from __future__ import annotations

import sqlite3
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Deque, Dict, Generator, List, Optional

from src.config import settings


_db: Optional[sqlite3.Connection] = None
_lock = Lock()
_ring_counters: Dict[str, int] = {}


def get_db() -> sqlite3.Connection:
    global _db
    with _lock:
        if _db is None:
            db_path = Path(settings.storage.db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            _db = sqlite3.connect(str(db_path), check_same_thread=False)
            _db.row_factory = sqlite3.Row
            _init_schema(_db)
        return _db


def _init_schema(db: sqlite3.Connection):
    """Every table here is bounded by construction: a fixed-capacity ring
    (`requests`, `task_samples`), fixed day-slots (`patterns`, `load_cycles`,
    `benchmark_results`), a fixed slot per time-bucket (`connection_samples`),
    or a single latest-row-per-key upsert (`ollama_state`,
    `gpu_hardware_samples`). Each write path overwrites its own oldest entry
    directly -- no periodic cleanup job and no VACUUM are needed, because
    table size can never grow past its bound regardless of traffic. The old
    accumulate-forever-then-DELETE design let monitor.db reach 10.8GB (see
    data/archive/monitor.db.bloated-*) because DELETE frees rows logically
    but SQLite never shrinks the file for it without a VACUUM.
    """
    db.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA cache_size=-32768;

        -- Fixed-capacity ring: `id` is a wrapped counter (id = n % capacity),
        -- not an autoincrement. INSERT OR REPLACE overwrites whatever
        -- occupied that slot `raw_ring_capacity` writes ago. See
        -- storage.ring_insert().
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY,
            timestamp REAL NOT NULL,
            service_name TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            method TEXT NOT NULL,
            client_ip TEXT,
            client_pid INTEGER,
            client_process TEXT,
            model TEXT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            ttft_ms REAL,
            tps REAL,
            total_latency_ms REAL,
            duration_ms REAL,
            status_code INTEGER,
            prompt_length_chars INTEGER,
            response_length_chars INTEGER,
            num_ctx INTEGER,
            temperature REAL,
            top_p REAL,
            stop_sequences TEXT,
            error TEXT,
            gpu_id INTEGER,
            gpu_memory_used_mb REAL,
            gpu_utilization_percent REAL,
            gpu_power_watts REAL,
            cpu_percent REAL,
            cpu_spillover BOOLEAN,
            swap_detected BOOLEAN,
            raw_request TEXT,
            raw_response TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp);
        CREATE INDEX IF NOT EXISTS idx_requests_service ON requests(service_name);
        CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model);
        CREATE INDEX IF NOT EXISTS idx_requests_client_ip ON requests(client_ip);

        -- Latest known state only, one row per service (PK) -- every
        -- consumer (/api/ollama/services, /api/health_status) only ever
        -- wants "what's resident right now," never history. See
        -- storage.upsert().
        CREATE TABLE IF NOT EXISTS ollama_state (
            service_name TEXT PRIMARY KEY,
            timestamp REAL NOT NULL,
            port INTEGER NOT NULL,
            models_json TEXT,
            tags_json TEXT,
            status TEXT
        );

        -- Discrete events, retained for exactly `event_retention_days`
        -- calendar days via day-slots that wrap: day_slot = epoch_day %
        -- retention_days. Writing into today's slot evicts any stale rows
        -- already there from a previous, different epoch day. See
        -- storage.day_bucket_insert().
        CREATE TABLE IF NOT EXISTS patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            day_slot INTEGER NOT NULL,
            epoch_day INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            pattern_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            description TEXT,
            related_request_id INTEGER,
            related_gpu_index INTEGER,
            related_load_cycle_id INTEGER,
            details_json TEXT,
            acknowledged BOOLEAN DEFAULT FALSE
        );

        CREATE INDEX IF NOT EXISTS idx_patterns_timestamp ON patterns(timestamp);
        CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type);
        CREATE INDEX IF NOT EXISTS idx_patterns_dedup ON patterns(pattern_type, related_request_id);
        CREATE INDEX IF NOT EXISTS idx_patterns_day_slot ON patterns(day_slot);

        -- The real unit of health on pipeline-parallel hardware: a load
        -- cycle, not a single request or a single GPU. Same day-slot
        -- retention as patterns -- event data, not a continuous stream.
        -- Rows are created 'pending' and later mutated by id (loaded /
        -- failed / evicted) -- day_slot only governs eviction, `id` stays a
        -- normal unique key for the lifetime of the cycle.
        CREATE TABLE IF NOT EXISTS load_cycles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            day_slot INTEGER NOT NULL,
            epoch_day INTEGER NOT NULL,
            service_name TEXT NOT NULL,
            start_ts REAL NOT NULL,
            ctx_requested INTEGER,
            model TEXT,
            outcome TEXT NOT NULL DEFAULT 'pending',
            load_completed_ts REAL,
            failed_ts REAL,
            duration_s REAL,
            triggering_client_ip TEXT,
            keep_alive_expires_at REAL,
            evicted_ts REAL,
            resident_duration_s REAL
        );

        CREATE INDEX IF NOT EXISTS idx_load_cycles_start ON load_cycles(start_ts);
        CREATE INDEX IF NOT EXISTS idx_load_cycles_service ON load_cycles(service_name);
        CREATE INDEX IF NOT EXISTS idx_load_cycles_outcome ON load_cycles(outcome);
        CREATE INDEX IF NOT EXISTS idx_load_cycles_day_slot ON load_cycles(day_slot);

        -- `connection_window_hours` sliding window at `connection_bucket_
        -- seconds` resolution. One row per (service_name, bucket_index); a
        -- bucket's index recurs every connection_window_hours, so writing
        -- it overwrites whatever it held one full window ago. Fixed row
        -- count forever: num_services * (window_hours * 3600 /
        -- bucket_seconds). See storage.connection_bucket_upsert().
        CREATE TABLE IF NOT EXISTS connection_samples (
            service_name TEXT NOT NULL,
            bucket_index INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            public_port INTEGER NOT NULL,
            established_count INTEGER NOT NULL,
            PRIMARY KEY (service_name, bucket_index)
        );

        CREATE INDEX IF NOT EXISTS idx_connection_samples_timestamp ON connection_samples(timestamp);

        -- Latest known hardware health only, one row per GPU (PK) -- ECC /
        -- retirement state changes rarely and every consumer only wants
        -- current status, never a trend line.
        CREATE TABLE IF NOT EXISTS gpu_hardware_samples (
            gpu_index INTEGER PRIMARY KEY,
            timestamp REAL NOT NULL,
            ecc_corrected_volatile INTEGER,
            ecc_uncorrected_volatile INTEGER,
            retired_pages_pending BOOLEAN,
            throttle_reasons TEXT
        );

        -- Fixed-capacity ring, same scheme as `requests`.
        CREATE TABLE IF NOT EXISTS task_samples (
            id INTEGER PRIMARY KEY,
            timestamp REAL NOT NULL,
            service_name TEXT NOT NULL,
            task_id INTEGER,
            total_tokens INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_task_samples_timestamp ON task_samples(timestamp);

        -- Fixed-capacity ring, same scheme as `requests`. Deliberately never
        -- stores the raw prompt/response body -- only a short (<= ~8-word)
        -- derived summary, mechanical or LLM-generated (see
        -- main.py:prompt_mirror). This is inference traffic that can carry
        -- proprietary source or secrets; the full body only ever exists
        -- transiently in-process for the duration of one request.
        CREATE TABLE IF NOT EXISTS prompt_summaries (
            id INTEGER PRIMARY KEY,
            timestamp REAL NOT NULL,
            service_name TEXT NOT NULL,
            model TEXT,
            endpoint TEXT NOT NULL,
            summary TEXT NOT NULL,
            source TEXT NOT NULL  -- 'llm' | 'truncated'
        );

        CREATE INDEX IF NOT EXISTS idx_prompt_summaries_timestamp ON prompt_summaries(timestamp);

        -- Manually-triggered benchmark runs -- naturally low volume, same
        -- day-slot retention as patterns/load_cycles for consistency (no
        -- separate cleanup path to maintain).
        CREATE TABLE IF NOT EXISTS benchmark_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            day_slot INTEGER NOT NULL,
            epoch_day INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            pool_name TEXT NOT NULL,
            port INTEGER NOT NULL,
            model TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'generate',
            success BOOLEAN NOT NULL,
            error TEXT,
            total_duration_ms REAL,
            load_duration_ms REAL,
            prompt_eval_count INTEGER,
            prompt_eval_duration_ms REAL,
            eval_count INTEGER,
            eval_duration_ms REAL,
            tokens_per_second REAL,
            ttft_ms REAL,
            response_chars INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_benchmark_timestamp ON benchmark_results(timestamp);
        CREATE INDEX IF NOT EXISTS idx_benchmark_pool_model ON benchmark_results(pool_name, model);
        CREATE INDEX IF NOT EXISTS idx_benchmark_day_slot ON benchmark_results(day_slot);
    """)


@contextmanager
def transaction() -> Generator[sqlite3.Connection, None, None]:
    db = get_db()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise


def execute(query: str, params: tuple = ()) -> sqlite3.Cursor:
    db = get_db()
    return db.execute(query, params)


def query_all(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    return [dict(row) for row in execute(query, params).fetchall()]


def query_one(query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    row = execute(query, params).fetchone()
    return dict(row) if row else None


def upsert(table: str, data: Dict[str, Any]) -> None:
    """INSERT OR REPLACE keyed by the table's own primary key (e.g.
    `service_name` on ollama_state, `gpu_index` on gpu_hardware_samples) --
    always exactly one row per key, so no history ever accumulates."""
    cols = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))
    sql = f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})"
    db = get_db()
    db.execute(sql, tuple(data.values()))
    db.commit()


def ring_insert(table: str, data: Dict[str, Any], capacity: int) -> int:
    """Insert into a fixed-capacity ring-buffer table (`requests`,
    `task_samples`). `id` is a wrapped counter, not an autoincrement, so
    INSERT OR REPLACE always overwrites whatever occupied that slot
    `capacity` writes ago -- the table can never hold more than `capacity`
    rows, regardless of traffic. Counter resets to 0 on process restart,
    which just means the next `capacity` writes re-evict slots 0..N in
    order rather than true LRU order -- self-corrects within one cycle."""
    with _lock:
        n = _ring_counters.get(table, 0)
        _ring_counters[table] = n + 1
    row_id = n % capacity
    row = {"id": row_id, **data}
    cols = ", ".join(row.keys())
    placeholders = ", ".join(["?"] * len(row))
    db = get_db()
    db.execute(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})", tuple(row.values()))
    db.commit()
    return row_id


def upgrade_ring_row(table: str, row_id: int, timestamp: float, data: Dict[str, Any]) -> None:
    """Update a row previously written by ring_insert(), guarded by the
    timestamp it was written with -- if the ring wrapped around and that
    slot was already overwritten by a newer row before this update arrives
    (only possible under very high write volume), the guard makes this a
    silent no-op instead of corrupting the newer row. Used to upgrade a
    ring_insert's synchronous placeholder with a slower async result (e.g.
    an LLM-generated prompt summary replacing the mechanical fallback)."""
    set_cols = ", ".join(f"{k} = ?" for k in data.keys())
    db = get_db()
    db.execute(
        f"UPDATE {table} SET {set_cols} WHERE id = ? AND timestamp = ?",
        (*data.values(), row_id, timestamp),
    )
    db.commit()


def day_bucket_insert(table: str, data: Dict[str, Any], timestamp: float, retention_days: int) -> int:
    """Insert into a table retained for exactly `retention_days` calendar
    days via day-slots that wrap: day_slot = epoch_day % retention_days.
    Writing into today's slot first evicts any stale rows already sitting
    there from a previous, different epoch day -- eviction is a side effect
    of the write path itself, not a periodic job, and only ever does
    anything once per slot per day (every other write is a no-op delete)."""
    epoch_day = int(timestamp // 86400)
    day_slot = epoch_day % retention_days
    db = get_db()
    db.execute(f"DELETE FROM {table} WHERE day_slot = ? AND epoch_day != ?", (day_slot, epoch_day))
    row = {"day_slot": day_slot, "epoch_day": epoch_day, **data}
    cols = ", ".join(row.keys())
    placeholders = ", ".join(["?"] * len(row))
    cursor = db.execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", tuple(row.values()))
    db.commit()
    return cursor.lastrowid


def connection_bucket_upsert(service_name: str, public_port: int, timestamp: float, established_count: int) -> None:
    """48h-style sliding window (width: settings.storage.connection_window_
    hours) at connection_bucket_seconds resolution: one row per (service,
    bucket_index), and a bucket's index recurs every connection_window_hours
    -- writing it overwrites whatever it held one full window ago. Fixed
    row count forever, no cleanup needed."""
    bucket_seconds = settings.storage.connection_bucket_seconds
    num_buckets = int(settings.storage.connection_window_hours * 3600 / bucket_seconds)
    bucket_index = int(timestamp // bucket_seconds) % num_buckets
    db = get_db()
    db.execute("""
        INSERT OR REPLACE INTO connection_samples
            (service_name, bucket_index, timestamp, public_port, established_count)
        VALUES (?, ?, ?, ?, ?)
    """, (service_name, bucket_index, timestamp, public_port, established_count))
    db.commit()


# ---------------------------------------------------------------------------
# GPU load samples (util/mem/temp/power): pure live/snapshot data, kept in
# memory only -- never written to disk. Nothing currently queries this past
# a live dashboard refresh, and it changes every couple of seconds forever;
# persisting it was the single largest contributor to unbounded growth. A
# restart loses the last `gpu_sample_memory_minutes` of chart history and
# refills within a couple of poll cycles.
# ---------------------------------------------------------------------------

_gpu_memory: Dict[int, Deque[Dict[str, Any]]] = {}


def _gpu_memory_capacity() -> int:
    poll_s = max(1, settings.collectors.gpu_poll_interval_seconds)
    return max(1, int(settings.storage.gpu_sample_memory_minutes * 60 / poll_s))


def store_gpu_sample(sample: Dict[str, Any]) -> None:
    dq = _gpu_memory.get(sample["gpu_index"])
    if dq is None:
        dq = deque(maxlen=_gpu_memory_capacity())
        _gpu_memory[sample["gpu_index"]] = dq
    dq.append(sample)


def query_gpu_samples(cutoff: float = 0.0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dq in _gpu_memory.values():
        rows.extend(s for s in dq if s["timestamp"] > cutoff)
    return rows


def latest_gpu_samples() -> List[Dict[str, Any]]:
    return [dq[-1] for dq in _gpu_memory.values() if dq]
