from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Generator, List, Optional

from src.config import settings


_db: Optional[sqlite3.Connection] = None
_lock = Lock()


def get_db() -> sqlite3.Connection:
    global _db
    with _lock:
        if _db is None:
            db_path = Path(settings.storage.db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            _db = sqlite3.connect(str(db_path), check_same_thread=False)
            _db.row_factory = sqlite3.Row
            _init_schema(_db)
            _migrate_schema(_db)
        return _db


def _init_schema(db: sqlite3.Connection):
    """Create tables if they don't exist."""
    db.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA cache_size=-32768;

        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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

        CREATE TABLE IF NOT EXISTS gpu_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            gpu_index INTEGER NOT NULL,
            name TEXT,
            memory_used_mb REAL,
            memory_total_mb REAL,
            memory_free_mb REAL,
            gpu_utilization_percent REAL,
            memory_utilization_percent REAL,
            temperature_c REAL,
            power_watts REAL,
            power_limit_watts REAL
        );

        CREATE INDEX IF NOT EXISTS idx_gpu_samples_timestamp ON gpu_samples(timestamp);
        CREATE INDEX IF NOT EXISTS idx_gpu_samples_gpu ON gpu_samples(gpu_index);

        CREATE TABLE IF NOT EXISTS ollama_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            service_name TEXT NOT NULL,
            port INTEGER NOT NULL,
            models_json TEXT,
            tags_json TEXT,
            status TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_ollama_state_timestamp ON ollama_state(timestamp);

        CREATE TABLE IF NOT EXISTS patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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

        -- The real unit of health on pipeline-parallel hardware: a load cycle,
        -- not a single request or a single GPU. Every reload storm, timeout
        -- cascade, and near-zero keep_alive eviction diagnosed on this box
        -- shows up here as one row.
        CREATE TABLE IF NOT EXISTS load_cycles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            service_name TEXT NOT NULL,
            start_ts REAL NOT NULL,
            ctx_requested INTEGER,
            model TEXT,
            outcome TEXT NOT NULL DEFAULT 'pending',  -- pending | success | failed
            load_completed_ts REAL,
            failed_ts REAL,
            duration_s REAL,
            triggering_client_ip TEXT,   -- best-effort, from the nearest connection_samples row
            keep_alive_expires_at REAL,
            evicted_ts REAL,
            resident_duration_s REAL
        );

        CREATE INDEX IF NOT EXISTS idx_load_cycles_start ON load_cycles(start_ts);
        CREATE INDEX IF NOT EXISTS idx_load_cycles_service ON load_cycles(service_name);
        CREATE INDEX IF NOT EXISTS idx_load_cycles_outcome ON load_cycles(outcome);

        -- Periodic `ss` snapshots of established connections to each service's
        -- public-facing port. This is what actually caught the 12-14
        -- connection pileup that caused a cascading failure storm -- nothing
        -- in the request/access-log layer can see a queued-but-not-yet-served
        -- connection.
        CREATE TABLE IF NOT EXISTS connection_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            service_name TEXT NOT NULL,
            public_port INTEGER NOT NULL,
            established_count INTEGER NOT NULL,
            peers_json TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_connection_samples_timestamp ON connection_samples(timestamp);
        CREATE INDEX IF NOT EXISTS idx_connection_samples_service ON connection_samples(service_name);

        -- GPU hardware health, distinct from GPU load (gpu_samples above).
        -- Polled slowly (default 60s) since ECC counters and retirement
        -- state change rarely -- this is a health checklist, not a live
        -- chart, and was completely absent before (only util/mem/temp/power
        -- were ever sampled).
        CREATE TABLE IF NOT EXISTS gpu_hardware_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            gpu_index INTEGER NOT NULL,
            ecc_corrected_volatile INTEGER,
            ecc_uncorrected_volatile INTEGER,
            retired_pages_pending BOOLEAN,
            throttle_reasons TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_gpu_hw_timestamp ON gpu_hardware_samples(timestamp);
        CREATE INDEX IF NOT EXISTS idx_gpu_hw_gpu ON gpu_hardware_samples(gpu_index);

        -- Per-task token counts from `slot release` log lines -- the only
        -- place real prompt/completion sizes exist without full request-
        -- body capture. Persisted (not just broadcast live) so the
        -- dashboard can show "was this fixed-signature heartbeat traffic
        -- or genuinely varied real work" over a historical window, not
        -- just at the instant you happen to be watching.
        CREATE TABLE IF NOT EXISTS task_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            service_name TEXT NOT NULL,
            task_id INTEGER,
            total_tokens INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_task_samples_timestamp ON task_samples(timestamp);

        CREATE TABLE IF NOT EXISTS benchmark_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    """)


def _migrate_schema(db: sqlite3.Connection):
    """`CREATE TABLE IF NOT EXISTS` doesn't add columns to a table that
    already exists on disk with an older shape. This box's monitor.db
    predates `related_load_cycle_id` -- add it if missing rather than
    requiring a fresh database."""
    cols = {row[1] for row in db.execute("PRAGMA table_info(patterns)").fetchall()}
    if "related_load_cycle_id" not in cols:
        db.execute("ALTER TABLE patterns ADD COLUMN related_load_cycle_id INTEGER")
        db.commit()

    # Rows written before this rebuild used the raw systemd unit name
    # ("ollama-unified.service") for service_name; everything since uses
    # the friendly config name ("gpu-unified"). Without this, the same
    # service shows up as two separate rows in every by-service breakdown
    # until 7-day retention ages the old rows out on its own -- normalize
    # once at startup instead of waiting a week for it to stop being
    # confusing. Keyed off the config, not hardcoded, so it still works if
    # the systemd unit name ever changes again.
    marker = db.execute(
        "SELECT value FROM _migrations WHERE key = 'service_name_normalized'"
    ).fetchone() if _table_exists(db, "_migrations") else None
    if marker is None:
        db.execute("CREATE TABLE IF NOT EXISTS _migrations (key TEXT PRIMARY KEY, value TEXT)")
        for svc in settings.get_ollama_services():
            old_name = svc.systemd_service or f"ollama-{svc.name}.service"
            if old_name == svc.name:
                continue
            for table in ("requests",):
                db.execute(f"UPDATE {table} SET service_name = ? WHERE service_name = ?", (svc.name, old_name))
        db.execute(
            "INSERT OR REPLACE INTO _migrations (key, value) VALUES ('service_name_normalized', ?)",
            (str(time.time()),),
        )
        db.commit()

    # The pre-fix `error`-column bug (raw journald JSON stuffed into
    # `error` unconditionally) produced a real false-positive
    # quota_exhaustion pattern in testing against this box's own legacy
    # data -- purge any already-stored pattern rows that match the same
    # shape the now-fixed detector would reject (error not starting with
    # the GIN line prefix).
    db.execute("""
        DELETE FROM patterns WHERE pattern_type = 'quota_exhaustion'
        AND id IN (
            SELECT p.id FROM patterns p JOIN requests r ON p.related_request_id = r.id
            WHERE r.error IS NOT NULL AND r.error NOT LIKE '[GIN]%'
        )
    """)
    db.commit()


def _table_exists(db: sqlite3.Connection, name: str) -> bool:
    return db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


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


def insert(table: str, data: Dict[str, Any]) -> int:
    """Insert a row and return the lastrowid."""
    cols = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))
    sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
    cursor = execute(sql, tuple(data.values()))
    get_db().commit()
    return cursor.lastrowid


def cleanup_old_data():
    """Delete data older than retention_days."""
    cutoff = time.time() - (settings.storage.retention_days * 86400)
    db = get_db()
    db.execute("DELETE FROM requests WHERE timestamp < ?", (cutoff,))
    db.execute("DELETE FROM gpu_samples WHERE timestamp < ?", (cutoff,))
    db.execute("DELETE FROM ollama_state WHERE timestamp < ?", (cutoff,))
    db.execute("DELETE FROM patterns WHERE timestamp < ?", (cutoff,))
    db.execute("DELETE FROM load_cycles WHERE start_ts < ?", (cutoff,))
    db.execute("DELETE FROM connection_samples WHERE timestamp < ?", (cutoff,))
    db.execute("DELETE FROM gpu_hardware_samples WHERE timestamp < ?", (cutoff,))
    db.execute("DELETE FROM task_samples WHERE timestamp < ?", (cutoff,))
    db.commit()
    # WAL mode's automatic checkpoint only fires when no other readers hold the
    # WAL open; under continuous polling that can starve indefinitely, letting the
    # WAL file grow unbounded (observed: 7.3GB WAL on an 8.4GB db after ~6 days).
    # DELETE without a following checkpoint doesn't shrink anything on disk either.
    # TRUNCATE checkpoints then truncates the WAL file back to zero bytes.
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")