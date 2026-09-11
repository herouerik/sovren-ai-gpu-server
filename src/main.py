from __future__ import annotations

import sys
from pathlib import Path


def _verify_running_from_project_venv() -> None:
    """Fail loudly, immediately, before any real import happens -- rather
    than the confusing `TypeError: unhashable type: 'dict'` a mismatched
    global jinja2 produces deep inside template rendering on the first
    request. A bare `uvicorn src.main:app` resolves to whatever `uvicorn`
    is first on PATH (often `~/.local/bin/uvicorn`), which runs against
    globally-installed packages instead of this project's pinned
    `requirements.txt` versions -- `run.sh` exists specifically to avoid
    this, but nothing stopped someone from bypassing it. Confirmed live:
    global jinja2 3.1.2 has this exact caching bug, the pinned 3.1.4 does
    not.
    """
    project_root = Path(__file__).resolve().parent.parent
    venv_dir = project_root / ".venv"
    if not venv_dir.exists():
        return  # no .venv at all yet (e.g. fresh checkout, first-time setup) -- let pip/venv errors surface normally
    # sys.prefix (not sys.executable) is the correct check here: a venv's
    # bin/python is typically a symlink to the base interpreter it was
    # created from, so resolving sys.executable follows that symlink right
    # back out of .venv even when correctly invoked through it. sys.prefix
    # reflects the active environment regardless of that symlink.
    if Path(sys.prefix).resolve() != venv_dir.resolve():
        sys.stderr.write(
            "\n"
            "=========================================================================\n"
            "  sovren-ai-gpu-server: not running from this project's .venv\n"
            "-------------------------------------------------------------------------\n"
            f"  Running interpreter : {sys.executable}\n"
            f"  Expected under      : {venv_dir}\n"
            "\n"
            "  A bare `uvicorn src.main:app` (or a global `python`) resolves to\n"
            "  whatever's first on PATH, which runs against globally-installed\n"
            "  packages instead of this project's pinned requirements.txt versions.\n"
            "  This has previously caused a silent `TypeError: unhashable type:\n"
            "  'dict'` from a mismatched global jinja2 on the very first request.\n"
            "\n"
            "  Run it via the wrapper instead:\n"
            "      ./run.sh\n"
            "  or invoke uvicorn through the venv's own interpreter explicitly:\n"
            "      .venv/bin/python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8082\n"
            "=========================================================================\n\n"
        )
        sys.exit(1)


_verify_running_from_project_venv()

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import httpx

from src.config import settings
from src.storage import (
    get_db, execute, query_all, query_one, ring_insert, upgrade_ring_row,
    attach_prompt_metrics, cache_raw_prompt, get_raw_prompt,
    latest_gpu_samples, query_gpu_samples,
)
from src.collectors import GPUCollector, OllamaStateCollector, LogTailer, ConnectionsCollector, GPUHardwareCollector
from src.lifecycle import get_tracker
from src.patterns import analyze_and_store

# Global state
gpu_collector: GPUCollector = GPUCollector()
ollama_state_collector: OllamaStateCollector = OllamaStateCollector()
log_tailer: LogTailer = LogTailer()
connections_collector: ConnectionsCollector = ConnectionsCollector()
gpu_hardware_collector: GPUHardwareCollector = GPUHardwareCollector()
ws_connections: List[WebSocket] = []
STARTED_AT: float = time.time()
# Guards the prompt-summarizer against exactly the kind of overload it can
# itself cause -- OLLAMA_NUM_PARALLEL=1 on every summarizer service
# configured so far means it can only truly work on one call at a time. A
# burst of mirrored prompts fires one independent asyncio.create_task per
# prompt; queueing the rest behind a semaphore just delays the pileup
# instead of fixing it (each waiting task still holds real memory/asyncio
# state, and by the time its turn comes the summary may be pointless).
# Instead: if the summarizer is already busy, skip immediately and leave
# the mechanical-truncation fallback in place -- never queue. Plain bool,
# not a lock/semaphore: single-threaded asyncio event loop, no `await`
# between the check and the set, so there's no race to guard against.
SUMMARIZER_BUSY = False


async def broadcast_ws(message: Dict[str, Any]):
    """Broadcast message to all connected WebSocket clients."""
    dead = []
    for ws in ws_connections:
        try:
            await ws.send_json(message)
        except:
            dead.append(ws)
    for ws in dead:
        ws_connections.remove(ws)


async def collector_orchestrator():
    """Background task that runs all collectors and pattern analysis."""
    last_gpu_hw = 0.0
    last_wal_checkpoint = 0.0
    wal_checkpoint_interval_seconds = 60
    tracker = get_tracker()
    while True:
        try:
            # GPU samples
            gpu_samples = await gpu_collector.collect()
            await gpu_collector.store_samples(gpu_samples)
            await broadcast_ws({"type": "gpu_update", "data": [s.__dict__ for s in gpu_samples]})

            # Ollama state — also feeds the lifecycle tracker (eviction
            # detection, keep_alive_expires_at capture)
            ollama_states = await ollama_state_collector.collect()
            await ollama_state_collector.store_states(ollama_states)
            for state in ollama_states:
                await tracker.handle_ollama_state(state.service_name, state.models)

            # Connections — the signal that caught the real pileup incident
            conn_samples = await connections_collector.collect()
            await connections_collector.store_samples(conn_samples)
            if conn_samples:
                await broadcast_ws({"type": "connections_update", "data": conn_samples})

            # GPU hardware health — slow cadence, rarely changes
            now = time.time()
            if now - last_gpu_hw > settings.collectors.gpu_hardware_poll_interval_seconds:
                hw_samples = await gpu_hardware_collector.collect()
                await gpu_hardware_collector.store_samples(hw_samples)
                last_gpu_hw = now

            # Pattern analysis
            patterns = await analyze_and_store()
            if patterns:
                await broadcast_ws({"type": "patterns", "data": [
                    {"timestamp": p.timestamp, "type": p.pattern_type,
                     "severity": p.severity, "description": p.description}
                    for p in patterns
                ]})

            # WAL checkpoint: independent of row retention -- under
            # continuous concurrent reads (API GETs), SQLite's automatic
            # checkpoint can starve indefinitely and let the WAL journal
            # file itself grow unboundedly even though every table is now
            # bounded. TRUNCATE checkpoints then truncates it back to zero.
            if now - last_wal_checkpoint > wal_checkpoint_interval_seconds:
                get_db().execute("PRAGMA wal_checkpoint(TRUNCATE)")
                last_wal_checkpoint = now

        except Exception as e:
            print(f"Collector error: {e}")

        await asyncio.sleep(settings.collectors.gpu_poll_interval_seconds)


async def log_capture_orchestrator():
    """Captures ollama logs: access-log requests, load-cycle lifecycle
    events, and per-task slot timing, from the same journald tail."""
    tracker = get_tracker()
    # entry.service_name is the actual systemd unit; map it back to the
    # config's public_port for lifecycle connection snapshots.
    service_lookup = {
        (svc.systemd_service or f"ollama-{svc.name}.service"): svc
        for svc in settings.get_ollama_services()
    }
    while True:
        try:
            for systemd_name, svc in service_lookup.items():
                if not svc.enabled:
                    continue
                entries = await log_tailer.tail_service(systemd_name)
                for entry in entries:
                    # Every extractor below tags events with `entry.service_name`,
                    # which is the raw systemd unit ("ollama-unified.service").
                    # Everywhere else (connection_samples, health_status
                    # queries) keys off the friendly config name
                    # ("gpu-unified") instead -- normalize to that here, once,
                    # rather than have two service-name conventions that
                    # silently never join.
                    req_info = log_tailer.extract_request_info(entry)
                    if req_info:
                        req_info["service_name"] = svc.name
                        ring_insert("requests", {
                            "timestamp": req_info["timestamp"],
                            "service_name": req_info["service_name"],
                            "endpoint": req_info["path"],
                            "method": req_info["method"],
                            "client_ip": req_info["client_ip"],
                            "status_code": req_info["status_code"],
                            "duration_ms": req_info["duration_ms"],
                            "error": req_info["error"],
                        }, capacity=settings.storage.raw_ring_capacity)
                        await broadcast_ws({"type": "request", "data": req_info})
                        continue  # a line is one of these three kinds, never more than one

                    lifecycle_event = log_tailer.extract_lifecycle_event(entry)
                    if lifecycle_event:
                        lifecycle_event["service_name"] = svc.name
                        await tracker.handle_event(lifecycle_event, public_port=svc.effective_public_port())
                        await broadcast_ws({"type": "lifecycle_event", "data": lifecycle_event})
                        continue

                    slot_info = log_tailer.extract_slot_info(entry)
                    if slot_info:
                        slot_info["service_name"] = svc.name
                        ring_insert("task_samples", {
                            "timestamp": slot_info["timestamp"],
                            "service_name": slot_info["service_name"],
                            "task_id": slot_info["task_id"],
                            "total_tokens": slot_info["total_tokens"],
                            "ttft_ms": slot_info["ttft_ms"],
                            "prefill_tps": slot_info["prefill_tps"],
                            "decode_tps": slot_info["decode_tps"],
                        }, capacity=settings.storage.raw_ring_capacity)
                        # Best-effort: attach these same numbers to the
                        # Recent Prompts row this task most likely came
                        # from -- see storage.attach_prompt_metrics(). Skip
                        # entirely if ttft_ms is missing (e.g. an errored/
                        # cancelled task with no timing lines) -- writing
                        # nulls would leave the row matchable as "still
                        # pending" and risk a later task attaching to it
                        # instead of its own.
                        if slot_info["ttft_ms"] is not None:
                            attach_prompt_metrics(slot_info["service_name"], slot_info["timestamp"], {
                                "total_tokens": slot_info["total_tokens"],
                                "ttft_ms": slot_info["ttft_ms"],
                                "prefill_tps": slot_info["prefill_tps"],
                                "decode_tps": slot_info["decode_tps"],
                            })
                        await broadcast_ws({"type": "slot_release", "data": slot_info})
        except Exception as e:
            print(f"Log capture error: {e}")
        await asyncio.sleep(settings.collectors.log_poll_interval_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting sovren-ai-gpu-server...")
    collector_task = asyncio.create_task(collector_orchestrator())
    log_task = asyncio.create_task(log_capture_orchestrator())
    try:
        yield
    finally:
        # Shutdown
        print("Shutting down...")
        collector_task.cancel()
        log_task.cancel()
        try:
            await collector_task
        except asyncio.CancelledError:
            pass
        try:
            await log_task
        except asyncio.CancelledError:
            pass
        gpu_collector.cleanup()
        await ollama_state_collector.close()


app = FastAPI(
    title="sovren-ai-gpu-server",
    description="GPU Ollama Traffic Monitor",
    version="0.1.0",
    lifespan=lifespan
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# =============================================================================
# API ROUTES
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/requests")
async def get_requests(
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    service: Optional[str] = None,
    model: Optional[str] = None,
    endpoint: Optional[str] = None,
    client_ip: Optional[str] = None,
    since: Optional[float] = None,
    has_error: Optional[bool] = None
):
    """Get paginated request log with filters."""
    conditions = []
    params = []

    if service:
        conditions.append("service_name = ?")
        params.append(service)
    if model:
        conditions.append("model = ?")
        params.append(model)
    if endpoint:
        conditions.append("endpoint = ?")
        params.append(endpoint)
    if client_ip:
        conditions.append("client_ip = ?")
        params.append(client_ip)
    if since:
        conditions.append("timestamp > ?")
        params.append(since)
    if has_error is not None:
        conditions.append("error IS NOT NULL" if has_error else "error IS NULL")

    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    total = query_one(f"SELECT COUNT(*) as c FROM requests {where}", tuple(params))["c"]

    rows = execute(f"""
        SELECT * FROM requests
        {where}
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    """, tuple(params + [limit, offset])).fetchall()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "requests": [dict(r) for r in rows]
    }


@app.get("/api/metrics/summary")
async def get_metrics_summary(
    window_seconds: int = Query(300, ge=60, le=86400)
):
    """Aggregated metrics by model, service, GPU."""
    cutoff = time.time() - window_seconds

    # By model (fallback to service_name when model is null)
    by_model = [dict(r) for r in execute("""
        SELECT COALESCE(model, service_name) as model_or_service, service_name,
               COUNT(*) as request_count,
               AVG(duration_ms) as avg_latency_ms,
               SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as error_count
        FROM requests
        WHERE timestamp > ?
        GROUP BY COALESCE(model, service_name), service_name
        ORDER BY request_count DESC
    """, (cutoff,)).fetchall()]

    # TTFT/prefill/decode come from task_samples (llama.cpp's own per-task
    # timing, parsed from "prompt eval time" / "eval time" log lines), not
    # from `requests` -- the GIN access log structurally can't carry them.
    # Kept as two separate rates (not one blended "tokens/sec") because
    # they behave very differently across models/context sizes -- same
    # distinction as sovren-ai-benchmarking's own dashboard. Merged in by
    # service_name since `requests.model` is always null in practice (see
    # README) and model_or_service already falls back to service_name.
    task_metrics_by_service = {
        r["service_name"]: dict(r)
        for r in execute("""
            SELECT service_name,
                   AVG(ttft_ms) as avg_ttft_ms,
                   AVG(prefill_tps) as avg_prefill_tps,
                   AVG(decode_tps) as avg_decode_tps
            FROM task_samples
            WHERE timestamp > ? AND (ttft_ms IS NOT NULL OR prefill_tps IS NOT NULL OR decode_tps IS NOT NULL)
            GROUP BY service_name
        """, (cutoff,)).fetchall()
    }
    for row in by_model:
        tm = task_metrics_by_service.get(row["service_name"], {})
        row["avg_ttft_ms"] = tm.get("avg_ttft_ms")
        row["avg_prefill_tps"] = tm.get("avg_prefill_tps")
        row["avg_decode_tps"] = tm.get("avg_decode_tps")

    # By GPU
    by_gpu = execute("""
        SELECT gpu_id,
               COUNT(*) as request_count,
               AVG(gpu_memory_used_mb) as avg_vram_mb,
               AVG(gpu_utilization_percent) as avg_util,
               AVG(gpu_power_watts) as avg_power_w
        FROM requests
        WHERE timestamp > ? AND gpu_id IS NOT NULL
        GROUP BY gpu_id
    """, (cutoff,)).fetchall()

    # By caller (IP)
    # One row per (client_ip, service_name) -- multiple pools now exist
    # (gpu-unified, meta, and more as they come online), and a caller's
    # traffic pattern per pool is the actually useful signal. The frontend
    # pivots this into one row per IP with a column per service, rather
    # than multiplying rows.
    by_caller = execute("""
        SELECT client_ip, service_name,
               COUNT(*) as request_count,
               AVG(duration_ms) as avg_latency_ms,
               SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as errors
        FROM requests
        WHERE timestamp > ? AND client_ip IS NOT NULL
        GROUP BY client_ip, service_name
        ORDER BY request_count DESC
        LIMIT 60
    """, (cutoff,)).fetchall()

    # By endpoint
    by_endpoint = execute("""
        SELECT endpoint, method,
               COUNT(*) as request_count,
               AVG(duration_ms) as avg_latency_ms,
               SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as errors
        FROM requests
        WHERE timestamp > ?
        GROUP BY endpoint, method
        ORDER BY request_count DESC
    """, (cutoff,)).fetchall()

    return {
        "window_seconds": window_seconds,
        "by_model": by_model,
        "by_gpu": [dict(r) for r in by_gpu],
        "by_caller": [dict(r) for r in by_caller],
        "by_endpoint": [dict(r) for r in by_endpoint]
    }


@app.get("/api/metrics/timeseries")
async def get_timeseries(
    metric: str = Query(..., description="Metric: latency, tps, ttft, requests, gpu_util, gpu_mem, gpu_power, gpu_temp"),
    group_by: str = Query("model", description="Group by: model, service, gpu, caller"),
    window_seconds: int = Query(3600, ge=60, le=86400),
    bucket_seconds: int = Query(60, ge=10, le=3600)
):
    """Time-series data for charts."""
    cutoff = time.time() - window_seconds

    if metric in ("latency", "tps", "ttft", "requests"):
        group_col = {"model": "COALESCE(model, service_name)", "service": "service_name",
                      "gpu": "gpu_id", "caller": "client_ip"}.get(group_by, "COALESCE(model, service_name)")
        agg = {"latency": "AVG(duration_ms)", "tps": "0", "ttft": "0", "requests": "COUNT(*)"}[metric]

        rows = execute(f"""
            SELECT
                (CAST(timestamp / ? AS INTEGER) * ?) as bucket,
                {group_col} as grp,
                {agg} as value
            FROM requests
            WHERE timestamp > ?
            GROUP BY bucket, grp
            ORDER BY bucket
        """, (bucket_seconds, bucket_seconds, cutoff)).fetchall()

    elif metric in ("gpu_util", "gpu_mem", "gpu_power", "gpu_temp"):
        # Pure live/snapshot data, served from the in-memory ring -- see
        # storage.query_gpu_samples() -- bucketed here in Python instead of
        # SQL GROUP BY.
        field = {"gpu_util": "gpu_utilization_percent", "gpu_mem": "memory_used_mb",
                  "gpu_power": "power_watts", "gpu_temp": "temperature_c"}[metric]
        buckets: Dict[tuple, List[float]] = {}
        for s in query_gpu_samples(cutoff):
            bucket = int(s["timestamp"] // bucket_seconds) * bucket_seconds
            buckets.setdefault((bucket, s["gpu_index"]), []).append(s[field])
        rows = [{"bucket": b, "grp": g, "value": sum(vals) / len(vals)} for (b, g), vals in buckets.items()]

    else:
        return JSONResponse({"error": "Unknown metric"}, status_code=400)

    # Pivot for frontend
    data: Dict[int, Dict[str, float]] = {}
    for r in rows:
        bucket = int(r["bucket"])
        grp = str(r["grp"]) if r["grp"] is not None else "unknown"
        val = r["value"]
        if bucket not in data:
            data[bucket] = {}
        data[bucket][grp] = val

    return {
        "metric": metric,
        "group_by": group_by,
        "bucket_seconds": bucket_seconds,
        "series": [{"timestamp": ts, **vals} for ts, vals in sorted(data.items())]
    }


@app.get("/api/gpus")
async def get_gpus():
    """Current GPU state -- pure live/snapshot data, served from the
    in-memory ring (see storage.latest_gpu_samples()), not SQL."""
    cutoff = time.time() - 10
    service_by_gpu = {
        r["gpu_id"]: r["service_name"]
        for r in execute("SELECT DISTINCT gpu_id, service_name FROM requests WHERE timestamp > ?", (cutoff,)).fetchall()
    }
    return [
        {**s, "service_name": service_by_gpu.get(s["gpu_index"])}
        for s in latest_gpu_samples()
    ]


@app.get("/api/load_cycles")
async def get_load_cycles(
    limit: int = Query(100, le=2000),
    service: Optional[str] = None,
    outcome: Optional[str] = None,
    since: Optional[float] = None
):
    """Load-cycle history: the real unit of health on this hardware.

    Each row is one model load attempt, from "starting llama-server" through
    to success/failure/eviction -- this is what a reload storm, a load-
    cancel cascade, or a near-zero keep_alive eviction actually look like.
    """
    conditions = []
    params: List[Any] = []
    if service:
        conditions.append("service_name = ?")
        params.append(service)
    if outcome:
        conditions.append("outcome = ?")
        params.append(outcome)
    if since:
        conditions.append("start_ts > ?")
        params.append(since)
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    rows = execute(f"""
        SELECT * FROM load_cycles {where} ORDER BY start_ts DESC LIMIT ?
    """, tuple(params + [limit])).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/health_status")
async def get_health_status():
    """One consolidated answer to 'is this box healthy right now' — the
    hero-card data. Deliberately not a generic metrics dump: every field
    here maps to a specific failure mode this box has actually hit.
    """
    result = {}
    for svc in settings.get_ollama_services():
        if not svc.enabled:
            result[svc.name] = {"enabled": False}
            continue

        latest_cycle = query_one("""
            SELECT * FROM load_cycles WHERE service_name = ? ORDER BY start_ts DESC LIMIT 1
        """, (svc.name,))

        # `load_cycles` only has rows for reloads this monitor has personally
        # witnessed since it started -- a model loaded before that (the
        # common case right after (re)starting the monitor itself) has no
        # matching row, and load_cycles-only logic reported IDLE/no-model for
        # a service that was genuinely, stably loaded. `ollama_state` (a
        # direct /api/ps poll, populated immediately regardless of monitor
        # uptime) is the actual ground truth for "is something resident
        # right now" -- use it as that, and load_cycles only for
        # reload-history detail (loaded_since timing, churn counts).
        state_row = query_one("""
            SELECT models_json FROM ollama_state
            WHERE service_name = ? ORDER BY timestamp DESC LIMIT 1
        """, (svc.name,))
        resident_model = None
        if state_row and state_row["models_json"]:
            models = json.loads(state_row["models_json"])
            if models:
                resident_model = models[0]
        resident_expires_at = None
        if resident_model and resident_model.get("expires_at"):
            from datetime import datetime
            try:
                resident_expires_at = datetime.fromisoformat(
                    resident_model["expires_at"].replace("Z", "+00:00")
                ).timestamp()
            except ValueError:
                pass

        cutoff_5m = time.time() - 300
        reload_count_5m = query_one("""
            SELECT COUNT(*) as c FROM load_cycles WHERE service_name = ? AND start_ts > ?
        """, (svc.name, cutoff_5m))["c"]

        cutoff_15m = time.time() - settings.patterns.ctx_churn_window_seconds
        distinct_ctx_15m = query_one("""
            SELECT COUNT(DISTINCT ctx_requested) as c FROM load_cycles
            WHERE service_name = ? AND start_ts > ? AND ctx_requested IS NOT NULL
        """, (svc.name, cutoff_15m))["c"]

        conn = query_one("""
            SELECT established_count FROM connection_samples
            WHERE service_name = ? ORDER BY timestamp DESC LIMIT 1
        """, (svc.name,))

        # Status derivation, in priority order — matches the actual
        # diagnostic sequence used on this box, worst first.
        if conn and conn["established_count"] >= settings.patterns.connection_pileup_threshold:
            status = "DEGRADED"
            status_reason = f"{conn['established_count']} connections piled up"
        elif latest_cycle and latest_cycle["outcome"] == "pending" and \
                (time.time() - latest_cycle["start_ts"]) > 30:
            status = "RELOADING"
            status_reason = f"loading for {time.time() - latest_cycle['start_ts']:.0f}s"
        elif reload_count_5m >= settings.patterns.reload_storm_count_threshold:
            status = "DEGRADED"
            status_reason = f"{reload_count_5m} reloads in 5 min"
        elif distinct_ctx_15m >= settings.patterns.ctx_churn_distinct_values_threshold:
            status = "DEGRADED"
            status_reason = f"{distinct_ctx_15m} different context sizes in 15 min"
        elif resident_model:
            # Ground truth: /api/ps says something is loaded right now, and
            # none of the worse conditions above applied. True whether or
            # not this monitor happened to witness the load itself.
            status = "STABLE"
            status_reason = None
        else:
            status = "IDLE"
            status_reason = None

        loaded_since = None
        if latest_cycle and latest_cycle["outcome"] == "success" and not latest_cycle["evicted_ts"]:
            loaded_since = latest_cycle["load_completed_ts"]

        result[svc.name] = {
            "enabled": True,
            "status": status,
            "status_reason": status_reason,
            "model": (resident_model["name"] if resident_model else None) or
                     (latest_cycle["model"] if latest_cycle else None),
            "ctx": (resident_model.get("context_length") if resident_model else None) or
                   (latest_cycle["ctx_requested"] if latest_cycle else None),
            "loaded_since": loaded_since,
            "keep_alive_expires_at": resident_expires_at or
                                      (latest_cycle["keep_alive_expires_at"] if latest_cycle else None),
            "reload_count_5m": reload_count_5m,
            "distinct_ctx_15m": distinct_ctx_15m,
            "established_connections": conn["established_count"] if conn else 0,
        }
    return result


@app.get("/api/task_samples")
async def get_task_samples(window_seconds: int = Query(3600, ge=60, le=86400)):
    """Recent per-task token counts, for telling real varied work apart
    from a fixed-signature health-check heartbeat (same token count every
    time). Also flags the single most common signature in the window as
    the likely heartbeat, so the UI doesn't need to guess."""
    cutoff = time.time() - window_seconds
    rows = execute("""
        SELECT timestamp, service_name, task_id, total_tokens
        FROM task_samples WHERE timestamp > ? ORDER BY timestamp
    """, (cutoff,)).fetchall()
    samples = [dict(r) for r in rows]

    from collections import Counter
    signature_counts = Counter(s["total_tokens"] for s in samples)
    likely_heartbeat_signature = signature_counts.most_common(1)[0][0] if signature_counts else None

    return {
        "samples": samples,
        "likely_heartbeat_signature": likely_heartbeat_signature,
        "real_count": sum(1 for s in samples if s["total_tokens"] != likely_heartbeat_signature),
        "heartbeat_count": sum(1 for s in samples if s["total_tokens"] == likely_heartbeat_signature),
    }


@app.get("/api/connections")
async def get_connections(window_seconds: int = Query(3600, ge=60, le=172800)):
    """Connection-count time series, per service — the pileup signal.
    Backed by a connection_window_hours (default 48h) ring, at
    connection_bucket_seconds (default 30s) resolution."""
    cutoff = time.time() - window_seconds
    rows = execute("""
        SELECT timestamp, service_name, established_count
        FROM connection_samples WHERE timestamp > ? ORDER BY timestamp
    """, (cutoff,)).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/gpu_hardware")
async def get_gpu_hardware():
    """Latest hardware-health sample per GPU: ECC + page retirement.

    This is a checklist, not a chart — these values change rarely, and the
    interesting state is binary (clean vs. not), not a trend line. The
    table is a plain upsert keyed by gpu_index, so this is just every row.
    """
    rows = execute("SELECT * FROM gpu_hardware_samples ORDER BY gpu_index").fetchall()
    return [dict(r) for r in rows]


@app.get("/api/ollama/services")
async def get_ollama_services():
    """Ollama service status and loaded models. The table is a plain
    upsert keyed by service_name, so this is just every row."""
    rows = execute("SELECT * FROM ollama_state").fetchall()
    return [dict(r) for r in rows]


@app.get("/api/benchmarks")
async def get_benchmarks(
    limit: int = Query(200, le=2000),
    pool_name: Optional[str] = None,
    model: Optional[str] = None,
    kind: Optional[str] = None
):
    """Get stored benchmark run results."""
    conditions = []
    params = []

    if pool_name:
        conditions.append("pool_name = ?")
        params.append(pool_name)
    if model:
        conditions.append("model = ?")
        params.append(model)
    if kind:
        conditions.append("kind = ?")
        params.append(kind)

    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    rows = execute(f"""
        SELECT * FROM benchmark_results
        {where}
        ORDER BY timestamp DESC
        LIMIT ?
    """, tuple(params + [limit])).fetchall()

    return [dict(r) for r in rows]


@app.get("/api/patterns")
async def get_patterns(
    limit: int = Query(50, le=500),
    since: Optional[float] = None,
    pattern_type: Optional[str] = None,
    severity: Optional[str] = None,
    acknowledged: Optional[bool] = None
):
    """Get detected patterns/anomalies."""
    conditions = []
    params = []

    if since:
        conditions.append("timestamp > ?")
        params.append(since)
    if pattern_type:
        conditions.append("pattern_type = ?")
        params.append(pattern_type)
    if severity:
        conditions.append("severity = ?")
        params.append(severity)
    if acknowledged is not None:
        conditions.append("acknowledged = ?")
        params.append(acknowledged)

    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    rows = execute(f"""
        SELECT * FROM patterns
        {where}
        ORDER BY timestamp DESC
        LIMIT ?
    """, tuple(params + [limit])).fetchall()

    return [dict(r) for r in rows]


@app.post("/api/patterns/{pattern_id}/acknowledge")
async def acknowledge_pattern(pattern_id: int):
    execute("UPDATE patterns SET acknowledged = 1 WHERE id = ?", (pattern_id,))
    return {"ok": True}


# =============================================================================
# PROMPT INSIGHT — recent-prompts panel
# =============================================================================
#
# This app never sees request bodies on its own -- Ollama's own logs never
# contain them, at any verbosity (confirmed live against this box's actual
# journald output). /api/prompt_mirror exists to receive a non-blocking
# COPY of each request from a reverse proxy's `mirror` directive (see
# README "Prompt insight"). The proxy discards whatever this endpoint
# returns, so it never affects real request latency or reliability -- if
# this app is down, real traffic is entirely unaffected.
#
# Deliberately never stores the raw prompt/response body -- only a short
# derived summary (mechanical truncation, or a real ~8-word summary from
# `prompt_insight.summarizer_service` if configured and reachable). This is
# inference traffic that can carry proprietary source or secrets; the full
# body only ever exists transiently in-process for one request.

def _truncate_to_words(text: str, max_words: int, max_chars: int) -> str:
    text = " ".join(text.split())
    if not text:
        return "(empty prompt)"
    words = text.split(" ")
    was_word_truncated = len(words) > max_words
    truncated = " ".join(words[:max_words])
    was_char_truncated = len(truncated) > max_chars
    if was_char_truncated:
        truncated = truncated[:max_chars].rstrip()
    if was_word_truncated or was_char_truncated:
        truncated += "…"
    return truncated


def _extract_prompt_text(endpoint: str, body: Dict[str, Any]) -> Optional[str]:
    if endpoint.endswith("/api/generate"):
        return body.get("prompt")
    # /api/chat (Ollama native) and /v1/chat/completions (OpenAI-compatible)
    # use the identical {"messages": [{"role":..., "content":...}]} shape --
    # same extraction either way.
    if endpoint.endswith("/api/chat") or endpoint.endswith("/v1/chat/completions"):
        messages = body.get("messages") or []
        for m in reversed(messages):
            if m.get("role") == "user" and m.get("content"):
                return m["content"]
        if messages:
            return messages[-1].get("content")
    return None


def _looks_like_verbatim_echo(text: str, prompt_text: str) -> bool:
    """Reject LLM output that's clearly not a summary. Observed live on this
    box: small models asked to summarize a diff- or JSON-shaped prompt often
    slip into completion mode and echo/continue the input instead of
    describing it. Two tells:
    - structural: a code fence or embedded newline -- a real one-or-two-
      sentence summary is always plain single-line text.
    - content: the output is itself a long verbatim substring of the
      prompt it was asked to summarize. Catches single-line echoes a code
      fence/newline check misses entirely -- observed live: given a diff-
      shaped prompt, the model echoed just its filename line back
      ("diff -u tests/test_x.py.orig tests/test_x.py"), one line, no
      fence, so the structural check alone let it straight through.
    20 chars is deliberately short -- a genuine summary reusing a filename
    or a few words from the prompt is normal and fine; this is aimed at
    "the whole output is a chunk of the input," not "shares vocabulary
    with it." A false positive here just means falling back to the
    mechanical truncation, which is already a reasonable result on its
    own -- not something worth tuning finer than this.
    """
    stripped = text.strip()
    if "```" in stripped or "\n" in stripped:
        return True
    if len(stripped) >= 20 and stripped in prompt_text:
        return True
    return False


async def _summarize_via_llm(row_id: int, timestamp: float, prompt_text: str):
    """Runs as a fire-and-forget background task -- never blocks the mirror
    response. Failure (summarizer not running, timeout, bad/echoed output)
    just means the mechanical placeholder ring_insert() already wrote stays
    put; there's nothing to roll back."""
    cfg = settings.prompt_insight
    summarizer = settings.get_ollama_service(cfg.summarizer_service) if cfg.summarizer_service else None
    if not summarizer:
        return
    global SUMMARIZER_BUSY
    if SUMMARIZER_BUSY:
        return  # already working on another prompt -- skip, leave the mechanical fallback in place
    instruction = (
        f"Summarize the following prompt in one or two plain sentences, at most "
        f"{cfg.fallback_max_chars} characters, describing what is being asked for. "
        "No code, no markdown, no diffs, no quotes -- plain text only, and never "
        "repeat any of the prompt's own text verbatim:\n\n" + prompt_text[:4000]
    )
    SUMMARIZER_BUSY = True
    try:
        async with httpx.AsyncClient(timeout=cfg.summarizer_timeout_seconds) as client:
            resp = await client.post(
                f"http://127.0.0.1:{summarizer.port}/api/generate",
                json={"model": summarizer.model, "prompt": instruction, "stream": False,
                      # num_ctx pinned -- left to Ollama's auto-sizing (based on prompt
                      # length), every request could pick a different context size, and
                      # with OLLAMA_MAX_LOADED_MODELS=1 every size change forces a full
                      # model reload. Observed live: alternating short/long mirrored
                      # prompts flipped this between 4096 and 32768 on nearly every call
                      # (~2s reload each) -- exactly what put this service into DEGRADED
                      # via ctx_churn. 4096 comfortably covers instruction +
                      # prompt_text[:4000] + the num_predict budget below.
                      "options": {"num_predict": 80, "num_ctx": 4096}},
            )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        if _looks_like_verbatim_echo(raw, prompt_text):
            return  # model echoed/continued the input -- leave the mechanical fallback in place
        summary = _truncate_to_words(raw, cfg.fallback_max_words, cfg.fallback_max_chars)
        if summary and summary != "(empty prompt)":
            upgrade_ring_row("prompt_summaries", row_id, timestamp, {"summary": summary, "source": "llm"})
    except Exception as e:
        print(f"Prompt summarizer error: {e}")
    finally:
        SUMMARIZER_BUSY = False


@app.post("/api/prompt_mirror")
async def prompt_mirror(request: Request):
    """Receives a mirrored copy of a real request from the reverse proxy.
    Always returns fast -- summarization (if configured) happens in a
    background task, never blocking this response."""
    if not settings.prompt_insight.enabled:
        return {"ok": True}
    try:
        body = await request.json()
    except Exception:
        return {"ok": True}  # not JSON (e.g. a mirrored GET /api/tags) -- nothing to summarize

    endpoint = request.headers.get("x-original-uri", request.url.path)
    service_name = request.headers.get("x-service-name", "unknown")
    prompt_text = _extract_prompt_text(endpoint, body)
    if not prompt_text:
        return {"ok": True}

    cfg = settings.prompt_insight
    timestamp = time.time()
    row_id = ring_insert("prompt_summaries", {
        "timestamp": timestamp,
        "service_name": service_name,
        "model": body.get("model"),
        "endpoint": endpoint,
        "summary": _truncate_to_words(prompt_text, cfg.fallback_max_words, cfg.fallback_max_chars),
        "source": "truncated",
    }, capacity=cfg.ring_capacity)
    cache_raw_prompt(row_id, timestamp, prompt_text)

    if cfg.summarizer_service:
        asyncio.create_task(_summarize_via_llm(row_id, timestamp, prompt_text))

    return {"ok": True}


@app.get("/api/prompt_summaries")
async def get_prompt_summaries(limit: int = Query(20, le=200)):
    rows = execute("SELECT * FROM prompt_summaries ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/prompt_raw/{row_id}")
async def get_prompt_raw(row_id: int, timestamp: float = Query(...)):
    """The actual mirrored prompt text behind a Recent Prompts row's
    summary -- in memory only (see storage.cache_raw_prompt), never in
    data/monitor.db. `timestamp` must match the row exactly, guarding
    against a ring slot that's since been overwritten by a newer prompt."""
    text = get_raw_prompt(row_id, timestamp)
    if text is None:
        return JSONResponse({"error": "not available (evicted or predates last restart)"}, status_code=404)
    return {"text": text}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_connections.append(ws)
    try:
        while True:
            await ws.receive_text()  # Keep alive
    except WebSocketDisconnect:
        ws_connections.remove(ws)


# =============================================================================
# HEALTH / STATUS
# =============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": time.time(),
        "started_at": STARTED_AT,
        "collectors": {
            "gpu": gpu_collector._initialized,
            "ollama": True,
            "log_tailer": True
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8082, log_level="info")