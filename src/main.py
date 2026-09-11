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

from src.config import settings
from src.storage import get_db, execute, query_all, query_one, ring_insert, latest_gpu_samples, query_gpu_samples
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
                        }, capacity=settings.storage.raw_ring_capacity)
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
    by_model = execute("""
        SELECT COALESCE(model, service_name) as model_or_service, service_name,
               COUNT(*) as request_count,
               AVG(duration_ms) as avg_latency_ms,
               SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as error_count
        FROM requests
        WHERE timestamp > ?
        GROUP BY COALESCE(model, service_name), service_name
        ORDER BY request_count DESC
    """, (cutoff,)).fetchall()

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
    by_caller = execute("""
        SELECT client_ip,
               COUNT(*) as request_count,
               AVG(duration_ms) as avg_latency_ms,
               SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as errors
        FROM requests
        WHERE timestamp > ? AND client_ip IS NOT NULL
        GROUP BY client_ip
        ORDER BY request_count DESC
        LIMIT 20
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
        "by_model": [dict(r) for r in by_model],
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
        "collectors": {
            "gpu": gpu_collector._initialized,
            "ollama": True,
            "log_tailer": True
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8082, log_level="info")