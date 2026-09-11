# sovren-ai-gpu-server — GPU Ollama Traffic Monitor

A standalone FastAPI web app for monitoring a GPU server running Ollama. Tracks model
load cycles, connection health, GPU hardware faults, and request traffic, and surfaces
whether the box is actually healthy right now — not just a metrics dump.

## Features

- **Load-cycle tracking**: correlates journald's starting/loaded/failed lines into the
  real unit of health on pipeline-parallel GPU hardware — a model load cycle, not a
  single request or a single GPU (a request never maps to one GPU on this hardware;
  every request touches all of them)
- **Connection pileup detection**: periodic `ss` sampling of the public port — the only
  way to see a queued-but-not-yet-served connection before it ever completes or fails
- **GPU hardware health**: real ECC/page-retirement checks via `nvidia-smi -q`, separate
  from compute/VRAM load telemetry
- **Real-work vs. heartbeat detection**: tells a fixed-signature automated health-check
  probe apart from genuinely varied inference traffic
- **Source identification**: distinguishes callers by IP; per-request PID/process
  attribution is not implemented (see [Data Schema](#data-schema))
- **Pattern detection**: reload storms, load-cancel timeout cascades, context-size
  churn, near-zero keep_alive evictions, connection pileups, latency spikes, delete
  attempts, quota/rate-limit signals, GPU starvation, GPU hardware faults, CPU spillover
- **Dashboard**: single-page, health-first layout — a hero status card, a color-coded
  load-cycle timeline, a GPU hardware checklist, live compute strip, and alerts

## Architecture

```
+-------------------------------------------------------------------------+
|                        sovren-ai-gpu-server                             |
+-------------------------------------------------------------------------+
|  FastAPI App (port 8082)                                                |
|  +- /api/health_status   -> Consolidated "is this healthy right now"    |
|  +- /api/load_cycles     -> Load-cycle history (the real health unit)   |
|  +- /api/connections     -> Established-connection time series          |
|  +- /api/gpu_hardware    -> Latest ECC / page-retirement per GPU        |
|  +- /api/task_samples    -> Real-work vs. heartbeat token signatures    |
|  +- /api/requests        -> Query logged requests                       |
|  +- /api/metrics         -> Aggregated metrics by model/GPU/caller      |
|  +- /api/gpus            -> Live GPU load state (NVML)                  |
|  +- /api/ollama          -> Ollama service state (/api/ps, /tags)       |
|  +- /api/patterns        -> Detected anomalies & patterns               |
|  +- /ws                  -> WebSocket for live updates                  |
+-------------------------------------------------------------------------+
|  Background Collectors                                                  |
|  +- LogTailer + LifecycleTracker -> journald -> requests + load_cycles  |
|  |                                   + task_samples (src/lifecycle.py)  |
|  +- GPUCollector          -> NVML load polling (2s interval)            |
|  +- GPUHardwareCollector  -> nvidia-smi ECC/retired-pages (60s)         |
|  +- ConnectionsCollector  -> `ss` established-connection sampling (5s)  |
|  +- OllamaStateCollector  -> /api/ps, /api/tags polling (5s)            |
|  +- PatternAnalyzer       -> Runs on collected data each tick           |
+-------------------------------------------------------------------------+
|  Storage -- every table bounded by construction, no cleanup job (*)     |
|  +- load_cycles table         -> load attempts, 14-day day-slots       |
|  +- connection_samples table  -> 48h ring, 30s buckets                 |
|  +- gpu_hardware_samples      -> latest-only upsert, one row per GPU   |
|  +- task_samples table        -> slot-line token counts, fixed ring    |
|  +- requests table            -> API calls (GIN log fields), fixed ring|
|  +- gpu_samples                -> in-memory only, never persisted (*)  |
|  +- ollama_state table        -> latest-only upsert, one row/service   |
|  +- patterns table            -> detected anomalies, 14-day day-slots  |
+-------------------------------------------------------------------------+
```
(*) See "Storage retention" below -- src/storage.py's module docstring has
the full mechanism.

## Quick Start

Runs anywhere for dashboard/API development, but real GPU and request data
only appear when run on the actual Ollama GPU server (Linux + NVIDIA + systemd
— see [Requirements](#requirements) below).

```bash
cd sovren-ai-gpu-server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# static/ and data/ aren't committed but are required at startup
mkdir -p static data

# Run the monitor
./run.sh
# or: python -m src.main

# Open dashboard at http://localhost:8082
```

## Requirements

The three live data sources are hard platform dependencies, not optional:

- **GPU load metrics** — `GPUCollector` reads stats via NVML (`pynvml`), which needs
  an NVIDIA GPU + driver. On anything else (e.g. Apple Silicon), `nvmlInit()`
  fails every poll and the in-memory GPU sample ring stays empty — the GPU
  strip will show nothing, silently.
- **GPU hardware health** — `GPUHardwareCollector` shells out to `nvidia-smi -q -d
  ECC,PAGE_RETIREMENT`. Same platform requirement as above; cards without ECC
  reporting enabled at the driver level (consumer GPUs, e.g. an RTX in the same
  box as datacenter cards) report `N/A` for the ECC fields, which is expected —
  `retired_pages_pending` is still meaningful on those cards.
- **Request/lifecycle capture** — `LogTailer` tails Ollama's logs via
  `journalctl -u <service>`, which only exists on systemd-managed Linux. Ollama
  traffic on a machine without journald (e.g. the macOS Ollama.app) is never
  ingested, regardless of how many calls you make. `ConnectionsCollector` similarly
  needs `ss` (iproute2, standard on any modern Linux).

Note also that `config.yaml`'s `server.host`/`server.port` are currently
unused — `src/main.py`'s `__main__` block hardcodes `port=8082`. Change that
line (or invoke uvicorn directly, see [Development](#development)) to run on a
different port.

**Always start it via `./run.sh` or `.venv/bin/python -m uvicorn ...`, never a
bare `uvicorn src.main:app`.** A bare `uvicorn` resolves to whatever's first on
`PATH` (often `~/.local/bin/uvicorn`), which runs against globally-installed
packages instead of this project's pinned `requirements.txt` versions — this
has previously produced a silent `TypeError: unhashable type: 'dict'` from a
mismatched global jinja2 on the very first request. `src/main.py` now checks
this at import time and exits immediately with a clear message instead of
letting that surface as a confusing 500 later.

## Configuration

`config.yaml` (committed, describes this box's actual current topology — one
unified 6-GPU pool, with `public_port` pointed at a reverse proxy that blocks
`DELETE` and forwards everything else through. The proxy itself is
infrastructure you own separately — this tool doesn't ship or require one,
it just distinguishes `port` (Ollama's real bind) from `public_port` (what
clients actually connect to) so collectors watch the right one):

```yaml
ollama:
  services:
    - name: "gpu-unified"
      port: 18434                # Ollama's real (internal-only) bind
      public_port: 11434         # what LAN clients actually connect to
      gpu_ids: [1, 2]
      model: "qwen3-coder-next:Q4_K_M"
      systemd_service: "ollama-unified.service"
    - name: "meta"
      port: 11435
      gpu_ids: [0]
      model: "qwen2.5-coder:7b"
      systemd_service: "ollama-meta.service"
      enabled: false            # provisioned, intentionally never started —
                                 # not an error state, don't alert on it

collectors:
  gpu_poll_interval_seconds: 2
  gpu_hardware_poll_interval_seconds: 60
  ollama_state_poll_interval_seconds: 5
  log_poll_interval_seconds: 1
  connections_poll_interval_seconds: 5

storage:
  db_path: "data/monitor.db"
  event_retention_days: 14        # patterns/alerts, load_cycles, benchmark_results
  connection_window_hours: 48
  connection_bucket_seconds: 30
  raw_ring_capacity: 100000       # requests, task_samples
  gpu_sample_memory_minutes: 240  # in-memory only, never persisted

patterns:
  latency_spike_threshold_ms: 5000
  swap_spike_multiplier: 5.0
  quota_exhaustion_keywords: ["quota exceeded", "quota exhausted", "rate limit", "429"]
  cpu_spillover_threshold_percent: 80
  reload_storm_count_threshold: 5
  reload_storm_window_seconds: 300
  ctx_churn_distinct_values_threshold: 2
  ctx_churn_window_seconds: 900
  near_zero_keep_alive_seconds: 5
  connection_pileup_threshold: 6

server:
  host: "0.0.0.0"  # currently unused, see Requirements
  port: 8082        # currently unused, see Requirements
```

A service's `enabled: false` flag matters: patterns and the hero status card
skip disabled services entirely rather than reporting them as unreachable —
`ollama-meta.service` on this box is real and intentionally dormant, not dead.
`public_port` matters when a reverse proxy sits in front of Ollama (as it does
here): collectors that need to watch real LAN traffic (`ConnectionsCollector`,
the lifecycle tracker's trigger-IP snapshot) watch `public_port`; `port` is
still used to query `/api/ps`/`/api/tags` directly.

### Storage retention

There is no cleanup job and no periodic `DELETE` sweep. Every table is
bounded by the shape of its own writes, not by a background process --
`data/monitor.db` was seen to grow to 10.8GB over a few weeks under the old
accumulate-then-DELETE design (`DELETE` frees rows logically, but SQLite
never shrinks the file for it without a `VACUUM`, which never ran). Each
table uses whichever of these fits the data:

- **In-memory only, never persisted** (`gpu_samples`) — pure live/snapshot
  data that changes every couple of seconds forever. A restart loses the
  last `gpu_sample_memory_minutes` of chart history and refills within a
  couple of poll cycles.
- **Latest-only upsert, one row per key** (`ollama_state`, `gpu_hardware_samples`)
  — every consumer only ever wants "what's true right now," never history.
- **Fixed-capacity ring buffer** (`requests`, `task_samples`) — `id` is a
  wrapped counter (`id = n % raw_ring_capacity`), not an autoincrement;
  `INSERT OR REPLACE` overwrites whatever occupied that slot
  `raw_ring_capacity` writes ago. The table can never exceed that many rows,
  regardless of traffic.
- **Fixed slot per time-bucket** (`connection_samples`) — one row per
  `(service_name, bucket_index)`; a bucket's index recurs every
  `connection_window_hours`, so writing it overwrites what it held one full
  window ago.
- **Fixed day-slots** (`patterns`, `load_cycles`, `benchmark_results`) —
  `day_slot = epoch_day % event_retention_days`. Writing into today's slot
  evicts any stale rows already there from a different epoch day first --
  eviction is a side effect of the write path, not a scheduled job.

See `src/storage.py`'s module docstring and the `ring_insert` /
`day_bucket_insert` / `connection_bucket_upsert` / `upsert` helpers for the
exact mechanism each table uses.

## Data Schema

### load_cycles table (the real health signal)
14-day day-slot retention (see [Storage retention](#storage-retention)) --
`day_slot`/`epoch_day` omitted below, they only govern eviction:
```sql
CREATE TABLE load_cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    service_name TEXT NOT NULL,
    start_ts REAL NOT NULL,            -- when "starting llama-server" fired
    ctx_requested INTEGER,             -- the -c value on that launch
    model TEXT,
    outcome TEXT NOT NULL DEFAULT 'pending',  -- pending | success | failed | superseded
    load_completed_ts REAL,            -- when "model loaded" fired
    failed_ts REAL,                    -- when "Load failed" fired
    duration_s REAL,
    triggering_client_ip TEXT,         -- best-effort `ss` snapshot at start_ts
    keep_alive_expires_at REAL,        -- from /api/ps once resident
    evicted_ts REAL,                   -- when the model disappeared from /api/ps
    resident_duration_s REAL
);
```
This is what actually explains a reload storm, a load-cancel timeout cascade, or a
near-zero `keep_alive` self-eviction — none of which a single request or a single
GPU's telemetry can show, since a reload happens *before* any request completes and
touches every GPU on this hardware, not one.

### requests table
Fixed-capacity ring, `raw_ring_capacity` rows (see [Storage retention](#storage-retention))
-- `id` is a wrapped counter, not an autoincrement:
```sql
CREATE TABLE requests (
    id INTEGER PRIMARY KEY,
    timestamp REAL NOT NULL,
    service_name TEXT NOT NULL,
    endpoint TEXT NOT NULL,            -- /api/generate, /api/chat, /api/ps, /api/tags
    method TEXT NOT NULL,
    client_ip TEXT,
    status_code INTEGER,
    duration_ms REAL,
    error TEXT,                        -- the raw GIN line, only on status >= 400
    -- Present in the schema for future use, always NULL today (see note below):
    client_pid INTEGER, client_process TEXT, model TEXT, prompt_tokens INTEGER,
    completion_tokens INTEGER, total_tokens INTEGER, ttft_ms REAL, tps REAL,
    total_latency_ms REAL, num_ctx INTEGER, gpu_id INTEGER, cpu_percent REAL,
    raw_request JSON, raw_response JSON
);
```

> `LogTailer.extract_request_info` parses Ollama's GIN *access-log* line, which only
> ever carries status/duration/client_ip/method/path — the model name, token counts,
> and `num_ctx` live in the request body, which this line structurally cannot contain.
> Do not re-add a TODO to fix that here; it isn't fixable from this line. Real
> per-request token counts and timing come from `task_samples` (parsed from `slot
> release` log lines instead) and load-level `num_ctx` facts come from `load_cycles`.
> The per-request `gpu_id` column is also permanently unpopulated by design: this
> hardware serves every request across all 6 GPUs in a pipeline, so "which GPU served
> this request" isn't a question with an answer here.

### task_samples table
Same fixed-capacity ring as `requests`:
```sql
CREATE TABLE task_samples (
    id INTEGER PRIMARY KEY,
    timestamp REAL NOT NULL,
    service_name TEXT NOT NULL,
    task_id INTEGER,
    total_tokens INTEGER               -- from `slot release: ... n_tokens=`
);
```
`/api/task_samples` also reports the single most-repeated token count in the window as
`likely_heartbeat_signature` — a fixed automated health-check probe sends the same
prompt every time and gets the same token count back; real work doesn't.

### connection_samples / gpu_hardware_samples / ollama_state
Periodic snapshots (`ss` established-connection counts; `nvidia-smi -q` ECC/retired-page
state; `/api/ps` + `/api/tags`). `connection_samples` is a 48h/30s ring bucket keyed by
`(service_name, bucket_index)`; `gpu_hardware_samples` and `ollama_state` are latest-only
upserts, one row per GPU/service. See `src/storage.py` for exact columns and the
[Storage retention](#storage-retention) section above for the mechanism.

## Pattern Detection

| Pattern | Detection Logic |
|---|---|
| **reload_storm** | N+ load cycles in a rolling window (default: 5 in 300s) |
| **load_cancelled** | A load failed via client-side cancellation, not a crash — a caller's timeout is shorter than this model's load time |
| **ctx_churn** | 2+ distinct `num_ctx` values requested for the same service within a window (default 900s) — forces a reload on every switch |
| **near_zero_keep_alive** | A load succeeded, then evicted within seconds — not a crash, a caller set an explicit near-zero `keep_alive` |
| **connection_pileup** | N+ established connections to the public port (default threshold 6) — a caller is retrying faster than the queue drains |
| **latency_spike** | `duration_ms > threshold` (default 5s) on a real (non-heartbeat) request |
| **delete_attempt** | A `DELETE` hit the public endpoint — logged whether or not the proxy guard blocked it |
| **quota_exhaustion** | A real GIN error line (`error LIKE '[GIN]%'`) contains a quota/rate-limit keyword |
| **gpu_starvation** | High VRAM, near-zero compute — informational; often just idle-but-loaded on this hardware |
| **gpu_hardware_fault** | Pending page retirement, or any uncorrected ECC error — real hardware degradation |
| **cpu_spillover** | System-wide CPU above threshold while all GPUs are idle — inference may be running on CPU instead of GPU |

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/health_status` | Consolidated per-service health: STABLE / RELOADING / DEGRADED / IDLE / disabled |
| `GET /api/load_cycles` | Load-cycle history, filterable by service/outcome/since |
| `GET /api/connections` | Established-connection time series |
| `GET /api/gpu_hardware` | Latest ECC/retired-page sample per GPU |
| `GET /api/task_samples` | Recent per-task token counts + likely heartbeat signature |
| `GET /api/requests` | Paginated request log with filters |
| `GET /api/metrics/summary` | Aggregated stats by model/GPU/caller |
| `GET /api/metrics/timeseries` | Time-series data for charts |
| `GET /api/gpus` | Current GPU load state |
| `GET /api/ollama/services` | Ollama service status & loaded models |
| `GET /api/patterns` | Detected anomalies |
| `WS /ws` | Live updates (GPU samples, connections, lifecycle events, patterns) |

## Dashboard

Open `http://localhost:8082` — single-page, health-first layout:

- **Hero status card** — per service: STABLE / RELOADING / DEGRADED / IDLE, resident
  model, context size, loaded-since time, keep_alive countdown
- **Load-cycle timeline** — color-coded swimlane (green=success, red=failed/cancelled,
  gray=superseded, pulsing yellow=loading now), hover for exact timing/trigger
- **GPU hardware checklist** — ECC/retired-pages per GPU, a checklist not a chart
- **GPU compute strip** — live per-GPU utilization
- **Connections chart** — established-connection count over time
- **Real work vs. heartbeat** — how much of recent traffic is genuine vs. a health probe
- **Patterns & alerts**, **requests by model/service**, **requests by caller IP**

## Development

```bash
# Run with auto-reload
uvicorn src.main:app --reload --host 0.0.0.0 --port 8082

# Run tests (integration tests against the UI/API skip gracefully if
# nothing is listening on :8082 — start the server first for those to run)
pytest tests/

# Check DB
sqlite3 data/monitor.db ".schema"
```

## License

This project is released under the [MIT License](LICENSE) — see `LICENSE` for
the full text.
