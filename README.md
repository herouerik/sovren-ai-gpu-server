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
- **Prompt insight** *(opt-in, see [Prompt insight](#prompt-insight))*: one-line,
  scrollable summaries (up to `fallback_max_chars`, default 200) of the most recent
  prompts — mechanical by default, or a real summary from a small local model if you
  point one at it — plus real per-request TTFT/prefill/decode metrics, attached
  asynchronously once each request actually completes
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
|  +- /api/prompt_summaries-> Recent one-line prompt summaries (opt-in)   |
|  +- /api/prompt_mirror   -> Reverse-proxy mirror target (opt-in)        |
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
|  +- prompt_summaries table    -> summaries only, fixed ring (opt-in)   |
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
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# config.yaml is this box's own real topology and is gitignored -- copy
# the checked-in template and edit it for your machine (see Configuration)
cp config.yaml.example config.yaml

# static/ and data/ aren't committed but are required at startup
mkdir -p static data

# Run the monitor
./run.sh
# or: python -m src.main

# Open dashboard at http://localhost:8082
```

> **macOS note:** use `python3` (not `python`) for the venv step above — macOS
> doesn't ship a `python` command by default, only `python3`, so `python -m venv`
> fails with something like `command not found: python`/`no such file or
> directory: pip` (there's no `.venv` to hold a `pip` if the venv step never
> ran). Once the venv is created with `python3`, everything inside it —
> `pip`, `python`, `./run.sh`, `python -m src.main` — resolves correctly on its
> own, no further changes needed. This is purely a venv-creation quirk, not a
> platform requirement: as noted in [Requirements](#requirements) below, the
> dashboard/API run fine on macOS for development; only the three live data
> sources (GPU/NVML, GPU hardware/`nvidia-smi`, and log capture/`journalctl`)
> are Linux+NVIDIA-only and degrade to empty data instead of erroring.

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

`config.yaml` is gitignored — it describes one specific box's actual current
topology (which differs per machine: the real GPU server vs. a laptop doing
dashboard dev, say), so it can't be a single committed file shared by every
checkout. `config.yaml.example` is the committed template; copy it to
`config.yaml` and edit the copy for your own box (see [Quick
Start](#quick-start)). The example below matches this project's own GPU
server: one unified 6-GPU pool, with `public_port` pointed at a reverse
proxy that blocks `DELETE` and forwards everything else through. The proxy
itself is infrastructure you own separately — this tool doesn't ship one
running by default, it just distinguishes `port` (Ollama's real bind) from
`public_port` (what clients actually connect to) so collectors watch the
right one. Worked examples of that proxy tier — DELETE-guard, rate
limiting, and the mirror this app needs for Prompt insight below, for both
Linux (the GPU server's own setup) and macOS — are in `deploy/`, see
[Prompt insight](#prompt-insight)):

> **Upgrading an existing checkout** (e.g. the GPU server itself): before
> pulling this change, back up your real `config.yaml`
> (`cp config.yaml config.yaml.bak`) — it used to be a tracked file, so a
> plain `git pull` on an unmodified working tree will delete it outright
> once it's removed from tracking upstream (`git` applies the upstream
> "file removed" diff to your identical local copy). After pulling, if
> `config.yaml` is gone, just restore it: `cp config.yaml.bak config.yaml`
> (it's gitignored now, so this won't happen again).

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
  event_retention_days: 14        # patterns/alerts, load_cycles
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

prompt_insight:      # see "Prompt insight" below -- off by default
  enabled: false
  summarizer_service: "meta"
  summarizer_timeout_seconds: 20
  fallback_max_words: 40
  fallback_max_chars: 200
  ring_capacity: 100
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
- **Fixed day-slots** (`patterns`, `load_cycles`) —
  `day_slot = epoch_day % event_retention_days`. Writing into today's slot
  evicts any stale rows already there from a different epoch day first --
  eviction is a side effect of the write path, not a scheduled job.

See `src/storage.py`'s module docstring and the `ring_insert` /
`day_bucket_insert` / `connection_bucket_upsert` / `upsert` helpers for the
exact mechanism each table uses.

### Prompt insight

The "Recent Prompts" dashboard panel needs to know what a request actually
said, and this app has no access to that on its own — confirmed live
against this box's real journald output, Ollama's logs never contain
prompt/response content at any verbosity, only access-log lines and
slot-timing metadata. The only way to see it is to have something in the
traffic path hand this app a copy.

**Off by default** (`prompt_insight.enabled: false`) and harmless to leave
on with nothing configured — `/api/prompt_mirror` just never receives
anything, so nothing is ever stored.

**How to turn it on** — add an nginx `mirror` directive to whatever already
proxies to Ollama. This sends a non-blocking *copy* of each request; nginx
discards the copy's response, so it never adds latency to the real request
and never affects reliability if this app is down.

`deploy/nginx-ollama.conf.example` is a full worked example — the mirror
block this feature needs, plus the DELETE-guard and per-path rate limit
this fleet actually runs in front of Ollama (see the comments in that file
for why each piece is there; the mirror is required for this feature, the
rest is hardening you may or may not want). `deploy/conf.d-rate-limits.conf.example`
is its companion (`limit_req_zone` has to live outside the `server {}`
block). Both are `.example` files, not active config — copy them into your
own nginx tree, replace `127.0.0.1:18434` and `127.0.0.1:8082` if your
Ollama/monitor bind elsewhere, and adjust `X-Service-Name` to match a
`name` in this app's `config.yaml`. If you only want the mirror and
nothing else, everything below the two `if ($request_method = DELETE)`
blocks and the `limit_req` lines is optional.

Then `sudo nginx -t && sudo systemctl reload nginx`, and set
`prompt_insight.enabled: true` in `config.yaml` (requires restarting this
app — YAML config is only read at startup).

**On macOS** (Ollama.app, Homebrew nginx, no systemd) the mechanics differ
enough to warrant a separate example: `deploy/nginx-ollama-macos.conf.example`
covers moving Ollama's own bind off the port nginx needs to take over
(`launchctl setenv OLLAMA_HOST ...` + relaunching Ollama.app), Homebrew's
`servers/*.conf` layout instead of `sites-available`, and `brew services`
instead of `systemctl`. Same mirror mechanism underneath; only the install
steps and paths change.

**Summarization** — every mirrored prompt gets a mechanical fallback first
(first `fallback_max_words` words / `fallback_max_chars` chars, whichever
is shorter, stored immediately, synchronously). If
`prompt_insight.summarizer_service` names a reachable entry in
`ollama.services`, a background task then asks it for a real one-or-two-
sentence summary (same `fallback_max_chars` cap applied to whatever it
returns, so a model that ignores the length instruction still can't blow
past it) and upgrades the stored row in place — this never blocks the
mirror response, and a summarizer that's unset, unreachable, or slow just
means every prompt stays on the mechanical fallback, silently, no error.
Point it at hardware that isn't part of your real inference pool if you
have any (an otherwise-idle GPU, a CPU-only Ollama instance) — pointing it
at the same pool being monitored means summarization competes with real
inference for capacity, and can time out under load (observed directly
while building this: a real generate call to the busy production pool on
this box timed out at 20s with zero bytes back).

Each row also gets real per-request TTFT/prefill_tps/decode_tps, attached
asynchronously once the underlying request actually completes (see
`storage.attach_prompt_metrics()`) — there's no shared id between a
mirrored prompt and its eventual completion, so this is a best-effort
correlation by timing: reliable (not a guess) for any service running
`OLLAMA_NUM_PARALLEL=1`, since exactly one request is ever in flight at a
time. Shows "pending…" until that arrives.

**No journald at all (e.g. macOS)** — `ttft_ms`/`decode_tps` can never be
attached this way, since they're parsed from journald's own slot-timing
lines; every real prompt would show "pending…" forever with no way to tell
a genuinely still-in-flight request from one that already succeeded or
failed. Setting a service's `access_log_path` (in `config.yaml`) to your
reverse proxy's own access log file gives a coarser but platform-agnostic
fallback: `AccessLogTailer` (`src/collectors.py`) tails that file directly
(no journald needed) and, on each completed `POST` to an inference
endpoint, best-effort correlates it the same way (`storage.
attach_prompt_status()`) to set `status` (`pending` / `completed` /
`failed`) and `latency_ms` on the Recent Prompts row. No token counts or
TTFT breakdown — just whether the real request eventually returned, and
with what HTTP status (nginx's own `499` for a client that gave up before
Ollama responded, e.g.). Unset (the default): this feature is simply off.

**Privacy** — `data/monitor.db` only ever stores the derived summary
(bounded by `fallback_max_chars`), never the raw prompt or response body.
This matters because inference traffic through a coding-agent pool can
carry proprietary source or secrets; don't widen `_extract_prompt_text`
in `src/main.py` to persist more than that in the database without
thinking through what you're now storing at rest.

One deliberate, narrow exception: hovering a summary in the dashboard
shows the real prompt text it came from (`GET /api/prompt_raw/{id}`),
for when a summary is too compressed or too strange to make sense of on
its own. That text lives only in server memory (`storage.
cache_raw_prompt`), capped at 4000 chars per entry and the same ring size
as `prompt_summaries` — it is never written to `data/monitor.db`, never
backed up, and gone on restart. Still real inference content sitting in
RAM for a while, so this is a real (if bounded and ephemeral) exception
to "only the summary is ever kept" — worth knowing if you're deploying
this somewhere the process's own memory needs to be trusted, not just the
database file.

## Watchdog

Off by default (`watchdog.enabled: false`) — this is the one feature in
this app that takes a real production action (`systemctl restart`) instead
of only observing. Built from a real incident: a request was mid-generation,
healthy, actively decoding tokens -- then a GPU hit a PCIe Uncorrectable
(Non-Fatal) error mid-CUDA-op, and llama-server went silent instantly. No
crash, no error surfaced anywhere in its own logs, no further progress
ever — just the process pegged near 100% CPU and every subsequent request
admitted then aborted, for hours, until a human happened to notice.

**Detection** (`src/watchdog.py`): a service counts as wedged when real
traffic arrived but llama-server logged **zero** real progress — not even a
slow prefill/decode tick — for the entire `wedge_window_seconds` (default
1200s/20min). "Progress" means a `slot print_timing` or `slot launch_slot_`
journald line (`LogTailer.is_progress_line()`), tracked per-service in
`service_activity.last_progress_ts`.

"Real traffic arrived" is read from **`prompt_summaries`, not the
`requests` table** — this is load-bearing, not a style choice, and got
this wrong twice before shipping (both caught by replaying the logic
against real historical incident data rather than trusting it on paper):

1. `requests` is fed by Ollama's own GIN access-log line, written only
   once a request *completes*. A genuine wedge is, by definition, a
   request that never completes — so during the real 2026-09-25 incident,
   `requests` has **zero** rows for the entire 6-hour wedge. A signal that
   goes silent exactly when there's something to judge can only ever fire
   by accident.
2. That accident happened: this box's own `keepwarm_gpu_ollama.py` cron
   sends an empty-prompt keep-alive touch to Ollama's internal bind every
   5 minutes, bypassing nginx, always `200` in ~230ms, zero real
   inference. Counting that as "traffic present" made three separate,
   genuinely healthy, merely-*idle* periods look wedged and triggered
   three unnecessary restarts in under 15 hours — each one a real, if
   brief, outage of its own (the cold-load gap after any restart), and
   each one silently spending a slot of `max_restarts_per_day` for
   nothing real.

`prompt_summaries` fixes both: it's written synchronously the moment
nginx's mirror delivers a copy of a request — at *arrival* time, not
completion — so it sees traffic a genuine wedge swallows entirely, and
since keepwarm's calls bypass nginx and never reach the mirror, they never
appear here at all (no IP heuristic needed — the data source itself
excludes them by construction). The real cost: **this feature requires
`prompt_insight.enabled: true`** and a working nginx mirror (see [Prompt
insight](#prompt-insight)). Without it, `prompt_summaries` never has real
data for any service, and the watchdog has no safe way to tell "wedged"
from "quiet, no one asked" — so it logs one message and stays silent
rather than falling back to `requests` and repeating mistake #1.

The same `last_progress_ts` signal is also what tells a genuine wedge
apart from this fleet's other real incident (an oversized accumulated
context + a client timeout shorter than this hardware's real prefill
throughput — looked identical from the outside, but was real work, just
slow). That incident logged progress every ~10-15s for its whole
multi-minute duration and eventually completed; a genuine wedge logs
nothing, for the entire window, no matter how long you wait.

**Remediation**: `sudo -n systemctl restart <systemd_service>` for the
wedged service. `-n` (non-interactive) fails fast with a clear message if
the sudoers grant below isn't set up, instead of hanging on a password
prompt nothing can ever answer. Every attempt — whether it actually ran,
or was suppressed — is logged to `watchdog_actions` (`GET
/api/watchdog_actions`, and the dashboard's "Watchdog Actions" panel), so
it's always auditable why a service did or didn't get restarted.

**Safety knobs**, both needed because a restart is a real, consequential
action:
- `restart_cooldown_seconds` (default 1200) — won't attempt another
  restart for the same service within this long of the last attempt, so a
  restart gets time to actually take effect (cold load + some real
  traffic) before being re-judged.
- `max_restarts_per_day` (default 3) — hard ceiling regardless of
  cooldown. If a service wedges this many times in a rolling 24h,
  something a restart doesn't fix is going on (failing hardware, not a
  transient hang) — this stops trying and raises a `service_wedged`
  pattern for a human instead of restart-looping forever.

**Required sudoers grant** — scoped to exactly the restart command(s) this
needs, nothing broader:
```bash
sudo visudo -f /etc/sudoers.d/gpu-server-watchdog
```
```
<user> ALL=(root) NOPASSWD: /usr/bin/systemctl restart ollama-unified.service, /usr/bin/systemctl restart ollama-meta.service
```
Replace `<user>` with whatever user runs this app, and the two unit names
with your own services' `systemd_service` values. Without this grant, every
restart attempt fails cleanly (logged to `watchdog_actions` as `failed`,
detail `sudo: a password is required`) rather than hanging — the watchdog
still detects and alerts, it just can't act.

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
    total_tokens INTEGER,   -- from `slot release: ... n_tokens=`
    ttft_ms REAL,           -- from `... prompt eval time = N ms ...` (the prefill phase)
    prefill_tps REAL,       -- same line's own tokens/sec, for the prefill phase
    decode_tps REAL         -- from `... eval time = ... N tokens per second)`, the generation phase
);
```
`ttft_ms`/`prefill_tps`/`decode_tps` are llama.cpp's own per-task timing, not estimated
and not derived from the GIN access log (which structurally can't carry them — see
`requests` below). Kept as two separate rates, not one blended "tokens/sec" — prefill is
highly parallel and normally much faster than decode, and the ratio between them varies a
lot across models and context sizes. `/api/metrics/summary`'s `by_model` rows include the
window average of all three, joined in by `service_name`.

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
| **reload_storm** | N+ load cycles in a rolling window (default: 5 in 300s) — excludes fast-rejects (see below) |
| **load_cancelled** | A load failed via client-side cancellation, not a crash — a caller's timeout is shorter than this model's load time. Excludes fast-rejects (see below): a `failed` row that resolved in under `fast_reject_max_duration_seconds` (default 15s) was never a real load attempt in the first place, so this description would be actively wrong for it |
| **ctx_churn** | 2+ distinct `num_ctx` values requested for the same service within a window (default 900s) — forces a reload on every switch |
| **model_churn** | 2+ distinct models requested for the same service within the same window — models fighting over one single-model-at-a-time pool. Deliberately does *not* fire for a slow, intentional model switch (e.g. a bandit comparing sovereign candidates every few hours) — a single reload event can never produce 2 distinct values on its own; only 2+ separate reloads inside the same short window do |
| **rejected_model_mismatch** | N+ requests (default 3 in 300s) naming a model that doesn't match what's actually resident — distinct from `model_churn`, which only sees real load attempts via `load_cycles`. A caller whose `num_ctx`/model mismatch is severe enough trips Ollama's own fast-reject admission check (instant 503, no load attempt, no journald "starting llama-server" line at all) — `load_cycles`-based detection is structurally blind to this. Reads the mirrored request's own declared `model` field instead (`prompt_summaries`), so it catches rejected-before-ever-attempting-to-load traffic too — real, wasted requests fighting a single-model-at-a-time pool even though nothing ever actually reloads |
| **near_zero_keep_alive** | A load succeeded, then evicted within seconds — not a crash, a caller set an explicit near-zero `keep_alive` |
| **connection_pileup** | N+ established connections to the public port (default threshold 6) — a caller is retrying faster than the queue drains |
| **latency_spike** | `duration_ms > threshold` (default 5s) on a real (non-heartbeat) request |
| **delete_attempt** | A `DELETE` hit the public endpoint — logged whether or not the proxy guard blocked it |
| **quota_exhaustion** | A real GIN error line (`error LIKE '[GIN]%'`) contains a quota/rate-limit keyword |
| **gpu_starvation** | High VRAM, near-zero compute — informational; often just idle-but-loaded on this hardware |
| **gpu_hardware_fault** | Pending page retirement, or any uncorrected ECC error — real hardware degradation |
| **cpu_spillover** | System-wide CPU above threshold while all GPUs are idle — inference may be running on CPU instead of GPU |
| **service_wedged** | Real traffic arrived but llama-server logged zero progress for the whole detection window — see [Watchdog](#watchdog). Only emitted when `watchdog.enabled: true` |

**Fast-rejects, and why `reload_storm`/`load_cancelled` exclude them**: a `load_cycles` row with
`outcome = 'failed'` isn't always a real cold-load attempt that failed partway through. A caller
sending a `num_ctx`/model that doesn't match what's resident can trigger a `starting` →`failed`
pair that resolves in milliseconds-to-seconds — real hardware on this fleet has never completed a
genuine cold load (success or failure) in under ~90s, so anything faster was never actually
loading weights. One stale, already-fixed source of these (a misconfigured health-check probe,
see the handover doc's Turns 70-73) can leave hundreds of these rows sitting in the 14-day
retention window, which would otherwise make a perfectly healthy pool's Load Cycle Timeline look
almost entirely red and could fire false `reload_storm`/`load_cancelled` alerts. Both the
dashboard timeline (rendered as a distinct brown "fast-reject" segment, not red) and these two
detectors exclude `failed` rows under `patterns.fast_reject_max_duration_seconds` (default 15s) —
this is a display/detection distinction only, the underlying `load_cycles` rows are untouched.

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
| `GET /api/requested_models` | Every distinct model actually *requested* per service in a window (default 1h), with a count and whether it matches what's resident — the data source for the "Requested Models" dashboard panel and the `rejected_model_mismatch` pattern; catches wasted/rejected traffic that never shows up in `load_cycles` |
| `GET /api/prompt_summaries` | Recent one-line prompt summaries (see [Prompt insight](#prompt-insight)) |
| `GET /api/prompt_raw/{id}` | Real prompt text behind one summary, in-memory only (see [Prompt insight](#prompt-insight)) |
| `POST /api/prompt_mirror` | Reverse-proxy mirror target — not for direct use, see [Prompt insight](#prompt-insight) |
| `GET /api/watchdog_actions` | Full audit trail of auto-restart attempts, run or suppressed (see [Watchdog](#watchdog)) |
| `WS /ws` | Live updates (GPU samples, connections, lifecycle events, patterns) |

## Dashboard

Open `http://localhost:8082` — single-page, health-first layout:

- **Hero status card** — per service: STABLE / RELOADING / DEGRADED / IDLE, resident
  model, context size, loaded-since time, keep_alive countdown
- **Load-cycle timeline** — color-coded swimlane (green=success, red=failed/cancelled,
  brown=fast-reject — a mismatched request rejected in seconds, never a real load attempt,
  see [Pattern Detection](#pattern-detection)'s `fast_reject_max_duration_seconds` note,
  gray=superseded, pulsing yellow=loading now), hover for exact timing/trigger
- **GPU hardware checklist** — ECC/retired-pages per GPU, a checklist not a chart
- **GPU compute strip** — live per-GPU utilization
- **Connections chart** — established-connection count over time
- **Real work vs. heartbeat** — how much of recent traffic is genuine vs. a health probe
- **Patterns & alerts**, **requested models vs. resident** (every model actually requested
  per pool, not just what's loaded — flags any row that doesn't match what's resident, the
  only place a rejected-before-loading mismatch is visible; see `rejected_model_mismatch`
  in [Pattern Detection](#pattern-detection)), **requests by model/service**, **requests by
  caller IP**
- **Recent prompts** — scrollable, one-line-each summaries of the last 20 prompts
  (empty until [Prompt insight](#prompt-insight) is configured)
- **Watchdog actions** — audit trail of auto-restart attempts, run or suppressed
  (empty unless [Watchdog](#watchdog) is enabled)
- **Header** — live clock plus "monitoring since" (this process's own start time,
  not the history depth of any individual table — see [Storage retention](#storage-retention)
  for what each table actually covers)

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
