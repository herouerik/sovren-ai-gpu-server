from __future__ import annotations

import asyncio
import json
import re
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import psutil
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlDeviceGetTemperature, nvmlDeviceGetPowerUsage

from src.config import settings


@dataclass
class GPUSample:
    timestamp: float
    gpu_index: int
    name: str
    memory_used_mb: float
    memory_total_mb: float
    memory_free_mb: float
    gpu_utilization_percent: float
    memory_utilization_percent: float
    temperature_c: float
    power_watts: float
    power_limit_watts: float


class GPUCollector:
    def __init__(self):
        self._initialized = False
        self._handles = []

    def _init_nvml(self):
        if not self._initialized:
            nvmlInit()
            for i in range(8):  # Support up to 8 GPUs
                try:
                    handle = nvmlDeviceGetHandleByIndex(i)
                    self._handles.append(handle)
                except:
                    break
            self._initialized = True

    async def collect(self) -> List[GPUSample]:
        self._init_nvml()
        samples = []
        now = time.time()

        for i, handle in enumerate(self._handles):
            try:
                mem = nvmlDeviceGetMemoryInfo(handle)
                util = nvmlDeviceGetUtilizationRates(handle)
                temp = nvmlDeviceGetTemperature(handle, 0)  # GPU core
                power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W

                # Get power limit
                try:
                    power_limit = nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                except:
                    power_limit = 0.0

                name_bytes = nvmlDeviceGetName(handle)
                name = name_bytes.decode() if isinstance(name_bytes, bytes) else name_bytes

                samples.append(GPUSample(
                    timestamp=now,
                    gpu_index=i,
                    name=name,
                    memory_used_mb=mem.used / 1024 / 1024,
                    memory_total_mb=mem.total / 1024 / 1024,
                    memory_free_mb=mem.free / 1024 / 1024,
                    gpu_utilization_percent=util.gpu,
                    memory_utilization_percent=util.memory,
                    temperature_c=temp,
                    power_watts=power,
                    power_limit_watts=power_limit
                ))
            except Exception as e:
                print(f"Error collecting GPU {i}: {e}")

        return samples

    async def store_samples(self, samples: List[GPUSample]):
        """Pure live/snapshot data -- kept in memory only, never persisted
        to disk. See storage.store_gpu_sample()."""
        from src.storage import store_gpu_sample
        for s in samples:
            store_gpu_sample(asdict(s))

    def cleanup(self):
        if self._initialized:
            nvmlShutdown()


@dataclass
class OllamaServiceState:
    timestamp: float
    service_name: str
    port: int
    models: List[Dict[str, Any]]  # From /api/ps
    tags: List[Dict[str, Any]]    # From /api/tags
    status: str  # "healthy", "unhealthy", "starting"


class OllamaStateCollector:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=5.0)

    async def collect(self) -> List[OllamaServiceState]:
        states = []
        for svc in settings.get_ollama_services():
            try:
                port = svc.port
                base = f"http://127.0.0.1:{port}"

                ps_resp = await self.client.get(f"{base}/api/ps")
                tags_resp = await self.client.get(f"{base}/api/tags")

                models = ps_resp.json().get("models", []) if ps_resp.is_success else []
                tags = tags_resp.json().get("models", []) if tags_resp.is_success else []

                states.append(OllamaServiceState(
                    timestamp=time.time(),
                    service_name=svc.name,
                    port=port,
                    models=models,
                    tags=tags,
                    status="healthy"
                ))
            except Exception as e:
                states.append(OllamaServiceState(
                    timestamp=time.time(),
                    service_name=svc.name,
                    port=port,
                    models=[],
                    tags=[],
                    status=f"error: {e}"
                ))

        return states

    async def store_states(self, states: List[OllamaServiceState]):
        """Latest known state only, one row per service. See storage.upsert()."""
        from src.storage import upsert
        for s in states:
            upsert("ollama_state", {
                "service_name": s.service_name,
                "timestamp": s.timestamp,
                "port": s.port,
                "models_json": json.dumps(s.models),
                "tags_json": json.dumps(s.tags),
                "status": s.status,
            })

    async def close(self):
        await self.client.aclose()


@dataclass
class LogEntry:
    timestamp: float
    service_name: str
    level: str
    message: str
    raw_line: str


class LogTailer:
    """Tails journald for Ollama service logs."""
    def __init__(self):
        self._positions: Dict[str, float] = {}
        self._last_entry: Dict[str, tuple] = {}
        # task_id -> {"ttft_ms": ..., "tokens_per_second": ...}, populated as
        # "prompt eval time" / "eval time" lines stream in and consumed
        # (popped) the moment that task's slot release line arrives. See
        # _track_task_metrics().
        self._task_metrics: Dict[int, Dict[str, float]] = {}

    async def tail_service(self, systemd_service: str) -> List[LogEntry]:
        """Get new log lines since the last call for this service.

        journalctl --since is inclusive and only has second resolution, so the
        same boundary line can reappear across consecutive calls — dedupe on
        (timestamp, raw_line) against the last-seen entry from this service
        rather than trusting the window edge to never overlap.
        """
        # First call for a service has no prior position — a fixed 30s lookback
        # bootstraps it without ever fetching unbounded history.
        since_time = self._positions.get(systemd_service, time.time() - 30)
        last_seen = self._last_entry.get(systemd_service)
        try:
            result = subprocess.run([
                "journalctl", "-u", systemd_service,
                "--since", f"@{int(since_time)}",
                "--no-pager", "-o", "json"
            ], capture_output=True, text=True, timeout=5)

            entries = []
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    try:
                        entry = json.loads(line)
                        ts = float(entry.get("__REALTIME_TIMESTAMP", "0")) / 1_000_000
                        msg = entry.get("MESSAGE", "")
                        level = entry.get("PRIORITY", "6")
                        if last_seen is not None and (ts, line) <= last_seen:
                            continue
                        entries.append(LogEntry(
                            timestamp=ts,
                            service_name=systemd_service,
                            level=level,
                            message=msg,
                            raw_line=line
                        ))
                    except:
                        pass

            if entries:
                last = entries[-1]
                self._last_entry[systemd_service] = (last.timestamp, last.raw_line)
                self._positions[systemd_service] = last.timestamp
            return entries
        except Exception as e:
            print(f"Log tail error for {systemd_service}: {e}")
            return []

    def extract_request_info(self, entry: LogEntry) -> Optional[Dict[str, Any]]:
        """Parse GIN access log format to extract request details.

        NOTE ON SCOPE: a GIN access-log line can only ever tell us status,
        duration, client IP, method, and path -- the model name, num_ctx,
        keep_alive, and token counts live in the request BODY, which never
        appears here. Do not add TODOs to "extract model from path" again;
        that data structurally cannot come from this line. Real request
        content (model, token counts, timing) comes from the `slot` lines
        extracted by `extract_slot_info()` below instead, and load-level
        facts (ctx, outcome) come from `extract_lifecycle_event()`.
        """
        # Format: [GIN] 2026/07/16 - 20:03:10 | 200 | 2.406617ms | 192.0.2.10 | GET "/api/tags"
        match = re.search(
            r'\[GIN\].*?\|\s*(\d{3})\s*\|\s*([\d.]+)(ms|s|µs)\s*\|\s*(\S+)\s*\|\s*(\w+)\s*"([^"]+)"',
            entry.message
        )
        if not match:
            return None

        status, duration_val, duration_unit, client_ip, method, path = match.groups()

        # Convert duration to ms
        duration_ms = float(duration_val)
        if duration_unit == "s":
            duration_ms *= 1000
        elif duration_unit == "µs":
            duration_ms /= 1000

        return {
            "timestamp": entry.timestamp,
            "service_name": entry.service_name,
            "status_code": int(status),
            "duration_ms": duration_ms,
            "client_ip": client_ip,
            "method": method,
            "endpoint": path,  # alias for frontend compatibility
            "path": path,
            # Only a real failure now, not "every line" -- the previous
            # version stuffed raw_log in here unconditionally, which made
            # `error IS NULL` false for every row and silently disabled
            # every pattern detector that filtered on it.
            "error": entry.raw_line if int(status) >= 400 else None,
            "raw_log": entry.raw_line
        }

    # ------------------------------------------------------------------
    # Lifecycle events: model load start/success/failure. This is the
    # signal that actually explains reload storms, load-cancel timeout
    # cascades, and context-size churn -- none of which a GIN access-log
    # line can show, since a reload happens *before* any request completes.
    # ------------------------------------------------------------------

    _RE_STARTING = re.compile(r'msg="starting llama-server".*?--model\s+(\S+).*?-c\s+(\d+)')
    _RE_LOADED = re.compile(r'srv\s+llama_server:\s+model loaded')
    _RE_LOAD_FAILED = re.compile(r'msg="Load failed"\s+model=(\S+)\s+error="([^"]*)"')

    def extract_lifecycle_event(self, entry: LogEntry) -> Optional[Dict[str, Any]]:
        """Parse the three journald lines that bound a model load cycle."""
        m = self._RE_STARTING.search(entry.message)
        if m:
            model_path, ctx = m.groups()
            return {
                "kind": "starting",
                "timestamp": entry.timestamp,
                "service_name": entry.service_name,
                "model": model_path.rsplit("/", 1)[-1],
                "ctx_requested": int(ctx),
            }

        if self._RE_LOADED.search(entry.message):
            return {
                "kind": "loaded",
                "timestamp": entry.timestamp,
                "service_name": entry.service_name,
            }

        m = self._RE_LOAD_FAILED.search(entry.message)
        if m:
            model_path, error = m.groups()
            return {
                "kind": "failed",
                "timestamp": entry.timestamp,
                "service_name": entry.service_name,
                "model": model_path.rsplit("/", 1)[-1],
                "error": error,
            }

        return None

    # ------------------------------------------------------------------
    # Slot lines: the only place real per-request token counts and timing
    # actually appear. Not a full substitute for request-body capture
    # (no client IP, no num_ctx here), but real signal where the GIN line
    # has none -- distinguishes an actual varied inference job from a
    # fixed-signature health-check ping (same prompt/response size every
    # time), which is exactly how the SubstrateAuditor heartbeat vs. real
    # opencode traffic was told apart by hand all session.
    # ------------------------------------------------------------------

    _RE_SLOT_RELEASE = re.compile(r'task\s+(\d+)\s+\|\s+stop processing:\s+n_tokens\s*=\s*(\d+)')

    # llama.cpp's own per-task timing summary, logged shortly before slot
    # release -- "prompt eval time" is the prefill phase before the first
    # generated token (TTFT), "eval time" is the generation phase and
    # already carries the average tokens/sec for that task. Neither is
    # estimated here, just parsed straight from the server's own numbers.
    _RE_PROMPT_EVAL_TIME = re.compile(r'task\s+(\d+)\s+\|\s+prompt eval time\s*=\s*([\d.]+)\s*ms')
    _RE_EVAL_TIME = re.compile(
        r'task\s+(\d+)\s+\|\s+eval time\s*=\s*[\d.]+\s*ms\s*/\s*\d+\s*tokens\s*'
        r'\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)'
    )
    _MAX_TRACKED_TASKS = 500  # safety net against a leak if slot release is never seen for a task

    def _track_task_metrics(self, entry: LogEntry) -> None:
        """Opportunistically capture TTFT and avg TPS per task_id as their
        summary lines stream by, for extract_slot_info() to attach once
        that task's slot release line arrives (see _RE_SLOT_RELEASE)."""
        m = self._RE_PROMPT_EVAL_TIME.search(entry.message)
        if m:
            task_id, ttft_ms = m.groups()
            self._task_metrics.setdefault(int(task_id), {})["ttft_ms"] = float(ttft_ms)
            return
        m = self._RE_EVAL_TIME.search(entry.message)
        if m:
            task_id, tps = m.groups()
            self._task_metrics.setdefault(int(task_id), {})["tokens_per_second"] = float(tps)
            if len(self._task_metrics) > self._MAX_TRACKED_TASKS:
                self._task_metrics.clear()

    def extract_slot_info(self, entry: LogEntry) -> Optional[Dict[str, Any]]:
        self._track_task_metrics(entry)
        m = self._RE_SLOT_RELEASE.search(entry.message)
        if not m:
            return None
        task_id, n_tokens = m.groups()
        metrics = self._task_metrics.pop(int(task_id), {})
        return {
            "timestamp": entry.timestamp,
            "service_name": entry.service_name,
            "task_id": int(task_id),
            "total_tokens": int(n_tokens),
            "ttft_ms": metrics.get("ttft_ms"),
            "tokens_per_second": metrics.get("tokens_per_second"),
        }


class ConnectionsCollector:
    """Samples established connections to each service's public port via `ss`.

    This is the one signal that caught the real incident on this box: 12-14
    piled-up connections from a single retry-looping caller, invisible to
    both GPU telemetry (nothing was computing) and the request log (nothing
    had completed yet to log). `ss` is queried directly rather than parsed
    from anywhere else -- there is no other place this count exists.
    """

    async def collect(self) -> List[Dict[str, Any]]:
        samples = []
        now = time.time()
        for svc in settings.get_ollama_services():
            if not svc.enabled:
                continue
            port = svc.effective_public_port()
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ss", "-tn", "state", "established",
                    f"( sport = :{port} or dport = :{port} )",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                lines = stdout.decode(errors="ignore").strip().split("\n")[1:]  # skip header
                samples.append({
                    "timestamp": now,
                    "service_name": svc.name,
                    "public_port": port,
                    "established_count": len([l for l in lines if l.strip()]),
                })
            except Exception as e:
                print(f"Connections collector error for {svc.name}: {e}")
        return samples

    async def store_samples(self, samples: List[Dict[str, Any]]):
        """48h-style ring, bucketed at connection_bucket_seconds resolution
        -- see storage.connection_bucket_upsert(). `peers` is collected for
        live use only (nothing persists or reads it back after this poll)."""
        from src.storage import connection_bucket_upsert
        for s in samples:
            connection_bucket_upsert(s["service_name"], s["public_port"], s["timestamp"], s["established_count"])


class GPUHardwareCollector:
    """Slow-polling GPU hardware health: ECC errors and page retirement.

    Distinct from GPUCollector's load telemetry (util/mem/temp/power, polled
    every ~2s) -- these counters change rarely, and `Pending Page Blacklist`
    is the actual "is this card dying" signal, not utilization or memory.
    Uses `nvidia-smi` directly rather than pynvml: pynvml's ECC/retired-page
    calls return N/A on cards with ECC reporting disabled at the driver
    level (true for GPU 0 on this box) and are awkward to query in bulk;
    nvidia-smi's CSV output handles that uniformly.
    """

    # nvidia-smi -q's ECC block nests "Single Bit"/"Double Bit" -> "Total"
    # under BOTH "Volatile" (since last reset) and "Aggregate" (lifetime) --
    # isolate the Volatile section first or "Aggregate"'s numbers (often
    # 4294967295, i.e. uint32 -1 / "counter not supported") get picked up
    # instead. See docs/... this was verified against this box's real
    # output, not guessed at.
    _RE_VOLATILE_BLOCK = re.compile(r"Volatile\s*\n(.*?)\n\s*Aggregate", re.DOTALL)
    _RE_SINGLE_BIT_TOTAL = re.compile(r"Single Bit.*?Total\s*:\s*(\d+|N/A)", re.DOTALL)
    _RE_DOUBLE_BIT_TOTAL = re.compile(r"Double Bit.*?Total\s*:\s*(\d+|N/A)", re.DOTALL)
    _RE_BLACKLIST = re.compile(r"Pending Page Blacklist\s*:\s*(Yes|No|N/A)")

    async def collect(self) -> List[Dict[str, Any]]:
        now = time.time()
        try:
            count_proc = await asyncio.create_subprocess_exec(
                "nvidia-smi", "--query-gpu=index", "--format=csv,noheader",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            count_out, _ = await asyncio.wait_for(count_proc.communicate(), timeout=10)
            gpu_count = len([l for l in count_out.decode().strip().split("\n") if l.strip()])
        except Exception as e:
            print(f"GPU hardware collector error (count): {e}")
            return []

        samples = []
        for idx in range(gpu_count):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "nvidia-smi", "-i", str(idx), "-q", "-d", "ECC,PAGE_RETIREMENT",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                text = stdout.decode(errors="ignore")
            except Exception as e:
                print(f"GPU hardware collector error (GPU {idx}): {e}")
                continue

            def _first_match(pattern: re.Pattern, s: str) -> Optional[str]:
                m = pattern.search(s)
                return m.group(1) if m else None

            volatile_match = self._RE_VOLATILE_BLOCK.search(text)
            volatile_text = volatile_match.group(1) if volatile_match else ""
            corrected = _first_match(self._RE_SINGLE_BIT_TOTAL, volatile_text)
            uncorrected = _first_match(self._RE_DOUBLE_BIT_TOTAL, volatile_text)
            blacklist = _first_match(self._RE_BLACKLIST, text)

            samples.append({
                "timestamp": now,
                "gpu_index": idx,
                "ecc_corrected_volatile": int(corrected) if corrected and corrected.isdigit() else None,
                "ecc_uncorrected_volatile": int(uncorrected) if uncorrected and uncorrected.isdigit() else None,
                "retired_pages_pending": blacklist == "Yes",
                "throttle_reasons": "",
            })
        return samples

    async def store_samples(self, samples: List[Dict[str, Any]]):
        """Latest known state only, one row per GPU. See storage.upsert()."""
        from src.storage import upsert
        for s in samples:
            upsert("gpu_hardware_samples", s)