from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import psutil

from src.config import settings
from src.storage import get_db, execute, day_bucket_insert, latest_gpu_samples


@dataclass
class Pattern:
    timestamp: float
    pattern_type: str
    severity: str
    description: str
    related_request_id: Optional[int] = None
    related_gpu_index: Optional[int] = None
    related_load_cycle_id: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


class PatternAnalyzer:
    def __init__(self):
        self._baselines: Dict[str, Dict[str, float]] = {}

    async def analyze(self) -> List[Pattern]:
        patterns = []
        # Load-cycle-based detectors — these are the ones that actually
        # matched real incidents this box hit (reload storms, timeout
        # cascades, context churn, near-zero keep_alive self-eviction).
        patterns.extend(await self._detect_reload_storm())
        patterns.extend(await self._detect_load_cancelled())
        patterns.extend(await self._detect_ctx_churn())
        patterns.extend(await self._detect_model_churn())
        patterns.extend(await self._detect_near_zero_keep_alive())
        # Connection-based
        patterns.extend(await self._detect_connection_pileup())
        # Request-log-based (real data now that `error` and `duration_ms`
        # are populated correctly)
        patterns.extend(await self._detect_latency_spikes())
        patterns.extend(await self._detect_delete_attempts())
        patterns.extend(await self._detect_quota_exhaustion())
        # GPU-based
        patterns.extend(await self._detect_gpu_starvation())
        patterns.extend(await self._detect_gpu_hardware_fault())
        patterns.extend(await self._detect_cpu_spillover())
        return patterns

    # ------------------------------------------------------------------
    # Load-cycle detectors
    # ------------------------------------------------------------------

    async def _detect_reload_storm(self) -> List[Pattern]:
        """N+ load attempts in a rolling window — the headline symptom of
        every incident this box hit (config drift, timeout cascades,
        sentinel-404 retry loops all showed up as this first)."""
        threshold = settings.patterns.reload_storm_count_threshold
        window = settings.patterns.reload_storm_window_seconds
        cutoff = time.time() - window

        rows = execute("""
            SELECT service_name, COUNT(*) as c, MIN(id) as first_id, MAX(id) as last_id
            FROM load_cycles WHERE start_ts > ? GROUP BY service_name
        """, (cutoff,)).fetchall()

        patterns = []
        for row in rows:
            if row["c"] >= threshold:
                patterns.append(Pattern(
                    timestamp=time.time(),
                    pattern_type="reload_storm",
                    severity="critical" if row["c"] >= threshold * 2 else "warning",
                    description=f"{row['service_name']}: {row['c']} model reloads in the last "
                                f"{window}s (threshold {threshold})",
                    related_load_cycle_id=row["last_id"],
                    details={"service": row["service_name"], "count": row["c"], "window_seconds": window},
                ))
        return patterns

    async def _detect_load_cancelled(self) -> List[Pattern]:
        """A load that failed via client-side cancellation, not an actual
        crash — distinguishes "someone's timeout is shorter than this
        model's load time" from a real hardware/OOM failure."""
        cutoff = time.time() - 300
        rows = execute("""
            SELECT id, service_name, model, ctx_requested, duration_s, timestamp
            FROM (SELECT *, start_ts as timestamp FROM load_cycles)
            WHERE outcome = 'failed' AND failed_ts > ?
        """, (cutoff,)).fetchall()

        patterns = []
        for row in rows:
            patterns.append(Pattern(
                timestamp=row["timestamp"],
                pattern_type="load_cancelled",
                severity="warning",
                description=f"{row['service_name']}: load cancelled after {row['duration_s']:.0f}s "
                            f"(ctx={row['ctx_requested']}) — a caller's timeout is shorter than this "
                            f"model's load time",
                related_load_cycle_id=row["id"],
                details={"service": row["service_name"], "model": row["model"],
                          "ctx_requested": row["ctx_requested"], "duration_s": row["duration_s"]},
            ))
        return patterns

    async def _detect_ctx_churn(self) -> List[Pattern]:
        """More than one distinct num_ctx requested for the same model
        within a window — the actual multi-day root cause chased across
        this box's whole incident history (config drift between repos,
        then a client omitting num_ctx, then a hardcoded mismatch)."""
        threshold = settings.patterns.ctx_churn_distinct_values_threshold
        window = settings.patterns.ctx_churn_window_seconds
        cutoff = time.time() - window

        rows = execute("""
            SELECT service_name, GROUP_CONCAT(DISTINCT ctx_requested) as values_str,
                   COUNT(DISTINCT ctx_requested) as distinct_count, MAX(id) as last_id
            FROM load_cycles
            WHERE start_ts > ? AND ctx_requested IS NOT NULL
            GROUP BY service_name
        """, (cutoff,)).fetchall()

        patterns = []
        for row in rows:
            if row["distinct_count"] >= threshold:
                patterns.append(Pattern(
                    timestamp=time.time(),
                    pattern_type="ctx_churn",
                    severity="warning",
                    description=f"{row['service_name']}: {row['distinct_count']} different context "
                                f"sizes requested in {window}s ({row['values_str']}) — forces a full "
                                f"reload on every switch",
                    related_load_cycle_id=row["last_id"],
                    details={"service": row["service_name"], "values": row["values_str"],
                              "distinct_count": row["distinct_count"]},
                ))
        return patterns

    async def _detect_model_churn(self) -> List[Pattern]:
        """Same idea as ctx_churn, but tracking model identity instead of
        context size -- two different models sharing the same context size
        would otherwise never trip that check. Same window/threshold: a
        deliberate model switch every few hours (e.g. a slow bandit
        comparing sovereign candidates) never fires this, since a single
        reload event can't produce 2 distinct values on its own -- it
        takes >=2 separate reload events inside the same window, which
        only genuine rapid flapping between models produces."""
        threshold = settings.patterns.ctx_churn_distinct_values_threshold
        window = settings.patterns.ctx_churn_window_seconds
        cutoff = time.time() - window

        rows = execute("""
            SELECT service_name, GROUP_CONCAT(DISTINCT model) as values_str,
                   COUNT(DISTINCT model) as distinct_count, MAX(id) as last_id
            FROM load_cycles
            WHERE start_ts > ? AND model IS NOT NULL
            GROUP BY service_name
        """, (cutoff,)).fetchall()

        patterns = []
        for row in rows:
            if row["distinct_count"] >= threshold:
                patterns.append(Pattern(
                    timestamp=time.time(),
                    pattern_type="model_churn",
                    severity="warning",
                    description=f"{row['service_name']}: {row['distinct_count']} different models "
                                f"requested in {window}s ({row['values_str']}) — models fighting for "
                                f"the same single-model-at-a-time pool",
                    related_load_cycle_id=row["last_id"],
                    details={"service": row["service_name"], "values": row["values_str"],
                              "distinct_count": row["distinct_count"]},
                ))
        return patterns

    async def _detect_near_zero_keep_alive(self) -> List[Pattern]:
        """A load succeeded, then evicted almost immediately — not a
        crash, a caller explicitly setting a near-zero keep_alive. Looks
        identical to a crash on a GPU chart; this is the only way to tell
        the two apart."""
        threshold = settings.patterns.near_zero_keep_alive_seconds
        cutoff = time.time() - 3600

        rows = execute("""
            SELECT id, service_name, model, load_completed_ts, evicted_ts, resident_duration_s
            FROM load_cycles
            WHERE outcome = 'success' AND evicted_ts IS NOT NULL
              AND resident_duration_s IS NOT NULL AND resident_duration_s <= ?
              AND load_completed_ts > ?
        """, (threshold, cutoff)).fetchall()

        patterns = []
        for row in rows:
            patterns.append(Pattern(
                timestamp=row["evicted_ts"],
                pattern_type="near_zero_keep_alive",
                severity="warning",
                description=f"{row['service_name']}: model resident for only "
                            f"{row['resident_duration_s']:.1f}s after a successful load — a caller set "
                            f"an explicit near-zero keep_alive, this is not a crash",
                related_load_cycle_id=row["id"],
                details={"service": row["service_name"], "model": row["model"],
                          "resident_duration_s": row["resident_duration_s"]},
            ))
        return patterns

    # ------------------------------------------------------------------
    # Connection-based
    # ------------------------------------------------------------------

    async def _detect_connection_pileup(self) -> List[Pattern]:
        """N+ established connections to the public port — the signal
        that caught the real 12-14-connection cascading failure storm.
        Nothing in GPU telemetry or the request log can see a
        queued-but-not-yet-served connection; only `ss` can."""
        threshold = settings.patterns.connection_pileup_threshold
        cutoff = time.time() - 60

        rows = execute("""
            SELECT service_name, established_count, timestamp
            FROM connection_samples
            WHERE timestamp = (
                SELECT MAX(timestamp) FROM connection_samples cs2 WHERE cs2.service_name = connection_samples.service_name
            ) AND timestamp > ?
        """, (cutoff,)).fetchall()

        patterns = []
        for row in rows:
            if row["established_count"] >= threshold:
                patterns.append(Pattern(
                    timestamp=row["timestamp"],
                    pattern_type="connection_pileup",
                    severity="critical" if row["established_count"] >= threshold * 2 else "warning",
                    description=f"{row['service_name']}: {row['established_count']} connections piled up "
                                f"(threshold {threshold}) — a caller is retrying faster than the queue drains",
                    details={"service": row["service_name"], "established_count": row["established_count"]},
                ))
        return patterns

    # ------------------------------------------------------------------
    # Request-log-based
    # ------------------------------------------------------------------

    async def _detect_latency_spikes(self) -> List[Pattern]:
        """Individual slow requests — real signal now that `duration_ms`
        (not the never-populated `total_latency_ms`) is what's checked."""
        threshold = settings.patterns.latency_spike_threshold_ms
        cutoff = time.time() - 300

        rows = execute("""
            SELECT id, service_name, endpoint, client_ip, duration_ms, timestamp
            FROM requests
            WHERE timestamp > ? AND duration_ms > ? AND error IS NULL
            ORDER BY timestamp DESC
        """, (cutoff, threshold)).fetchall()

        patterns = []
        for row in rows:
            patterns.append(Pattern(
                timestamp=row["timestamp"],
                pattern_type="latency_spike",
                severity="warning" if row["duration_ms"] < threshold * 2 else "critical",
                description=f"High latency ({row['duration_ms']:.0f}ms) on {row['endpoint']} "
                            f"via {row['service_name']}",
                related_request_id=row["id"],
                details={"service": row["service_name"], "endpoint": row["endpoint"],
                          "latency_ms": row["duration_ms"], "client_ip": row["client_ip"],
                          "threshold_ms": threshold},
            ))
        return patterns

    async def _detect_delete_attempts(self) -> List[Pattern]:
        """A DELETE hit the public endpoint. A reverse proxy in front may
        already block it (403), but the attempt is still worth surfacing —
        DELETE against a loaded model is destructive, and whether it was
        actually blocked is worth knowing, not just assuming."""
        cutoff = time.time() - 3600
        rows = execute("""
            SELECT id, service_name, client_ip, status_code, timestamp
            FROM requests
            WHERE timestamp > ? AND method = 'DELETE'
            ORDER BY timestamp DESC
        """, (cutoff,)).fetchall()

        patterns = []
        for row in rows:
            blocked = row["status_code"] == 403
            patterns.append(Pattern(
                timestamp=row["timestamp"],
                pattern_type="delete_attempt",
                severity="info" if blocked else "critical",
                description=f"DELETE from {row['client_ip']} on {row['service_name']} — "
                            f"{'blocked by proxy guard' if blocked else 'NOT BLOCKED, status ' + str(row['status_code'])}",
                related_request_id=row["id"],
                details={"service": row["service_name"], "client_ip": row["client_ip"],
                          "status_code": row["status_code"], "blocked": blocked},
            ))
        return patterns

    async def _detect_quota_exhaustion(self) -> List[Pattern]:
        """Known limitation: GIN access-log lines never carry the response
        body, so this can only match keywords against status/path/IP text
        (via the `error` field, now correctly populated only on real
        errors) — it will not see a "quota exceeded" message that only
        exists in a JSON error body. Kept for the 429/status-code cases,
        which it can see.

        `error LIKE '[GIN]%'` matters, not just cosmetics: rows written
        before the `error`-column fix (Turn: this rebuild) stuffed the raw
        journald JSON envelope into `error` unconditionally, and fields
        like `_CAP_EFFECTIVE` coincidentally contain digit runs that match
        naive substring keywords like "429" — this produced a real,
        demonstrated false positive in testing against this box's own
        legacy data. Anchoring to the GIN line shape filters that out."""
        keywords = settings.patterns.quota_exhaustion_keywords
        cutoff = time.time() - 3600

        placeholders = " OR ".join(["error LIKE ?"] * len(keywords))
        params = [f"%{kw}%" for kw in keywords] + [cutoff]

        rows = execute(f"""
            SELECT id, service_name, endpoint, error, timestamp, client_ip
            FROM requests
            WHERE error IS NOT NULL AND error LIKE '[GIN]%' AND ({placeholders}) AND timestamp > ?
            ORDER BY timestamp DESC
        """, tuple(params)).fetchall()

        patterns = []
        for row in rows:
            patterns.append(Pattern(
                timestamp=row["timestamp"],
                pattern_type="quota_exhaustion",
                severity="critical",
                description=f"Possible quota/rate-limit signal on {row['endpoint']}: {row['error'][:100]}",
                related_request_id=row["id"],
                details={"service": row["service_name"], "endpoint": row["endpoint"],
                          "error": row["error"], "client_ip": row["client_ip"]},
            ))
        return patterns

    # ------------------------------------------------------------------
    # GPU-based
    # ------------------------------------------------------------------

    async def _detect_gpu_starvation(self) -> List[Pattern]:
        """High VRAM, near-zero compute — a model is resident but idle.
        On this box's pipeline-parallel hardware this is often benign
        (waiting between requests), so keep the threshold generous.
        Reads the in-memory GPU sample ring (see storage.latest_gpu_
        samples()) -- this data is never persisted to disk."""
        cutoff = time.time() - 300

        patterns = []
        for sample in latest_gpu_samples():
            if sample["timestamp"] <= cutoff:
                continue  # collector stalled -- nothing fresh for this GPU
            mem_pct = (sample["memory_used_mb"] / sample["memory_total_mb"]) * 100 if sample["memory_total_mb"] else 0
            util = sample["gpu_utilization_percent"]
            if mem_pct > 90 and util < 10:
                patterns.append(Pattern(
                    timestamp=sample["timestamp"],
                    pattern_type="gpu_starvation",
                    severity="info",
                    description=f"GPU {sample['gpu_index']} ({sample['name']}): {mem_pct:.0f}% VRAM used, "
                                f"{util:.0f}% compute — likely idle-but-loaded, not necessarily unhealthy",
                    related_gpu_index=sample["gpu_index"],
                    details={"gpu_index": sample["gpu_index"], "gpu_name": sample["name"],
                              "memory_percent": mem_pct, "utilization_percent": util},
                ))
        return patterns

    async def _detect_gpu_hardware_fault(self) -> List[Pattern]:
        """Real hardware degradation: pending page retirement or
        uncorrected ECC errors. This is the actual "is the card dying"
        signal — utilization and memory never show it."""
        rows = execute("SELECT * FROM gpu_hardware_samples").fetchall()

        patterns = []
        for row in rows:
            if row["retired_pages_pending"]:
                patterns.append(Pattern(
                    timestamp=row["timestamp"],
                    pattern_type="gpu_hardware_fault",
                    severity="critical",
                    description=f"GPU {row['gpu_index']}: pending page retirement — real hardware "
                                f"degradation, not a load/driver issue",
                    related_gpu_index=row["gpu_index"],
                    details={"gpu_index": row["gpu_index"]},
                ))
            if row["ecc_uncorrected_volatile"] and row["ecc_uncorrected_volatile"] > 0:
                patterns.append(Pattern(
                    timestamp=row["timestamp"],
                    pattern_type="gpu_hardware_fault",
                    severity="critical",
                    description=f"GPU {row['gpu_index']}: {row['ecc_uncorrected_volatile']} uncorrected "
                                f"ECC errors",
                    related_gpu_index=row["gpu_index"],
                    details={"gpu_index": row["gpu_index"],
                              "ecc_uncorrected": row["ecc_uncorrected_volatile"]},
                ))
        return patterns

    async def _detect_cpu_spillover(self) -> List[Pattern]:
        """System-wide CPU spillover during active GPU compute — simplified
        to a live psutil check rather than per-request attribution (the
        original design assumed a request->GPU->CPU mapping that doesn't
        exist on pipeline-parallel hardware where every request touches
        every GPU). A real prior incident on this fleet (AHA-1083, wrong
        Ollama port routed inference to a CPU-serving fallback) is exactly
        the failure mode this exists to catch."""
        threshold = settings.patterns.cpu_spillover_threshold_percent
        cpu_pct = psutil.cpu_percent(interval=None)
        if cpu_pct == 0.0:
            return []  # first call after process start, psutil needs a baseline

        recent = [s for s in latest_gpu_samples() if s["timestamp"] > time.time() - 10]
        max_gpu_util = max((s["gpu_utilization_percent"] or 0 for s in recent), default=0)

        if cpu_pct > threshold and recent and max_gpu_util < 5:
            return [Pattern(
                timestamp=time.time(),
                pattern_type="cpu_spillover",
                severity="warning",
                description=f"System CPU at {cpu_pct:.0f}% while all GPUs are idle — inference may be "
                            f"running on CPU instead of GPU (check routing/port config)",
                details={"cpu_percent": cpu_pct, "threshold_percent": threshold},
            )]
        return []

    async def store_patterns(self, patterns: List[Pattern]):
        if not patterns:
            return

        db = get_db()
        cooldown_seconds = settings.patterns.baseline_window_minutes * 60
        for p in patterns:
            if p.related_request_id is not None:
                existing = db.execute(
                    "SELECT 1 FROM patterns WHERE pattern_type = ? AND related_request_id = ? LIMIT 1",
                    (p.pattern_type, p.related_request_id)
                ).fetchone()
            elif p.related_load_cycle_id is not None:
                # Load-cycle-scoped patterns (reload_storm, ctx_churn) embed
                # live counts in their description that drift every poll —
                # same reasoning as the gpu_index branch below: dedupe on
                # (type, description) within the cooldown window instead of
                # trying to match the description text exactly forever.
                existing = db.execute(
                    "SELECT 1 FROM patterns WHERE pattern_type = ? AND timestamp > ? "
                    "AND description = ? LIMIT 1",
                    (p.pattern_type, p.timestamp - cooldown_seconds, p.description)
                ).fetchone()
            elif p.related_gpu_index is not None:
                existing = db.execute(
                    "SELECT 1 FROM patterns WHERE pattern_type = ? AND related_gpu_index = ? AND timestamp > ? LIMIT 1",
                    (p.pattern_type, p.related_gpu_index, p.timestamp - cooldown_seconds)
                ).fetchone()
            else:
                existing = db.execute(
                    "SELECT 1 FROM patterns WHERE pattern_type = ? AND description = ? AND timestamp > ? LIMIT 1",
                    (p.pattern_type, p.description, p.timestamp - cooldown_seconds)
                ).fetchone()
            if existing:
                continue

            day_bucket_insert("patterns", {
                "timestamp": p.timestamp,
                "pattern_type": p.pattern_type,
                "severity": p.severity,
                "description": p.description,
                "related_request_id": p.related_request_id,
                "related_gpu_index": p.related_gpu_index,
                "related_load_cycle_id": p.related_load_cycle_id,
                "details_json": json.dumps(p.details) if p.details else None,
            }, timestamp=p.timestamp, retention_days=settings.storage.event_retention_days)


async def analyze_and_store():
    """Convenience function to run analysis and store results."""
    analyzer = PatternAnalyzer()
    patterns = await analyzer.analyze()
    await analyzer.store_patterns(patterns)
    return patterns
