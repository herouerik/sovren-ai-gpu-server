"""Detects a genuinely wedged Ollama service and, if enabled, restarts it.

Built from a real incident (2026-09-25): task 2848714 was mid-generation,
healthy, decoding at 32.6 tok/s -- then GPU 5 threw a PCIe Uncorrectable
(Non-Fatal) error mid-CUDA-op, and llama-server went silent instantly. No
crash, no error surfaced anywhere, no further progress ever -- just ~100%
CPU spin and every subsequent request admitted-then-aborted, for over 6
hours until a human noticed and restarted the service by hand.

Distinguishing a genuine wedge from this fleet's OTHER real incident
(2026-09-22/23: oversized accumulated context + a client timeout shorter
than this hardware's real prefill throughput -- looked identical from the
outside, but was real work, just slow) is the entire point of this module,
not incidental. That earlier incident logged real progress (a `slot
print_timing` line) every ~10-15s for the whole multi-minute duration and
eventually completed successfully. The 2026-09-25 wedge logged ZERO
progress lines, for hours, despite real traffic continuing to arrive and
get admitted (task IDs kept incrementing). `service_activity.
last_progress_ts` -- updated by LogTailer.is_progress_line(), see
collectors.py -- is what makes telling these apart mechanical instead of
requiring a human to read logs by hand every time, the way both real
incidents this fleet has had were actually diagnosed.

Off by default (`watchdog.enabled: false`) -- this is the one part of this
app that takes a real production action rather than only observing. See
README "Watchdog" for the sudoers grant it needs and the safety knobs
(cooldown, daily cap) that keep it from restart-looping if a restart isn't
actually the fix for whatever's wrong.
"""
from __future__ import annotations

import asyncio
import time
from typing import List, Optional, Tuple

from src.config import settings
from src.storage import execute, last_progress_ts, day_bucket_insert
from src.patterns import Pattern, PatternAnalyzer

_INFERENCE_ENDPOINTS = ("/api/generate", "/api/chat", "/v1/chat/completions")


async def _restart_service(systemd_service: str) -> Tuple[bool, str]:
    """`sudo -n` (non-interactive) fails fast with a clear stderr message
    if the sudoers NOPASSWD grant isn't set up, instead of hanging forever
    on a password prompt nothing can answer."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "sudo", "-n", "systemctl", "restart", systemd_service,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode == 0:
            return True, "systemctl restart succeeded"
        return False, f"systemctl restart failed (exit {proc.returncode}): {stderr.decode(errors='ignore')[:300]}"
    except Exception as e:
        return False, f"systemctl restart errored: {e}"


def _restarts_today(service_name: str) -> int:
    cutoff = time.time() - 86400
    row = execute(
        "SELECT COUNT(*) as c FROM watchdog_actions "
        "WHERE service_name = ? AND action = 'restart' AND outcome = 'success' AND timestamp > ?",
        (service_name, cutoff),
    ).fetchone()
    return row["c"] if row else 0


def _last_restart_attempt_ts(service_name: str) -> Optional[float]:
    row = execute(
        "SELECT MAX(timestamp) as t FROM watchdog_actions WHERE service_name = ? AND action = 'restart'",
        (service_name,),
    ).fetchone()
    return row["t"] if row and row["t"] is not None else None


async def check_and_remediate_wedged_services() -> List[Pattern]:
    cfg = settings.watchdog
    if not cfg.enabled:
        return []

    now = time.time()
    window_cutoff = now - cfg.wedge_window_seconds
    placeholders = ",".join("?" * len(_INFERENCE_ENDPOINTS))
    patterns: List[Pattern] = []

    for svc in settings.get_ollama_services():
        if not svc.enabled or not svc.systemd_service:
            continue

        row = execute(f"""
            SELECT COUNT(*) as total
            FROM requests
            WHERE service_name = ? AND method = 'POST' AND timestamp > ?
              AND endpoint IN ({placeholders})
        """, (svc.name, window_cutoff, *_INFERENCE_ENDPOINTS)).fetchone()
        total = row["total"] or 0
        if total < cfg.min_real_requests_in_window:
            continue  # not enough attempted traffic in the window to judge either way

        # Deliberately NOT "was there a 200 in the requests table" -- this
        # box's own keepwarm_gpu_ollama.py cron sends an empty-prompt
        # keep_alive touch every 5 minutes that always returns 200 in
        # ~230ms (confirmed live: zero llama-server progress lines at the
        # same timestamp) without ever exercising real inference. Counting
        # that as "healthy" would mask a genuine wedge forever on this box,
        # since keepwarm guarantees a 200 every 5 minutes regardless of
        # whether real generation works at all -- caught by replaying this
        # exact query against the 2026-09-25 incident's own historical
        # data before shipping. last_progress_ts is immune to this: it's
        # only ever set by a real `print_timing`/`launch_slot_` line, which
        # a bodyless keep_alive touch never produces.
        progress = last_progress_ts(svc.name)
        if progress is not None and progress > window_cutoff:
            continue  # real progress within the window -- healthy, or legitimately slow (Sep 22/23 case)

        # Wedge condition met: real traffic arrived, but llama-server
        # logged zero real progress for the entire window.
        last_attempt = _last_restart_attempt_ts(svc.name)
        restarts_today = _restarts_today(svc.name)

        if last_attempt is not None and now - last_attempt < cfg.restart_cooldown_seconds:
            outcome = "suppressed_cooldown"
            remaining = cfg.restart_cooldown_seconds - (now - last_attempt)
            detail = f"restart already attempted {now - last_attempt:.0f}s ago, {remaining:.0f}s left on cooldown"
        elif restarts_today >= cfg.max_restarts_per_day:
            outcome = "suppressed_daily_cap"
            detail = f"{restarts_today} restarts already succeeded today (cap {cfg.max_restarts_per_day}) -- restarts aren't fixing this, needs a human"
        else:
            ok_restart, detail = await _restart_service(svc.systemd_service)
            outcome = "success" if ok_restart else "failed"

        day_bucket_insert("watchdog_actions", {
            "timestamp": now, "service_name": svc.name, "action": "restart",
            "outcome": outcome, "detail": detail,
        }, timestamp=now, retention_days=settings.storage.event_retention_days)

        # Description deliberately stays stable per outcome bucket (no live
        # countdowns/counts in the text itself) so store_patterns()'s exact-
        # string dedup actually suppresses repeat alerts about the same
        # ongoing state -- full detail still lands in `details` and in the
        # permanent watchdog_actions audit row above either way.
        summaries = {
            "success": f"{svc.name}: wedged -- auto-restarted {svc.systemd_service}",
            "failed": f"{svc.name}: wedged -- auto-restart attempt FAILED, needs a human",
            "suppressed_cooldown": f"{svc.name}: wedged -- restart suppressed (cooldown active)",
            "suppressed_daily_cap": f"{svc.name}: wedged -- restart suppressed (daily cap reached), needs a human",
        }
        patterns.append(Pattern(
            timestamp=now, pattern_type="service_wedged", severity="critical",
            description=summaries[outcome],
            details={"service": svc.name, "outcome": outcome, "requests_in_window": total, "detail": detail},
        ))

    if patterns:
        await PatternAnalyzer().store_patterns(patterns)
    return patterns
