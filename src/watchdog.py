"""Detects a genuinely wedged Ollama service and, if enabled, restarts it.

Built from a real incident (2026-09-25): a request was mid-generation,
healthy, actively decoding -- then GPU 5 threw a PCIe Uncorrectable
(Non-Fatal) error mid-CUDA-op, and llama-server went silent instantly. No
crash, no error surfaced anywhere, no further progress ever -- just ~100%
CPU spin and every subsequent request admitted-then-aborted, for over 6
hours until a human happened to notice and restarted the service by hand.

Distinguishing a genuine wedge from this fleet's OTHER real incident
(2026-09-22/23: oversized accumulated context + a client timeout shorter
than this hardware's real prefill throughput -- looked identical from the
outside, but was real work, just slow) is the entire point of this module,
not incidental. That earlier incident logged real progress (a `slot
print_timing` line) every ~10-15s for its whole multi-minute duration and
eventually completed; a genuine wedge logs nothing, ever, no matter how
long real traffic keeps arriving. `service_activity.last_progress_ts` --
updated by LogTailer.is_progress_line(), see collectors.py -- is what
makes this distinction mechanical.

TWO real bugs were caught and fixed in this module before it could do
real damage, both by replaying its own logic against real historical data
rather than trusting it on paper -- worth understanding why, since the
same mistake is easy to reintroduce:

1. Originally checked `requests` (fed by Ollama's own GIN access-log
   lines, written only on request COMPLETION) for "did anything succeed".
   That table structurally CANNOT see a request that never completes --
   which is exactly what a genuine wedge is. During the real 2026-09-25
   incident, `requests` has ZERO rows for the entire 6-hour wedge, because
   nothing ever finished long enough to get a GIN line. A signal that
   goes silent exactly when there's traffic to judge is useless for that
   traffic -- it can only ever fire by accident (e.g. an unrelated cron's
   own successful pings happening to satisfy an "any request" count).

2. That accident is exactly what happened next: this box's own
   keepwarm_gpu_ollama.py cron sends an empty-prompt keep_alive touch to
   Ollama's internal bind every 5 minutes, bypassing nginx entirely,
   which always returns 200 in ~230ms and produces zero llama-server
   progress lines (confirmed live). Counting it as "traffic present"
   made three separate, genuinely healthy, merely-idle periods look
   wedged and triggered three unnecessary restarts in under 15 hours --
   each one a real, if brief, outage of its own (the cold-load gap that
   follows any restart), and each one silently spending a slot of
   max_restarts_per_day for nothing.

The fix for both: use `prompt_summaries` instead of `requests` as the
"was real traffic attempted" signal. It's written synchronously the
moment nginx's mirror delivers a copy of a request -- at ARRIVAL time,
not completion -- so it sees traffic a genuine wedge swallows entirely.
And since keepwarm's calls go directly to Ollama's internal bind and
never pass through nginx, they never reach the mirror and never appear
here at all -- no IP heuristic needed, the data source itself already
excludes them by construction.

This means the watchdog has a real, hard dependency: `prompt_insight.
enabled: true` and a working nginx mirror (see README "Prompt insight").
Without it, `prompt_summaries` never has real data for any service, and
this module has no way to safely tell "wedged" apart from "quiet, no one
asked" -- so it stays silent rather than guessing from a signal (like
`requests`) proven unable to answer the question.

Off by default (`watchdog.enabled: false`) -- this is the one part of
this app that takes a real production action rather than only observing.
See README "Watchdog" for the sudoers grant it needs and the safety
knobs (cooldown, daily cap) that keep it from restart-looping if a
restart isn't actually the fix for whatever's wrong.
"""
from __future__ import annotations

import asyncio
import time
from typing import List, Optional, Tuple

from src.config import settings
from src.storage import execute, last_progress_ts, day_bucket_insert
from src.patterns import Pattern, PatternAnalyzer

_watchdog_warned_prompt_insight_off = False


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

    global _watchdog_warned_prompt_insight_off
    if not settings.prompt_insight.enabled:
        if not _watchdog_warned_prompt_insight_off:
            print("Watchdog: prompt_insight.enabled is false, so prompt_summaries never has real "
                  "traffic data for any service -- watchdog has no safe signal and will stay silent "
                  "until it's enabled (see README Watchdog / Prompt insight).")
            _watchdog_warned_prompt_insight_off = True
        return []

    now = time.time()
    window_cutoff = now - cfg.wedge_window_seconds
    patterns: List[Pattern] = []

    for svc in settings.get_ollama_services():
        if not svc.enabled or not svc.systemd_service:
            continue

        # prompt_summaries, not `requests`: written at request ARRIVAL
        # (nginx's mirror), not completion (Ollama's own GIN log) -- see
        # module docstring for why that distinction is the whole fix.
        row = execute(
            "SELECT COUNT(*) as total FROM prompt_summaries WHERE service_name = ? AND timestamp > ?",
            (svc.name, window_cutoff),
        ).fetchone()
        total = row["total"] or 0
        if total < cfg.min_real_requests_in_window:
            continue  # not enough real traffic in the window to judge either way

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
