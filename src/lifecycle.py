"""Correlates raw lifecycle events (starting/loaded/failed + /api/ps polling)
into `load_cycles` rows -- the real unit of health on pipeline-parallel
hardware where a single request never maps to a single GPU.

This is the same method used by hand, dozens of times, to diagnose every
real incident on this box across a multi-day investigation: watch for
"starting llama-server" (with its -c value), see whether "model loaded" or
"Load failed" follows, and separately watch /api/ps for when a successfully
loaded model disappears again (eviction) and what `expires_at` it carried.
Coding it here means the dashboard can show it live instead of requiring a
manual journalctl session every time something looks wrong.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

from src.storage import get_db


class LifecycleTracker:
    def __init__(self):
        # service_name -> currently-open load_cycles row id (outcome='pending' or 'success')
        self._open_cycle: Dict[str, int] = {}
        # service_name -> model name currently believed resident, per last /api/ps poll
        self._resident_model: Dict[str, Optional[str]] = {}

    async def _snapshot_peers(self, service_name: str, public_port: int) -> Optional[str]:
        """Best-effort: who's connected to the public port right now.

        Taken synchronously at the moment a load starts, same as the manual
        `ss` snapshots used throughout this box's incident investigations --
        a periodic background sample would too often miss the narrow window
        a reload actually fires in.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "ss", "-tn", "state", "established",
                f"( sport = :{public_port} or dport = :{public_port} )",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3)
            lines = [l for l in stdout.decode(errors="ignore").strip().split("\n")[1:] if l.strip()]
            import re
            ips = set()
            for line in lines:
                m = re.search(r"(\d+\.\d+\.\d+\.\d+):\d+\s*$", line.split()[-1] if line.split() else "")
                if m:
                    ips.add(m.group(1))
            return ",".join(sorted(ips)) if ips else None
        except Exception:
            return None

    async def handle_event(self, event: Dict[str, Any], public_port: Optional[int] = None):
        db = get_db()
        service_name = event["service_name"]

        if event["kind"] == "starting":
            # A new load starting while a previous one is still "pending"
            # means that one never resolved before being replaced -- mark
            # it superseded rather than leaving it dangling forever.
            prev_id = self._open_cycle.get(service_name)
            if prev_id is not None:
                row = db.execute("SELECT outcome FROM load_cycles WHERE id = ?", (prev_id,)).fetchone()
                if row and row["outcome"] == "pending":
                    db.execute(
                        "UPDATE load_cycles SET outcome = 'superseded', duration_s = ? WHERE id = ?",
                        (event["timestamp"] - self._start_ts(db, prev_id), prev_id),
                    )

            peers = await self._snapshot_peers(service_name, public_port) if public_port else None
            cur = db.execute("""
                INSERT INTO load_cycles (service_name, start_ts, ctx_requested, model, outcome, triggering_client_ip)
                VALUES (?, ?, ?, ?, 'pending', ?)
            """, (service_name, event["timestamp"], event["ctx_requested"], event["model"], peers))
            db.commit()
            self._open_cycle[service_name] = cur.lastrowid

        elif event["kind"] == "loaded":
            cid = self._open_cycle.get(service_name)
            if cid is None:
                return
            start_ts = self._start_ts(db, cid)
            db.execute("""
                UPDATE load_cycles
                SET outcome = 'success', load_completed_ts = ?, duration_s = ?
                WHERE id = ? AND outcome = 'pending'
            """, (event["timestamp"], event["timestamp"] - start_ts, cid))
            db.commit()

        elif event["kind"] == "failed":
            cid = self._open_cycle.get(service_name)
            if cid is None:
                return
            start_ts = self._start_ts(db, cid)
            db.execute("""
                UPDATE load_cycles
                SET outcome = 'failed', failed_ts = ?, duration_s = ?
                WHERE id = ? AND outcome = 'pending'
            """, (event["timestamp"], event["timestamp"] - start_ts, cid))
            db.commit()
            self._open_cycle.pop(service_name, None)

    @staticmethod
    def _start_ts(db, cycle_id: int) -> float:
        row = db.execute("SELECT start_ts FROM load_cycles WHERE id = ?", (cycle_id,)).fetchone()
        return row["start_ts"] if row else time.time()

    async def handle_ollama_state(self, service_name: str, models: list):
        """Call once per /api/ps poll. Detects eviction (resident model
        disappeared) and captures `expires_at` for the currently-loaded
        cycle the first time it's observed.
        """
        db = get_db()
        current_model = models[0]["name"] if models else None
        previous_model = self._resident_model.get(service_name)

        cid = self._open_cycle.get(service_name)
        if cid is not None and current_model:
            row = db.execute("SELECT outcome, keep_alive_expires_at FROM load_cycles WHERE id = ?", (cid,)).fetchone()
            if row and row["outcome"] == "success" and row["keep_alive_expires_at"] is None:
                expires_at_str = models[0].get("expires_at")
                if expires_at_str:
                    from datetime import datetime
                    try:
                        expires_ts = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00")).timestamp()
                        db.execute(
                            "UPDATE load_cycles SET keep_alive_expires_at = ? WHERE id = ?",
                            (expires_ts, cid),
                        )
                        db.commit()
                    except ValueError:
                        pass

        if previous_model and not current_model:
            # Resident model vanished -- eviction. Close out the most recent
            # successful, not-yet-evicted cycle for this service.
            row = db.execute("""
                SELECT id, load_completed_ts FROM load_cycles
                WHERE service_name = ? AND outcome = 'success' AND evicted_ts IS NULL
                ORDER BY start_ts DESC LIMIT 1
            """, (service_name,)).fetchone()
            if row:
                now = time.time()
                resident_s = now - row["load_completed_ts"] if row["load_completed_ts"] else None
                db.execute("""
                    UPDATE load_cycles SET evicted_ts = ?, resident_duration_s = ? WHERE id = ?
                """, (now, resident_s, row["id"]))
                db.commit()
            self._open_cycle.pop(service_name, None)

        self._resident_model[service_name] = current_model


_tracker: Optional[LifecycleTracker] = None


def get_tracker() -> LifecycleTracker:
    global _tracker
    if _tracker is None:
        _tracker = LifecycleTracker()
    return _tracker
