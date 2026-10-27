"""Age-based retention and cleansing of everything that can reveal what was asked.

The tables in storage.py are bounded by COUNT (ring buffers), which on a quiet
machine can keep a prompt summary for months. This module adds an AGE limit:

* database rows older than `retention.max_age_days` are deleted from the tables
  that derive from requests (`prompt_summaries`, `requests`, `task_samples`);
  `secure_delete` zeroes the freed pages and the WAL is truncated, so the old
  text does not linger in the file;
* files older than that in the directories listed under `retention.log_dirs`
  are deleted (for example rotated nginx/Ollama logs). Only regular files that
  match the configured patterns directly inside a configured directory are
  touched: no recursion, no symlinks, nothing outside the listed directories.
  An active, still-growing log is never deleted by age; rotate it
  (deploy/newsyslog-sovren.conf.example on macOS) so old data becomes a rotated file.

Run it three ways: inside the monitor (retention.enabled, every
`interval_hours`), from cron/launchd (`python -m src.retention`), or by hand
with `--dry-run` to see what would go. Disabled when `max_age_days` is 0.
"""
from __future__ import annotations

import argparse
import fnmatch
import sqlite3
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DB_TABLES = ("prompt_summaries", "requests", "task_samples")


def prune_db(db: sqlite3.Connection, max_age_days: float, now: Optional[float] = None, dry_run: bool = False) -> Dict[str, int]:
    """Delete rows older than the cutoff from the request-derived tables. Returns rows affected per table."""
    cutoff = (now if now is not None else time.time()) - max_age_days * 86400
    out: Dict[str, int] = {}
    existing = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not dry_run:
        db.execute("PRAGMA secure_delete=ON")
    for table in DB_TABLES:
        if table not in existing:
            continue
        n = db.execute(f"SELECT COUNT(*) FROM {table} WHERE timestamp < ?", (cutoff,)).fetchone()[0]
        out[table] = n
        if n and not dry_run:
            db.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff,))
    if not dry_run:
        db.commit()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return out


def prune_files(dirs: Iterable[str], patterns: Iterable[str], max_age_days: float, now: Optional[float] = None,
                dry_run: bool = False) -> List[str]:
    """Delete regular files older than the cutoff that match `patterns` directly inside each listed directory."""
    cutoff = (now if now is not None else time.time()) - max_age_days * 86400
    pats = list(patterns) or ["*"]
    gone: List[str] = []
    for d in dirs:
        base = Path(d).expanduser()
        if not base.is_dir() or base.is_symlink():
            continue
        for p in base.iterdir():
            if p.is_symlink() or not p.is_file():
                continue
            if not any(fnmatch.fnmatch(p.name, pat) for pat in pats):
                continue
            if p.stat().st_mtime < cutoff:
                gone.append(str(p))
                if not dry_run:
                    p.unlink(missing_ok=True)
    return gone


def run_once(settings, dry_run: bool = False, now: Optional[float] = None) -> Dict[str, object]:
    cfg = settings.retention
    if cfg.max_age_days <= 0:
        return {"enabled": False}
    from src import storage
    rows = prune_db(storage.get_db(), cfg.max_age_days, now=now, dry_run=dry_run)
    files = prune_files(cfg.log_dirs, cfg.log_patterns, cfg.max_age_days, now=now, dry_run=dry_run)
    return {"enabled": True, "dry_run": dry_run, "rows": rows, "files": files}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Apply the configured retention (age limit) now.")
    ap.add_argument("--dry-run", action="store_true", help="show what would be deleted, delete nothing")
    args = ap.parse_args(argv)
    from src.config import settings
    res = run_once(settings, dry_run=args.dry_run)
    if not res["enabled"]:
        print("retention disabled (retention.max_age_days is 0)")
        return 0
    verb = "would delete" if args.dry_run else "deleted"
    print(f"{verb} rows: {res['rows']}")
    print(f"{verb} {len(res['files'])} file(s)")
    for f in res["files"]:
        print("  ", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
