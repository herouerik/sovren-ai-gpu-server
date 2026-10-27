"""Unit tests for src/retention.py (temp DB and temp dirs only; never touches real data)."""
import os
import sqlite3
import time

from src import retention

DAY = 86400


def _db():
    db = sqlite3.connect(":memory:")
    for t in retention.DB_TABLES:
        db.execute(f"CREATE TABLE {t} (id INTEGER PRIMARY KEY, timestamp REAL NOT NULL, note TEXT)")
    return db


def test_prune_db_deletes_only_old_rows():
    db, now = _db(), time.time()
    for t in retention.DB_TABLES:
        db.execute(f"INSERT INTO {t} (timestamp, note) VALUES (?, 'old')", (now - 20 * DAY,))
        db.execute(f"INSERT INTO {t} (timestamp, note) VALUES (?, 'new')", (now - 1 * DAY,))
    res = retention.prune_db(db, 14, now=now)
    assert res == {t: 1 for t in retention.DB_TABLES}
    for t in retention.DB_TABLES:
        assert [r[0] for r in db.execute(f"SELECT note FROM {t}")] == ["new"]


def test_prune_db_dry_run_changes_nothing():
    db, now = _db(), time.time()
    db.execute("INSERT INTO requests (timestamp, note) VALUES (?, 'old')", (now - 30 * DAY,))
    assert retention.prune_db(db, 14, now=now, dry_run=True)["requests"] == 1
    assert db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 1


def _touch(path, age_days):
    path.write_text("x")
    t = time.time() - age_days * DAY
    os.utime(path, (t, t))


def test_prune_files_scope(tmp_path):
    logs, other = tmp_path / "logs", tmp_path / "other"
    logs.mkdir(); other.mkdir()
    _touch(logs / "access.log.3.gz", 30)      # old + matches -> deleted
    _touch(logs / "access.log.4.gz", 2)       # recent -> kept
    _touch(logs / "access.log", 30)           # old but active log, no pattern match -> kept
    _touch(other / "keep.gz", 30)             # other dir -> kept
    (logs / "sub").mkdir(); _touch(logs / "sub" / "deep.gz", 30)   # no recursion -> kept
    link = logs / "link.gz"; link.symlink_to(other / "keep.gz")     # symlink -> never followed
    gone = retention.prune_files([str(logs)], ["*.log.*", "*.gz"], 14)
    assert gone == [str(logs / "access.log.3.gz")]
    assert (logs / "access.log").exists() and (other / "keep.gz").exists() and (logs / "sub" / "deep.gz").exists() and link.is_symlink()


def test_disabled_when_zero():
    class C:  # minimal settings stand-in
        class retention:
            max_age_days = 0
    assert retention.run_once(C) == {"enabled": False}
