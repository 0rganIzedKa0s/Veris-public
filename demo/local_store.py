"""Local SQLite store for session captures — the Orin's NVMe is the backend.

Replaces the Supabase mirror. Every describe/walk/rating record the hub
appends to demo/sessions/*.jsonl is also inserted here, and every frame JPEG
the hub writes gets an index row, so one SQL query answers "every frame of
mode X", "every thumbs-down", "ratings joined to their capture" without
parsing JSONL. The JSONL + JPEGs on disk stay the source of truth: the DB is
rebuildable from them at any time with `python3 -m demo.local_store backfill`.

Stdlib only (sqlite3 + threading), same fire-and-forget contract as before:
a write failure never breaks or slows a capture.

Layout (default demo/veris.sqlite, override with VERIS_DB=/path/to.sqlite):
  captures  one row per describe/walk record (mode + source + kind columns
            instead of one table per mode; `raw` holds the full record)
  ratings   thumbs + notes, joined back to captures on (mode, frame)
  frames    one row per frame JPEG the hub wrote (path relative to sessions/)
"""
import json
import os
import pathlib
import sqlite3
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parent
SESSIONS = ROOT / "sessions"

_MODES = ("describe", "outdoor_nav", "indoor_nav", "text_reading", "general")
_CAP_COLS = ("ts", "context", "question", "variants", "output",
             "ttft_s", "total_s", "aborted")

_SCHEMA = """
create table if not exists captures (
  id        integer primary key,
  ts        real,
  kind      text not null default 'describe',   -- 'describe' | 'walk'
  mode      text,
  source    text,                               -- 'veris' | 'generic'
  frame     text,                               -- frame-*.jpg in sessions/
  context   text,
  question  text,
  variants  text,
  output    text,
  ttft_s    real,
  total_s   real,
  aborted   integer,
  raw       text                                -- full record as JSON
);
create unique index if not exists captures_ts_kind_uq on captures (ts, kind);
create index if not exists captures_mode_idx  on captures (mode, kind);
create index if not exists captures_frame_idx on captures (frame);

create table if not exists ratings (
  id      integer primary key,
  ts      real,
  mode    text,
  source  text,
  frame   text,
  output  text,
  rating  text,                                 -- 'up' | 'down' | null
  note    text,
  raw     text
);
create unique index if not exists ratings_ts_uq on ratings (ts);
create index if not exists ratings_frame_idx on ratings (frame);

create table if not exists frames (
  name    text primary key,                     -- frame-*.jpg
  mode    text,
  ts      real,
  bytes   integer
);
create index if not exists frames_mode_idx on frames (mode);
"""

_lock = threading.Lock()
_conn = None


def db_path() -> pathlib.Path:
    return pathlib.Path(os.environ.get("VERIS_DB") or (ROOT / "veris.sqlite"))


def enabled() -> bool:
    return os.environ.get("VERIS_DB", "").strip() != "off"


def _connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        p = db_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(p, check_same_thread=False, timeout=5)
        c.execute("pragma journal_mode=wal")
        c.execute("pragma synchronous=normal")
        c.executescript(_SCHEMA)
        _conn = c
    return _conn


def _mode_of(rec: dict) -> str:
    m = rec.get("mode") or "describe"
    return m if m in _MODES else "describe"


def _source_of(mode) -> str | None:
    if not mode:
        return None
    return "generic" if mode == "general" else "veris"


def _bg(fn) -> None:
    threading.Thread(target=fn, daemon=True).start()


def _insert_record(c: sqlite3.Connection, rec: dict) -> None:
    if rec.get("kind") == "rating":
        mode = rec.get("mode")
        c.execute(
            "insert or ignore into ratings (ts, mode, source, frame, output, rating, note, raw)"
            " values (?,?,?,?,?,?,?,?)",
            (rec.get("ts"), mode, _source_of(mode), rec.get("frame"),
             rec.get("output"), rec.get("rating"), rec.get("note"),
             json.dumps(rec)))
    else:
        mode = _mode_of(rec)
        vals = [rec.get(k) for k in _CAP_COLS]
        if vals[-1] is not None:
            vals[-1] = int(bool(vals[-1]))  # aborted -> 0/1
        c.execute(
            "insert or ignore into captures (ts, context, question, variants, output,"
            " ttft_s, total_s, aborted, kind, mode, source, frame, raw)"
            " values (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (*vals, rec.get("kind") or "describe", mode, _source_of(mode),
             rec.get("frame"), json.dumps(rec)))


def push_record(rec: dict) -> None:
    """Store one session record (describe / walk / rating). Never raises."""
    if not enabled():
        return

    def _do():
        try:
            with _lock:
                c = _connect()
                _insert_record(c, rec)
                c.commit()
        except Exception:
            pass  # JSONL on disk is the source of truth

    _bg(_do)


def push_frame(mode: str, name: str, data: bytes) -> None:
    """Index one frame JPEG the hub just wrote to sessions/<name>."""
    if not enabled():
        return
    mode = mode if mode in _MODES else "describe"

    def _do():
        try:
            ts = int(name[len("frame-"):-len(".jpg")]) / 1000 if name.startswith("frame-") else None
            with _lock:
                c = _connect()
                c.execute("insert or replace into frames (name, mode, ts, bytes) values (?,?,?,?)",
                          (name, mode, ts, len(data)))
                c.commit()
        except Exception:
            pass

    _bg(_do)


# ---------------------------------------------------------------- CLI ------

def backfill(sessions: pathlib.Path = SESSIONS) -> dict:
    """Replay every sessions/*.jsonl + frame-*.jpg into the DB. Idempotent."""
    counts = {"records": 0, "frames": 0, "bad_lines": 0}
    with _lock:
        c = _connect()
        for f in sorted(sessions.glob("session-*.jsonl")):
            for line in f.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except Exception:
                    counts["bad_lines"] += 1
                    continue
                _insert_record(c, rec)
                counts["records"] += 1
        # frames: mode comes from the record that references it, else 'describe'
        mode_by_frame = dict(c.execute(
            "select frame, mode from captures where frame is not null"))
        for p in sorted(sessions.glob("frame-*.jpg")):
            ts = int(p.name[6:-4]) / 1000
            c.execute("insert or ignore into frames (name, mode, ts, bytes) values (?,?,?,?)",
                      (p.name, mode_by_frame.get(p.name, "describe"), ts, p.stat().st_size))
            counts["frames"] += 1
        c.commit()
    return counts


def stats() -> dict:
    with _lock:
        c = _connect()
        out = {"db": str(db_path()),
               "captures": dict(c.execute("select mode, count(*) from captures group by mode")),
               "walks": c.execute("select count(*) from captures where kind='walk'").fetchone()[0],
               "ratings": dict(c.execute("select coalesce(rating,'note'), count(*) from ratings group by 1")),
               "frames": c.execute("select count(*) from frames").fetchone()[0]}
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if cmd == "backfill":
        print("backfill:", backfill())
    print(json.dumps(stats(), indent=2))
