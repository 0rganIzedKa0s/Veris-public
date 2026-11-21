"""Pull demo session logs + frames off the Orin into demo/sessions/.

  uv run python dataset/sync_sessions.py           # do the pull
  uv run python dataset/sync_sessions.py --dry-run  # show what would transfer

The demo hub on the Orin writes one JSONL per session (generation records +
kind:"rating" feedback records) plus the frame-*.jpg it captured, all under
~/veris-app/demo/sessions/. This is the first stage of the session-data
flywheel: bring that raw feedback home so build_sft_set.py can mine it.

Idempotent and non-destructive: uses rsync --ignore-existing, so a frame or
log already present locally is never re-fetched or overwritten. Safe to run on
a cron. Key-based ssh to user@192.168.55.1 is assumed to work.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL_SESSIONS = HERE.parent / "demo" / "sessions"

ORIN_SSH = os.environ.get("ORIN_SSH", "user@192.168.55.1")
# trailing slash on the source => copy the directory *contents*, not the dir
REMOTE_SESSIONS = os.environ.get("ORIN_SESSIONS", "~/veris-app/demo/sessions/")


def build_cmd(dry_run: bool) -> list[str]:
    # --ignore-existing: never touch a file we already have (no overwrites).
    # -t preserves mtimes so the guard is stable across runs.
    flags = ["-rt", "--ignore-existing"]
    if dry_run:
        flags.append("--dry-run")
    flags += ["--out-format=%n"]  # print each transferred path, nothing else
    return [
        "rsync", *flags,
        "-e", "ssh -o ConnectTimeout=10 -o BatchMode=yes",
        f"{ORIN_SSH}:{REMOTE_SESSIONS}",
        f"{LOCAL_SESSIONS}/",
    ]


def parse_transfers(stdout: str) -> tuple[list[str], list[str]]:
    """Split rsync's per-file output into (logs, frames), ignoring dir lines."""
    logs, frames = [], []
    for line in stdout.splitlines():
        name = line.strip()
        if not name or name.endswith("/"):
            continue
        if name.endswith(".jsonl"):
            logs.append(name)
        elif name.endswith(".jpg"):
            frames.append(name)
    return logs, frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would transfer without copying anything")
    args = ap.parse_args()

    LOCAL_SESSIONS.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(args.dry_run)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        print("error: rsync not found on PATH", file=sys.stderr)
        return 2
    except subprocess.TimeoutExpired:
        print("error: sync timed out talking to the Orin", file=sys.stderr)
        return 2

    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        print(f"error: rsync exited {proc.returncode} "
              f"(is {ORIN_SSH} reachable, key-based ssh working?)", file=sys.stderr)
        return proc.returncode

    logs, frames = parse_transfers(proc.stdout)
    verb = "would pull" if args.dry_run else "pulled"
    print(f"sync_sessions: {verb} {len(logs)} new log(s) + {len(frames)} new frame(s) "
          f"from {ORIN_SSH} into {LOCAL_SESSIONS}")
    for name in logs:
        print(f"  log   {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
