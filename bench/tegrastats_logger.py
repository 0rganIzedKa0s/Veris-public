"""Runs ON THE JETSON: parse `tegrastats` into timestamped JSONL.

  python3 bench/tegrastats_logger.py --out bench/tegra.jsonl

Captures per-sample: total module power (VDD_IN mW), CPU/GPU rail power where
exposed, GPU load %, RAM used/total MB, and the hottest temperature. Rows are
merged with bench results by timestamp in analyze.py.

Stdlib only — no pip installs needed on the board.
"""

import argparse
import json
import re
import subprocess
import time

RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")
GPU_RE = re.compile(r"GR3D_FREQ (\d+)%")
TEMP_RE = re.compile(r"(\w+)@([\d.]+)C")
# Orin power rails, e.g. "VDD_IN 4712mW/4712mW" (instant/avg)
POWER_RE = re.compile(r"(VDD_\w+|VIN_\w+) (\d+)mW/(\d+)mW")


def parse_line(line: str) -> dict:
    row = {"ts": time.time()}
    if m := RAM_RE.search(line):
        row["ram_used_mb"], row["ram_total_mb"] = int(m.group(1)), int(m.group(2))
    if m := GPU_RE.search(line):
        row["gpu_pct"] = int(m.group(1))
    temps = {name: float(v) for name, v in TEMP_RE.findall(line)}
    if temps:
        row["temp_max_c"] = max(temps.values())
        row["temps"] = temps
    for rail, inst, _avg in POWER_RE.findall(line):
        row[f"power_{rail.lower()}_mw"] = int(inst)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tegra.jsonl")
    p.add_argument("--interval-ms", type=int, default=500)
    args = p.parse_args()

    proc = subprocess.Popen(
        ["tegrastats", "--interval", str(args.interval_ms)],
        stdout=subprocess.PIPE, text=True,
    )
    with open(args.out, "a") as f:
        try:
            for line in proc.stdout:
                f.write(json.dumps(parse_line(line)) + "\n")
                f.flush()
        except KeyboardInterrupt:
            pass
        finally:
            proc.terminate()


if __name__ == "__main__":
    main()
