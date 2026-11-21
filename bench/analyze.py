"""Summarize bench results, merging tegrastats power/thermal samples.

  uv run python bench/analyze.py --results bench/results.jsonl \
      --tegra bench/tegra.jsonl

Prints a per-(tag, mode) table: TTFT, total, tok/s, and — when tegrastats rows
overlap a generation window — mean power, peak RAM, max temp during
generation. Also emits bench/summary.json for downstream use.
"""

import argparse
import json
import statistics as st
from pathlib import Path


def load_jsonl(path):
    if not path or not Path(path).exists():
        return []
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def window_stats(tegra, t0, t1):
    rows = [r for r in tegra if t0 <= r.get("ts", 0) <= t1]
    if not rows:
        return {}
    out = {}
    power = [r["power_vdd_in_mw"] for r in rows if "power_vdd_in_mw" in r]
    if power:
        out["power_mean_w"] = round(st.mean(power) / 1000, 2)
        out["power_max_w"] = round(max(power) / 1000, 2)
    ram = [r["ram_used_mb"] for r in rows if "ram_used_mb" in r]
    if ram:
        out["ram_max_mb"] = max(ram)
    temps = [r["temp_max_c"] for r in rows if "temp_max_c" in r]
    if temps:
        out["temp_max_c"] = max(temps)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="bench/results.jsonl")
    p.add_argument("--tegra", default=None)
    args = p.parse_args()

    results = load_jsonl(args.results)
    tegra = load_jsonl(args.tegra)
    if not results:
        raise SystemExit("no results found")

    groups: dict[tuple, list] = {}
    for r in results:
        groups.setdefault((r["tag"], r["mode"]), []).append(r)

    summary = []
    hdr = f"{'tag':<22}{'mode':<14}{'n':>3}{'ttft_s':>8}{'total_s':>9}{'tok/s':>8}{'W avg':>7}{'MB max':>8}{'C max':>7}"
    print(hdr)
    print("-" * len(hdr))
    for (tag, mode), rows in sorted(groups.items()):
        ttft = st.median([r["ttft_s"] for r in rows if r["ttft_s"]])
        total = st.median([r["total_s"] for r in rows])
        tps_vals = [r["tok_per_s"] for r in rows if r.get("tok_per_s")]
        tps = st.median(tps_vals) if tps_vals else None
        agg = {}
        if tegra:
            per_run = [window_stats(tegra, r["ts_start"], r["ts_end"]) for r in rows]
            per_run = [w for w in per_run if w]
            if per_run:
                for k in ("power_mean_w", "ram_max_mb", "temp_max_c"):
                    vals = [w[k] for w in per_run if k in w]
                    if vals:
                        agg[k] = round(max(vals) if k != "power_mean_w" else st.mean(vals), 2)
        row = {"tag": tag, "mode": mode, "n": len(rows), "ttft_s": round(ttft, 2),
               "total_s": round(total, 2), "tok_per_s": tps, **agg}
        summary.append(row)
        print(f"{tag:<22}{mode:<14}{len(rows):>3}{ttft:>8.2f}{total:>9.2f}"
              f"{tps or 0:>8.1f}{agg.get('power_mean_w', 0) or '':>7}"
              f"{agg.get('ram_max_mb', '') or '':>8}{agg.get('temp_max_c', '') or '':>7}")

    Path("bench/summary.json").write_text(json.dumps(summary, indent=1))
    print("\nwrote bench/summary.json")


if __name__ == "__main__":
    main()
