"""Record one run per sample through the live hub, for the offline copy.

Run on the board, where the hub is (the page's recorded replays must be what
the board really did): each sample is sent through /api/describe in its own
mode and in the generic baseline, and the streamed text, its timing and the
pipeline's trace are written back into demo/static/samples/recorded.json.

    python3 ops/record_samples.py [hub-url=http://localhost:8000]
"""

import json
import pathlib
import subprocess
import sys
import time

S = pathlib.Path(__file__).resolve().parents[1] / "demo/static/samples"
HUB = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

man = json.loads((S / "recorded.json").read_text())
for slug, s in man["samples"].items():
    for mode in (s["mode"], "general"):
        t0, deltas, run = time.time(), [], {}
        p = subprocess.Popen(["curl", "-sN", "-F", f"frame=@{S / s['file']}", "-F", f"mode={mode}",
                              f"{HUB}/api/describe"], stdout=subprocess.PIPE, text=True)
        for line in p.stdout:
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            if "error" in ev:
                sys.exit(f"{slug}/{mode}: {ev['error']}")
            if "delta" in ev:
                deltas.append([round(time.time() - t0, 3), ev["delta"]])
            if "trace" in ev:
                run["trace"] = ev["trace"]
            if ev.get("done"):
                run |= {k: ev[k] for k in ("ttft_s", "total_s", "sources") if ev.get(k)}
        assert deltas and run.get("ttft_s"), f"{slug}/{mode}: nothing streamed"
        s["runs"][mode] = {"text": "".join(d[1] for d in deltas), **run, "deltas": deltas}
        print(slug, mode, run["ttft_s"], run["total_s"], run.get("sources"), file=sys.stderr)
        time.sleep(7)       # the hub allows ten model calls a minute per address

status = json.loads(subprocess.run(["curl", "-s", f"{HUB}/api/status"], capture_output=True, text=True).stdout)
man["recorded_on"] = time.strftime("%Y-%m-%d")
man["model"] = ("gate, detector, text reader and outdoor/indoor/reader adapters on Qwen2.5-VL-3B (Q4_K_M)"
                if status.get("pipeline") == "multi" else man["model"])
(S / "recorded.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
