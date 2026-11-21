"""Replay a simulated downtown walk against the context engine and print what
the pack would receive.

  uv run python -m context_engine.walk_trace [--url http://localhost:8090]

Trace: south along N High St from E Gay St toward W Broad St in downtown
Columbus, OH, then a turn onto W Broad — chosen to exercise: street change,
heading swing, crossings, dense POIs, and the composer cache (fixes are
~15-25 m apart; composition should re-run only a few times).
"""

import argparse
import json
import time
import urllib.request

TRACE = [
    # (lat, lon, heading_deg, speed_mps)
    (39.96480, -83.00075, 185, 1.3),
    (39.96458, -83.00078, 185, 1.3),
    (39.96436, -83.00081, 185, 1.3),   # ~25m — same block, expect cache
    (39.96408, -83.00085, 185, 1.4),   # ~55m from start — recompose
    (39.96380, -83.00088, 185, 1.3),
    (39.96358, -83.00092, 230, 0.9),   # slowing toward corner
    (39.96345, -83.00125, 275, 1.2),   # turned onto W Broad — heading swing
    (39.96338, -83.00168, 275, 1.3),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8090")
    args = p.parse_args()
    for i, (lat, lon, hdg, spd) in enumerate(TRACE):
        body = json.dumps({"lat": lat, "lon": lon, "heading_deg": hdg,
                           "speed_mps": spd, "timestamp": time.time()}).encode()
        req = urllib.request.Request(f"{args.url}/context", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=30) as r:
            out = json.load(r)
        dt = time.time() - t0
        print(f"[fix {i}] ({out['composed_by']}, {dt:.2f}s, llm_calls={out['llm_calls_total']})")
        print(f"  {out['context']}\n")


if __name__ == "__main__":
    main()
