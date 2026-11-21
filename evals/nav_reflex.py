"""Tier 1 speech policy on synthetic track sequences (no model, no images).

Runs veris.reflex.demo() (approaching person, pole passed on the side, pole
ahead, person leaving the corridor, planner-raised traffic light; its asserts
must pass) and a crowded-scene timing case: 40 objects every frame for 300
frames. Times are this desktop's CPU for the policy alone (tracking + geometry
+ speech choice), detector excluded.

  .venv/bin/python evals/nav_reflex.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from veris.ranging import Camera  # noqa: E402
from veris.reflex import Reflex, demo  # noqa: E402


def crowded(n_obj=40, n_frames=300):
    rng = np.random.default_rng(0)
    cam = Camera.from_hfov(1280, 720)
    rf = Reflex(lambda d: d, cam)
    x = rng.uniform(0, 1200, n_obj); y = rng.uniform(300, 600, n_obj)
    lat = []
    for k in range(n_frames):
        x += rng.normal(0, 2, n_obj)
        dets = [(xi, yi, xi + 60, yi + 110, 0.8, "person") for xi, yi in zip(x, y)]
        t0 = time.perf_counter(); rf.update(dets, k / 30); lat.append(time.perf_counter() - t0)
    lat = np.array(lat[10:]) * 1e3
    return {"objects_per_frame": n_obj, "median": round(float(np.median(lat)), 3),
            "p99": round(float(np.percentile(lat, 99)), 3)}


if __name__ == "__main__":
    res = {"command": ".venv/bin/python evals/nav_reflex.py",
           "split": "synthetic tracks projected through a 1.6 m, 10 deg-down, 80 deg HFOV camera; "
                    "30 fps; 1% box-edge noise",
           "scenarios_utterances": demo(), "crowded_policy_ms": crowded()}
    (REPO / "evals/results/nav_reflex.json").write_text(json.dumps(res, indent=1))
    print(res["crowded_policy_ms"])
