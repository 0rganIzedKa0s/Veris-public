"""Gate indoor/outdoor cross-check on photos the gate never trained on.

COCO val2017 rows of data/experts/manifest.jsonl (split == "val"; scene from
the COCO-category rule in train/experts/build.py, not a human label), clean
and through wearcam outdoor_walk / indoor_walk at native resolution, for
gate_v1.onnx (Places-only indoor head) and gate.onnx (v2, + street rows).
Also indoor_p on the three hand-picked test_frames.

    .venv/bin/python evals/gate_crosscheck.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from evals.gate_eval import auroc  # noqa: E402
from veris.gate import Gater  # noqa: E402
from veris.wearcam import degrade  # noqa: E402

MODELS = {"v1": REPO / "data/gate/gate_v1.onnx", "v2": REPO / "data/gate/gate.onnx"}
FRAMES = ("crossing.jpg", "kitchen.jpg", "sign.jpg")


def main() -> None:
    rows = [r for r in map(json.loads, open(REPO / "data/experts/manifest.jsonl"))
            if r["split"] == "val"]
    imgs = [cv2.imread(str(REPO / "data/coco/images/val" / r["image"])) for r in rows]
    y = np.array([r["scene"] == "indoor" for r in rows])
    res = {"command": ".venv/bin/python evals/gate_crosscheck.py",
           "dataset": "COCO val2017 scene subset, data/experts/manifest.jsonl split == val "
                      "(never used to train the gate)",
           "n": {"indoor": int(y.sum()), "outdoor": int((~y).sum())},
           "note": "scene label = COCO category rule in train/experts/build.py, not a human "
                   "indoor/outdoor label; wearcam applied at native COCO resolution",
           "models": {k: str(v.relative_to(REPO)) for k, v in MODELS.items()}}
    views = {"clean": imgs,
             **{p: [degrade(im, p, seed=i) for i, im in enumerate(imgs)]
                for p in ("outdoor_walk", "indoor_walk")}}
    for name, path in MODELS.items():
        g = Gater(path)
        out = {}
        for view, ims in views.items():
            p = np.array([g.probs(im) for im in ims])
            pred = p[:, 5] >= 0.5
            out[view] = {"acc": round(float((pred == y).mean()), 4),
                         "indoor_acc": round(float(pred[y].mean()), 4),
                         "outdoor_acc": round(float((~pred[~y]).mean()), 4),
                         "auroc": round(auroc(y.astype(int), p[:, 5]), 4),
                         "usable_rate": round(float((p[:, 4] < g.t).mean()), 4)}
        out["test_frames_indoor_p"] = {
            f: round(g.run(cv2.imread(str(REPO / "test_frames" / f))).indoor_p, 4) for f in FRAMES}
        res[name] = out
        print(name, json.dumps(out), flush=True)
    # the deployed vote: gate v2 corrected by the reflex detector's street classes
    from train.nav.build_oi import CLASSES
    from veris.packet import indoor_with_objects
    from veris.reflex import onnx_detector
    detect = onnx_detector(str(REPO / "data/nav/dfine_s_oi18_640.onnx"), CLASSES)
    g, out = Gater(MODELS["v2"]), {}
    for view, ims in views.items():
        p = np.array([indoor_with_objects(g.probs(im)[5], [d[5] for d in detect(im)]) for im in ims])
        pred = p >= 0.5
        out[view] = {"acc": round(float((pred == y).mean()), 4),
                     "indoor_acc": round(float(pred[y].mean()), 4),
                     "outdoor_acc": round(float((~pred[~y]).mean()), 4),
                     "auroc": round(auroc(y.astype(int), p), 4)}
    out["test_frames_indoor_p"] = {
        f: round(indoor_with_objects(g.run(im).indoor_p, [d[5] for d in detect(im)]), 4)
        for f in FRAMES for im in [cv2.imread(str(REPO / "test_frames" / f))]}
    res["v2_plus_reflex_street_classes"] = out
    print("v2+reflex", json.dumps(out), flush=True)
    (REPO / "evals/results/gate_coco_crosscheck.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
