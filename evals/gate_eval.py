"""Tier 0 gate: held-out accuracy, threshold choice, ONNX parity, latency.

Everything except the parity check goes through veris.gate.Gater, i.e. the
exported ONNX that ships, not the PyTorch checkpoint.

Splits (from data/gate/manifest.jsonl, built by train/gate/build_data.py):
- threshold chosen on `val` (10% of the VizWiz train images, never trained on)
- quality/text reported on `test` = VizWiz official val images
- indoor reported on `test` = 20 of the 100 Places365 val images per category

    .venv/bin/python evals/gate_eval.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import onnx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from veris import gate  # noqa: E402
from veris.wearcam import PROFILES, Profile, degrade  # noqa: E402
from dataclasses import replace  # noqa: E402

ROOT = REPO / "data" / "gate"
OUT = REPO / "evals" / "results" / "gate.json"
FRR_TARGET = 0.05           # chosen operating point: <= 5% false rejects on val
PROFS = ("outdoor_walk", "indoor_walk", "indoor_worst")


def auroc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y) == 1
    if y.all() or not y.any():
        return None
    r = rankdata(s)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def binary(y, p, t=0.5) -> dict:
    y, pred = np.asarray(y) == 1, np.asarray(p) >= t
    tp, fp, fn = (y & pred).sum(), (~y & pred).sum(), (y & ~pred).sum()
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return {"n": int(len(y)), "pos": int(y.sum()), "acc": round(float((y == pred).mean()), 4),
            "f1": None if f1 is None else round(float(f1), 4),
            "auroc": None if (a := auroc(y, p)) is None else round(a, 4)}


_g = None


def _one(job):
    global _g
    _g = _g or gate.Gater(threads=1)
    i, path, prof = job
    img = cv2.imread(str(ROOT / path))
    if prof:
        # applied at the stored 256 px; long_side keeps degrade from resizing
        img = degrade(img, replace(PROFILES[prof], long_side=256), seed=i)
    return _g.probs(img)


def probs(rows, prof=None):
    with Pool(8) as p:
        return np.array(p.map(_one, [(i, r["path"], prof) for i, r in enumerate(rows)], 32))


def main() -> None:
    rows = [json.loads(l) for l in open(ROOT / "manifest.jsonl")]
    viz_val = [r for r in rows if r["src"] == "vizwiz" and r["split"] == "val"]
    viz_test = [r for r in rows if r["src"] == "vizwiz" and r["split"] == "test"]
    pl_test = [r for r in rows if r["src"] == "places" and r["split"] == "test"]
    g = gate.Gater(threads=1)
    res: dict = {"command": "flock /tmp/veris-gpu.lock .venv/bin/python train/gate/train.py && "
                            ".venv/bin/python evals/gate_eval.py",
                 "model": str(gate.MODEL.relative_to(REPO)),
                 "splits": {
                     "threshold": f"VizWiz train images, md5(name)%10==0 (`val`), N={len(viz_val)}",
                     "quality_text": f"VizWiz official val images (`test`), N={len(viz_test)}; "
                                     "1,700 random + the 300 already in dataset/frames",
                     "indoor": f"Places365 val, per category files 81-100 by name (`test`), "
                               f"N={len(pl_test)}; Places365 test labels are private",
                     "train": "VizWiz train (90% of 7,200 fetched, 2,500 enriched for rare flaws) + "
                              "Places365 val files 1-70 per category; files 71-80 = val"},
                 "labels": {"flaw": ">=2 of 5 VizWiz votes", "unusable": "unrecognizable >= 3 of 5",
                            "text": "VizWiz-Captions text_detected", "indoor": "IO_places365.txt"}}

    # threshold on val: largest t with <= FRR_TARGET of usable frames rejected
    pv = probs(viz_val)
    yv = np.array([r["unusable"] for r in viz_val])
    good = np.sort(pv[yv == 0, 4])
    t = float(good[int(np.ceil((1 - FRR_TARGET) * len(good))) - 1]) + 1e-6
    res["threshold"] = {"unusable_t": round(t, 4), "rule": f"FRR <= {FRR_TARGET:.0%} on val usable frames",
                        "val_frr": round(float((pv[yv == 0, 4] >= t).mean()), 4),
                        "val_catch": round(float((pv[yv == 1, 4] >= t).mean()), 4),
                        "val_n_usable": int((yv == 0).sum()), "val_n_unusable": int(yv.sum())}

    # VizWiz test: quality + text
    pt = probs(viz_test)
    y = {k: np.array([r[k] for r in viz_test]) for k in gate.OUTS if k != "indoor"}
    res["quality_test"] = {k: binary(y[k], pt[:, j]) for j, k in enumerate(gate.OUTS[:5])}
    yu = y["unusable"]
    curve = []
    for target in (0.01, 0.02, 0.05, 0.10, 0.20):
        tt = float(good[int(np.ceil((1 - target) * len(good))) - 1]) + 1e-6
        curve.append({"val_frr_target": target, "t": round(tt, 4),
                      "test_frr": round(float((pt[yu == 0, 4] >= tt).mean()), 4),
                      "test_catch": round(float((pt[yu == 1, 4] >= tt).mean()), 4)})
    rej = pt[:, 4] >= t
    issue_ok = [gate.ISSUES[int(np.argmax(pt[i, :4]))] in
                [k for k in gate.ISSUES if viz_test[i][k]] for i in np.where(rej & (yu == 1))[0]]
    res["usable_gate_test"] = {
        "t": round(t, 4), "n_usable": int((yu == 0).sum()), "n_unusable": int(yu.sum()),
        "false_reject_rate": round(float(rej[yu == 0].mean()), 4),
        "catch_rate": round(float(rej[yu == 1].mean()), 4),
        "issue_matches_a_voted_flaw": round(float(np.mean(issue_ok)), 4) if issue_ok else None,
        "curve": curve}
    res["text_test"] = {"clean": binary(y["text"], pt[:, 6])}
    for prof in PROFS:
        pp = probs(viz_test, prof)
        res["text_test"][prof] = binary(y["text"], pp[:, 6])
        print(prof, "text done", flush=True)

    # Places test: indoor, clean and wearcam; every frame is a usable scene
    res["indoor_test"], res["places_reject_rate"] = {}, {}
    yi = np.array([r["indoor"] for r in pl_test])
    for prof in (None,) + PROFS:
        pp = probs(pl_test, prof)
        name = prof or "clean"
        res["indoor_test"][name] = binary(yi, pp[:, 5])
        res["places_reject_rate"][name] = {"n": len(pl_test),
                                           "reject_rate": round(float((pp[:, 4] >= t).mean()), 4),
                                           "mean_p_blur": round(float(pp[:, 0].mean()), 4),
                                           "mean_p_dark": round(float(pp[:, 1].mean()), 4)}
        print(name, res["indoor_test"][name], flush=True)
    res["places_reject_rate"]["note"] = ("Places scenes are all usable, so the reject rate "
                                         "is a false-reject rate; degrade applied at 256 px")

    # best_of: one scene, 5 frames from still to a 90 deg/s head turn at 1/30 s
    rng, hits, hits_le15, g1 = np.random.default_rng(0), 0, 0, gate.Gater(threads=1, unusable_t=t)
    speeds = np.array([0, 15, 30, 60, 90])
    sample = pl_test[::25]
    for i, r in enumerate(sample):
        img = cv2.imread(str(ROOT / r["path"]))
        order = rng.permutation(5)
        frames = [degrade(img, Profile(long_side=256, head_deg_s=float(speeds[k]), light=0.5),
                          seed=i) for k in order]
        b = g1.best_of(frames)
        hits += b is not None and speeds[order[b]] == 0
        hits_le15 += b is not None and speeds[order[b]] <= 15
    res["best_of_test"] = {"n_scenes": len(sample), "frames_per_scene": 5,
                           "head_deg_s": speeds.tolist(), "exposure_s": 1 / 30, "light": 0.5,
                           "picked_still": round(hits / len(sample), 4),
                           "picked_le_15deg_s": round(hits_le15 / len(sample), 4)}

    # ONNX vs PyTorch on 64 test images
    import timm
    import torch
    m = timm.create_model("mobilenetv3_small_100", num_classes=7)
    m.load_state_dict(torch.load(ROOT / "gate.pt", map_location="cpu"))
    m.eval()
    diffs = []
    for r in viz_test[:32] + pl_test[:32]:
        x = gate.preprocess(cv2.imread(str(ROOT / r["path"])))
        with torch.no_grad():
            a = m(torch.from_numpy(x)).numpy()
        diffs.append(np.abs(a - g.sess.run(None, {"image": x})[0]).max())
    model = onnx.load(gate.MODEL)
    res["export"] = {
        "opset": model.opset_import[0].version,
        "input": [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim],
        "params": int(sum(np.prod(i.dims) for i in model.graph.initializer)),
        "torch_params": int(sum(p.numel() for p in m.parameters())),
        "file_mb": round(os.path.getsize(gate.MODEL) / 1e6, 2),
        "max_abs_logit_diff_vs_torch": float(max(diffs)), "parity_n": len(diffs)}

    # CPU latency, median of 200, model only and full run() on a 1280x720 frame
    frame = cv2.imread(str(ROOT / viz_test[0]["path"]))
    frame = cv2.resize(frame, (1280, 720))
    x = gate.preprocess(frame)
    lat = {"cpu": next((l.split(":")[1].strip() for l in open("/proc/cpuinfo")
                        if l.startswith("model name")), platform.processor()),
           "cores": os.cpu_count(), "runs": 200,
           "loadavg_1m_at_start": os.getloadavg()[0],
           "note": "shared dev box; other agents were training, so treat as an upper bound"}
    for name, th in (("single_thread", 1), ("default_threads", 0)):
        gg = gate.Gater(threads=th)
        for fn, key in ((lambda: gg.sess.run(None, {"image": x}), "model_ms"),
                        (lambda: gg.run(frame), "run_1280x720_ms")):
            for _ in range(20):
                fn()
            ts = []
            for _ in range(200):
                t0 = time.perf_counter()
                fn()
                ts.append((time.perf_counter() - t0) * 1e3)
            lat[f"{name}_{key}"] = round(float(np.median(ts)), 3)
    lat["loadavg_1m_at_end"] = os.getloadavg()[0]
    res["latency"] = lat
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("threshold", "usable_gate_test", "export", "latency")},
                     indent=1))


if __name__ == "__main__":
    main()
