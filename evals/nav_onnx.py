"""ONNX export parity and desktop-CPU timing for the Tier 1 detector.

Parity: PyTorch fp32 (CPU) vs onnxruntime (CPU) on the first N held-out
images: every PyTorch detection above 0.3 is matched to the same-class ONNX
detection with the highest IoU; reports the worst IoU and score gap. Timing is this desktop's CPU, NOT the Orin; the Orin
number comes from bench/orin/nav_trt.sh.

  .venv/bin/python evals/nav_onnx.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from transformers import DFineForObjectDetection

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from veris.ranging import Camera  # noqa: E402
from veris.reflex import Reflex, onnx_detector  # noqa: E402

CKPT = REPO / "train/nav/runs/dfine_s_oi18"
ONNX = REPO / "data/nav/dfine_s_oi18_640.onnx"
OI = REPO / "data/nav/oi"
N = 8


THR = 0.3


def dets(p, b):
    q, c = np.nonzero(p > THR)
    xyxy = np.c_[b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2]
    return [(ci, p[qi, ci], xyxy[qi]) for qi, ci in zip(q, c)]


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = w * h
    return float(i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-12))


def prep(img):
    x = cv2.resize(img, (640, 640))[:, :, ::-1].transpose(2, 0, 1)[None]
    return np.ascontiguousarray(x, np.float32) / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--onnx", type=Path, default=ONNX)
    ap.add_argument("--out", type=Path, default=REPO / "evals/results/nav_onnx.json")
    a = ap.parse_args()
    d = json.load(open(OI / "heldout.json"))
    classes = [c["name"] for c in sorted(d["categories"], key=lambda c: c["id"])]
    m = DFineForObjectDetection.from_pretrained(a.ckpt).eval()
    sess = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
    rows, counts = [], []
    for im in d["images"][:N]:
        x = prep(cv2.imread(str(OI / "images/heldout" / im["file_name"])))
        with torch.no_grad():
            o = m(pixel_values=torch.from_numpy(x))
        lg, bx = sess.run(None, {"pixel_values": x})
        # compare detections, not raw query slots: low-score queries can swap
        # order between runtimes when encoder top-k scores tie
        pt, on = dets(torch.sigmoid(o.logits)[0].numpy(), o.pred_boxes[0].numpy()), \
            dets(1 / (1 + np.exp(-lg[0])), bx[0])
        for c, s, b in pt:
            cand = [(iou(b, b2), abs(s - s2)) for c2, s2, b2 in on if c2 == c]
            best = max(cand) if cand else (0.0, s)
            rows.append({"iou": best[0], "score_diff": best[1]})
        counts.append((len(pt), len(on)))
    x = prep(cv2.imread(str(OI / "images/heldout" / d["images"][0]["file_name"])))
    for _ in range(3):
        sess.run(None, {"pixel_values": x})
    lat = []
    for _ in range(20):
        t0 = time.perf_counter(); sess.run(None, {"pixel_values": x}); lat.append(time.perf_counter() - t0)
    # whole reflex step (preprocess + model + postprocess + track + policy) on one frame
    img = cv2.imread(str(OI / "images/heldout" / d["images"][0]["file_name"]))
    rf = Reflex(onnx_detector(str(a.onnx), classes), Camera.from_hfov(img.shape[1], img.shape[0]))
    step = []
    for k in range(20):
        t0 = time.perf_counter(); rf.step(img, k / 30); step.append(time.perf_counter() - t0)
    res = {
        "command": ".venv/bin/python evals/nav_onnx.py",
        "onnx": str(a.onnx), "onnx_mb": round(a.onnx.stat().st_size / 1e6, 2),
        "opset": 17, "input": "1x3x640x640 RGB 0..1",
        "split": "Open Images held-out (heldout.json), first N images", "n": N,
        "parity": {"threshold": THR, "n_torch_dets": len(rows),
                   "det_counts_equal": all(p == o for p, o in counts),
                   "min_matched_iou": round(min(r["iou"] for r in rows), 4) if rows else None,
                   "max_score_diff": round(float(max(r["score_diff"] for r in rows)), 4) if rows else None},
        "desktop_cpu_ms": {"ort_median": round(1e3 * float(np.median(lat)), 1),
                           "reflex_step_median": round(1e3 * float(np.median(step)), 1),
                           "note": "desktop x86 CPU, not the Orin"},
    }
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
