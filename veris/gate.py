"""Tier 0: the gate. Runs on every frame, before anything expensive.

A MobileNetV3-small exported to ONNX (data/gate/gate.onnx; trained by
train/gate/train.py, measured by evals/gate_eval.py) gives seven
probabilities: blur, dark, bright, obstructed, unusable, indoor, text.

- usable = p(unusable) < UNUSABLE_T. The threshold is set for a low
  false-reject rate on held-out VizWiz frames: telling a blind user to
  "hold still" when the frame was fine is the costly error.
- issue (only when unusable) = the most likely of blur/dark/bright/obstructed.
- sharpness = 1 - p(blur), what best_of() ranks by.

onnxruntime only: no torch, no training code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort

from veris.packet import Gate

MODEL = Path(__file__).resolve().parents[1] / "data" / "gate" / "gate.onnx"
OUTS = ("blur", "dark", "bright", "obstructed", "unusable", "indoor", "text")
ISSUES = OUTS[:4]
# chosen by evals/gate_eval.py (val split, false-reject target); see
# evals/results/gate.json -> threshold
UNUSABLE_T = 0.4218       # evals/results/gate.json threshold.unusable_t: 5% false rejects on val
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(bgr: np.ndarray) -> np.ndarray:
    """Whole frame (never cropped) -> 1x3x224x224 float32, ImageNet norm."""
    x = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA)
    x = (x[:, :, ::-1].astype(np.float32) / 255 - MEAN) / STD
    return x.transpose(2, 0, 1)[None].copy()


class Gater:
    def __init__(self, path: str | Path = MODEL, threads: int = 1,
                 unusable_t: float = UNUSABLE_T):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads        # 0 = onnxruntime default
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.t = unusable_t

    def probs(self, bgr: np.ndarray) -> np.ndarray:
        logit = self.sess.run(None, {"image": preprocess(bgr)})[0][0]
        return 1 / (1 + np.exp(-logit))

    def run(self, bgr: np.ndarray) -> Gate:
        p = self.probs(bgr)
        usable = bool(p[4] < self.t)
        return Gate(usable=usable,
                    issue=None if usable else ISSUES[int(np.argmax(p[:4]))],
                    indoor_p=float(p[5]), text_p=float(p[6]),
                    sharpness=float(1 - p[0]))

    def best_of(self, frames: Sequence[np.ndarray]) -> int | None:
        """Index of the sharpest usable frame of the last N; None if none is
        usable (the caller says "hold still" instead of describing blur)."""
        gates = [self.run(f) for f in frames]
        ok = [i for i, g in enumerate(gates) if g.usable]
        return max(ok, key=lambda i: gates[i].sharpness) if ok else None


def demo() -> None:
    import sys

    from veris.wearcam import degrade

    assert "torch" not in sys.modules
    g = Gater()
    # a held-out Places365 test frame: a pharmacy (indoor, shelf labels)
    img = cv2.imread(str(MODEL.parent / "img" / "places" / "Places365_val_00027753.jpg"))
    assert img is not None, "run train/gate/build_data.py first"
    clean = g.run(img)
    assert clean.usable and clean.issue is None, clean
    # the same scene through a dim room with a fast head turn
    worst = g.run(degrade(img, "indoor_worst", seed=1))
    assert worst.sharpness < clean.sharpness, (worst, clean)
    # a lens cap is unusable and called dark
    black = g.run(np.zeros((720, 1280, 3), np.uint8))
    assert not black.usable and black.issue == "dark", black
    frames = [degrade(img, "indoor_worst", seed=1), img, black_img := np.zeros_like(img)]
    assert g.best_of(frames) == 1
    assert g.best_of([black_img]) is None
    print("clean", clean)
    print("worst", worst)
    print("black", black)


if __name__ == "__main__":
    demo()
