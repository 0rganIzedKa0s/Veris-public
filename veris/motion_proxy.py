"""Camera-motion proxy: wearer motion state from consecutive frames.

The bench rig has no temple IMU yet, so this stands in for it: consecutive
head-camera frames -> Motion (stationary / walking / turning) plus a
head-yaw delta, feeding Arbiter.update_motion() through the exact same API
the IMU driver will use (docs/sensor-fusion-switching.md, "concrete asks").

Method — PIL + numpy only, no OpenCV, <50 ms per frame pair on CPU:

1. Decode with PIL JPEG draft mode (fast reduced-scale decode), grayscale,
   downsample to a fixed working width (default 96 px, aspect preserved).
2. Mean absolute pixel difference (MAD, 0-255) against the previous frame:
   "is anything changing at all".
3. Global horizontal shift: compare the two frames' column-mean intensity
   profiles at every integer shift within +/-max_shift_px and keep the
   minimum-error shift — but accept it only when it beats the zero-shift
   error by a clear margin (shift_confidence). Walking bob and parallax
   decorrelate frames *without* a coherent shift, and a low-confidence
   match must not synthesize yaw.
4. Classify: accepted shift large enough -> TURNING; else MAD above the
   stationary floor -> WALKING; else STATIONARY. The accepted shift
   converts to degrees via the camera's horizontal FOV and accumulates
   into a running yaw proxy for the arbiter's stale-frame guard.

Sign convention: positive yaw = pan right (clockwise from above). Panning
right moves scene content left in the image, so yaw_delta = -shift * deg/px.

Known limits (fine for a proxy): pure forward translation reads as WALKING
regardless of speed; exposure swings and large moving objects add MAD that
can promote stationary -> walking. Thresholds are per frame *pair*, so they
assume a roughly steady capture cadence (~2 Hz in the replay scenarios).
"""

import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .arbiter import Motion


@dataclass(frozen=True)
class MotionEstimate:
    motion: Motion
    yaw_delta_deg: float   # yaw change over this frame pair (+ = pan right)
    yaw_deg: float         # cumulative yaw proxy since reset()
    mad: float             # mean abs pixel diff at working resolution, 0-255
    shift_px: int          # accepted coherent horizontal shift, working px
    elapsed_ms: float      # decode + estimate time for this frame


class MotionProxy:
    def __init__(
        self,
        width: int = 96,             # working width in px
        hfov_deg: float = 60.0,      # camera horizontal field of view
        mad_stationary: float = 2.0, # MAD below this -> stationary
        yaw_turn_deg: float = 1.5,   # |yaw delta| per pair at/above this -> turning
        max_shift_px: int = 16,      # shift search range at working width
        shift_confidence: float = 0.5,  # accept shift only if err < this * err(0)
    ):
        self.width = width
        self.hfov_deg = hfov_deg
        self.mad_stationary = mad_stationary
        self.yaw_turn_deg = yaw_turn_deg
        self.max_shift_px = max_shift_px
        self.shift_confidence = shift_confidence
        self.yaw_deg = 0.0
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        """Forget the previous frame and zero the yaw accumulator."""
        self._prev = None
        self.yaw_deg = 0.0

    # ── per-frame entry point ────────────────────────────────────────
    def process(self, frame) -> MotionEstimate | None:
        """Feed the next frame (path or PIL.Image). Returns None for the
        first frame of a sequence (no pair yet) or on a resolution change."""
        t0 = time.perf_counter()
        cur = self._load(frame)
        prev, self._prev = self._prev, cur
        if prev is None or prev.shape != cur.shape:
            return None

        mad = float(np.abs(cur - prev).mean())
        shift = self._coherent_shift(prev, cur)
        yaw_delta = -shift * (self.hfov_deg / self.width)

        if abs(yaw_delta) >= self.yaw_turn_deg:
            motion = Motion.TURNING
        elif mad >= self.mad_stationary:
            motion = Motion.WALKING
        else:
            motion = Motion.STATIONARY

        self.yaw_deg += yaw_delta
        return MotionEstimate(
            motion=motion,
            yaw_delta_deg=yaw_delta,
            yaw_deg=self.yaw_deg,
            mad=mad,
            shift_px=shift,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )

    # ── internals ────────────────────────────────────────────────────
    def _load(self, frame) -> np.ndarray:
        if isinstance(frame, Image.Image):
            img = frame
        else:
            img = Image.open(frame)
            try:
                # JPEG fast path: decode at reduced scale straight from DCT
                img.draft("L", (self.width, self.width))
            except Exception:
                pass
        g = img.convert("L")
        w, h = g.size
        th = max(1, round(self.width * h / w))
        g = g.resize((self.width, th), Image.BILINEAR)
        return np.asarray(g, dtype=np.float32)

    def _coherent_shift(self, prev: np.ndarray, cur: np.ndarray) -> int:
        """Best global horizontal shift of scene content prev -> cur
        (positive = content moved right), 0 unless clearly evidenced."""
        p = prev.mean(axis=0)
        c = cur.mean(axis=0)
        if float(p.std()) < 1.0:      # featureless (blank wall): no evidence
            return 0
        n = p.size

        def err(s: int) -> float:
            if s >= 0:
                d = c[s:] - p[: n - s] if s else c - p
            else:
                d = c[: n + s] - p[-s:]
            return float(np.abs(d).mean())

        err0 = err(0)
        if err0 == 0.0:               # identical frames
            return 0
        best_s, best_e = 0, err0
        for s in range(-self.max_shift_px, self.max_shift_px + 1):
            if s == 0:
                continue
            e = err(s)
            if e < best_e:
                best_s, best_e = s, e
        if best_e < self.shift_confidence * err0:
            return best_s
        return 0
