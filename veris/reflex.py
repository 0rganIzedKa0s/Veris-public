"""Tier 1 reflex: detections -> IoU tracks -> Obj records -> at most one short
spoken observation per frame. No LLM anywhere on this path.

The detector is injected: any callable frame -> [(x1, y1, x2, y2, score, label)]
in pixels. `onnx_detector` wraps the exported D-FINE model; tests pass a fake.

Speech priority (higher level wins, then nearer):
  3  closing fast: ttc < 3 s and in or beside the walking corridor
  2  in path within 4 m (or its base is below the frame: closer than the
     nearest visible ground)
  1  the planner raised this class (Situation.class_priors > 1)
  0  silence
A track speaks once per level; it speaks again only when its level rises and
its cooldown has passed. Phrases are observations (docs/style-guide.md U1):
"Pole, 1 o'clock, two meters." Never advice.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from veris.packet import Obj, Situation, clock_from_bearing, side_from_bearing
from veris.ranging import CORRIDOR_MARGIN, Camera, bearing_deg, in_path, range_m, ttc_s

Det = tuple[float, float, float, float, float, str]

# Classes whose box bottom is where they meet the ground. Signs and lights hang
# in the air, so their box bottom says nothing about range.
GROUNDED = {"person", "car", "bus", "truck", "bicycle", "motorcycle", "street light",
            "fire hydrant", "waste container", "bench", "chair", "table", "door",
            "stairs", "dog"}
SPOKEN = {"street light": "pole", "waste container": "bin"}
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight"]
SAY_RANGE_MAX = 8.0     # m; at 8 m a 2 deg pitch error is already -15%/+22% (nav_ranging.json)
CLOSING_TTC = 3.0
PATH_RANGE = 4.0
SIDE_SLACK = 0.5        # m beyond the corridor still counted for closing objects


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


@dataclass
class Track:
    id: int
    label: str
    box: tuple
    conf: float
    hist: deque = field(default_factory=deque)     # (t, box)
    hits: int = 0
    last_t: float = 0.0
    spoken_level: int = 0
    spoken_t: float = -1e9
    level: int = 0
    obj: Obj | None = None
    near: bool = False          # base below the frame edge


def distance_words(o: Obj, near_limit_m: float | None) -> str | None:
    """near_limit_m: set when the base is below the frame, i.e. the object is
    closer than the nearest ground the camera sees."""
    if near_limit_m is not None:
        return f"under {WORDS[max(1, min(8, math.ceil(near_limit_m)))]} meters"
    if o.dist_m is not None and o.dist_m < 1.0:
        return "within reach"
    if o.dist_m is None or o.dist_m > SAY_RANGE_MAX:
        return None
    n = max(1, round(o.dist_m))
    return f"{WORDS[n]} meter" + ("s" if n > 1 else "")


def utterance(o: Obj, level: int, near_limit_m: float | None = None,
              clock_face: bool = True) -> str:
    where = f"{clock_from_bearing(o.bearing_deg)} o'clock" if clock_face \
        else side_from_bearing(o.bearing_deg)
    parts = [SPOKEN.get(o.label, o.label).capitalize(), where]
    d = distance_words(o, near_limit_m)
    if d:
        parts.append(d)
    if level == 3:
        parts.append("approaching")
    return ", ".join(parts) + "."


class Reflex:
    def __init__(self, detector: Callable[[np.ndarray], list[Det]], cam: Camera,
                 iou_thr: float = 0.3, max_age_s: float = 0.5, min_hits: int = 3,
                 window_s: float = 0.7, cooldown_s: float = 2.0, min_gap_s: float = 1.0,
                 clock_face: bool = True):
        self.detector, self.cam = detector, cam
        self.iou_thr, self.max_age_s, self.min_hits = iou_thr, max_age_s, min_hits
        self.window_s, self.cooldown_s, self.min_gap_s = window_s, cooldown_s, min_gap_s
        self.clock_face = clock_face
        self.tracks: list[Track] = []
        self.next_id = 0
        self.last_say_t = -1e9

    def step(self, frame, t: float, situation: Situation = Situation()):
        dets = self.detector(frame)
        return self.update(dets, t, situation)

    def update(self, dets: list[Det], t: float, situation: Situation = Situation()):
        """Tracks + speech for one frame. Returns (objects, utterance or None)."""
        self._associate(dets, t)
        live = [tr for tr in self.tracks if tr.last_t == t and tr.hits >= self.min_hits]
        for tr in live:
            self._measure(tr)
            tr.level = self._level(tr, situation)
            # forget a spoken level once it has dropped and the cooldown passed,
            # so a later rise (re-entering the path) speaks again
            if tr.level < tr.spoken_level and t - tr.spoken_t >= self.cooldown_s:
                tr.spoken_level = tr.level
        say = None
        cands = [tr for tr in live if tr.level > tr.spoken_level
                 and (tr.spoken_level == 0 or t - tr.spoken_t >= self.cooldown_s)]
        if cands and (t - self.last_say_t >= self.min_gap_s or max(c.level for c in cands) == 3):
            pri = situation.class_priors
            best = min(cands, key=lambda c: (-c.level, -pri.get(c.label, 1.0),
                                             c.obj.dist_m if c.obj.dist_m is not None else
                                             (0.0 if c.near else 99.0)))
            near = float(range_m(self.cam, self.cam.cx, self.cam.height - 1)) if best.near else None
            say = utterance(best.obj, best.level, near, self.clock_face)
            best.spoken_level, best.spoken_t, self.last_say_t = best.level, t, t
        return [tr.obj for tr in live], say

    def _associate(self, dets: list[Det], t: float) -> None:
        self.tracks = [tr for tr in self.tracks if t - tr.last_t <= self.max_age_s]
        pairs = sorted(((iou(tr.box, d), i, j) for i, tr in enumerate(self.tracks)
                        for j, d in enumerate(dets) if tr.label == d[5]), reverse=True)
        used_t, used_d = set(), set()
        for v, i, j in pairs:      # ponytail: greedy IoU, Hungarian if crowds swap ids
            if v < self.iou_thr or i in used_t or j in used_d:
                continue
            used_t.add(i); used_d.add(j)
            self._hit(self.tracks[i], dets[j], t)
        for j, d in enumerate(dets):
            if j not in used_d:
                tr = Track(self.next_id, d[5], tuple(d[:4]), d[4])
                self.next_id += 1
                self.tracks.append(tr)
                self._hit(tr, d, t)

    def _hit(self, tr: Track, d: Det, t: float) -> None:
        tr.box, tr.conf, tr.last_t = tuple(d[:4]), d[4], t
        tr.hits += 1
        tr.hist.append((t, tr.box))
        while tr.hist and t - tr.hist[0][0] > self.window_s:
            tr.hist.popleft()

    def _scale_series(self, tr: Track):
        """Box height, or width if the window ever touches top/bottom; None if both clip."""
        W, H = self.cam.width, self.cam.height
        bx = [b for _, b in tr.hist]
        if all(b[1] > 1 and b[3] < H - 2 for b in bx):
            return [b[3] - b[1] for b in bx]
        if all(b[0] > 1 and b[2] < W - 2 for b in bx):
            return [b[2] - b[0] for b in bx]
        return None

    def _measure(self, tr: Track) -> None:
        x1, y1, x2, y2 = tr.box
        u = (x1 + x2) / 2
        grounded = tr.label in GROUNDED
        tr.near = grounded and y2 >= self.cam.height - 2
        vb = min(y2, self.cam.height - 1)
        dist = float(range_m(self.cam, u, vb)) if grounded and not tr.near else None
        if dist is not None and not math.isfinite(dist):
            dist = None
        s = self._scale_series(tr)
        ttc = ttc_s([h[0] for h in tr.hist], s) if grounded and s else None
        tr.obj = Obj(tr.label, float(bearing_deg(self.cam, u, vb)), tr.conf, dist, ttc,
                     grounded and in_path(self.cam, x1, x2, vb), tr.id)

    def _level(self, tr: Track, situation: Situation) -> int:
        o = tr.obj
        if o.label in GROUNDED:
            x1, _, x2, y2 = tr.box
            vb = min(y2, self.cam.height - 1)
            beside = in_path(self.cam, x1, x2, vb, margin=CORRIDOR_MARGIN + SIDE_SLACK)
            if o.ttc_s is not None and o.ttc_s < CLOSING_TTC and beside:
                return 3
            if o.in_path and (tr.near or (o.dist_m is not None and o.dist_m <= PATH_RANGE)):
                return 2
        if situation.class_priors.get(o.label, 1.0) > 1.0:
            return 1
        return 0


def onnx_detector(path: str, classes: list[str], thr: float = 0.5, size: int = 640,
                  providers=("CPUExecutionProvider",)):
    """The exported D-FINE model (train/nav/export_onnx.py) as a Det callable.
    Input: BGR uint8 frame. D-FINE's processor resizes to size x size and
    scales to 0..1 with no mean/std normalisation."""
    import cv2
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=list(providers))

    def detect(frame: np.ndarray) -> list[Det]:
        h, w = frame.shape[:2]
        x = cv2.resize(frame, (size, size))[:, :, ::-1].transpose(2, 0, 1)[None]
        logits, boxes = sess.run(None, {"pixel_values": x.astype(np.float32) / 255})
        p = 1 / (1 + np.exp(-logits[0]))
        q, c = np.nonzero(p > thr)
        out = []
        for qi, ci in zip(q, c):
            cx, cy, bw, bh = boxes[0, qi]
            out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w,
                        (cy + bh / 2) * h, float(p[qi, ci]), classes[ci]))
        return out
    return detect


# ---- synthetic check -------------------------------------------------------

def _project(cam: Camera, X: float, Z: float, w_m: float, h_m: float):
    """Pixel box of an upright w x h object whose base centre is at ground (X, Z)."""
    th = math.radians(cam.pitch_deg)

    def px(x, y_up, z):
        yc = (cam.height_m - y_up) * math.cos(th) - z * math.sin(th)
        zc = (cam.height_m - y_up) * math.sin(th) + z * math.cos(th)
        return cam.cx + cam.fx * x / zc, cam.cy + cam.fy * yc / zc
    xl, yb = px(X - w_m / 2, 0, Z)
    xr, _ = px(X + w_m / 2, 0, Z)
    _, yt = px(X, h_m, Z)
    return (max(0.0, xl), max(0.0, yt), min(cam.width - 1.0, xr), min(cam.height - 1.0, yb))


def _run(cam, scene, n, fps=30, situation=Situation(), noise=0.0):
    """scene(k, t) -> list of (label, X, Z, w, h). Returns [(t, utterance)]."""
    rng = np.random.default_rng(0)
    rf = Reflex(lambda dets: dets, cam)
    said, lat = [], []
    for k in range(n):
        t = k / fps
        dets = []
        for label, X, Z, w, h in scene(k, t):
            if Z <= 0.3:
                continue
            b = np.array(_project(cam, X, Z, w, h))
            # detector noise on each edge, in proportion to the box; an edge
            # clamped at the frame border stays there, as a real detector's does
            free = (b != [0.0, 0.0, cam.width - 1.0, cam.height - 1.0])
            b += free * noise * np.r_[b[2] - b[0], b[3] - b[1]].repeat(2)[[0, 2, 1, 3]] \
                * rng.standard_normal(4)
            if b[2] - b[0] > 2 and b[0] < cam.width - 2 and b[2] > 2:
                dets.append((*b, 0.9, label))
        t0 = time.perf_counter()
        _, say = rf.step(dets, t, situation)
        lat.append(time.perf_counter() - t0)
        if say:
            said.append((round(t, 2), say))
    return said, lat


def demo() -> dict:
    cam = Camera.from_hfov(1280, 720, 80, 1.6, 10.0)
    walk = 1.3
    lat, out = [], {}

    # 1. a person walking at us down the corridor from 12 m, user walking too
    said, l = _run(cam, lambda k, t: [("person", 0.1, 12 - 2 * walk * t, 0.5, 1.7)], 150, noise=0.01)
    lat += l
    out["approaching_person"] = said
    assert said and said[0][1].startswith("Person, 12 o'clock") and said[0][1].endswith("approaching.")
    assert 12 / (2 * walk) - said[0][0] > 2.5, "must speak with >2.5 s to contact left"
    assert len(said) <= 2, "one track must not chatter"

    # 2. a pole 1.6 m to the right, passed on the side: never in the path, silent
    said, l = _run(cam, lambda k, t: [("street light", 1.6, 9 - walk * t, 0.25, 3.5)], 200, noise=0.01)
    lat += l
    out["pole_passed_on_side"] = said
    assert said == []

    # 3. a pole dead ahead: spoken as a pole, with a distance, a bounded number of times
    said, l = _run(cam, lambda k, t: [("street light", 0.2, 7 - walk * t, 0.25, 3.5)], 130, noise=0.01)
    lat += l
    out["pole_ahead"] = said
    assert said and said[0][1].startswith("Pole, 12 o'clock, ") and "meter" in said[0][1]
    assert len(said) <= 2

    # 4. a person standing 3 m ahead (user stopped) steps out of the corridor:
    #    spoken once while in path, then silence
    def leaving(k, t):
        return [("person", 0.0 if t < 1.0 else min(2.0, 0.0 + 2.0 * (t - 1.0)), 3.0, 0.5, 1.7)]
    said, l = _run(cam, leaving, 150, noise=0.01)
    lat += l
    out["person_leaves_corridor"] = said
    assert len(said) == 1 and said[0][1] == "Person, 12 o'clock, three meters." and said[0][0] < 0.2

    # 5. the planner raises traffic lights at a crossing: spoken once; without the prior, silent
    tl = lambda k, t: [("traffic light", -5.0, 8.0, 0.4, 3.0)]
    pri = Situation("street_crossing", class_priors={"traffic light": 3.0}, source="offline_rules")
    said, l = _run(cam, tl, 60, situation=pri)
    lat += l
    assert said == [(0.07, "Traffic light, 11 o'clock.")], said
    out["traffic_light_with_prior"] = said
    assert _run(cam, tl, 60)[0] == []

    lat = np.array(lat) * 1e3
    out["policy_ms_per_frame"] = {"median": round(float(np.median(lat)), 3),
                                  "p99": round(float(np.percentile(lat, 99)), 3),
                                  "max": round(float(lat.max()), 3), "n_frames": len(lat)}
    for k, v in out.items():
        print(f"{k}: {v}")
    return out


if __name__ == "__main__":
    demo()
