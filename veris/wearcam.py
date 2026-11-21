"""Wearable-camera degradation model.

Open datasets are shot by sighted photographers holding still. A head-worn
camera is not. This turns a clean image into what a wearable would have
captured, so every eval can report a clean number and a wearable number.

Parameters come from the 2026-11-18 market research (sources in
docs/multi-model-evidence.md):

- Stream resolution: devices carry 5-13 MP sensors but hand 720p-1080p to the
  model (Ray-Ban Meta 1080p30 video, Glass EE2 / Envision 1080p30, Brilliant
  Frame 720x720). Default long side here is 1280.
- Field of view: 83 deg diagonal (Glass EE2) to about 100 deg (Ray-Ban Meta
  gen 2). Wide lenses bend straight lines, modelled as barrel distortion.
- Head motion while walking: vertical bob near 2 Hz, pitch velocity about
  30 deg/s typical and 90 deg/s worst case.
- Blur follows from that: smear_px = deg_per_s * exposure_s * px_per_deg.
  At 1280 px across 80 deg (16 px/deg), 30 deg/s and a 1/30 s indoor exposure
  smear 16 px; outdoors at 1/500 s the same motion smears 1 px.
- Fixed focus tuned near 0.6 m (Glass EE2), so close text is soft.

Exposure times and rolling-shutter readout are not published by any vendor.
The indoor 1/30 s and outdoor 1/500 s figures are engineering assumptions and
are the first thing to replace once the real camera is measured.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np


@dataclass(frozen=True)
class Profile:
    long_side: int = 1280
    hfov_deg: float = 80.0
    exposure_s: float = 1 / 30
    head_deg_s: float = 30.0        # angular speed during the exposure
    vertical_bias: float = 0.8      # share of motion that is pitch (bob) vs yaw
    barrel_k1: float = -0.18        # negative = barrel
    roll_deg: float = 4.0           # max head tilt either way
    defocus_sigma: float = 0.0      # px, for text closer than the focus distance
    light: float = 1.0              # 1 = well lit; 0.1 = dim room
    read_noise: float = 2.0         # sensor noise in 8-bit levels at light=1
    jpeg_q: int = 70


PROFILES = {
    # still user, daylight: the best a wearable gets
    "outdoor_still": Profile(exposure_s=1 / 500, head_deg_s=5, jpeg_q=80),
    "outdoor_walk": Profile(exposure_s=1 / 500, head_deg_s=30, jpeg_q=75),
    "indoor_still": Profile(exposure_s=1 / 30, head_deg_s=3, light=0.5),
    "indoor_walk": Profile(exposure_s=1 / 30, head_deg_s=30, light=0.5),
    # head turning while walking in a dim room
    "indoor_worst": Profile(exposure_s=1 / 15, head_deg_s=90, light=0.15,
                            roll_deg=8, jpeg_q=50),
    # document held closer than the 0.6 m focus distance
    "reading_close": Profile(exposure_s=1 / 30, head_deg_s=3, light=0.5,
                             defocus_sigma=1.8, barrel_k1=-0.10),
}


def smear_px(p: Profile) -> float:
    return p.head_deg_s * p.exposure_s * (p.long_side / p.hfov_deg)


def _motion_kernel(length: float, angle_deg: float) -> np.ndarray:
    n = max(3, int(np.ceil(length)) | 1)
    k = np.zeros((n, n), np.float32)
    c = n // 2
    dx, dy = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    for t in np.linspace(-length / 2, length / 2, n * 4):
        x, y = int(round(c + t * dx)), int(round(c + t * dy))
        if 0 <= x < n and 0 <= y < n:
            k[y, x] += 1
    return k / k.sum()


def degrade(img: np.ndarray, profile: str | Profile = "indoor_walk",
            seed: int = 0) -> np.ndarray:
    """BGR uint8 in, BGR uint8 out. Deterministic for a given seed."""
    p = PROFILES[profile] if isinstance(profile, str) else profile
    rng = np.random.default_rng(seed)

    h, w = img.shape[:2]
    s = p.long_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # lens: barrel distortion, then head roll
    if p.barrel_k1:
        f = 0.5 * w / np.tan(np.radians(p.hfov_deg / 2))
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
        dist = np.array([p.barrel_k1, 0, 0, 0], np.float64)
        # the clean image is the undistorted view; map it through the lens
        mx, my = cv2.initUndistortRectifyMap(K, -dist, None, K, (w, h), cv2.CV_32FC1)
        img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if p.roll_deg:
        roll = rng.uniform(-p.roll_deg, p.roll_deg)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), roll, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    out = img.astype(np.float32)

    if p.defocus_sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), p.defocus_sigma)

    px = smear_px(replace(p, long_side=max(h, w)))
    if px >= 1.5:
        vertical = rng.random() < p.vertical_bias
        angle = (90 if vertical else 0) + rng.uniform(-20, 20)
        out = cv2.filter2D(out, -1, _motion_kernel(px, angle))

    # light: fewer photons, then the camera's gain lifts signal and noise alike
    if p.light < 1 or p.read_noise:
        photons = np.clip(out, 0, 255) * p.light
        shot = rng.normal(0, 1, out.shape).astype(np.float32) * np.sqrt(photons + 1e-3)
        read = rng.normal(0, p.read_noise, out.shape).astype(np.float32)
        out = (photons + shot + read) / p.light

    out = np.clip(out, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, p.jpeg_q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian on a 320-px copy. Higher is sharper."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 320 / max(g.shape)
    if s < 1:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    # sensor noise is high-frequency too; smooth it out so a noisy dim frame
    # does not score as sharp
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def demo() -> None:
    # the arithmetic the docstring promises
    assert abs(smear_px(PROFILES["indoor_walk"]) - 16.0) < 0.1
    assert smear_px(PROFILES["outdoor_walk"]) < 1.1

    # a structured scene: blocks and edges, not noise
    yy, xx = np.mgrid[0:720, 0:1280]
    img = np.dstack([((xx // 80 + yy // 80) % 2 * 160 + 40).astype(np.uint8)] * 3).copy()
    cv2.putText(img, "EXIT", (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 16)
    a = degrade(img, "indoor_walk", seed=3)
    assert a.shape == img.shape and a.dtype == np.uint8
    assert np.array_equal(a, degrade(img, "indoor_walk", seed=3))      # deterministic
    # more motion and less light must cost sharpness, in order
    order = [sharpness(degrade(img, n, seed=3))
             for n in ("outdoor_still", "indoor_walk", "indoor_worst")]
    assert order[0] > order[1] > order[2], order
    print("smear px:", {n: round(smear_px(p), 1) for n, p in PROFILES.items()})
    print("sharpness:", [round(x) for x in order])


if __name__ == "__main__":
    demo()
