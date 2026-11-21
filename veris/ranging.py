"""Tier 1 geometry: range, bearing, closing time and walking corridor from
pixels alone. numpy only; no model, no depth network.

Camera frame: x right, y down, z forward along the optical axis. The camera
sits `height_m` above flat ground, pitched down by `pitch_deg` (positive =
looking down), no roll. World frame: same x, y straight down, z forward and
horizontal. Ranges are horizontal ground-plane distances.

Validated on KITTI in evals/nav_ranging.py. The flat-ground assumption is
the ceiling: far away, a degree of pitch error (a head nod, a slope) moves
the ground point by metres. See that result before speaking a far distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CORRIDOR_HALF_W = 0.4      # m, half a walking body
CORRIDOR_MARGIN = 0.2      # m, sway + arm swing


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    height_m: float = 1.6
    pitch_deg: float = 0.0

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float = 80.0,
                  height_m: float = 1.6, pitch_deg: float = 10.0) -> "Camera":
        f = 0.5 * width / math.tan(math.radians(hfov_deg) / 2)
        return cls(width, height, f, f, width / 2, height / 2, height_m, pitch_deg)


def _ray(cam: Camera, u, v):
    """Pixel -> world-frame ray (X right, Y down, Z forward-horizontal)."""
    rx = (np.asarray(u, float) - cam.cx) / cam.fx
    ry = (np.asarray(v, float) - cam.cy) / cam.fy
    s, c = math.sin(math.radians(cam.pitch_deg)), math.cos(math.radians(cam.pitch_deg))
    return rx, ry * c + s, -ry * s + c


def ground_point(cam: Camera, u, v):
    """Where the ray through pixel (u, v) meets the ground: (lateral X, forward Z)
    in metres, NaN where the ray is at or above the horizon."""
    X, Y, Z = _ray(cam, u, v)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(Y > 1e-6, cam.height_m / Y, np.nan)
    return X * t, Z * t


def range_m(cam: Camera, u, v_bottom):
    """Ground range to an object standing on the ground, from the column and
    the pixel row of its box bottom. NaN above the horizon."""
    X, Z = ground_point(cam, u, v_bottom)
    return np.hypot(X, Z)


def bearing_deg(cam: Camera, u, v=None):
    """Signed horizontal bearing, negative = left. Valid above the horizon too."""
    X, _, Z = _ray(cam, u, cam.cy if v is None else v)
    return np.degrees(np.arctan2(X, Z))


def in_path(cam: Camera, x1: float, x2: float, v_bottom: float,
            half_w: float = CORRIDOR_HALF_W, margin: float = CORRIDOR_MARGIN) -> bool:
    """Does the object's footprint (its box bottom edge projected to the ground)
    overlap the walking corridor straight ahead?"""
    (xl, xr), _ = ground_point(cam, np.array([x1, x2]), v_bottom)
    if not np.isfinite(xl):
        return False
    lim = half_w + margin
    return bool(xr >= -lim and xl <= lim)


def ttc_s(times, scales, min_span_s: float = 0.3, max_s: float = 20.0) -> float | None:
    """Time to contact from looming: tau = s / (ds/dt) = 1 / (d ln s / dt).

    The slope of ln(scale) is a Theil-Sen estimate (median of all pairwise
    slopes), so one jittery box moves it little and cannot fire it alone.
    None when the window is too short, the object is not growing, or tau is
    beyond `max_s`."""
    t = np.asarray(times, float)
    ls = np.log(np.maximum(np.asarray(scales, float), 1e-6))
    if len(t) < 3 or t[-1] - t[0] < min_span_s:
        return None
    i, j = np.triu_indices(len(t), 1)
    dt = t[j] - t[i]
    ok = dt > 1e-6
    rate = float(np.median((ls[j] - ls[i])[ok] / dt[ok]))
    # the median slope is the rate near mid-window; at constant closing speed
    # tau falls one second per second, so carry it forward to the last frame
    tau = 1.0 / rate - (t[-1] - float(np.median(t))) if rate > 0 else math.inf
    return tau if 0 < tau <= max_s else None


def demo() -> None:
    # level camera 1.6 m up: an object 8 m ahead has its foot at row cy + f*1.6/8
    cam = Camera(1280, 720, 800, 800, 640, 360, 1.6, 0.0)
    assert abs(range_m(cam, 640, 360 + 800 * 1.6 / 8) - 8.0) < 1e-9
    assert np.isnan(range_m(cam, 640, 300))                     # above horizon
    assert abs(bearing_deg(cam, 640 + 800)) - 45 < 1e-9 and bearing_deg(cam, 0) < 0

    # pitched camera: build the pixel of a known ground point, recover it
    cam = Camera.from_hfov(1280, 720, 80, 1.6, 15.0)
    X, Z = 1.0, 5.0
    th = math.radians(15)
    xc, yc, zc = X, 1.6 * math.cos(th) - Z * math.sin(th), 1.6 * math.sin(th) + Z * math.cos(th)
    u, v = cam.cx + cam.fx * xc / zc, cam.cy + cam.fy * yc / zc
    gx, gz = ground_point(cam, u, v)
    assert abs(gx - X) < 1e-9 and abs(gz - Z) < 1e-9
    assert abs(bearing_deg(cam, u, v) - math.degrees(math.atan2(X, Z))) < 1e-9

    # corridor: a 30 cm pole 0.5 m to the side is in it, 1.5 m to the side is not
    def box_at(xc_m, z_m, w_m=0.3):
        us = []
        for x in (xc_m - w_m / 2, xc_m + w_m / 2):
            yc_, zc_ = 1.6 * math.cos(th) - z_m * math.sin(th), 1.6 * math.sin(th) + z_m * math.cos(th)
            us.append(cam.cx + cam.fx * x / zc_)
            vb = cam.cy + cam.fy * yc_ / zc_
        return us[0], us[1], vb
    assert in_path(cam, *box_at(0.5, 4.0)) and not in_path(cam, *box_at(1.5, 4.0))

    # looming: an object closing at 1.5 m/s from 6 m, 10 fps -> tau = Z / 1.5
    t = np.arange(8) / 10
    z = 6 - 1.5 * t
    s = 200 / z
    assert abs(ttc_s(t, s) - z[-1] / 1.5) < 0.15
    jit = s.copy(); jit[-1] *= 1.15                  # one jittery frame on a static object
    assert ttc_s(t, np.full(8, 50.0)) is None
    assert ttc_s(t, np.r_[np.full(7, 50.0), 50 * 1.15]) is None
    assert ttc_s(t[:2], s[:2]) is None
    print("ranging ok; tau", round(ttc_s(t, s), 2), "true", round(z[-1] / 1.5, 2),
          "| jittered", round(ttc_s(t, jit), 2))


if __name__ == "__main__":
    demo()
