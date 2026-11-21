"""Tests for veris/motion_proxy.py — camera-motion proxy feeding the arbiter.

  uv run --with pytest --with numpy --with pillow python -m pytest tests/test_motion_proxy.py -q
"""

import numpy as np
import pytest
from PIL import Image

from veris.arbiter import Motion
from veris.motion_proxy import MotionProxy


def img_from_columns(profile, height=40):
    """Grayscale image whose column-mean profile is exactly `profile` (every
    row identical), so _coherent_shift's column-mean comparison is driven
    precisely by the values under test."""
    arr = np.tile(np.asarray(profile, dtype=np.uint8), (height, 1))
    return Image.fromarray(arr, mode="L")


# Deterministic, non-periodic textures (no randomness -> no flaky tests).
T = np.array([(i * 53) % 197 for i in range(200)], dtype=np.float64)
U = np.array([(i * 91 + 17) % 211 for i in range(200)], dtype=np.float64)  # decorrelated from T


def test_first_frame_returns_none():
    p = MotionProxy()
    assert p.process(img_from_columns(T[20:116])) is None


def test_identical_frames_are_stationary():
    p = MotionProxy()
    frame = img_from_columns(T[20:116])
    p.process(frame)
    est = p.process(frame)
    assert est.motion is Motion.STATIONARY
    assert est.mad == 0.0
    assert est.shift_px == 0


def test_pan_right_is_turning_with_positive_yaw():
    """Content moves left in-frame -> pan right -> positive yaw (module
    docstring's sign convention)."""
    p = MotionProxy()
    p.process(img_from_columns(T[20:116]))
    est = p.process(img_from_columns(T[25:121]))  # content shifted left 5px
    assert est.motion is Motion.TURNING
    assert est.shift_px == -5
    assert est.yaw_delta_deg > 0
    assert est.yaw_deg == est.yaw_delta_deg


def test_pan_left_is_negative_yaw():
    p = MotionProxy()
    p.process(img_from_columns(T[25:121]))
    est = p.process(img_from_columns(T[20:116]))  # content shifted right 5px
    assert est.motion is Motion.TURNING
    assert est.shift_px == 5
    assert est.yaw_delta_deg < 0


def test_decorrelated_change_without_shift_is_walking_not_turning():
    """Frame-to-frame change with no coherent global shift (walking bob,
    parallax) must not synthesize a fake turn."""
    p = MotionProxy()
    p.process(img_from_columns(T[20:116]))
    est = p.process(img_from_columns(U[20:116]))
    assert est.shift_px == 0
    assert est.motion is Motion.WALKING


def test_featureless_frame_never_synthesizes_shift():
    """A blank/uniform frame has no evidence for a shift (std < 1), even
    across an exposure change between frames."""
    p = MotionProxy()
    p.process(img_from_columns([100] * 96))
    est = p.process(img_from_columns([150] * 96))
    assert est.shift_px == 0
    assert est.mad == 50.0
    assert est.motion is Motion.WALKING  # exposure swing reads as motion (documented limit)


def test_resolution_change_returns_none_and_resets_pairing():
    p = MotionProxy()
    p.process(Image.new("L", (200, 150), 128))
    assert p.process(Image.new("L", (200, 100), 128)) is None  # different aspect -> different working shape
    est = p.process(Image.new("L", (200, 100), 128))  # same aspect as previous -> pairs again
    assert est is not None
    assert est.motion is Motion.STATIONARY


def test_reset_clears_pairing_and_yaw_accumulator():
    p = MotionProxy()
    p.process(img_from_columns(T[20:116]))
    p.process(img_from_columns(T[25:121]))
    assert p.yaw_deg != 0.0
    p.reset()
    assert p.yaw_deg == 0.0
    assert p.process(img_from_columns(T[20:116])) is None  # forgot the previous frame


def test_yaw_accumulates_across_frames():
    p = MotionProxy()
    p.process(img_from_columns(T[20:116]))
    p.process(img_from_columns(T[25:121]))  # +1 turn, shift of 5
    first_yaw = p.yaw_deg
    est = p.process(img_from_columns(T[30:126]))  # another turn, shift of 5
    assert p.yaw_deg == pytest.approx(first_yaw + est.yaw_delta_deg)


def test_elapsed_ms_is_nonnegative():
    p = MotionProxy()
    p.process(img_from_columns(T[20:116]))
    est = p.process(img_from_columns(T[20:116]))
    assert est.elapsed_ms >= 0.0


def test_accepts_file_path(tmp_path):
    frame = img_from_columns(T[20:116])
    path = tmp_path / "frame.jpg"
    frame.convert("RGB").save(path)
    p = MotionProxy()
    p.process(str(path))
    est = p.process(str(path))
    assert est.motion is Motion.STATIONARY
