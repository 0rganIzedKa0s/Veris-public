"""Unit tests for veris/pack_loop.py control logic, with lightweight fakes so
the two paths the replay scenarios don't hit are covered directly: the
stale-frame drop, and the "mode is read only between generations" invariant.

  uv run --with pytest python -m pytest tests/replay/test_pack_loop.py -q
"""

from veris.arbiter import Arbiter, Motion
from veris.motion_proxy import MotionEstimate
from veris.pack_loop import Clock, DescribeResult, PackLoop


class FakeMotion:
    """Returns scripted MotionEstimates (or None) instead of decoding images."""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0

    def process(self, path):
        est = self.script[self.i]
        self.i += 1
        return est


class FakeDescribe:
    def __init__(self, duration_s=2.0):
        self.duration_s = duration_s

    def describe(self, mode, frame_path, context):
        return DescribeResult(text=f"[{mode}]", duration_s=self.duration_s)


class NoContext:
    def poll(self, fix):
        return None


def _est(motion, yaw):
    return MotionEstimate(motion, yaw, yaw, mad=5.0, shift_px=0, elapsed_ms=0.0)


def _loop(motion):
    clock = Clock(0.0)
    arb = Arbiter(now=clock)
    arb._mode_since = clock()
    loop = PackLoop(arb, motion, NoContext(), FakeDescribe())
    return loop, clock, arb


def test_stale_frame_guard_drops_the_pending_description():
    """A frame captured at yaw~0; the head turns >STALE_YAW_DEG(30) during the
    generation; at the boundary the finished description is dropped, not spoken."""
    motion = FakeMotion([None, _est(Motion.TURNING, 40.0)])
    loop, clock, arb = _loop(motion)
    events = [
        {"t": 0.0, "type": "user_command", "mode": "describe"},
        {"t": 0.0, "type": "frame", "path": "a.png"},   # captured at yaw 0
        {"t": 0.5, "type": "frame", "path": "b.png"},   # turned to yaw 40 mid-gen
    ]
    trace = loop.run(events, clock)
    drops = [e.payload for e in trace if e.kind == "drop"]
    assert drops == ["stale:a.png"]
    assert not any(e.kind == "speak" for e in trace)


def test_mode_is_read_only_between_generations():
    """A user command mid-generation does not retag the in-flight generation;
    it takes effect for the next one."""
    motion = FakeMotion([None, _est(Motion.STATIONARY, 0.0), _est(Motion.STATIONARY, 0.0)])
    loop, clock, arb = _loop(motion)
    events = [
        {"t": 0.0, "type": "user_command", "mode": "describe"},
        {"t": 0.0, "type": "frame", "path": "a.png"},        # gen starts: describe
        {"t": 1.0, "type": "user_command", "mode": "outdoor_nav"},  # mid-gen switch
        {"t": 2.5, "type": "frame", "path": "b.png"},
    ]
    trace = loop.run(events, clock)
    starts = [e.payload for e in trace if e.kind == "gen_start"]
    assert starts[0] == "describe"          # in-flight gen keeps its snapshotted mode
    assert "outdoor_nav" in starts[1:]      # command applies to a later generation
    # the first generation still spoke as describe (mode snapshot honoured)
    assert trace[trace.index(next(e for e in trace if e.kind == "speak"))].payload == "[describe]"
