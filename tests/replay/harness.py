"""Deterministic replay harness for veris/pack_loop.py.

A scenario is a JSONL file: an optional first ``config`` line followed by
time-ordered sensor events. The harness materialises synthetic *frame* specs
into real PNG files on disk (so the loop is handed genuine frame paths and the
real MotionProxy runs), wires up stub context/describe clients, and drives
PackLoop.run to completion, returning the emitted trace.

Frames are described by a texture name + offset into a fixed, non-periodic
1-D profile (same construction as tests/test_motion_proxy.py). This makes the
resulting Motion state deterministic:

  - repeat the same (tex, off)      -> identical frames   -> STATIONARY
  - alternate tex T <-> U, same off -> decorrelated       -> WALKING
  - same tex, off shifted by N px   -> coherent shift      -> TURNING

Event schema (one JSON object per line):
  {"type": "config", "describe_duration_s": 2.0,
   "context_map": {"street": ["outdoor_nav", "street_sidewalk"], ...}}
  {"t": 0.0, "type": "frame", "tex": "T", "off": 20}
  {"t": 1.0, "type": "context", "fix": "street"}
  {"t": 1.5, "type": "tof", "nearest_m": 0.8}
  {"t": 2.0, "type": "barge_in"}
  {"t": 2.0, "type": "user_command", "mode": "text_reading"}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from veris.arbiter import Arbiter
from veris.motion_proxy import MotionProxy
from veris.pack_loop import Clock, DescribeResult, PackLoop

SCENARIOS = Path(__file__).parent / "scenarios"

# Deterministic, non-periodic textures (mirror tests/test_motion_proxy.py).
TEX = {
    "T": np.array([(i * 53) % 197 for i in range(256)], dtype=np.uint8),
    "U": np.array([(i * 91 + 17) % 211 for i in range(256)], dtype=np.uint8),
}
_FRAME_WIDTH = 96
_FRAME_HEIGHT = 40


class ContextStub:
    """Cloud context engine stub: a fixed fix-label -> proposal lookup."""

    def __init__(self, context_map: dict[str, tuple[str, str]]):
        self._map = context_map

    def poll(self, fix):
        prop = self._map.get(fix)
        return tuple(prop) if prop is not None else None


class DescribeStub:
    """VLM stub: canned text tagged with the mode/frame, fixed duration.
    Records every call so tests can inspect what the VLM was asked to do."""

    def __init__(self, duration_s: float):
        self.duration_s = duration_s
        self.calls: list[tuple[str, str, str | None]] = []

    def describe(self, mode, frame_path, context) -> DescribeResult:
        self.calls.append((mode, frame_path, context))
        name = Path(frame_path).stem
        return DescribeResult(text=f"[{mode}] {name}", duration_s=self.duration_s)


def _materialise_frame(ev: dict, out_dir: Path, index: int) -> str:
    """Render a frame spec to a PNG and return its path."""
    profile = TEX[ev["tex"]][ev["off"]: ev["off"] + _FRAME_WIDTH]
    arr = np.tile(profile, (_FRAME_HEIGHT, 1))
    path = out_dir / f"frame_{index:03d}.png"
    Image.fromarray(arr, mode="L").save(path)
    return str(path)


def load_scenario(name: str, frame_dir: Path):
    """Parse a scenario file, materialising its frames under frame_dir.

    Returns (events, clock, loop, describe_stub) ready to run."""
    path = SCENARIOS / name
    config: dict = {}
    events: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        obj = json.loads(line)
        if obj.get("type") == "config":
            config = obj
        else:
            events.append(obj)

    frame_dir.mkdir(parents=True, exist_ok=True)
    for i, ev in enumerate(events):
        if ev["type"] == "frame":
            ev["path"] = _materialise_frame(ev, frame_dir, i)

    clock = Clock(0.0)
    arbiter = Arbiter(now=clock)
    arbiter._mode_since = clock()          # anchor residence to the scenario start
    context_map = config.get("context_map", {})
    describe = DescribeStub(config.get("describe_duration_s", 2.0))
    loop = PackLoop(
        arbiter=arbiter,
        motion_proxy=MotionProxy(),
        context_client=ContextStub(context_map),
        describe_client=describe,
    )
    return events, clock, loop, describe


def run_scenario(name: str, frame_dir: Path):
    """Load and run a scenario; return (trace, loop, describe_stub)."""
    events, clock, loop, describe = load_scenario(name, frame_dir)
    trace = loop.run(events, clock)
    return trace, loop, describe


def modes(trace) -> list[str]:
    return [e.payload for e in trace if e.kind == "gen_start"]


def kinds(trace, kind) -> list[str]:
    return [e.payload for e in trace if e.kind == kind]
