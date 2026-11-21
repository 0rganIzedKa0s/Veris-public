"""Headless two-channel orchestration loop for the pack.

Ties the mode arbiter (veris/arbiter.py), the camera-motion proxy
(veris/motion_proxy.py), a cloud context client and a VLM describe client into
the loop specified in docs/sensor-fusion-switching.md. The loop itself owns no
policy — thresholds and mode logic live in the arbiter — it only sequences the
two channels around the (slow) VLM generation:

- Two channels. The REFLEX channel (ToF hazard, barge-in) queues interrupt
  events the instant a sensor fires and never blocks the VLM. The MODE channel
  is consulted only *between* generations, so a switch costs zero latency. Both
  are read at generation boundaries.
- A generation describes exactly one frame and runs for a duration reported by
  the describe client. Sensor events that fall inside that window are the
  "mid-generation" async traffic: applied to the arbiter as they arrive (so
  motion/yaw/context keep tracking and interrupts queue) but only *acted on* —
  interrupts emitted, next mode re-read — at the following boundary.
- Stale-frame guard: if head yaw has moved past the arbiter's threshold since
  the described frame was captured, the finished description is dropped instead
  of spoken; it would mislead.
- Reflex outranks describe (subsumption): any interrupt drained at a boundary
  preempts the just-finished description — higher layers suppress lower ones.

Everything is injected, so the same loop runs on-device (real ToF/IMU/VLM
drivers) or under tests/replay (stubs + synthetic frames). Nothing here imports
torch, a camera, or a network client; those arrive as the four collaborators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .arbiter import Arbiter, Interrupt
from .motion_proxy import MotionProxy


# ── injected collaborators ───────────────────────────────────────────
@dataclass(frozen=True)
class DescribeResult:
    text: str
    duration_s: float          # wall-clock the VLM will take for this frame


class DescribeClient(Protocol):
    """The VLM. describe() returns the finished text plus how long it took,
    so the loop knows the generation window without owning a real clock."""

    def describe(self, mode: str, frame_path: str, context: str | None) -> DescribeResult: ...


class ContextClient(Protocol):
    """The cloud/phone context engine. poll() maps a GPS/venue fix to a
    (suggested_mode, situation) proposal, or None if it has nothing to say."""

    def poll(self, fix: Any) -> tuple[str, str] | None: ...


class Clock:
    """Settable virtual clock shared with the arbiter (arbiter.now = clock)."""

    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def set(self, t: float) -> None:
        self.t = t


# ── loop output trace ────────────────────────────────────────────────
@dataclass(frozen=True)
class Emit:
    """One observable thing the loop did, in order. kind is one of:
    gen_start | speak | interrupt | drop. payload carries the detail; for
    gen_start it is the mode used."""

    t: float
    kind: str
    payload: str


class PackLoop:
    def __init__(
        self,
        arbiter: Arbiter,
        motion_proxy: MotionProxy,
        context_client: ContextClient,
        describe_client: DescribeClient,
    ):
        self.arb = arbiter
        self.motion = motion_proxy
        self.ctx = context_client
        self.describe = describe_client

        self.trace: list[Emit] = []
        self._situation: str | None = None
        self._latest: tuple[str, float] | None = None   # (frame_path, yaw_at_capture)
        self._gen: dict | None = None                    # in-flight generation

    # ── async sensor inputs (may arrive at any time, incl. mid-generation) ──
    def on_context(self, fix: Any) -> None:
        prop = self.ctx.poll(fix)
        if prop is not None:
            self.arb.update_context(*prop)
            self._situation = prop[1]

    def on_tof(self, nearest_m: float | None) -> None:
        self.arb.update_tof(nearest_m)            # reflex: may queue a hazard

    def on_barge_in(self) -> None:
        self.arb.user_speaking()                  # reflex: queues a barge_in

    def on_user_command(self, mode: str) -> None:
        self.arb.user_command(mode)               # instant, outranks inference

    def on_frame(self, frame_path: str) -> None:
        """A new camera frame is available: feed the motion proxy and record it
        as the latest describable frame. Does NOT itself start a generation."""
        est = self.motion.process(frame_path)
        if est is not None:
            self.arb.update_motion(est.motion, est.yaw_deg)
            self._latest = (frame_path, est.yaw_deg)
        else:
            # first frame of a sequence: no pair yet, yaw reference is 0
            self._latest = (frame_path, 0.0)

    # ── replay / on-device driver ────────────────────────────────────
    def run(self, events: list[dict], clock: Clock) -> list[Emit]:
        """Drive the loop through a time-ordered event stream. Each event is a
        dict with a float ``t`` and a ``type`` in
        {frame, context, tof, barge_in, user_command}. Returns the trace."""
        for ev in sorted(events, key=lambda e: e["t"]):
            self._advance_to(ev["t"], clock)      # close generations finished before now
            clock.set(ev["t"])
            self._apply(ev, clock)
        # flush the final in-flight generation so its output is resolved
        if self._gen is not None:
            clock.set(self._gen["end"])
            self._close_generation(clock)
        return self.trace

    def _apply(self, ev: dict, clock: Clock) -> None:
        typ = ev["type"]
        if typ == "frame":
            self.on_frame(ev["path"])
            if self._gen is None:                 # loop was idle -> begin work
                self._start_generation(clock)
        elif typ == "context":
            self.on_context(ev["fix"])
        elif typ == "tof":
            self.on_tof(ev["nearest_m"])
        elif typ == "barge_in":
            self.on_barge_in()
        elif typ == "user_command":
            self.on_user_command(ev["mode"])
        else:
            raise ValueError(f"unknown event type: {typ!r}")

    # ── generation lifecycle ─────────────────────────────────────────
    def _advance_to(self, t: float, clock: Clock) -> None:
        """Close (and chain) every generation whose window ends at or before t."""
        while self._gen is not None and self._gen["end"] <= t:
            end = self._gen["end"]
            clock.set(end)
            self._close_generation(clock)
            self._start_generation(clock)         # next gen begins at the boundary
            if self._gen is not None and self._gen["end"] <= end:
                break                             # defensive: never spin on 0-length

    def _start_generation(self, clock: Clock) -> None:
        """Read the mode between generations and kick off the next describe."""
        if self._latest is None:
            return
        frame_path, yaw_capture = self._latest
        mode = self.arb.mode                      # <- the only place mode is read
        result = self.describe.describe(mode, frame_path, self._situation)
        start = clock()
        self._gen = {
            "frame": frame_path,
            "mode": mode,
            "yaw_capture": yaw_capture,
            "result": result,
            "end": start + max(result.duration_s, 1e-9),
        }
        self.trace.append(Emit(start, "gen_start", mode))

    def _close_generation(self, clock: Clock) -> None:
        """At the boundary: drain the reflex channel, then decide the pending
        description's fate (stale -> drop, preempted -> drop, else speak)."""
        gen, self._gen = self._gen, None
        if gen is None:
            return
        t = clock()
        interrupts: list[Interrupt] = self.arb.drain_interrupts()
        for it in interrupts:
            self.trace.append(Emit(t, "interrupt", f"{it.kind}:{it.detail}"))

        if not self.arb.frame_guard(gen["yaw_capture"]):
            self.trace.append(Emit(t, "drop", f"stale:{gen['frame']}"))
            return
        if interrupts:                            # reflex suppresses the describe output
            self.trace.append(Emit(t, "drop", f"preempted:{gen['frame']}"))
            return
        self.trace.append(Emit(t, "speak", gen["result"].text))
