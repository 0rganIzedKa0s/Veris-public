"""Mode arbiter — fuses context signals into mode decisions + interrupts.

Design rules (see docs/sensor-fusion-switching.md for the evidence):
- Two channels. The REFLEX channel (ToF proximity, barge-in) emits interrupt
  events immediately and never waits for, or blocks, the VLM. The MODE
  channel decides what the next VLM call should be; it is consulted between
  generations, so switching adds zero latency to the loop.
- Slow signals propose, fast signals confirm. The cloud context engine
  proposes a mode (low rate, ~30-60s); local motion/depth/voice evidence
  gates when the proposal is believed.
- Hysteresis beats accuracy. A wrong-but-stable mode reads better to users
  than rapid flapping: candidate modes must persist for DWELL_S before a
  switch, and the current mode is held for at least MIN_RESIDENCE_S —
  except explicit user commands, which switch instantly (user intent
  outranks all inference).
- Anything time-sensitive the arbiter learns (stale head pose, hazard) is
  attached to the *next* frame's prompt, not injected mid-generation.

Hardware-free: signals arrive via update_*() calls, so the same arbiter runs
against real IMU/ToF drivers, the camera-motion proxy, or the simulator in
tests/test_arbiter.py.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

MODES = ("outdoor_nav", "indoor_nav", "text_reading", "describe")
OUTDOOR_SITUATIONS = ("street_sidewalk", "street_crossing", "transit_stop", "parking_lot")

DWELL_S = 4.0            # candidate must persist this long before we switch
MIN_RESIDENCE_S = 10.0   # don't leave a mode sooner than this (except commands)
HAZARD_NEAR_M = 1.2      # ToF closer than this while moving -> interrupt
STALE_YAW_DEG = 30.0     # head turned more than this since capture -> stale
GATE_TAU_S = 6.0         # leak time constant of the per-frame gate evidence
GATE_IN, GATE_OUT = 0.7, 0.3   # indoor evidence hysteresis band (hold inside)
GPS_W = 0.1              # planner's in/out vote, a nudge: a confident gate (~0.85)
                         # minus the nudge still clears GATE_IN, so vision outranks GPS
# Tuned on simulator dev seeds (evals/switching_eval.py operating_points_dev):
# a longer TAU or larger GPS_W trades doorway latency for fewer flaps.
TEXT_ON, TEXT_OFF = 0.75, 0.4  # text evidence band; text only counts when still
GATE_STALE_S = 5.0       # no gate frame for this long -> vision has no vote


class Motion(str, Enum):
    STATIONARY = "stationary"
    WALKING = "walking"
    TURNING = "turning"


@dataclass
class Interrupt:
    kind: str       # "hazard" | "barge_in"
    detail: str
    ts: float


@dataclass
class Arbiter:
    now: Callable[[], float]
    mode: str = "describe"
    motion: Motion = Motion.STATIONARY
    _mode_since: float = 0.0
    _candidate: str | None = None
    _candidate_since: float = 0.0
    _context_mode: str | None = None
    _context_situation: str = "unknown"
    _yaw_deg: float = 0.0
    _tof_m: float | None = None
    _gate_t: float | None = None
    _indoor_ev: float = 0.5
    _text_ev: float = 0.0
    _vision_indoor: bool | None = None
    _vision_text: bool = False
    interrupts: list[Interrupt] = field(default_factory=list)

    # ── signal inputs ────────────────────────────────────────────────
    def update_context(self, suggested_mode: str, situation: str):
        """Low-rate proposal from the cloud context engine."""
        if suggested_mode in MODES:
            self._context_mode = suggested_mode
            self._context_situation = situation
        self._evaluate()

    def update_motion(self, motion: Motion, yaw_deg: float | None = None):
        """High-rate: IMU walking-state + head yaw (or camera-motion proxy)."""
        self.motion = motion
        if yaw_deg is not None:
            self._yaw_deg = yaw_deg
        self._evaluate()

    def update_gate(self, indoor_p: float, text_p: float = 0.0):
        """Every frame: the Tier 0 gate's indoor/text votes. Accumulated as
        leaky evidence (time constant GATE_TAU_S) with a hysteresis band, so
        one wrong frame cannot move the mode; a decisive verdict outranks the
        planner's GPS-derived situation (GPS lags 10-35 s at doorways)."""
        t = self.now()
        dt = GATE_TAU_S if self._gate_t is None else max(t - self._gate_t, 0.02)
        a = 1.0 - math.exp(-dt / GATE_TAU_S)
        gps = 0.0 if self._context_situation == "unknown" else (
            -0.5 if self._context_situation in OUTDOOR_SITUATIONS else 0.5)
        self._indoor_ev += a * (indoor_p + GPS_W * gps - self._indoor_ev)
        self._text_ev += a * (text_p - self._text_ev)
        self._gate_t = t
        if self._indoor_ev > GATE_IN:
            self._vision_indoor = True
        elif self._indoor_ev < GATE_OUT:
            self._vision_indoor = False
        if self._text_ev > TEXT_ON:
            self._vision_text = True
        elif self._text_ev < TEXT_OFF:
            self._vision_text = False
        self._evaluate()

    def _vision(self) -> tuple[bool | None, bool]:
        """(indoor verdict or None, text verdict), only while gate frames are fresh."""
        if self._gate_t is None or self.now() - self._gate_t > GATE_STALE_S:
            return None, False
        return self._vision_indoor, self._vision_text

    def update_tof(self, nearest_m: float | None):
        """High-rate: forward ToF nearest obstacle. Reflex channel."""
        self._tof_m = nearest_m
        if (nearest_m is not None and nearest_m < HAZARD_NEAR_M
                and self.motion is Motion.WALKING):
            self.interrupts.append(Interrupt(
                "hazard", f"obstacle {nearest_m:.1f} m ahead", self.now()))

    def user_command(self, mode: str):
        """Explicit voice/button command: instant, no hysteresis."""
        if mode in MODES:
            self._set_mode(mode)

    def user_speaking(self):
        """VAD fired while TTS playing -> barge-in interrupt."""
        self.interrupts.append(Interrupt("barge_in", "user speech", self.now()))

    # ── decision logic ───────────────────────────────────────────────
    def _desired_mode(self) -> str:
        """Local motion evidence gates the cloud proposal."""
        ctx = self._context_mode or self.mode
        indoor, text = self._vision()
        # vision outranks GPS on in/out; the planner still names the venue
        if self.motion is Motion.WALKING:
            if indoor is not None:
                return "indoor_nav" if indoor else "outdoor_nav"
        elif text:
            return "text_reading"
        elif indoor:
            return ctx if ctx in ("describe", "text_reading") else "describe"
        elif indoor is False:
            return ctx if ctx in ("outdoor_nav", "text_reading") else "outdoor_nav"
        if self.motion is Motion.WALKING:
            # moving users need nav regardless of what the venue suggests;
            # indoor/outdoor comes from the situation
            if self._context_situation in OUTDOOR_SITUATIONS + ("unknown",):
                return "outdoor_nav"
            return "indoor_nav"
        # stationary: trust the context engine (describe/text_reading/indoor)
        return ctx

    def _evaluate(self):
        t = self.now()
        desired = self._desired_mode()
        if desired == self.mode:
            self._candidate = None
            return
        if self._candidate != desired:
            self._candidate, self._candidate_since = desired, t
            return
        residence_ok = (t - self._mode_since) >= MIN_RESIDENCE_S
        dwell_ok = (t - self._candidate_since) >= DWELL_S
        if residence_ok and dwell_ok:
            self._set_mode(desired)

    def _set_mode(self, mode: str):
        self.mode = mode
        self._mode_since = self.now()
        self._candidate = None

    # ── consumed by the pipeline between generations ─────────────────
    def frame_guard(self, yaw_at_capture_deg: float) -> bool:
        """False if the head has turned too far since this frame was taken —
        the pending description would mislead; drop or recapture."""
        return abs(self._yaw_deg - yaw_at_capture_deg) <= STALE_YAW_DEG

    def drain_interrupts(self) -> list[Interrupt]:
        out, self.interrupts = self.interrupts, []
        return out
