"""Simulated-sensor scenarios for the mode arbiter.

  uv run python -m pytest tests/test_arbiter.py -q
"""

from veris.arbiter import Arbiter, Motion, DWELL_S, MIN_RESIDENCE_S


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def advance(self, s): self.t += s


def make():
    c = Clock()
    a = Arbiter(now=c)
    a._mode_since = c.t
    return a, c


def test_walk_to_restaurant_switches_with_dwell():
    """Walking down the street, context flips to restaurant; the switch should
    happen only after residence + dwell, and land on indoor_nav (still moving)."""
    a, c = make()
    a.user_command("outdoor_nav")
    a.update_motion(Motion.WALKING)
    c.advance(MIN_RESIDENCE_S + 1)
    a.update_context("indoor_nav", "restaurant")
    assert a.mode == "outdoor_nav"          # proposal alone doesn't switch
    c.advance(DWELL_S / 2); a.update_motion(Motion.WALKING)
    assert a.mode == "outdoor_nav"          # dwell not reached
    c.advance(DWELL_S); a.update_motion(Motion.WALKING)
    assert a.mode == "indoor_nav"           # stable proposal, moving indoors


def test_gps_flapping_does_not_flap_mode():
    """Context engine oscillates (GPS jitter at a doorway): mode must not."""
    a, c = make()
    a.user_command("outdoor_nav")
    a.update_motion(Motion.WALKING)
    c.advance(MIN_RESIDENCE_S + 1)
    for _ in range(6):                       # alternating proposals, 1s apart
        a.update_context("indoor_nav", "restaurant"); c.advance(1)
        a.update_context("outdoor_nav", "street_sidewalk"); c.advance(1)
    assert a.mode == "outdoor_nav"           # candidate never dwelled long enough


def test_user_command_is_instant():
    a, c = make()
    a.user_command("outdoor_nav")
    a.user_command("text_reading")           # zero dwell, zero residence
    assert a.mode == "text_reading"


def test_walking_overrides_stationary_modes():
    """Sitting in a restaurant (describe), then standing up and walking:
    mode must migrate to indoor_nav despite the venue context."""
    a, c = make()
    a.user_command("describe")
    a.update_context("describe", "restaurant")
    c.advance(MIN_RESIDENCE_S + 1)
    a.update_motion(Motion.WALKING)
    c.advance(DWELL_S + 0.1)
    a.update_motion(Motion.WALKING)
    assert a.mode == "indoor_nav"


def test_hazard_interrupt_fires_only_while_moving():
    a, c = make()
    a.update_motion(Motion.STATIONARY)
    a.update_tof(0.8)
    assert a.drain_interrupts() == []        # standing near a wall is fine
    a.update_motion(Motion.WALKING)
    a.update_tof(0.8)
    ints = a.drain_interrupts()
    assert len(ints) == 1 and ints[0].kind == "hazard"


def test_stale_frame_guard():
    a, c = make()
    a.update_motion(Motion.TURNING, yaw_deg=0.0)
    assert a.frame_guard(yaw_at_capture_deg=0.0)
    a.update_motion(Motion.TURNING, yaw_deg=45.0)
    assert not a.frame_guard(yaw_at_capture_deg=0.0)   # head moved 45°


def test_barge_in():
    a, c = make()
    a.user_speaking()
    ints = a.drain_interrupts()
    assert ints and ints[0].kind == "barge_in"
