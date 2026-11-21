"""Replay-scenario assertions for veris/pack_loop.py.

  uv run --with pytest --with numpy --with pillow python -m pytest tests/replay -q

Each test runs a JSONL scenario end to end through the real PackLoop + Arbiter
+ MotionProxy (only the context/describe clients are stubbed) and asserts the
mode and interrupt sequences the two-channel loop must produce.
"""

from tests.replay.harness import kinds, modes, run_scenario


def test_street_to_shop_switches_only_between_generations(tmp_path):
    """Walking the whole time; context flips street->shop at t=12. The mode
    rides out residence+dwell and switches outdoor_nav->indoor_nav at a
    generation boundary, never mid-generation. No reflexes fire."""
    trace, loop, describe = run_scenario("street_to_shop.jsonl", tmp_path)
    seq = modes(trace)

    # starts and stays outdoor_nav, ends indoor_nav
    assert seq[0] == "outdoor_nav"
    assert seq[-1] == "indoor_nav"
    # exactly one switch, and it goes outdoor -> indoor
    switches = [(a, b) for a, b in zip(seq, seq[1:]) if a != b]
    assert switches == [("outdoor_nav", "indoor_nav")]

    # the switch lands only after MIN_RESIDENCE_S(10)+DWELL_S(4) past the start
    first_indoor = next(e for e in trace if e.kind == "gen_start" and e.payload == "indoor_nav")
    assert first_indoor.t >= 14.0
    # every gen_start sits on a generation boundary (multiple of the 2.0s window)
    assert all(e.t % 2.0 == 0.0 for e in trace if e.kind == "gen_start")

    assert kinds(trace, "interrupt") == []
    assert kinds(trace, "drop") == []
    # the VLM was actually asked to describe in both modes
    assert {c[0] for c in describe.calls} == {"outdoor_nav", "indoor_nav"}


def test_static_room_stays_describe_and_speaks(tmp_path):
    """Stationary in a room: mode is describe throughout and every generation
    is spoken — nothing stale, no interrupts, no drops."""
    trace, loop, describe = run_scenario("static_room.jsonl", tmp_path)

    assert set(modes(trace)) == {"describe"}
    assert kinds(trace, "interrupt") == []
    assert kinds(trace, "drop") == []
    spoken = [e for e in trace if e.kind == "speak"]
    assert len(spoken) >= 3
    assert all(t.startswith("[describe]") for t in kinds(trace, "speak"))


def test_hazard_midgen_preempts_pending_description(tmp_path):
    """A ToF hazard fires while a generation is in flight. At the next boundary
    the reflex is emitted and it preempts (drops) the just-finished
    description; the loop then recovers and speaks the following generation."""
    trace, loop, describe = run_scenario("hazard_midgen.jsonl", tmp_path)

    ints = kinds(trace, "interrupt")
    assert len(ints) == 1 and ints[0].startswith("hazard:")

    drops = kinds(trace, "drop")
    assert len(drops) == 1 and drops[0].startswith("preempted:")

    # the interrupt and its preemption drop happen at the same boundary...
    hz = next(e for e in trace if e.kind == "interrupt")
    dr = next(e for e in trace if e.kind == "drop")
    assert hz.t == dr.t
    # ...and the description that was dropped was NOT spoken, but a later one is
    kinds_in_order = [e.kind for e in trace]
    assert kinds_in_order.index("interrupt") < kinds_in_order.index("speak")
    assert len([e for e in trace if e.kind == "speak"]) >= 1
    assert modes(trace) == ["outdoor_nav", "outdoor_nav", "outdoor_nav"]
