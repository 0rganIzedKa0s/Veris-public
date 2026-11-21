"""Unit tests for the always-on front door (ops/guard/veris_guard.py).

This process decides where a resume link lands, so its two failure modes both
matter: sending visitors to a board that is gone, and sending them to the
recorded copy while the live demo is up and working.

  uv run python -m pytest tests/test_guard.py -q
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "veris_guard", ROOT / "ops" / "guard" / "veris_guard.py")
guard = importlib.util.module_from_spec(_spec)
sys.modules["veris_guard"] = guard
_spec.loader.exec_module(guard)


def watch(start_up, probes):
    """Drive a BoardWatch through a fixed probe sequence, one loop body per item."""
    w = guard.BoardWatch()
    w.up = start_up
    seen_states = []
    for seen in probes:
        w._streak = w._streak + 1 if seen == w._last else 1
        w._last = seen
        if w._streak >= guard.FLAPS and seen != w.up:
            w.up = seen
        seen_states.append(w.up)
    return seen_states


def test_single_dropped_probe_does_not_bounce_a_live_visitor():
    """One timeout on a working board must not send anyone to the recorded copy."""
    assert watch(True, [False, True, True]) == [True, True, True]


def test_sustained_outage_switches_over():
    assert watch(True, [False, False]) == [True, False]


def test_one_lucky_response_from_a_rebooting_board_does_not_switch_back():
    """llama-server restarts mid-reboot; a single OK must not route visitors
    into a board that is about to stop answering again."""
    assert watch(False, [True, False, True, False]) == [False, False, False, False]


def test_recovery_switches_back():
    assert watch(False, [True, True]) == [False, True]


def test_starts_pessimistic():
    """Before the first probe resolves, the recorded copy is the safe answer:
    it always works, whereas the live URL may not."""
    assert guard.BoardWatch().up is False


def test_target_follows_state():
    w = guard.BoardWatch()
    w.up = True
    assert w.target() == guard.LIVE_URL
    w.up = False
    assert w.target() == guard.RECORDED_URL


def test_probe_rejects_a_non_veris_json_body(monkeypatch):
    """Some captive portals and parked domains answer 200 with JSON. Requiring
    the hub's own shape stops those from reading as a healthy board."""
    class R:
        status = 200
        def read(self): return b'{"hello": "world"}'
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(guard.urllib.request, "urlopen", lambda *a, **k: R())
    assert guard.BoardWatch().probe() is False


def test_probe_accepts_the_hubs_status_shape(monkeypatch):
    class R:
        status = 200
        def read(self): return b'{"orin": {"status": "unreachable: x"}}'
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(guard.urllib.request, "urlopen", lambda *a, **k: R())
    # the hub answering is enough: that page replays recordings locally even
    # when llama-server itself is down
    assert guard.BoardWatch().probe() is True


def test_probe_survives_a_dead_host(monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(guard.urllib.request, "urlopen", boom)
    assert guard.BoardWatch().probe() is False
