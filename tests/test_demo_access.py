"""Unit tests for the demo hub's public-access protections.

The demo URL is printed on a resume, so it is ungated on purpose. What stands
between an open URL and a 15 W board is this file's subject: per-IP + global
rate limits, single-occupancy of the model slot, and a cache in front of the
endpoints every open tab polls.

  uv run python -m pytest tests/test_demo_access.py -q
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from demo import app as demo  # noqa: E402


class FakeRequest:
    """Just enough of starlette's Request for _client_key."""

    class _Client:
        def __init__(self, host):
            self.host = host

    def __init__(self, peer="1.2.3.4", forwarded=None):
        self.client = self._Client(peer) if peer else None
        self.headers = {"x-forwarded-for": forwarded} if forwarded else {}


def setup_function():
    demo._hits.clear()
    demo._release_slot()
    demo._warmed_instance = None


# --- client identity -------------------------------------------------------

def test_direct_peer_is_the_key():
    assert demo._client_key(FakeRequest(peer="9.9.9.9")) == "9.9.9.9"


def test_forwarded_for_trusted_only_from_loopback():
    """Behind the Tailscale funnel every peer is 127.0.0.1, so XFF is the only
    thing separating visitors — but from a non-loopback peer it is spoofable
    and must be ignored."""
    via_funnel = FakeRequest(peer="127.0.0.1", forwarded="203.0.113.7, 10.0.0.1")
    assert demo._client_key(via_funnel) == "203.0.113.7"

    spoofed = FakeRequest(peer="8.8.8.8", forwarded="203.0.113.7")
    assert demo._client_key(spoofed) == "8.8.8.8"


def test_client_key_is_length_capped():
    junk = FakeRequest(peer="127.0.0.1", forwarded="x" * 500)
    assert len(demo._client_key(junk)) <= 64


def test_missing_client_does_not_crash():
    assert demo._client_key(FakeRequest(peer=None)) == "unknown"


# --- rate limiting ---------------------------------------------------------

def test_per_ip_limit_blocks_after_quota():
    now = time.time()
    per_min = demo._LIMITS["model"][0][1]
    for i in range(per_min):
        assert demo._check_limits("model", "ip-a", now) == 0, f"call {i} blocked early"
    assert demo._check_limits("model", "ip-a", now) > 0


def test_one_ip_cannot_exhaust_another():
    now = time.time()
    for _ in range(demo._LIMITS["model"][0][1]):
        demo._check_limits("model", "ip-a", now)
    assert demo._check_limits("model", "ip-a", now) > 0
    assert demo._check_limits("model", "ip-b", now) == 0


def test_global_cap_stops_a_distributed_crawl():
    """The paid path calls Google Maps and OpenAI, so a per-IP cap alone is not
    enough — a thousand IPs each staying under it would still spend the money."""
    now = time.time()
    per_min_global = demo._LIMITS["paid"][0][2]
    for i in range(per_min_global):
        assert demo._check_limits("paid", f"ip-{i}", now) == 0
    assert demo._check_limits("paid", "ip-fresh", now) > 0


def test_window_slides_so_the_block_is_temporary():
    now = time.time()
    for _ in range(demo._LIMITS["model"][0][1]):
        demo._check_limits("model", "ip-a", now)
    assert demo._check_limits("model", "ip-a", now) > 0
    # one window later the old hits have aged out
    assert demo._check_limits("model", "ip-a", now + 61) == 0


def test_blocked_calls_do_not_extend_the_block():
    """A blocked request must not be recorded, or a client that keeps retrying
    would never be let back in."""
    now = time.time()
    for _ in range(demo._LIMITS["model"][0][1]):
        demo._check_limits("model", "ip-a", now)
    for _ in range(50):
        demo._check_limits("model", "ip-a", now + 1)
    assert demo._check_limits("model", "ip-a", now + 61) == 0


def test_key_table_is_bounded():
    now = time.time()
    for i in range(demo._MAX_KEYS + 200):
        demo._check_limits("write", f"ip-{i}", now)
    assert len(demo._hits) <= demo._MAX_KEYS + 2


def test_every_routed_path_has_a_limit_class():
    for cls in demo._PATH_CLASS.values():
        assert cls in demo._LIMITS


# --- the board's single model slot -----------------------------------------

def test_slot_is_exclusive_and_reusable():
    assert demo._take_slot() is True
    assert demo._take_slot() is False, "llama-server runs -np 1; two at once must not pass"
    demo._release_slot()
    assert demo._take_slot() is True


def test_stale_slot_self_heals():
    """A stream torn down without running its finally must not wedge the demo
    shut until someone restarts the hub."""
    demo._take_slot()
    demo._slot_taken_at = time.time() - demo._SLOT_STALE_S - 1
    assert demo._take_slot() is True


# --- polled-endpoint cache -------------------------------------------------

def test_cached_computes_once_within_ttl():
    calls = []

    def compute():
        calls.append(1)
        return len(calls)

    demo._cache.pop("t", None)
    assert demo._cached("t", 60.0, compute) == 1
    assert demo._cached("t", 60.0, compute) == 1
    assert len(calls) == 1, "ten open tabs must not mean ten ssh calls to the board"


def test_cached_recomputes_after_ttl():
    calls = []
    demo._cache.pop("t2", None)
    demo._cached("t2", 0.0, lambda: calls.append(1))
    demo._cached("t2", 0.0, lambda: calls.append(1))
    assert len(calls) == 2


# --- bundled sample images -------------------------------------------------
# The offline demo is only honest if recorded.json actually lines up with the
# images and modes the UI offers; a silent mismatch would show a description of
# one photo under another.

import json  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "demo" / "static" / "samples"


def _manifest():
    return json.loads((SAMPLES / "recorded.json").read_text())


def test_every_sample_has_its_image():
    for slug, s in _manifest()["samples"].items():
        assert (SAMPLES / s["file"]).exists(), f"{slug}: missing {s['file']}"
        assert s["file"] == f"{slug}.jpg"


def test_every_sample_records_its_own_mode_and_the_generic_baseline():
    """Compare needs both halves offline, and Describe needs the sample's own
    mode — the UI's offline path assumes exactly these two exist."""
    from veris.prompts import MODES
    for slug, s in _manifest()["samples"].items():
        assert s["mode"] in MODES, f"{slug}: unknown mode {s['mode']}"
        assert s["mode"] in s["runs"], f"{slug}: no run for its own mode"
        assert "general" in s["runs"], f"{slug}: no generic run for Compare"


def test_recorded_runs_are_replayable():
    for slug, s in _manifest()["samples"].items():
        for name, run in s["runs"].items():
            where = f"{slug}/{name}"
            assert run["deltas"], f"{where}: no streamed deltas to replay"
            assert run["text"] == "".join(d[1] for d in run["deltas"]), \
                f"{where}: text does not match its deltas"
            offsets = [d[0] for d in run["deltas"]]
            assert offsets == sorted(offsets), f"{where}: delta offsets out of order"
            assert run["ttft_s"] and run["total_s"], f"{where}: missing timings"
            assert run["ttft_s"] <= run["total_s"], f"{where}: ttft after total"


def test_captions_are_present_for_screen_readers():
    """A blind visitor cannot see which frame is loaded; the alt text and blurb
    ARE the picture for them, so an empty one is a broken control."""
    for slug, s in _manifest()["samples"].items():
        assert len(s["alt"]) > 40, f"{slug}: alt text too thin"
        assert s["blurb"] and s["credit"] and s["credit_url"], f"{slug}: missing caption"


def test_credits_file_lists_every_shipped_image():
    credits = (SAMPLES / "CREDITS.md").read_text()
    for slug in _manifest()["samples"]:
        assert f"`{slug}.jpg`" in credits, f"{slug}.jpg is not credited"


# --- llama-server cold-start warm-up ----------------------------------------
# The board's llama-server segfaults if the first request a fresh instance
# sees carries an image; the hub sends one text completion first, per instance.

def _counter():
    calls = []
    return calls, (lambda: calls.append(1) or True)


def test_new_llm_instance_is_warmed_exactly_once():
    calls, warm = _counter()
    assert demo._ensure_llm_warm(instance_of=lambda: "4242", warm=warm)
    assert demo._ensure_llm_warm(instance_of=lambda: "4242", warm=warm)
    assert len(calls) == 1


def test_restarted_llm_instance_is_warmed_again():
    calls, warm = _counter()
    pid = ["4242"]
    demo._ensure_llm_warm(instance_of=lambda: pid[0], warm=warm)
    pid[0] = "4300"  # systemd restarted the unit: new MainPID
    demo._ensure_llm_warm(instance_of=lambda: pid[0], warm=warm)
    assert len(calls) == 2


def test_failed_warm_up_is_retried_on_the_next_describe():
    attempts = []
    outcome = [False]
    warm = lambda: attempts.append(1) or outcome[0]
    assert demo._ensure_llm_warm(instance_of=lambda: "4242", warm=warm) is False
    outcome[0] = True
    assert demo._ensure_llm_warm(instance_of=lambda: "4242", warm=warm) is True
    assert len(attempts) == 2


def test_without_systemd_the_hub_warms_once_then_again_after_a_crash_signature():
    calls, warm = _counter()
    demo._ensure_llm_warm(instance_of=lambda: None, warm=warm)
    demo._ensure_llm_warm(instance_of=lambda: None, warm=warm)
    assert len(calls) == 1
    demo._note_llm_crash_signature()  # a describe streamed zero tokens
    demo._ensure_llm_warm(instance_of=lambda: None, warm=warm)
    assert len(calls) == 2


def test_llm_instance_ignores_a_stopped_unit(monkeypatch):
    class R:
        def __init__(self, out): self.stdout = out
    monkeypatch.setattr(demo.subprocess, "run", lambda *a, **k: R("0\n"))
    assert demo._llm_instance() is None
    monkeypatch.setattr(demo.subprocess, "run", lambda *a, **k: R("3148\n"))
    assert demo._llm_instance() == "3148"


# --- liveness probe for the recorded copy ------------------------------------
# The GitHub Pages copy asks /api/live cross-origin; nothing else may.

def _fake_status(board_ok=True):
    return {"orin": {"status": "ok" if board_ok else "unreachable"},
            "context_engine": {"status": "ok"}}


def test_live_probe_reports_board_state_from_the_status_cache(monkeypatch):
    from starlette.testclient import TestClient
    demo._cache.clear()
    monkeypatch.setattr(demo, "_status_uncached", lambda: _fake_status(board_ok=False))
    with TestClient(demo.app) as c:
        j = c.get("/api/live").json()
    assert j == {"live": True, "board": False, "context_engine": True}


def test_live_probe_grants_cors_to_the_pages_origin_only():
    pages = "https://0rganizedka0s.github.io"
    assert demo._live_headers(pages)["Access-Control-Allow-Origin"] == pages
    assert demo._live_headers(pages)["Vary"] == "Origin"
    for other in ("https://evil.example", "https://0rganizedka0s.github.io.evil.example",
                  "http://0rganizedka0s.github.io", None):
        assert "Access-Control-Allow-Origin" not in demo._live_headers(other)


def test_no_other_endpoint_sends_cors(monkeypatch):
    from starlette.testclient import TestClient
    demo._cache.clear()
    monkeypatch.setattr(demo, "_status_uncached", _fake_status)
    with TestClient(demo.app) as c:
        r = c.get("/api/status", headers={"Origin": "https://0rganizedka0s.github.io"})
        assert "access-control-allow-origin" not in {k.lower() for k in r.headers}
        r = c.get("/api/live", headers={"Origin": "https://0rganizedka0s.github.io"})
        assert r.headers["access-control-allow-origin"] == "https://0rganizedka0s.github.io"


def test_live_probe_is_in_the_cheap_poll_class():
    assert demo._PATH_CLASS["/api/live"] == "poll"
