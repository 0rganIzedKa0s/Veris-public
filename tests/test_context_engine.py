"""Context engine tests — all external HTTP is mocked.

  uv run --with pytest --with httpx python -m pytest tests/test_context_engine.py -q

Covers: maps degradation chain (Google -> OSM -> bare facts), retry/backoff,
lookup TTL cache, both composer backends, the composition cache, the daily
OpenAI spend guard, the "if visible, mention first" phrasing guarantee /
never-assert-presence rule, vision-vote blending, and the FastAPI endpoints.
"""

import io
import json
import urllib.error

import pytest

from context_engine import maps
from context_engine.maps import PlaceFacts
from context_engine import composer as comp
from context_engine.composer import (
    Composer, SpendGuard, VisionVote, VisionVoteStore,
    blend_vision, compose_rules, ensure_expectation_phrasing,
    infer_situation_rules, strip_expectation_sentences,
)


# ---- helpers -----------------------------------------------------------------

class FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def openai_body(situation="restaurant", mode="indoor_nav",
                look_for=None, context_text="You are at the door."):
    look_for = ["menus", "open seats"] if look_for is None else look_for
    return {"choices": [{"message": {"content": json.dumps({
        "situation": situation, "suggested_mode": mode,
        "look_for": look_for, "context_text": context_text,
    })}}]}


GEOCODE = {"results": [{"address_components": [
    {"long_name": "High Street", "types": ["route"]},
    {"long_name": "Columbus", "types": ["locality", "political"]},
]}]}

PLACES = {"results": [
    {"name": "Cedar Kitchen", "types": ["restaurant", "food", "point_of_interest", "establishment"],
     "geometry": {"location": {"lat": 39.96410, "lng": -83.00075}}},
    {"name": "Northside Coffee", "types": ["cafe", "point_of_interest"],
     "geometry": {"location": {"lat": 39.96420, "lng": -83.00100}}},
]}

NOMINATIM = {"address": {"road": "High Street", "city": "Columbus"}}
OVERPASS = {"elements": [
    {"lat": 39.96405, "lon": -83.00080, "tags": {"name": "Cedar Kitchen", "amenity": "restaurant"}},
    {"lat": 39.96390, "lon": -83.00060, "tags": {"highway": "crossing"}},
]}


def http_error(code):
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(b"{}"))


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    maps.clear_cache()
    monkeypatch.setenv("CTX_HTTP_RETRIES", "0")           # tests opt in to retries
    monkeypatch.setattr(maps.time, "sleep", lambda s: None)
    yield
    maps.clear_cache()


def mock_http(monkeypatch, handler):
    """Route both maps' and composer's urlopen through `handler(url) -> resp`."""
    def fake_urlopen(req, timeout=None):
        url = req.get_full_url() if hasattr(req, "get_full_url") else req
        return handler(url)
    monkeypatch.setattr(maps.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(comp.urllib.request, "urlopen", fake_urlopen)


def google_ok(url):
    if "maps.googleapis.com/maps/api/geocode" in url:
        return FakeResp(GEOCODE)
    if "maps.googleapis.com/maps/api/place" in url:
        return FakeResp(PLACES)
    raise AssertionError(f"unexpected URL {url}")


# ---- maps: parsing + degradation chain ----------------------------------------

def test_google_lookup_parses(monkeypatch):
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "k")
    mock_http(monkeypatch, google_ok)
    facts = maps.lookup(39.9640, -83.0007)
    assert facts.source == "google"
    assert facts.street == "High Street"
    assert facts.locality == "Columbus"
    assert [p["name"] for p in facts.nearby] == ["Cedar Kitchen", "Northside Coffee"]
    assert facts.nearby[0]["distance_m"] <= facts.nearby[1]["distance_m"]
    # point_of_interest / establishment stripped from types
    assert "point_of_interest" not in facts.nearby[0]["types"]


def test_google_failure_falls_back_to_osm(monkeypatch):
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "k")

    def handler(url):
        if "googleapis" in url:
            raise http_error(500)
        if "nominatim" in url:
            return FakeResp(NOMINATIM)
        if "overpass" in url:
            return FakeResp(OVERPASS)
        raise AssertionError(url)
    mock_http(monkeypatch, handler)
    facts = maps.lookup(39.9640, -83.0007)
    assert facts.source == "osm"
    assert facts.street == "High Street"
    assert any("crossing" in (p["types"] or []) for p in facts.nearby)


def test_all_backends_fail_returns_bare_facts(monkeypatch):
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "k")

    def handler(url):
        raise urllib.error.URLError("network down")
    mock_http(monkeypatch, handler)
    facts = maps.lookup(39.9640, -83.0007)
    assert facts.source == "none"
    assert facts.street is None and facts.nearby == []
    # the rules composer still yields a usable string on bare facts
    assert compose_rules(facts, None, None) == "No location information available."


def test_bare_facts_not_cached(monkeypatch):
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "")
    calls = {"n": 0}

    def handler(url):
        calls["n"] += 1
        raise urllib.error.URLError("down")
    mock_http(monkeypatch, handler)
    maps.lookup(39.9640, -83.0007)
    n1 = calls["n"]
    maps.lookup(39.9640, -83.0007)   # failure was not cached -> retried
    assert calls["n"] > n1


# ---- maps: retries / backoff ---------------------------------------------------

def test_retry_recovers_from_transient_5xx(monkeypatch):
    monkeypatch.setenv("CTX_HTTP_RETRIES", "2")
    attempts = {"n": 0}

    def handler(url):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise http_error(503)
        return FakeResp({"ok": True})
    mock_http(monkeypatch, handler)
    assert maps._get("https://maps.googleapis.com/x") == {"ok": True}
    assert attempts["n"] == 3


def test_no_retry_on_permanent_4xx(monkeypatch):
    monkeypatch.setenv("CTX_HTTP_RETRIES", "3")
    attempts = {"n": 0}

    def handler(url):
        attempts["n"] += 1
        raise http_error(403)
    mock_http(monkeypatch, handler)
    with pytest.raises(urllib.error.HTTPError):
        maps._get("https://maps.googleapis.com/x")
    assert attempts["n"] == 1        # bad key: retrying can't heal it


def test_retries_exhaust_then_raise(monkeypatch):
    monkeypatch.setenv("CTX_HTTP_RETRIES", "2")
    attempts = {"n": 0}

    def handler(url):
        attempts["n"] += 1
        raise urllib.error.URLError("timeout")
    mock_http(monkeypatch, handler)
    with pytest.raises(urllib.error.URLError):
        maps._get("https://maps.googleapis.com/x")
    assert attempts["n"] == 3        # 1 try + 2 retries


def test_backoff_sleeps_between_retries(monkeypatch):
    monkeypatch.setenv("CTX_HTTP_RETRIES", "2")
    monkeypatch.setenv("CTX_HTTP_BACKOFF_S", "0.5")
    sleeps = []
    monkeypatch.setattr(maps.time, "sleep", sleeps.append)

    def handler(url):
        raise http_error(500)
    mock_http(monkeypatch, handler)
    with pytest.raises(urllib.error.HTTPError):
        maps._get("https://maps.googleapis.com/x")
    assert sleeps == [0.5, 1.0]      # exponential


# ---- maps: TTL cache -----------------------------------------------------------

def test_lookup_cache_hits_within_ttl(monkeypatch):
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "k")
    monkeypatch.setenv("CTX_MAPS_CACHE_TTL_S", "120")
    calls = {"n": 0}

    def handler(url):
        calls["n"] += 1
        return google_ok(url)
    mock_http(monkeypatch, handler)
    t = {"now": 1000.0}
    f1 = maps.lookup(39.9640, -83.0007, now=lambda: t["now"])
    assert calls["n"] == 2           # geocode + places
    t["now"] += 30
    f2 = maps.lookup(39.96401, -83.00071, now=lambda: t["now"])  # same ~11m cell
    assert calls["n"] == 2 and f2 is f1
    t["now"] += 200                  # past TTL
    maps.lookup(39.9640, -83.0007, now=lambda: t["now"])
    assert calls["n"] == 4


# ---- composer: rules path -------------------------------------------------------

def facts_street(nearby=None):
    return PlaceFacts(lat=39.9640, lon=-83.0007, street="High Street",
                      locality="Columbus", source="google", nearby=nearby or [])


def test_compose_rules_street_crossing_pois():
    facts = facts_street(nearby=[
        {"name": "crossing", "types": ["crossing"], "distance_m": 12, "bearing_deg": 205},
        {"name": "Cedar Kitchen", "types": ["restaurant"], "distance_m": 18, "bearing_deg": 295},
    ])
    text = compose_rules(facts, heading_deg=205, speed_mps=1.3)
    assert "walking on High Street" in text
    assert "in Columbus" in text and "heading south-west" in text
    assert "pedestrian crossing is about 12 m ahead" in text
    assert "Cedar Kitchen about 18 m on your right" in text


def test_infer_situation_rules_venue_vs_street():
    venue = facts_street(nearby=[
        {"name": "Cedar Kitchen", "types": ["restaurant"], "distance_m": 10, "bearing_deg": 0}])
    assert infer_situation_rules(venue, 0.2)[0] == "restaurant"
    assert infer_situation_rules(venue, 1.5)[0] == "street_sidewalk"  # moving -> street
    assert infer_situation_rules(PlaceFacts(lat=0, lon=0), 0.0)[0] == "unknown"


def test_rules_result_has_if_visible_guarantee(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = Composer(use_llm=False)
    result, by = c.compose(facts_street(), heading_deg=205, speed_mps=1.3)
    assert by == "rules"
    assert result["look_for"]
    assert "If visible, mention first:" in result["context_text"]


# ---- composer: LLM path, fallback, cache ----------------------------------------

def test_llm_path_and_composition_cache(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls = {"n": 0}

    def handler(url):
        assert "api.openai.com" in url
        calls["n"] += 1
        return FakeResp(openai_body(
            context_text="You are on High Street. If menus are visible, mention them first."))
    mock_http(monkeypatch, handler)
    c = Composer(use_llm=True)
    r1, by1 = c.compose(facts_street(), 205, 1.3)
    assert by1 == "llm" and r1["situation"] == "restaurant" and calls["n"] == 1
    # ~11 m away, same street, same heading -> cached, no second LLM call
    near = facts_street(); near.lat += 0.0001
    r2, by2 = c.compose(near, 205, 1.3)
    assert by2 == "llm+cache" and calls["n"] == 1
    # move >40 m -> recompose
    far = facts_street(); far.lat += 0.001
    _, by3 = c.compose(far, 205, 1.3)
    assert by3 == "llm" and calls["n"] == 2


def test_stopping_at_a_venue_recomposes_despite_the_distance_cache(monkeypatch):
    """Found live on 2026-09-02: walk past a restaurant (composed as street),
    stop inside it 30 m later -> the distance cache returned the street
    composition; and the reverse, walking away, kept saying "you are at"."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    venue = facts_street(nearby=[
        {"name": "Cedar Kitchen", "types": ["restaurant"], "distance_m": 10, "bearing_deg": 0}])
    c = Composer(use_llm=False)
    r1, by1 = c.compose(venue, 180, 1.3)          # walking past
    assert by1 == "rules" and r1["situation"] == "street_sidewalk"
    r2, by2 = c.compose(venue, 180, 0.0)          # stopped, same fix
    assert by2 == "rules" and r2["situation"] == "restaurant"
    r3, by3 = c.compose(venue, 180, 0.2)          # still stopped (GPS jitter)
    assert by3 == "rules+cache" and r3["situation"] == "restaurant"
    r4, by4 = c.compose(venue, 180, 0.45)         # dead band: keeps "still"
    assert by4 == "rules+cache"
    r5, by5 = c.compose(venue, 180, 1.4)          # walks away
    assert by5 == "rules" and r5["situation"] == "street_sidewalk"


def test_llm_error_falls_back_to_rules(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def handler(url):
        raise urllib.error.URLError("openai down")
    mock_http(monkeypatch, handler)
    c = Composer(use_llm=True)
    result, by = c.compose(facts_street(), 205, 1.3)
    assert by == "rules"
    assert "High Street" in result["context_text"]


# ---- composer: daily spend guard -------------------------------------------------

def test_spend_guard_hard_stops_at_cap(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls = {"n": 0}

    def handler(url):
        calls["n"] += 1
        return FakeResp(openai_body())
    mock_http(monkeypatch, handler)
    c = Composer(use_llm=True, daily_cap=2)
    f = facts_street()
    for i in range(4):                       # 4 distinct spots -> 4 recompositions
        f2 = facts_street(); f2.lat += 0.002 * i
        _, by = c.compose(f2, 205, 1.3)
    assert calls["n"] == 2                   # hard stop past the cap
    assert by == "rules"                     # last composition fell back
    assert c.guard.calls_today == 2


def test_spend_guard_resets_next_utc_day():
    t = {"now": 1_750_000_000.0}
    g = SpendGuard(daily_cap=1, now=lambda: t["now"])
    assert g.allow(); g.record()
    assert not g.allow()
    t["now"] += 86400
    assert g.allow()
    assert g.calls_today == 0


def test_failed_llm_attempt_still_counts_against_cap(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def handler(url):
        raise urllib.error.URLError("down")
    mock_http(monkeypatch, handler)
    c = Composer(use_llm=True, daily_cap=5)
    c.compose(facts_street(), 205, 1.3)
    assert c.guard.calls_today == 1


# ---- phrasing guarantee / never-assert-presence -----------------------------------

def test_ensure_phrasing_appends_when_missing():
    out = ensure_expectation_phrasing("You are on Main Street.", ["menus", "open seats"])
    assert out == "You are on Main Street. If visible, mention first: menus, open seats."


def test_ensure_phrasing_strips_presence_assertions():
    text = ("You are at Cedar Kitchen. There are menus on the counter. "
            "If open seats are visible, mention them first.")
    out = ensure_expectation_phrasing(text, ["menus", "open seats"])
    assert "There are menus" not in out                 # asserted presence -> dropped
    assert "If open seats are visible" in out           # correct framing -> kept


def test_ensure_phrasing_noop_without_expectations():
    assert ensure_expectation_phrasing("You are on Main Street.", []) == "You are on Main Street."


def test_llm_output_sanitized_end_to_end(monkeypatch):
    """Even if the LLM disobeys and asserts presence, the composed text obeys
    the never-assert-presence rule."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def handler(url):
        return FakeResp(openai_body(
            look_for=["menus", "a waiter"],
            context_text="You are inside Cedar Kitchen. There are menus at the counter and a waiter nearby."))
    mock_http(monkeypatch, handler)
    result, by = Composer(use_llm=True).compose(facts_street(), 205, 0.2)
    assert by == "llm"
    assert "There are menus" not in result["context_text"]
    assert "If visible, mention first: menus, a waiter." in result["context_text"]


def test_strip_expectation_sentences():
    text = "You are on Main Street. If visible, mention first: menus."
    assert strip_expectation_sentences(text) == "You are on Main Street."


# ---- vision-vote blending -----------------------------------------------------

def street_result():
    return {"situation": "street_sidewalk", "suggested_mode": "outdoor_nav",
            "look_for": ["obstacles in the path", "curb edges", "crossings"],
            "context_text": ("You are walking on High Street. "
                             "If visible, mention first: obstacles in the path, curb edges, crossings.")}


def test_vote_confirms_matching_situation():
    r = {"situation": "restaurant", "suggested_mode": "indoor_nav",
         "look_for": ["menus"], "context_text": "x. If visible, mention first: menus."}
    out = blend_vision(r, VisionVote("restaurant", 0.6, ts=0))
    assert out["situation_source"] == "vision_confirmed"
    assert out["situation"] == "restaurant"             # unchanged


def test_confident_vote_overrides_gps_heuristic():
    out = blend_vision(street_result(), VisionVote("restaurant", 0.9, ts=0))
    assert out["situation_source"] == "vision_override"
    assert out["situation"] == "restaurant"
    assert out["suggested_mode"] == "indoor_nav"
    assert "a waiter or host" in out["look_for"]
    # expectation sentence re-generated for the NEW look_for
    assert "If visible, mention first: a waiter or host" in out["context_text"]
    assert "curb edges" not in out["context_text"]


def test_weak_disagreeing_vote_does_not_override():
    out = blend_vision(street_result(), VisionVote("restaurant", 0.5, ts=0))
    assert out["situation_source"] == "maps"
    assert out["situation"] == "street_sidewalk"


def test_below_min_confidence_vote_ignored():
    out = blend_vision(street_result(), VisionVote("restaurant", 0.2, ts=0))
    assert out["situation_source"] == "maps"


def test_blend_does_not_mutate_cached_result():
    r = street_result()
    blend_vision(r, VisionVote("restaurant", 0.9, ts=0))
    assert r["situation"] == "street_sidewalk"          # original untouched


def test_vote_store_freshness_window():
    t = {"now": 1000.0}
    s = VisionVoteStore(ttl_s=60, now=lambda: t["now"])
    assert s.current() is None
    s.submit("restaurant", 0.9)
    assert s.current().scene_label == "restaurant"
    t["now"] += 61
    assert s.current() is None                          # stale vote expired


def test_vote_store_clamps_future_ts_and_confidence():
    t = {"now": 1000.0}
    s = VisionVoteStore(ttl_s=60, now=lambda: t["now"])
    v = s.submit("cafe", 1.7, ts=99999.0)
    assert v.ts == 1000.0 and v.confidence == 1.0


def test_unmapped_label_overrides_situation_only():
    out = blend_vision(street_result(), VisionVote("elevator lobby", 0.9, ts=0))
    assert out["situation"] == "elevator_lobby"
    assert out["suggested_mode"] == "outdoor_nav"       # kept: no mapping for mode
    assert out["look_for"] == street_result()["look_for"]


# ---- service endpoints ---------------------------------------------------------

@pytest.fixture()
def client(monkeypatch):
    from starlette.testclient import TestClient
    from context_engine import service
    monkeypatch.setattr(service, "composer", Composer(use_llm=False))
    monkeypatch.setattr(service, "votes", VisionVoteStore(ttl_s=60))
    def fake_lookup(lat, lon):
        f = facts_street(nearby=[
            {"name": "Cedar Kitchen", "types": ["restaurant"], "distance_m": 15, "bearing_deg": 300}])
        f.lat, f.lon = lat, lon      # keep the composer's move-distance cache honest
        return f
    monkeypatch.setattr(service, "lookup", fake_lookup)
    return TestClient(service.app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "llm_calls_today" in body and "llm_daily_cap" in body


def test_context_endpoint_shape(client):
    r = client.post("/context", json={"lat": 39.9640, "lon": -83.0007,
                                      "heading_deg": 205, "speed_mps": 1.3})
    assert r.status_code == 200
    body = r.json()
    assert body["situation"] == "street_sidewalk"
    assert body["situation_source"] == "maps"
    assert body["composed_by"] == "rules"
    assert "High Street" in body["context"]
    assert "If visible, mention first:" in body["context"]
    assert body["facts"]["source"] == "google"


def test_vision_vote_flow_confirm_and_override(client):
    # vote "restaurant" -> stationary user near Cedar Kitchen: rules also say restaurant -> confirmed
    r = client.post("/vision_vote", json={"scene_label": "restaurant", "confidence": 0.9})
    assert r.status_code == 200 and r.json()["accepted"] is True
    body = client.post("/context", json={"lat": 39.9640, "lon": -83.0007,
                                         "heading_deg": 205, "speed_mps": 0.2}).json()
    assert body["situation"] == "restaurant"
    assert body["situation_source"] == "vision_confirmed"
    # walking fast -> maps say street, confident vote overrides back to restaurant
    body = client.post("/context", json={"lat": 39.9800, "lon": -83.0200,
                                         "heading_deg": 205, "speed_mps": 1.5}).json()
    assert body["situation"] == "restaurant"
    assert body["situation_source"] == "vision_override"
    assert body["suggested_mode"] == "indoor_nav"


def test_vision_vote_validation(client):
    assert client.post("/vision_vote", json={"scene_label": "x", "confidence": 1.5}).status_code == 422
    assert client.post("/vision_vote", json={"scene_label": "", "confidence": 0.5}).status_code == 422
    assert client.post("/vision_vote", json={"confidence": 0.5}).status_code == 422
