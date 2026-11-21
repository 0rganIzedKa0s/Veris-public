"""Context composer: place facts + motion state -> compact context pre-prompt.

Two backends:
- LLM (OpenAI gpt-4o-mini): better phrasing, costs money — so composition is
  CACHED and only re-runs when the situation materially changes, and a daily
  spend guard hard-stops LLM calls past a configurable cap (rules take over).
- Rules: free, deterministic, always available; also the fallback on any API
  error. Quality is close because the output format is deliberately rigid.

The output is one short paragraph in second person, consumed verbatim by the
pack VLM's `--context` slot. It must contain only stable, map-derived facts —
never guesses about what the camera might see (the frame always wins). Both
backends run through `ensure_expectation_phrasing`, which enforces that
guarantee mechanically: any sentence that mentions a `look_for` expectation
without "if ... visible" framing is dropped, and an "If visible, mention
first: ..." sentence is guaranteed present whenever expectations exist.

Vision-vote feedback: the pack VLM can report what the camera actually saw
(POST /vision_vote on the service). A fresh, confident vote is blended into
the situation inference AFTER composition-cache retrieval, so it takes effect
immediately without invalidating the (paid) composition cache:
- vote agrees with the map-derived situation -> situation_source
  "vision_confirmed" (the GPS venue heuristic is corroborated)
- vote disagrees and confidence >= override threshold -> the vote wins:
  situation / suggested_mode / look_for are replaced and the expectation
  sentence is re-generated (GPS is 10-30 m wrong near buildings; the camera
  is ground truth).

Env knobs (all optional):
  CTX_OPENAI_DAILY_CAP     max LLM calls per UTC day (default 300)
  CTX_SPEND_STATE_FILE     spend-guard persistence file (default
                           <tempdir>/veris-ctx-spend.json) so a restart
                           can't reset the daily counter
  CTX_OPENAI_TIMEOUT_S     OpenAI request timeout (default 15)
  CTX_VISION_VOTE_TTL_S    vote freshness window in seconds (default 60)
  CTX_VISION_MIN_CONF      min confidence for a vote to count (default 0.4)
  CTX_VISION_OVERRIDE_CONF min confidence to override maps (default 0.7)
"""

import json
import math
import os
import re
import tempfile
import time
import urllib.request
from dataclasses import dataclass

from .maps import PlaceFacts, _env_float, _env_int

_COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]


def openai_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "")


def compass(deg: float) -> str:
    return _COMPASS[round(deg / 45) % 8]


def relative_side(bearing_deg: float, heading_deg: float | None) -> str:
    """Where a POI sits relative to the user's heading."""
    if heading_deg is None:
        return f"to the {compass(bearing_deg)}"
    d = (bearing_deg - heading_deg + 360) % 360
    if d < 30 or d > 330:
        return "ahead"
    if d < 150:
        return "on your right"
    if d < 210:
        return "behind you"
    return "on your left"


def compose_rules(facts: PlaceFacts, heading_deg: float | None, speed_mps: float | None) -> str:
    bits = []
    where = facts.street or facts.locality
    if where:
        head = f", heading {compass(heading_deg)}" if heading_deg is not None else ""
        moving = "walking" if (speed_mps or 0) > 0.4 else "currently stopped"
        bits.append(f"You are {moving} on {where}" + (f" in {facts.locality}" if facts.street and facts.locality else "") + head + ".")
    crossings = [p for p in facts.nearby if "crossing" in (p.get("types") or [])]
    if crossings:
        c = crossings[0]
        bits.append(f"A pedestrian crossing is about {c['distance_m']} m {relative_side(c['bearing_deg'], heading_deg)}.")
    pois = [p for p in facts.nearby if p.get("name") and "crossing" not in (p.get("types") or [])][:3]
    if pois:
        frag = "; ".join(
            f"{p['name']} about {p['distance_m']} m {relative_side(p['bearing_deg'], heading_deg)}" for p in pois
        )
        bits.append("Nearby: " + frag + ".")
    return " ".join(bits) or "No location information available."


# ---- "if visible, mention first" guarantee ----------------------------------

_VISIBLE_RE = re.compile(r"\bif\b[^.!?]*\bvisible\b", re.IGNORECASE)


def _split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def strip_expectation_sentences(text: str) -> str:
    """Remove any 'if visible' expectation sentences (used before re-phrasing
    when a vision vote replaces look_for)."""
    return " ".join(s for s in _split_sentences(text) if not _VISIBLE_RE.search(s))


def ensure_expectation_phrasing(text: str, look_for: list[str]) -> str:
    """Enforce the never-assert-presence rule on composed text.

    - Sentences that mention a look_for expectation WITHOUT 'if ... visible'
      framing assert presence of something the camera may not see -> dropped.
    - If look_for is non-empty, the text is guaranteed to contain an
      'If visible, mention first: ...' sentence.
    """
    if not look_for:
        return text
    kept = []
    for s in _split_sentences(text):
        low = s.lower()
        mentions = any(item.lower() in low for item in look_for if item)
        if mentions and not _VISIBLE_RE.search(s):
            continue
        kept.append(s)
    if not any(_VISIBLE_RE.search(s) for s in kept):
        kept.append("If visible, mention first: " + ", ".join(look_for) + ".")
    return " ".join(kept)


_LLM_SYSTEM = """You are the situational context engine of an assistive device for a blind \
pedestrian. Input: map facts (street, nearby places with types/distance/bearing) and the \
user's heading/speed. Output STRICT JSON with these keys:
  "situation": one short label for where the user most plausibly is or is entering \
(e.g. "restaurant", "cafe", "street_sidewalk", "street_crossing", "retail_store", \
"transit_stop", "office_building", "residential", "parking_lot", "unknown"). \
IMPORTANT heuristic: GPS is ~10-30 m inaccurate near/inside buildings. If the user is \
stationary or very slow (speed < 0.4 m/s) and a venue is within ~30 m, assume they are \
AT or INSIDE that venue — the situation is the venue type, and look_for should be \
venue-interior expectations. Only a clearly moving user is "street_sidewalk".
  "suggested_mode": exactly one of "outdoor_nav" | "indoor_nav" | "text_reading" | "describe" \
(moving outdoors -> outdoor_nav; inside/entering a venue -> indoor_nav or describe; \
at a menu/signage-heavy spot and stationary -> text_reading)
  "look_for": array of up to 4 short noun phrases the camera should prioritize IF VISIBLE, \
specific to the situation (restaurant: "a waiter or host", "open seats", "menus"; \
street: "the crossing signal", "curb edges"). These are expectations, NOT facts.
  "context_text": max 3 sentences, second person, present tense. Sentence 1: where the user \
is (street/venue, heading). Sentence 2 (optional): nearest 1-2 relevant places with \
egocentric side (convert bearings using heading) and rounded distance. Sentence 3: the \
expectation framing, phrased ONLY as "If X is visible, mention it first" — never assert \
that anything IS present. Never invent places not in the input."""


def compose_llm(facts: PlaceFacts, heading_deg: float | None, speed_mps: float | None) -> dict:
    payload = {
        "model": "gpt-4o-mini",
        "max_tokens": 220,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM},
            {"role": "user", "content": json.dumps({
                "facts": facts.to_dict(),
                "heading_deg": heading_deg,
                "speed_mps": speed_mps,
            })},
        ],
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {openai_key()}"},
    )
    with urllib.request.urlopen(req, timeout=_env_float("CTX_OPENAI_TIMEOUT_S", 15.0)) as r:
        out = json.load(r)
    parsed = json.loads(out["choices"][0]["message"]["content"])
    look_for = [str(x) for x in (parsed.get("look_for") or [])][:4]
    return {
        "situation": parsed.get("situation", "unknown"),
        "suggested_mode": parsed.get("suggested_mode", "describe"),
        "look_for": look_for,
        "context_text": ensure_expectation_phrasing(
            parsed.get("context_text", "").strip(), look_for),
    }


# place types -> (situation, suggested_mode, look_for) for the keyless fallback
_SITUATION_RULES = [
    ({"restaurant", "cafe", "bar", "bakery", "fast_food"}, "restaurant", "indoor_nav",
     ["a waiter or host", "open seats", "menus", "the counter"]),
    ({"supermarket", "store", "shop", "clothing_store", "convenience_store", "shopping_mall"},
     "retail_store", "indoor_nav", ["aisle signs", "the checkout", "shelves", "staff"]),
    ({"bus_station", "transit_station", "train_station", "subway_station", "bus_stop"},
     "transit_stop", "outdoor_nav", ["the stop sign or shelter", "arriving vehicles", "route numbers"]),
    ({"crossing"}, "street_crossing", "outdoor_nav",
     ["the crossing signal", "curb edges", "turning vehicles"]),
]

_STREET_LOOK = ["obstacles in the path", "curb edges", "crossings"]


def infer_situation_rules(facts: PlaceFacts, speed_mps: float | None):
    """Nearest-place-type heuristic: something within ~25 m and user slow ->
    they're plausibly at/in it. Otherwise: on the street."""
    slow = (speed_mps or 0) < 0.6
    for p in facts.nearby:
        types = set(p.get("types") or [])
        if p["distance_m"] <= 25 and slow:
            for typeset, situation, mode, look in _SITUATION_RULES:
                if types & typeset:
                    return situation, mode, list(look)
    if facts.street:
        return "street_sidewalk", "outdoor_nav", list(_STREET_LOOK)
    return "unknown", "describe", []


# ---- vision-vote feedback ----------------------------------------------------

@dataclass
class VisionVote:
    scene_label: str
    confidence: float
    ts: float


class VisionVoteStore:
    """Holds the latest scene vote from the pack VLM. A vote is only usable
    within a freshness window (the user moves; a 5-minute-old 'restaurant'
    vote says nothing about now)."""

    def __init__(self, ttl_s: float | None = None, now=time.time):
        self.ttl_s = _env_float("CTX_VISION_VOTE_TTL_S", 60.0) if ttl_s is None else ttl_s
        self._now = now
        self._vote: VisionVote | None = None

    def submit(self, scene_label: str, confidence: float, ts: float | None = None) -> VisionVote:
        now = self._now()
        ts = now if ts is None else min(ts, now)  # clock skew: never in the future
        self._vote = VisionVote(scene_label=scene_label.strip(),
                                confidence=max(0.0, min(1.0, confidence)), ts=ts)
        return self._vote

    def current(self) -> VisionVote | None:
        v = self._vote
        if v is None or self._now() - v.ts > self.ttl_s:
            return None
        return v


def _canonical_for_label(label: str):
    """Map a free-form vision scene label onto the situation taxonomy.
    Returns (situation, suggested_mode, look_for) — mode/look may be None for
    labels we can't map (the situation label itself is still usable)."""
    lab = label.strip().lower().replace(" ", "_")
    for typeset, situation, mode, look in _SITUATION_RULES:
        if lab in typeset or lab == situation:
            return situation, mode, list(look)
    if lab in ("street", "sidewalk", "street_sidewalk", "outdoor", "outdoors"):
        return "street_sidewalk", "outdoor_nav", list(_STREET_LOOK)
    return lab, None, None


def blend_vision(result: dict, vote: VisionVote | None,
                 min_conf: float | None = None,
                 override_conf: float | None = None) -> dict:
    """Blend a fresh vision vote into a composed result (non-mutating).

    Adds "situation_source": "maps" | "vision_confirmed" | "vision_override".
    """
    min_conf = _env_float("CTX_VISION_MIN_CONF", 0.4) if min_conf is None else min_conf
    override_conf = _env_float("CTX_VISION_OVERRIDE_CONF", 0.7) if override_conf is None else override_conf
    out = dict(result)
    out["situation_source"] = "maps"
    if vote is None or vote.confidence < min_conf:
        return out
    situation, mode, look = _canonical_for_label(vote.scene_label)
    if situation == out.get("situation"):
        out["situation_source"] = "vision_confirmed"
        return out
    if vote.confidence >= override_conf:
        out["situation"] = situation
        if mode is not None:
            out["suggested_mode"] = mode
        if look is not None:
            out["look_for"] = look
            out["context_text"] = ensure_expectation_phrasing(
                strip_expectation_sentences(out.get("context_text", "")), look)
        out["situation_source"] = "vision_override"
    return out


# ---- daily spend guard -------------------------------------------------------

class SpendGuard:
    """Counts OpenAI calls per UTC day and hard-stops past a cap. A blocked
    call falls back to the rules composer — degraded phrasing, zero spend."""

    def __init__(self, daily_cap: int | None = None, now=time.time):
        self.daily_cap = _env_int("CTX_OPENAI_DAILY_CAP", 300) if daily_cap is None else daily_cap
        self._now = now
        self._day: str | None = None
        self.calls_today = 0

    def _roll(self):
        day = time.strftime("%Y-%m-%d", time.gmtime(self._now()))
        if day != self._day:
            self._day, self.calls_today = day, 0

    def allow(self) -> bool:
        self._roll()
        return self.calls_today < self.daily_cap

    def record(self):
        self._roll()
        self.calls_today += 1


class Composer:
    """Caches composition; re-runs only on material change (move >40 m, new
    street, heading swing >60°, stopping or starting to walk) — protects the OpenAI budget and mirrors how
    the real cloud service should behave. A SpendGuard additionally caps LLM
    calls per day. Vision votes are blended after cache retrieval so camera
    feedback applies immediately without re-billing composition."""

    def __init__(self, use_llm: bool = True, daily_cap: int | None = None, now=time.time):
        self.use_llm = use_llm and bool(openai_key())
        self.guard = SpendGuard(daily_cap=daily_cap, now=now)
        self._last: dict | None = None
        self.llm_calls = 0

    def _changed(self, facts: PlaceFacts, heading_deg, motion: str) -> bool:
        if self._last is None:
            return True
        dm = _dist_m(self._last["lat"], self._last["lon"], facts.lat, facts.lon)
        if dm > 40 or self._last["street"] != facts.street:
            return True
        if motion != self._last["motion"]:
            # stopping at a venue / walking away from one changes the
            # situation (the venue heuristic keys on speed) even when the
            # fix barely moves — 30 m is inside GPS error near buildings
            return True
        if heading_deg is not None and self._last["heading"] is not None:
            swing = abs((heading_deg - self._last["heading"] + 180) % 360 - 180)
            if swing > 60:
                return True
        return False

    def _compose_base(self, facts: PlaceFacts, heading_deg, speed_mps) -> tuple[dict, str]:
        motion = _motion_state(speed_mps, self._last["motion"] if self._last else None)
        if not self._changed(facts, heading_deg, motion):
            return self._last["result"], self._last["by"] + "+cache"
        result, by = None, "rules"
        if self.use_llm and self.guard.allow():
            self.guard.record()  # count the attempt: the API bills it either way
            try:
                result, by = compose_llm(facts, heading_deg, speed_mps), "llm"
                self.llm_calls += 1
            except Exception:
                result = None
        if result is None:
            situation, mode, look = infer_situation_rules(facts, speed_mps)
            text = ensure_expectation_phrasing(
                compose_rules(facts, heading_deg, speed_mps), look)
            result = {"situation": situation, "suggested_mode": mode,
                      "look_for": look, "context_text": text}
        self._last = {"lat": facts.lat, "lon": facts.lon, "street": facts.street,
                      "heading": heading_deg, "motion": motion, "result": result, "by": by}
        return result, by

    def compose(self, facts: PlaceFacts, heading_deg=None, speed_mps=None,
                vision_vote: VisionVote | None = None) -> tuple[dict, str]:
        """Returns (result, composer_used). result keys: situation,
        suggested_mode, look_for, context_text, situation_source."""
        base, by = self._compose_base(facts, heading_deg, speed_mps)
        return blend_vision(base, vision_vote), by


def _motion_state(speed_mps, prev: str | None) -> str:
    """"moving" | "still", with a dead band so GPS speed noise around the
    threshold cannot flap the (billed) composition on every fix."""
    s = speed_mps or 0.0
    if s > 0.6:
        return "moving"
    if s < 0.3:
        return "still"
    return prev or "still"


def _dist_m(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))
