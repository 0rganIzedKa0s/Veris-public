"""Tier 3 offline planner: phone GPS fix -> packet.Situation, airgapped.

Reads only a local OpenStreetMap index (data/maps/index.pkl, built from the
cached Overpass extracts by `python -m sim.gps fetch`). This module imports no
network client: on the device it must work with the radio off.

It is a drop-in for the cloud context engine (context_engine/composer.py):
same situation vocabulary, same suggested_mode vocabulary (arbiter.MODES),
same "if visible" look-for guarantee. What it adds:

- an indoor/outdoor estimate from GNSS quality (accuracy, dropouts) plus the
  building footprints, held with hysteresis; the cloud engine has only the
  "slow and a venue within 25 m" guess, which it keeps here as a fallback;
- class_priors: which detector classes matter at this spot;
- route_hint: the next user-relative instruction, from a route polyline or a
  phone navigation instruction plus the user's heading.

The composer's motion-state cache rule carries over: a venue adopted while the
user stood still is dropped the moment they start walking away.

`llm_prompt(facts)` / `parse_llm(text)` are the optional on-device LLM polish
step (the 3B base in text-only mode); the rules path is complete without it.

Call `Planner.plan` once per second. `plan(None)` means "no fix arrived this
second", which is itself indoor evidence.
"""

from __future__ import annotations

import json
import math
import pickle
import re
from pathlib import Path

import numpy as np

from .arbiter import MODES
from .packet import Situation

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "data" / "maps" / "index.pkl"

# ---- vocabulary shared with context_engine/composer.py ----------------------
# Copied, not imported: composer imports urllib and this module must not.
# tests/test_planner.py asserts the copies still match.

SITUATIONS = ("restaurant", "cafe", "street_sidewalk", "street_crossing",
              "retail_store", "transit_stop", "office_building", "residential",
              "parking_lot", "unknown")

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
# situations the composer's rules never produce; look-for in the same spirit
_EXTRA_LOOK = {
    "office_building": ["doors", "signs and room numbers", "stairs or elevators", "people"],
    "parking_lot": ["moving cars", "curb edges", "the walkway"],
}
_COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
_VISIBLE_RE = re.compile(r"\bif\b[^.!?]*\bvisible\b", re.IGNORECASE)


def compass(deg: float) -> str:
    return _COMPASS[round(deg / 45) % 8]


def relative_side(bearing_deg: float, heading_deg: float | None) -> str:
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


def _split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def ensure_expectation_phrasing(text: str, look_for: list[str]) -> str:
    """Same guarantee as the composer: never assert that an expectation is
    present; always end with an 'If visible, mention first' sentence."""
    if not look_for:
        return text
    kept = []
    for s in _split_sentences(text):
        mentions = any(item.lower() in s.lower() for item in look_for if item)
        if mentions and not _VISIBLE_RE.search(s):
            continue
        kept.append(s)
    if not any(_VISIBLE_RE.search(s) for s in kept):
        kept.append("If visible, mention first: " + ", ".join(look_for) + ".")
    return " ".join(kept)


# ---- OSM tags -> the composer's place types ---------------------------------

_SHOP = {"clothes": "clothing_store", "convenience": "convenience_store",
         "supermarket": "supermarket", "mall": "shopping_mall", "bakery": "bakery",
         "coffee": "cafe"}
_OFFICE_BUILDINGS = {"university", "college", "school", "office", "public", "civic",
                     "government", "library", "hospital", "dormitory", "commercial"}
WALKABLE = {"footway", "path", "pedestrian", "steps", "residential", "service",
            "living_street", "tertiary", "secondary", "primary", "unclassified",
            "cycleway", "tertiary_link", "secondary_link", "primary_link", "track",
            "corridor"}


def osm_types(tags: dict) -> list[str]:
    t = []
    if a := tags.get("amenity"):
        t.append(a)
    if s := tags.get("shop"):
        t.append(_SHOP.get(s, "store"))
    hw = tags.get("highway")
    if hw in ("bus_stop", "crossing"):
        t.append(hw)
    if tags.get("public_transport") == "platform" and tags.get("bus") == "yes" and "bus_stop" not in t:
        t.append("bus_stop")
    if tags.get("railway") == "station":
        t.append("train_station")
    return t


def venue_situation(types) -> str | None:
    types = set(types)
    for typeset, sit, _, _ in _SITUATION_RULES[:2]:
        if types & typeset:
            return sit
    if types & (_OFFICE_BUILDINGS | {"university", "college", "library", "townhall"}):
        return "office_building"
    return None


# ---- index ------------------------------------------------------------------

def _levels(tags) -> float:
    for k, scale in (("building:levels", 1.0), ("height", 1 / 3.5)):
        try:
            return float(str(tags.get(k, "")).split(";")[0].replace("m", "")) * scale
        except ValueError:
            pass
    return 0.0


def build_index(osm_paths) -> dict:
    """Overpass JSON extracts -> one compact dict of numpy arrays + lists."""
    pois, blds, parking, coords, edges = [], [], [], {}, []
    for path in osm_paths:
        for e in json.loads(Path(path).read_text())["elements"]:
            tags = e.get("tags", {})
            if e["type"] == "node":
                types = osm_types(tags) + (["entrance"] if tags.get("entrance") else [])
                if types:
                    pois.append({"name": tags.get("name"), "types": types, "lat": e["lat"],
                                 "lon": e["lon"], "signals": tags.get("crossing") == "traffic_signals"})
                continue
            geom = [(g["lat"], g["lon"]) for g in e.get("geometry") or [] if g]
            if len(geom) < 2:
                continue
            hw = tags.get("highway")
            if hw in WALKABLE and tags.get("foot") != "no" and tags.get("access") not in ("private", "no"):
                for a, b, pa, pb in zip(e["nodes"], e["nodes"][1:], geom, geom[1:]):
                    coords[a], coords[b] = pa, pb
                    edges.append((a, b, hw, tags.get("name"), tags.get("footway") == "crossing"))
                mid = geom[len(geom) // 2]
                if hw == "steps":
                    pois.append({"name": None, "types": ["steps"], "lat": mid[0], "lon": mid[1]})
                if tags.get("footway") == "crossing":
                    pois.append({"name": None, "types": ["crossing"], "lat": mid[0], "lon": mid[1],
                                 "signals": tags.get("crossing") == "traffic_signals"})
            closed = len(geom) > 3 and e["nodes"][0] == e["nodes"][-1]
            if not closed:
                continue
            poly = np.array(geom)
            types = osm_types(tags)
            if "building" in tags:
                blds.append({"name": tags.get("name"), "types": types + [tags["building"]],
                             "levels": _levels(tags), "poly": poly})
            elif tags.get("amenity") == "parking":
                parking.append(poly)
            if types and tags.get("name") and tags.get("amenity") != "parking":
                c = poly.mean(0)
                pois.append({"name": tags["name"], "types": types, "lat": c[0], "lon": c[1]})
    ids = list(coords)
    pos = {n: i for i, n in enumerate(ids)}
    return {
        "pois": pois,
        "poi_ll": np.array([(p["lat"], p["lon"]) for p in pois]),
        "buildings": blds,
        "bld_box": np.array([(*b["poly"].min(0), *b["poly"].max(0)) for b in blds]),
        "parking": parking,
        "node_ll": np.array([coords[n] for n in ids]),
        "edges": np.array([(pos[a], pos[b]) for a, b, *_ in edges], dtype=np.int64),
        "edge_kind": [k for _, _, k, _, _ in edges],
        "edge_name": [n for _, _, _, n, _ in edges],
        "edge_crossing": np.array([c for *_, c in edges], dtype=bool),
    }


def load_index(path=INDEX) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---- geometry (local tangent plane, metres) ---------------------------------

def to_xy(ll, lat0: float, lon0: float) -> np.ndarray:
    """(n,2) lat/lon -> (n,2) east/north metres around (lat0, lon0)."""
    ll = np.asarray(ll, dtype=float).reshape(-1, 2)
    return np.column_stack(((ll[:, 1] - lon0) * 111320.0 * math.cos(math.radians(lat0)),
                            (ll[:, 0] - lat0) * 110540.0))


def from_xy(xy, lat0: float, lon0: float) -> np.ndarray:
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    return np.column_stack((lat0 + xy[:, 1] / 110540.0,
                            lon0 + xy[:, 0] / (111320.0 * math.cos(math.radians(lat0)))))


def bearing(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dx, dy)) % 360


def point_in_poly(x: float, y: float, poly_xy: np.ndarray) -> bool:
    px, py = poly_xy[:, 0], poly_xy[:, 1]
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    hit = ((py > y) != (qy > y)) & (x < (qx - px) * (y - py) / np.where(qy == py, 1e-12, qy - py) + px)
    return bool(hit.sum() % 2)


def seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance from point p to segments a[i]-b[i]; also the projection fraction."""
    d = b - a
    L2 = np.maximum((d ** 2).sum(1), 1e-9)
    u = np.clip(((p - a) * d).sum(1) / L2, 0, 1)
    return np.hypot(*(a + d * u[:, None] - p).T), u


# ---- route hints ------------------------------------------------------------

def _r5(m: float) -> int:
    return max(5, int(round(m / 5.0)) * 5)


def _turn_word(delta: float) -> str | None:
    """Signed change of direction (degrees, + = clockwise) -> instruction."""
    a = abs(delta)
    if a < 30:
        return None
    side = "right" if delta > 0 else "left"
    if a > 150:
        return "turn around"
    return f"bear {side}" if a < 60 else f"turn {side}"


_INSTR_TURN = re.compile(r"\b(slight|sharp)?\s*(left|right)\b", re.I)
_INSTR_HEAD = re.compile(r"\b(?:head|continue|walk)\s+(north|south|east|west|northeast|northwest|southeast|southwest)\b", re.I)
_DIRS = {"north": 0, "northeast": 45, "east": 90, "southeast": 135, "south": 180,
         "southwest": 225, "west": 270, "northwest": 315}


def route_hint(lat: float, lon: float, heading: float | None, route,
               crossings_ll: np.ndarray | None = None) -> str | None:
    """User-relative next instruction.

    route is either a polyline [[lat, lon], ...] or a phone instruction
    {"text": "Turn left onto Oak St", "lat": .., "lon": ..} whose lat/lon is
    the manoeuvre point."""
    if not route:
        return None
    if isinstance(route, dict):
        return _instruction_hint(lat, lon, heading, route)
    xy = to_xy(route, lat, lon)
    if len(xy) < 2:
        return None
    a, b = xy[:-1], xy[1:]
    seglen = np.hypot(*(b - a).T)
    cum = np.concatenate(([0.0], np.cumsum(seglen)))
    dist, u = seg_dist(np.zeros(2), a, b)
    seg_brg = np.degrees(np.arctan2(*(b - a).T)) % 360
    cost = dist.copy()
    if heading is not None:
        # a route that doubles back passes the user twice; take the leg the
        # user is walking along unless the other one is much nearer
        dev = np.abs((seg_brg - heading + 180) % 360 - 180)
        cost += np.where(dev > 90, 20.0, 0.0)
    i = int(np.argmin(cost))
    s0 = cum[i] + u[i] * seglen[i]
    total = cum[-1]
    events = []                                       # (along-route metres ahead, text)
    if crossings_ll is not None and len(crossings_ll):
        cxy = to_xy(crossings_ll, lat, lon)
        cxy = cxy[np.hypot(*cxy.T) < total - s0 + 50]
        if len(cxy):                                   # crossings x segments at once
            d = b - a
            L2 = np.maximum((d ** 2).sum(1), 1e-9)
            cu = np.clip((((cxy[:, None, :] - a) * d).sum(2)) / L2, 0, 1)
            cd = np.hypot(*(a + d * cu[..., None] - cxy[:, None, :]).transpose(2, 0, 1))
            j = cd.argmin(1)
            r = np.arange(len(cxy))
            sc = cum[j] + cu[r, j] * seglen[j]
            for s_c in sc[(cd[r, j] < 8) & (sc > s0 - 2)]:
                events.append((s_c - s0, "crossing"))
    # turns: compare the direction of the nearest legs longer than 2 m
    for k in range(i + 1, len(xy) - 1):
        if cum[k] - s0 > 80:                      # only the next 80 m matter
            break
        before = next((seg_brg[m] for m in range(k - 1, i - 1, -1) if seglen[m] > 2), None)
        after = next((seg_brg[m] for m in range(k, len(a)) if seglen[m] > 2), None)
        if before is None or after is None:
            continue
        w = _turn_word((after - before + 180) % 360 - 180)
        if w and w != "turn around":
            events.append((cum[k] - s0, w))
            break
    remaining = total - s0
    if remaining < 30:
        events.append((remaining, "destination"))
    # is the user facing along the route at all?
    if heading is not None and dist[i] < 25:
        dev = (seg_brg[i] - heading + 180) % 360 - 180
        w = _turn_word(dev)
        if w and abs(dev) > 60:
            return f"{w} to follow the route" if w != "turn around" else "turn around to follow the route"
    if not events:
        return f"continue straight for {_r5(remaining)} m"
    d, what = min(events)
    if d < 3:
        return f"{what} now" if what != "destination" else "destination here"
    return f"{what} in {_r5(d)} m"


def _instruction_hint(lat, lon, heading, ins: dict) -> str | None:
    text = str(ins.get("text", ""))
    d, brg = 0.0, None
    if "lat" in ins and "lon" in ins:
        dx, dy = to_xy([[ins["lat"], ins["lon"]]], lat, lon)[0]
        d, brg = math.hypot(dx, dy), bearing(dx, dy)
    when = "now" if d < 5 else f"in {_r5(d)} m"
    if re.search(r"\b(destination|arrive)", text, re.I):
        side = relative_side(brg, heading) if brg is not None else ""
        return f"destination {when}" + (f", {side}" if side and d >= 5 else "")
    if re.search(r"\bcross\b", text, re.I):
        return f"crossing {when}"
    m = _INSTR_HEAD.search(text)
    if m:
        target = _DIRS[m.group(1).lower()]
        if heading is None:
            return f"head {m.group(1).lower()}"
        w = _turn_word((target - heading + 180) % 360 - 180)
        return f"{w} now" if w else "continue straight"
    m = _INSTR_TURN.search(text)
    if m:
        mod = (m.group(1) or "").lower()
        verb = "bear" if mod == "slight" else ("turn sharp" if mod == "sharp" else "turn")
        return f"{verb} {m.group(2).lower()} {when}"
    return None


# ---- class priors -----------------------------------------------------------
# Open Images detector labels (veris/reflex.py speaks these).

def class_priors(situation: str, near: dict) -> dict[str, float]:
    p: dict[str, float] = {}

    def up(label, w):
        p[label] = max(p.get(label, 1.0), w)

    if situation == "street_crossing" or near.get("crossing", 1e9) < 30:
        up("Traffic light", 3.0); up("Car", 2.5)
    if situation == "transit_stop" or near.get("bus_stop", 1e9) < 20:
        up("Bus", 3.0); up("Bench", 2.0)
    if near.get("steps", 1e9) < 30:
        up("Stairs", 3.0)
    if near.get("entrance", 1e9) < 20 or situation in ("restaurant", "retail_store", "office_building"):
        up("Door", 2.5)
    if situation == "parking_lot":
        up("Car", 3.0)
    if situation == "restaurant":
        up("Chair", 2.0); up("Table", 2.0)
    if situation == "street_sidewalk":
        up("Car", 1.5); up("Bicycle", 1.5)
    return p


# ---- planner ----------------------------------------------------------------

ACC_OPEN_M, ACC_INDOOR_M = 8.0, 25.0     # accuracy ramp: open sky .. indoors
INDOOR_ON, INDOOR_OFF = 0.6, 0.35        # hysteresis on the leaky indoor score
INDOOR_ALPHA = 0.25                      # leak per fix (time constant ~3.5 s)
VENUE_R_M = 25.0                         # composer: "a venue within ~25 m"
CROSSING_R_M = 12.0
BUS_WAIT_S = 40                          # still beside a stop this long beats a nearby crossing
SPEED_TRUST_ACC_M = 20.0                 # GNSS speed from a worse fix is noise
UNTRUSTED_HOLD_S = 10


def _motion_state(speed, prev):
    """Composer's dead band: >0.6 moving, <0.3 still, hold in between."""
    if speed is None:
        return prev or "still"
    if speed > 0.6:
        return "moving"
    if speed < 0.3:
        return "still"
    return prev or "still"


class Planner:
    def __init__(self, index_path=INDEX, index: dict | None = None):
        ix = index if index is not None else load_index(index_path)
        self.ix = ix
        self._crossing_ll = np.array([(p["lat"], p["lon"]) for p in ix["pois"] if "crossing" in p["types"]])
        named = [i for i, n in enumerate(ix["edge_name"]) if n]
        seg = ix["edges"][named]
        self._street_a, self._street_b = ix["node_ll"][seg[:, 0]], ix["node_ll"][seg[:, 1]]
        self._street_name = [ix["edge_name"][i] for i in named]
        self._park_box = np.array([(*p.min(0), *p.max(0)) for p in ix["parking"]]).reshape(-1, 4)
        self.indoor_score = 0.0
        self.indoor = False
        self.motion = None
        self._still_s = 0                    # seconds (plan calls) spent still
        self._untrusted = 0                  # consecutive fixes too poor to trust speed
        self._venue: dict | None = None      # adopted while still; dropped on walking away
        self._last = None                    # (lat, lon) of the last fix
        self._heading = None
        self._prev_key = None
        self.changed = True                  # material change -> worth an LLM polish call
        self.last_facts: dict = {}
        self.last: Situation = Situation()

    # -- map queries --
    def _nearby(self, lat, lon, r=120.0):
        ix = self.ix
        if not len(ix["poi_ll"]):
            return []
        xy = to_xy(ix["poi_ll"], lat, lon)
        d = np.hypot(*xy.T)
        out = []
        for i in np.flatnonzero(d < r)[np.argsort(d[d < r])]:
            p = ix["pois"][i]
            out.append({**{k: p[k] for k in ("name", "types")}, "distance_m": float(d[i]),
                        "bearing_deg": bearing(*xy[i])})
        return out

    def _building_at(self, lat, lon):
        box = self.ix["bld_box"]
        if not len(box):
            return None
        m = np.flatnonzero((box[:, 0] <= lat) & (box[:, 2] >= lat) & (box[:, 1] <= lon) & (box[:, 3] >= lon))
        for i in m:
            b = self.ix["buildings"][i]
            if point_in_poly(0, 0, to_xy(b["poly"], lat, lon)):
                return b
        return None

    def _street(self, lat, lon):
        # prefilter to ~300 m, then exact point-to-segment distance
        m = np.flatnonzero((np.abs(self._street_a[:, 0] - lat) < 0.003) &
                           (np.abs(self._street_a[:, 1] - lon) < 0.004))
        if not len(m):
            return None, 1e9
        d, _ = seg_dist(np.zeros(2), to_xy(self._street_a[m], lat, lon), to_xy(self._street_b[m], lat, lon))
        i = int(np.argmin(d))
        return self._street_name[m[i]], float(d[i])

    def _in_parking(self, lat, lon):
        box = self._park_box
        m = np.flatnonzero((box[:, 0] <= lat) & (box[:, 2] >= lat) & (box[:, 1] <= lon) & (box[:, 3] >= lon))
        return any(point_in_poly(0, 0, to_xy(self.ix["parking"][i], lat, lon)) for i in m)

    # -- the rules path --
    def plan(self, fix: dict | None, route=None) -> Situation:
        if fix is None or fix.get("lat") is None:
            # no fix this second: GNSS dropout is indoor evidence
            self._still_s += self.motion != "moving"
            self.indoor_score += INDOOR_ALPHA * (1.0 - self.indoor_score)
            if self.indoor_score > INDOOR_ON and not self.indoor:
                self.indoor = True
                if self._last is not None:
                    return self._decide(*self._last, None, None, route)
            return self.last
        lat, lon = fix["lat"], fix["lon"]
        acc = fix.get("accuracy_m") or 10.0
        speed = fix.get("speed_mps")
        heading = fix.get("heading_deg")
        self._heading = heading
        prev_motion = self.motion
        trusted = acc <= SPEED_TRUST_ACC_M
        self._untrusted = 0 if trusted else self._untrusted + 1
        # a poor fix's speed is noise: hold the motion state, and after
        # UNTRUSTED_HOLD_S of poor fixes assume still (the IMU knows better)
        self.motion = _motion_state(speed if trusted else (None if self._untrusted < UNTRUSTED_HOLD_S else 0.0),
                                    prev_motion)
        if prev_motion == "still" and self.motion == "moving":
            self._venue = None                    # walking away invalidates a stationary venue
        self._still_s = 0 if self.motion == "moving" else self._still_s + 1
        x = min(1.0, max(0.0, (acc - ACC_OPEN_M) / (ACC_INDOOR_M - ACC_OPEN_M)))
        bld = self._building_at(lat, lon)
        if bld is not None and self.motion == "still":
            x = min(1.0, x + 0.3)
        self.indoor_score += INDOOR_ALPHA * (x - self.indoor_score)
        if self.indoor_score > INDOOR_ON:
            self.indoor = True
        elif self.indoor_score < INDOOR_OFF:
            self.indoor = False
        self._last = (lat, lon)
        return self._decide(lat, lon, heading, speed, route, bld)

    def _decide(self, lat, lon, heading, speed, route, bld=None) -> Situation:
        nearby = self._nearby(lat, lon)
        near = {}
        for p in nearby:
            for t in p["types"]:
                near.setdefault(t, p["distance_m"])
        venues = [p for p in nearby if p.get("name") and venue_situation(p["types"])]
        still = self.motion != "moving"
        street, street_d = self._street(lat, lon)
        venue = None
        if self.indoor:
            if self._venue is not None:
                venue = self._venue
            else:
                # the fix wanders 20-50 m indoors; a venue in the footprint wins,
                # else the nearest named venue within indoor error
                if bld is None:
                    bld = self._building_at(lat, lon)
                inside = [v for v in venues if v["distance_m"] < 40]
                venue = inside[0] if inside else (
                    {"name": bld["name"], "types": bld["types"], "distance_m": 0.0, "bearing_deg": 0.0}
                    if bld is not None else None)
            situation = (venue_situation(venue["types"]) if venue else None) or "office_building"
            mode = "describe" if still else "indoor_nav"
        else:
            close = [v for v in venues if v["distance_m"] <= VENUE_R_M]
            if still and close and self.indoor_score > 0.3:
                # composer heuristic: stopped by a venue with weakening GNSS
                venue = close[0]
                situation, mode = venue_situation(venue["types"]), "indoor_nav"
            elif still and near.get("bus_stop", 1e9) <= 20 and self._still_s >= BUS_WAIT_S:
                # waiting longer than any signal cycle beside a stop: a bus wait
                situation, mode = "transit_stop", "outdoor_nav"
            elif near.get("crossing", 1e9) <= CROSSING_R_M:
                situation, mode = "street_crossing", "outdoor_nav"
            elif still and near.get("bus_stop", 1e9) <= 20:
                situation, mode = "transit_stop", "outdoor_nav"
            elif self._in_parking(lat, lon):
                situation, mode = "parking_lot", "outdoor_nav"
            elif street_d < 150 or near:
                situation, mode = "street_sidewalk", "outdoor_nav"
            else:
                situation, mode = "unknown", "describe"
        if venue is not None and still:
            self._venue = venue
        look = self._look_for(situation)
        hint = route_hint(lat, lon, heading, route, self._crossing_ll) if route else None
        text = self._text(situation, venue, street, nearby, heading, speed, still)
        sit = Situation(situation=situation, suggested_mode=mode, look_for=tuple(look),
                        context_text=ensure_expectation_phrasing(text, look),
                        source="offline_rules", class_priors=class_priors(situation, near),
                        route_hint=hint)
        key = (situation, mode, self.motion, street, round(lat, 3), round(lon, 3))
        self.changed, self._prev_key = key != self._prev_key, key
        self.last_facts = {"street": street, "indoor": self.indoor,
                           "indoor_score": round(self.indoor_score, 2), "motion": self.motion,
                           "heading_deg": heading, "speed_mps": speed,
                           "venue": venue and venue.get("name"),
                           "nearby": [{"name": p["name"], "types": p["types"][:3],
                                       "distance_m": round(p["distance_m"]),
                                       "side": relative_side(p["bearing_deg"], heading)}
                                      for p in nearby[:8]],
                           "route_hint": hint, "rules": {"situation": situation, "suggested_mode": mode}}
        self.last = sit
        return sit

    @staticmethod
    def _look_for(situation):
        for _, sit, _, look in _SITUATION_RULES:
            if sit == situation:
                return list(look)
        if situation == "street_sidewalk":
            return list(_STREET_LOOK)
        return list(_EXTRA_LOOK.get(situation, []))

    @staticmethod
    def _text(situation, venue, street, nearby, heading, speed, still) -> str:
        if venue is not None:
            name = venue.get("name") or "a building"
            doing = "standing or seated" if still else "walking"
            return f"You are probably inside {name}, {doing}."
        bits = []
        if street:
            head = f", heading {compass(heading)}" if heading is not None else ""
            bits.append(f"You are {'currently stopped' if still else 'walking'} near {street}{head}.")
        cross = [p for p in nearby if "crossing" in p["types"]]
        if cross:
            c = cross[0]
            bits.append(f"A pedestrian crossing is about {round(c['distance_m'])} m "
                        f"{relative_side(c['bearing_deg'], heading)}.")
        pois = [p for p in nearby if p.get("name") and "crossing" not in p["types"]][:3]
        if pois:
            bits.append("Nearby: " + "; ".join(
                f"{p['name']} about {round(p['distance_m'])} m {relative_side(p['bearing_deg'], heading)}"
                for p in pois) + ".")
        return " ".join(bits) or "No map information for this spot."


# ---- optional on-device LLM polish -------------------------------------------

def llm_prompt(facts: dict) -> str:
    """Prompt for the 3B base in text-only mode. `facts` is Planner.last_facts."""
    return (
        "You are the situational planner of a wearable for a blind pedestrian. "
        "Below are map facts from an offline map and the phone's GPS. Reply with "
        "ONE JSON object and nothing else, with keys:\n"
        f'"situation": one of {list(SITUATIONS)};\n'
        f'"suggested_mode": one of {list(MODES)} (moving outdoors -> outdoor_nav; '
        "moving inside -> indoor_nav; stationary inside -> describe; stationary at "
        "signs or a menu -> text_reading);\n"
        '"look_for": up to 4 short noun phrases the camera should mention first IF VISIBLE '
        "(expectations, not facts);\n"
        '"context_text": at most 3 sentences, second person. Where the user is, the '
        "nearest 1-2 places with their side, and expectations phrased only as "
        '"If X is visible, mention it first". Never invent places not in the facts;\n'
        '"class_priors": object mapping detector classes (e.g. "Traffic light", "Car", '
        '"Bus", "Bench", "Stairs", "Door") to weights between 0.5 and 5.\n'
        "The rules engine already proposed facts.rules; keep it unless the facts "
        "clearly say otherwise.\n"
        "facts: " + json.dumps(facts, separators=(",", ":"), default=str))


def parse_llm(text: str) -> Situation | None:
    """Strict: one JSON object, every field validated, else None (keep rules)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        d = json.loads(t)
    except (ValueError, TypeError):
        return None
    if not isinstance(d, dict):
        return None
    sit, mode = d.get("situation"), d.get("suggested_mode")
    look, ctx = d.get("look_for", []), d.get("context_text", "")
    pri = d.get("class_priors", {})
    if sit not in SITUATIONS or mode not in MODES or not isinstance(ctx, str) or len(ctx) > 600:
        return None
    if not isinstance(look, list) or len(look) > 4 or not all(isinstance(x, str) and 0 < len(x) <= 60 for x in look):
        return None
    if not isinstance(pri, dict) or not all(
            isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool)
            and 0.25 <= v <= 5 for k, v in pri.items()):
        return None
    return Situation(situation=sit, suggested_mode=mode, look_for=tuple(look),
                     context_text=ensure_expectation_phrasing(ctx.strip(), look),
                     source="offline_llm", class_priors={k: float(v) for k, v in pri.items()})


# ---- location questions: the second LLM's job --------------------------------
# Measured on the board (evals/results/planner_llm_*.json): asked to re-decide
# situation and mode, the 3B base is slower and less accurate than the rules
# above. Asked where the nearest X is, it is right 93% of the time in about a
# second. So rules decide, and the model answers questions rules cannot parse.

NOT_ON_MAP = "Not on the map nearby."
_SIDE_WORD = {"on your left": "left", "on your right": "right", "ahead": "ahead",
              "behind you": "behind"}
_METRES = re.compile(r"\b(\d+)\s*(?:m\b|meters?|metres?)")


def _fact_label(p: dict) -> str:
    kind = p["types"][0].replace("_", " ")
    return f"{p['name']} ({kind})" if p.get("name") else kind


def facts_text(facts: dict) -> str:
    """Planner.last_facts as one compact line: fewer tokens than JSON."""
    venue = facts.get("venue") or {}
    where = f"inside {venue['name']}" if venue.get("name") \
        else f"on {facts['street']}" if facts.get("street") else "at an unnamed spot"
    near = "; ".join(f"{_fact_label(p)} {p['distance_m']} m {p['side']}"
                     for p in facts["nearby"])
    return f"You are {where}, {facts['motion']}. Nearby: {near or 'nothing mapped'}."


def locate_prompt(facts: dict, question: str) -> str:
    return f"Map facts: {facts_text(facts)}\nQuestion: {question}"


def locate_guard(answer: str, facts: dict, question: str = "") -> str:
    """Let an answer through only if the distance and side it gives belong to a
    mapped place that the answer names or the question asked for. Anything
    else becomes a plain 'not on the map', so a borrowed distance is never
    spoken."""
    t, q = answer.lower(), question.lower()
    m = _METRES.search(t)
    if not m:
        return answer                      # a refusal, or nothing checkable
    d = int(m.group(1))
    for p in facts["nearby"]:
        if p["distance_m"] == d and _SIDE_WORD[p["side"]] in t:
            kind = p["types"][0].replace("_", " ")
            name = (p.get("name") or "").lower()
            if kind in t or kind in q or (name and (name in t or name in q)):
                return answer
    return NOT_ON_MAP
