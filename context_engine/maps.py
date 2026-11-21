"""Maps enrichment: GPS fix -> structured place facts.

Degradation chain (each step only on failure of the previous):
  1. Google Maps Platform (reverse geocoding + Places Nearby) — user has credit
  2. OpenStreetMap (Nominatim + Overpass), keyless
  3. bare PlaceFacts(source="none") — the composer's rules path still yields a
     usable (if empty) context string, so the pipeline never 500s on maps.

All HTTP goes through `_fetch`, which applies a timeout plus bounded retries
with exponential backoff on transient failures (network errors, timeouts,
HTTP 5xx / 429). Successful lookups are cached by ~11 m grid cell with a TTL
so a stationary user doesn't re-bill Google every fix.

Env knobs (all optional):
  CTX_HTTP_RETRIES       extra attempts after the first (default 2)
  CTX_HTTP_BACKOFF_S     base backoff, doubles per retry (default 0.5)
  CTX_MAPS_CACHE_TTL_S   lookup cache TTL in seconds (default 120)
"""

import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict

UA = {"User-Agent": "veris-context-engine/0.1 (assistive device prototype)"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def google_key() -> str:
    return os.environ.get("GOOGLE_MAPS_API_KEY", "")


@dataclass
class PlaceFacts:
    lat: float
    lon: float
    street: str | None = None
    locality: str | None = None
    nearby: list[dict] = field(default_factory=list)  # {name, types, distance_m, bearing_deg}
    source: str = "none"

    def to_dict(self):
        return asdict(self)


def _fetch(req: urllib.request.Request, timeout: float,
           retries: int | None = None, backoff_s: float | None = None):
    """urlopen + json with bounded retries and exponential backoff.

    Retries transient failures only: network errors, timeouts, HTTP 5xx and
    429. Other 4xx (bad key, bad request) fail immediately — retrying can't
    heal them and just burns time.
    """
    retries = _env_int("CTX_HTTP_RETRIES", 2) if retries is None else retries
    backoff_s = _env_float("CTX_HTTP_BACKOFF_S", 0.5) if backoff_s is None else backoff_s
    last_err: Exception = RuntimeError("unreachable")
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code < 500 and e.code != 429:
                raise
        except Exception as e:  # URLError, socket timeout, malformed JSON
            last_err = e
        if attempt < retries:
            time.sleep(backoff_s * (2 ** attempt))
    raise last_err


def _get(url: str, timeout: float = 8):
    return _fetch(urllib.request.Request(url, headers=UA), timeout)


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def lookup_google(lat: float, lon: float, radius_m: int = 120) -> PlaceFacts:
    key = google_key()
    facts = PlaceFacts(lat=lat, lon=lon, source="google")
    geo = _get(
        "https://maps.googleapis.com/maps/api/geocode/json?"
        + urllib.parse.urlencode({"latlng": f"{lat},{lon}", "key": key})
    )
    if geo.get("results"):
        comps = geo["results"][0].get("address_components", [])
        for c in comps:
            if "route" in c["types"]:
                facts.street = c["long_name"]
            if "locality" in c["types"] or "sublocality" in c["types"]:
                facts.locality = facts.locality or c["long_name"]
    places = _get(
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
        + urllib.parse.urlencode(
            {"location": f"{lat},{lon}", "radius": radius_m, "key": key}
        )
    )
    for p in places.get("results", []):
        ptypes = [t for t in p.get("types", []) if t not in ("point_of_interest", "establishment")]
        loc = p["geometry"]["location"]
        facts.nearby.append({
            "name": p.get("name"),
            "types": ptypes[:3],
            "distance_m": round(_haversine_m(lat, lon, loc["lat"], loc["lng"])),
            "bearing_deg": round(_bearing_deg(lat, lon, loc["lat"], loc["lng"])),
        })
    facts.nearby.sort(key=lambda x: x["distance_m"])
    facts.nearby = facts.nearby[:8]
    return facts


def lookup_osm(lat: float, lon: float, radius_m: int = 120) -> PlaceFacts:
    facts = PlaceFacts(lat=lat, lon=lon, source="osm")
    rev = _get(
        "https://nominatim.openstreetmap.org/reverse?"
        + urllib.parse.urlencode({"lat": lat, "lon": lon, "format": "jsonv2"})
    )
    addr = rev.get("address", {})
    facts.street = addr.get("road")
    facts.locality = addr.get("town") or addr.get("city") or addr.get("suburb")
    q = f"""[out:json][timeout:8];
(node(around:{radius_m},{lat},{lon})[amenity];
 node(around:{radius_m},{lat},{lon})[shop];
 node(around:{radius_m},{lat},{lon})[highway=crossing];);
out body 20;"""
    try:
        req = urllib.request.Request(
            "https://overpass-api.de/api/interpreter",
            data=urllib.parse.urlencode({"data": q}).encode(),
            headers=UA,
        )
        elems = _fetch(req, timeout=12).get("elements", [])
        for e in elems:
            tags = e.get("tags", {})
            name = tags.get("name") or tags.get("amenity") or tags.get("shop") or tags.get("highway")
            facts.nearby.append({
                "name": name,
                "types": [t for t in (tags.get("amenity"), tags.get("shop"), tags.get("highway")) if t],
                "distance_m": round(_haversine_m(lat, lon, e["lat"], e["lon"])),
                "bearing_deg": round(_bearing_deg(lat, lon, e["lat"], e["lon"])),
            })
        facts.nearby.sort(key=lambda x: x["distance_m"])
        facts.nearby = facts.nearby[:8]
    except Exception:
        pass  # Overpass is flaky; street/locality alone is still useful
    return facts


# ---- lookup cache -----------------------------------------------------------
# Keyed by ~11 m grid cell (4 decimal places). A stationary or slow user emits
# many fixes inside one cell; only the first should hit the paid APIs.

_CACHE: dict[tuple, tuple[float, PlaceFacts]] = {}
_CACHE_MAX = 512


def _cache_key(lat: float, lon: float) -> tuple:
    return (round(lat, 4), round(lon, 4))


def clear_cache():
    _CACHE.clear()


def lookup(lat: float, lon: float, now=time.time) -> PlaceFacts:
    ttl = _env_float("CTX_MAPS_CACHE_TTL_S", 120.0)
    key = _cache_key(lat, lon)
    hit = _CACHE.get(key)
    if hit is not None and now() - hit[0] < ttl:
        return hit[1]
    facts: PlaceFacts | None = None
    if google_key():
        try:
            facts = lookup_google(lat, lon)
        except Exception:
            facts = None
    if facts is None:
        try:
            facts = lookup_osm(lat, lon)
        except Exception:
            facts = PlaceFacts(lat=lat, lon=lon, source="none")
    if facts.source != "none":  # don't cache total failure; retry next fix
        if len(_CACHE) >= _CACHE_MAX:
            oldest = min(_CACHE, key=lambda k: _CACHE[k][0])
            del _CACHE[oldest]
        _CACHE[key] = (now(), facts)
    return facts
