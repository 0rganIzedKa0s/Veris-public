# Context engine — cloud deployment notes

The context engine (`context_engine/`) is the cloud half of the dual-LLM
architecture: phone GPS fix in → maps enrichment → compact context pre-prompt
out, consumed verbatim by the pack VLM's `--context` slot. It is a small
FastAPI app with **no heavyweight dependencies** (fastapi + uvicorn only; all
outbound HTTP is stdlib), so it deploys anywhere a container runs.

## API

| Endpoint | Body | Purpose |
|---|---|---|
| `POST /context` | `{lat, lon, heading_deg?, speed_mps?, timestamp?}` | GPS fix → `{context, situation, situation_source, suggested_mode, look_for, composed_by, facts}` |
| `POST /vision_vote` | `{scene_label, confidence (0-1), ts?}` | Pack VLM reports what the camera saw; fresh votes confirm/override the GPS venue heuristic in later `/context` calls |
| `GET /health` | — | `{status, llm, llm_calls_today, llm_daily_cap}` |

## Build & run (local or any container host)

```sh
# from the repo root
docker build -f context_engine/Dockerfile -t veris-context-engine .
docker run --rm -p 8090:8090 --env-file .env veris-context-engine
curl -s localhost:8090/health
```

For a cloud host (Fly.io / Cloud Run / a $5 VPS): push the image, set the two
API keys as secrets, point the phone app at `https://<host>/context`. One
instance is plenty — state is per-process (composition cache, vote store,
spend counter) and the phone is a single client. If you ever scale out,
pin a client to one instance (session affinity) or move the vote store to
Redis first.

## Environment

Required (either; both degrade gracefully if absent):

- `OPENAI_API_KEY` — gpt-4o-mini phrasing. Absent/over-cap → deterministic
  rules composer.
- `GOOGLE_MAPS_API_KEY` — primary maps. Absent/failing → OSM
  (Nominatim + Overpass, keyless) → bare facts. The service never 500s on
  upstream failure.

Optional knobs (defaults in parentheses):

- `CTX_HTTP_RETRIES` (2), `CTX_HTTP_BACKOFF_S` (0.5) — maps retry policy;
  transient failures only (5xx/429/network), exponential backoff.
- `CTX_MAPS_CACHE_TTL_S` (120) — lookup cache per ~11 m grid cell.
- `CTX_OPENAI_DAILY_CAP` (300), `CTX_OPENAI_TIMEOUT_S` (15) — daily spend
  guard: hard stop past the cap, rules fallback, resets at UTC midnight.
  300 calls/day ≈ well under $0.05/day at gpt-4o-mini prices.
- `CTX_VISION_VOTE_TTL_S` (60), `CTX_VISION_MIN_CONF` (0.4),
  `CTX_VISION_OVERRIDE_CONF` (0.7) — vision-vote freshness and thresholds.

## Cost model

- Google: 1 geocode + 1 places-nearby per *uncached* fix; the ~11 m/TTL cache
  means a stationary user costs one lookup per 2 minutes.
- OpenAI: composition re-runs only on material change (>40 m move, street
  change, >60° heading swing) and is capped daily. Everything else is served
  from the composition cache.

## Tests

```sh
uv run --with pytest --with httpx python -m pytest tests/test_context_engine.py -q
```

All external HTTP is mocked; no keys or network needed.
