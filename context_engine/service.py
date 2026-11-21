"""Context engine HTTP service — the API a real cloud deployment would expose.

  uv run uvicorn context_engine.service:app --port 8090

POST /context
  {"lat": .., "lon": .., "heading_deg": .., "speed_mps": .., "timestamp": ..}
  -> {"context": "...", "situation": "...", "situation_source": "maps|vision_confirmed|vision_override",
      "composed_by": "llm|rules|llm+cache", "facts": {...}}

POST /vision_vote
  {"scene_label": "restaurant", "confidence": 0.85, "ts": <unix seconds, optional>}
  The pack VLM's read of the scene. Votes stay live for CTX_VISION_VOTE_TTL_S
  (default 60 s) and are blended into subsequent /context responses: a vote
  agreeing with the GPS venue heuristic confirms it; a confident disagreeing
  vote overrides it (the camera beats a 10-30 m GPS error near buildings).

The phone would POST its GPS fix here and relay the returned `context` string
to the pack over BT — the string drops into the VLM's --context slot.
"""

import os
import pathlib

from fastapi import FastAPI
from pydantic import BaseModel, Field

# load .env before importing modules that read env at import time
_env = pathlib.Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from .maps import lookup  # noqa: E402
from .composer import Composer, VisionVoteStore  # noqa: E402

app = FastAPI(title="veris-context-engine")
composer = Composer(use_llm=True)
votes = VisionVoteStore()


class Fix(BaseModel):
    lat: float
    lon: float
    heading_deg: float | None = None
    speed_mps: float | None = None
    timestamp: float | None = None


class Vote(BaseModel):
    scene_label: str = Field(min_length=1, max_length=64)
    confidence: float = Field(ge=0.0, le=1.0)
    ts: float | None = None


@app.post("/context")
def context(fix: Fix):
    facts = lookup(fix.lat, fix.lon)
    result, by = composer.compose(facts, fix.heading_deg, fix.speed_mps,
                                  vision_vote=votes.current())
    return {"context": result["context_text"],
            "situation": result["situation"],
            "situation_source": result["situation_source"],
            "suggested_mode": result["suggested_mode"],
            "look_for": result["look_for"],
            "composed_by": by, "llm_calls_total": composer.llm_calls,
            "facts": facts.to_dict()}


@app.post("/vision_vote")
def vision_vote(vote: Vote):
    v = votes.submit(vote.scene_label, vote.confidence, vote.ts)
    return {"accepted": True, "scene_label": v.scene_label,
            "confidence": v.confidence, "ts": v.ts, "ttl_s": votes.ttl_s}


@app.get("/health")
def health():
    return {"status": "ok", "llm": composer.use_llm,
            "llm_calls_today": composer.guard.calls_today,
            "llm_daily_cap": composer.guard.daily_cap}
