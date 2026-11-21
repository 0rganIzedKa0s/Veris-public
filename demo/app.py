"""Veris demo hub — serves the demo UI and relays frames to the Orin.

  uv run uvicorn demo.app:app --host 0.0.0.0 --port 8000

Open http://localhost:8000 on the dev box (USB cam) or, once Tailscale is up,
https://<orin-or-devbox-tailnet-name> from the phone. getUserMedia needs a
secure context: localhost counts; for LAN-phone testing without Tailscale use
`--ssl-keyfile/--ssl-certfile` with mkcert certs.
"""

import base64
import json
import os
import pathlib
import subprocess
import threading
import time
import urllib.request
import uuid

from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# .env for context-engine proxying
_env = pathlib.Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from veris.prompts import MODES, build_messages  # noqa: E402
from demo import local_store  # noqa: E402  (SQLite on the Orin NVMe; VERIS_DB=off disables)

ORIN = os.environ.get("ORIN_URL", "http://192.168.55.1:8080")
ORIN_SSH = os.environ.get("ORIN_SSH", "user@192.168.55.1")
CTX_ENGINE = os.environ.get("CTX_ENGINE_URL", "http://localhost:8090")
ROOT = pathlib.Path(__file__).resolve().parent
SESSIONS = ROOT / "sessions"
SESSIONS.mkdir(exist_ok=True)
SESSION_LOG = SESSIONS / f"session-{int(time.time())}.jsonl"


def _append_session_record(rec: dict) -> None:
    """Append one JSONL record with a single O_APPEND write so concurrent
    describes (auto-timer overlap) can't interleave partial lines."""
    line = (json.dumps(rec) + "\n").encode()
    fd = os.open(SESSION_LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)
    local_store.push_record(rec)  # best-effort SQLite index; JSONL is truth

MODELS = {
    "3b": {"gguf": "~/models/qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
           "mmproj": "~/models/qwen2.5-vl-3b/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
           "extra": "-ngl 99 -c 4096", "min_power": "any"},
    "7b": {"gguf": "~/models/qwen2.5-vl-7b/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
           "mmproj": "~/models/qwen2.5-vl-7b/mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf",
           "extra": "-c 2048", "min_power": "25W+ only; needs reboot-fresh memory"},
}

app = FastAPI(title="veris-demo-hub")

# ── The multi-model pipeline (VERIS_PIPELINE=multi) ──────────────────────
# With it on, a frame goes through gate -> detector -> reader or expert
# (demo/pipeline.py) and llama-server holds the stock base plus three
# adapters. Off (the default), every mode is one prompt on Veris-3B-v1, as
# before. "general" is the untuned prompt on whatever base is loaded.
PIPELINE = os.environ.get("VERIS_PIPELINE", "") == "multi"
_pipe = None
_pipe_lock = threading.Lock()


def _pipeline():
    global _pipe
    with _pipe_lock:
        if _pipe is None:
            from demo.pipeline import Pipeline
            _pipe = Pipeline(ORIN)
        return _pipe

# ── Public access and throttling ─────────────────────────────────────────
# The demo is deliberately ungated. The link is printed on a resume and has to
# open for anyone, on any device, with no code to type. Links handed out
# earlier carry a `?k=...` query string from when a code WAS required —
# unknown query params are ignored, so every one of those keeps working.
#
# Throttling is what replaces the gate, because an open URL points at (a) one
# 15 W board whose llama-server runs a SINGLE slot and (b) metered Google Maps
# and OpenAI calls in the context engine. Limits are per client IP, and also
# global for the paid path so a distributed crawl cannot outrun a per-IP cap.
_LIMITS: dict[str, list[tuple[int, int, int]]] = {
    # class    (window_s, per_ip, global)
    "model":  [(60, 10, 40), (3600, 120, 500)],       # GPU: one slot, 4-15 s each
    "paid":   [(60, 5, 15), (3600, 30, 120), (86400, 120, 500)],  # Maps + OpenAI
    "asr":    [(60, 6, 20), (3600, 60, 240)],         # ssh + whisper on the CPU
    "upload": [(60, 20, 60), (3600, 300, 900)],       # disk only, no model call
    "write":  [(60, 30, 150)],
    # Each open tab polls status+telemetry every 5s (24/min), so a visitor
    # with three tabs must not trip this. Cheap to allow: these are served
    # from the cache below, so the board pays once regardless.
    "poll":   [(60, 150, 3000)],
    "page":   [(60, 90, 1800)],
}
_PATH_CLASS = {
    "/api/describe": "model", "/api/capture": "upload", "/api/context": "paid",
    "/api/transcribe": "asr", "/api/voice_ask": "asr",
    "/api/rate": "write", "/api/status": "poll", "/api/telemetry": "poll",
    "/api/live": "poll",
}
_MAX_KEYS = 4096  # bound the table; a spoofed-XFF flood must not eat memory

_hits: dict[tuple[str, str], list[float]] = {}
_hits_lock = threading.Lock()


def _client_key(request: Request) -> str:
    """Identify the caller for rate-limiting.

    Behind Tailscale Serve/Funnel the hub only ever sees 127.0.0.1 as the peer,
    so without X-Forwarded-For every visitor on Earth would share one bucket.
    XFF is trusted ONLY when the immediate peer is loopback (i.e. it came from
    tailscaled on this host); from any other peer it is attacker-controlled.
    """
    peer = request.client.host if request.client else "unknown"
    if peer in ("127.0.0.1", "::1"):
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[0].strip()[:64]
    return peer


def _check_limits(cls: str, key: str, now: float) -> int:
    """Return 0 if allowed, else the seconds to wait. Records the hit."""
    windows = _LIMITS[cls]
    longest = max(w for w, _, _ in windows)
    with _hits_lock:
        if len(_hits) > _MAX_KEYS:  # drop buckets with nothing recent in them
            for k in [k for k, v in _hits.items() if not v or now - v[-1] > longest]:
                _hits.pop(k, None)
            if len(_hits) > _MAX_KEYS:
                _hits.clear()
        mine = _hits.setdefault((cls, key), [])
        everyone = _hits.setdefault((cls, "*"), [])
        for lst in (mine, everyone):
            cut = now - longest
            while lst and lst[0] < cut:
                lst.pop(0)
        for window, per_ip, per_all in windows:
            edge = now - window
            if sum(1 for t in mine if t >= edge) >= per_ip:
                return max(1, int(window - (now - next(t for t in mine if t >= edge))))
            if sum(1 for t in everyone if t >= edge) >= per_all:
                return max(1, int(window - (now - next(t for t in everyone if t >= edge))))
        mine.append(now)
        everyone.append(now)
    return 0


@app.middleware("http")
async def throttle(request: Request, call_next):
    cls = _PATH_CLASS.get(request.url.path, "page")
    wait = _check_limits(cls, _client_key(request), time.time())
    if wait:
        return JSONResponse(
            {"error": "Rate limit reached. This demo runs on a single Jetson "
                      f"Orin Nano, so it is throttled. Try again in {wait}s.",
             "retry_after": wait},
            status_code=429, headers={"Retry-After": str(wait)})
    return await call_next(request)


# ── The board's single model slot ────────────────────────────────────────
# llama-server runs with -np 1: two overlapping describes do not go faster,
# they queue inside the server and both testers watch a stalled stream. Refuse
# the second one immediately with an honest message instead. The timestamp
# guard releases a slot whose stream died without running its finally (a torn
# TCP connection), so the demo can never wedge itself shut.
_SLOT_STALE_S = 180.0
_slot_taken_at: float | None = None
_slot_lock = threading.Lock()


def _take_slot() -> bool:
    global _slot_taken_at
    with _slot_lock:
        now = time.time()
        if _slot_taken_at is not None and now - _slot_taken_at < _SLOT_STALE_S:
            return False
        _slot_taken_at = now
        return True


def _release_slot() -> None:
    global _slot_taken_at
    with _slot_lock:
        _slot_taken_at = None


# ── Cold-start warm-up for llama-server ──────────────────────────────────
# The board's llama-server build (b9870, 2026-07-03) segfaults if the FIRST
# request a fresh instance processes carries an image — reproduced 5 of 5
# times on 2026-09-02 after the move; send any text-only completion first and
# every image after it works. Systemd restarts the server in 15 s and the next
# visitor's frame kills it again, so after a power cycle the public demo would
# sit in a crash loop until someone happened to send plain text. Every image
# reaches the server through this hub, so the hub warms each new instance
# itself: one four-token text completion before the first frame it sends.
#
# The instance is identified by its systemd MainPID when the hub runs on the
# board (`systemctl show` needs no sudo). Where that is unavailable (the dev
# box driving a remote board) the hub warms once per hub process, and again
# after any describe that came back with no tokens — the crash signature — so
# a restart costs at most one lost describe instead of a loop.
_LLM_UNIT = os.environ.get("VERIS_LLM_UNIT", "veris-llm")
_warm_lock = threading.Lock()
_warmed_instance: str | None = None  # MainPID last warmed, or "hub" in fallback mode


def _llm_instance() -> str | None:
    """MainPID of the llama-server unit, or None if systemd can't tell us."""
    try:
        out = subprocess.run(["systemctl", "show", "-p", "MainPID", "--value", _LLM_UNIT],
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return None
    return out if out.isdigit() and out != "0" else None


def _warm_text_completion() -> bool:
    req = urllib.request.Request(
        f"{ORIN}/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": "Say ready."}],
                         "max_tokens": 4, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            r.read()
        return True
    except Exception:
        return False


def _ensure_llm_warm(instance_of=None, warm=None) -> bool:
    """Warm the running llama-server instance once before it sees an image.
    Returns False only when the warm-up request itself failed."""
    global _warmed_instance
    instance_of = instance_of or _llm_instance
    warm = warm or _warm_text_completion
    with _warm_lock:
        inst = instance_of() or "hub"
        if _warmed_instance == inst:
            return True
        if warm():
            _warmed_instance = inst
            _evict_model_files()
            return True
        return False


def _evict_model_files() -> None:
    """Drop the model files from the page cache once the server has loaded
    them. The board's GPU allocator cannot reclaim page cache: with 2.9 GB of
    it held, free memory is about 100 MB and the second large image kills the
    server with a CUDA out-of-memory error. Only where the hub runs beside
    the server."""
    if "localhost" not in ORIN and "127.0.0.1" not in ORIN:
        return
    for f in pathlib.Path.home().glob("models/**/*.gguf"):
        try:
            fd = os.open(f, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except OSError:
            pass


def _guard_llm_load() -> None:
    """Keep llama-server's load from failing, and catch it when it does.

    The server reads 3 GB of model files as it loads, and the board's GPU
    allocator cannot reclaim page cache. Unless those files are evicted while
    it loads, its last and largest buffer (the vision tower's, 705 MB) fails
    to allocate: the server then reports healthy and dies on its first image.
    So this evicts while the server is not healthy, and when a new instance
    comes up it looks for the failed allocation in that instance's log and
    restarts the unit (the sudoers file allows exactly this)."""
    seen, tries = None, 0
    while True:
        try:
            with urllib.request.urlopen(f"{ORIN}/health", timeout=2) as r:
                ok = b"ok" in r.read()
        except Exception:
            ok = False
        if not ok:
            _evict_model_files()
            time.sleep(0.5)
            continue
        inst = _llm_instance()
        if inst and inst != seen:
            seen = inst
            try:
                log = subprocess.run(["journalctl", f"_PID={inst}", "--no-pager", "-o", "cat"],
                                     capture_output=True, text=True, timeout=30).stdout
            except Exception:
                log = ""
            if "cudaMalloc failed" in log and tries < 5:    # five in a row: something else is wrong
                tries += 1
                print(f"llama-server {inst}: vision buffer failed to allocate, restarting", flush=True)
                subprocess.run(["sudo", "-n", "systemctl", "restart", _LLM_UNIT], timeout=90)
                continue
            tries = 0
        time.sleep(5)


if "localhost" in ORIN or "127.0.0.1" in ORIN:      # the hub runs beside the server
    threading.Thread(target=_guard_llm_load, daemon=True).start()


def _preload_pipeline() -> None:
    """Load the small models once llama-server is up, so the first visitor
    does not wait for them and the language model gets the GPU's large
    blocks first."""
    while not _ensure_llm_warm():
        time.sleep(5)
    _pipeline()


if PIPELINE:
    threading.Thread(target=_preload_pipeline, daemon=True).start()


def _note_llm_crash_signature() -> None:
    """A describe that streamed no tokens is what a mid-request crash looks
    like from here: forget the warm state so the next describe re-warms."""
    global _warmed_instance
    with _warm_lock:
        _warmed_instance = None


# ── Cache for the polled endpoints ───────────────────────────────────────
# /api/status and /api/telemetry are hit every 5 s by EVERY open tab, and
# telemetry costs an ssh + tegrastats round trip. Uncached, ten viewers meant
# ten ssh sessions every five seconds — the board spent more effort reporting
# its own health than describing anything. One computation is shared by all
# callers in the TTL; the per-key lock is what stops a cache miss from firing
# N identical ssh calls at once.
_cache: dict[str, tuple[float, object]] = {}
_cache_locks: dict[str, threading.Lock] = {}
_cache_meta_lock = threading.Lock()


def _cached(key: str, ttl: float, compute):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    with _cache_meta_lock:
        lock = _cache_locks.setdefault(key, threading.Lock())
    with lock:
        hit = _cache.get(key)  # another thread may have filled it while we waited
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
        value = compute()
        _cache[key] = (time.time(), value)
        return value


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

# The Orin's own camera was removed: its capture route (`/api/orin_frame`) and
# the `source=orin` describe/capture path let any anonymous visitor trigger a
# live photo from the board's physical camera and read a description of the
# room it sits in. The camera has been physically disconnected; this code is
# gone so the endpoint can never be reachable again. Visitors send frames from
# their own device camera or pick a bundled sample.


@app.get("/")
def index():
    # no-store: the demo iterates fast and stale cached JS keeps "hiding" new
    # buttons; the page is tiny, always fetch fresh
    return FileResponse(ROOT / "static" / "index.html",
                        headers={"Cache-Control": "no-store"})


@app.get("/api/status")
def status():
    return _cached("status", 4.0, _status_uncached)


# ── Liveness probe for the recorded copy ─────────────────────────────────
# The always-up recorded copy of the demo (GitHub Pages) asks this before it
# tells a visitor the board is reachable and offers the live page. It is the
# hub's ONLY cross-origin endpoint: a GET that reveals nothing the status
# endpoint doesn't, served from the same cache so every open recorded tab
# probing once a minute costs the board nothing extra, and CORS is granted to
# the Pages origin alone. Everything else on the hub stays same-origin.
_LIVE_ORIGINS = frozenset(o.strip() for o in os.environ.get(
    "VERIS_LIVE_ORIGINS", "https://0rganizedka0s.github.io").split(",") if o.strip())


def _live_headers(origin: str | None) -> dict:
    headers = {"Cache-Control": "no-store"}
    if origin in _LIVE_ORIGINS:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Vary"] = "Origin"
    return headers


@app.get("/api/live")
def live(request: Request):
    s = _cached("status", 4.0, _status_uncached)
    body = {"live": True,
            "board": s["orin"].get("status") == "ok",
            "context_engine": s["context_engine"].get("status") == "ok"}
    return JSONResponse(body, headers=_live_headers(request.headers.get("origin")))


def _status_uncached():
    try:
        with urllib.request.urlopen(f"{ORIN}/health", timeout=3) as r:
            orin = json.load(r)
    except Exception as e:
        orin = {"status": f"unreachable: {e}"}
    try:
        with urllib.request.urlopen(f"{CTX_ENGINE}/health", timeout=3) as r:
            ctx = json.load(r)
    except Exception as e:
        ctx = {"status": f"unreachable: {e}"}
    return {"orin": orin, "context_engine": ctx, "modes": sorted(MODES),
            "pipeline": "multi" if PIPELINE else "single", "pipeline_ready": _pipe is not None,
            "models": {k: v["min_power"] for k, v in MODELS.items()},
            "session_log": str(SESSION_LOG)}


@app.post("/api/context")
def get_context(payload: dict):
    """Proxy the browser's geolocation to the context engine."""
    req = urllib.request.Request(f"{CTX_ENGINE}/context",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except Exception as e:
        return JSONResponse({"error": f"context engine unreachable: {e}"},
                            status_code=502)


def _llama_deltas(payload: dict):
    req = urllib.request.Request(f"{ORIN}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            choices = json.loads(line[6:]).get("choices") or [{}]
            delta = choices[0].get("delta", {}).get("content")
            if delta:
                yield delta


def stream_generation(payload: dict | None, rec_base: dict, on_done=None, events=None):
    """SSE generator for a llama-server chat completion, or, with `events`,
    for the pipeline's stream: one trace dict (sent as a `trace` event), then
    (text, source) pairs.

    Module-level (not a closure) so tests can drive it directly. The session
    record is appended in a `finally`: a client disconnect mid-stream closes
    the generator, raising GeneratorExit (a BaseException) at the suspended
    yield — an `except Exception` after the loop would never run, silently
    dropping the record while the frame JPEG stays on disk. Aborted records
    keep the partial output and are marked "aborted": true.
    """
    t0, ttft, text, done, extra = time.time(), None, [], False, {}
    try:
        try:
            for ev in (_llama_deltas(payload) if events is None else events):
                if isinstance(ev, dict):
                    extra["pipeline"] = ev
                    yield f"data: {json.dumps({'trace': ev})}\n\n"
                    continue
                delta = ev
                if isinstance(ev, tuple):
                    delta, src = ev
                    sources = extra.setdefault("sources", [])
                    if src not in sources[-1:]:     # a new tier starts speaking
                        sources.append(src)
                        if text and not text[-1][-1:].isspace():
                            delta = " " + delta
                if ttft is None:
                    ttft = round(time.time() - t0, 2)
                text.append(delta)
                yield f"data: {json.dumps({'delta': delta})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        meta = {"done": True, "ttft_s": ttft, "total_s": round(time.time() - t0, 2),
                "frame": rec_base.get("frame")}
        if "sources" in extra:
            meta["sources"] = extra["sources"]
        yield f"data: {json.dumps(meta)}\n\n"
        done = True
    finally:
        rec = dict(rec_base, output="".join(text), ttft_s=ttft,
                   total_s=round(time.time() - t0, 2), **extra)
        if not done:
            rec["aborted"] = True
        if ttft is None:
            _note_llm_crash_signature()
        _append_session_record(rec)
        if on_done:
            on_done()

@app.post("/api/describe")
async def describe(
    frame: UploadFile | None = None,
    mode: str = Form("describe"),
    context: str = Form(""),
    question: str = Form(""),
    variants: str = Form(""),
    max_tokens: int = Form(0),
):
    # Hold the board's single model slot for the WHOLE request, including the
    # ssh frame grab, then hand the release to the SSE generator once the
    # stream owns the work. `handed_off` is what keeps every early error path
    # from leaking the slot.
    if not _take_slot():
        return JSONResponse(
            {"error": "The board is describing something for another visitor — "
                      "it runs one at a time. Try again in a few seconds.",
             "retry_after": 10},
            status_code=429, headers={"Retry-After": "10"})
    handed_off = False
    try:
        if frame is None:
            return JSONResponse({"error": "no frame uploaded"}, status_code=422)
        data = await frame.read()
        # enforce the per-mode image budget (TTFT scales ~linearly with pixels;
        # nav modes run 896px = ~5s TTFT, text_reading keeps 1280px legibility)
        from io import BytesIO
        from PIL import Image
        multi = PIPELINE and mode != "general"      # "general" stays the untuned baseline
        budget = 1280 if multi else MODES.get(mode, MODES["describe"]).max_image_side
        img = Image.open(BytesIO(data))
        if max(img.size) > budget:
            img.thumbnail((budget, budget))
            buf = BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=88)
            data = buf.getvalue()
        ts = time.time()
        frame_path = SESSIONS / f"frame-{int(ts*1000)}.jpg"
        frame_path.write_bytes(data)
        local_store.push_frame(mode, frame_path.name, data)

        if multi:
            rec_base = {"ts": ts, "frame": frame_path.name, "mode": mode,
                        "context": context, "question": question, "variants": variants}
            _ensure_llm_warm()
            resp = StreamingResponse(
                stream_generation(None, rec_base, on_done=_release_slot,
                                  events=_pipeline().run(data, mode, question, context)),
                media_type="text/event-stream")
            handed_off = True
            return resp

        m = MODES.get(mode, MODES["describe"])
        msgs = build_messages(m, str(frame_path), context or None, question or None,
                              tuple(v for v in variants.split(",") if v))
        b64 = base64.b64encode(data).decode()
        oai = []
        for msg in msgs:
            if isinstance(msg["content"], str):
                oai.append(msg)
            else:
                content = []
                for part in msg["content"]:
                    if part["type"] == "image":
                        content.append({"type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                    else:
                        content.append(part)
                oai.append({"role": msg["role"], "content": content})
        # cache_prompt: llama-server reuses the constant system-prompt KV prefix
        # (~0.9s saved on new frames) and returns <1s on repeated/static scenes
        payload = {"messages": oai, "max_tokens": max_tokens or m.max_new_tokens,
                   "temperature": 0, "stream": True, "cache_prompt": True}
        if PIPELINE:        # every adapter off for this request: the bare base
            payload["lora"] = _pipeline().llm._lora(None)

        rec_base = {"ts": ts, "frame": frame_path.name, "mode": mode,
                    "context": context, "question": question, "variants": variants}
        # A failed warm-up is not fatal here: the stream below reports the
        # connection error the same way it always has.
        _ensure_llm_warm()
        resp = StreamingResponse(
            stream_generation(payload, rec_base, on_done=_release_slot),
            media_type="text/event-stream")
        handed_off = True
        return resp
    finally:
        if not handed_off:
            _release_slot()


@app.post("/api/capture")
async def capture(
    frame: UploadFile | None = None,
    mode: str = Form("outdoor_nav"),
    context: str = Form(""),
):
    """Store one frame as training feedstock — NO model call.

    A walking session exists to harvest nav imagery, and the targets get
    written later by the 7B teacher on the dev box, so running the 3B here
    would burn seconds and battery per frame to produce output nobody reads.
    Skipping it is what makes a walk sustainable at a 4s cadence.

    outdoor_nav/indoor_nav have zero training examples in v1 and v2 (both were
    built from VizWiz, which has no egocentric walking imagery) — this is the
    intake for that gap: v2+ training data must be stratified by mode.
    """
    if frame is None:
        return JSONResponse({"error": "no frame uploaded"}, status_code=422)
    data = await frame.read()

    from io import BytesIO
    from PIL import Image
    budget = MODES.get(mode, MODES["describe"]).max_image_side
    img = Image.open(BytesIO(data))
    if max(img.size) > budget:
        img.thumbnail((budget, budget))
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=88)
        data = buf.getvalue()

    ts = time.time()
    frame_path = SESSIONS / f"frame-{int(ts*1000)}.jpg"
    frame_path.write_bytes(data)
    local_store.push_frame(mode, frame_path.name, data)
    _append_session_record({"ts": ts, "frame": frame_path.name, "mode": mode,
                            "context": context, "kind": "walk"})
    return {"ok": True, "frame": frame_path.name, "bytes": len(data)}


@app.post("/api/rate")
def rate(payload: dict):
    payload["ts"] = time.time()
    payload["kind"] = "rating"
    _append_session_record(payload)
    return {"ok": True}


@app.post("/api/voice_ask")
def voice_ask(payload: dict | None = None):
    """Record from the mic ON THE ORIN, transcribe on its CPU, return the
    transcript. The UI then feeds it to /api/describe as the question —
    full hands-free interaction with the GPU untouched until the VLM call."""
    # Both values reach a shell (bash -c on the board, or ssh). NEVER
    # interpolate the raw request body: `model` is caller-controlled and a
    # value like "base; curl x|sh #" would run arbitrary code as the board
    # user. Whitelist the model and clamp seconds to a small integer.
    try:
        secs = max(1, min(int((payload or {}).get("seconds", 4)), 15))
    except (TypeError, ValueError):
        secs = 4
    model = (payload or {}).get("model", "base")
    if model not in ("tiny", "base", "small"):
        return JSONResponse({"error": "unsupported model"}, status_code=422)
    cmd_str = f"bash ~/veris-app/listen.sh {secs} {model}"
    cmd = ["bash", "-c", cmd_str] if os.environ.get("TELEMETRY_LOCAL") else \
          ["ssh", "-o", "ConnectTimeout=5", ORIN_SSH, cmd_str]
    t0 = time.time()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=secs + 30).stdout.strip()
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "voice capture timed out"}, status_code=502)
    return {"transcript": out, "wall_s": round(time.time() - t0, 2),
            "asr_s": round(time.time() - t0 - secs, 2)}


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile):
    """Phone-mic path: browser records 16kHz mono WAV, we transcribe it with
    whisper.cpp on the Orin's CPU. Audio only ever crosses the encrypted
    tailnet; nothing leaves the device."""
    data = await audio.read()
    t0 = time.time()
    # unique path per request: a fixed /tmp name means a failed upload leaves a
    # STALE wav that whisper happily transcribes, and simultaneous clients
    # clobber each other's audio
    wav = f"/tmp/phone_query-{uuid.uuid4().hex}.wav"
    whisper = ("$HOME/whisper.cpp/build/bin/whisper-cli "
               "-m $HOME/whisper.cpp/models/ggml-tiny.en.bin "
               f"-f {wav} -np -nt -t 4 2>/dev/null")
    try:
        if os.environ.get("TELEMETRY_LOCAL"):
            pathlib.Path(wav).write_bytes(data)
            try:
                out = subprocess.run(["bash", "-c", whisper], capture_output=True,
                                     text=True, timeout=60).stdout
            finally:
                pathlib.Path(wav).unlink(missing_ok=True)
        else:
            up = subprocess.run(["ssh", ORIN_SSH, f"cat > {wav}"], input=data,
                                capture_output=True, timeout=30)
            if up.returncode != 0:
                err = up.stderr.decode(errors="replace").strip()
                return JSONResponse(
                    {"error": "audio upload to Orin failed: "
                              f"{err or f'ssh exit {up.returncode}'}"},
                    status_code=502)
            try:
                out = subprocess.run(["ssh", ORIN_SSH, whisper], capture_output=True,
                                     text=True, timeout=60).stdout
            finally:
                try:  # best-effort cleanup; never mask the transcript
                    subprocess.run(["ssh", ORIN_SSH, f"rm -f {wav}"],
                                   capture_output=True, timeout=15)
                except Exception:
                    pass
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "transcription timed out"}, status_code=502)
    return {"transcript": out.strip().replace("\n", " "),
            "asr_s": round(time.time() - t0, 2)}


@app.get("/api/telemetry")
def telemetry():
    """One tegrastats sample, parsed on the board (no persistent logger needed)."""
    return _cached("telemetry", 5.0, _telemetry_uncached)


def _telemetry_uncached():
    try:
        if os.environ.get("TELEMETRY_LOCAL"):  # hub running ON the Orin
            cmd = ["bash", "-c", "timeout 3 tegrastats --interval 500 | head -1"]
        else:
            cmd = ["ssh", "-o", "ConnectTimeout=4", ORIN_SSH,
                   "timeout 3 tegrastats --interval 500 | head -1"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
        sys.path.insert(0, str(ROOT.parent / "bench"))
        from tegrastats_logger import parse_line
        return parse_line(out) if out else {"error": "no sample"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
