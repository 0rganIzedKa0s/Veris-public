# Veris demo webapp — architecture draft v1

Goal: a webapp served from the dev box (not the Orin) that lets anyone point a
live camera at the world and watch the models **running on the Orin** describe
it — switching model, prompt version, mode, and power profile live, with real
latency/power telemetry. This is both the demo surface for stakeholders and
the data-collection surface for the test base.

## The key simplification: the browser IS the camera adapter

Serve one webapp over HTTPS from the dev box. `getUserMedia()` gives us
whatever camera belongs to the device that opens the page:

- **Phone camera** → open the page on the phone (same WiFi). This doubles as
  the honest head-cam simulation: hold or strap the phone, walk around.
- **USB camera** → open the page in a browser on the dev box itself and pick
  the USB cam from the device dropdown.

No native app, no RTSP, no per-camera plumbing. HTTPS is required for camera
access off-localhost → self-signed cert via `mkcert` (one-time install of its
CA on the phone) or a `--insecure-demo` fallback using Chrome's
`unsafely-treat-insecure-origin-as-secure` flag.

The phone browser can also supply **geolocation + compass heading** via the
Geolocation/DeviceOrientation APIs — a real feed for the context-engine slot
(the "phone GPS" box in the product architecture) with zero extra work.

## Topology

```
[Phone browser]──┐  WebRTC/WS frames + GPS     ┌── SSH ──► model lifecycle
                 ├──► [Dev box: FastAPI hub] ◄─┤
[Dev-box browser]┘        │        ▲           └── HTTP ──► llama-server :8080
   (USB cam)              │        │ SSE tokens      (Orin, USB-net 192.168.55.1)
                          ▼        │                 tegrastats WS ◄── Orin
                   [session logs: frame + output + metrics + votes]
```

### Components

1. **Frame ingester** (browser → hub): capture a JPEG from the video element
   on a timer or button press (`canvas.toBlob`, ~1280px long side — the
   validated image budget), POST/WS to the hub. Full WebRTC video is
   unnecessary: the Orin consumes ~1 frame per generation, so shipping
   discrete JPEGs is simpler and matches the real pipeline.
2. **Inference relay** (hub → Orin): reuses `veris/prompts.py` (modes,
   variants, context injection) and the bench client's OpenAI-format code.
   Streams tokens back to the browser over SSE. Browser speaks them with the
   Web Speech API (`speechSynthesis`) — streamed clause-by-clause TTS, the
   same UX the headset will have.
3. **Model manager** (hub → Orin over SSH): start/stop `llama-server` with a
   chosen GGUF + power mode, using the exact recipes hardened in
   `bench/run_matrix*.sh` (prep_mem, --no-mmap, -np 1, health poll). A model
   registry JSON lists: GGUF path on Orin, prompt-pack version, context
   length, notes. Fine-tuned checkpoints get added here as they land →
   "version A/B" dropdown in the UI.
4. **Telemetry panel**: `tegrastats_logger.py` already emits JSONL on the
   Orin; the hub tails it over SSH and forwards via WebSocket. UI shows live
   watts, RAM, temp, plus per-request TTFT / tok/s from the relay.
5. **Session recorder**: every generation appends frame path + all knob
   settings + output + metrics to a session JSONL, with a thumbs-up/down +
   free-text box in the UI. **This is the test-base feedback harness** — a
   blind tester (or their sighted assistant) can rate descriptions in situ,
   and the records become fine-tuning/eval data directly.

### UI (single page)

- Video preview + "Describe" button + auto-capture toggle (every N s)
- Knobs: model/version, mode (outdoor_nav / indoor_nav / text_reading /
  describe), variants (clock-face, etc.), power mode, free-question box,
  simulated/real context toggle
- Streaming description panel with speak-aloud, TTFT and tok/s badges
- Telemetry strip: W / RAM / °C sparklines
- Session log tab with rate + annotate

## Latency expectations (set demo expectations honestly)

From the Orin benchmarks: ~12s TTFT (3B) / ~17s (7B) at 15W+ per cold frame,
5-7+ tok/s decode on 7B. The demo should display "capturing → encoding →
first word" progress so the wait reads as pipeline stages, not a hang. Frame
downscale to 1280px client-side (already validated) keeps prefill sane.

## Phase 2: remote access — "Orin at home, phone anywhere"

Requirement: whip out the phone anywhere and stream to the Orin sitting at
home, no laptop involved.

**Recommended: Tailscale.** Two modes depending on audience:

- **Private demo (you + trusted testers): `tailscale serve`.** Install
  Tailscale on the Orin, the dev box, and each phone (app store); all join a
  private tailnet, E2E-encrypted, no port forwarding, no public exposure.
  Downside for data collection: every phone must be a tailnet member, i.e.
  someone signs into the Tailscale app with an account — you do NOT want to
  put your own account on a stranger's phone. Use per-tester **user invites**
  (they sign in with their own account) if you go this route.
- **Data-collection at scale: `tailscale funnel` (the distribution setup).**
  Funnel exposes the Orin hub's HTTPS endpoint to the public internet through
  Tailscale's relay, on the same `*.ts.net` name with a real Let's Encrypt
  cert. Testers open one URL in any browser — **no app, no account, nothing
  installed on their phone.** The video still terminates on your Orin (Tailscale
  just relays the TLS); it is not a third-party server. Because the URL is
  public, the hub throttles rather than gates: per-IP sliding-window limits per
  endpoint class (GPU, paid-API, ssh, polling), a global cap on the metered
  context-engine path, and a single-occupancy guard on the model slot, since
  llama-server runs `-np 1` and a second concurrent describe only makes both
  testers wait. Enable Funnel once in the tailnet admin console, then
  `tailscale funnel --bg --https=443 http://localhost:8000` on the board.
  Links issued while the old `ACCESS_CODE` gate existed carry a `?k=...` query
  string; it is simply ignored now, so those keep working.
  This is the recommended path for "hand it to N
  people and collect as much as possible." Requires a one-time admin-console
  toggle (MagicDNS + HTTPS certs + `funnel` nodeAttr).
- The hub webapp moves from the dev box to the **Orin itself** for this mode
  (it's a lightweight FastAPI app; the Orin can serve it beside llama-server).
  Phone browser → `https://orin.<tailnet>.ts.net` → same UI, same flow.
  Tailscale's `tailscale cert` even issues a real HTTPS cert, which solves
  the getUserMedia requirement more cleanly than mkcert.
- The dev box stays optional: for demos at home it can still host the hub;
  the model-manager SSH target is the same either way.
- Latency: frame upload (~200KB JPEG) over LTE adds ~100-300 ms — noise
  against the 12-17s generation time.
- Privacy story stays intact: the video never touches a third-party server;
  Tailscale only brokers an encrypted peer-to-peer tunnel. (DERP relays can
  carry encrypted traffic when NAT traversal fails — still E2E encrypted.)
- WiFi dependency note: the Orin joins the local WiFi, so it's reachable
  whenever it has power + WiFi. Tailscale reconnects automatically after
  reboots.

Fallbacks considered: Cloudflare Tunnel (public URL — worse privacy story),
port forwarding + DDNS (exposes the Orin — no), WebRTC TURN (overkill while
frames-per-generation ≈ 0.08 fps). Tailscale wins on all axes for a
prototype.

## Build plan (rough)

1. `demo/` FastAPI hub: static page, /frame endpoint, SSE relay, prompts
   import — 1 day.
2. Model manager SSH wrapper + registry — 0.5 day.
3. Telemetry WS + UI strip — 0.5 day.
4. Session recorder + rating UI — 0.5 day.
5. mkcert + phone onboarding docs — 0.5 day.

Everything reuses code that already exists and is tested (prompts, bench
client conversion, matrix server recipes, tegrastats logger).
