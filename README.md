# Veris

The vision half of a privacy-first assistive wearable for blind and low-vision
users: a Jetson Orin Nano pack + head-mounted camera that describes the world
through an open-ear headset, with **all video processed on-device**. The only
external contact is phone GPS feeding a cloud "context engine" that pre-prompts
the local model (a dual-LLM architecture) — no image ever leaves the device.

End-to-end prototype, built and verified on real hardware: live camera →
a pipeline of small models and three experts on one language model, all on the
Orin → streamed spoken descriptions, with situational mode switching,
on-device voice queries, a phone demo served over Tailscale, and scripted
training pipelines that shipped their models to the device.

## Live demo

**[Try the demo — always up](https://0rganizedka0s.github.io/Veris-public/)**
— a recorded copy of the webapp that needs no hardware. Four sample images,
each replaying a real run captured from the Jetson Orin over the actual
inference path: verbatim text, measured timings, streamed at the speed it
really ran. Compare replays both halves, so you can hear the pipeline
against the untuned model. Labelled as recordings throughout — nothing here is
presented as live.

**[Try it live on the board](https://veris-orin.tail68ae71.ts.net/?k=efzcjzgxo0)**
— the same page, talking to a Jetson Orin Nano devkit over a Tailscale funnel:
your camera → on-device models → spoken description, plus voice questions and
GPS-driven mode switching. No sign-in; it is rate-limited per IP instead,
because there is one 15 W board with a single model slot behind that link.

The board is a device on a desk, so **that second link is not always up** —
which is exactly why the first one exists. Both pages open with a walkthrough
explaining what any of this is, and both work with no camera.

## Highlights

- **Five models in 8 GB, airgapped** — a gate that rejects blurred frames, a
  detector that speaks hazards from a template at 21 fps, an OCR reader, and
  outdoor, indoor and reading experts as per-request adapters on one
  Qwen2.5-VL-3B, with an offline map planner. Measured on the board with no
  network interface: first word in 1.2 s outdoors, 1.7 s indoors, 0.55 s for a
  short sign and 2.4 s for a receipt, each inside a target set from published
  latency research. Design in `docs/multi-model-architecture.md`; every
  number, its script and its limits in `docs/multi-model-evidence.md`.
- **Scored on open datasets** — COCO and Open Images for the experts and the
  detector, KITTI for ranging, ICDAR 2015 (shot with Google Glass) for signs,
  WildReceipt for documents: 74% of receipt lines read from a third of the
  image tokens the full frame needs.
- **Perceived latency cut from 13.5 s to 3.6 s cold — and 0.15 s on repeated
  scenes** (3.7× / ~90×) — time-to-first-spoken-word driven down via an
  image-budget sweep, prompt-KV caching, and clause-by-clause streaming TTS.
  Measured ladder (including rejected approaches) in `docs/latency-roadmap.md`.
- **Fine-tuned model deployed to the edge** — Veris-3B-v1: QLoRA on 653
  curated examples (judged 7B-teacher distillation + 344 hallucinations
  rescued into honesty targets + mirror-augmented spatial pairs), trained
  overnight on a single RTX 3070, quantized to GGUF Q4_K_M, serving on the
  Orin at 15 W. Invented details on unanswerable images **halved**, style
  keep-rate 23% → 40%, zero over-refusal, no speed penalty.
- **Eval-gated training** — behavioral gates for laterality, shuffled-image
  shortcut learning, and context-bleed; the v1 adapter passed laterality
  86% → 100% and context-bleed 9 → 0 (`dataset/evals/`).
- **Dual-LLM architecture live** — phone GPS → cloud context engine
  (Google Maps + gpt-4o-mini, OSM/rules fallback, spend-guarded, Dockerized)
  → `{situation, suggested_mode, look_for, context}` → automatic mode
  switching and situational priming of the on-device VLM, with a
  regression-tested guard that priming never invents unseen objects.
- **Data flywheel** — every interaction and 👍/👎 rating lands in session
  logs, syncs off the board, and is mined back into training rows for the
  next fine-tune; every capture is also indexed in a local SQLite DB on the board.
- **Grounded in research** — description style, mode design, and every
  training-data rule trace to a cited evidence base on BLV user preferences
  (`docs/style-guide.md`; the full evidence review and training playbook are
  kept in a private companion repo).
- **Accessible by construction** — full screen-reader pass on the demo
  (ARIA live-region streaming, focus management, WCAG AA contrast), plus a
  keyboard-navigable walkthrough and an honest offline mode that explains
  itself instead of failing silently.

## System map

```
[phone GPS] → context_engine/ (cloud role: Maps+LLM → situation, mode, "look for") ─┐
                                                                                    ▼
[camera frame] → gate → detector (hazards, object packet) → OCR reader ─┐           │
                 veris/conductor.py decides which tier answers          ▼           ▼
                 llama-server on Orin: Qwen2.5-VL-3B + outdoor / indoor / reader adapters
                                                                                    │
[demo/ webapp] ← streamed description + TTS + which tier answered ←────────────────┘
[veris/arbiter.py + pack_loop.py]  ← the device's mode/interrupt brain (sensor-ready)
[dataset/]  ← the model factory: teacher → judge → curate → QLoRA → GGUF → device
```

**Stack:** Python 3.12 · FastAPI · PyTorch / Transformers / PEFT (QLoRA) ·
llama.cpp on Jetson Orin Nano · whisper.cpp (on-device STT) · LLaMA-Factory ·
Docker · SQLite · Tailscale.

## Components

| Path | What it is |
|---|---|
| `veris/prompts.py` | The four modes (outdoor_nav, indoor_nav, text_reading, describe) + `general` untuned baseline; per-mode image budgets and user variants |
| `veris/model.py` | Dev-box VLM wrapper (Qwen2.5-VL, 4-bit) |
| `veris/conductor.py` | Which tier speaks and when: gate, reflex, reader, expert; keeps the model cache on the expert in use |
| `veris/gate.py`, `reflex.py`, `ranging.py`, `reader.py` | The fast tiers: frame quality and scene votes, detection + tracking + distance, OCR with a text-region mosaic for signs and a page crop for documents |
| `veris/experts.py`, `packet.py`, `planner.py` | The three adapters and their prompts, the packet the fast tiers hand them, and the offline map planner |
| `train/`, `evals/`, `sim/` | Training for the gate, detector and adapters; every evaluation with its result file; the GPS simulator (the simulator and its map extract are kept in the private repo) |
| `veris/arbiter.py` | Mode-switch state machine: hysteresis, reflex interrupts, stale-frame guard (7 scenario tests) |
| `veris/pack_loop.py` + `veris/motion_proxy.py` | Two-channel device loop + camera-motion IMU stand-in, with deterministic replay harness (`tests/replay/`) |
| `context_engine/` | GPS → Google/OSM → gpt-4o-mini/rules → `{situation, suggested_mode, look_for, context}`; vision-vote feedback; spend guard; Dockerfile |
| `demo/` | The webapp (`demo/pipeline.py` runs the multi-model pipeline behind it): phone/USB/Orin cameras **or bundled sample images**, streamed descriptions + TTS, voice ask (whisper on-device), Compare (generic vs Veris), ratings → session logs, guided walkthrough, offline replay of recorded runs, per-IP throttling, screen-reader accessible |
| `dataset/` | Fine-tuning factory: VizWiz frames, 7B teacher gen, gpt-4o-mini judge, abstention rescue, session flywheel, SFT builder, QLoRA config, eval gates |
| `bench/` | Orin benchmark harness (latency/power/thermals) + latency experiments |
| `docs/` | Research, results, and design docs — benchmarks, latency roadmap, sensor-fusion design, description style guide, accessibility notes |
| `ops/` | Boot-resilience stack for the board (systemd units, watchdog, uptime check), the always-on front door that keeps the public link working while the board is off (`ops/guard/`), the public-repo export, and the static-demo build |

## Quickstart

Everything runs from this repo with [uv](https://docs.astral.sh/uv/). The
Jetson is reachable at `192.168.55.1` over USB device mode (or your tailnet);
API keys and per-machine settings live in `.env` — see `.env.example`.

```bash
# Demo hub on this PC (the Orin serves its own copy for the phone)
uv run uvicorn demo.app:app --port 8000          # http://localhost:8000
uv run uvicorn context_engine.service:app --port 8090

# Phone from anywhere: expose the hub with Tailscale Serve/Funnel
# (see docs/demo-webapp-architecture.md)

# Board services after a reboot
ssh <board-user>@192.168.55.1 'bash ~/veris-app/start_services.sh'
# Model select (base vs tuned):
# The flags argument REPLACES the stored one — always pass the image cap
# (see docs/orin-benchmarks.md):
ssh <board-user>@192.168.55.1 'VERIS_MODEL=$HOME/models/qwen2.5-vl-3b/Veris-3B-v1-Q4_K_M.gguf bash ~/veris-app/restart_llm.sh "--image-max-tokens 1600"'

# Put the multi-model pipeline behind the board's demo (back: bench/orin/restore_demo.sh)
bash ops/orin/deploy_multimodel.sh

# Build the always-up recorded copy of the demo (-> ./site, publish to Pages)
bash ops/build_static_demo.sh

# Tests
uv run --with pytest python -m pytest tests/ -q

# Rebuild the training set / retrain
uv run python dataset/sync_sessions.py
uv run python dataset/build_sft_set.py
cd ~/LLaMA-Factory && .venv/bin/llamafactory-cli train ~/Desktop/veris/dataset/sft/qlora_3b.yaml
```

## Privacy

Session data, captured frames, model weights, and credentials are all
gitignored — nothing user-identifying is committed. The demo's four sample photos are openly
licensed and credited in `demo/static/samples/CREDITS.md`. Training imagery comes
from the public [VizWiz](https://vizwiz.org/) dataset (© VizWiz contributors,
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)) plus
consenting-tester sessions that never enter version control. The VizWiz
annotation payloads themselves aren't redistributed here (they contain OCR'd
text from blind users' photos); the pipeline rebuilds every derived file from
the official VizWiz download.

## License

Copyright © 2026 Dhira Vadlamudi. Source is shared publicly for portfolio and evaluation
purposes only — all rights reserved. No permission is granted to use, copy,
modify, or distribute this code or its models.
