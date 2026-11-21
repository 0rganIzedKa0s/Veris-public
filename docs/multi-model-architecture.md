# Multi-model architecture

Status: built and measured on the board (2026-11-18 to 19). Measured results
are in `docs/multi-model-evidence.md`; nothing in this file is a result.

## The problem with one model

Veris-3B-v1 serves every mode by swapping the system prompt. That leaves four
gaps:

- A 3.6 s vision-language model (VLM) cannot warn about a pole. Navigation
  needs a different kind of model, not a faster prompt.
- Text reading pays for 1280 px (13.7 s to first token) and can still invent
  a word.
- v1 was trained under one prompt, so what it treats as important drifts in
  the other modes.
- The planner needs the cloud, so the device is not airgapped.

## Latency targets (market research, 2026-11-18)

Time to the first cue or spoken word. Sources are in the evidence doc.

| Use | Target | Ceiling | Why |
|---|---|---|---|
| Hazard while walking | 0.3 s cue, 0.5 s word | 1.0 s | At 1.3 m/s, one second spends 1.3 m of a roughly 1.5 m warning budget |
| Outdoor description | 1.5 s | 4 s | Over 2 s feels unnatural, over 4 s degrades; outdoor scenes go stale as the user walks |
| Indoor description | 2 s | 5 s | User is usually stationary; 1.5 s rated best, 5 s the satisfaction limit |
| Short text (signs) | 1 s | 2 s | Shipping products do continuous offline OCR |
| Documents | 3 s, then stream | 10 s | 10 s is the attention limit |

No single model meets all five. The tiers below are set by these numbers.

## Tiers

```
camera frame
   │
   ▼
Tier 0  GATE      every frame, ~5 ms     usable? indoor/outdoor? text present?
   │
   ├──► Tier 1  REFLEX   every frame, ~30 ms    objects, bearing, range, closing speed
   │       │                └─► speaks hazards directly (templates, no LLM)
   │       ▼
   ├──► Tier 1b READER OCR   on demand, ~0.3 s    exact words + confidence
   │       │                └─► speaks short text directly
   │       ▼
   │    PACKET  (veris/packet.py)  ◄──── Tier 3  PLANNER  every 30-60 s
   │       │                              GPS -> situation, look-for, class priors
   ▼       ▼
Tier 2  EXPERTS   one 3B base in memory, one adapter active per request
        outdoor · indoor · reader
```

**Five models, one of them shared four ways:**

| # | Model | Kind | Job | Trained on |
|---|---|---|---|---|
| 0 | Gate | small CNN, multi-head | reject blurry/dark frames, vote indoor/outdoor/text | VizWiz quality labels, Places365 |
| 1 | Reflex | real-time detector + geometry | navigation hazards, object packet | Open Images subset, with the original detector filling in the people and vehicles Open Images leaves unlabelled; ranging checked on KITTI |
| 2a | Outdoor expert | adapter on Qwen2.5-VL-3B | outdoor description | GT-verified teacher outputs, paired with the packets the real detector writes |
| 2b | Indoor expert | adapter on the same base | indoor description | same |
| 2c | Reader expert | adapter on the same base | read text-region mosaics, say what is unreadable | ICDAR 2015 train, targets built from its transcripts |
| 2d | Document reader | the same base with no adapter, the reader's prompt | read a page, a menu or a receipt from a crop of its text | none; scored on WildReceipt |
| 3 | Planner | rules over an offline map; the same base with no adapter for questions | GPS → situation; location answers | none (rules + prompt), scored against the simulator |

Four VLMs do not fit in 8 GB. One base plus adapters does: the base and the
vision tower load once, and llama-server selects the adapter per request, so
a switch needs no reload (measured: about 0.2 s on the first request after).

The adapters touch attention layers only. An adapter applied at run time is
extra work on every token, and on the board an all-layer adapter costs 25% of
decode speed where an attention-only one costs 3%. The projector is frozen so
all experts share one vision tower.

## What is specific to this problem

Each of these uses a fact about the wearable that a general VLM pipeline
cannot assume.

1. **Geometry is measured, not generated.** A head camera sits at a known
   height with a known pitch. Range to anything standing on the ground
   follows from the pixel row of its base; bearing follows from the pixel
   column; closing time follows from how fast its box grows. None of it needs
   a depth network or an LLM. The experts receive this as text and narrate
   it. The 3B's weakest skill (left/right, distance) is removed from its job.
   The packet says "left", "ahead" or "right" in words. A 70-degree camera
   only spans 11 to 1 o'clock, and the experts repeat a side word more
   reliably than they translate a clock hour.

2. **Experts are trained on the packet, so they can look at less.** If
   laterality and distance arrive as text, the image no longer has to carry
   them, and the image budget can drop below the 768 px floor that laterality
   forced. Fewer vision tokens is the only thing that moves time to first
   token on this board.

3. **Ground-truth-verified distillation.** Training targets come from the 7B
   teacher, but they are accepted by annotation, not by another LLM: an
   outdoor/indoor target is kept only if every object it names is in the
   ground-truth boxes and every left/right it states matches them; a reader
   target is kept only if its quoted words equal the ground-truth transcript.
   The filter is exact, free and reproducible. In training, the verified
   target is paired with the packet the real detector writes for that image,
   false detections included, so the expert learns that the packet can be
   wrong and the image decides.

   The detector also serves two thresholds. Hazards are spoken from 0.45
   confidence, because a missed pole costs more than a false warning. The
   packet handed to a describing expert lists only detections from 0.7 up,
   because there the costs are reversed: the expert can see the image
   itself, and it repeats a false object it is handed.

4. **OCR finds the text; the expert reads only that.** Measured on Google
   Glass footage, OCR alone is too unreliable to be the reader (its words
   are 95% right only at a confidence that discards four in five). So the
   OCR detector's job is to say where text is and how tall its letters are.
   Each block is cropped from the full-resolution frame and packed into one
   small mosaic, which the reader expert reads with OCR's guesses as hints.
   A sign costs tens of image tokens at native size, where a 1280 px frame
   costs about 1,300. A menu held close would cost 600, so when a sheet
   goes over 256 tokens its blocks are scaled down by letter height, as far
   as needed and never below 24 px print. Scaling everything would be
   cheaper still, but it costs recall on small scene text (measured). OCR's guesses are passed as
   hints only when it could read at least 30% of the lines; below that the
   hint is mostly "[unreadable]" and talks the reader out of text it can
   see. A block in which OCR is sure of every line (0.999 or better) is
   spoken at once, without the language model, and the expert reads only
   what is left.

   A page, a menu or a receipt (12 or more text lines) takes a different
   road. Finding the lines is a third of OCR's work; recognising 60 of them
   takes the board's CPU three seconds, and on receipts the hints lower
   recall. A mosaic also throws away the layout, which is what makes a page
   readable, and the reader expert is trained on signs. So a document is
   cropped to its text from line detection alone, capped at 334 image
   tokens, and read by the base model with the reader's prompt and the
   adapter turned off: 0.74 of receipt lines read against 0.48 for the
   mosaic and the expert, in the same 2.4 s.

5. **The second LLM is the first one with no adapter, and it answers; it
   does not decide.** Airgapped, the planner's decisions (situation, mode,
   class priors) come from rules over an offline map extract. Measured on the
   board, the 3B base is slower and less accurate than those rules at
   deciding, so it is not asked to. What it does is answer location questions
   in language ("where is the nearest crossing?") over the same map facts, in
   about a second, behind a check that refuses any distance not attached to a
   mapped place. It is the same weights with every adapter off, so it costs
   no memory.

6. **The planner tunes the fast tier, not only the prompt.** Its `class_priors`
   re-rank which detections are worth speaking at this location, and its
   `suggested_mode` is one vote in the switch.

7. **The cache follows the expert in use.** The language model keeps one
   prompt cache. The conductor sends the first usable frame through at
   boot, and warms the new expert on the first camera frame after any
   switch, in a thread so the reflex loop never waits. An earlier version
   warmed the candidate during its 4 s dwell; measured, that took the cache
   away from the expert the user could still ask (1.5 s against 1.2 s), so
   a candidate is not warmed until it wins.

8. **Bad frames never reach the VLM.** Blind-taken frames are often blurred or
   dark. The gate says "hold still" in milliseconds, or picks the sharpest
   frame of the last 0.6 s, where the VLM would spend seconds describing
   blur. Older frames are never used: the user has turned their head since.

9. **The tiers correct each other.** A scene classifier calls a street
   crossing "indoor" when a wall fills the frame. The detector is already
   running, so a car, bus, traffic light or sign in its output pulls the
   gate's indoor vote down before the switch sees it.

## Switching criteria

Three rules, in priority order. The first two are never overridden by
inference.

1. **Reflex is always on.** It is not a mode. A hazard speaks over any expert.
2. **A user command switches instantly**, with no hysteresis.
3. **Otherwise evidence accumulates per mode** and the existing dwell (4 s)
   and residence (10 s) rules decide when a switch is believed:

| Signal | Source | Rate | Pushes toward |
|---|---|---|---|
| `indoor_p` | gate, lowered when the reflex sees street objects | every frame | indoor / outdoor |
| `text_p` high while stationary | gate | every frame | reader |
| walking | motion proxy / IMU | every frame | nav-led output; expert only on change or request |
| `suggested_mode` | planner | 30-60 s | whichever it names |
| GNSS quality drop | phone | 1 Hz | indoor |

Vision outranks GPS when they disagree, because GPS is known to lag 10-35 s
at doorways (`docs/sensor-fusion-switching.md`).

**While walking, the expert stays quiet unless the scene changes.** The reflex
tier keeps a set of tracked objects; an expert description is triggered by a
request or by a change in that set, not by a timer.

## Repository layout

| Path | Contents |
|---|---|
| `veris/packet.py` | the tier contract |
| `veris/wearcam.py` | wearable-camera degradation model |
| `veris/gate.py` | Tier 0 inference |
| `veris/ranging.py`, `veris/reflex.py` | Tier 1 geometry, tracking, speech policy |
| `veris/reader.py` | Tier 1b OCR, reading order, confidence gating |
| `veris/planner.py` | Tier 3 offline planner |
| `veris/experts.py` | Tier 2 registry and llama-server client |
| `veris/conductor.py` | ties the tiers together; extends `arbiter.py` |
| `sim/gps.py` | GPS simulation engine with ground-truth situations |
| `train/<model>/` | dataset builders, configs, training scripts |
| `evals/` | one script per claim; each writes `evals/results/<name>.json` |
| `bench/orin/` | on-device timing and airgap checks |
| `data/` | downloads and generated sets (gitignored) |

## Operating notes learned on the board

- Cap llama-server's prompt cache (`--cache-ram 128`). Its default limit is
  8 GB of RAM; after a few hundred requests it had grown by 1.8 GB and left
  no memory for anything else.
- The GPU allocator cannot reclaim page cache. Evict the model files from the
  cache while the server loads (`bench/orin/e2e.py` `evict`, or the systemd
  unit's `drop_caches`), or the 705 MB vision buffer fails to allocate and
  the server dies on its first image while `/health` still says ok.
- Build TensorRT engines with the language model stopped. The detector's
  build needs over 3 GB and takes 17 minutes; it is a one-off install step.
- Run clients on the board. Requests sent over the USB network link can stall
  before they reach the server.

## Rules for everyone building this

- **GPU jobs take the lock**: `flock /tmp/veris-gpu.lock <command>`. One 8 GB
  card is shared by every training run.
- **Every claim has a script and a JSON result.** No number goes in a doc
  unless `evals/results/*.json` holds it and a command reproduces it.
- **Evaluate on clean and on `wearcam`-degraded images.** Report both.
- **Test splits are never trained on.** Record the split in the result file.
- Inference modules import no training code and no network client.
