# Multi-model evidence

Every number here is copied from a file under `evals/results/` and can be
regenerated with the command beside it. Design is in
`docs/multi-model-architecture.md`. Started 2026-11-18.

**Status key:** ✅ measured · ⏳ job queued or running, number not in yet.
Nothing marked ⏳ should be read as a result.

## 1. What "fast enough" means

Targets come from market research run on 2026-11-18 (sources in §8). No
vendor publishes an end-to-end latency for a blind-navigation wearable, and no
study gives a latency tolerance specific to blind users of description apps;
the targets below rest on walking-speed arithmetic and general voice-assistant
studies.

| Use | Target | Ceiling | Basis |
|---|---|---|---|
| Hazard while walking | 0.3 s cue, 0.5 s word | 1.0 s | 1.2-1.4 m/s walking; a 1.5 s reaction already needs about 1.5 m of warning |
| Outdoor description | 1.5 s | 4 s | Over 2 s feels unnatural and over 4 s degrades (arXiv 2507.22352); WorldScribe's layered pipeline reached 1.44 s |
| Indoor description | 2 s | 5 s | In-vehicle assistant studies: 1.5 s rated best, 5 s the satisfaction limit |
| Short text (signs) | 1 s | 2 s | Seeing AI and Envision read short text continuously, offline |
| Documents | 3 s, then stream | 10 s | Nielsen's 10 s attention limit |

## 2. Measured on the Jetson Orin Nano 8GB

15 W mode, JetPack 6.2, llama.cpp build 9870, TensorRT 10.3. Base model
Qwen2.5-VL-3B Q4_K_M with three adapters loaded and selected per request.

### 2.1 Time to first word against the targets

One run, airgapped, trained adapters, everything loaded
(`orin_e2e_airgap.json` ✅, `bash bench/orin/run_e2e_airgap.sh live`);
steady-state medians from `orin_experts.json` ✅.

| Use | Path | Measured | Target | Ceiling | |
|---|---|---|---|---|---|
| Hazard | detector + tracker + template | 48 ms per frame sustained (21 fps); spoken on the 3rd frame of an approach | 0.3 s cue | 1.0 s | met |
| Outdoor description | outdoor expert, 448 px | 1.18 s steady; 1.26 s on the first request after boot, 1.27 s after a switch | 1.5 s | 4 s | met |
| Indoor description | indoor expert, 560 px | 1.70 s on the first request after a switch | 2 s | 5 s | met |
| Short text | OCR speaks the block it is sure of | 0.55 s ("FIRE EXIT." on a street frame) | 1 s | 2 s | met |
| Documents | base model with the reader's prompt, on the page cropped to its text | 8 receipt photos: median 2.40 s, slowest 2.51 s; a café menu 2.53 s | 3 s | 10 s | met |
| Blurred frame | gate rejects, spoken cue | no model call | n/a | | |
| Location question | base model, no adapter, text only | 0.43 s to first word, 1.3 s total | n/a | | |

- The hazard path stops at text. Speech synthesis adds about 0.17 s per
  clause (Piper on this board, measured 2026-07-04,
  `docs/latency-roadmap.md`), which puts the first spoken hazard word near
  0.4 to 0.5 s after the object is first tracked. That figure is arithmetic
  from two measurements, not one end-to-end measurement.
- The first outdoor request used to take 1.6 s: nothing had been through
  the vision tower yet. The conductor now sends the first usable frame
  through once at boot, and warms the expert in use on the first frame
  after any switch, in a thread so no camera frame waits for it.
- Documents took four steps (section 6.2). A mosaic at native size took
  5.2 s, 3.4 s with print scaled, and 2.3 s once it was built from line
  detection alone and capped at 334 image tokens. Receipts then showed that
  the mosaic read under half the lines, so the page is now cropped to its
  text and read by the base model: 2.4 s, at 0.74 of lines read instead of
  0.48. A whole receipt takes a median 20 s to read out; the words stream
  as they come.
- By image size, time to first word is 1.23 s at 448 px, 1.91 s at 560 px
  and 2.4 to 2.6 s at 768 px for all three adapters, at 13.5 tokens per
  second.

### 2.2 Small models beside the loaded language model

| Model | TensorRT FP16 | CPU | Engine | File |
|---|---|---|---|---|
| Gate (MobileNetV3-small, 1.5 M params) | 4.9 ms | 6.4 ms on 4 cores | 4.0 MB | `orin_small_models.json` ✅ |
| Detector (D-FINE-S, 18 classes, 640 px) | 22.9 ms | not runnable (the board's onnxruntime lacks one op) | 25.5 MB | `orin_small_models.json` ✅ |

The detector figure is for the retrained model, timed with the language
model stopped (its engine had just been rebuilt); the first model measured
23.3 ms with the language model loaded.

Inside the full pipeline the gate call measures 18 ms and the reflex call
35 ms per frame, because each includes resizing a 1280 px frame on the CPU.

### 2.3 Everything loaded at once

`orin_e2e_airgap.json` ✅: 7,619 MB total. Available memory was 5,942 MB
before loading, 2,671 MB with the language model and three adapters up,
2,409 MB with gate, detector, OCR and planner added, and never fell below
1,158 MB during the run.

The model files must be evicted from the page cache after the server loads
(`bench/orin/set_adapters.sh` does it). Without that, free memory was 95 MB
with 2.9 GB of cache the GPU allocator cannot reclaim, and the second
600-token image killed the server with a CUDA out-of-memory error.

### 2.4 Airgapped

The same run executes inside `unshare -r -n`: a network namespace that
contains only loopback. Recorded from inside it: interfaces `lo` only, no
routes, `connect(1.1.1.1:443)` fails with "Network is unreachable", DNS
lookup fails. llama-server, both TensorRT engines, OCR and the planner all
start and run inside that namespace. Reproduce:
`bash bench/orin/run_e2e_airgap.sh <adapter-tag>`.

The offline planner is also evaluated with `socket.socket` patched to raise
(`planner.json`).

### 2.5 Adapter cost, and why the experts adapt attention only

Measured on the board at 448 px (`orin_experts_standin.json`,
`orin_overhead_attn.json`, `orin_overhead_qv.json`) ✅:

| Layers adapted | Time to first token | Decode |
|---|---|---|
| All linear layers (the v1 recipe) | 1.42 s | 10.7 tok/s |
| Attention only | 1.23 s | 13.8 tok/s |
| No adapter | n/a | 14.3 tok/s |

Switching adapters costs 0.13 s on the first request after the switch
(1.36 s against 1.23 s, trained adapters, `orin_experts.json` ✅).

## 3. Gate (Tier 0)

`gate.json` ✅ `python evals/gate_eval.py`. Trained on VizWiz quality labels
(photos taken by blind people) and Places365, then fine-tuned with 5,552
street-level COCO and Open Images photos. Tested on 2,000 VizWiz official val
images and 7,300 held-out Places365 images. The first version's results are
kept in `gate_v1.json`.

| Head | Clean | Walking blur indoors | Worst case |
|---|---|---|---|
| Indoor vs outdoor (accuracy) | 0.896 | 0.875 | 0.842 |
| Text present (accuracy) | 0.879 | 0.794 | 0.561 |

- Unusable-frame filter: AUROC 0.945. At the chosen threshold it wrongly
  rejects 3.6% of usable frames and catches 68.5% of unusable ones.
- Picks the still frame out of five 95.9% of the time, and a frame from a
  head turning at 15°/s or slower 99.7% of the time.
- **Cross-check on data from neither training source**
  (`gate_coco_crosscheck.json` ✅, 300 COCO val images):

  | | First version | Fine-tuned |
  |---|---|---|
  | Indoor vs outdoor, clean | 0.837 | 0.893 |
  | Outdoor scenes only | 0.720 | 0.867 |
  | Indoor scenes only | 0.953 | 0.920 |
  | With outdoor walking blur | 0.827 | 0.907 |

  The first version called street scenes indoor, which flipped the mode
  during an outdoor walk in the end-to-end run. The fine-tune fixes most of
  that. One of the three end-to-end test frames (a crossing) is still scored
  indoor at 0.69, down from 0.95.
- Cost of the fine-tune: Places accuracy fell by 0.004 to 0.005, and the
  still-frame pick fell from 99.0% to 95.9%.

## 4. Reflex (Tier 1)

### 4.1 Geometry, on KITTI labels (no images, no model)

`nav_ranging.json` ✅ `python evals/nav_ranging.py`. 7,481 frames; un-truncated,
un-occluded boxes; camera height 1.65 m.

| Quantity | Result |
|---|---|
| Bearing from the box's pixel column | 0.17° median error (pedestrians, n=2,404) |
| Range from the box's bottom row | 9.1% median error (pedestrians), 16.4% (cars) |
| Time to contact from box growth, 5-frame window | 4.0% median error on pedestrians with exact boxes; 26.8% with 3% box jitter |

Limit that shapes the product: range depends on head pitch. One degree of
pitch error moves the pedestrian range error from 9% to 31%. For a camera
pitched 15° down, a 2° error turns 2 m into 1.86-2.15 m, 8 m into 6.8-9.8 m
and 12 m into 9.5-16.3 m. So the reflex speaks a distance only up to 8 m
(`veris/reflex.py` `SAY_RANGE_MAX`) and gives direction alone beyond that.
KITTI's level camera cannot see ground nearer than about 6 m, so the 0-5 m
band, the one that matters most, is covered by geometry here and not yet by
data.

Closing objects (`nav_ranging.json`, 0.7 s window): 97.8% of objects under
3 s to contact are flagged with no false alarms; with 6% box jitter, 89.4%
flagged and 1.2% false alarms.

### 4.2 Speech policy

`nav_reflex.json` ✅: 0.17 ms per frame median; 9.3 ms with 40 objects in view.

### 4.3 Detector

D-FINE-S (Apache-2.0) fine-tuned for 85 minutes on 8,898 Open Images `train`
images with 18 classes, including four COCO lacks: door, stairs, street light
(spoken as "pole"; Open Images has no pole class) and waste container.
Held out: 1,500 images from Open Images `validation` and `test`.

`nav_detect_pl.json` ✅ (deployed model) and `nav_detect.json` ✅ (first
fine-tune), `python evals/nav_detect.py [--ckpt … --out …]`. mAP50:

| | Clean | Walking outdoors | Walking indoors |
|---|---|---|---|
| Deployed, all 18 classes | 69.9 | 66.0 | 56.6 |
| Deployed, the 13 classes shared with COCO | 76.4 | 73.0 | 63.4 |
| Original COCO checkpoint, same 13 classes | 78.9 | 71.5 | 57.4 |
| First fine-tune, all 18 classes | 66.4 | 61.3 | 50.5 |
| First fine-tune, the 13 shared classes | 72.5 | 67.2 | 55.8 |

**The first fine-tune unlearned people and cars, and the evaluation caught
it.** Open Images boxes a class only where a human verified it, so an
unlabelled person in a photo chosen for its stairs was trained as background.
The original checkpoint found 12,956 unlabelled people in the training images
beside 6,955 labelled ones, and 6,055 unlabelled cars beside 2,614
(`train/nav/pseudo_label.py`). Retraining with those boxes filled in, same
recipe and time:

| Clean mAP50 | Original | First fine-tune | Deployed |
|---|---|---|---|
| person | 74.2 | 50.4 | 71.0 |
| car | 76.2 | 58.8 | 72.9 |
| bench | 71.2 | 47.0 | 59.1 |
| door | none | 36.5 | 38.3 |
| stairs | none | 39.1 | 46.6 |
| street light | none | 35.2 | 34.8 |
| waste container | none | 91.6 | 93.1 |

- On clean images the deployed model is 2.5 points under the original on
  the shared classes. Under wearable blur it is ahead (73.0 against 71.5
  outdoors, 63.4 against 57.4 indoors), because it trains on degraded images.
- Still weak: door, stairs and street light (35 to 47), traffic light
  (62.7, down from 73.1 in the original) and bench.
- Held-out counts are thin for dog (7 images), bench (22) and stop sign (77).
- The exported ONNX file matches the PyTorch model exactly on 8 images
  (`nav_onnx.json` ✅).

## 5. Experts (Tier 2)

**Training.** The 7B teacher described 610 COCO train2017 images given the
ground-truth object packet. An output is kept only if it names no absent
COCO object and states no left/right the boxes contradict
(`train/experts/verify.py`): 266 of 309 outdoor and 248 of 301 indoor
outputs passed. Each kept target is then paired with one of five packets
(`data/experts/sft_stats.json`): the packet the real detector writes for
that image (45%), the full ground-truth packet, one with objects dropped,
none at all, or one with a planted object that is not in the image. With
mirrored copies and 200 replay rows the sets are 602 and 554 rows; each
adapter trains in about 15 minutes on the RTX 3070
(`bash train/experts/train_loop.sh outdoor indoor`).

**Held-out scoring** ✅ `experts_v2.json`, `experts_v2_thr.json`,
`experts_all_boxes.json`: COCO val2017, 100 images per scene, never trained
on. `python -m evals.experts_eval --configs ...`, then `--rescore`.
"Coverage" is the share of the image's object categories the description
names. "Invented" counts descriptions naming a COCO object that has no box
anywhere in the image. Sides are checked wherever the description says left
or right about an object whose boxes are all on one side.

| Outdoor | Image | Coverage | Descriptions with an invented object | Sides stated, right |
|---|---|---|---|---|
| Base model, no packet | 768 px | 0.47 | 4 of 100 | 11, all right |
| Expert, perfect packet | 448 px | 0.83 | 1 | 92, 96% |
| Expert, real detector packet at 0.45 | 448 px | 0.70 | 17 | 69, 94% |
| **Expert, real detector packet at 0.7 (deployed)** | 448 px | 0.70 | 12 | 74, 95% |
| Expert, no packet | 448 px | 0.56 | 10 | 30, 93% |

| Indoor | Image | Coverage | Descriptions with an invented object | Sides stated, right |
|---|---|---|---|---|
| Base model, no packet | 768 px | 0.65 | 43 of 100 | 26, 92% |
| Expert, perfect packet | 448 px | 0.84 | 16 | 90, 97% |
| Expert, real detector packet at 0.45 | 448 px | 0.65 | 35 | 68, 93% |
| **Expert, real detector packet at 0.7 (deployed)** | 448 px | 0.65 | 32 | 62, 94% |
| Expert, no packet | 448 px | 0.65 | 31 | 58, 97% |

What the tables say:

- **The packet does what it was built for.** At 448 px, where the board
  answers in 1.2 s, the expert names half again as many of the outdoor
  objects as the base model does at 768 px (about 2.4 s), and it tells the
  user which side they are on over six times as often, 95% correctly. The base model
  mostly avoids saying left or right at all.
- **Indoors the expert matches the base model's coverage and beats it on
  the other columns**, in half the words (30 against 58 per description).
- **Outdoors it invents more than the base model**, 12 descriptions in 100
  against 4. The cause is measured: 9 of its 14 invented mentions were in
  the detector's packet, and 4 of those are a vehicle that is there but
  named as the wrong kind (the detector says truck where COCO says car).
  With a perfect packet the figure is 1 in 100. The detector's precision is
  the limit, not the language model.
- **Two things were done about that.** The second training round paired
  targets with real detector packets instead of perfect ones, which cut
  invented objects from 21 to 17 outdoor and 42 to 35 indoor descriptions
  in 100 (`experts_all_boxes_v1.json` against `experts_all_boxes.json`).
  Then the packet threshold was raised to 0.7 for descriptions only, which
  cut outdoor from 17 to 12 at no cost in coverage. Hazard warnings keep
  0.45, because there a miss costs more than a false alarm.
- **A planted object is repeated in 15% of outdoor and 13% of indoor
  descriptions** (`expert+fakeside`). The expert does look at the image,
  but it is not immune to a wrong packet.
- 100 images per scene puts about ±4 points on each count; differences
  smaller than that are not findings.
- "Dining table" is 16 of the 40 invented indoor mentions for the expert
  and 26 of 55 for the base model. The checker maps any "table" to COCO's
  dining table, so indoor figures are overstated for every row alike.
- Words say "left", "ahead" or "right". The first round used clock hours;
  with a perfect packet its stated sides were right 90% of the time, and
  96% once the packet used side words (`experts_v1/experts_side.json`).

## 6. Reader (Tier 1b)

`reader_ocr.json` ✅ `python evals/reader_ocr.py`. PP-OCRv5 mobile
detection + English recognition (12.7 MB together), end to end on ICDAR 2015
test, which was shot with Google Glass (500 images, 2,096 words):

| Condition | Word recall | Word precision |
|---|---|---|
| Clean | 0.578 | 0.620 |
| Walking outdoors | 0.438 | 0.566 |
| Walking indoors (16 px smear) | 0.023 | 0.096 |

- Spoken-word precision reaches 0.95 only at confidence 0.999, where recall
  is 0.197. At 0.99, 97% of spoken words are within one character of the
  truth at about twice that recall.
- Recall by input size: 0.578 at 1280 px, 0.460 at 768, 0.394 at 640.
- Conclusion: OCR alone is not the reader. It finds the text; the model reads
  those regions at native resolution from a mosaic.

### 6.1 Mosaic against the full frame

`reader_fovea.json` ✅ `python evals/reader_fovea.py`. Same ICDAR 2015 test
images, read first by the **untrained** base model and then by the trained
reader adapter (1,015 training rows built from ICDAR 2015 train transcripts;
19 minutes on the RTX 3070). Words are scored per image without positions, and words
read from regions the dataset marks illegible count as invented, so precision
is understated equally for every row. Time to first token is on the RTX 3070,
not the board.

| Clean, 261 images | Word recall | Word precision | Image tokens | First token |
|---|---|---|---|---|
| Full frame at 1280 px | 0.564 | 0.495 | 1,196 | 0.94 s |
| Mosaic + OCR hints | 0.585 | 0.533 | 80 | 0.28 s |
| Mosaic, no hints | 0.641 | 0.515 | 80 | 0.28 s |
| OCR alone at 0.99 confidence | 0.362 | 0.728 | none | n/a |
| **Trained reader**, mosaic, hints always | 0.526 | 0.682 | 80 | |
| **Trained reader**, mosaic, no hints | 0.547 | 0.648 | 80 | |
| **Trained reader**, hints when OCR read 30% of lines or more (deployed) | 0.560 | 0.661 | 80 | |

| Walking outdoors, 131 images | Word recall | Word precision | Image tokens |
|---|---|---|---|
| Full frame at 1280 px | 0.519 | 0.541 | 1,196 |
| Mosaic + OCR hints | 0.417 | 0.592 | 64 |
| **Trained reader**, hints always | 0.382 | 0.692 | 64 |
| **Trained reader**, no hints | 0.407 | 0.548 | 64 |
| **Trained reader**, deployed hint rule | 0.431 | 0.674 | 64 |

- On clean frames the mosaic reads at least as well as the full frame for
  one fifteenth of the image tokens.
- Under walking blur the mosaic loses a fifth of the recall, because the OCR
  detector misses regions and the model never sees them. It also invents
  less (0.31 of words against 0.41).
- With the untrained model, hints cost recall (0.585 against 0.641) for a
  small precision gain.
- Training moves the reader from guessing to declining: on clean frames the
  share of spoken words that are not in the image falls from 0.43 to 0.32,
  and recall falls 2.5 points. It still gets one word in three wrong, so
  this is a reader for signs and labels, not for medicine doses.
- The hint rule came from a failure on the board. On a café menu OCR could
  read 8 of 32 lines; given 24 "[unreadable]" hints the trained reader
  answered "too blurry to read", and with no hints it read the menu. The
  threshold sweep in the result file (`g_hint_threshold_sweep`) shows the
  other side: never hinting drops walking precision from 0.69 to 0.55.
  Hinting at 30% or more keeps the menu and stays within 4 points of the
  best precision in the sweep on both conditions. The menu is one example
  at 25%, so the threshold has little margin.
- The mosaic canvas grows from 448 to 768 px before any region is shrunk or
  dropped. The first version kept the menu's logo and dropped its body text.

**Print size.** `reader_fovea_glyph.json` ✅
(`FOVEA_TAG=_glyph evals/reader_fovea.py prep|gpu|score`). The menu's
letters are 38 px tall, so its native-size mosaic cost 596 image tokens and
5.2 s to the first word. Scaling every block to 24 px print cuts that to 334
tokens and 3.4 s, and the transcript got better, not worse ("smoked ham
hock" where the larger sheet gave "smoked ham, rock"). The same scaling
applied to every ICDAR mosaic costs recall, 0.560 to 0.468 on clean frames
and 0.431 to 0.376 walking, for a saving that does not matter there (80
tokens to 27). So print is scaled only when the native sheet is over 256
image tokens, and only as far as needed, never below 24 px. That rule
changes 8 of 258 clean ICDAR sheets, so the table above stands for the
deployed path. The menu itself now takes the document path (section 6.2);
the print rule is left for frames under 12 lines whose mosaic is still over
256 tokens. The 24 px floor rests on that one menu.

**Short text.** On a real street frame OCR is never sure of every line:
there is a plate or a shop front it cannot read. The first rule (speak from
OCR only when every line is sure) therefore almost never fired outside a
test image. Now the block OCR is sure of is spoken at once and the reader
expert is given only the rest: "FIRE EXIT." at 0.55 s on the board, with the
expert's answer about the remaining text following at 2.5 s
(`orin_e2e_airgap.json`, scene "short sign on a street"; the sign is drawn
onto a street photo).

### 6.2 Documents

`reader_docs.json` ✅
`flock /tmp/veris-gpu.lock .venv/bin/python evals/reader_docs.py`. WildReceipt
test split, phone photos of receipts: of those at least 1000 px on the long
side, the first 100 in a seeded order, downscaled to the 1280 px frame. 3,711
ground-truth lines of three or more characters. A line counts as read when it
appears in the output, ignoring case and spacing. Precision is a lower bound:
a line read with one wrong character counts fully against it. Every reading
stops at 300 new tokens.

| Path | Lines read | Precision | Image tokens | First word on the board |
|---|---|---|---|---|
| Base model, full 1280 px frame | 0.723 | 0.712 | 1,063 | about 7 s (computed) |
| Mosaic of text blocks, capped, reader expert | 0.478 | 0.813 | 304 | 2.26 s |
| Mosaic, not capped, reader expert | 0.504 | 0.850 | 418 | about 3 s (computed) |
| Mosaic, not capped, OCR hints, reader expert | 0.327 | 0.713 | 418 | |
| Page cropped to its text, capped, reader expert | 0.599 | 0.837 | 334 | |
| **Deployed: page cropped to its text, capped, base model with the reader's prompt** | **0.738** | **0.795** | 334 | **2.40 s** |

Computed times use the board's measured 0.4 s for line detection plus 6.3 ms
per image token. The two measured times are 8 of these receipts on the board:
`orin_e2e_airgap.json` for the deployed path (slowest 2.51 s) and
`orin_e2e_airgap_mosaic_docs.json` for the mosaic (slowest 2.48 s).

- The deployed path reads as many lines as the base model on the full
  frame, more precisely, from a third of the image tokens. The full frame
  would take about 7 s to the first word.
- The mosaic that helps on signs hurts on a page. Cut into blocks and
  repacked, a receipt loses its layout: 0.48 of lines against 0.60 for the
  plain crop, same expert.
- The reader expert is the wrong reader for print. It is trained on signs
  and on saying so when it cannot read one; on 100 receipts it said
  "something I can't read" 630 times from the crop. The base model with the
  same prompt and the same crop reads 0.74 against 0.60. Turning the adapter
  off is a per-request setting, so this costs no memory and no load time.
- OCR hints lower recall on receipts (0.50 to 0.33) and recognising the
  lines costs the board about three seconds, so a document is read on line
  detection alone.
- A reading stops at 300 tokens. Receipts under 40 lines (70 of the 100) are
  read at 0.89; longer ones at 0.58, because the reading is cut off.

## 7. Planner (Tier 3) and the dual-LLM split

### 7.1 Offline rules planner against the GPS simulator

`planner.json` ✅ `python -m evals.planner_eval`. Seven scenarios, five test
seeds, 16,844 simulated fixes with a correlated GNSS error model, networking
disabled.
The simulator (`sim/gps.py`), its map extract and its tests are kept in the
private repo: they walk real streets.

| | Offline planner | Cloud engine's rules fallback, run offline |
|---|---|---|
| Situation | 0.838 | 0.653 |
| Indoor / outdoor | 0.970 | 0.935 |
| Mode | 0.949 | 0.863 |

8 ms per fix on the dev box, 2.9 ms on the board. Weakest scenario: parking
lot (situation 0.61).

### 7.2 Switching

`switching.json` ✅ `python -m evals.switching_eval`. The arbiter driven by the
planner alone, then with per-frame gate votes at three error rates.

| Condition | Time in correct mode | Doorway switch, median | Flaps per hour |
|---|---|---|---|
| Planner only | 0.954 | 30 s | 0.4 |
| + gate votes, 5% wrong | 0.973 | 15 s | 0.2 |
| + gate votes, 15% wrong | 0.960 | 17 s | 3.2 |
| + gate votes, 30% wrong | 0.852 | 14 s | 14.5 |

Vision recovers about half of the GPS doorway lag while the gate is right at
least 85% of the time. At 30% error it does harm, which is why the gate's
street-scene weakness (§3) is being fixed before this is trusted.

### 7.3 What the second LLM is for

Both measured on the board, base model with every adapter off, text only.

**Deciding** (`planner_llm_decide.json` ✅, n=84): asked to re-decide
situation and mode as JSON, the model's output passed strict validation 54%
of the time, lowered situation accuracy from 0.726 to 0.607, never improved
the mode, and took 10.8 s per call. Rules decide; the model does not.

**Answering** (`planner_llm_locate.json` ✅, n=251): asked "where is the
nearest X?" over the same map facts:

| | Model alone | With the fact guard |
|---|---|---|
| Place is on the map: side and distance both right | 0.934 | 0.934 |
| Place is not on the map: says so | 0.762 | 0.952 |
| Place is not on the map: claims it exists | 0.190 | 0.000 |

0.21 s to first token, 1.0 s total. The guard (`veris/planner.py`
`locate_guard`) passes an answer only if its distance and side belong to a
mapped place that the answer names or the question asked for.

## 8. Wearable camera assumptions and sources

`veris/wearcam.py` degrades open-dataset images before evaluation.

- Ray-Ban Meta: 12 MP, 3024×4032 stills, 1080p30 video (gen 1), up to 3K30
  (gen 2), about 100° field of view per Ray-Ban's marketing.
- Envision Glasses (Google Glass EE2): 8 MP, 1080p30, 83° diagonal, fixed
  focus best at 0.6 m (Google's support page).
- Brilliant Labs Frame: 1280×720 cropped to 720×720.
- Head motion while walking: vertical bob about 2 Hz, pitch velocity about
  30°/s (Hausamann 2019; Pozzo 1990 as cited in Zubair 2016).

Not published by any vendor, so assumed here: exposure time (1/30 s indoors,
1/500 s outdoors) and rolling-shutter readout. These are the first values to
replace once the real camera is measured.

Sources: support.google.com/glass-enterprise/customer/answer/9220200 ·
docs.brilliant.xyz/frame/hardware · androidcentral.com/wearables/ray-ban-meta-gen-2 ·
arxiv.org/html/2408.06627v1 (WorldScribe) · arxiv.org/html/2507.22352v1 ·
nngroup.com/articles/response-times-3-important-limits ·
frontiersin.org/article/10.3389/fnhum.2019.00179/full ·
users.soe.ucsc.edu/~manduchi/papers/sensors2010.pdf

## 9. Known limits

- **Outdoor descriptions name an object that is not there in 12 of 100
  images** (base model: 4), mostly repeating the detector. The detector's
  weak classes are the ones Open Images labels thinly: door 38, street light
  35, stairs 47 mAP50.
- **The reader gets about one word in three wrong** on Google Glass scene
  text, and under indoor walking blur nothing is readable (the gate says
  "hold still" instead).
- **A quarter of receipt lines are not read** (0.74 on 100 WildReceipt
  photos), and a reading stops at 300 tokens, so receipts of 40 lines or
  more are read at 0.58. Reading a whole receipt aloud takes about 20 s.
- Document timing on the board is 8 receipts and one menu. A dense A4 page
  is untested: at the 334-token cap its print would be smaller than a
  receipt's. The 12-line switch to the document path and the cap are set
  from these receipts, the menu and ICDAR (7 of 261 clean ICDAR frames have
  12 or more lines). The hint threshold (30%) and the print floor (24 px)
  rest on ICDAR plus the one menu.
- Expert scoring is 100 images per scene, about ±4 points per count.
- Expert training packets carry no distances (COCO has no camera geometry),
  so the experts have not been trained on the metric distances they receive
  on the device.
- GNSS error model and the gate's error pattern in the switching eval are
  assumptions, not measurements; motion state there is ground truth.
- The approach in the end-to-end run is a zoom on one frame, and the short
  sign is drawn onto a photo; neither is a recorded walk.
- Speech synthesis is not inside the end-to-end timing.
- The page-cache eviction is done by the bench scripts. The installed
  systemd unit does not do it yet (installing it needs root on the board).
- Every board number is 15 W mode on the dev kit; thermals in an enclosure
  are untested.
