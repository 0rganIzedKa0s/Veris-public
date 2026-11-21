# Sensor integration ideas — mic, ToF, IMU

Brainstorm for folding the other pack sensors into the vision pipeline. Each
idea is tagged with the research finding it serves (from the private
evidence review) and a rough effort guess. The organizing principle
from the research: **two channels** — a low-latency reflex channel for safety
(never waits for the VLM) and the descriptive channel (VLM). Sensors mostly
feed the reflex channel and *ground* the descriptive one.

## Microphone

1. **Voice queries (the on-demand trigger).** Wake word or hold-a-button →
   whisper.cpp (tiny/base runs comfortably on the Orin beside the 3B VLM;
   tight beside 7B — schedule ASR between generations). Research: users
   overwhelmingly preferred conversational interaction (87% of requests).
   Effort: low — whisper.cpp is mature. This is the highest-value mic use.
2. **Barge-in.** User speech immediately pauses TTS output. Speech is
   sequential audio bandwidth — the #1 verbosity complaint becomes tolerable
   when the user can always interrupt. Effort: low (VAD gate on TTS).
3. **Ambient hazard sounds** (approaching vehicle, siren, reversing-truck
   beeper, bicycle bell): a small audio-event classifier (YAMNet-class, ~ms
   inference) feeding the reflex channel → "Stop —" interrupts. Complements
   vision at night/occlusion; blind users already lean on hearing, so only
   surface what the user plausibly missed (configurable). Effort: medium.
4. **Sound-source direction** (needs 2+ mics): "the voice calling you is on
   your left." Effort: medium-high; later.
5. **Privacy note:** all audio stays on-device (the marketing pillar). No
   cloud ASR.

## ToF / depth

1. **Ground-truth distances for the describer.** The VLM guesses distances
   and sides badly (documented failure + our own laterality bugs). Fuse
   detector boxes with ToF depth → inject verified facts into the prompt:
   "detected: doorway, 2.1 m, 15° left". The VLM narrates; sensors measure.
   Research: quantitative spatial estimates are wanted but current systems
   refuse them. Effort: medium; highest quality-per-effort of all sensor
   fusions and it directly patches our worst VLM failure.
2. **Head-height obstacle reflex.** The classic cane-gap: branches, signs,
   truck mirrors at face height. ToF cone forward → distance threshold →
   immediate "Stop — obstacle at head height" interrupt, bypassing the VLM
   entirely (12s TTFT is uselessly slow for this). Research: ~1 m precision
   needed at crossings; hazards-first ordering. Effort: low-medium. Do this
   early — it's the safety story for the whole product.
3. **Drop-off detection** (curbs, stair-down, platform edges): downward ToF
   ray; sudden depth increase → interrupt. Effort: medium (mounting angle).
4. **Doorway/gap confirmation** in indoor_nav: depth silhouette verifies
   "opening ahead" before the system says it. Kills a whole class of
   hallucinated doorways. Effort: medium.
5. **Auto verbosity by clutter:** mean scene depth < 2 m (crowded) → terse
   cues; open space → allow richer description. Effort: low once depth flows.

## IMU

1. **Heading re-anchor / stale-description guard.** Egocentric directions
   rot the moment the head turns. IMU yaw delta > ~30° since frame capture →
   drop or re-caption the pending description ("things may have moved —
   describing again"). With a 12s TTFT this is not optional; it's
   correctness. Effort: low. Do first among IMU ideas.
2. **Frame selection by motion.** Capture bursts, keep the frame with lowest
   angular velocity → sharper input (VizWiz: blurry frames are the #1 real-
   world quality problem). Also: gate auto-capture to pauses in head motion —
   users naturally stop to "look." Effort: low.
3. **Walking-state machine** (IMU step detection): walking → outdoor/indoor
   nav mode, terse, hazard-priority; stationary → describe mode allowed,
   richer. This is the seamless mode-switching brain stem, with GPS/context
   picking indoor vs outdoor. Research: context-dependent content is the
   core finding. Effort: medium.
4. **Dead reckoning indoors** between GPS fixes: step count + heading gives
   the context engine a rough indoor track ("about 20 m down the corridor
   since the entrance"). Effort: medium-high (drift), pairs with landmark
   re-anchoring from vision.
5. **Fall detection** → loud check-in prompt, then optional SOS via phone.
   Effort: low-medium; big trust win for users and families.
6. **Spatial audio anchoring** (with the open-ear headset): IMU head pose +
   object bearing → play the cue *from the object's direction*. Research:
   spatialized audio guidance has low attentional load. Effort: high;
   prototype later.

## Fusion sketch

```
reflex channel (≤100 ms):  ToF thresholds ─┐
                           audio events ───┼─► interrupt arbiter ─► TTS "Stop —…"
                           (IMU fall) ─────┘        (preempts everything)

grounding channel:         detector boxes + ToF depth + IMU heading + GPS/POI
                           ─► fact table ─► injected into VLM prompt per frame

descriptive channel:       camera frame ─► VLM (mode prompt + fact table)
                           ─► streamed TTS (interruptible, barge-in aware)
```

Priorities if I had to order the roadmap: mic voice queries + barge-in →
ToF head-height reflex → IMU heading guard + frame selection → ToF-grounded
distances in prompts → walking-state auto-mode → the rest.
