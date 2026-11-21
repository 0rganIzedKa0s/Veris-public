# Sensor fusion for seamless switching & hands-free interaction

Design spec for ToF + mic + IMU/head-tracking, built on an independently
verified research review (2026-07-04) plus what's already implemented and
measured on our hardware. Companion code: `veris/arbiter.py` (+ 7 simulation tests),
`~/veris-app/listen.sh` (on-device voice), demo `🎤 Ask`.

## Architecture: subsumption, not central control

The literature's recommendation matches what we built: run every sensor
behavior continuously and asynchronously; higher-priority reflexes *suppress*
lower layers' outputs rather than asking permission (subsumption
architecture). Concretely for Veris:

```
Layer 3 (ms):    ToF hazard reflex ──────────► preempts TTS, plays "Stop —"
Layer 2 (ms):    barge-in (VAD during TTS) ──► halts speech
Layer 1 (<10ms): mode arbiter (IMU+GPS+light fusion + cloud proposals)
Layer 0 (3.6s):  VLM describe loop — never consulted for anything above it
```

Nothing above Layer 0 ever waits on the VLM; the arbiter is read *between*
generations, so switching costs zero latency by construction.

## Mode-switch accuracy: what the evidence says

1. **GPS alone will lie at exactly the moment that matters.** Receivers keep
   reporting fixes 10–35s *after* walking indoors (worst near wide/glass
   entrances) — so the cloud context engine's venue inference is structurally
   late at doorways. This validates the arbiter's rule that cloud signals
   only *propose* and local signals confirm.
2. **Fused indoor/outdoor detection is a solved problem at 96–100%** with
   <10ms inference: 9-axis IMU @10Hz + adaptive light-sensor threshold + GPS
   horizontal-accuracy (5m threshold) [DeepIOD, 97–98% in-the-wild]. A
   GNSS-SNR hybrid hit 100% over 20 transition points.
3. **Ready-made hysteresis numbers**: GNSS SNR dead band **25 dB (indoor) /
   33 dB (outdoor)** — inside the band, hold state. Entering top-bounded
   spaces: last-2s SNR average dropping ≥20% below the outdoor mean.
   These slot directly into `Arbiter` as an indoor/outdoor input signal.
4. **Barometer = near-instant door-event bell** (99.3–99.8% detection of
   door open/close from pressure transients) — but building-wide and
   HVAC-dependent; use as a *dwell-shortener* (a door event during a pending
   indoor↔outdoor candidate confirms it early), never as sole trigger.
5. **Walking/stationary/vehicle from IMU**: Random-Forest-class models on 5s
   accel+gyro windows reach 81–93% for transport modes; walking vs
   stationary (all we need for mode gating) is the easy end. 5s windows
   also happen to match our DWELL_S≈4-5s.

**Expected outcome**: cloud proposal (what venue) + fused IOD (in or out) +
IMU walking state (moving or not) covers the mode matrix
{outdoor_nav, indoor_nav, text_reading, describe} with each axis backed by a
96%+ signal, and hysteresis absorbing the residual errors. The failure mode
that remains is *venue misidentification* (GPS says restaurant, user is in
the shop next door) — which the describe output itself corrects ("you appear
to be in a clothing store" → feed back to the arbiter as a vision vote;
cheap to add once fine-tuned).

## Hands-free interaction: three channels, all off the GPU

1. **Voice (built, measured)**: C920 mic → whisper.cpp **CPU** tiny.en =
   ~1.4s utterance→transcript on our board; GPU untouched. Upgrade path:
   NVIDIA WhisperTRT (0.64s tiny.en / 0.86s base.en for 20s audio, ~440–490MB)
   — 2–3× faster but GPU-resident, and **no one has measured it contending
   with a running 3B VLM**; benchmark before adopting. Missing piece for
   full hands-free: wake-word/VAD trigger (openWakeWord/Silero claims didn't
   survive verification — measure ourselves; both are CPU-light in practice).
2. **Head gestures (recommend to sensor team)**: single IMU at the glasses
   temple classifies 6 gestures at 100% (lab, DTW-based, 100Hz), earbud IMU
   7 gestures at 94.3%; Apple ships nod=yes/shake=no on AirPods Pro. For
   blind users this is the *socially quiet* channel: nod to accept
   "read the menu?", shake to dismiss, double-nod to repeat. Latency is
   gesture duration (~1s) + trivial classification.
3. **Head-pose as attention pointer**: "describe where I'm facing" — head
   yaw selects the crop/region of interest sent to the VLM. Zero extra
   sensors beyond the same temple IMU; also drives the stale-frame guard
   already in the arbiter (>30° turn since capture → drop description).

## ToF: the reflex layer (sensor team BOM input)

- **Proof it works**: a published head-mounted ToF assistive wearable
  achieves **<320ms sensing-to-feedback** — inside human collision-response
  time and ~11× faster than our VLM path. This is the "Stop —" channel.
- **Part guidance**: multi-zone ToF (ST **VL53L5CX / VL53L7CX**, 4×4–8×8
  zones, multi-target, ~4m range, ~$15) over single-zone: zone patterns
  distinguish doorway-vs-wall and detect head-height overhangs; a downward
  zone row covers drop-offs. Single-zone can only say "something close."
- Arbiter integration is already coded: `update_tof()` → hazard interrupt,
  gated on walking (standing near a wall ≠ hazard — tested).

## Radar (added to the pack spec 2026-07-04 — knowledge-based, not yet research-verified)

mmWave radar complements rather than duplicates ToF, and the split is clean:

- **ToF measures static geometry** (doorways, poles, drop-offs, head-height
  obstacles) precisely at short range but says nothing about motion until
  frames are differenced — slow and noisy for fast objects.
- **Radar measures radial velocity directly** (Doppler): "something
  approaching at 4 m/s from the left" in a single measurement, tens of
  meters of range, and — unlike camera and ToF — indifferent to darkness,
  rain, fog, and glare. It's the right sensor for the *moving-hazard*
  reflex: vehicles at crossings, cyclists on shared paths, an approaching
  person behind you.
- Distance-in-steps output pairs naturally: radar/ToF give ground-truth
  meters, the composer converts (÷0.75) to steps for users on that variant —
  numbers come from sensors, never the VLM.
- Part classes to evaluate: 60GHz presence/motion modules (cheap, short
  range) vs automotive-derived 77GHz modules (TI IWR6843-class, ~$30-60,
  point clouds + velocity per target). The 8x8 ToF + one IWR-class radar +
  temple IMU is a complete reflex-sensor suite.
- Arbiter integration mirrors ToF: `update_radar(targets)` feeding hazard
  interrupts gated by closing velocity rather than distance alone (a parked
  car 2m away is fine; anything closing >2 m/s from any bearing is not).

## Honest caveats (from adversarial verification)

- Nearly all context-detection accuracy numbers come from *smartphone*
  studies; transfer to head-worn placement is untested (head IMU orientation
  differs — expect retuning, not failure).
- The 100% figures are small-sample/lab (20 transition points; 419 gesture
  samples); in-the-wild-while-walking accuracy is unmeasured.
- WhisperTRT-beside-VLM GPU contention and wake-word latencies are the two
  gaps we must measure ourselves.
- Barometer signal is building-wide, not user-localized.

## Concrete asks for the sensor team

1. IMU at the glasses temple (100Hz, MPU9250-class) — feeds walking state,
   head gestures, head-pose pointer, stale-frame guard. Highest value/effort.
2. VL53L7CX forward + slight down-tilt — reflex channel + doorway signatures.
3. Light sensor + barometer on the pack (cheap, passive) — IOD fusion inputs.
4. Wire everything to the arbiter's `update_*()` API (veris/arbiter.py);
   thresholds live in one place at the top of that file, current values:
   DWELL_S=4, MIN_RESIDENCE_S=10, HAZARD_NEAR_M=1.2, STALE_YAW_DEG=30, plus
   the GNSS dead band 25/33 dB when SNR plumbing exists.
