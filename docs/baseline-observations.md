# Run observations — Qwen2.5-VL 3B vs 7B, 4-bit (2026-07-03)

## Round 3: Qwen2.5-VL-7B-Instruct nf4 (6.2 GB VRAM — fits the 8 GB budget)

The two 3B capability failures largely resolve at 7B:

- **Laterality:** 7B placed the door, bench, and counter on the correct sides
  in indoor_nav; one minor slip remains (light-switch wall side in describe).
  3B was wrong or unstable on nearly every lateral claim.
- **Occlusion honesty:** on the blocked menu, 7B transcribed only the visible
  fragments verbatim ("Sticky pork / relish, wam / & chilli card") instead of
  inventing completions ("chilli cake" at 3B). It still doesn't *announce*
  the occlusion — that stays on the fine-tuning target list.
- **Scene salience:** 7B spotted the person with a cane immediately ahead in
  the crossing frame — the single most important element, which 3B missed in
  every run.

New issues found at 7B, both fixed by prompt edits (describe mode):
- Hallucinated "two people ahead" in an empty kitchen — induced by the
  unconditional "say how many people are present" instruction; now
  conditional.
- Described the scene instead of answering a direct question; now
  answer-first.

Cost: TTFT 1.4-2.1 s warm vs 0.8 s at 3B; totals 4-11 s vs 2-9 s. Jetson
Orin Nano will be slower than the 3070 (memory-bandwidth-bound decode).

**Recommendation:** develop the description style and fine-tuning data
against 7B behavior; keep 3B as the latency fallback and as a distillation
student (fine-tune 3B on 7B-quality outputs) if Jetson throughput demands it.
Both fit 8 GB at nf4.

## Rounds 1-2 detail below

## Round 2: research-backed prompts v1 (after docs/style-guide.md)

Fixed by prompting:
- **Generic-advice filler is gone.** outdoor_nav now reports concrete frame
  content ("A blue truck is parked at your left, blocking part of the path").
- **Source attribution in text_reading** ("The chalkboard says… The framed
  sign says…") arrived for free from the example.
- **describe** structure (place → elements → appearance) follows the spec.

Still broken at 3B — candidate fine-tuning targets, and the questions to
answer with a 7B comparison:
- **Occlusion honesty:** even with an explicit worked example ("Fresh co—"),
  the model still silently completed the blocked lunch menu items. Rule +
  example is not enough; needs fine-tuning data (frames with occluded text,
  target output flags the occlusion) or a bigger model.
- **Laterality is unstable run to run:** door reported "on your right" in one
  indoor_nav run and correctly "ahead, slightly left" in another with a
  near-identical prompt; Q&A placed the counter "directly in front" (it's
  right). Confirms the architecture's detection stage should supply verified
  bearings to the LLM prompt rather than trusting VLM spatial reasoning.
- **outdoor_nav now under-reports:** one sentence, missed the tricycle on the
  right and the pedestrian ahead. The single-vehicle example may anchor it;
  try two examples or an explicit checklist pass.
- Minor hallucinations persist ("door slightly open" — it's closed).

Latency unchanged: TTFT 0.8-0.9 s warm, well within tolerance given
streaming.

## Round 1: placeholder prompts

Latency on RTX 3070 (≈ Orin Nano 8 GB budget, though Nano tok/s will be lower):
TTFT ~1 s warm, ~10 tok/s. Streaming works, so TTS could start speaking ~1 s
after frame capture.

## Failure modes observed (targets for prompt work / fine-tuning)

1. **Generic advice instead of observation** (outdoor_nav): model produced
   "Stay alert for vehicles, cyclists, and pedestrians… Pay attention to the
   traffic light if visible" — safety-flavored filler describing nothing in
   the actual frame (which had a truck ahead-left, a motor tricycle right, a
   pedestrian ahead). Worst possible failure class for a blind user: sounds
   helpful, contains zero information. Prompt must forbid advice/imperatives
   and demand concrete, frame-specific observations.

2. **No admission of occlusion** (text_reading): the menu board's lunch items
   are partially blocked by a framed sign; the model silently completed them
   into plausible-sounding dishes instead of saying the text was cut off,
   despite the prompt rule. 3B instruction-following isn't strong enough for
   a negative rule stated once — needs few-shot examples or fine-tuning.
   Otherwise text reading was near-verbatim and well-ordered.

3. **Left/right reliability suspect** (indoor_nav + Q&A): counter/stove is on
   the user's right in the frame; model said "to your left". Lateral errors
   are actively harmful in navigation. Needs eval set with ground-truth
   sides; consider fine-tune emphasis or a detection-model cross-check
   (the vision-processing stage in the architecture could supply verified
   object bearings to the LLM).

4. **Context injection works mechanically** (same description with/without
   context, slightly more confident phrasing) but the placeholder prompts
   don't yet tell the model *how* to use context. Style guide should define:
   context disambiguates and localizes ("the crossing at 5th Ave"), never
   overrides what the frame shows.

## What already works

- Verbatim menu reading with prices, correct ordering, multi-source
  attribution (chalkboard vs framed sign).
- Direct answers to on-demand questions without preamble.
- Verbosity budgets respected per mode.
- No AI/camera meta-talk after the system prompt forbids it.
