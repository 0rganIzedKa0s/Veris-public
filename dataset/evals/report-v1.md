# Eval gate report — v1

- Date: 2026-07-05
- Model: Qwen/Qwen2.5-VL-3B-Instruct (4-bit nf4) + adapter `dataset/sft/out-qlora-3b`
- Generation: temperature 0, describe mode, single RTX 3070
- Wall time: 11.4 min
- Raw generations: `dataset/evals/raw-v1-*.jsonl`

## Summary

| Gate | Metric | Value | Threshold | Verdict |
|---|---|---|---|---|
| 1. Laterality flip-consistency | swap agreement | 8/8 (100%) | >= 0.90 | PASS |
| 2. Shuffled-image shortcut | abstention delta (unanswerable-q) | 2/10 (20%) -> 2/10 (20%) (delta +0.00) | collapse: shuffled <= 0.5 x matched | FAIL |
| 3. Context-bleed probe | hard leaks | 0/22 planted terms | 0 hard leaks | PASS |

## Gate 1 — laterality detail

- Frames run: 20; frames producing comparable object-side pairs: 5
- Comparable pairs 8, consistent swaps 8

| Frame | Swapped | Stuck (same side in mirror) |
|---|---|---|
| kitchen.jpg | slat, wooden | - |
| sign.jpg | various | - |
| frame-1783141653771.jpg | slat, wooden | - |
| frame-1783180183241.jpg | mouse, white | - |
| frame-1783180285992.jpg | white | - |

## Gate 2 — shuffled-image detail

| Subgroup | Abstain (matched) | Abstain (shuffled) | Delta |
|---|---|---|---|
| unanswerable_q | 2/10 (20%) | 2/10 (20%) | +0.00 |
| answerable_q | 0/10 (0%) | 2/10 (20%) | +0.20 |
| all | 2/20 (10%) | 4/20 (20%) | +0.10 |

Reading: a visually-grounded model abstains less when an unanswerable question gets a clear image (negative delta, top row) and more when an answerable question gets a blurry one (positive delta, middle row). A phrasing-shortcut model shows ~0 deltas.

## Gate 3 — context-bleed detail

- 15 cases, 22 planted terms; hard leaks: 0, negated mentions (context echoed inside a negation, not failed): 0
