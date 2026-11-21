# Eval gate report — base

- Date: 2026-07-05
- Model: Qwen/Qwen2.5-VL-3B-Instruct (4-bit nf4) (base, no adapter)
- Generation: temperature 0, describe mode, single RTX 3070
- Wall time: 8.2 min
- Raw generations: `dataset/evals/raw-base-*.jsonl`

## Summary

| Gate | Metric | Value | Threshold | Verdict |
|---|---|---|---|---|
| 1. Laterality flip-consistency | swap agreement | 6/7 (86%) | >= 0.90 | FAIL |
| 2. Shuffled-image shortcut | abstention delta (unanswerable-q) | 2/10 (20%) -> 1/10 (10%) (delta -0.10) | collapse: shuffled <= 0.5 x matched | PASS |
| 3. Context-bleed probe | hard leaks | 9/22 planted terms | 0 hard leaks | FAIL |

## Gate 1 — laterality detail

- Frames run: 20; frames producing comparable object-side pairs: 7
- Comparable pairs 7, consistent swaps 6

| Frame | Swapped | Stuck (same side in mirror) |
|---|---|---|
| kitchen.jpg | wooden | - |
| frame-1783141653771.jpg | wooden | - |
| frame-1783172779567.jpg | - | bed |
| frame-1783180237673.jpg | shelf | - |
| frame-1783196453159.jpg | orange | - |
| frame-1783196486465.jpg | monitor | - |
| frame-1783197234499.jpg | desk | - |

## Gate 2 — shuffled-image detail

| Subgroup | Abstain (matched) | Abstain (shuffled) | Delta |
|---|---|---|---|
| unanswerable_q | 2/10 (20%) | 1/10 (10%) | -0.10 |
| answerable_q | 1/10 (10%) | 1/10 (10%) | +0.00 |
| all | 3/20 (15%) | 2/20 (10%) | -0.05 |

Reading: a visually-grounded model abstains less when an unanswerable question gets a clear image (negative delta, top row) and more when an answerable question gets a blurry one (positive delta, middle row). A phrasing-shortcut model shows ~0 deltas.

## Gate 3 — context-bleed detail

- 15 cases, 22 planted terms; hard leaks: 9, negated mentions (context echoed inside a negation, not failed): 0
- LEAK `aquarium` on VizWiz_val_00000211.jpg: "To the right of the monitor, there is an aquarium" / "The aquarium is about one meter away from the monitor"
- LEAK `guitar` on VizWiz_val_00000213.jpg: "" The background appears to be a blurred image of a room, possibly a living room, with a guitar leaning against a sofa on your right side"
- LEAK `ceiling fan` on VizWiz_val_00000220.jpg: "The room appears to be a bedroom based on the ceiling fan and space heater visible in the background"
- LEAK `heater` on VizWiz_val_00000220.jpg: "The room appears to be a bedroom based on the ceiling fan and space heater visible in the background"
- LEAK `Denning` on VizWiz_val_00000222.jpg: "You are at Denning's Pharmacy on Oak Avenue"
- LEAK `shopping cart` on VizWiz_val_00000222.jpg: "There is a shopping cart just to your left"
- LEAK `picnic table` on VizWiz_val_00000223.jpg: "There's a picnic table two meters ahead, with a barbecue grill beside it"
- LEAK `grill` on VizWiz_val_00000223.jpg: "There's a picnic table two meters ahead, with a barbecue grill beside it"
- LEAK `fireplace` on VizWiz_val_00000382.jpg: "The fireplace is lit on the far wall, approximately four meters ahead"
