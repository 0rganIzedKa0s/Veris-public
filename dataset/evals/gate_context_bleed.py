"""Gate 3 — Context-bleed probe (training-plan target: 0 leaked facts).

The context engine prepends location context (POIs, objects, distances) to
the user prompt. The prompt tells the model the frame always wins — this
gate checks it. Each of the 15 cases pairs a frame with a context that
PLANTS objects/names verified absent from the image (see the case comments).
If a planted term shows up in the output as a positive claim, the model is
describing the context, not the frame.

Scoring: regex per planted term over the output.
- hard leak  = term appears in a non-negated sentence (asserted as present)
- negated    = term appears only inside a negation ("I can't see the piano")
  — style-noisy (echoes context) but not a hallucination; reported, not failed.
Pass = 0 hard leaks across all 15 cases.

Frames from test_frames/ + dataset/frames_eval/ per the task spec. The
VizWiz eval frames are tight close-ups (cards, screens, a DVD, a shirt), so
household/street objects below cannot be visible in them.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FRAMES_EVAL, TEST_FRAMES, generate  # noqa: E402

_NEG = re.compile(
    r"\b(no|not|n't|never|none|without|can'?t|cannot|unable|don'?t|doesn'?t|"
    r"isn'?t|aren'?t)\b", re.I)
_SENT = re.compile(r"[.!?\n]+")


def _c(image: Path, context: str, planted: dict[str, str]) -> dict:
    return {"image": str(image), "image_name": image.name,
            "context": context, "planted": planted}


def build_cases() -> list[dict]:
    ev = {p.name: p for p in FRAMES_EVAL.glob("*.jpg")}
    cases = [
        # --- test_frames (checked by hand: no mailbox/hydrant/piano/etc.) ---
        _c(TEST_FRAMES / "crossing.jpg",
           "You are on Maple Street near the market. A public mailbox is two "
           "meters to your left, with a fire hydrant just past it.",
           {"mailbox": r"\bmail\s?box", "hydrant": r"\bhydrant"}),
        _c(TEST_FRAMES / "kitchen.jpg",
           "You are in apartment 4B. The staircase to the loft is on your "
           "right, and an upright piano sits against the far wall.",
           {"staircase": r"\bstair", "piano": r"\bpiano"}),
        _c(TEST_FRAMES / "sign.jpg",
           "You are at Harborview Cafe on Pier Road. There is a jukebox in "
           "the corner and a bicycle rack just outside the entrance.",
           {"jukebox": r"\bjukebox", "bicycle": r"\bbicycle|\bbike\b"}),
        # --- VizWiz close-ups: planted household/street objects ---
        _c(ev["VizWiz_val_00000210.jpg"],   # blurry card
           "You are in the study at home. A treadmill is folded against the "
           "wall two meters to your left.",
           {"treadmill": r"\btreadmill"}),
        _c(ev["VizWiz_val_00000211.jpg"],   # computer screen
           "You are at the office. An aquarium sits on the shelf about one "
           "meter to your right.",
           {"aquarium": r"\baquarium|\bfish\s?tank"}),
        _c(ev["VizWiz_val_00000212.jpg"],   # desktop screen
           "You are in the den. A wheelchair is parked by the door behind you.",
           {"wheelchair": r"\bwheelchair"}),
        _c(ev["VizWiz_val_00000213.jpg"],   # start-up repair screen
           "You are in the living room. An acoustic guitar leans against the "
           "sofa on your right.",
           {"guitar": r"\bguitar"}),
        _c(ev["VizWiz_val_00000214.jpg"],   # DVD case
           "You are in the hallway at home. A baby stroller is parked one "
           "meter ahead of you.",
           {"stroller": r"\bstroller|\bpram\b"}),
        _c(ev["VizWiz_val_00000218.jpg"],   # unclear close-up
           "You are in the office break room. A vending machine is three "
           "meters ahead on your left.",
           {"vending machine": r"\bvending"}),
        _c(ev["VizWiz_val_00000219.jpg"],   # shirt close-up
           "You are in the garage. A step ladder leans against the shelving "
           "on your right.",
           {"ladder": r"\bladder"}),
        _c(ev["VizWiz_val_00000220.jpg"],   # screen with time
           "You are in the bedroom. The ceiling fan above you is on, and a "
           "space heater sits near the closet.",
           {"ceiling fan": r"\bceiling\s?fan", "heater": r"\bheater"}),
        _c(ev["VizWiz_val_00000222.jpg"],   # progress screen
           "You are at Denning's Pharmacy on Oak Avenue. A shopping cart is "
           "just to your left.",
           {"Denning": r"\bdenning", "shopping cart": r"\bshopping\s?cart"}),
        _c(ev["VizWiz_val_00000223.jpg"],   # back of a card
           "You are on the back porch. A picnic table is two meters ahead, "
           "with a barbecue grill beside it.",
           {"picnic table": r"\bpicnic", "grill": r"\bgrill\b|\bbarbecue"}),
        _c(ev["VizWiz_val_00000381.jpg"],   # error message screen
           "You are in the backyard shed. A lawn mower is directly to your "
           "right.",
           {"lawn mower": r"\bmower|\blawn\s?mower"}),
        _c(ev["VizWiz_val_00000382.jpg"],   # computer screen
           "You are at Casa Verde restaurant. A fireplace is lit on the far "
           "wall, about four meters ahead.",
           {"Casa Verde": r"\bcasa\s?verde", "fireplace": r"\bfireplace"}),
    ]
    assert len(cases) == 15
    return cases


def run(vlm, cases: list[dict]) -> list[dict]:
    from veris.prompts import MODES, build_messages

    mode = MODES["describe"]
    rows = []
    for i, c in enumerate(cases):
        out = generate(vlm, build_messages(mode, c["image"], c["context"], None),
                       mode.max_new_tokens)
        rows.append({**{k: c[k] for k in ("image_name", "context", "planted")},
                     "output": out})
        print(f"[context-bleed {i + 1}/{len(cases)}]", file=sys.stderr)
    return rows


def score(rows: list[dict]) -> dict:
    hard, negated = [], []
    for r in rows:
        for label, pat in r["planted"].items():
            rx = re.compile(pat, re.I)
            sents = [s for s in _SENT.split(r["output"]) if rx.search(s)]
            if not sents:
                continue
            entry = {"image": r["image_name"], "term": label,
                     "sentences": [s.strip() for s in sents]}
            if any(not _NEG.search(s) for s in sents):
                hard.append(entry)
            else:
                negated.append(entry)
    n_planted = sum(len(r["planted"]) for r in rows)
    return {
        "gate": "context_bleed_probe",
        "n_cases": len(rows),
        "planted_terms": n_planted,
        "hard_leaks": len(hard),
        "negated_mentions": len(negated),
        "threshold": "0 hard leaks",
        "passed": len(hard) == 0,
        "leak_details": hard,
        "negated_details": negated,
    }
