"""Gate 1 — Laterality flip-consistency (training-plan target: >= 90%).

Run the model on each spatial frame and on its horizontal mirror. A model
that actually *sees* left/right must swap the sides consistently: every
object it places on the left in the original should land on the right in the
mirror. A model guessing sides from language priors will not swap.

Scoring is deterministic: for every occurrence of "left"/"right" we collect
nearby content words (same sentence, +/- WINDOW tokens, stopwords removed)
as the objects being lateralized. An object with an unambiguous side in BOTH
runs is a comparable pair; the pair is consistent iff the sides are opposite.
agreement = consistent / comparable.

Frames: test_frames/ + held-out C920 session frames (never in training),
which are real scenes with spatial layout — VizWiz eval frames are close-ups
and rarely elicit directional language.
"""

import re
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    HERE,
    TEST_FRAMES,
    generate,
    pct,
    session_frames_spaced,
)

FLIP_CACHE = HERE / ".flip_cache"

_SIDE = re.compile(r"\b(left|right)\b", re.I)
_SENT = re.compile(r"[.!?\n]+")
_TOKEN = re.compile(r"[a-z]+")
WINDOW = 5

# Function words + words that describe sides/space rather than objects.
STOP = set(
    """the a an and or but your you yours their his her its of on to in at is
    are was were be been am it this that there here with within about near
    far ahead behind front back side sides left right top bottom upper lower
    just also only very not no nothing none which what when where who while
    can cannot cant could may might will would should has have had one two
    three four five six seven eight nine ten meter meters arm reach hand
    another other each both all some any most more less than from into onto
    against along across toward towards away over under above below next
    close nearby slightly directly roughly out up down""".split()
)


def build_cases(n: int = 20) -> list[Path]:
    frames = sorted(TEST_FRAMES.glob("*.jpg")) + session_frames_spaced()
    return frames[:n]


def flipped_path(frame: Path) -> Path:
    FLIP_CACHE.mkdir(exist_ok=True)
    out = FLIP_CACHE / f"{frame.stem}_flip.jpg"
    if not out.exists():
        Image.open(frame).transpose(Image.FLIP_LEFT_RIGHT).save(out, quality=92)
    return out


def side_pairs(text: str) -> dict[str, str]:
    """Map object-token -> side ('left'/'right'); tokens seen with both
    sides in one output are ambiguous and dropped."""
    seen: dict[str, set] = {}
    for sent in _SENT.split(text.lower()):
        tokens = _TOKEN.findall(sent)
        for i, tok in enumerate(tokens):
            if tok not in ("left", "right"):
                continue
            lo, hi = max(0, i - WINDOW), i + WINDOW + 1
            for w in tokens[lo:hi]:
                if len(w) >= 3 and w not in STOP and w not in ("left", "right"):
                    seen.setdefault(w, set()).add(tok)
    return {w: s.pop() for w, s in seen.items() if len(s) == 1}


def run(vlm, cases: list[Path]) -> list[dict]:
    from veris.prompts import MODES, build_messages

    mode = MODES["describe"]
    rows = []
    for i, frame in enumerate(cases):
        out_orig = generate(vlm, build_messages(mode, str(frame), None, None),
                            mode.max_new_tokens)
        out_flip = generate(vlm, build_messages(mode, str(flipped_path(frame)),
                                                None, None), mode.max_new_tokens)
        rows.append({"image": frame.name, "output_orig": out_orig,
                     "output_flip": out_flip})
        print(f"[laterality {i + 1}/{len(cases)}]", file=sys.stderr)
    return rows


def score(rows: list[dict]) -> dict:
    comparable = consistent = 0
    frames_with_pairs = 0
    details = []
    for r in rows:
        po, pf = side_pairs(r["output_orig"]), side_pairs(r["output_flip"])
        shared = set(po) & set(pf)
        good = {w for w in shared if po[w] != pf[w]}
        if shared:
            frames_with_pairs += 1
        comparable += len(shared)
        consistent += len(good)
        details.append({
            "image": r["image"],
            "comparable": sorted(shared),
            "swapped": sorted(good),
            "stuck": sorted(shared - good),
        })
    agreement = consistent / comparable if comparable else None
    return {
        "gate": "laterality_flip_consistency",
        "n_frames": len(rows),
        "frames_with_comparable_pairs": frames_with_pairs,
        "comparable_pairs": comparable,
        "consistent_swaps": consistent,
        "agreement": agreement,
        "agreement_str": pct(consistent, comparable),
        "threshold": ">= 0.90",
        "passed": (agreement is not None and agreement >= 0.90),
        "details": details,
    }
