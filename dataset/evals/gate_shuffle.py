"""Gate 2 — Shuffled-image shortcut test (textual-shortcut ablation).

Naive abstention SFT teaches models to say "I can't tell" from the QUESTION
phrasing, not from looking. The ablation: run each
held-out question twice — once with its real image, once with a random WRONG
image — and compare abstention-language frequency.

Pairing is cross-group on purpose:
- questions from genuinely-unanswerable frames get a clearly ANSWERABLE
  image  -> a grounded model's abstention rate should COLLAPSE;
- questions from answerable frames get an UNANSWERABLE (blurry/misframed)
  image -> a grounded model's abstention rate should RISE.
A phrasing-shortcut model shows ~zero delta in both directions.

Reported metric is the delta; the pass criterion (from the training plan's
"abstention rate must collapse when images are randomized") is applied to
the unanswerable-question subgroup: shuffled rate <= half the matched rate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ABSTAIN_RE,
    FRAMES_EVAL,
    generate,
    genuine_unanswerable,
    pct,
    val_annotations,
)

N_PER_GROUP = 10  # 20 cases total, 40 generations


def build_cases(n_per_group: int = N_PER_GROUP) -> list[dict]:
    ann = val_annotations()
    gu, ans = [], []
    for frame in sorted(FRAMES_EVAL.glob("*.jpg")):
        a = ann.get(frame.name)
        if not a or not (a.get("question") or "").strip():
            continue
        human = [x["answer"] for x in a.get("answers", [])]
        rec = {"image": str(frame), "image_name": frame.name,
               "question": a["question"].strip()}
        if genuine_unanswerable(human):
            rec["group"] = "unanswerable_q"
            gu.append(rec)
        elif a.get("answerable") == 1:
            rec["group"] = "answerable_q"
            ans.append(rec)
    gu, ans = gu[:n_per_group], ans[:n_per_group]
    # Deterministic cross-group wrong images: GU question i gets the image of
    # answerable case i; answerable question i gets the image of GU case i+1
    # (offset so no case can receive an image related to its own).
    for i, c in enumerate(gu):
        c["wrong_image"] = ans[i % len(ans)]["image"]
    for i, c in enumerate(ans):
        c["wrong_image"] = gu[(i + 1) % len(gu)]["image"]
    return gu + ans


def run(vlm, cases: list[dict]) -> list[dict]:
    from veris.prompts import MODES, build_messages

    mode = MODES["describe"]
    rows = []
    for i, c in enumerate(cases):
        out_m = generate(vlm, build_messages(mode, c["image"], None,
                                             c["question"]), mode.max_new_tokens)
        out_s = generate(vlm, build_messages(mode, c["wrong_image"], None,
                                             c["question"]), mode.max_new_tokens)
        rows.append({**{k: c[k] for k in ("image_name", "question", "group")},
                     "wrong_image": Path(c["wrong_image"]).name,
                     "output_matched": out_m, "output_shuffled": out_s})
        print(f"[shuffle {i + 1}/{len(cases)}]", file=sys.stderr)
    return rows


def _rate(rows, key):
    hits = sum(1 for r in rows if ABSTAIN_RE.search(r[key]))
    return hits, len(rows)


def score(rows: list[dict]) -> dict:
    out = {"gate": "shuffled_image_shortcut", "n_cases": len(rows)}
    groups = {"all": rows,
              "unanswerable_q": [r for r in rows if r["group"] == "unanswerable_q"],
              "answerable_q": [r for r in rows if r["group"] == "answerable_q"]}
    for name, g in groups.items():
        m_hit, n = _rate(g, "output_matched")
        s_hit, _ = _rate(g, "output_shuffled")
        out[name] = {
            "abstain_matched": pct(m_hit, n),
            "abstain_shuffled": pct(s_hit, n),
            "delta": round((s_hit - m_hit) / n, 3) if n else None,
        }
    gu = groups["unanswerable_q"]
    m_hit, n = _rate(gu, "output_matched")
    s_hit, _ = _rate(gu, "output_shuffled")
    # Grounded = abstention on unanswerable-phrased questions collapses once
    # the image is actually answerable. Vacuously ungrounded if it never
    # abstained in the matched run (nothing to collapse — flagged, not passed).
    out["threshold"] = "unanswerable_q shuffled rate <= 0.5 * matched rate (matched > 0)"
    out["passed"] = bool(n and m_hit > 0 and s_hit <= 0.5 * m_hit)
    return out
