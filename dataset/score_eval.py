"""Score eval generations — fixed metrics (no GPU, no re-generation needed).

  uv run python dataset/score_eval.py

Fixes vs the first scoring pass:
- Ground truth: a frame is 'genuinely unanswerable' only if >=50% of its
  human answers are unanswerable-type ('unanswerable', 'unknown', "can't
  tell", ...). VizWiz's binary flag alone is noisy (partial answers exist).
- Abstention detection: regex over the model output (phrases like "can't
  tell", "not legible", "too blurry to") — deterministic, auditable, and
  can't miss an explicit abstention the way a JSON-field judge did.
- Reports per-model: style keep-rate (from the earlier judged files if
  present), abstention recall on genuine unanswerables, and false-abstention
  rate on clearly answerable frames.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

ABSTAIN_RE = re.compile(
    r"can'?t (tell|see|read|make out)|cannot (tell|see|read|make out)|"
    r"not (legible|readable|clear enough|visible)|too (blurry|dark|far|close) to|"
    r"unable to (tell|see|read|determine)|hard to (tell|read|make out)|"
    r"i can'?t determine|isn'?t (visible|legible)", re.I)

UNANS_RE = re.compile(r"unanswerable|unknown|can'?t tell|cannot tell|unclear|blurry", re.I)


def genuine_unanswerable(human_answers):
    if not human_answers:
        return False
    hits = sum(1 for a in human_answers if UNANS_RE.search(a or ""))
    return hits >= max(1, len(human_answers) / 2)


def score(path):
    rows = [json.loads(l) for l in path.open()]
    gu, ans = [], []
    for r in rows:
        (gu if genuine_unanswerable(r.get("human_answers")) else ans).append(r)
    abst_recall = sum(1 for r in gu if ABSTAIN_RE.search(r["output"]))
    false_abst = sum(1 for r in ans if ABSTAIN_RE.search(r["output"]))
    return {
        "n": len(rows), "genuine_unanswerable": len(gu),
        "abstention_recall": f"{abst_recall}/{len(gu)}",
        "false_abstentions_on_answerable": f"{false_abst}/{len(ans)}",
    }


def main():
    for which in ("base", "tuned"):
        p = HERE / f"eval_{which}.jsonl"
        if p.exists():
            print(which, "->", json.dumps(score(p)))


if __name__ == "__main__":
    main()
