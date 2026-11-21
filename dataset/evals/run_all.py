"""Run every v2 eval gate for a given model/adapter and write a report.

  uv run python dataset/evals/run_all.py --model base
  uv run python dataset/evals/run_all.py --model tuned \
      --adapter dataset/sft/out-qlora-3b --tag v1

Loads the VLM ONCE (8 GB budget: one model at a time), runs:
  gate 1  laterality flip-consistency   (>= 90% swap agreement)
  gate 2  shuffled-image shortcut test  (abstention delta, must collapse)
  gate 3  context-bleed probe           (0 hard leaks on 15 cases)

Outputs, all under dataset/evals/:
  raw-<tag>-<gate>.jsonl   every generation (re-scoreable without a GPU)
  report-<tag>.md          summary table + per-gate detail

Re-score existing raw files without the GPU:  --score-only --tag <tag>
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_ADAPTER, HERE, MODEL_ID, load_vlm, read_jsonl, write_jsonl  # noqa: E402
import gate_context_bleed  # noqa: E402
import gate_laterality  # noqa: E402
import gate_shuffle  # noqa: E402

GATES = [
    ("laterality", gate_laterality),
    ("shuffle", gate_shuffle),
    ("context_bleed", gate_context_bleed),
]


def render_report(tag: str, model: str, adapter: str | None,
                  scores: dict[str, dict], wall_s: float) -> str:
    lat, shf, ctx = scores["laterality"], scores["shuffle"], scores["context_bleed"]
    ok = lambda s: "PASS" if s["passed"] else "FAIL"  # noqa: E731

    lines = [
        f"# Eval gate report — {tag}",
        "",
        f"- Date: {date.today().isoformat()}",
        f"- Model: {MODEL_ID} (4-bit nf4)" + (f" + adapter `{adapter}`" if adapter else " (base, no adapter)"),
        f"- Generation: temperature 0, describe mode, single RTX 3070",
        f"- Wall time: {wall_s / 60:.1f} min",
        f"- Raw generations: `dataset/evals/raw-{tag}-*.jsonl`",
        "",
        "## Summary",
        "",
        "| Gate | Metric | Value | Threshold | Verdict |",
        "|---|---|---|---|---|",
        f"| 1. Laterality flip-consistency | swap agreement | "
        f"{lat['agreement_str']} | {lat['threshold']} | {ok(lat)} |",
        f"| 2. Shuffled-image shortcut | abstention delta "
        f"(unanswerable-q) | {shf['unanswerable_q']['abstain_matched']} -> "
        f"{shf['unanswerable_q']['abstain_shuffled']} "
        f"(delta {shf['unanswerable_q']['delta']:+.2f}) | collapse: shuffled <= 0.5 x matched | {ok(shf)} |",
        f"| 3. Context-bleed probe | hard leaks | "
        f"{ctx['hard_leaks']}/{ctx['planted_terms']} planted terms | "
        f"{ctx['threshold']} | {ok(ctx)} |",
        "",
        "## Gate 1 — laterality detail",
        "",
        f"- Frames run: {lat['n_frames']}; frames producing comparable "
        f"object-side pairs: {lat['frames_with_comparable_pairs']}",
        f"- Comparable pairs {lat['comparable_pairs']}, consistent swaps "
        f"{lat['consistent_swaps']}",
        "",
        "| Frame | Swapped | Stuck (same side in mirror) |",
        "|---|---|---|",
    ]
    for d in lat["details"]:
        if d["comparable"]:
            lines.append(f"| {d['image']} | {', '.join(d['swapped']) or '-'} | "
                         f"{', '.join(d['stuck']) or '-'} |")
    lines += [
        "",
        "## Gate 2 — shuffled-image detail",
        "",
        "| Subgroup | Abstain (matched) | Abstain (shuffled) | Delta |",
        "|---|---|---|---|",
    ]
    for g in ("unanswerable_q", "answerable_q", "all"):
        s = shf[g]
        lines.append(f"| {g} | {s['abstain_matched']} | {s['abstain_shuffled']} "
                     f"| {s['delta']:+.2f} |")
    lines += [
        "",
        "Reading: a visually-grounded model abstains less when an unanswerable "
        "question gets a clear image (negative delta, top row) and more when an "
        "answerable question gets a blurry one (positive delta, middle row). A "
        "phrasing-shortcut model shows ~0 deltas.",
        "",
        "## Gate 3 — context-bleed detail",
        "",
        f"- {ctx['n_cases']} cases, {ctx['planted_terms']} planted terms; "
        f"hard leaks: {ctx['hard_leaks']}, negated mentions (context echoed "
        f"inside a negation, not failed): {ctx['negated_mentions']}",
    ]
    for e in ctx["leak_details"]:
        lines.append(f"- LEAK `{e['term']}` on {e['image']}: "
                     + " / ".join(f"\"{s}\"" for s in e["sentences"]))
    for e in ctx["negated_details"]:
        lines.append(f"- negated `{e['term']}` on {e['image']}: "
                     + " / ".join(f"\"{s}\"" for s in e["sentences"]))
    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["base", "tuned"], required=True)
    p.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    p.add_argument("--tag", default=None, help="report suffix (default: model name)")
    p.add_argument("--score-only", action="store_true",
                   help="re-score existing raw-<tag>-*.jsonl, no GPU")
    args = p.parse_args()
    tag = args.tag or args.model
    adapter = args.adapter if args.model == "tuned" else None

    t0 = time.time()
    raws: dict[str, list] = {}
    if args.score_only:
        for name, _ in GATES:
            raws[name] = read_jsonl(HERE / f"raw-{tag}-{name}.jsonl")
    else:
        vlm = load_vlm(args.model, adapter)
        for name, mod in GATES:
            rows = mod.run(vlm, mod.build_cases())
            raws[name] = rows
            write_jsonl(HERE / f"raw-{tag}-{name}.jsonl", rows)

    scores = {name: mod.score(raws[name]) for name, mod in GATES}
    report = render_report(tag, args.model, adapter, scores, time.time() - t0)
    out = HERE / f"report-{tag}.md"
    out.write_text(report)
    print(json.dumps({n: {k: v for k, v in s.items()
                          if k not in ("details", "leak_details", "negated_details")}
                      for n, s in scores.items()}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
