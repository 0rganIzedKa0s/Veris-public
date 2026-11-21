"""The airgapped second LLM: the board's base model, no adapter, text only,
working over offline map facts. Two jobs are measured, against the simulator.

    decide   the model re-decides situation and mode as JSON. The rules
             planner (veris/planner.py) already does this with no model, so
             this asks what the model adds on top.
    locate   the model answers "where is the nearest X?" in one sentence from
             the same facts, and must say so when X is not on the map. Rules
             cannot do this; scoring is exact against the facts.

    .venv/bin/python -m evals.planner_llm_eval prompts
    bash bench/orin/run_on_board.sh data/planner_llm evals.planner_llm_eval run decide
    bash bench/orin/run_on_board.sh data/planner_llm evals.planner_llm_eval run locate
    .venv/bin/python -m evals.planner_llm_eval score decide
    .venv/bin/python -m evals.planner_llm_eval score locate
"""

from __future__ import annotations

import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
D = ROOT / "data/planner_llm"
SEEDS = (0, 1)                     # test seeds; planner thresholds were tuned on 100-104
MAX_PER_RUN = 6                    # cap prompts per scenario/seed so no scenario dominates
WHERE = "Jetson Orin Nano 8GB, llama-server, base Q4_K_M, all adapters off, text only"

ABSENT = ["pharmacy", "bank", "toilet", "bus stop", "crossing", "cafe", "elevator",
          "post office", "bench", "entrance"]


def prompts() -> None:
    from sim import gps
    from veris.planner import Planner, llm_prompt, locate_prompt

    D.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    n = m = 0
    with open(D / "prompts.jsonl", "w") as fd, open(D / "locate.jsonl", "w") as fl:
        for name in gps.SCENARIOS:
            for seed in SEEDS:
                pl, k = Planner(), 0
                for r in gps.simulate(name, seed):
                    if r["fix"] is None:
                        continue
                    s = pl.plan(r["fix"])
                    if not pl.changed or k >= MAX_PER_RUN:
                        continue
                    k += 1
                    n += 1
                    tr, f = r["truth"], pl.last_facts
                    rid = f"{name}/{seed}/{int(r['t'])}"
                    fd.write(json.dumps({
                        "id": rid, "prompt": llm_prompt(f),
                        "truth": {"situation": tr["situation"], "mode": tr["mode"],
                                  "indoor": tr["indoor"]},
                        "rules": {"situation": s.situation, "mode": s.suggested_mode},
                    }) + "\n")
                    # two things that are on the map, one that is not
                    first: dict[str, dict] = {}
                    for p in f["nearby"]:
                        first.setdefault(p["types"][0].replace("_", " "), p)
                    for what in rng.sample(sorted(first), min(2, len(first))):
                        p = first[what]
                        fl.write(json.dumps({"id": f"{rid}/{what}", "prompt": locate_prompt(f, f"Where is the nearest {what}?"),
                                             "nearby": f["nearby"],
                                             "expect": {"side": p["side"],
                                                        "distance_m": p["distance_m"]}}) + "\n")
                        m += 1
                    kinds = {t.replace("_", " ") for p in f["nearby"] for t in p["types"]}
                    names = " ".join((p.get("name") or "").lower() for p in f["nearby"])
                    gone = [a for a in ABSENT if a not in kinds and a not in names]
                    if gone:
                        what = rng.choice(gone)
                        fl.write(json.dumps({"id": f"{rid}/absent:{what}",
                                             "prompt": locate_prompt(f, f"Where is the nearest {what}?"),
                                             "nearby": f["nearby"], "expect": None}) + "\n")
                        m += 1
    print(f"{n} decide prompts, {m} locate prompts -> {D}")


def run(task: str, url: str = "http://127.0.0.1:8080") -> None:
    """Board side. Standard library only."""
    from veris.experts import LlamaClient

    src = "prompts.jsonl" if task == "decide" else "locate.jsonl"
    c = LlamaClient(url, timeout=300)
    rows = [json.loads(l) for l in open(D / src)]
    with open(D / f"outputs_{task}.jsonl", "w") as f:
        for i, r in enumerate(rows):
            text = c.run("planner", None, "", r["prompt"],
                         max_tokens=260 if task == "decide" else 40)
            f.write(json.dumps({"id": r["id"], "text": text,
                                "ttft_s": round(c.last.ttft_s, 3),
                                "total_s": round(c.last.total_s, 3),
                                "tokens": c.last.tokens}) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"[{i}/{len(rows)}] ttft {c.last.ttft_s:.2f}s total {c.last.total_s:.1f}s",
                      flush=True)


def _timing(O: list[dict]) -> dict:
    return {"ttft_median_s": round(statistics.median(o["ttft_s"] for o in O), 3),
            "total_median_s": round(statistics.median(o["total_s"] for o in O), 2),
            "tokens_median": statistics.median(o["tokens"] for o in O)}


def score_decide() -> dict:
    from veris.planner import parse_llm

    P = {r["id"]: r for r in map(json.loads, open(D / "prompts.jsonl"))}
    O = [json.loads(l) for l in open(D / "outputs_decide.jsonl")]
    n = len(O)
    parsed = sit_llm = mode_llm = sit_rules = mode_rules = changed = better = worse = 0
    for o in O:
        p = P[o["id"]]
        tr, ru = p["truth"], p["rules"]
        sit_rules += ru["situation"] == tr["situation"]
        mode_rules += ru["mode"] == tr["mode"]
        s = parse_llm(o["text"])
        if s is None:                       # unparseable -> the device keeps the rules
            sit_llm += ru["situation"] == tr["situation"]
            mode_llm += ru["mode"] == tr["mode"]
            continue
        parsed += 1
        sit_llm += s.situation == tr["situation"]
        mode_llm += s.suggested_mode == tr["mode"]
        if s.suggested_mode != ru["mode"]:
            changed += 1
            better += s.suggested_mode == tr["mode"]
            worse += ru["mode"] == tr["mode"]
    return {
        "n": n, "seeds": list(SEEDS),
        "parse_rate": round(parsed / n, 4),
        "situation_acc": {"rules": round(sit_rules / n, 4),
                          "rules_plus_llm": round(sit_llm / n, 4)},
        "mode_acc": {"rules": round(mode_rules / n, 4),
                     "rules_plus_llm": round(mode_llm / n, 4)},
        "llm_changed_mode": changed, "of_which_fixed": better, "of_which_broke": worse,
        **_timing(O),
    }


_SIDE = {"on your left": "left", "on your right": "right", "ahead": "ahead",
         "behind you": "behind"}
_NOT = re.compile(r"\b(not on the map|no\b|not\b|isn't|don't|cannot|can't|none|unknown)", re.I)


def score_locate() -> dict:
    from veris.planner import NOT_ON_MAP, locate_guard

    P = {r["id"]: r for r in map(json.loads, open(D / "locate.jsonl"))}
    O = [json.loads(l) for l in open(D / "outputs_locate.jsonl")]

    def tally(guarded: bool) -> dict:
        pres = both_ok = absent = declined = named_other = invented = 0
        for o in O:
            row = P[o["id"]]
            exp = row["expect"]
            text = locate_guard(o["text"], row, row["prompt"].split("Question:")[1]) if guarded else o["text"]
            t = text.lower()
            nums = [int(x) for x in re.findall(r"\b(\d+)\s*(?:m\b|meters?|metres?)", t)]
            if exp is None:
                absent += 1
                asked = o["id"].split("absent:")[1]
                if not nums and (_NOT.search(t) or text == NOT_ON_MAP):
                    declined += 1
                elif asked not in t:
                    named_other += 1        # true statement about a different mapped thing
                else:
                    invented += 1           # claims the asked-for place exists
                continue
            pres += 1
            both_ok += _SIDE[exp["side"]] in t and exp["distance_m"] in nums
        return {"on_map_n": pres, "on_map_correct": round(both_ok / pres, 4),
                "not_on_map_n": absent, "declined": round(declined / absent, 4),
                "named_another_place": round(named_other / absent, 4),
                "invented_the_place": round(invented / absent, 4)}

    return {
        "n": len(O), "seeds": list(SEEDS),
        "model_alone": tally(False), "with_guard": tally(True),
        **_timing(O),
        "scoring": "correct = side word and exact metre figure match the map facts; "
                   "for absent places: declined / gave facts about a different named "
                   "place / claimed the asked-for place exists",
    }


def score(task: str) -> None:
    res = {"decide": score_decide, "locate": score_locate}[task]()
    res.update(where=WHERE, time=time.strftime("%Y-%m-%d %H:%M"),
               command=f"python -m evals.planner_llm_eval prompts | run {task} (board) | score {task}")
    json.dump(res, open(ROOT / f"evals/results/planner_llm_{task}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prompts":
        prompts()
    else:
        {"run": run, "score": score}[cmd](sys.argv[2])
