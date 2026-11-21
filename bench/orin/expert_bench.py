"""Time the experts on a running llama-server (the board, or any other).

Measures what the latency targets are written in: time to first token.

    steady    each expert at each image budget, a different frame every
              request, so nothing but the system prefix can be cached
    switch    alternating experts on fresh frames: is the first request
              after a switch slower than the second?
    planner   the bare base, text only (the airgapped second LLM)

Run it ON the board. Over the USB network link a request of an unlucky size
stalls before it reaches the server, which looks like a hung model. So the
frames are encoded here first, copied over, and the client (stdlib only) runs
next to the server:

    .venv/bin/python -m bench.orin.expert_bench prep --frames data/coco/images/val
    bash bench/orin/run_expert_bench.sh --n 8 --out orin_experts
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from veris.experts import EXPERTS, LlamaClient  # noqa: E402

# a packet of realistic length, so prefill includes what production prefills
SENSE = ("<sense>\nobjects: person 12 o'clock 3 m approaching (in path); "
         "pole 1 o'clock within reach (in path); car 10 o'clock 8 m; "
         "bench 2 o'clock 4 m\nplace: street sidewalk; if visible: a crossing [GPS]\n</sense>")
PLANNER_FACTS = ("Facts: stationary; on Main St facing north; crossing with "
                 "signals 14 m ahead; bus stop 30 m ahead on the right; cafe entrance "
                 "8 m to the left. Write one sentence of context and name the mode.")


SIDES = (448, 560, 768)


def prep(src: Path, dst: Path, count: int) -> None:
    """Encode the first `count` frames at each budget into dst/<side>/."""
    import cv2
    for side in SIDES:
        (dst / str(side)).mkdir(parents=True, exist_ok=True)
        for p in sorted(src.glob("*.jpg"))[:count]:
            img = cv2.imread(str(p))
            s = side / max(img.shape[:2])
            if s < 1:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst / str(side) / p.name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("encoded", count, "frames x", SIDES, "->", dst)


def jpeg(path: Path, side: int) -> bytes:
    """`path` is a frame name under <frames>/448; the same name exists per side."""
    return (path.parent.parent / str(side) / path.name).read_bytes()


def summarise(rows: list[dict]) -> dict:
    t = sorted(r["ttft_s"] for r in rows)
    rate = [r["tokens"] / (r["total_s"] - r["ttft_s"]) for r in rows
            if r["tokens"] > 1 and r["total_s"] > r["ttft_s"]]
    return {"n": len(rows),
            "ttft_median_s": round(statistics.median(t), 3),
            "ttft_min_s": round(t[0], 3), "ttft_max_s": round(t[-1], 3),
            "tok_s": round(statistics.median(rate), 1) if rate else None,
            "total_median_s": round(statistics.median(r["total_s"] for r in rows), 2)}


def one(c: LlamaClient, expert: str, img: bytes | None, max_tokens: int = 40) -> dict:
    text = c.run(expert, img, SENSE if img else "", PLANNER_FACTS if not img else None,
                 max_tokens=max_tokens)
    return {"ttft_s": c.last.ttft_s, "total_s": c.last.total_s,
            "tokens": c.last.tokens, "text": text}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "prep"])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--frames", default=str(ROOT / "data/bench_frames"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--sides", type=int, nargs="+", default=list(SIDES))
    ap.add_argument("--out", default="orin_experts")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    if a.cmd == "prep":
        return prep(Path(a.frames), ROOT / "data/bench_frames", 160)

    frames = sorted((Path(a.frames) / "448").glob("*.jpg"))
    need = a.n * (len(a.sides) * 3 + 4) + 2
    assert len(frames) >= need, f"need {need} distinct frames, found {len(frames)}"
    fresh = iter(frames)                      # every request gets an unseen frame
    c = LlamaClient(a.url, timeout=300)
    res: dict = {"_meta": {"url": a.url, "adapters": c.adapter_ids(), "note": a.note,
                           "time": time.strftime("%Y-%m-%d %H:%M"),
                           "command": " ".join(["bench/orin/" + Path(sys.argv[0]).name, *sys.argv[1:]])}}

    one(c, "outdoor", jpeg(next(fresh), 448))     # warm-up, discarded

    for expert in ("outdoor", "indoor", "reader"):
        for side in a.sides:
            rows = [one(c, expert, jpeg(next(fresh), side)) for _ in range(a.n)]
            res[f"{expert}@{side}"] = summarise(rows)
            res[f"{expert}@{side}"]["sample"] = rows[-1]["text"]
            print(f"{expert}@{side}", res[f"{expert}@{side}"], flush=True)

    # switch cost: A A B B A A B B ... on fresh frames at one budget
    first, second = [], []
    seq = ["outdoor", "indoor"] * a.n
    for expert in seq:
        first.append(one(c, expert, jpeg(next(fresh), 448)))
        second.append(one(c, expert, jpeg(next(fresh), 448)))
    res["switch"] = {"first_after_switch": summarise(first),
                     "second_same_expert": summarise(second)}
    res["switch"]["cost_s"] = round(res["switch"]["first_after_switch"]["ttft_median_s"]
                                    - res["switch"]["second_same_expert"]["ttft_median_s"], 3)
    print("switch", res["switch"], flush=True)

    rows = [one(c, "planner", None, max_tokens=60) for _ in range(a.n)]
    res["planner_text_only"] = summarise(rows)
    res["planner_text_only"]["sample"] = rows[-1]["text"]
    print("planner", res["planner_text_only"], flush=True)

    out = ROOT / "evals/results" / f"{a.out}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
