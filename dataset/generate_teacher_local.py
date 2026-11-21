"""Teacher generation on the dev box (RTX 3070, 7B @ 4-bit via transformers).

Same output schema as generate_teacher.py, no Orin involvement — the board
stays on demo duty. Higher-quality inputs than the edge path (no latency
constraint), which is what training targets should be anyway.

  uv run python dataset/generate_teacher_local.py --limit 300
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from veris.model import VLM  # noqa: E402
from veris.prompts import MODES, build_messages  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--out", default=str(HERE / "teacher_raw.jsonl"))
    args = p.parse_args()

    ann = {a["image"]: a for a in json.load(open(HERE / "val.json"))}
    frames = sorted((HERE / "frames").glob("*.jpg"))[: args.limit]
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["image"] for l in out_path.open() if l.strip()}
        print(f"resuming: {len(done)} done", file=sys.stderr)

    print("loading 7B teacher on the 3070...", file=sys.stderr)
    vlm = VLM(model_id="Qwen/Qwen2.5-VL-7B-Instruct", quantize=True)
    mode = MODES["describe"]

    def gen(image: Path, question: str | None) -> dict:
        msgs = build_messages(mode, str(image), None, question)
        t0 = time.time()
        text = "".join(vlm.generate_stream(msgs, mode.max_new_tokens))
        return {"text": text.strip(), "wall_s": round(time.time() - t0, 2)}

    with out_path.open("a") as f:
        for i, frame in enumerate(frames):
            if frame.name in done:
                continue
            a = ann.get(frame.name, {})
            recs = []
            q = (a.get("question") or "").strip()
            if q:
                recs.append({"kind": "vqa", "question": q, **gen(frame, q)})
            recs.append({"kind": "describe", "question": None, **gen(frame, None)})
            f.write(json.dumps({"image": frame.name, "answerable": a.get("answerable"),
                                "human_answers": [x["answer"] for x in a.get("answers", [])][:3],
                                "gens": recs, "ts": time.time()}) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"[{i}/{len(frames)}] {frame.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
