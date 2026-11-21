"""Eval gate: base 3B vs fine-tuned adapter on 30 held-out VizWiz frames.

  uv run python dataset/eval_adapter.py --model base
  uv run python dataset/eval_adapter.py --model tuned --adapter dataset/sft/out-qlora-3b

Generates with the Veris describe prompt, judges with the same gpt-4o-mini
rubric as training curation, prints keep/fix/drop + abstention correctness on
the 10 unanswerable frames. The go/no-go: tuned keep-rate and abstention
recall must beat base.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["base", "tuned"], required=True)
    p.add_argument("--adapter", default=str(HERE / "sft" / "out-qlora-3b"))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out_path = Path(args.out or (HERE / f"eval_{args.model}.jsonl"))

    from veris.model import VLM  # noqa deferred: slow import
    from veris.prompts import MODES, build_messages

    vlm = VLM(model_id="Qwen/Qwen2.5-VL-3B-Instruct", quantize=True)
    if args.model == "tuned":
        from peft import PeftModel
        vlm.model = PeftModel.from_pretrained(vlm.model, args.adapter)
        print("adapter loaded", file=sys.stderr)

    ann = {a["image"]: a for a in json.load(open(HERE / "val.json"))}
    mode = MODES["describe"]
    frames = sorted((HERE / "frames_eval").glob("*.jpg"))
    with out_path.open("w") as f:
        for i, frame in enumerate(frames):
            a = ann.get(frame.name, {})
            q = (a.get("question") or "").strip() or None
            msgs = build_messages(mode, str(frame), None, q)
            t0 = time.time()
            text = "".join(vlm.generate_stream(msgs, mode.max_new_tokens)).strip()
            f.write(json.dumps({"image": frame.name, "question": q,
                                "answerable": a.get("answerable"),
                                "human_answers": [x["answer"] for x in a.get("answers", [])][:3],
                                "output": text, "wall_s": round(time.time() - t0, 2)}) + "\n")
            print(f"[{i+1}/{len(frames)}]", file=sys.stderr)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
