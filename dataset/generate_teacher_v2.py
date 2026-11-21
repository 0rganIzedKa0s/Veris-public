"""v2 teacher generation — MODE-STRATIFIED (each example under its own mode's
system prompt, fixing v1's describe-only training). Runs on the dev 3070.

  uv run python dataset/generate_teacher_v2.py

Reads dataset/v2_manifest.json (image, mode, question, answerable, answers),
generates with the 7B under the manifest mode's system prompt, appends to
dataset/teacher_raw_v2.jsonl. Resumable.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from veris.model import VLM  # noqa: E402
from veris.prompts import MODES, build_messages  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    man = json.load(open(HERE / "v2_manifest.json"))
    out_path = HERE / "teacher_raw_v2.jsonl"
    done = set()
    if out_path.exists():
        done = {(json.loads(l)["image"], json.loads(l)["mode"]) for l in out_path.open() if l.strip()}
        print(f"resuming: {len(done)} done", file=sys.stderr)

    print("loading 7B teacher...", file=sys.stderr)
    vlm = VLM(model_id="Qwen/Qwen2.5-VL-7B-Instruct", quantize=True)

    with out_path.open("a") as f:
        for i, m in enumerate(man):
            key = (m["image"], m["mode"])
            frame = HERE / "frames_v2" / m["image"]
            if key in done or not frame.exists():
                continue
            mode = MODES[m["mode"]]
            q = (m.get("question") or "").strip() or None
            msgs = build_messages(mode, str(frame), None, q)
            t0 = time.time()
            text = "".join(vlm.generate_stream(msgs, mode.max_new_tokens)).strip()
            f.write(json.dumps({"image": m["image"], "mode": m["mode"], "question": q,
                                "answerable": m.get("answerable"),
                                "human_answers": m.get("answers", []),
                                "text": text, "wall_s": round(time.time() - t0, 2)}) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"[{i}/{len(man)}]", file=sys.stderr)


if __name__ == "__main__":
    main()
