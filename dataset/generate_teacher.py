"""Teacher-generation: 7B answers real VizWiz blind-user questions in the
Veris style; outputs become 3B fine-tuning candidates.

  uv run python dataset/generate_teacher.py --limit 300

Per frame we generate:
- an answer to the REAL VizWiz question (context-aware finding: what users
  actually asked is the best training signal), in describe-mode style
- a plain describe-mode description (no question)
VizWiz's unanswerable/blurry images are kept on purpose: the style guide's
"say you can't tell" rule needs positive training examples, and the
annotation's `answerable` flag lets the judge check the teacher actually
abstained.

Requires the 7B server running on the Orin (25W, fresh boot —
~/veris-app/run_7b_server.sh on the board).
"""

import argparse
import base64
import json
import sys
import time
import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from veris.prompts import MODES, build_messages  # noqa: E402

BASE = "http://192.168.55.1:8080"
HERE = Path(__file__).resolve().parent


def frame_b64(path: Path, max_side: int) -> str:
    img = Image.open(path)
    img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def ask(image: Path, question: str | None) -> dict:
    mode = MODES["describe"]
    msgs = build_messages(mode, str(image), None, question)
    oai = [msgs[0], {"role": "user", "content": [
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{frame_b64(image, mode.max_image_side)}"}},
        {"type": "text", "text": msgs[1]["content"][1]["text"]}]}]
    # request shape mirrors the SUCCESSFUL 7B benchmark exactly: streaming,
    # default cache_prompt — the non-streaming variant coincided with crashes
    payload = {"messages": oai, "max_tokens": mode.max_new_tokens,
               "temperature": 0, "stream": True}
    last_err = None
    for attempt in range(3):
        t0 = time.time()
        try:
            req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            parts = []
            with urllib.request.urlopen(req, timeout=600) as r:
                for raw in r:
                    line = raw.decode().strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    choices = json.loads(line[6:]).get("choices") or [{}]
                    d = choices[0].get("delta", {}).get("content")
                    if d:
                        parts.append(d)
            if not parts:
                raise RuntimeError("empty stream")
            return {"text": "".join(parts).strip(),
                    "wall_s": round(time.time() - t0, 2)}
        except Exception as e:
            last_err = e
            print(f"  request failed (attempt {attempt+1}): {e}", file=sys.stderr)
            time.sleep(15 * (attempt + 1))
    raise SystemExit(f"server unreachable after retries: {last_err}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--out", default=str(HERE / "teacher_raw.jsonl"))
    args = p.parse_args()

    ann = {a["image"]: a for a in json.load(open(HERE / "val.json"))}
    frames = sorted((HERE / "frames").glob("*.jpg"))[: args.limit]
    done = set()
    out_path = Path(args.out)
    if out_path.exists():
        done = {json.loads(l)["image"] for l in out_path.open() if l.strip()}
        print(f"resuming: {len(done)} already generated", file=sys.stderr)

    with out_path.open("a") as f:
        for i, frame in enumerate(frames):
            if frame.name in done:
                continue
            a = ann.get(frame.name, {})
            recs = []
            q = (a.get("question") or "").strip()
            if q:
                r = ask(frame, q)
                recs.append({"kind": "vqa", "question": q, **r})
            r = ask(frame, None)
            recs.append({"kind": "describe", "question": None, **r})
            row = {"image": frame.name, "answerable": a.get("answerable"),
                   "human_answers": [x["answer"] for x in a.get("answers", [])][:3],
                   "gens": recs, "ts": time.time()}
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"[{i}/{len(frames)}] {frame.name} "
                      f"({sum(g['wall_s'] for g in recs):.0f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
