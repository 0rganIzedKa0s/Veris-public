"""Rubric judge: gpt-4o-mini (with the image) scores each teacher output
against the style guide before it becomes training data.

  uv run python dataset/judge_outputs.py

Checks per output: (1) no invented objects (judge sees the image), (2) if the
VizWiz annotation says unanswerable, the output must abstain, (3) no advice/
filler ("be careful", "it appears"), (4) egocentric spatial language, (5)
matches human consensus answers when they exist. Verdict keep|fix|drop with a
reason — 'fix' rows get the judge's corrected text as the training target.
Cost: ~$0.10 per 300 images with gpt-4o-mini.
"""

import argparse
import base64
import json
import os
import pathlib
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
_env = HERE.parent / ".env"
for line in _env.read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

RUBRIC = """You are auditing a scene description generated for a BLIND user against their style guide. \
You can see the image; the model's output is below, with the real question a blind person asked \
(if any), whether the dataset marks the image answerable, and human consensus answers.

Verdict rules:
- drop: invents objects/text not in the image; asserts a confident answer the image cannot \
support (blur/wrong framing/occlusion) instead of saying so; contradicts human consensus.
- fix: right content, style violations (advice like "be careful", filler like "it appears", \
camera/AI meta-talk, non-egocentric directions, vague counts). Provide corrected_text.
- keep: accurate and on-style.

HUMAN CONSENSUS IS THE ARBITER. Sighted annotators saw this image; you are working from a \
low-detail copy, so do NOT overrule them with your own read of legibility. Never justify a \
verdict with "the image is too blurry/unclear to confirm" when the humans confirmed it — they \
could see it and you cannot.

1. Output agrees with human consensus -> keep, no matter how specific it is. Being precise is \
the goal, not a risk: "Basil Leaves", "The can contains Coca-Cola", a captcha read as "T36M" \
are all keeps when the humans said the same thing. Allow paraphrase and extra correct detail \
("crushed tomatoes" for "tomatoes", a full brand name for a generic one).
2. Consensus is "unanswerable" (or answerable=0):
   - output declines the uncertain part -> keep. This is the target behaviour, and declining \
AGREES with that consensus. Naming the concrete cause (blur, darkness, framing, occlusion) is \
better still.
       keep: "The flavor is not clearly visible on the packaging."
       keep: "I can't see the mail — a hand is covering most of it."
   - output names a specific identity, brand, or label text anyway -> drop as invention. \
Hedges do not rescue it: "it appears to be X" asserts X.
       drop: "The label says \\"Menu Board\\" on the wall."
       drop: "This appears to be a can of soup."
3. Output contradicts a definite consensus -> drop.
4. No consensus available -> fall back to judging the image, and only then treat an \
unsupportable specific as invention.

Judge honesty against what the humans reported, not answer rate and not your own confidence.

Return JSON: {"verdict": "keep|fix|drop", "reason": "...", "corrected_text": "..." (fix only)}"""


def judge(image_path, gen, question, answerable, human_answers, mode=None):
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    user = {"output": gen, "question": question, "answerable": answerable,
            "human_consensus": human_answers}
    if mode:  # v2 is mode-stratified; style rules differ per mode
        user["mode"] = mode
    payload = {
        "model": "gpt-4o-mini", "max_tokens": 300, "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                {"type": "text", "text": json.dumps(user)}]}],
    }
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    return json.loads(out["choices"][0]["message"]["content"])


def _iter_units(raw):
    """Yield (image, kind, text, question, answerable, human_answers) for both
    schemas: v1 nests several `gens` per image under a `kind`; v2 is flat, one
    row per (image, mode) with the mode standing in as the kind."""
    for row in raw:
        if "gens" in row:                       # v1
            for g in row["gens"]:
                yield (row["image"], g["kind"], g["text"], g.get("question"),
                       row.get("answerable"), row.get("human_answers"))
        else:                                   # v2 (mode-stratified)
            yield (row["image"], row["mode"], row["text"], row.get("question"),
                   row.get("answerable"), row.get("human_answers"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="teacher_raw.jsonl")
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--out", default="teacher_judged.jsonl")
    ap.add_argument("--limit", type=int, default=0,
                    help="judge at most N units (pilot a cost estimate first)")
    a = ap.parse_args()

    raw = [json.loads(l) for l in (HERE / a.raw).open() if l.strip()]
    out_path = HERE / a.out
    done = set()
    if out_path.exists():
        done = {(json.loads(l)["image"], json.loads(l)["kind"])
                for l in out_path.open() if l.strip()}
        print(f"resuming: {len(done)} judged", file=sys.stderr)
    counts = {"keep": 0, "fix": 0, "drop": 0}
    n = 0
    with out_path.open("a") as f:
        for image, kind, text, question, answerable, human in _iter_units(raw):
            img = HERE / a.frames / image
            if (image, kind) in done or not img.exists():
                continue
            if a.limit and n >= a.limit:
                break
            try:
                v = judge(img, text, question, answerable, human, mode=kind)
            except Exception as e:
                print(f"judge error {image}: {e}", file=sys.stderr)
                time.sleep(2)
                continue
            counts[v.get("verdict", "drop")] = counts.get(v.get("verdict", "drop"), 0) + 1
            n += 1
            f.write(json.dumps({"image": image, "kind": kind,
                                "question": question,
                                "teacher_text": text, **v}) + "\n")
            f.flush()
    print("verdicts:", counts)


if __name__ == "__main__":
    main()
