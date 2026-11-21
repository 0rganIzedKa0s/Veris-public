"""Convert judge-dropped outputs into abstention/honesty training targets.

The teacher hallucinated on these images; gpt-4o-mini (seeing the image)
writes what SHOULD have been said in the Veris style — usually an abstention
naming the concrete visual cause (blur, framing, darkness, occlusion), plus
whatever CAN honestly be stated. Diverse visual causes are load-bearing:
uniform abstention text teaches phrasing shortcuts, not looking.

  uv run python dataset/abstention_rescue.py
"""

import base64
import json
import os
import pathlib
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
for line in (HERE.parent / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

SYSTEM = """You write GOLD-STANDARD answers for a blind user's assistive device, to replace a \
model output that was rejected for inventing details or answering an unanswerable image. \
You can see the image. Style: second person, present tense, 1-4 short sentences, no advice, \
no filler, no "it appears". If the image cannot support an answer, say you can't tell and name \
the CONCRETE visual reason (too blurry, subject out of frame, too dark, object blocking the \
view, too close to the camera) — then state anything that IS honestly visible. If the question \
IS answerable, answer it accurately and concisely.
Return JSON: {"target": "...", "abstained": true|false, "visual_cause": "blur|framing|dark|occlusion|too_close|none"}"""


def rescue(image_path, question, human_answers):
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    payload = {
        "model": "gpt-4o-mini", "max_tokens": 250, "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                {"type": "text", "text": json.dumps({
                    "question": question or "Describe what's in front of me.",
                    "human_consensus_answers": human_answers})}]}],
    }
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(json.load(r)["choices"][0]["message"]["content"])


def main():
    judged = [json.loads(l) for l in (HERE / "teacher_judged.jsonl").open()]
    raw = {r["image"]: r for r in (json.loads(l) for l in (HERE / "teacher_raw.jsonl").open())}
    drops = [j for j in judged if j["verdict"] == "drop"]
    out_path = HERE / "abstention_rescued.jsonl"
    done = set()
    if out_path.exists():
        done = {(json.loads(l)["image"], json.loads(l)["kind"]) for l in out_path.open()}
    causes = {}
    with out_path.open("a") as f:
        for j in drops:
            key = (j["image"], j["kind"])
            img = HERE / "frames" / j["image"]
            if key in done or not img.exists():
                continue
            try:
                r = rescue(img, j.get("question"), raw.get(j["image"], {}).get("human_answers"))
            except Exception as e:
                print(f"rescue error {j['image']}: {e}", file=sys.stderr)
                time.sleep(2)
                continue
            causes[r.get("visual_cause", "?")] = causes.get(r.get("visual_cause", "?"), 0) + 1
            f.write(json.dumps({"image": j["image"], "kind": j["kind"],
                                "question": j.get("question"), **r}) + "\n")
            f.flush()
    print("visual causes:", causes)


if __name__ == "__main__":
    main()
