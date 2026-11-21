"""Latency experiment sweep against the Orin's llama-server.

Uses non-streaming requests so llama.cpp's own `timings` block gives exact
prompt-eval (prefill) vs predict (decode) numbers, separating vision+prefill
cost from decode cost per experiment.

  uv run python bench/latency_sweep.py --tag baseline
  uv run python bench/latency_sweep.py --tag fa --cache-prompt --sizes 896,768,640
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


def frame_b64(path: str, max_side: int) -> str:
    img = Image.open(path)
    img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def run_once(image, mode_name, max_side, cache_prompt):
    m = MODES[mode_name]
    msgs = build_messages(m, image, None, None)
    oai = [msgs[0], {"role": "user", "content": [
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{frame_b64(image, max_side)}"}},
        {"type": "text", "text": msgs[1]["content"][1]["text"]}]}]
    payload = {"messages": oai, "max_tokens": m.max_new_tokens, "temperature": 0,
               "stream": False, "cache_prompt": cache_prompt, "timings_per_token": False}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    wall = time.time() - t0
    t = out.get("timings", {})
    return {
        "wall_s": round(wall, 2),
        "prompt_n": t.get("prompt_n"),
        "prefill_s": round(t.get("prompt_ms", 0) / 1000, 2),
        "prefill_tps": round(t.get("prompt_per_second") or 0, 1),
        "decode_n": t.get("predicted_n"),
        "decode_tps": round(t.get("predicted_per_second") or 0, 1),
        "text": out["choices"][0]["message"]["content"].strip(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--sizes", default="896")
    p.add_argument("--cache-prompt", action="store_true")
    p.add_argument("--repeat", type=int, default=2,
                   help="repeats per cell; with --cache-prompt the 2nd run shows warm-prefix gain")
    p.add_argument("--cells", default="crossing.jpg:outdoor_nav,kitchen.jpg:indoor_nav")
    p.add_argument("--out", default="bench/latency_sweep.jsonl")
    args = p.parse_args()

    cells = [c.split(":") for c in args.cells.split(",")]
    with open(args.out, "a") as f:
        for size in [int(s) for s in args.sizes.split(",")]:
            for img, mode in cells:
                for rep in range(args.repeat):
                    r = run_once(f"test_frames/{img}", mode, size, args.cache_prompt)
                    row = {"tag": args.tag, "size": size, "mode": mode, "rep": rep,
                           "cache_prompt": args.cache_prompt, **r}
                    f.write(json.dumps(row) + "\n")
                    print(f"[{args.tag} {size}px {mode} rep{rep}] wall {r['wall_s']}s | "
                          f"prefill {r['prefill_s']}s ({r['prompt_n']} tok @ {r['prefill_tps']}/s) | "
                          f"decode {r['decode_n']} tok @ {r['decode_tps']}/s")


if __name__ == "__main__":
    main()
