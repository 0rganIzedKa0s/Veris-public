"""Benchmark client — drives an OpenAI-compatible VLM server (llama-server on
the Jetson) with the same per-mode prompts as the local pipeline, and records
one JSONL row per generation.

Run from the dev box; the server runs on the Nano:
  uv run python bench/client.py --base-url http://192.168.55.1:8080 \
      --tag "7B-Q4_K_M@25W" --runs 3 --out bench/results.jsonl

Measures: TTFT (first streamed token), total wall time, completion tokens,
tok/s. Pair with tegrastats_logger.py on the Jetson for power/thermals; rows
carry timestamps for the merge.
"""

import argparse
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from veris.prompts import MODES, build_messages  # noqa: E402

DEFAULT_MATRIX = [
    ("test_frames/crossing.jpg", "outdoor_nav"),
    ("test_frames/kitchen.jpg", "indoor_nav"),
    ("test_frames/sign.jpg", "text_reading"),
    ("test_frames/kitchen.jpg", "describe"),
]


def _load_downscaled(image_path: str, max_side: int = 1280) -> bytes:
    """Downscale to <=1280px long side — matches the pipeline's MAX_PIXELS cap
    and keeps the mtmd image compute buffer inside the Nano's 8GB budget."""
    from io import BytesIO
    from PIL import Image
    img = Image.open(image_path)
    img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def to_openai_messages(messages, image_path: str, max_side: int = 1280):
    """Convert transformers-style messages to OpenAI chat format with a data URL."""
    b64 = base64.b64encode(_load_downscaled(image_path, max_side)).decode()
    out = []
    for m in messages:
        if isinstance(m["content"], str):
            out.append(m)
            continue
        content = []
        for part in m["content"]:
            if part["type"] == "image":
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            else:
                content.append(part)
        out.append({"role": m["role"], "content": content})
    return out


def stream_chat(base_url: str, payload: dict):
    """POST /v1/chat/completions with stream=true; yield (chunk_text, usage|None)."""
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            choices = chunk.get("choices") or [{}]
            delta = choices[0].get("delta", {}).get("content")
            usage = chunk.get("usage")
            if delta or usage:
                yield delta or "", usage


def bench_one(base_url: str, image: str, mode_name: str, context: str | None,
              model: str | None = None, max_side: int = 1280):
    mode = MODES[mode_name]
    messages = to_openai_messages(build_messages(mode, image, context, None), image,
                                  max_side)
    payload = {
        "messages": messages,
        "max_tokens": mode.max_new_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if model:
        # vLLM requires a model name; llama-server ignores it. cache_prompt is
        # llama.cpp-specific, so omit it for other engines (for cold numbers on
        # vLLM start the server with --no-enable-prefix-caching).
        payload["model"] = model
    else:
        # every real frame is new — measure cold-prompt TTFT, not cache hits
        payload["cache_prompt"] = False
    text, usage = [], None
    start = time.time()
    ttft = None
    for delta, u in stream_chat(base_url, payload):
        if delta and ttft is None:
            ttft = time.time() - start
        text.append(delta)
        usage = u or usage
    total = time.time() - start
    completion_tokens = (usage or {}).get("completion_tokens")
    return {
        "ts_start": start,
        "ts_end": start + total,
        "image": image,
        "mode": mode_name,
        "ttft_s": round(ttft, 3) if ttft else None,
        "total_s": round(total, 3),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "tok_per_s": round(completion_tokens / (total - ttft), 2)
        if completion_tokens and ttft and total > ttft else None,
        "output": "".join(text).strip(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--tag", required=True,
                   help="Config label recorded on every row, e.g. '7B-Q4_K_M@25W'")
    p.add_argument("--runs", type=int, default=3, help="Repeats per matrix cell")
    p.add_argument("--context", default=None)
    p.add_argument("--out", default="bench/results.jsonl")
    p.add_argument("--warmup", type=int, default=1,
                   help="Unrecorded warmup generations before measuring")
    p.add_argument("--model", default=None,
                   help="Model name for the payload (needed by vLLM; llama-server infers)")
    p.add_argument("--max-side", type=int, default=1280,
                   help="Long-side pixel budget for frames (default 1280)")
    args = p.parse_args()

    for _ in range(args.warmup):
        bench_one(args.base_url, *DEFAULT_MATRIX[0], args.context,
                  args.model, args.max_side)
        print("warmup done", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    with out.open("a") as f:
        for image, mode_name in DEFAULT_MATRIX:
            for run in range(args.runs):
                row = bench_one(args.base_url, image, mode_name, args.context,
                                args.model, args.max_side)
                row.update(tag=args.tag, run=run, max_side=args.max_side)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"{args.tag} {mode_name} run{run}: "
                      f"ttft {row['ttft_s']}s total {row['total_s']}s "
                      f"{row['tok_per_s'] or '?'} tok/s", file=sys.stderr)
    print(f"appended to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
