"""CLI: describe still frames as the wearable device would.

Examples:
  uv run veris frame.jpg --mode outdoor_nav
  uv run veris frame.jpg --mode outdoor_nav --context "Main St heading north, approaching 5th Ave intersection"
  uv run veris frame.jpg --question "Is there an empty seat?"
  uv run veris frames/*.jpg --mode indoor_nav --speak
"""

import argparse
import shutil
import subprocess
import sys

from .prompts import MODES, build_messages


def main() -> int:
    p = argparse.ArgumentParser(prog="veris", description="Assistive frame describer (demo)")
    p.add_argument("images", nargs="+", help="Still frame(s) to describe")
    p.add_argument("--mode", choices=sorted(MODES), default="describe")
    p.add_argument("--context", help="Simulated context prompt from the phone/cloud engine")
    p.add_argument("--question", help="Free-form on-demand question (overrides mode's default)")
    p.add_argument("--variant", action="append", default=[],
                   choices=["clock_face", "numeric_confidence", "imperial"],
                   help="Per-user style variant (repeatable)")
    p.add_argument("--model", default=None, help="HF model id (default: Qwen2.5-VL-3B-Instruct)")
    p.add_argument("--no-quant", action="store_true", help="Load full bf16 instead of 4-bit")
    p.add_argument("--max-tokens", type=int, default=None, help="Override mode's token budget")
    p.add_argument("--speak", action="store_true", help="Pipe output through espeak-ng if installed")
    args = p.parse_args()

    from .model import DEFAULT_MODEL, VLM  # deferred: torch import is slow

    mode = MODES[args.mode]
    print(f"loading {args.model or DEFAULT_MODEL} ({'4-bit' if not args.no_quant else 'bf16'})...",
          file=sys.stderr)
    vlm = VLM(model_id=args.model or DEFAULT_MODEL, quantize=not args.no_quant)

    speak = args.speak and shutil.which("espeak-ng")
    if args.speak and not speak:
        print("(espeak-ng not found; printing only)", file=sys.stderr)

    for image in args.images:
        if len(args.images) > 1:
            print(f"\n== {image} [{mode.name}] ==")
        messages = build_messages(mode, image, args.context, args.question,
                                  variant_keys=tuple(args.variant))
        parts = []
        for chunk in vlm.generate_stream(
            messages, max_new_tokens=args.max_tokens or mode.max_new_tokens,
            temperature=mode.temperature,
        ):
            parts.append(chunk)
            print(chunk, end="", flush=True)
        print()
        s = vlm.last_stats
        print(f"  [ttft {s.ttft_s:.2f}s | total {s.total_s:.2f}s | ~{s.tokens_per_s:.0f} tok/s]",
              file=sys.stderr)
        if speak:
            subprocess.run(["espeak-ng", "".join(parts)], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
