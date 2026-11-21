"""Shared plumbing for the v2 eval gates.

Design constraints for the eval gates:
- one model loaded at a time (8 GB VRAM budget) -> load_vlm() is called once
  by run_all.py and the vlm object is passed into every gate;
- N <= 20 cases per gate;
- deterministic scoring (regex, temperature=0) so reports are auditable and
  re-scoreable from the raw jsonl without touching the GPU.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # dataset/evals
DATASET = HERE.parent                            # dataset/
ROOT = DATASET.parent                            # repo root

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DATASET))

# Reuse the audited abstention regex from the v1 scoring pass — one source of
# truth for what counts as abstention language.
from score_eval import ABSTAIN_RE, genuine_unanswerable  # noqa: E402

FRAMES_EVAL = DATASET / "frames_eval"
TEST_FRAMES = ROOT / "test_frames"
SESSION_FRAMES = ROOT / "demo" / "sessions"
VAL_JSON = DATASET / "val.json"

DEFAULT_ADAPTER = DATASET / "sft" / "out-qlora-3b"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def load_vlm(model: str, adapter: str | None = None):
    """Load the 4-bit base VLM, optionally with the QLoRA adapter (peft)."""
    from veris.model import VLM  # slow import, deferred

    vlm = VLM(model_id=MODEL_ID, quantize=True)
    if model == "tuned":
        from peft import PeftModel

        vlm.model = PeftModel.from_pretrained(
            vlm.model, str(adapter or DEFAULT_ADAPTER)
        )
        print("adapter loaded", file=sys.stderr)
    return vlm


def generate(vlm, messages, max_new_tokens: int) -> str:
    return "".join(vlm.generate_stream(messages, max_new_tokens)).strip()


def val_annotations() -> dict:
    return {a["image"]: a for a in json.load(VAL_JSON.open())}


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()]


def session_frames_spaced(min_gap_s: float = 30.0) -> list[Path]:
    """Held-out C920 session frames, deduped: bursts collapse to their first
    frame (timestamps are epoch ms in the filename)."""
    frames = sorted(SESSION_FRAMES.glob("frame-*.jpg"))
    picked, last = [], None
    for f in frames:
        try:
            ts = int(f.stem.split("-")[1]) / 1000.0
        except (IndexError, ValueError):
            continue
        if last is None or ts - last >= min_gap_s:
            picked.append(f)
            last = ts
    return picked


def pct(num: int, den: int) -> str:
    return f"{num}/{den}" + (f" ({num / den:.0%})" if den else "")
