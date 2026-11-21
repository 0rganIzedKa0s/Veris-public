"""Build the SFT training set from judged teacher outputs.

  uv run python dataset/build_sft_set.py

- keep  -> teacher_text is the target
- fix   -> judge's corrected_text is the target (style repaired)
- drop  -> excluded
- Spatial examples (mention left/right) are mirror-augmented: horizontally
  flipped image + left<->right swapped text (the flip-consistency trick for
  laterality). Doubles spatial data for free.

Session flywheel (demo/sessions/*.jsonl, synced from the Orin by
dataset/sync_sessions.py): user-rated on-device outputs are mined too —
thumbs-up outputs become SFT targets verbatim; thumbs-down (or a change-note)
outputs go to dataset/rewrite_queue.jsonl for later correction. Deduped by
(frame sha256, resolved user text) across sessions and against the existing
training set. No LLM is called here.

Output: dataset/sft/veris_sft.json (LLaMA-Factory sharegpt format with image
paths) + dataset/sft/images/ (originals + _flip variants) +
dataset/rewrite_queue.jsonl. `--dry-run` prints counts and writes nothing.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from veris.prompts import MODES, build_messages  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "sft"
IMG_OUT = OUT / "images"
SESSIONS_DIR = HERE.parent / "demo" / "sessions"
REWRITE_QUEUE = HERE / "rewrite_queue.jsonl"

_SWAP = re.compile(r"\b(left|right)\b", re.I)


def mirror_text(t: str) -> str:
    def sub(m):
        w = m.group(0)
        repl = "right" if w.lower() == "left" else "left"
        return repl.capitalize() if w[0].isupper() else repl
    return _SWAP.sub(sub, t)


# ---------------------------------------------------------------------------
# Session-log flywheel: mine user-rated on-device outputs from demo sessions.
# ---------------------------------------------------------------------------

def frame_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_session_records(path: Path) -> list[dict]:
    """Parse one session JSONL, skipping blank/corrupt lines (the demo hub
    appends live, so a truncated last line is normal)."""
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records


def join_ratings(records: list[dict]) -> tuple[list[tuple[dict, list[dict]]], int]:
    """Join kind:"rating" feedback records to the generation each one rates.

    The UI posts {output, rating, note} with no frame reference, so the join
    key is the output text: the nearest *preceding* generation record whose
    output matches. A rating with no output (sent before any description)
    joins the nearest preceding generation of any output. Returns
    ([(generation, [feedback, ...]), ...] in file order, n_unmatched).
    """
    feedback: dict[int, list[dict]] = {}
    unmatched = 0
    for i, rec in enumerate(records):
        if rec.get("kind") != "rating":
            continue
        target_idx = None
        for j in range(i - 1, -1, -1):
            cand = records[j]
            if cand.get("kind") == "rating" or "output" not in cand:
                continue
            if rec.get("output") and cand.get("output") != rec["output"]:
                continue
            target_idx = j
            break
        if target_idx is None:
            unmatched += 1
            continue
        feedback.setdefault(target_idx, []).append(rec)
    return [(records[i], fb) for i, fb in sorted(feedback.items())], unmatched


def existing_keys(examples: list[dict]) -> set[tuple[str, str]]:
    """(frame sha256, user text) keys for examples already in the training set."""
    seen = set()
    for ex in examples:
        img = IMG_OUT / Path(ex["images"][0]).name
        if not img.exists():
            continue
        q = ex["messages"][1]["content"].removeprefix("<image>")
        seen.add((frame_hash(img), q))
    return seen


def collect_session_data(sessions_dir: Path, seen: set[tuple[str, str]]):
    """Mine rated session generations into SFT candidates + a rewrite queue.

    - thumbs-up -> the on-device output is the SFT target, verbatim
    - thumbs-down, or a change-note without a thumbs-up -> rewrite-queue entry
      for later correction (nothing here calls an LLM)
    Dedup key is (frame sha256, resolved user text); `seen` is pre-seeded from
    the existing training set and mutated, so duplicates across sessions and
    against existing data are dropped. Returns (ups, queue, counts).
    """
    ups, queue = [], []
    counts = {"sessions": 0, "generations": 0, "feedback": 0, "unmatched": 0,
              "sft": 0, "queued": 0, "dup": 0, "missing_frame": 0, "empty": 0}
    for f in sorted(sessions_dir.glob("session-*.jsonl")):
        records = load_session_records(f)
        if not records:
            continue
        counts["sessions"] += 1
        counts["generations"] += sum(
            1 for r in records if r.get("kind") != "rating" and "output" in r)
        joined, unmatched = join_ratings(records)
        counts["unmatched"] += unmatched
        for gen, fbs in joined:
            counts["feedback"] += len(fbs)
            frame_name = gen.get("frame") or ""
            frame_src = sessions_dir / frame_name
            if not frame_name or not frame_src.exists():
                counts["missing_frame"] += 1
                continue
            output = (gen.get("output") or "").strip()
            if not output:
                counts["empty"] += 1
                continue
            m = MODES.get(gen.get("mode"), MODES["describe"])
            variant_keys = tuple(v for v in (gen.get("variants") or "").split(",") if v)
            msgs = build_messages(m, str(frame_src), gen.get("context") or None,
                                  gen.get("question") or None, variant_keys)
            user_text = msgs[1]["content"][1]["text"]
            key = (frame_hash(frame_src), user_text)
            if key in seen:
                counts["dup"] += 1
                continue
            seen.add(key)
            ratings = {fb.get("rating") for fb in fbs}
            notes = [n for fb in fbs if (n := (fb.get("note") or "").strip())]
            if "down" in ratings or (notes and "up" not in ratings):
                queue.append({
                    "session": f.name, "ts": gen.get("ts"), "frame": frame_name,
                    "frame_sha256": key[0], "mode": gen.get("mode"),
                    "context": gen.get("context", ""),
                    "question": gen.get("question", ""), "user_text": user_text,
                    "output": output,
                    "rating": "down" if "down" in ratings else None,
                    "notes": notes,
                })
                counts["queued"] += 1
            elif "up" in ratings:
                ups.append({"frame_src": frame_src, "system": msgs[0]["content"],
                            "user_text": user_text, "target": output})
                counts["sft"] += 1
    return ups, queue, counts


def main():
    ap = argparse.ArgumentParser(description="Build the Veris SFT set")
    ap.add_argument("--dry-run", action="store_true",
                    help="mine sessions and print the count summary; write nothing")
    ap.add_argument("--sessions", type=Path, default=SESSIONS_DIR,
                    help="session-log directory (default demo/sessions)")
    args = ap.parse_args()
    dry = args.dry_run

    OUT.mkdir(exist_ok=True)
    IMG_OUT.mkdir(exist_ok=True)
    judged = [json.loads(l) for l in (HERE / "teacher_judged.jsonl").open()]
    mode = MODES["describe"]
    system = mode.system_prompt

    examples, n_flip = [], 0
    # rescued abstention/honesty targets (from judge-dropped hallucinations)
    rescued_path = HERE / "abstention_rescued.jsonl"
    rescued = [json.loads(l) for l in rescued_path.open()] if rescued_path.exists() else []
    for r in rescued:
        src = HERE / "frames" / r["image"]
        if not src.exists() or not (r.get("target") or "").strip():
            continue
        dst = IMG_OUT / r["image"]
        if not dst.exists() and not dry:
            dst.write_bytes(src.read_bytes())
        user_text = r.get("question") or mode.user_prompt.format(context="")
        examples.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "<image>" + user_text},
                {"role": "assistant", "content": r["target"]},
            ],
            "images": [f"images/{r['image']}"],
        })
    n_rescued = len(examples)

    for row in judged:
        if row.get("verdict") == "drop":
            continue
        target = row.get("corrected_text") if row.get("verdict") == "fix" else row["teacher_text"]
        if not target or not target.strip():
            continue
        src = HERE / "frames" / row["image"]
        if not src.exists():
            continue
        dst = IMG_OUT / row["image"]
        if not dst.exists() and not dry:
            dst.write_bytes(src.read_bytes())
        user_text = row["question"] or mode.user_prompt.format(context="")
        examples.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "<image>" + user_text},
                {"role": "assistant", "content": target},
            ],
            "images": [f"images/{row['image']}"],
        })
        # mirror augmentation for spatial examples
        if _SWAP.search(target):
            flip_name = row["image"].replace(".jpg", "_flip.jpg")
            flip_path = IMG_OUT / flip_name
            if not flip_path.exists() and not dry:
                Image.open(src).transpose(Image.FLIP_LEFT_RIGHT).save(flip_path, quality=92)
            examples.append({
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "<image>" + mirror_text(user_text)},
                    {"role": "assistant", "content": mirror_text(target)},
                ],
                "images": [f"images/{flip_name}"],
            })
            n_flip += 1

    # --- session flywheel: user-rated on-device outputs ---
    seen = existing_keys(examples)
    n_session = 0
    counts = {"sessions": 0, "generations": 0, "feedback": 0, "unmatched": 0,
              "sft": 0, "queued": 0, "dup": 0, "missing_frame": 0, "empty": 0}
    queue = []
    if args.sessions.is_dir():
        ups, queue, counts = collect_session_data(args.sessions, seen)
        for u in ups:
            name = u["frame_src"].name
            dst = IMG_OUT / name
            if not dst.exists() and not dry:
                dst.write_bytes(u["frame_src"].read_bytes())
            examples.append({
                "messages": [
                    {"role": "system", "content": u["system"]},
                    {"role": "user", "content": "<image>" + u["user_text"]},
                    {"role": "assistant", "content": u["target"]},
                ],
                "images": [f"images/{name}"],
            })
            n_session += 1
            if _SWAP.search(u["target"]):
                flip_name = name.replace(".jpg", "_flip.jpg")
                flip_path = IMG_OUT / flip_name
                if not flip_path.exists() and not dry:
                    Image.open(u["frame_src"]).transpose(
                        Image.FLIP_LEFT_RIGHT).save(flip_path, quality=92)
                examples.append({
                    "messages": [
                        {"role": "system", "content": u["system"]},
                        {"role": "user", "content": "<image>" + mirror_text(u["user_text"])},
                        {"role": "assistant", "content": mirror_text(u["target"])},
                    ],
                    "images": [f"images/{flip_name}"],
                })
                n_flip += 1

    if not dry:
        (OUT / "veris_sft.json").write_text(json.dumps(examples, indent=1))
        with REWRITE_QUEUE.open("w") as f:
            for row in queue:
                f.write(json.dumps(row) + "\n")

    kept = len(examples) - n_flip - n_rescued - n_session
    print(f"SFT set: {len(examples)} examples ({kept} keep/fix + {n_rescued} rescued-abstention "
          f"+ {n_session} session-rated + {n_flip} mirrored), from {len(judged)} judged outputs")
    print(f"sessions: {counts['sessions']} files, {counts['generations']} generations, "
          f"{counts['feedback']} feedback records joined -> {counts['sft']} thumbs-up SFT, "
          f"{counts['queued']} queued for rewrite; skipped {counts['dup']} dup, "
          f"{counts['missing_frame']} missing-frame, {counts['empty']} empty-output, "
          f"{counts['unmatched']} unmatched ratings")
    if dry:
        print("dry run: nothing written")
    else:
        print(f"wrote {OUT/'veris_sft.json'} and {REWRITE_QUEUE} ({len(queue)} entries)")


if __name__ == "__main__":
    main()
