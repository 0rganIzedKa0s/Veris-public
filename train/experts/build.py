"""Build the outdoor and indoor experts' training sets by ground-truth-verified
distillation.

    select  pick COCO images per scene, download them, write the manifest
    teach   7B teacher writes a description for each training image, seeing
            the image and its ground-truth object packet; every output is
            checked against the boxes (train/experts/verify.py)
    sft     keep the outputs that passed, add packet variants and mirrored
            twins, write LLaMA-Factory datasets and configs

    .venv/bin/python -m train.experts.build select
    flock /tmp/veris-gpu.lock .venv/bin/python -m train.experts.build teach
    .venv/bin/python -m train.experts.build sft

COCO train2017 feeds training; val2017 is held out for evals/experts_eval.py.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from veris.experts import EXPERTS, user_text  # noqa: E402
from veris.packet import Obj, Packet  # noqa: E402
from train.experts.verify import check  # noqa: E402

COCO = ROOT / "data/coco"
OUT = ROOT / "data/experts"
HFOV_DEG = 70.0          # COCO has no camera metadata; a phone-like assumption

STREET = {"car", "bus", "truck", "traffic light", "stop sign", "fire hydrant",
          "parking meter", "bench", "bicycle", "motorcycle"}
INDOOR_ONLY = {"couch", "bed", "toilet", "tv", "refrigerator", "oven", "sink", "microwave"}
INDOOR = INDOOR_ONLY | {"chair", "dining table", "potted plant", "laptop"}
OUTDOOR_ONLY = STREET - {"bench", "bicycle"} | {"train", "airplane", "boat"}
OBSTACLES = {"person", "bicycle", "motorcycle", "bench", "fire hydrant", "chair",
             "dog", "suitcase", "potted plant", "parking meter", "car"}

_SHARED = """\
You are the eyes of a wearable device for a blind user. You see one frame from \
their head-mounted camera, facing where they face. Follow these rules exactly:
- Speak to the user in second person. Present tense, active voice, short sentences.
- Most important information first.
- Report only what is actually visible. NEVER give advice or warnings. State observations.
- Use counts and specifics: "two people ahead", never "some" or "a few".
- If you cannot determine something, say "I can't tell". Never guess.
- Never mention AI, cameras, images, or photos. No greetings, no filler, no "it appears".
- A detector has already measured where things are. Its list follows the request. \
Use its directions exactly as written for the objects it lists. Do not name a person, \
vehicle, animal or piece of furniture unless it is in that list or plainly visible."""

TEACHER = {
    "outdoor": _SHARED + """

MODE: outdoors. In 2 to 4 short sentences, in this order:
1. Anything moving toward the user or standing in their path.
2. The path itself: where it is clear, and curbs, steps, poles or uneven ground.
3. What kind of place this is and one or two landmarks, only if clear.
Leave out scenery and parked vehicles that do not constrain where the user can walk.""",
    "indoor": _SHARED + """

MODE: indoors. In 2 to 4 short sentences, in this order:
1. What kind of room this is, if clear.
2. Obstacles in the user's way at any height, and openings: doorways, corridors, stairs.
3. Furniture and large objects, nearest first, with their direction.""",
}


def packet_for(gt: list[dict], flip: bool = False, drop: int = 0,
               fake: Obj | None = None, rng: random.Random | None = None) -> Packet:
    """Ground-truth boxes -> the packet a perfect detector would write."""
    top = sorted(gt, key=lambda g: -g["area"])[:6]
    if drop and rng and len(top) > 1:
        for _ in range(min(drop, len(top) - 1)):
            top.pop(rng.randrange(len(top)))
    objs = []
    for g in top:
        cx = 1 - g["cx"] if flip else g["cx"]
        objs.append(Obj(
            g["cat"], (cx - 0.5) * HFOV_DEG,
            in_path=g["cat"] in OBSTACLES and 0.38 < cx < 0.62 and g["bottom"] > 0.7))
    if fake:
        objs.append(fake)
    return Packet(objects=tuple(objs))


# ── det: the packets the real detector writes for the training images ──
def det(thr: float = 0.45) -> None:
    """Run the reflex detector over the teacher's images and store its boxes in
    the manifest's gt shape, so packet_for() builds the packet the student
    will really get on the device: misses, false positives and all. The
    targets stay ground-truth-verified, so a row whose packet lists something
    absent teaches the student to check the packet against the image."""
    import cv2

    from train.nav.build_oi import CLASSES
    from veris.reflex import onnx_detector

    detect = onnx_detector(str(sorted((ROOT / "data/nav").glob("*.onnx"))[-1]), CLASSES, thr)
    out = {}
    for t in map(json.loads, open(OUT / "teacher.jsonl")):
        img = cv2.imread(str(COCO / "images/train" / t["image"]))
        h, w = img.shape[:2]
        out[t["image"]] = [{"cat": label, "cx": round(float(x1 + x2) / 2 / w, 4),
                            "bottom": round(float(y2) / h, 4),
                            "area": round(float((x2 - x1) * (y2 - y1)) / (w * h), 4)}
                           for x1, y1, x2, y2, conf, label in detect(img)]
    json.dump(out, open(OUT / "det_train.json", "w"))
    print(f"{len(out)} images, {sum(map(len, out.values())) / len(out):.1f} detections each")


# ── select ───────────────────────────────────────────────────────────
def _candidates(split: str) -> dict[str, list[dict]]:
    d = json.load(open(COCO / f"annotations/instances_{split}2017.json"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    by_img: dict[int, list] = {}
    for a in d["annotations"]:
        if not a["iscrowd"]:
            by_img.setdefault(a["image_id"], []).append(a)
    out: dict[str, list[dict]] = {"outdoor": [], "indoor": []}
    for im in d["images"]:
        w, h = im["width"], im["height"]
        if min(w, h) < 400:
            continue
        gt = []
        for a in by_img.get(im["id"], []):
            x, y, bw, bh = a["bbox"]
            area = bw * bh / (w * h)
            if area >= 0.004:
                gt.append({"cat": cats[a["category_id"]], "cx": round((x + bw / 2) / w, 4),
                           "bottom": round((y + bh) / h, 4), "area": round(area, 4)})
        names = {g["cat"] for g in gt}
        if not 2 <= len(gt) <= 14:
            continue
        if names & STREET and not names & INDOOR_ONLY:
            scene = "outdoor"
        elif len(names & INDOOR) >= 2 and not names & OUTDOOR_ONLY:
            scene = "indoor"
        else:
            continue
        out[scene].append({"image": im["file_name"], "split": split, "scene": scene,
                           "url": im["coco_url"], "w": w, "h": h, "gt": gt})
    return out


def _fetch(row: dict) -> bool:
    dst = COCO / "images" / row["split"] / row["image"]
    if dst.exists() and dst.stat().st_size > 1000:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(row["url"], timeout=30) as r:
                dst.write_bytes(r.read())
            return True
        except Exception:
            time.sleep(1 + attempt)
    return False


def select(n_train: int, n_eval: int, seed: int) -> None:
    rng = random.Random(seed)
    rows = []
    for split, n in (("train", n_train), ("val", n_eval)):
        for scene, cand in _candidates(split).items():
            cand.sort(key=lambda r: r["image"])
            rng.shuffle(cand)
            rows += cand[:n]
            print(f"{split}/{scene}: {len(cand)} candidates, took {min(n, len(cand))}")
    with ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(_fetch, rows))
    rows = [r for r, good in zip(rows, ok) if good]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"manifest: {len(rows)} rows, {ok.count(False)} downloads failed")


# ── teach ────────────────────────────────────────────────────────────
def _manifest(split: str) -> list[dict]:
    return [r for r in map(json.loads, open(OUT / "manifest.jsonl")) if r["split"] == split]


def teach(limit: int | None) -> None:
    from veris.model import VLM

    out_path = OUT / "teacher.jsonl"
    done = {json.loads(l)["image"] for l in open(out_path)} if out_path.exists() else set()
    pending = [r for r in _manifest("train") if r["image"] not in done]
    # alternate scenes so a partial run is balanced
    by = {s: [r for r in pending if r["scene"] == s] for s in ("outdoor", "indoor")}
    todo = [r for pair in zip(by["outdoor"], by["indoor"]) for r in pair][:limit]
    print(f"teacher: {len(done)} done, {len(todo)} to do", flush=True)
    vlm = VLM(model_id="Qwen/Qwen2.5-VL-7B-Instruct", quantize=True)
    n_ok = 0
    with open(out_path, "a") as f:
        for i, r in enumerate(todo):
            clock = hash_bit(r["image"])
            sense = packet_for(r["gt"]).to_prompt(r["scene"], clock_face=clock)
            msgs = [
                {"role": "system", "content": TEACHER[r["scene"]]},
                {"role": "user", "content": [
                    {"type": "image", "image": str(COCO / "images/train" / r["image"])},
                    {"type": "text", "text": user_text(EXPERTS[r["scene"]], sense)}]},
            ]
            t0 = time.time()
            target = "".join(vlm.generate_stream(msgs, 110)).strip()
            res = check(target, r["gt"])
            n_ok += res["ok"]
            f.write(json.dumps({"image": r["image"], "scene": r["scene"], "clock": clock,
                                "target": target, "check": res,
                                "wall_s": round(time.time() - t0, 2)}) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"[{i}/{len(todo)}] kept {n_ok}  {r['image']}", flush=True)
    print(f"teacher done: kept {n_ok}/{len(todo)}", flush=True)


def hash_bit(name: str) -> bool:
    """Stable per-image coin: half the set speaks in clock hours, half in sides."""
    return sum(name.encode()) % 2 == 0


# ── sft ──────────────────────────────────────────────────────────────
_SIDE = re.compile(r"\b(left|right)\b", re.I)
_HOUR = re.compile(r"\b(\d{1,2})( o'clock)", re.I)


def mirror_text(t: str) -> str:
    def side(m):
        w = m.group(0)
        r = "right" if w.lower() == "left" else "left"
        return r.capitalize() if w[0].isupper() else r

    def hour(m):
        h = int(m.group(1))
        return f"{12 - h if 1 <= h <= 11 else h}{m.group(2)}"

    return _HOUR.sub(hour, _SIDE.sub(side, t))


def sft(seed: int, replay: int) -> None:
    from PIL import Image

    rng = random.Random(seed)
    gt_of = {r["image"]: r for r in _manifest("train")}
    taught = [json.loads(l) for l in open(OUT / "teacher.jsonl")]
    sft_dir = OUT / "sft"
    (OUT / "flip").mkdir(parents=True, exist_ok=True)
    sft_dir.mkdir(exist_ok=True)
    old = json.load(open(ROOT / "dataset/sft/veris_sft.json"))
    info, stats = {}, {}
    det_path = OUT / "det_train.json"
    det_of = json.load(open(det_path)) if det_path.exists() else {}

    for scene in ("outdoor", "indoor"):
        ex = EXPERTS[scene]
        rows, counts = [], {"det": 0, "full": 0, "dropped": 0, "none": 0, "fake": 0, "mirror": 0}
        cats_seen = sorted({g["cat"] for t in taught for g in gt_of[t["image"]]["gt"]})
        passed = [t for t in taught if t["scene"] == scene and t["check"]["ok"]]

        def row(img_path: str, sense: str, target: str) -> dict:
            return {"messages": [
                {"role": "system", "content": ex.system},
                {"role": "user", "content": "<image>" + user_text(ex, sense)},
                {"role": "assistant", "content": target}], "images": [img_path]}

        for t in passed:
            gt = gt_of[t["image"]]["gt"]
            src = COCO / "images/train" / t["image"]
            # the packet the student sees is not always the one the teacher saw:
            # detectors miss things and occasionally invent them
            u = rng.random()
            kind = "full" if u < 0.6 else "dropped" if u < 0.8 else "none" if u < 0.9 else "fake"
            if det_of and u < 0.45:
                kind = "det"        # the real detector's packet, errors included
            for flip in ((False, True) if rng.random() < 0.5 else (False,)):
                fake = None
                if kind == "fake":
                    absent = [c for c in cats_seen if c not in {g["cat"] for g in gt}]
                    fake = Obj(rng.choice(absent), rng.uniform(-30, 30))
                pk = packet_for(det_of[t["image"]] if kind == "det" else gt, flip=flip,
                                drop=rng.randint(1, 2) if kind == "dropped" else 0,
                                fake=fake, rng=rng)
                # left/ahead/right, never clock hours: a 70 degree camera spans only
                # 11 to 1 o'clock, so the hour cannot carry the side
                # (evals/results/experts_all_boxes.json: side accuracy 0.90 -> 0.96)
                sense = "" if kind == "none" else pk.to_prompt(scene, clock_face=False)
                target, path = t["target"], src
                if flip:
                    path = OUT / "flip" / t["image"]
                    if not path.exists():
                        Image.open(src).convert("RGB").transpose(
                            Image.FLIP_LEFT_RIGHT).save(path, quality=92)
                    target = mirror_text(target)
                    counts["mirror"] += 1
                rows.append(row(str(path), sense, target))
            counts[kind] += 1

        # honesty replay: the v1 VizWiz set carries abstention and question answering
        for r in rng.sample(old, min(replay, len(old))):
            msgs = [{"role": "system", "content": ex.system}] + \
                   [m for m in r["messages"] if m["role"] != "system"]
            rows.append({"messages": msgs,
                         "images": [str(ROOT / "dataset/sft" / r["images"][0])]})
        rng.shuffle(rows)
        json.dump(rows, open(sft_dir / f"{scene}.json", "w"), indent=1)
        info[f"veris_{scene}"] = {
            "file_name": f"{scene}.json", "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                     "assistant_tag": "assistant", "system_tag": "system"}}
        stats[scene] = {"teacher_outputs": sum(t["scene"] == scene for t in taught),
                        "passed_verification": len(passed), **counts,
                        "replay": min(replay, len(old)), "rows": len(rows)}
        (sft_dir / f"qlora_{scene}.yaml").write_text(CONFIG.format(
            sft=sft_dir, scene=scene, out=ROOT / f"train/experts/out-{scene}"))
    # merge: the reader's builder adds its own entry to the same file
    info_path = sft_dir / "dataset_info.json"
    merged = json.load(open(info_path)) if info_path.exists() else {}
    merged.update(info)
    json.dump(merged, open(info_path, "w"), indent=1)
    json.dump(stats, open(OUT / "sft_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


# The v1 recipe (dataset/sft/qlora_3b.yaml) with three deliberate changes:
# - attention layers only. Adapters are applied at run time on the board, and
#   measured there (evals/results/orin_overhead_*.json) an all-layer adapter
#   costs 25% of decode speed while an attention-only one costs 3%.
# - projector frozen, so every expert shares one vision tower and one mmproj.
# - smaller image budget: the packet carries geometry, so the expert is
#   trained to look at less.
CONFIG = """\
model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
image_max_pixels: 313600
trust_remote_code: true
stage: sft
do_train: true
finetuning_type: lora
lora_target: q_proj,k_proj,v_proj,o_proj
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
freeze_vision_tower: true
freeze_multi_modal_projector: true
quantization_bit: 4
double_quantization: true
dataset_dir: {sft}
dataset: veris_{scene}
template: qwen2_vl
cutoff_len: 2048
output_dir: {out}
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 2
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
gradient_checkpointing: true
logging_steps: 5
save_steps: 100
plot_loss: true
report_to: none
overwrite_output_dir: true
"""


def demo() -> None:
    gt = [{"cat": "car", "cx": 0.2, "bottom": 0.8, "area": 0.1},
          {"cat": "person", "cx": 0.5, "bottom": 0.9, "area": 0.05}]
    p = packet_for(gt).to_prompt("outdoor")
    assert "car 11 o'clock" in p and "person 12 o'clock (in path)" in p, p
    assert "car 1 o'clock" in packet_for(gt, flip=True).to_prompt("outdoor")
    assert mirror_text("A car at 10 o'clock on your left. Left turn at 12 o'clock.") == \
        "A car at 2 o'clock on your right. Right turn at 12 o'clock."
    assert len(packet_for(gt, drop=1, rng=random.Random(0)).objects) == 1
    print("build ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "teach", "det", "sft", "demo"])
    ap.add_argument("--n-train", type=int, default=900)
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--replay", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    {"select": lambda: select(a.n_train, a.n_eval, a.seed),
     "teach": lambda: teach(a.limit),
     "det": det,
     "sft": lambda: sft(a.seed, a.replay),
     "demo": demo}[a.cmd]()
