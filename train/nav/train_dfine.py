"""Fine-tune D-FINE-S (ustc-community/dfine-small-obj2coco, Apache-2.0) on the
18-class Open Images navigation set. Class-head rows for classes COCO already
has are copied from the COCO head, so they start trained.

  flock /tmp/veris-gpu.lock .venv/bin/python train/nav/train_dfine.py --minutes 85
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import DFineForObjectDetection

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import OIDet, collate  # noqa: E402
from build_oi import CLASSES  # noqa: E402

BASE = "ustc-community/dfine-small-obj2coco"
COCO = {"person": "person", "bicycle": "bicycle", "car": "car", "motorcycle": "motorbike",
        "bus": "bus", "truck": "truck", "traffic light": "traffic light",
        "fire hydrant": "fire hydrant", "stop sign": "stop sign", "bench": "bench",
        "dog": "dog", "chair": "chair", "table": "diningtable"}


def build_model():
    coco = DFineForObjectDetection.from_pretrained(BASE)
    c2i = {v: int(k) for k, v in coco.config.id2label.items()}
    m = DFineForObjectDetection.from_pretrained(
        BASE, num_labels=len(CLASSES), ignore_mismatched_sizes=True,
        id2label=dict(enumerate(CLASSES)), label2id={c: i for i, c in enumerate(CLASSES)})
    old, new = coco.state_dict(), m.state_dict()
    copied = []
    for k, v in new.items():
        if old[k].shape == v.shape or old[k].shape[0] not in (80, 81):
            continue
        for i, c in enumerate(CLASSES):
            if c in COCO:
                v[i] = old[k][c2i[COCO[c]]]
        if old[k].shape[0] == 81:           # denoising embed: last row is the pad class
            v[-1] = old[k][-1]
        copied.append(k)
    m.load_state_dict(new)
    print("seeded class rows in", copied)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=85)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="train/nav/runs/dfine_s_oi18")
    ap.add_argument("--data", default="train",
                    help="annotation file in data/nav/oi; train_pl = with pseudo_label.py's boxes")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    dev = "cuda"
    m = build_model().to(dev)
    ds = OIDet(a.data, train=True, images="train")
    dl = torch.utils.data.DataLoader(ds, a.bs, shuffle=True, num_workers=6, collate_fn=collate,
                                     drop_last=True, persistent_workers=True)
    bb = [p for n, p in m.named_parameters() if "backbone" in n]
    rest = [p for n, p in m.named_parameters() if "backbone" not in n]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr / 10}, {"params": rest, "lr": a.lr}],
                            weight_decay=1e-4)
    budget = a.minutes * 60
    t0 = time.time()
    step, ep, log, skipped = 0, 0, [], 0
    m.train()
    done = False
    while not done:
        ep += 1
        for x, y in dl:
            frac = (time.time() - t0) / budget
            if frac >= 1:
                done = True
                break
            # 500-step warmup, then cosine over the wall-clock budget
            scale = min(1.0, (step + 1) / 500) * 0.5 * (1 + math.cos(math.pi * frac))
            for g, base in zip(opt.param_groups, (a.lr / 10, a.lr)):
                g["lr"] = base * max(scale, 0.02)
            x = x.to(dev, non_blocking=True)
            y = [{k: v.to(dev) for k, v in t.items()} for t in y]
            # bf16, not fp16: the first fp16 run hit inf in the Hungarian cost
            # matrix ("cost matrix is infeasible") within 50 steps
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = m(pixel_values=x, labels=y).loss
            except ValueError as e:         # non-finite predictions reach the matcher
                loss, err = None, e
            if loss is None or not torch.isfinite(loss):
                skipped += 1
                print(json.dumps({"skip": step, "why": str(err) if loss is None else "nan loss"}),
                      flush=True)
                if skipped > 20:
                    raise SystemExit("too many non-finite steps")
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 0.1)
            opt.step()
            step += 1
            if step % 50 == 0:
                rec = {"step": step, "epoch": ep, "loss": round(loss.item(), 4),
                       "min": round((time.time() - t0) / 60, 1),
                       "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
                log.append(rec)
                print(json.dumps(rec), flush=True)
        m.save_pretrained(out)          # every epoch, so a kill leaves a usable checkpoint
    json.dump({"base": BASE, "train_split": "Open Images V7 train (head-of-CSV sample), "
               f"data/nav/oi/{a.data}.json", "images": len(ds), "steps": step, "epochs": ep,
               "batch": a.bs, "lr": a.lr, "skipped_steps": skipped, "precision": "bf16 autocast", "minutes": round((time.time() - t0) / 60, 1),
               "aug": "hflip, zoom-crop 0.6-1, wearcam p=0.3 (outdoor/indoor still/walk)",
               "log": log}, open(out / "train_log.json", "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
