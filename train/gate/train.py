"""Train the Tier 0 gate and export it to ONNX.

One MobileNetV3-small (ImageNet weights, 1.5 M parameters) with a 7-way
linear output, read as three heads:
    quality  [blur, dark, bright, obstructed, unusable]   VizWiz votes + synthetic
    indoor   [indoor]                                     Places365
    text     [text]                                       VizWiz-Captions
Every sample carries a mask; a head is trained only where its label exists.

Synthetic quality: Places images go through veris.wearcam.degrade with a
random profile (motion, defocus, low light) and a random exposure error
(auto-gain ran out -> dark; clipped highlights -> bright). Blur is labelled
from the smear the profile produces, dark/bright from the result's luminance;
ambiguous middles are masked out instead of guessed.

    flock /tmp/veris-gpu.lock .venv/bin/python train/gate/train.py              # v1
    flock /tmp/veris-gpu.lock .venv/bin/python train/gate/train.py --finetune   # v2

v2 (--finetune) starts from data/gate/gate_v1.pt, adds street_rows() for the
indoor head and runs 4 epochs at a lower learning rate.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from veris.wearcam import Profile, degrade  # noqa: E402

ROOT = REPO / "data" / "gate"
OUTS = ("blur", "dark", "bright", "obstructed", "unusable", "indoor", "text")
SIDE, IN = 256, 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
EPOCHS, PER_EPOCH, BATCH = 14, 16384, 128


def to_tensor(bgr: np.ndarray) -> np.ndarray:
    """Same preprocessing as veris/gate.py: whole frame -> 224x224 RGB, ImageNet norm."""
    x = cv2.resize(bgr, (IN, IN), interpolation=cv2.INTER_AREA)
    x = (x[:, :, ::-1].astype(np.float32) / 255 - MEAN) / STD
    return x.transpose(2, 0, 1).copy()


def luma_labels(img: np.ndarray) -> tuple[float | None, float | None]:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m, clip = g.mean(), (g >= 250).mean()
    dark = 1.0 if m < 40 else 0.0 if m > 70 else None
    bright = 1.0 if clip > 0.35 else 0.0 if clip < 0.05 else None
    return dark, bright


def synth(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float | None]:
    """Wearable degradation with a known blur label (None = ambiguous)."""
    exp = float(rng.choice([1 / 500, 1 / 120, 1 / 30, 1 / 15]))
    head = float(rng.uniform(0, 90))
    defocus = float(rng.choice([0, 0, 0, 0.8, 1.6, 3.0]))     # px at 256
    p = Profile(long_side=SIDE, exposure_s=exp, head_deg_s=head,
                light=float(rng.choice([1.0, 0.5, 0.15])), roll_deg=8,
                defocus_sigma=defocus, barrel_k1=float(rng.uniform(-0.2, 0)),
                jpeg_q=int(rng.integers(40, 90)))
    out = degrade(img, p, seed=int(rng.integers(1 << 31))).astype(np.float32)
    r = rng.random()
    if r < 0.25:                                  # under-exposed: gain ran out
        out *= rng.uniform(0.04, 0.5)
    elif r < 0.4:                                 # over-exposed: highlights clip
        out = out * rng.uniform(1.8, 4.0) + rng.uniform(0, 60)
    out = np.clip(out, 0, 255).astype(np.uint8)
    smear = head * exp * SIDE / 80                # px at 256 (5 px = 25 px at 1280)
    blur = 1.0 if smear >= 5 or defocus >= 1.6 else 0.0 if smear <= 1.5 and defocus == 0 else None
    return out, blur


class GateSet(Dataset):
    def __init__(self, rows: list[dict], train: bool):
        self.rows, self.train = rows, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        rng = np.random.default_rng() if self.train else np.random.default_rng(i)
        img = cv2.imread(str(ROOT / r["path"]), cv2.IMREAD_COLOR)
        if img.shape[:2] != (SIDE, SIDE):        # street rows are stored full size
            img = cv2.resize(img, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
        y, m = np.zeros(7, np.float32), np.zeros(7, np.float32)

        def put(k, v):
            if v is not None:
                y[OUTS.index(k)], m[OUTS.index(k)] = v, 1

        if r["src"] in ("coco", "oi"):           # indoor label only; quality/text masked
            put("indoor", r["indoor"])
            if self.train and rng.random() < 0.6:
                img, _ = synth(img, rng)
        elif r["src"] == "places":
            put("indoor", r["indoor"])
            if self.train and rng.random() < 0.6:
                img, blur = synth(img, rng)
                put("blur", blur)
            dark, bright = luma_labels(img)
            put("dark", dark)
            put("bright", bright)
        else:
            for k in OUTS:
                if k != "indoor":
                    put(k, r[k])
        if self.train:
            s = rng.uniform(0.75, 1.0)
            h = w = int(SIDE * s)
            y0, x0 = rng.integers(0, SIDE - h + 1), rng.integers(0, SIDE - w + 1)
            img = img[y0:y0 + h, x0:x0 + w]
            if rng.random() < 0.5:
                img = img[:, ::-1]
        return to_tensor(np.ascontiguousarray(img)), y, m


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from scipy.stats import rankdata
    pos = y == 1
    if pos.all() or not pos.any():
        return float("nan")
    r = rankdata(s)
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


@torch.no_grad()
def evaluate(model, loader, dev) -> dict:
    model.eval()
    P, Y, M = [], [], []
    for x, y, m in loader:
        P.append(torch.sigmoid(model(x.to(dev)).float()).cpu())
        Y.append(y)
        M.append(m)
    P, Y, M = (torch.cat(t).numpy() for t in (P, Y, M))
    return {k: auroc(Y[M[:, j] > 0, j], P[M[:, j] > 0, j]) for j, k in enumerate(OUTS)}


OI_STREET = {"car", "bus", "truck", "traffic light", "traffic sign", "stop sign",
             "street light", "fire hydrant"}
OI_ROOM = {"table", "chair"}


def street_rows() -> list[dict]:
    """v2: street- and room-level photos for the indoor head (Places outdoor is
    mostly landscapes; a head camera sees pavements, cars and shop fronts).
    - COCO train2017 rows of data/experts/manifest.jsonl, scene from the COCO
      category rule in train/experts/build.py (COCO val rows stay held out)
    - Open Images nav train images, outdoor only: at least one street class
      (car, bus, truck, traffic light/sign, stop sign, street light, fire
      hydrant) and no table or chair. No OI class implies indoors reliably, so
      OI contributes no indoor rows.
    Split: md5(name) % 10 == 0 -> val, else train."""
    def split(name):
        return "val" if int(hashlib.md5(name.encode()).hexdigest(), 16) % 10 == 0 else "train"

    rows = [{"path": str(REPO / "data/coco/images/train" / r["image"]), "src": "coco",
             "indoor": int(r["scene"] == "indoor"), "split": split(r["image"])}
            for r in map(json.loads, open(REPO / "data/experts/manifest.jsonl"))
            if r["split"] == "train"]
    oi = json.load(open(REPO / "data/nav/oi/train.json"))
    names = {c["id"]: c["name"] for c in oi["categories"]}
    cats: dict[int, set] = {}
    for a in oi["annotations"]:
        cats.setdefault(a["image_id"], set()).add(names[a["category_id"]])
    for im in oi["images"]:
        c = cats.get(im["id"], set())
        if c & OI_STREET and not c & OI_ROOM:
            rows.append({"path": str(REPO / "data/nav/oi/images/train" / im["file_name"]),
                         "src": "oi", "indoor": 0, "split": split(im["file_name"])})
    return rows


def main() -> None:
    torch.manual_seed(0)
    finetune = "--finetune" in sys.argv
    rows = [json.loads(l) for l in open(ROOT / "manifest.jsonl")]
    if finetune:
        rows += street_rows()
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    assert not any(r["split"] == "test" for r in tr)
    # share of every batch per source
    share = ({"vizwiz": 0.4, "places": 0.35, "coco": 0.15, "oi": 0.10} if finetune
             else {"vizwiz": 0.5, "places": 0.5})
    n = {k: sum(r["src"] == k for r in tr) for k in share}
    print("train rows", n, "val rows", len(va), flush=True)
    w = [share[r["src"]] / n[r["src"]] for r in tr]
    dl_tr = DataLoader(GateSet(tr, True), BATCH, num_workers=10, drop_last=True,
                       sampler=WeightedRandomSampler(w, PER_EPOCH), persistent_workers=True)
    dl_va = DataLoader(GateSet(va, False), 256, num_workers=10)

    dev = "cuda"
    model = timm.create_model("mobilenetv3_small_100.lamb_in1k", pretrained=not finetune,
                              num_classes=len(OUTS)).to(dev)
    epochs, lr = (EPOCHS, 2e-3) if not finetune else (4, 5e-4)
    if finetune:                                  # v2 continues from the v1 weights
        model.load_state_dict(torch.load(ROOT / "gate_v1.pt"))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    steps = epochs * (PER_EPOCH // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    best, log, t0 = -1.0, [], time.time()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, y, m in dl_tr:
            x, y, m = x.to(dev), y.to(dev), m.to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16):
                logit = model(x)
            loss = (F.binary_cross_entropy_with_logits(logit.float(), y, reduction="none") * m
                    ).sum(0) / m.sum(0).clamp(min=1)
            loss = loss.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        auc = evaluate(model, dl_va, dev)
        score = float(np.nanmean(list(auc.values())))
        log.append({"epoch": ep, "loss": tot / (PER_EPOCH // BATCH), "val_auroc": auc,
                    "mean": score, "min": round((time.time() - t0) / 60, 1)})
        print(json.dumps(log[-1]), flush=True)
        if score > best:
            best = score
            torch.save(model.state_dict(), ROOT / "gate.pt")
    (ROOT / ("train_log_v2.json" if finetune else "train_log.json")).write_text(json.dumps(log, indent=1))

    # ONNX export of the best epoch, fixed 1x3x224x224, opset 17
    model.load_state_dict(torch.load(ROOT / "gate.pt"))
    model = model.cpu().eval()
    torch.onnx.export(model, torch.zeros(1, 3, IN, IN), ROOT / "gate.onnx", opset_version=17,
                      input_names=["image"], output_names=["logits"], dynamo=False)
    print("exported", ROOT / "gate.onnx", f"best val mean AUROC {best:.4f}")


if __name__ == "__main__":
    main()
