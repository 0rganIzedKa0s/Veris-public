"""Tier 1 detector: mAP on the Open Images held-out split, clean and wearcam.

Held-out = OI `validation` + OI `test` images (data/nav/oi/heldout.json);
training used only OI `train`. Open Images boxes are not exhaustive, so each
class is scored only on images where a human verified that class (present or
absent), as the OI protocol does; group-of boxes are COCO `iscrowd` (ignored).

Wearcam: veris.wearcam.degrade(profile, seed=image id); GT boxes are moved by
the same barrel distortion and head roll (train/nav/data.warp_boxes). The
fine-tune saw wearcam degradation as augmentation (p=0.3), so its wearcam
numbers are in-distribution for the degradation model; the COCO checkpoint's
are not.

COCO baseline: the pretrained 80-class checkpoint the fine-tune started from,
scored on the 13 classes it shares with ours (dining table -> table).

  flock /tmp/veris-gpu.lock .venv/bin/python evals/nav_detect.py [--device cpu]
"""

import contextlib
import copy
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO as CocoApi
from pycocotools.cocoeval import COCOeval
from transformers import DFineForObjectDetection

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "train/nav"))
from data import OI, load_coco, warp_boxes  # noqa: E402
from train_dfine import BASE, COCO  # noqa: E402
from veris.wearcam import degrade  # noqa: E402

def _arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


CKPT = REPO / _arg("--ckpt", "train/nav/runs/dfine_s_oi18")
OUT = REPO / "evals/results" / _arg("--out", "nav_detect.json")
PROFILES = ("clean", "outdoor_walk", "indoor_walk")
DEV = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda"


@torch.no_grad()
def predict(model, d, profile, label_map, bs=16):
    """label_map: model class index -> our category id (unmapped classes dropped)."""
    dets, imgs, metas = [], [], []

    def flush():
        o = model(pixel_values=torch.from_numpy(np.stack(imgs)).to(DEV))   # fp32
        p = torch.sigmoid(o.logits.float())
        keep = torch.zeros(p.shape[-1], dtype=torch.bool, device=p.device)
        keep[list(label_map)] = True
        p[..., ~keep] = 0
        for k, (iid, w, h) in enumerate(metas):
            s, idx = p[k].flatten().topk(100)
            q, c = idx // p.shape[-1], idx % p.shape[-1]
            b = o.pred_boxes[k, q].float().cpu().numpy()
            for sc, ci, (cx, cy, bw, bh) in zip(s.tolist(), c.tolist(), b):
                dets.append({"image_id": iid, "category_id": label_map[ci], "score": sc,
                             "bbox": [float((cx - bw / 2) * w), float((cy - bh / 2) * h),
                                      float(bw * w), float(bh * h)]})
        imgs.clear(); metas.clear()

    for im in d["images"]:
        img = cv2.imread(str(OI / "images/heldout" / im["file_name"]))
        if profile != "clean":
            img = degrade(img, profile, seed=im["id"])
        x = cv2.resize(img, (640, 640))[:, :, ::-1].transpose(2, 0, 1)
        imgs.append(np.ascontiguousarray(x, np.float32) / 255)
        metas.append((im["id"], img.shape[1], img.shape[0]))
        if len(imgs) == bs:
            flush()
    if imgs:
        flush()
    return dets


def gt_for(d, profile):
    if profile == "clean":
        return d
    g = copy.deepcopy(d)
    by_img = {im["id"]: im for im in g["images"]}
    per = {}
    for a in g["annotations"]:
        per.setdefault(a["image_id"], []).append(a)
    for iid, A in per.items():
        im = by_img[iid]
        b = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                       a["bbox"][1] + a["bbox"][3]] for a in A])
        for a, (x1, y1, x2, y2) in zip(A, warp_boxes(b, im["width"], im["height"], profile, iid)):
            a["bbox"] = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            a["area"] = float((x2 - x1) * (y2 - y1))
    return g


def score(gt_dict, dets, cats):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = CocoApi()
        gt.dataset = gt_dict
        gt.createIndex()
        dt = gt.loadRes(dets) if dets else None
    names = {c["id"]: c["name"] for c in gt_dict["categories"]}
    per = {}
    for c in cats:
        imgs = [im["id"] for im in gt_dict["images"] if c in im["verified"]]
        n_gt = sum(1 for a in gt_dict["annotations"] if a["category_id"] == c
                   and not a["iscrowd"] and a["image_id"] in set(imgs))
        if n_gt == 0 or dt is None:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            E = COCOeval(gt, dt, "bbox")
            E.params.catIds, E.params.imgIds = [c], imgs
            E.evaluate(); E.accumulate(); E.summarize()
        per[names[c]] = {"mAP50": round(100 * E.stats[1], 1), "mAP50_95": round(100 * E.stats[0], 1),
                         "n_images": len(imgs), "n_boxes": n_gt}
    return {"mAP50": round(float(np.mean([v["mAP50"] for v in per.values()])), 1),
            "mAP50_95": round(float(np.mean([v["mAP50_95"] for v in per.values()])), 1),
            "n_classes": len(per), "per_class": per}


def main():
    d, _ = load_coco("heldout")
    classes = [c["name"] for c in sorted(d["categories"], key=lambda c: c["id"])]
    ft = DFineForObjectDetection.from_pretrained(CKPT).to(DEV).eval()
    base = DFineForObjectDetection.from_pretrained(BASE).to(DEV).eval()
    c2i = {v: int(k) for k, v in base.config.id2label.items()}
    shared = [classes.index(c) for c in COCO]
    base_map = {c2i[COCO[classes[i]]]: i for i in shared}
    res = {"command": "flock /tmp/veris-gpu.lock .venv/bin/python evals/nav_detect.py", "device": DEV,
           "split": "held-out: Open Images V7 validation + test images (never trained on); "
                    "training used Open Images V7 train only",
           "n_images": len(d["images"]), "n_boxes": len(d["annotations"]),
           "protocol": "pycocotools bbox AP per class on images where the class is human-"
                       "verified; macro mean over classes; IsGroupOf -> iscrowd; maxDets 100",
           "finetuned": {"ckpt": str(CKPT.relative_to(REPO))},
           "coco_pretrained_shared_classes": {"ckpt": BASE, "classes": [classes[i] for i in shared]},
           "finetuned_on_shared_classes": {}}
    for prof in PROFILES:
        g = gt_for(d, prof)
        dft = predict(ft, d, prof, {i: i for i in range(len(classes))})
        res["finetuned"][prof] = score(g, dft, range(len(classes)))
        sh = score(g, [x for x in dft if x["category_id"] in shared], shared)
        res["finetuned_on_shared_classes"][prof] = {k: sh[k] for k in ("mAP50", "mAP50_95")}
        res["coco_pretrained_shared_classes"][prof] = score(g, predict(base, d, prof, base_map), shared)
        print(prof, "ft", res["finetuned"][prof]["mAP50"], res["finetuned"][prof]["mAP50_95"],
              "| shared ft", sh["mAP50"], "coco", res["coco_pretrained_shared_classes"][prof]["mAP50"],
              flush=True)
        # written after every profile so a stopped run keeps what it finished
        res["profiles_done"] = PROFILES[:PROFILES.index(prof) + 1]
        OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
