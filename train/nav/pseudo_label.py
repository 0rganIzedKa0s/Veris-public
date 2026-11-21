"""Fill in the boxes Open Images leaves out, so the fine-tune does not unlearn
people and cars.

Open Images boxes a class only on images where a human verified it. A person
in a photo picked for its stairs is unlabelled, and training on that photo
teaches "person here = background". The first fine-tune lost 24 points of
person mAP50 that way (evals/results/nav_detect.json).

The original COCO checkpoint already knows 13 of the 18 classes. It labels the
training images for those classes; a detection is added when it scores at
least --thr and does not overlap a box that class already has there.

    flock /tmp/veris-gpu.lock .venv/bin/python train/nav/pseudo_label.py
writes data/nav/oi/train_pl.json (same schema as train.json; added boxes
carry "pseudo": score).
"""

import argparse
import json

import cv2
import numpy as np
import torch
from transformers import DFineForObjectDetection

from data import OI, load_coco
from train_dfine import BASE, COCO


def iou(b, B):
    """b: xywh; B: n x 4 xywh."""
    x0, y0 = np.maximum(b[0], B[:, 0]), np.maximum(b[1], B[:, 1])
    x1 = np.minimum(b[0] + b[2], B[:, 0] + B[:, 2])
    y1 = np.minimum(b[1] + b[3], B[:, 1] + B[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / (b[2] * b[3] + B[:, 2] * B[:, 3] - inter + 1e-9)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    d, anns = load_coco("train")
    classes = [c["name"] for c in sorted(d["categories"], key=lambda c: c["id"])]
    m = DFineForObjectDetection.from_pretrained(BASE).cuda().eval()
    c2i = {v: int(k) for k, v in m.config.id2label.items()}
    to_ours = {c2i[COCO[c]]: classes.index(c) for c in COCO}   # COCO index -> our id
    cols = torch.tensor(sorted(to_ours)).cuda()
    added = {c: 0 for c in COCO}
    nid = max(x["id"] for x in d["annotations"]) + 1
    ims = d["images"]
    for s in range(0, len(ims), a.bs):
        batch = ims[s:s + a.bs]
        x = np.stack([cv2.resize(cv2.imread(str(OI / "images/train" / im["file_name"])),
                                 (640, 640))[:, :, ::-1].transpose(2, 0, 1) for im in batch])
        o = m(pixel_values=torch.from_numpy(np.ascontiguousarray(x, np.float32) / 255).cuda())
        p = torch.sigmoid(o.logits.float())[..., cols]           # queries x shared classes
        for k, im in enumerate(batch):
            w, h = im["width"], im["height"]
            for q, j in (p[k] >= a.thr).nonzero().tolist():
                cid = to_ours[int(cols[j])]
                cx, cy, bw, bh = o.pred_boxes[k, q].float().tolist()
                box = [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h]
                same = np.array([g["bbox"] for g in anns.get(im["id"], [])
                                 if g["category_id"] == cid]).reshape(-1, 4)
                if len(same) and iou(box, same).max() >= 0.5:
                    continue                                     # already labelled
                ann = {"id": nid, "image_id": im["id"], "category_id": cid, "bbox": box,
                       "area": box[2] * box[3], "iscrowd": 0, "pseudo": round(p[k, q, j].item(), 3)}
                nid += 1
                d["annotations"].append(ann)
                anns.setdefault(im["id"], []).append(ann)        # dedupes later queries too
                added[classes[cid]] += 1
    json.dump(d, open(OI / "train_pl.json", "w"))
    before = {c: 0 for c in COCO}
    for g in d["annotations"]:
        if "pseudo" not in g and classes[g["category_id"]] in before:
            before[classes[g["category_id"]]] += 1
    print(json.dumps({"thr": a.thr, "images": len(ims), "labelled_before": before, "added": added}))


if __name__ == "__main__":
    assert iou([0, 0, 10, 10], np.array([[0, 0, 10, 10], [20, 20, 5, 5]])).round(3).tolist() == [1.0, 0.0]
    main()
