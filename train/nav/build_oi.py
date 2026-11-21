"""Build the Tier 1 detector set from Open Images V7 boxes (CC BY 4.0).

Splits (no image appears in both):
  train : OI `train`, from the first 800 MB of oidv6-train-annotations-bbox.csv
          (the file is sorted by random hex ImageID, so its head is a random
          ~35% sample of train images)
  heldout: OI `validation` + OI `test`, never trained on. Both are pooled
          because the rare classes have 6-34 images in either one alone.

Selection is greedy rarest-class-first: every class gets up to `quota` images
before common classes fill in. Images are resized to 640 px long side on
download. Writes COCO json; held-out images also carry `verified`, the class
ids a human checked (present or absent) on that image, because Open Images
boxes are not exhaustive and unverified classes must not count as false
positives.

  .venv/bin/python -I train/nav/build_oi.py
"""

import csv
import json
import sys
import threading
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "data/nav/oi"
CLASSES = ["person", "car", "bus", "truck", "bicycle", "motorcycle", "traffic light",
           "traffic sign", "stop sign", "street light", "fire hydrant", "waste container",
           "bench", "chair", "table", "door", "stairs", "dog"]
MID = {  # Open Images label -> our class. Person's gendered/age children are separate OI labels.
    "/m/01g317": "person", "/m/04yx4": "person", "/m/03bt1vf": "person", "/m/01bl7v": "person",
    "/m/05r655": "person", "/m/0k4j": "car", "/m/0pg52": "car", "/m/0h2r6": "car",
    "/m/01bjv": "bus", "/m/07r04": "truck", "/m/0199g": "bicycle", "/m/04_sv": "motorcycle",
    "/m/015qff": "traffic light", "/m/01mqdt": "traffic sign", "/m/02pv19": "stop sign",
    "/m/033rq4": "street light", "/m/01pns0": "fire hydrant", "/m/0bjyj5": "waste container",
    "/m/0cvnqh": "bench", "/m/01mzpv": "chair", "/m/04bcr3": "table", "/m/0h8n5zk": "table",
    "/m/078n6m": "table", "/m/01y9k5": "table", "/m/02dgv": "door", "/m/01lynh": "stairs",
    "/m/0bt9lr": "dog"}
CID = {c: i for i, c in enumerate(CLASSES)}
BUDGET = 3.9e9          # image bytes; CSVs already spent ~0.95 GB of the 5 GB
LONG = 640


def read_boxes(path, split, drop_last=False):
    boxes = defaultdict(list)
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) < 13 or row[2] not in MID:
                continue
            x1, x2, y1, y2 = map(float, row[4:8])
            boxes[(split, row[0])].append((CID[MID[row[2]]], x1, y1, x2, y2, int(row[10])))
    if drop_last:   # a byte-range cut may have split the last image's rows
        boxes.pop(max(boxes, key=lambda k: k[1]), None)
    return boxes


def select(boxes, quota, cap):
    per = defaultdict(list)
    for k, bs in boxes.items():
        for c in {b[0] for b in bs}:
            per[c].append(k)
    have, chosen = defaultdict(int), []
    seen = set()
    for c in sorted(per, key=lambda c: len(per[c])):
        for k in sorted(per[c]):      # sorted ids = a fixed random order
            if have[c] >= quota or len(chosen) >= cap:
                break
            if k in seen:
                continue
            seen.add(k)
            chosen.append(k)
            for cc in {b[0] for b in boxes[k]}:
                have[cc] += 1
    return chosen, dict(have)


spent = 0
lock = threading.Lock()


def fetch(key, out_dir):
    global spent
    split, iid = key
    dst = out_dir / f"{iid}.jpg"
    if dst.exists():
        img = cv2.imread(str(dst))
        return key, img.shape[1], img.shape[0]
    with lock:
        if spent > BUDGET:
            return key, None, None
    try:
        data = urllib.request.urlopen(
            f"https://s3.amazonaws.com/open-images-dataset/{split}/{iid}.jpg", timeout=30).read()
    except Exception as e:
        print("fail", iid, e, file=sys.stderr)
        return key, None, None
    with lock:
        spent += len(data)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return key, None, None
    s = LONG / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return key, img.shape[1], img.shape[0]


def verified_labels(paths):
    v = defaultdict(set)
    for p in paths:
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                if row["LabelName"] in MID:
                    v[row["ImageID"]].add(CID[MID[row["LabelName"]]])
    return v


def write(name, keys, boxes, verified=None):
    out_dir = ROOT / "images" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(24) as ex:
        res = list(ex.map(lambda k: fetch(k, out_dir), keys))
    images, anns = [], []
    for key, w, h in res:
        if w is None:
            continue
        iid = len(images)
        im = {"id": iid, "file_name": f"{key[1]}.jpg", "width": w, "height": h,
              "oi_split": key[0]}
        if verified is not None:
            im["verified"] = sorted(verified[key[1]] | {b[0] for b in boxes[key]})
        images.append(im)
        for c, x1, y1, x2, y2, grp in boxes[key]:
            bw, bh = (x2 - x1) * w, (y2 - y1) * h
            anns.append({"id": len(anns), "image_id": iid, "category_id": c,
                         "bbox": [x1 * w, y1 * h, bw, bh], "area": bw * bh,
                         "iscrowd": grp})
    cats = [{"id": i, "name": c} for i, c in enumerate(CLASSES)]
    json.dump({"images": images, "annotations": anns, "categories": cats},
              open(ROOT / f"{name}.json", "w"))
    per = defaultdict(lambda: [0, set()])
    for a in anns:
        per[a["category_id"]][0] += 1
        per[a["category_id"]][1].add(a["image_id"])
    print(name, len(images), "images", len(anns), "boxes")
    for i, c in enumerate(CLASSES):
        print(f"  {c:16s} boxes {per[i][0]:6d} images {len(per[i][1]):5d}")


if __name__ == "__main__":
    held = read_boxes(ROOT / "validation-annotations-bbox.csv", "validation")
    held.update(read_boxes(ROOT / "test-annotations-bbox.csv", "test"))
    train = read_boxes(ROOT / "train-annotations-bbox.head.csv", "train", drop_last=True)
    hk, _ = select(held, quota=150, cap=1500)
    tk, _ = select(train, quota=700, cap=9000)
    ver = verified_labels([ROOT / "validation-annotations-human-imagelabels-boxable.csv",
                           ROOT / "test-annotations-human-imagelabels-boxable.csv"])
    write("heldout", hk, held, ver)
    write("train", tk, train)
    print(f"image bytes downloaded this run: {spent / 1e9:.2f} GB")
