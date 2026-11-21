"""Build the Tier 0 gate dataset under data/gate/.

Sources (all fetched by this script, total < 6 GB):
- VizWiz image-quality annotations (flaw votes out of 5, `unrecognizable`)
- VizWiz-Captions annotations (`text_detected`)
- VizWiz images: reused from dataset/frames(_v2) where present, else fetched
  one by one from vizwiz.cs.colorado.edu/VizWiz_visualization_img/
- Places365 val_256 + its file list + IO_places365.txt (1 indoor, 2 outdoor)

Every image is stored as a 256x256 JPEG (whole frame squashed, never cropped:
framing is one of the things the gate judges).

Labels
- flaw (blur/dark/bright/obstructed): >= 2 of 5 annotators
- unusable: `unrecognizable` >= 3 of 5 (a majority could not tell what it is)
- text: VizWiz-Captions `text_detected`
- indoor: Places365 category -> IO_places365.txt

Splits (written into manifest.jsonl; nothing in `test` is ever trained on)
- VizWiz train  -> train (90%) / val (10%), by hash of the file name
- VizWiz val    -> test (official labels for VizWiz test are private)
- Places365 val -> per category, sorted by file name: 70 train / 10 val /
  20 test (Places365 test labels are private)

    .venv/bin/python -I train/gate/build_data.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import tarfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data" / "gate"
ANN, IMG = ROOT / "ann", ROOT / "img"
BUDGET = 6e9
SIDE = 256
N_VIZ_TRAIN = 6000      # fetched in addition to the 1,200 local train frames
N_VIZ_RARE = 2500       # of those, drawn from images with a rare flaw
N_VIZ_TEST = 1700       # fetched in addition to the 300 local val frames
LOCAL = [REPO / "dataset" / "frames", REPO / "dataset" / "frames_v2"]
URLS = {
    "quality.zip": "https://vizwiz.cs.colorado.edu/VizWiz_final/image_quality/annotations.zip",
    "captions.zip": "https://vizwiz.cs.colorado.edu/VizWiz_final/caption/annotations.zip",
    "IO_places365.txt": "https://raw.githubusercontent.com/CSAILVision/places365/master/IO_places365.txt",
    "filelist.tar": "http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar",
    "val_256.tar": "http://data.csail.mit.edu/places/places365/val_256.tar",
}
VIZ_IMG = "https://vizwiz.cs.colorado.edu/VizWiz_visualization_img/"
FLAWS = {"blur": "BLR", "dark": "DRK", "bright": "BRT", "obstructed": "OBS"}
spent = 0


def fetch(url: str, dest: Path) -> bytes | None:
    """Download once; count bytes against the budget."""
    global spent
    if dest.exists():
        return None
    data = urllib.request.urlopen(url, timeout=120).read()
    spent += len(data)
    if spent > BUDGET:
        sys.exit(f"download budget exceeded at {spent / 1e9:.2f} GB")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.rename(dest)
    return data


def squash(buf: bytes) -> bytes:
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("undecodable image")
    img = cv2.resize(img, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def vizwiz_image(name: str, url: str) -> Path | None:
    out = IMG / "vizwiz" / name
    if out.exists():
        return out
    try:
        local = next((d / name for d in LOCAL if (d / name).exists()), None)
        raw = local.read_bytes() if local else fetch(url, ROOT / "tmp" / name)
        if raw is None:                     # interrupted earlier run left tmp
            raw = (ROOT / "tmp" / name).read_bytes()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(squash(raw))
        (ROOT / "tmp" / name).unlink(missing_ok=True)
        return out
    except Exception as e:                  # one bad image must not stop the build
        print("skip", name, e, flush=True)
        return None


def main() -> None:
    ANN.mkdir(parents=True, exist_ok=True)
    for f in ("quality.zip", "captions.zip", "IO_places365.txt", "filelist.tar"):
        fetch(URLS[f], ANN / f)
    qual = {s: json.loads(zipfile.ZipFile(ANN / "quality.zip").read(f"{s}.json"))
            for s in ("train", "val")}
    caps = {}
    for s in ("train", "val"):
        c = json.loads(zipfile.ZipFile(ANN / "captions.zip").read(f"annotations/{s}.json"))
        caps.update({i["file_name"]: i for i in c["images"]})

    rng = random.Random(0)
    local = {p.name for d in LOCAL for p in d.glob("*.jpg")}

    def rare(r):
        return (any(r["flaws"][k] >= 2 for k in ("DRK", "BRT", "OBS"))
                or r["unrecognizable"] >= 3)

    tr = [r for r in qual["train"] if r["image"] not in local]
    rng.shuffle(tr)
    rare_pick = [r for r in tr if rare(r)][:N_VIZ_RARE]
    picked = {r["image"] for r in rare_pick}
    rand_pick = [r for r in tr if r["image"] not in picked][:N_VIZ_TRAIN - N_VIZ_RARE]
    va = [r for r in qual["val"] if r["image"] not in local]
    rng.shuffle(va)
    want = ([r for r in qual["train"] if r["image"] in local] + rare_pick + rand_pick
            + [r for r in qual["val"] if r["image"] in local] + va[:N_VIZ_TEST])

    with ThreadPoolExecutor(24) as ex:
        # the captions file names ivc.ischool.utexas.edu, which no longer resolves
        paths = list(ex.map(lambda r: vizwiz_image(r["image"], VIZ_IMG + r["image"]), want))
    rows = []
    for r, p in zip(want, paths):
        if p is None:
            continue
        name = r["image"]
        if name.startswith("VizWiz_val"):
            split = "test"
        else:
            split = "val" if int(hashlib.md5(name.encode()).hexdigest(), 16) % 10 == 0 else "train"
        row = {"path": str(p.relative_to(ROOT)), "src": "vizwiz", "split": split,
               "unusable": int(r["unrecognizable"] >= 3),
               "text": int(caps[name]["text_detected"]), "indoor": None,
               "rare_sampled": name in picked}
        row.update({k: int(r["flaws"][v] >= 2) for k, v in FLAWS.items()})
        rows.append(row)
    print(f"vizwiz: {len(rows)} images, {spent / 1e9:.2f} GB fetched", flush=True)

    # Places365 val
    pdir = IMG / "places"
    if not pdir.exists():
        fetch(URLS["val_256.tar"], ANN / "val_256.tar")
        with tarfile.open(ANN / "val_256.tar") as t:
            t.extractall(IMG / "places_tmp", filter="data")
        (IMG / "places_tmp" / "val_256").rename(pdir)
        (IMG / "places_tmp").rmdir()
    io = [int(l.split()[-1]) for l in (ANN / "IO_places365.txt").read_text().splitlines() if l.strip()]
    with tarfile.open(ANN / "filelist.tar") as t:
        member = next(m for m in t.getmembers() if m.name.endswith("places365_val.txt"))
        lines = t.extractfile(member).read().decode().split()
    by_cls: dict[int, list[str]] = {}
    for name, cls in zip(lines[::2], lines[1::2]):
        by_cls.setdefault(int(cls), []).append(name)
    for cls, names in by_cls.items():
        for i, name in enumerate(sorted(names)):
            split = "train" if i < 70 else "val" if i < 80 else "test"
            rows.append({"path": f"img/places/{name}", "src": "places", "split": split,
                         "indoor": int(io[cls] == 1), "cls": cls, "text": None,
                         "unusable": None, **{k: None for k in FLAWS}})
    with open(ROOT / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (ANN / "val_256.tar").unlink(missing_ok=True)    # extracted; keep disk tidy
    print(f"manifest: {len(rows)} rows, {spent / 1e9:.2f} GB fetched this run")


if __name__ == "__main__":
    main()
