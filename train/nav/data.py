"""COCO-json detection set for D-FINE, plus wearcam degradation that keeps the
boxes aligned (barrel distortion and head roll move objects; the boxes must
move with them)."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from veris.wearcam import PROFILES, degrade  # noqa: E402

OI = REPO / "data/nav/oi"
SIZE = 640


def warp_boxes(boxes, w, h, profile, seed):
    """Move xyxy pixel boxes the way wearcam.degrade(profile, seed) moves pixels.
    Mirrors degrade(): barrel first, then roll, roll drawn first from the seed's rng."""
    p = PROFILES[profile]
    if len(boxes) == 0:
        return boxes
    t = np.linspace(0, 1, 9)
    pts = []
    for x1, y1, x2, y2 in boxes:
        xs, ys = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        pts.append(np.r_[np.c_[xs, np.full(9, y1)], np.c_[xs, np.full(9, y2)],
                         np.c_[np.full(9, x1), ys], np.c_[np.full(9, x2), ys]])
    pts = np.concatenate(pts).astype(np.float64)
    if p.barrel_k1:
        f = 0.5 * w / np.tan(np.radians(p.hfov_deg / 2))
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
        # degrade() samples src = distort(dst, -k1); invert that for src -> dst
        pts = cv2.undistortPoints(pts[:, None], K, np.array([-p.barrel_k1, 0, 0, 0.0]),
                                  P=K)[:, 0]
    if p.roll_deg:
        roll = np.random.default_rng(seed).uniform(-p.roll_deg, p.roll_deg)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), roll, 1.0)
        pts = pts @ M[:, :2].T + M[:, 2]
    pts = pts.reshape(len(boxes), 36, 2)
    out = np.c_[pts[..., 0].min(1), pts[..., 1].min(1), pts[..., 0].max(1), pts[..., 1].max(1)]
    return np.clip(out, 0, [w, h, w, h])


def load_coco(name):
    d = json.load(open(OI / f"{name}.json"))
    anns = {}
    for a in d["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    return d, anns


class OIDet(torch.utils.data.Dataset):
    def __init__(self, name="train", train=True, wear_p=0.3, images=None):
        self.d, self.anns = load_coco(name)
        self.dir = OI / "images" / (images or name)
        self.train, self.wear_p = train, wear_p

    def __len__(self):
        return len(self.d["images"])

    def __getitem__(self, i):
        im = self.d["images"][i]
        img = cv2.imread(str(self.dir / im["file_name"]))
        h, w = img.shape[:2]
        A = self.anns.get(im["id"], [])
        b = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                       a["bbox"][1] + a["bbox"][3]] for a in A], np.float64).reshape(-1, 4)
        c = np.array([a["category_id"] for a in A], np.int64)
        if self.train:
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                img = img[:, ::-1].copy()
                b[:, [0, 2]] = w - b[:, [2, 0]]
            if rng.random() < 0.5:      # zoom crop: small far objects are the hard ones
                s = rng.uniform(0.6, 1.0)
                cw, ch = int(w * s), int(h * s)
                x0, y0 = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
                nb = np.clip(b - [x0, y0, x0, y0], 0, [cw, ch, cw, ch])
                area = lambda q: (q[:, 2] - q[:, 0]) * (q[:, 3] - q[:, 1])
                keep = area(nb) >= 0.4 * np.maximum(area(b), 1e-6)
                img, b, c, w, h = img[y0:y0 + ch, x0:x0 + cw], nb[keep], c[keep], cw, ch
            if rng.random() < self.wear_p:
                prof = rng.choice(["outdoor_still", "outdoor_walk", "indoor_still", "indoor_walk"])
                seed = int(rng.integers(1 << 30))
                img = degrade(np.ascontiguousarray(img), prof, seed)
                b = warp_boxes(b, w, h, prof, seed)
        keep = ((b[:, 2] - b[:, 0]) > 1) & ((b[:, 3] - b[:, 1]) > 1)
        b, c = b[keep], c[keep]
        x = cv2.resize(img, (SIZE, SIZE))[:, :, ::-1].transpose(2, 0, 1)
        x = torch.from_numpy(np.ascontiguousarray(x)).float() / 255
        cxcywh = np.c_[(b[:, 0] + b[:, 2]) / 2 / w, (b[:, 1] + b[:, 3]) / 2 / h,
                       (b[:, 2] - b[:, 0]) / w, (b[:, 3] - b[:, 1]) / h]
        return x, {"class_labels": torch.from_numpy(c),
                   "boxes": torch.from_numpy(cxcywh).float()}


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def demo():
    # a box warped with a zero-distortion profile must not move
    import dataclasses
    PROFILES["_id"] = dataclasses.replace(PROFILES["outdoor_walk"], barrel_k1=0, roll_deg=0)
    bx = np.array([[100.0, 50, 200, 300]])
    assert np.allclose(warp_boxes(bx, 640, 480, "_id", 1), bx)
    # check alignment empirically: paint the box, degrade, measure the painted blob
    img = np.zeros((480, 640, 3), np.uint8)
    cv2.rectangle(img, (400, 100), (500, 300), (255, 255, 255), -1)
    deg = degrade(img, "indoor_walk", 7)
    ys, xs = np.nonzero(deg[..., 0] > 128)
    pred = warp_boxes(np.array([[400.0, 100, 501, 301]]), 640, 480, "indoor_walk", 7)[0]
    meas = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
    assert np.abs(pred - meas).max() < 12, (pred, meas)       # blur smears ~8 px
    print("warp ok", pred.round(1), meas)


if __name__ == "__main__":
    demo()
