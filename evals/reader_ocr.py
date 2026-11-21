"""End-to-end word recognition of veris/reader.py on open scene-text sets.

    .venv/bin/python evals/reader_ocr.py            # -> evals/results/reader_ocr.json

Sets (test splits only; the confidence threshold is fitted on ICDAR15 *train*):
  icdar15    ICDAR 2015 Incidental Scene Text, ch4 test, 500 images (Google
             Glass, wearer not aiming). Mirror: hf datasets dlxjj/ICDAR2015.
  totaltext  Total-Text test, 300 images (aimed photos, curved text): the
             cleaner contrast set. Mirror: hf yunusserhat/Total-Text-Dataset.

Matching. The reader returns words with quads interpolated from CTC frame
positions, so matching is by location, not strict IoU: a predicted word and a
GT word are paired when the predicted word's centre lies inside the GT polygon
grown by half its glyph height; pairs with equal text are taken first, then
nearest; one-to-one. A word is correct when the pair's text is equal after
lower-casing and stripping leading/trailing punctuation. Predictions centred in
a don't-care region ("###" / "#") are ignored. GT entries holding several
space-separated words are split along their length by character count.
CER is edit distance / GT length over all location-matched GT words.
"Invented" = a spoken word whose text appears nowhere in that image's GT.

OCR output is cached under data/reader/cache/, so re-scoring is cheap.
"""

from __future__ import annotations

import glob
import json
import math
import os
import pickle
import re
import sys
import time
import unicodedata
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from veris import packet, reader, wearcam  # noqa: E402

RAW = ROOT / "data/reader/raw"
CACHE = ROOT / "data/reader/cache"
OUT = ROOT / "evals/results/reader_ocr.json"
PROFILES = ("outdoor_walk", "indoor_walk", "reading_close")
RESOLUTIONS = (1280, 960, 768, 640)
THRESHOLDS = [round(float(x), 2) for x in np.arange(0.0, 1.0, 0.02)] + [0.99, 0.995, 0.999]
HEIGHT_EDGES = (0, 8, 12, 16, 24, 32, 48, 10_000)
TARGET_PRECISION = 0.95


# -- ground truth -------------------------------------------------------------
def norm(s: str) -> str:
    s = re.sub(r"[‘’´`]", "'", s.lower())      # GT mixes apostrophes
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return re.sub(r"^[^\w]+|[^\w]+$", "", s)


def glyph_height(poly: np.ndarray) -> float:
    """Short side of a strip-shaped polygon: solves L+h = P/2, L*h = A, so it
    also works for curved text where minAreaRect overestimates."""
    a = abs(cv2.contourArea(poly.astype(np.float32)))
    p2 = cv2.arcLength(poly.astype(np.float32), True) / 2
    return float((p2 - math.sqrt(max(p2 * p2 - 4 * a, 0))) / 2)


def _split_entry(poly: np.ndarray, text: str):
    """A GT box holding 'Genaxis Theatre' -> one quad per word, by char count."""
    parts = text.split()
    if len(parts) <= 1 or len(poly) != 4:
        return [(poly, text)]
    q = reader._order_quad(poly.astype(np.float32))
    total, pos, out = len(text), 0, []
    for w in parts:
        i = text.index(w, pos)
        f0, f1 = i / total, (i + len(w)) / total
        pos = i + len(w)
        top = lambda f: q[0] + f * (q[1] - q[0])
        bot = lambda f: q[3] + f * (q[2] - q[3])
        out.append((np.array([top(f0), top(f1), bot(f1), bot(f0)]), w))
    return out


def load_icdar(split: str):
    """split: test | train -> [(image path, [(poly, text, dontcare)])]"""
    d = RAW / "icdar2015"
    imgs = sorted(glob.glob(str(d / f"{split}_img/*.jpg")),
                  key=lambda p: int(re.findall(r"\d+", Path(p).stem)[0]))
    out = []
    for p in imgs:
        gt = []
        for line in open(d / f"{split}_gt/gt_{Path(p).stem}.txt", encoding="utf-8-sig"):
            f = line.rstrip("\r\n").split(",", 8)
            if len(f) < 9:
                continue
            poly = np.array(f[:8], np.float32).reshape(4, 2)
            dc = f[8] == "###"
            for q, t in ([(poly, f[8])] if dc else _split_entry(poly, f[8])):
                gt.append((q, t, dc or not norm(t)))
        out.append((p, gt))
    return out


def load_totaltext():
    d = RAW / "totaltext"
    pat = re.compile(r"x: \[\[(.*?)\]\], y: \[\[(.*?)\]\], ornt: \[(.*?)\], "
                     r"transcriptions: \[u?(['\"])(.*?)\4\]", re.S)
    out = []
    for p in sorted(glob.glob(str(d / "Images/Test/*.jpg")),
                    key=lambda p: int(re.findall(r"\d+", Path(p).stem)[0])):
        txt = (d / f"txt_format/Test/poly_gt_{Path(p).stem}.txt").read_text()
        gt = []
        for m in pat.finditer(txt):
            poly = np.stack([np.array(m.group(1).split(), np.float32),
                             np.array(m.group(2).split(), np.float32)], 1)
            t = m.group(5)
            gt.append((poly, t, t == "#" or not norm(t)))
        out.append((p, gt))
    return out


# -- wearcam geometry for GT points -------------------------------------------
def warp_points(pts: np.ndarray, shape, profile: str, seed: int) -> np.ndarray:
    """Map points of the clean image into wearcam.degrade(img, profile, seed).
    Mirrors degrade()'s geometry: downscale, barrel remap, head roll (its first
    rng draw). Checked against degrade() itself in check_warp()."""
    p = wearcam.PROFILES[profile]
    h, w = shape[:2]
    s = p.long_side / max(h, w)
    pts = pts.astype(np.float64).copy()
    if s < 1:
        w2, h2 = round(w * s), round(h * s)
        pts[:, 0] *= w2 / w
        pts[:, 1] *= h2 / h
        w, h = w2, h2
    if p.barrel_k1:
        f = 0.5 * w / np.tan(np.radians(p.hfov_deg / 2))
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
        dist = np.array([p.barrel_k1, 0, 0, 0], np.float64)
        # degrade samples source = distort_{-k1}(dest); invert that
        pts = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, -dist, P=K).reshape(-1, 2)
    if p.roll_deg:
        roll = np.random.default_rng(seed).uniform(-p.roll_deg, p.roll_deg)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), roll, 1.0)
        pts = pts @ M[:, :2].T + M[:, 2]
    return pts.astype(np.float32)


def check_warp() -> float:
    img = np.zeros((720, 1280, 3), np.uint8)
    pts = np.array([[x, y] for x in range(80, 1280, 160) for y in range(60, 720, 120)], np.float32)
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), 6, (255, 255, 255), -1)
    worst = 0.0
    for prof in PROFILES:
        d = cv2.cvtColor(wearcam.degrade(img, prof, seed=7), cv2.COLOR_BGR2GRAY)
        n, _, _, cents = cv2.connectedComponentsWithStats((d > 100).astype(np.uint8))
        wp = warp_points(pts, img.shape, prof, 7)
        for q in wp:
            if 20 < q[0] < 1260 and 20 < q[1] < 700:
                worst = max(worst, float(np.min(np.linalg.norm(cents[1:] - q, axis=1))))
    assert worst < 3.0, f"warp_points disagrees with wearcam.degrade by {worst:.1f} px"
    return worst


# -- running the OCR ----------------------------------------------------------
_readers: dict = {}


def _ocr(task):
    """One image under one condition -> raw words + GT in OCR-input pixels."""
    path, gt, cond, res, det_side, seed, cfg = task
    key = (res, det_side, cfg)
    if key not in _readers:
        kw = dict(cfg)
        _readers[key] = reader.Reader(long_side=res, det_side=det_side, threads=1, **kw)
    r = _readers[key]
    img = cv2.imread(path)
    polys = [g[0] for g in gt]
    if cond != "clean":
        polys = [warp_points(q, img.shape, cond, seed) for q in polys]
        img = wearcam.degrade(img, cond, seed=seed)
    s = min(1.0, res / max(img.shape[:2]))
    t0 = time.perf_counter()
    words, _, _ = r.words(img)
    dt = time.perf_counter() - t0
    gts = [(q * s, t, dc) for q, (_, t, dc) in zip(polys, gt)]
    return dict(words=[(w.text, w.conf, w.quad, w.probs) for w in words], gt=gts, ocr_s=dt)


def run(items, cond, res, det_side=None, cfg=(), tag="", procs=12):
    name = f"{tag}_{cond}_{res}" + (f"_det{det_side}" if det_side else "")
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    tasks = [(p, gt, cond, res, det_side, i, cfg) for i, (p, gt) in enumerate(items)]
    t0 = time.time()
    with Pool(procs) as pool:
        out = pool.map(_ocr, tasks, chunksize=4)
    print(f"{name}: {len(out)} images in {time.time() - t0:.0f} s", flush=True)
    f.write_bytes(pickle.dumps(out))
    return out


# -- scoring ------------------------------------------------------------------
def edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def match_image(rec):
    """-> preds [(conf, correct, invented, text, within_one_edit_of_its_gt)],
          gts [(height_px, matched_conf, matched_pred_text, gt_text)]"""
    gts = [(q, norm(t), glyph_height(q)) for q, t, dc in rec["gt"] if not dc]
    dcs = [(q, glyph_height(q)) for q, t, dc in rec["gt"] if dc]
    bag = {t for _, t, _ in gts}
    preds = [(norm(w[0]), w[1], w[2].mean(0)) for w in rec["words"] if norm(w[0])]

    pairs = []
    for i, (pt, _, pc) in enumerate(preds):
        for j, (q, gt_t, h) in enumerate(gts):
            d = cv2.pointPolygonTest(q.astype(np.float32), (float(pc[0]), float(pc[1])), True)
            if d >= -0.5 * h:
                pairs.append((pt != gt_t, -d, i, j))
    pairs.sort()
    p2g, g2p = {}, {}
    for _, _, i, j in pairs:
        if i not in p2g and j not in g2p:
            p2g[i], g2p[j] = j, i

    out_p = []
    for i, (pt, c, pc) in enumerate(preds):
        if i not in p2g and any(cv2.pointPolygonTest(q.astype(np.float32),
                                                     (float(pc[0]), float(pc[1])), True)
                                >= -0.5 * h for q, h in dcs):
            continue                                    # don't-care region
        ok = i in p2g and gts[p2g[i]][1] == pt
        near = i in p2g and edit(pt, gts[p2g[i]][1]) <= 1
        out_p.append((c, ok, pt not in bag, pt, near))
    out_g = []
    for j, (_, gt_t, h) in enumerate(gts):
        i = g2p.get(j)
        out_g.append((h, preds[i][1] if i is not None else None,
                      preds[i][0] if i is not None else None, gt_t))
    return out_p, out_g


def score(recs, thresholds=THRESHOLDS):
    P, G = [], []
    for r in recs:
        p, g = match_image(r)
        P += p
        G += g
    conf = np.array([p[0] for p in P])
    ok = np.array([p[1] for p in P], bool)
    inv = np.array([p[2] for p in P], bool)
    near = np.array([p[4] for p in P], bool)
    n_gt = len(G)
    curve = []
    for t in thresholds:
        sp = conf >= t
        correct = int((ok & sp).sum())
        curve.append(dict(t=t, spoken=int(sp.sum()), correct=correct,
                          precision=round(correct / max(1, sp.sum()), 4),
                          recall=round(correct / max(1, n_gt), 4),
                          precision_within_1_edit=round(
                              float((near & sp).sum()) / max(1, sp.sum()), 4),
                          invented_rate=round(float((inv & sp).sum()) / max(1, sp.sum()), 4)))
    return dict(P=P, G=G, n_gt=n_gt, curve=curve)


def threshold_for(curve, target=TARGET_PRECISION):
    """Lowest threshold whose spoken-word precision reaches the target and
    stays there at every higher threshold with words left to speak."""
    ok = [c for c in curve if c["spoken"] >= 20]
    for k, c in enumerate(ok):
        if all(x["precision"] >= target for x in ok[k:]):
            return c["t"]
    return None


def at(curve, t):
    return min(curve, key=lambda c: abs(c["t"] - t))


def summary(sc, t):
    c = at(sc["curve"], t)
    matched = [(g[2], g[3]) for g in sc["G"] if g[1] is not None and g[1] >= t]
    cer = sum(edit(p, g) for p, g in matched) / max(1, sum(len(g) for _, g in matched))
    buckets = []
    for lo, hi in zip(HEIGHT_EDGES, HEIGHT_EDGES[1:]):
        gs = [g for g in sc["G"] if lo <= g[0] < hi]
        hit = sum(1 for g in gs if g[1] is not None and g[1] >= t and g[2] == g[3])
        buckets.append(dict(px=f"{lo}-{hi if hi < 10_000 else 'inf'}", n=len(gs),
                            recall=round(hit / max(1, len(gs)), 4)))
    return dict(threshold=c["t"], precision=c["precision"], recall=c["recall"],
                precision_within_1_edit=c["precision_within_1_edit"],
                invented_rate=c["invented_rate"], spoken=c["spoken"],
                cer_matched=round(cer, 4), n_matched=len(matched),
                recall_by_glyph_height_px=buckets)


def timing(items, res_list, n=40, det_side=None):
    """Single process, onnxruntime default threads, clean images."""
    out = {}
    for res in res_list:
        r = reader.Reader(long_side=res, det_side=det_side)
        imgs = [cv2.imread(p) for p, _ in items[:n]]
        r.words(imgs[0])
        ts = []
        for im in imgs:
            t0 = time.perf_counter()
            r.words(im)
            ts.append(time.perf_counter() - t0)
        out[str(res) + (f"_det{det_side}" if det_side else "")] = dict(
            n=n, mean_ms=round(1000 * float(np.mean(ts))),
            p90_ms=round(1000 * float(np.percentile(ts, 90))))
    return out


ENGINES = {
    # v6 det may expect mean/std 0.5 (rapidocr's config) rather than ImageNet
    # normalisation used here; not checked, see report
    "v5mobile_det+en_v5mobile_rec": dict(det_model="ch_PP-OCRv5_det_mobile.onnx",
                                         rec_model="en_PP-OCRv5_rec_mobile.onnx"),
    "v5mobile_det+en_v4mobile_rec": dict(det_model="ch_PP-OCRv5_det_mobile.onnx",
                                         rec_model="en_PP-OCRv4_rec_mobile.onnx"),
    "v6tiny_det+v6tiny_rec": dict(det_model="PP-OCRv6_det_tiny.onnx",
                                  rec_model="PP-OCRv6_rec_tiny.onnx"),
}


def model_sizes():
    return {f: os.path.getsize(reader.MODEL_DIR / f)
            for f in sorted(os.listdir(reader.MODEL_DIR)) if f.endswith(".onnx")}


def main():
    warp_err = check_warp()
    print(f"warp_points vs degrade: worst {warp_err:.2f} px")
    test = {"icdar15": load_icdar("test"), "totaltext": load_totaltext()}
    train = load_icdar("train")
    res_out = dict(
        command="uv run python evals/reader_ocr.py",
        engine="veris/reader.py (onnxruntime CPU)",
        default_models={"det": reader.DET_MODEL, "rec": reader.REC_MODEL},
        model_bytes=model_sizes(),
        splits=dict(icdar15="ICDAR2015 ch4 test (500)", totaltext="Total-Text test (300)",
                    calibration="ICDAR2015 ch4 train, first 500 images",
                    engine_selection="ICDAR2015 ch4 train, images 500-799 (clean, 1280)"),
        matching=__doc__.split("Matching.")[1].split("OCR output")[0].strip(),
        n_images={k: len(v) for k, v in test.items()},
        n_gt_words={k: sum(1 for _, g in v for x in g if not x[2]) for k, v in test.items()},
        warp_check_px=round(warp_err, 2),
    )

    # 1. engine selection on train images (never on test)
    sel = {}
    for name, cfg in ENGINES.items():
        sc = score(run(train[500:800], "clean", 1280, cfg=tuple(cfg.items()), tag=f"sel_{name}"))
        best = max(sc["curve"], key=lambda c: 2 * c["precision"] * c["recall"]
                   / max(1e-9, c["precision"] + c["recall"]))
        sel[name] = dict(at_t0=summary(sc, 0.0) | {"recall_by_glyph_height_px": None},
                         best_f1=best, t_for_p95=threshold_for(sc["curve"]))
    res_out["engine_selection"] = sel

    # 2. calibration on ICDAR15 train, pooled and per condition, at 1280
    cal = {}
    pooled = []
    for cond in ("clean",) + PROFILES:
        recs = run(train[:500], cond, 1280, tag="cal")
        pooled += recs
        cal[cond] = threshold_for(score(recs)["curve"])
    cal_pooled = score(pooled)
    cal["pooled"] = threshold_for(cal_pooled["curve"])
    rec_t = max(v for v in cal.values() if v is not None)
    res_out["calibration"] = dict(
        target_precision=TARGET_PRECISION, threshold_by_condition=cal,
        recommended_LEGIBLE_CONF=rec_t,
        rule="max over conditions of the lowest threshold at which spoken-word precision "
             "is >= target at it and every higher threshold (ICDAR15 train)",
        pooled_curve=cal_pooled["curve"])

    # 3. test sets: all conditions x resolutions
    results = {}
    for ds, items in test.items():
        results[ds] = {}
        for cond in ("clean",) + PROFILES:
            results[ds][cond] = {}
            for res in RESOLUTIONS:
                sc = score(run(items, cond, res, tag=ds))
                results[ds][cond][str(res)] = dict(
                    all_words=summary(sc, 0.0),
                    at_placeholder_0_60=summary(sc, packet.LEGIBLE_CONF),
                    at_recommended=summary(sc, rec_t),
                    test_threshold_for_p95=threshold_for(sc["curve"]),
                    curve=sc["curve"] if res == 1280 else None)
        # detection at 640 on the 1280 frame, recognition crops at 1280
        for cond in ("clean", "indoor_walk"):
            sc = score(run(items, cond, 1280, det_side=640, tag=ds))
            results[ds][cond]["1280_det640"] = dict(
                all_words=summary(sc, 0.0), at_recommended=summary(sc, rec_t),
                at_placeholder_0_60=summary(sc, packet.LEGIBLE_CONF))
    res_out["results"] = results
    res_out["timing_cpu_x86"] = dict(
        note="desktop x86 CPU, 12 cores, onnxruntime default threads, ICDAR15 test clean; "
             "not Orin numbers. The machine is shared with other agents: see loadavg",
        loadavg_before=os.getloadavg(),
        per_resolution=timing(test["icdar15"], RESOLUTIONS)
        | timing(test["icdar15"], [1280], det_side=640),
        pooled_worker_mean_ms={f"{ds}_{res}": round(1000 * float(np.mean(
            [r["ocr_s"] for r in run(test[ds], "clean", res, tag=ds)]))) for ds in test
            for res in RESOLUTIONS})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res_out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
