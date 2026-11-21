"""Does pixel geometry give usable range, bearing and closing time?

KITTI labels + calibration only, no images. Camera 1.65 m up, rectified and
roughly level (pitch assumed 0; road slope and real pitch show up as error).

Range: un-truncated, un-occluded Pedestrian / Cyclist / Car in the object
training labels. Predicted = ranging.range_m from the 2D box bottom row and
centre column. Ground truth "near" = horizontal distance from the camera to
the nearest point of the 3D box footprint (what you would walk into);
"centre" = distance to the 3D box centre, for reference.
Sensitivity: the same, with the assumed pitch off by +-1/+-2 deg and the
assumed height off by +-10 cm (a head nods and bobs).

Time to contact: KITTI tracking labels at 10 Hz. tau_2d = ranging.ttc_s on
2D box height over a window; GT tau = z / (-dz/dt) from a line fit to the 3D z
over the same window, for objects whose z is falling.

  .venv/bin/python evals/nav_ranging.py
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from veris.ranging import Camera, bearing_deg, range_m, ttc_s  # noqa: E402

K = REPO / "data/nav/kitti/training"
CLASSES = ("Pedestrian", "Cyclist", "Car")
BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 1e9)]
H = 1.65


def p2(path):
    for line in open(path):
        if line.startswith(("P2:", "P2 ")):
            return np.array(line.split()[1:], float).reshape(3, 4)


def cam_from(P, height_m=H, pitch_deg=0.0):
    return Camera(1242, 375, P[0, 0], P[1, 1], P[0, 2], P[1, 2], height_m, pitch_deg)


def footprint_near(x, z, l, w, ry):
    """Min horizontal distance from the camera to the box footprint edges."""
    c, s = np.cos(ry), np.sin(ry)
    pts = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    corners = np.c_[x + pts[:, 0] * c + pts[:, 1] * s, z - pts[:, 0] * s + pts[:, 1] * c]
    t = np.linspace(0, 1, 21)[:, None]
    edge = np.concatenate([corners[i] + t * (corners[(i + 1) % 4] - corners[i]) for i in range(4)])
    return float(np.hypot(edge[:, 0], edge[:, 1]).min())


def load_objects():
    rows = []
    for f in sorted((K / "label_2").glob("*.txt")):
        P = p2(K / "calib" / f.name)
        for line in open(f):
            a = line.split()
            if a[0] not in CLASSES or float(a[1]) != 0 or int(a[2]) != 0:
                continue
            x1, y1, x2, y2, h, w, l, x, y, z, ry = map(float, a[4:15])
            x2cam = x + P[0, 3] / P[0, 0]          # label is in cam0; image is cam2
            rows.append(dict(cls=a[0], P=P, u=(x1 + x2) / 2, vb=y2,
                             near=footprint_near(x2cam, z, l, w, ry),
                             centre=float(np.hypot(x2cam, z)),
                             bearing=float(np.degrees(np.arctan2(x2cam, z)))))
    return rows


def summarize(pred, gt):
    pred, gt = np.asarray(pred), np.asarray(gt)
    ok = np.isfinite(pred)
    out = {"n": int(len(gt)), "beyond_horizon": int((~ok).sum())}
    ape = np.abs(pred[ok] - gt[ok]) / gt[ok] * 100
    sgn = (pred[ok] - gt[ok]) / gt[ok] * 100
    out["median_ape_pct"] = round(float(np.median(ape)), 1) if ok.any() else None
    out["p90_ape_pct"] = round(float(np.percentile(ape, 90)), 1) if ok.any() else None
    out["median_signed_pct"] = round(float(np.median(sgn)), 1) if ok.any() else None
    out["by_bucket"] = {}
    for lo, hi in BUCKETS:
        m = (gt >= lo) & (gt < hi)
        mk = m & ok
        key = f"{lo}-{hi:g}" if hi < 1e9 else f"{lo}+"
        out["by_bucket"][key] = {
            "n": int(m.sum()), "beyond_horizon": int((m & ~ok).sum()),
            "median_ape_pct": round(float(np.median(np.abs(pred[mk] - gt[mk]) / gt[mk] * 100)), 1)
            if mk.any() else None}
    return out


def range_eval(rows, pitch=0.0, dh=0.0, ref="near"):
    res = {}
    for cls in CLASSES:
        rs = [r for r in rows if r["cls"] == cls]
        pred = [float(range_m(cam_from(r["P"], H + dh, pitch), r["u"], r["vb"])) for r in rs]
        res[cls] = summarize(pred, [r[ref] for r in rs])
    return res


def load_tracks():
    tracks = {}
    for f in sorted((K / "label_02").glob("*.txt")):
        for line in open(f):
            a = line.split()
            if a[2] not in CLASSES or int(a[3]) != 0 or int(a[4]) != 0:
                continue
            y1, y2, z = float(a[7]), float(a[9]), float(a[15])
            tracks.setdefault((f.stem, int(a[1]), a[2]), []).append((int(a[0]), y2 - y1, z))
    return tracks


def ttc_eval(tracks, win, thr=3.0, jitter=0.0):
    """jitter: relative Gaussian noise on box height, standing in for detector noise."""
    rng = np.random.default_rng(0)
    samples = []
    for (_, _, cls), obs in tracks.items():
        obs = sorted(obs)
        for i in range(win - 1, len(obs)):
            w = obs[i - win + 1:i + 1]
            if w[-1][0] - w[0][0] != win - 1:            # need contiguous frames
                continue
            t = np.array([o[0] for o in w]) / 10.0
            s = np.array([o[1] for o in w]) * (1 + jitter * rng.standard_normal(win))
            z = np.array([o[2] for o in w])
            dz = np.polyfit(t, z, 1)[0]
            gt = z[-1] / -dz if dz < -1e-3 else np.inf
            est = ttc_s(t, s, min_span_s=0.2, max_s=20.0)
            samples.append((cls, gt, np.inf if est is None else est))
    out = {}
    for cls in CLASSES + ("all",):
        S = [x for x in samples if cls in ("all", x[0])]
        gt = np.array([x[1] for x in S]); est = np.array([x[2] for x in S])
        app = np.isfinite(gt) & (gt <= 10)
        both = app & np.isfinite(est)
        d = {"n_windows": int(len(S)), "n_approaching_gt_le_10s": int(app.sum()),
             "coverage_pct": round(float(both.sum() / max(app.sum(), 1) * 100), 1),
             "median_ape_pct": round(float(np.median(np.abs(est[both] - gt[both]) / gt[both]) * 100), 1)
             if both.any() else None, "by_gt_tau": {}}
        for lo, hi in ((0, 3), (3, 6), (6, 10)):
            m = both & (gt >= lo) & (gt < hi)
            d["by_gt_tau"][f"{lo}-{hi}s"] = {
                "n": int((app & (gt >= lo) & (gt < hi)).sum()),
                "median_ape_pct": round(float(np.median(np.abs(est[m] - gt[m]) / gt[m]) * 100), 1)
                if m.any() else None}
        pos, neg = gt < thr, gt > 2 * thr
        d[f"alarm_at_{thr:g}s"] = {
            "recall_pct": round(float((est[pos] < thr).mean() * 100), 1) if pos.any() else None,
            "n_gt_below": int(pos.sum()),
            "false_alarm_pct": round(float((est[neg] < thr).mean() * 100), 2) if neg.any() else None,
            "n_gt_above_2x_or_receding": int(neg.sum())}
        out[cls] = d
    return out


def wearable_table():
    """Analytic: head camera 1.6 m, 15 deg down, 1280x720 at 80 deg HFOV. How far
    off is the range when the true pitch differs from the assumed one?"""
    th = np.radians(15)
    cam = Camera.from_hfov(1280, 720, 80, 1.6, 15)
    out = {}
    for r in (1, 2, 3, 5, 8, 12, 20):
        yc, zc = 1.6 * np.cos(th) - r * np.sin(th), 1.6 * np.sin(th) + r * np.cos(th)
        v = cam.cy + cam.fy * yc / zc
        row = {"row_px": round(float(v), 1)}
        for e in (-2, -1, 1, 2):
            c2 = Camera.from_hfov(1280, 720, 80, 1.6, 15 + e)
            p = float(range_m(c2, cam.cx, v))
            row[f"pitch{e:+d}"] = None if not np.isfinite(p) else round(p, 2)
        for e in (-0.1, 0.1):
            p = float(range_m(Camera.from_hfov(1280, 720, 80, 1.6 + e, 15), cam.cx, v))
            row[f"height{e:+.1f}"] = round(p, 2)
        out[f"{r}m"] = row
    return out


def main():
    rows = load_objects()
    br = {}
    for cls in CLASSES:
        rs = [r for r in rows if r["cls"] == cls]
        err = [abs(float(bearing_deg(cam_from(r["P"]), r["u"])) - r["bearing"]) for r in rs]
        br[cls] = {"n": len(rs), "median_abs_err_deg": round(float(np.median(err)), 2),
                   "p90_abs_err_deg": round(float(np.percentile(err, 90)), 2)}
    tracks = load_tracks()
    res = {
        "command": ".venv/bin/python evals/nav_ranging.py",
        "data": "KITTI object training labels+calib (7481 frames) and tracking training "
                "labels (21 sequences, 10 Hz); labels only, no images",
        "split": "KITTI training labels (nothing here is trained; pure geometry)",
        "filter": "truncated == 0 and occluded == 0",
        "camera": {"height_m": H, "pitch_deg_assumed": 0.0, "intrinsics": "per-frame P2"},
        "range_vs_nearest_footprint_point": range_eval(rows),
        "range_vs_box_centre": range_eval(rows, ref="centre"),
        "bearing_from_box_centre_column": br,
        "sensitivity": {f"pitch{e:+d}deg": range_eval(rows, pitch=e) for e in (-2, -1, 1, 2)}
        | {f"height{e:+.1f}m": range_eval(rows, dh=e) for e in (-0.1, 0.1)},
        "wearable_analytic_pitch15_h1.6": wearable_table(),
        "ttc": {f"window_{w}_frames": ttc_eval(tracks, w) for w in (3, 5, 8)},
        "ttc_with_box_jitter": {f"window_{w}_frames_jitter_{j:g}": ttc_eval(tracks, w, jitter=j)
                                for w in (5, 8) for j in (0.03, 0.06)},
        "range_note": "No 0-5 m samples: KITTI's camera is level with a 28 deg vertical FOV, so "
                      "the ground is out of frame closer than about 6 m. Near range is covered "
                      "only by the analytic wearable table.",
        "ttc_note": "Boxes are labels, not detections; the jitter rows add Gaussian noise to "
                    "box height as a stand-in for detector noise. KITTI ego is a car at up to ~15 m/s: closing speeds and taus differ "
                    "from walking. GT tau from 3D z includes ego motion, as looming does.",
    }
    out = REPO / "evals/results/nav_ranging.json"
    out.write_text(json.dumps(res, indent=1))
    for cls in CLASSES:
        r = res["range_vs_nearest_footprint_point"][cls]
        print(cls, r["n"], r["median_ape_pct"], {k: v["median_ape_pct"] for k, v in r["by_bucket"].items()})
    print("wrote", out)


if __name__ == "__main__":
    main()
