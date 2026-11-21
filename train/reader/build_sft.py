"""The reader expert's training set, targets built deterministically from
ground truth (no teacher model). ICDAR 2015 ch4 TRAIN only.

    .venv/bin/python train/reader/build_sft.py      # -> data/experts/sft/reader.json
    .venv/bin/python train/reader/build_sft.py --demo

Per image: OCR (veris.reader, LEGIBLE_CONF) finds text regions; reader.foveate
packs them into a mosaic; the user turn is exactly evals/reader_fovea.py path c:
"<image>" + user_text(EXPERTS["reader"], Packet(text=hint lines).to_prompt("reader")).

Target: per kept region, in mosaic order, the ground-truth words whose centre
falls inside that region's crop, in reading order, said in reader.compose's
format with read text in double quotes. A GT word is said only when the
pixels support it:
  - it is not "###" (annotators could not read it),
  - its glyph height in the mosaic is >= MIN_GLYPH_PX,
  - and, on wearcam-degraded frames, the OCR recognised it at any confidence
    with at most one substituted character, never a missing one, so a clipped
    word is not completed (blur that defeats the recogniser is assumed to defeat the
    model too, so the model is never trained to read pixels that are gone).
Everything else in a region is said as "something I can't read" at its
position; a region with no supported word is one can't-read statement; a
mosaic with none at all says the text is too blurry (degraded) or cannot be
read (clean).

Images cycle clean / outdoor_walk / reading_close; every 9th image also gets
an indoor_walk row (about 10% of rows).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))
import reader_ocr as E  # noqa: E402
from veris import packet, reader, wearcam  # noqa: E402
from veris.experts import EXPERTS, user_text  # noqa: E402

SFT = ROOT / "data/experts/sft"
MOSAICS = ROOT / "data/reader/sft/mosaic"
CONDS = ("clean", "outdoor_walk", "reading_close")
MIN_GLYPH_PX = 8
EX = EXPERTS["reader"]

QUOTED = re.compile(r'["“”]([^"“”]*)["“”]')
SAYS_UNREADABLE = re.compile(
    r"\b(can'?t|cannot|can not|couldn'?t|could not|unable to|not able to)\s+(be\s+)?"
    r"(read|make out)|\bunreadable\b|\billegible\b|\btoo (blurry|small|dark|faint|far)"
    r"|\bnot readable\b|\bcut off\b|\bhidden\b|\bblocked\b|\bobscured\b|\bblurr(y|ed)\b", re.I)


def _legible_words(ocr_lines) -> set[str]:
    out = set()
    for ln in ocr_lines:
        ln = ln if isinstance(ln, packet.TextLine) else packet.TextLine(**ln)
        if ln.legible:
            out |= {E.norm(w) for w in ln.text.split()}
    return out - {""}


def verify_target(target_text: str, gt_words, ocr_lines) -> tuple[bool, str]:
    """Exact acceptance check for a teacher-written reader target.

    Read text must be in double quotes (straight or curly). Comparison is
    case-insensitive with leading/trailing punctuation stripped.
      1. every quoted word is a legible OCR word AND a ground-truth word
      2. no read text outside quotes: no digits, no ALL-CAPS tokens of 2+ letters
      3. if OCR reported any unreadable / low-confidence text, the target says
         that something could not be read
      4. if OCR read at least one GT word legibly, the target quotes something
      5. no packet markup ("[unreadable]", "<sense>") and balanced quotes
    """
    t = target_text.replace("’", "'")
    if "[unreadable]" in t.lower() or "<sense" in t.lower():
        return False, "echoes packet markup"
    if (t.count('"') + t.count("“") + t.count("”")) % 2:
        return False, "unbalanced quotes"
    gt = {E.norm(w) for w in gt_words} - {""}
    legible = _legible_words(ocr_lines)
    quoted = [E.norm(w) for q in QUOTED.findall(t) for w in q.split()]
    quoted = [w for w in quoted if w]
    for w in quoted:
        if w not in legible:
            return False, f'quoted "{w}" is not a legible OCR word'
        if w not in gt:
            return False, f'quoted "{w}" is not in the ground truth'
    for tok in QUOTED.sub(" ", t).split():
        core = re.sub(r"^[^\w]+|[^\w]+$", "", tok)
        if re.search(r"\d", core) or (len(core) >= 2 and core.isalpha() and core.isupper()):
            return False, f'unquoted read text "{core}"'
    lines = [ln if isinstance(ln, packet.TextLine) else packet.TextLine(**ln) for ln in ocr_lines]
    if any(not ln.legible for ln in lines) and not SAYS_UNREADABLE.search(t):
        return False, "OCR reported unreadable text but the target does not say so"
    if legible & gt and not quoted:
        return False, "OCR read ground-truth words but the target quotes none"
    return True, "ok"


def _target(img_shape, boxes, kept, gt, ocr_words, degraded, scales):
    """-> (target text, quoted GT words, n regions with a read word)."""
    H, W = img_shape[:2]
    f = (W / 2) / math.tan(math.radians(reader.HFOV_DEG / 2))
    ocr = [E.norm(w.text) for w in ocr_words]
    taken = set()
    sources, bearings, quoted, n_read = [], [], [], 0
    for k, g in enumerate(kept):
        x0, y0, x1, y1 = boxes[g]
        p = 0.15 * (y1 - y0)
        x0, y0, x1, y1 = x0 - p, y0 - p, x1 + p, y1 + p
        inside = []
        for j, (q, t, dc) in enumerate(gt):
            cx, cy = q.mean(0)
            if j not in taken and x0 <= cx <= x1 and y0 <= cy <= y1:
                taken.add(j)
                inside.append((q, t, dc))
        parts = []
        if inside:
            hmed = float(np.median([E.glyph_height(q) for q, _, _ in inside]))
            inside.sort(key=lambda w: (round(w[0].mean(0)[1] / max(1.0, 0.5 * hmed)),
                                       w[0][:, 0].min()))
        for q, t, dc in inside:
            n = E.norm(t)
            ok = (not dc and E.glyph_height(q) * scales[k] >= MIN_GLYPH_PX
                  and (not degraded or any(len(o) == len(n) and E.edit(n, o) <= 1
                                           for o in ocr)))
            if ok and parts and parts[-1][1]:
                parts[-1] = (parts[-1][0] + " " + t, True)     # one quote per run
            else:
                parts.append((t if ok else "", ok))
            if ok:
                quoted.append(t)
        if not parts:
            parts = [("", False)]
        n_read += any(ok for _, ok in parts)
        sources.append(parts)
        bearings.append(math.degrees(math.atan(((x0 + x1) / 2 - W / 2) / f)))
    if not n_read:
        return ("There is text, but it is too blurry to read." if degraded
                else "There is text, but I can't read it."), [], 0
    return reader.compose(sources, bearings, quote=True), quoted, n_read


def _row(task):
    i, path, gt, cond = task
    if not hasattr(_row, "r"):
        _row.r = reader.Reader(threads=1)
    seed = 10_000 + i
    img = cv2.imread(path)
    polys = [g[0] for g in gt]
    if cond != "clean":
        polys = [E.warp_points(q, img.shape, cond, seed) for q in polys]
        img = wearcam.degrade(img, cond, seed=seed)
    words, quads, shape = _row.r.words(img)
    lines, boxes = reader.lines_and_boxes(words, quads, shape)
    k = img.shape[0] / shape[0]
    boxes = [tuple(v * k for v in b) for b in boxes]
    if not boxes:
        return None
    mosaic, kept = reader.foveate(img, boxes)
    # the scale each kept crop got in the mosaic (foveate shrinks wide crops only)
    scales = []
    for g in kept:
        x0, y0, x1, y1 = boxes[g]
        p = 0.15 * (y1 - y0)
        cw, ch = x1 - x0 + 2 * p, y1 - y0 + 2 * p
        scales.append(min(1.0, 448 / cw, 448 / ch))
    gtw = [(q * k / k, t, dc) for q, (_, t, dc) in zip(polys, gt)]
    target, quoted, n_read = _target(img.shape, boxes, kept, gtw, words,
                                     cond != "clean", scales)
    hint = [ln for g in kept for ln in lines if ln.group == g]
    sid = f"icdar15_train_{Path(path).stem}_{cond}"
    MOSAICS.mkdir(parents=True, exist_ok=True)
    mpath = MOSAICS / f"{sid}.png"
    cv2.imwrite(str(mpath), mosaic)
    gt_set = {E.norm(t) for _, t, dc in gt if not dc}
    assert all(E.norm(w) in gt_set for q in quoted for w in q.split()), sid
    return {"messages": [
        {"role": "system", "content": EX.system},
        {"role": "user", "content": "<image>" + user_text(
            EX, packet.Packet(text=tuple(hint)).to_prompt("reader"))},
        {"role": "assistant", "content": target}],
        "images": [str(mpath)],
        "_meta": dict(id=sid, cond=cond, regions=len(boxes), kept=len(kept),
                      regions_read=n_read, words_quoted=len(quoted),
                      mosaic_hw=list(mosaic.shape[:2]))}


def build() -> list[dict]:
    from train.experts.build import CONFIG

    items = E.load_icdar("train")
    tasks = [(i, p, gt, CONDS[i % 3]) for i, (p, gt) in enumerate(items)]
    tasks += [(i, p, gt, "indoor_walk") for i, (p, gt) in enumerate(items) if i % 9 == 0]
    with Pool(12) as pool:
        rows = [r for r in pool.map(_row, tasks, chunksize=8) if r]
    meta = [r.pop("_meta") for r in rows]
    SFT.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(SFT / "reader.json", "w"), indent=1)
    # read-merge-write: other builders own the other keys
    info_path = SFT / "dataset_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info["veris_reader"] = {
        "file_name": "reader.json", "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                 "assistant_tag": "assistant", "system_tag": "system"}}
    info_path.write_text(json.dumps(info, indent=1))
    (SFT / "qlora_reader.yaml").write_text(CONFIG.format(
        sft=SFT, scene="reader", out=ROOT / "train/experts/out-reader"))
    stats = dict(
        rows=len(rows), images_without_text_regions=len(tasks) - len(rows),
        by_condition=dict(Counter(m["cond"] for m in meta)),
        rows_reading_something=sum(m["regions_read"] > 0 for m in meta),
        rows_all_unreadable=sum(m["regions_read"] == 0 for m in meta),
        words_quoted=sum(m["words_quoted"] for m in meta),
        regions_kept_mean=round(float(np.mean([m["kept"] for m in meta])), 2),
        mosaic_tokens_mean=round(float(np.mean([m["mosaic_hw"][0] * m["mosaic_hw"][1] / 784
                                                for m in meta])), 1))
    json.dump(stats, open(SFT / "reader_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))
    return rows


def demo() -> None:
    L = lambda t, c=0.99: packet.TextLine(t, c)
    lines = [L("EMERGENCY EXIT"), L("", 0.3), L("PUSH", 0.99)]
    gt = ["Emergency", "Exit", "only", "PUSH"]
    ok = lambda s, g=gt, ln=lines: verify_target(s, g, ln)

    assert ok('The sign says "Emergency exit", then a line I can\'t read, '
              'then "push".') == (True, "ok")
    assert ok('The door sign says “EMERGENCY EXIT”. Below it, the text is '
              'too blurry to read. Lower down: “PUSH”.')[0]
    # completes the illegible line from world knowledge -> not a legible OCR word
    assert ok('The sign says "Emergency exit only", then "push". Some text is unreadable.') \
        == (False, 'quoted "only" is not a legible OCR word')
    # silent about the unreadable line
    assert ok('The sign says "Emergency exit" and "push".')[1].startswith("OCR reported")
    # read text left outside quotes
    assert ok('The sign says EMERGENCY EXIT. One line is unreadable.')[1].startswith("unquoted")
    assert ok('Platform 4. "push". The rest is unreadable.')[1] == 'unquoted read text "4"'
    # legible OCR word that the ground truth does not contain (an OCR misread)
    bad = [L("EMERGENCV EXIT")]
    assert ok('It says "Emergencv exit".', ln=bad) == \
        (False, 'quoted "emergencv" is not in the ground truth')
    assert ok('It says "exit"; the first word looks wrong.', ln=bad)[0]
    # nothing quoted although OCR read GT words
    assert ok("There is a sign I can't read.")[1].startswith("OCR read ground-truth")
    # nothing legible at all: abstaining is the right answer
    assert ok("There is a sign, but I can't read it.", ln=[L("", 0.1)])[0]
    assert not ok("There is a sign.", ln=[L("", 0.1)])[0]
    assert ok('It says "[unreadable] exit".')[1] == "echoes packet markup"
    assert ok('It says "push. Some text is unreadable.')[1] == "unbalanced quotes"
    # rows store lines as dicts; same verdicts
    assert ok('"push", then something I can\'t read, "emergency exit"',
              ln=[ln.__dict__ for ln in lines])[0]
    # deterministic targets: region order, quotes, can't-read at the right spot
    q = lambda x0, y0, x1, y1: np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    gt = [(q(10, 10, 60, 30), "PUSH", False), (q(70, 10, 120, 30), "###", True),
          (q(500, 300, 560, 320), "Exit", False)]
    W = reader.Word
    boxes = [(10, 10, 120, 30), (500, 300, 560, 320)]
    t, quoted, n = _target((720, 1280), boxes, [0, 1], gt, [], False, [1.0, 1.0])
    assert t == 'On your left: "PUSH", then something I can\'t read. Next: "Exit".', t
    assert quoted == ["PUSH", "Exit"] and n == 2
    # degraded frame: a GT word the recogniser could not get near is not read
    t, quoted, n = _target((720, 1280), boxes, [1], gt, [W("Exlt", 0.3, None, 0)], True, [1.0])
    assert t == '"Exit".' and quoted == ["Exit"], t
    t, _, n = _target((720, 1280), boxes, [1], gt, [W("xit", 0.3, None, 0)], True, [1.0])
    assert t == "There is text, but it is too blurry to read." and n == 0
    # shrunk below MIN_GLYPH_PX in the mosaic -> can't read
    t, _, _ = _target((720, 1280), boxes, [0, 1], gt, [], False, [0.2, 1.0])
    assert t.startswith("On your left: Something I can't read. "), t
    print("verify_target and targets: all checks pass")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    demo()
    if not a.demo:
        build()
