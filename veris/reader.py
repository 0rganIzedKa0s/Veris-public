"""Tier 1b: OCR, reading order and confidence gating.

PP-OCR mobile detection (DB) + recognition (CTC) through onnxruntime. OCR
supplies the characters and a confidence per word; a word under
packet.LEGIBLE_CONF leaves this module as an illegible TextLine with empty
text, so nothing downstream can see, speak or complete it.

    lines = Reader().read(bgr)          # list[TextLine], reading order
    speak_short(lines)                  # str for short text, None -> reader expert

Models (Apache-2.0, from the RapidOCR ONNX export of PaddleOCR):
    ch_PP-OCRv5_det_mobile.onnx   4.8 MB
    en_PP-OCRv5_rec_mobile.onnx   7.9 MB  (436-char English dictionary, embedded)
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from veris.packet import HINT_MIN_LEGIBLE, LEGIBLE_CONF, TextLine, side_from_bearing  # noqa: F401
from veris.wearcam import Profile

MODEL_DIR = Path(os.environ.get("VERIS_READER_MODELS",
                                Path(__file__).resolve().parent.parent / "data/reader/models"))
DET_MODEL = "ch_PP-OCRv5_det_mobile.onnx"
REC_MODEL = "en_PP-OCRv5_rec_mobile.onnx"
HFOV_DEG = Profile().hfov_deg

LONE_LETTER_CONF = 0.5

# speak_short hands over to the reader expert above these
SHORT_MAX_WORDS = 12
SPEAK_CONF = 0.999      # reader_ocr.json: spoken-word precision >= 0.95 only from here
SHORT_MAX_GROUPS = 2


@dataclass
class Word:
    text: str
    conf: float
    quad: np.ndarray        # 4x2 float, tl tr br bl, in input-image pixels
    line: int               # index of the detected line it came from
    probs: tuple = ()       # per-character CTC probabilities


def _word_conf(probs) -> float:
    return float(np.mean(probs))


def _sess(path: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Any 4 points -> tl, tr, br, bl."""
    xs = pts[np.argsort(pts[:, 0])]
    left = xs[:2][np.argsort(xs[:2, 1])]
    right = xs[2:][np.argsort(xs[2:, 1])]
    return np.array([left[0], right[0], right[1], left[1]], np.float32)


def _crop(img: np.ndarray, q: np.ndarray) -> np.ndarray:
    w = int(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[3] - q[2])))
    h = int(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])))
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, M, (max(w, 1), max(h, 1)),
                               borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)


class Reader:
    def __init__(self, model_dir: Path | str = MODEL_DIR, det_model: str = DET_MODEL,
                 rec_model: str = REC_MODEL, long_side: int = 1280,
                 det_side: int | None = None, threads: int = 0,
                 det_thresh: float = 0.3, box_thresh: float = 0.5, unclip: float = 1.6):
        md = Path(model_dir)
        self.det = _sess(md / det_model, threads)
        self.rec = _sess(md / rec_model, threads)
        chars = self.rec.get_modelmeta().custom_metadata_map["character"].splitlines()
        self.chars = [""] + chars + [" "]          # CTC blank first, space last
        self.long_side, self.det_side = long_side, det_side
        self.det_thresh, self.box_thresh, self.unclip = det_thresh, box_thresh, unclip

    # -- detection ---------------------------------------------------------
    def _detect(self, img: np.ndarray) -> list[np.ndarray]:
        h, w = img.shape[:2]
        side = self.det_side or max(h, w)
        s = side / max(h, w)
        rh, rw = max(32, round(h * s / 32) * 32), max(32, round(w * s / 32) * 32)
        x = cv2.resize(img, (rw, rh)).astype(np.float32) / 255.0
        x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
        pred = self.det.run(None, {"x": x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
        mask = cv2.dilate((pred > self.det_thresh).astype(np.uint8), np.ones((2, 2), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        quads = []
        for c in contours[:1000]:
            (cx, cy), (bw, bh), ang = cv2.minAreaRect(c)
            if min(bw, bh) < 3:
                continue
            box = cv2.boxPoints(((cx, cy), (bw, bh), ang))
            m = np.zeros_like(mask)
            cv2.fillPoly(m, [box.astype(np.int32)], 1)
            if cv2.mean(pred, m)[0] < self.box_thresh:
                continue
            # DB unclip of a rectangle: offset each side by area*ratio/perimeter
            d = bw * bh * self.unclip / (2 * (bw + bh))
            bw, bh = bw + 2 * d, bh + 2 * d
            if min(bw, bh) < 5:
                continue
            q = cv2.boxPoints(((cx, cy), (bw, bh), ang))
            q[:, 0] = np.clip(q[:, 0] * w / rw, 0, w - 1)
            q[:, 1] = np.clip(q[:, 1] * h / rh, 0, h - 1)
            q = _order_quad(q)
            if np.linalg.norm(q[0] - q[1]) > 3 and np.linalg.norm(q[0] - q[3]) > 3:
                quads.append(q)
        return quads

    # -- recognition -------------------------------------------------------
    def _recognize(self, img: np.ndarray, quads: list[np.ndarray]) -> list[Word]:
        crops = []
        for q in quads:
            c = _crop(img, q)
            crops.append(np.rot90(c) if c.shape[0] >= 1.5 * c.shape[1] else c)
        order = np.argsort([c.shape[1] / c.shape[0] for c in crops])
        words: list[Word] = []
        for b in range(0, len(order), 6):
            idx = order[b:b + 6]
            ratios = [crops[i].shape[1] / crops[i].shape[0] for i in idx]
            W = max(320, math.ceil(48 * max(ratios)))
            batch = np.zeros((len(idx), 3, 48, W), np.float32)
            rws = []
            for k, (i, r) in enumerate(zip(idx, ratios)):
                rw = min(W, max(1, math.ceil(48 * r)))
                x = cv2.resize(crops[i], (rw, 48)).astype(np.float32) / 255.0
                batch[k, :, :, :rw] = ((x - 0.5) / 0.5).transpose(2, 0, 1)
                rws.append(rw)
            probs = self.rec.run(None, {"x": batch})[0]
            for k, i in enumerate(idx):
                ws = self._ctc_words(probs[k], W / probs.shape[1] / rws[k], quads[i], int(i))
                if len(ws) == 1 and len(ws[0].text) == 1 and ws[0].text.isalpha():
                    # a line that is one lone letter is usually an edge fragment
                    # (ICDAR15 train: most confident false words were these), so it
                    # is reported as unreadable text, never spoken. ponytail: this
                    # also mutes a real one-letter sign ("P"); revisit with glyph size.
                    ws[0].conf = min(ws[0].conf, LONE_LETTER_CONF)
                words += ws
        return words

    def _ctc_words(self, p: np.ndarray, frame_frac: float, q: np.ndarray,
                   line: int) -> list[Word]:
        """Greedy CTC decode, split on spaces; each word gets the mean of its
        characters' probabilities and a quad interpolated from frame positions."""
        ids, conf = p.argmax(1), p.max(1)
        chars = []                                  # (char, prob, frame)
        for t, c in enumerate(ids):
            if c != 0 and (t == 0 or c != ids[t - 1]) and c < len(self.chars):
                chars.append((self.chars[c], float(conf[t]), t))
        out, cur = [], []
        for ch in chars + [(" ", 1.0, -1)]:
            if ch[0] == " ":
                if cur:
                    f0 = max(0.0, (cur[0][2] - 0.5) * frame_frac)
                    f1 = min(1.0, (cur[-1][2] + 1.5) * frame_frac)
                    top = lambda f: q[0] + f * (q[1] - q[0])
                    bot = lambda f: q[3] + f * (q[2] - q[3])
                    wq = np.array([top(f0), top(f1), bot(f1), bot(f0)], np.float32)
                    pr = tuple(c[1] for c in cur)
                    out.append(Word("".join(c[0] for c in cur), _word_conf(pr), wq, line, pr))
                cur = []
            else:
                cur.append(ch)
        return out

    # -- public ------------------------------------------------------------
    def prepare(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        s = self.long_side / max(h, w)
        return cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) \
            if s < 1 else bgr

    def layout(self, bgr: np.ndarray):
        """Detection only: where the text lines are, a third of read()'s time.
        Returns (frame at OCR scale, line quads); pass it on as `pre`."""
        img = self.prepare(bgr)
        return img, self._detect(img)

    def words(self, bgr: np.ndarray, pre=None):
        """Raw OCR on the frame after downscaling to long_side. Returns words
        (quads in the downscaled frame), every detected line quad (including
        lines that produced no characters) and the downscaled (h, w)."""
        img, quads = pre or self.layout(bgr)
        return self._recognize(img, quads), quads, img.shape[:2]

    def read(self, bgr: np.ndarray, legible_conf: float = LEGIBLE_CONF) -> list[TextLine]:
        return to_lines(*self.words(bgr), legible_conf)

    def read_regions(self, bgr: np.ndarray, legible_conf: float = LEGIBLE_CONF, pre=None):
        """read() plus one box per group (x0, y0, x1, y1) in full-resolution
        pixels of `bgr`; boxes[g] belongs to the lines with group == g."""
        words, quads, shape = self.words(bgr, pre)
        lines, boxes = lines_and_boxes(words, quads, shape, legible_conf)
        k = bgr.shape[0] / shape[0]
        return lines, [tuple(v * k for v in b) for b in boxes]


def to_lines(words: list[Word], quads: list[np.ndarray], shape,
             legible_conf: float = LEGIBLE_CONF) -> list[TextLine]:
    """Raw OCR -> TextLines in reading order, illegible runs with empty text."""
    return lines_and_boxes(words, quads, shape, legible_conf)[0]


def lines_and_boxes(words, quads, shape, legible_conf: float = LEGIBLE_CONF):
    H, W = shape[:2]
    per_line: dict[int, list[Word]] = {i: [] for i in range(len(quads))}
    for wd in words:
        per_line[wd.line].append(wd)
    f = (W / 2) / math.tan(math.radians(HFOV_DEG / 2))

    # one entry per detected line: geometry for grouping + its segments
    lines = []
    for i, q in enumerate(quads):
        cx, cy = q.mean(0)
        gh = float(min(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[0] - q[1])))
        # runs of legible / illegible words; an illegible run carries no text
        segs: list[tuple[str, float]] = []
        for wd in per_line[i] or [Word("", 0.0, q, i)]:
            ok = wd.conf >= legible_conf and wd.text.strip()
            if segs and (segs[-1][1] >= legible_conf) == bool(ok):
                t, c = segs[-1]
                segs[-1] = ((t + " " + wd.text) if ok else "", min(c, wd.conf))
            else:
                segs.append((wd.text if ok else "", wd.conf))
        lines.append(dict(x0=q[:, 0].min(), x1=q[:, 0].max(), y0=q[:, 1].min(),
                          y1=q[:, 1].max(), cx=cx, cy=cy, h=gh, segs=segs))

    groups = _group(lines)
    out: list[TextLine] = []
    boxes = []
    for g, members in enumerate(groups):
        boxes.append((min(m["x0"] for m in members), min(m["y0"] for m in members),
                      max(m["x1"] for m in members), max(m["y1"] for m in members)))
        for ln in members:
            bearing = math.degrees(math.atan((ln["cx"] - W / 2) / f))
            for text, conf in ln["segs"]:
                out.append(TextLine(text, round(conf, 3), round(bearing, 1),
                                    round(ln["h"] / H, 4), g))
    return out, boxes


CANVASES = (448, 560, 672, 768)     # mosaic side grows before any text is shrunk or dropped
SHRINKS = (0.8, 0.65, 0.5)          # then every region shrinks together, at the largest canvas
# Signs and labels are shown at native size: scaling scene text down to 24 px
# print costs 8 points of word recall (reader_fovea_glyph.json). A menu held
# close is different. Its 40 px letters make a sheet of 600 image tokens and a
# first word after 5 s. Only a sheet over SHEET_PX is scaled, block by block,
# to the tallest print in PRINT_STEPS that brings it under, and never below
# GLYPH_PX.
SHEET_PX = 448 * 448                # 256 image tokens: as fast as a 448 px description
PRINT_STEPS = (40, 32, 28)
GLYPH_PX = 24                       # ponytail: floor set on one menu; calibrate on a document set


def foveate(bgr: np.ndarray, regions, canvas: int = 448, max_regions: int = 8,
            pad: float = 0.15, gap: int = 4, max_canvas: int = 768, glyph_px=None,
            print_px: int | None = None):
    """Pack text regions, cut from the full-resolution frame, into one small
    mosaic so a VLM sees the glyphs at native resolution for few tokens.

    regions: boxes (x0, y0, x1, y1) or 4x2 quads in `bgr` pixels, already most
    prominent first (read_regions' group order). Each is grown by `pad` x its
    height on every side, then placed left to right in rows.

    The image budget follows the amount of text. Scattered words fit a
    `canvas`-wide sheet at native size. A menu or a page does not, so the
    sheet grows up to `max_canvas`; only if that is still too small are all
    regions shrunk together, and only then is anything dropped. (Dropping
    first loses the wrong thing: a logo and five bottle labels fill a small
    sheet and the menu between them is the region that no longer fits.)
    `glyph_px` (glyph height per region, frame pixels) turns "native size"
    into "`print_px`-tall print" (default GLYPH_PX): large print is scaled
    down first, and the crop margin follows the glyph height, not the block
    height.
    Returns (mosaic trimmed to its content, indices of the regions kept, in
    mosaic order, which is the order their hints are listed)."""
    regions = list(regions)[:max_regions]
    glyph = [g / (print_px or GLYPH_PX) for g in list(glyph_px)[:max_regions]] if glyph_px else None
    tries = [(c, 1.0) for c in CANVASES if canvas <= c <= max_canvas] or [(canvas, 1.0)]
    if glyph:
        # image tokens follow area, not shape: try every sheet width at full
        # height and keep the smallest sheet that holds every block
        fits = [(sh.shape[0] * sh.shape[1], n, sh, k) for n, (sh, k) in enumerate(
            _pack(bgr, regions, c, 1.0, pad, gap, glyph, True, tries[-1][0]) for c, _ in tries)
            if len(k) == len(regions)]
        if fits:
            return min(fits)[2:]
        tries = tries[-1:]
    tries += [(tries[-1][0], k) for k in SHRINKS]
    for side, scale in tries:
        sheet, kept = _pack(bgr, regions, side, scale, pad, gap, glyph)
        if len(kept) == len(regions):
            break
    return sheet, kept


def glyphs(lines, boxes, frame_h: int) -> list[float]:
    """Smallest glyph height in each block, frame pixels."""
    return [min((ln.height_frac for ln in lines if ln.group == g), default=0) * frame_h or GLYPH_PX
            for g in range(len(boxes))]


def mosaic(bgr: np.ndarray, lines, boxes):
    """What the reader expert is shown, and the lines that belong to the
    blocks that fit: native size when that is cheap, scaled print when not."""
    # ponytail: a block already spoken from OCR is left out, not blanked; one
    # lying inside a larger block is read again
    live = [g for g in range(len(boxes)) if any(ln.group == g for ln in lines)]
    boxes = [boxes[g] for g in live]
    glyph = [glyphs(lines, [None] * (live[-1] + 1), bgr.shape[0])[g] for g in live] if live else []
    sheet, kept = foveate(bgr, boxes)
    if sheet.shape[0] * sheet.shape[1] > SHEET_PX:
        for px in PRINT_STEPS + (GLYPH_PX,):
            sheet, kept = foveate(bgr, boxes, glyph_px=glyph, print_px=px)
            if sheet.shape[0] * sheet.shape[1] <= SHEET_PX:
                break
    return sheet, [ln for k in kept for ln in lines if ln.group == live[k]]


DOC_MIN_LINES = 12      # this many text lines is a page, a menu or a receipt, not a sign
# On the board the first word of a document comes 0.4 s (line detection) plus
# 6.3 ms per image token after the request (orin_e2e_airgap.json, documents).
# 334 tokens keeps that under the 3 s target; a denser sheet is scaled to fit.
DOC_PX = 512 * 512


def document_sheet(bgr: np.ndarray, pre, max_px: int | None = DOC_PX):
    """For a page, a menu or a receipt: the frame cropped to its text and sized
    to max_px. None when the text is a sign.

    Where the lines are is known a third of the way through OCR, and the crop
    needs nothing else. Recognising 60 lines takes the board's CPU three
    seconds and the hints they make lower recall on receipts, so a document
    is read on detection alone. A page keeps its layout: cutting it into a
    mosaic of blocks lost a third of the lines (evals/reader_docs.py)."""
    img, quads = pre
    if len(quads) < DOC_MIN_LINES:
        return None
    H, W = bgr.shape[:2]
    p = np.concatenate([np.asarray(q, float).reshape(-1, 2) for q in quads]) * (H / img.shape[0])
    m = 0.02 * max(H, W)
    x0, y0 = np.maximum(p.min(0) - m, 0).astype(int)
    x1, y1 = np.minimum(p.max(0) + m, (W, H)).astype(int)
    page = bgr[y0:y1, x0:x1]
    s = (max_px / (page.shape[0] * page.shape[1])) ** 0.5 if max_px else 1
    return cv2.resize(page, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else page


def _pack(bgr, regions, canvas, scale, pad, gap, glyph=None, strict=False, height=None):
    height = height or canvas
    H, W = bgr.shape[:2]
    boxes = []
    for i, r in enumerate(regions):
        r = np.asarray(r, np.float32).reshape(-1, 2)
        x0, y0 = r.min(0)
        x1, y1 = r.max(0)
        p = pad * (glyph[i] * GLYPH_PX if glyph else y1 - y0)   # ~glyph height; exact px not needed
        boxes.append((int(max(0, x0 - p)), int(max(0, y0 - p)),
                      int(min(W, x1 + p + 1)), int(min(H, y1 + p + 1))))
    area = [(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes]
    sheet = np.full((height, canvas, 3), 128, np.uint8)
    x = y = row_h = used_w = 0
    kept = []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        # show every region once: a smaller region lying inside this one has
        # its own crop, so blank it here or its words are read twice
        crop = crop.copy()
        for j, (a0, b0, a1, b1) in enumerate(boxes):
            if j != i and area[j] < area[i]:
                crop[max(b0, y0) - y0:max(min(b1, y1) - y0, 0),
                     max(a0, x0) - x0:max(min(a1, x1) - x0, 0)] = 128
        want = scale * (min(1.0, 1 / glyph[i]) if glyph else 1.0)    # glyph[i]: height / wanted height
        s = min(want, canvas / crop.shape[1], height / crop.shape[0])
        if strict and s < want:
            continue                                    # print would get too small: wants a larger sheet
        if s < 1:
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                              interpolation=cv2.INTER_AREA)
        ch, cw = crop.shape[:2]
        if x and x + cw > canvas:                       # new row
            x, y, row_h = 0, y + row_h + gap, 0
        if y + ch > height:
            continue                                    # too tall for what is left
        sheet[y:y + ch, x:x + cw] = crop
        kept.append(i)
        used_w = max(used_w, x + cw)
        row_h = max(row_h, ch)
        x += cw + gap
    used_h = y + row_h
    return sheet[:max(used_h, 28), :max(used_w, 28)].copy(), kept


def _group(lines: list[dict]) -> list[list[dict]]:
    """Lines -> blocks. Two lines share a block when their glyph heights are
    within 1.6x, they overlap horizontally (or sit on one row), and the
    vertical gap is under one line height. Within a block: top to bottom, then
    left to right on a row. Blocks: tallest glyphs first, then top-left."""
    n = len(lines)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = lines[i], lines[j]
            hm = max(a["h"], b["h"])
            if hm > 1.6 * min(a["h"], b["h"]):
                continue
            xover = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
            vgap = max(a["y0"], b["y0"]) - min(a["y1"], b["y1"])
            same_row = abs(a["cy"] - b["cy"]) < 0.5 * hm and xover > -1.5 * hm
            stacked = xover > 0 and vgap < 1.0 * hm
            if same_row or stacked:
                parent[find(i)] = find(j)
    blocks: dict[int, list[dict]] = {}
    for i in range(n):
        blocks.setdefault(find(i), []).append(lines[i])
    out = []
    for members in blocks.values():
        hmed = float(np.median([m["h"] for m in members]))
        # row index = centre y quantised by half a line height, then x
        members.sort(key=lambda m: (round(m["cy"] / (0.5 * hmed)), m["x0"]))
        out.append(members)
    out.sort(key=lambda ms: (-max(m["h"] for m in ms), min(m["y0"] for m in ms)))
    return out


def speak_short(lines: list[TextLine]) -> str | None:
    """Deterministic speech for short text (signs, labels). Legible words are
    read verbatim; illegible stretches are named where they occur. Returns
    None when the text needs the reader expert: any line under SPEAK_CONF
    (including unreadable ones), too many words, or too many sources."""
    said, rest = speak_sure(lines)
    return None if rest else said


def speak_sure(lines: list[TextLine]) -> tuple[str | None, list[TextLine]]:
    """Speak now what OCR is sure of; return the lines still to be read.

    A street frame is never only the sign: there is a licence plate or a
    shop front in the corner that OCR cannot read, and waiting for the
    language model because of it makes "EXIT" a 1.5 s answer. A block whose
    every line is at SPEAK_CONF is spoken at once (up to SHORT_MAX_GROUPS
    blocks and SHORT_MAX_WORDS words); the reader expert gets the rest."""
    if not lines:
        return "I can't see any text.", []
    groups: dict[int, list[TextLine]] = {}
    for ln in lines:
        groups.setdefault(ln.group, []).append(ln)
    sure, words = [], 0
    for g in sorted(groups):
        n = sum(len(ln.text.split()) for ln in groups[g])
        if all(ln.conf >= SPEAK_CONF and ln.text.strip() for ln in groups[g]) \
                and len(sure) < SHORT_MAX_GROUPS and words + n <= SHORT_MAX_WORDS:
            sure.append(g)
            words += n
    if not sure:
        return None, lines
    said = compose([[(ln.text, True) for ln in groups[g]] for g in sure],
                   [float(np.mean([ln.bearing_deg for ln in groups[g]])) for g in sure])
    return said, [ln for ln in lines if ln.group not in sure]


CANT_READ = "something I can't read"


def compose(sources: list[list[tuple[str, bool]]], bearings: list[float],
            quote: bool = False) -> str:
    """The one spoken format. sources: per source, its parts in order as
    (text, legible); bearings: one per source. Several sources are introduced
    by where they are. quote=True puts read text in double quotes (the reader
    expert's training targets; see train/reader/build_sft.verify_target)."""
    def say(parts: list[tuple[str, bool]]) -> str:
        out: list[str] = []
        for t, ok in parts:
            p = (f'"{t.strip()}"' if quote else t.strip()) if ok else CANT_READ
            if not (out and out[-1] == p == CANT_READ):
                out.append(p)
        s = ", then ".join(out) if not all(ok for _, ok in parts) else ". ".join(out)
        s = s[0].upper() + s[1:]
        return s if re.search(r"[.!?]\"?$", s) else s + "."

    if len(sources) == 1:
        return say(sources[0])
    out, prev_side, prev_blank = [], None, False
    for parts, b in zip(sources, bearings):
        blank = not any(ok for _, ok in parts)
        if blank and prev_blank:
            continue                          # one "can't read" covers a run of them
        side = side_from_bearing(b)
        where = "Next" if side == prev_side else (
            "Ahead" if side == "ahead" else f"On your {side}")
        out.append(f"{where}: {say(parts)}")
        prev_side, prev_blank = side, blank
    return " ".join(out)


def demo() -> None:
    import time
    mosaic_fn = globals()["mosaic"]     # `mosaic` is a local sheet below

    from veris.packet import Packet

    # synthetic sign: big two-line heading on the left, small label on the right
    img = np.full((720, 1280, 3), 235, np.uint8)
    cv2.putText(img, "EMERGENCY", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (20, 20, 20), 8)
    cv2.putText(img, "EXIT ONLY", (60, 320), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (20, 20, 20), 8)
    cv2.putText(img, "Room 214", (900, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
    r = Reader()
    lines = r.read(img)
    r.read(img)                                       # warm-up done; time the next ones
    t0 = time.perf_counter()
    for _ in range(5):
        r.read(img)
    ms = (time.perf_counter() - t0) / 5 * 1000
    print(lines)
    texts = [ln.text for ln in lines]
    assert texts[:2] == ["EMERGENCY", "EXIT ONLY"], texts      # prominent block, top-down
    assert lines[0].group == lines[1].group == 0
    assert "Room 214" in texts and lines[texts.index("Room 214")].group == 1
    assert lines[0].bearing_deg < 0 < lines[texts.index("Room 214")].bearing_deg
    assert lines[0].height_frac > lines[texts.index("Room 214")].height_frac
    assert all(ln.legible for ln in lines)
    said = speak_short(lines)
    assert said == "On your left: EMERGENCY. EXIT ONLY. On your right: Room 214.", said

    # gating: a forced-high threshold turns everything into empty illegible lines
    gated = r.read(img, legible_conf=1.01)
    assert gated and all(ln.text == "" and not ln.legible for ln in gated)
    assert speak_short(gated) is None                                # -> expert
    assert Packet(text=tuple(gated)).to_prompt("reader").count("[unreadable]") == len(gated)

    # composer rules on hand-made lines
    L = lambda t, c=0.9995, g=0, b=0.0: TextLine(t, c, b, 0.05, g)
    assert speak_short([L("PUSH")]) == "PUSH."
    assert speak_short([L("PUSH"), L("", 0.2)]) is None              # unreadable part
    assert speak_short([L("PUSH", 0.995)]) is None                   # legible, not sure enough
    assert speak_short([L("Total: $4.50")]) == "Total: $4.50."
    assert speak_short([L("w " * 13)]) is None                       # long -> expert
    assert speak_short([L("A", g=0), L("B", g=1), L("C", g=2)]) is None
    assert speak_short([]) == "I can't see any text."
    # the sure block is spoken at once; the unreadable one is left for the expert
    said2, rest = speak_sure([L("EXIT", g=0), L("", 0.2, g=1)])
    assert said2 == "EXIT." and [ln.group for ln in rest] == [1]
    assert speak_sure([L("EXIT", 0.99)]) == (None, [L("EXIT", 0.99)])
    assert compose([[("PUSH", True), ("", False), ("", False)]], [0.0], quote=True) == \
        '"PUSH", then something I can\'t read.'
    assert compose([[("", False), ("OPEN", True)], [("Exit", True)]], [-30.0, 2.0]) == \
        "On your left: Something I can't read, then OPEN. Ahead: Exit."
    assert compose([[("A", True)], [("", False)], [("", False)], [("B", True)]],
                   [-30.0, -30.0, 30.0, 30.0]) == \
        "On your left: A. Next: Something I can't read. On your right: B."

    # foveation: regions keep their order, glyphs keep their pixels
    lines, boxes = r.read_regions(img)
    mosaic, kept = foveate(img, boxes)
    assert kept == [0, 1] and len(boxes) == 2, (kept, boxes)
    assert mosaic.shape[1] <= 448 and mosaic.shape[0] <= 448
    big = [(0, 0, 900, 100)] * 3 + [(0, 0, 50, 20)] * 20
    m2, k2 = foveate(img, big)
    assert k2[:3] == [0, 1, 2] and len(k2) == 8                      # max_regions
    assert m2.shape[1] == 448                                        # wide crop shrunk
    m3, k3 = foveate(img, [(1000, 580, 1100, 610)])                  # small crop: native size
    assert k3 == [0] and m3.shape[:2] == (40, 110), m3.shape
    tm = time.perf_counter()
    for _ in range(20):
        foveate(img, boxes)
    # a sign is the same native-size sheet as before; with glyph heights,
    # close-up print is scaled to GLYPH_PX and small print is never squeezed
    # into the small sheet while a larger one is available
    sh, shown = mosaic_fn(img, lines, boxes)
    assert [ln.text for ln in shown] == [ln.text for ln in lines] and sh.shape == mosaic.shape
    page = np.full((1400, 1280, 3), 235, np.uint8)
    big, _ = foveate(page, [(100, 100, 1100, 800)], glyph_px=[48])       # 14 lines of 48 px print
    assert big.shape[1] <= 1000 * GLYPH_PX / 48 + 20, big.shape
    small, _ = foveate(page, [(100, 100, 800, 700)], glyph_px=[20])      # 20 px print, 700 px wide
    assert small.shape[1] >= 700, small.shape
    print(f"foveate: {mosaic.shape[1]}x{mosaic.shape[0]} mosaic, kept {kept}, "
          f"{(time.perf_counter() - tm) / 20 * 1000:.1f} ms")
    print(said)
    print(f"read() CPU latency at 1280x720: {ms:.0f} ms "
          f"({os.cpu_count()} cores, onnxruntime default threads)")


if __name__ == "__main__":
    demo()
