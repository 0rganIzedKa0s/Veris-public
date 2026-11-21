"""Documents: the deployed reading path on camera photos of receipts.

WildReceipt test split (phone photos of receipts in the wild), the images at
least 1000 px on the long side, in a fixed seeded order. Each is downscaled to
the 1280 px wearable frame and read six ways:

  a  the untrained base model on the full frame (text_reading prompt)
  r  the deployed path, as veris/conductor.py runs it: OCR line detection ->
     reader.document_sheet (the frame cropped to its text, capped at
     reader.DOC_PX) -> the base model with the reader's prompt. Under
     DOC_MIN_LINES lines it is a sign: speak_sure + reader.mosaic with hints
     -> trained reader adapter
  c  the same crop, read by the trained reader adapter
  m  the first document path: the page cut into text blocks and packed
     (reader.mosaic), capped at DOC_PX, read by the reader adapter
  u  as m, not capped
  h  as u, with every line recognised and passed as hints (the
     HINT_MIN_LEGIBLE rule)
Outputs are kept per path in data/reader/docs/raw_<path>.json; --paths reruns
only the ones named.

Scoring. WildReceipt transcribes each line without spaces, so both sides are
reduced to lower-case letters and digits. A ground-truth line of 3 or more
characters counts as read when it appears in the output. Precision is the
share of output characters that belong to a line read, so a line read with
one wrong character counts fully against it: it is a lower bound.

    flock /tmp/veris-gpu.lock .venv/bin/python evals/reader_docs.py [--n 100]
-> evals/results/reader_docs.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

from veris import reader  # noqa: E402
from veris.experts import EXPERTS, user_text  # noqa: E402
from veris.packet import HINT_MIN_LEGIBLE, Packet  # noqa: E402

WR = ROOT / "data/reader/wildreceipt/wildreceipt"
D = ROOT / "data/reader/docs"
OUT = ROOT / "evals/results/reader_docs.json"
FRAME = 1280
MAX_NEW = EXPERTS["reader"].max_tokens
PATHS = "arcmuh"


def key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def wildreceipt(n: int, seed: int = 0) -> list[tuple[np.ndarray, list[str]]]:
    """[(frame at 1280 px, ground-truth lines)] for the first n test photos."""
    rows = [json.loads(ln) for ln in open(WR / "test.txt")]
    rows = [r for r in rows if max(r["height"], r["width"]) >= 1000]
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:n]:
        img = cv2.imread(str(WR / r["file_name"]))
        s = FRAME / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        out.append((img, [a["text"] for a in r["annotations"]]))
    return out


def mosaic_sheet(img: np.ndarray, pre, max_px: int | None) -> np.ndarray:
    im, quads = pre
    lines, boxes = reader.lines_and_boxes([], quads, im.shape[:2])
    k = img.shape[0] / im.shape[0]
    sheet = reader.mosaic(img, lines, [tuple(v * k for v in b) for b in boxes])[0]
    s = (max_px / (sheet.shape[0] * sheet.shape[1])) ** 0.5 if max_px else 1
    return cv2.resize(sheet, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else sheet


def view(R: reader.Reader, img: np.ndarray, p: str):
    """-> (spoken from OCR, image for the model or None, hint lines, expert)."""
    pre = R.layout(img)
    said, sheet, lines, name = "", None, [], "reader"
    if len(pre[1]) >= reader.DOC_MIN_LINES and p in "rc":
        return said, reader.document_sheet(img, pre), lines, "document" if p == "r" else name
    if len(pre[1]) >= reader.DOC_MIN_LINES and p in "mu":
        return said, mosaic_sheet(img, pre, reader.DOC_PX if p == "m" else None), lines, name
    lines, boxes = R.read_regions(img, pre=pre)
    if p != "h" or len(pre[1]) < reader.DOC_MIN_LINES:      # a sign, on every path
        said, lines = reader.speak_sure(lines)
    if lines and boxes:
        sheet, lines = reader.mosaic(img, lines, boxes)
    if sum(ln.legible for ln in lines) < HINT_MIN_LEGIBLE * len(lines):
        lines = []
    return said or "", sheet, lines, name


def score(rows: list[dict]) -> dict:
    hit = tot = hit_c = tot_c = out_c = 0
    for r in rows:
        text = key(" ".join(re.findall(r'["“”]([^"“”]*)["“”]', r["text"])) or r["text"])
        gt = {g for g in map(key, r["gt"]) if len(g) >= 3}
        got = {g for g in gt if g in text}
        hit += len(got)
        tot += len(gt)
        hit_c += sum(map(len, got))
        tot_c += sum(map(len, gt))
        out_c += max(len(text), sum(map(len, got)))
    toks = [r["image_tokens"] for r in rows if r["image_tokens"]]
    return {"n_images": len(rows), "gt_lines": tot, "line_recall": round(hit / max(tot, 1), 4),
            "char_recall": round(hit_c / max(tot_c, 1), 4),
            "char_precision": round(hit_c / max(out_c, 1), 4),
            "image_tokens_mean": round(float(np.mean(toks)), 1), "image_tokens_max": max(toks),
            "ttft_s_median_3070": round(float(np.median([r["ttft_s"] for r in rows])), 3),
            "gen_s_mean_3070": round(float(np.mean([r["gen_s"] for r in rows])), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--adapter", default="train/experts/out-reader")
    ap.add_argument("--paths", default=PATHS)
    a = ap.parse_args()
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    from veris.model import MAX_PIXELS, VLM
    from veris.prompts import MODES, build_messages

    D.mkdir(parents=True, exist_ok=True)
    items = wildreceipt(a.n)
    vlm = VLM(model_id="Qwen/Qwen2.5-VL-3B-Instruct", quantize=True)
    vlm.processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=4 * 28 * 28, max_pixels=MAX_PIXELS)
    mode, ex = MODES["text_reading"], EXPERTS["reader"]

    def call(msgs):
        ii, _ = process_vision_info(msgs)
        tmpl = vlm.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        g = vlm.processor(text=[tmpl], images=ii, return_tensors="pt")["image_grid_thw"]
        t0 = time.perf_counter()
        text = "".join(vlm.generate_stream(msgs, MAX_NEW)).strip()
        return dict(text=text, image_tokens=int(g.prod(-1).sum()) // 4,
                    gen_s=round(time.perf_counter() - t0, 2), ttft_s=round(vlm.last_stats.ttft_s, 3))

    res: dict[str, list[dict]] = {p: [] if p in a.paths else json.load(open(D / f"raw_{p}.json"))
                                  for p in PATHS}
    for i, (img, gt) in enumerate(items if "a" in a.paths else []):   # a: base model, full frame
        cv2.imwrite(str(D / f"{i}.jpg"), img)
        res["a"].append(call(build_messages(mode, str(D / f"{i}.jpg"), None, None)) | {"gt": gt})
        print("a", i, flush=True) if i % 10 == 0 else None
    R = reader.Reader()

    def read(i, p, img, gt):
        said, sheet, lines, name = view(R, img, p)
        out = dict(text="", image_tokens=0, gen_s=0.0, ttft_s=0.0)
        if sheet is not None:
            cv2.imwrite(str(D / f"{i}_{p}.png"), sheet)
            out = call([{"role": "system", "content": ex.system},
                        {"role": "user", "content": [
                            {"type": "image", "image": str(D / f"{i}_{p}.png")},
                            {"type": "text", "text": user_text(
                                ex, Packet(text=tuple(lines)).to_prompt("reader"))}]}])
        out["text"] = f'"{said}" {out["text"]}' if said else out["text"]
        print(p, i, flush=True) if i % 10 == 0 else None
        return out | {"gt": gt, "expert": name, "hinted": bool(lines)}

    if "r" in a.paths:      # documents go to the base model; signs wait for the adapter
        res["r"] = [read(i, "r", img, gt) if len(R.layout(img)[1]) >= reader.DOC_MIN_LINES
                    else None for i, (img, gt) in enumerate(items)]
    vlm.model = PeftModel.from_pretrained(vlm.model, a.adapter).eval()
    res["r"] = [r or read(i, "r", *items[i]) for i, r in enumerate(res["r"])]
    for p in (p for p in "cmuh" if p in a.paths):
        res[p] = [read(i, p, img, gt) for i, (img, gt) in enumerate(items)]
    for p, rows in res.items():
        json.dump(rows, open(D / f"raw_{p}.json", "w"))
    OUT.write_text(json.dumps({
        "command": "flock /tmp/veris-gpu.lock .venv/bin/python evals/reader_docs.py",
        "dataset": f"WildReceipt test, long side >= 1000 px, seed 0, first {len(items)}, "
                   f"downscaled to {FRAME} px",
        "model": "Qwen/Qwen2.5-VL-3B-Instruct, bnb 4-bit, RTX 3070 (timings are not the board)",
        "max_new_tokens": MAX_NEW, "scoring": __doc__.split("Scoring.")[1].split("flock")[0].strip(),
        "a_base_full_frame_1280": score(res["a"]),
        "r_deployed_crop_base_model": score(res["r"]) | {
            "took_document_path": sum(r["expert"] == "document" for r in res["r"])},
        "c_crop_reader_adapter": score(res["c"]),
        "m_mosaic_capped_reader_adapter": score(res["m"]),
        "u_mosaic_not_capped": score(res["u"]),
        "h_mosaic_not_capped_with_hints": score(res["h"]) | {
            "hinted": sum(r["hinted"] for r in res["h"])},
        "examples": [{"gt": r["gt"][:12], "a": a_["text"][:400], "r": r["text"][:400]}
                     for a_, r in list(zip(res["a"], res["r"]))[:3]],
    }, indent=1))
    print(json.dumps({k: v for k, v in json.loads(OUT.read_text()).items() if k[1] == "_"}, indent=1))


if __name__ == "__main__":
    assert key("2621 Orangethorpe Ave, Fullerton.") == key("2621ORANGETHORPEAVE,FULLERTON.")
    assert score([dict(text='"Total 4.50" "zzz"', gt=["TOTAL4.50", "CASH"], image_tokens=10,
                       ttft_s=0.1, gen_s=1.0)])["line_recall"] == 0.5
    main()
