"""OCR-guided foveation vs today's full-frame reader, on ICDAR15 TEST.

    .venv/bin/python evals/reader_fovea.py prep                     # CPU: OCR, mosaics
    flock /tmp/veris-gpu.lock .venv/bin/python evals/reader_fovea.py gpu --minutes 38
    .venv/bin/python evals/reader_fovea.py score        # -> evals/results/reader_fovea.json

Paths compared (base Qwen2.5-VL-3B, 4-bit, veris.model.VLM):
  a  full frame, 1280 px long side, prompts.MODES["text_reading"]   (today's reader)
  b  OCR alone: legible words at conf >= 0.99 and >= 0.999
  c  mosaic of OCR-detected regions (reader.foveate), EXPERTS["reader"] prompt,
     OCR words as hints: user_text(ex, Packet(text=lines).to_prompt("reader"))
  d  as c without hints
Conditions: clean on every image; wearcam outdoor_walk on every other image
(paths a, c only), to fit the GPU budget. Images run in a fixed seeded order
until the time budget ends; only images with every path done are scored.

Scoring. VLM text has no geometry, so this is reader_ocr's normalisation
(lower case, edge punctuation stripped, apostrophes/accents folded) applied
as a per-image bag of words. Read words: the words inside double quotes when
the output has any, else every word minus a fixed narration stoplist
(NARRATION). GT words that are in NARRATION are dropped for every path so all
paths share one GT. A read word is correct when it matches an unmatched GT
word exactly; remaining words are paired greedily by smallest edit distance
(at most half the GT length) for CER. Invented = read word absent from the
image's GT words. Don't-care "###" regions are excluded from GT; words read
from them count as invented for every path alike.

Image tokens: image_grid_thw.prod() / 4 from the Qwen processor, which is what
the board's prefill scales with. The VLM processor's default floor
(veris.model.MIN_PIXELS = 256 tokens) would upscale small mosaics, so the
processor is rebuilt here with a 4-token floor.
"""

from __future__ import annotations

import argparse
import os
import json
import random
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))
import reader_ocr as E  # noqa: E402
from veris import reader, wearcam  # noqa: E402
from veris.experts import EXPERTS, user_text  # noqa: E402
from veris.packet import Packet, TextLine  # noqa: E402

# FOVEA_TAG=_glyph scores the trained reader on the deployed sheet
# (reader.mosaic: blocks scaled to GLYPH_PX print) -> reader_fovea_glyph.json
TAG = os.environ.get("FOVEA_TAG", "")
NATIVE = ROOT / "data/reader/fovea"
D = ROOT / f"data/reader/fovea{TAG}"
RAW_OUT = D / "gen.jsonl"
OUT = ROOT / f"evals/results/reader_fovea{TAG}.json"
if TAG:                         # GLYPH_PX=20 FOVEA_TAG=_glyph20: try another print height
    reader.GLYPH_PX = int(os.environ.get("GLYPH_PX", reader.GLYPH_PX))
SEED = 0
MAX_NEW = 200
NARRATION = set("""the a an sign signs says say said reads read reading text board label
labels poster banner it its this that there here is are and of on in to at with
rest word words line lines can't cannot can not unreadable blurry too small i see
which what front left right ahead top bottom above below then something written
some other another first second third next part partly hidden blocked cut off
also one two three letters letter numbers number show shows showing appears
readable visible your tell possibly""".split())


def _prep(task):
    i, path, cond = task
    if not hasattr(_prep, "r"):
        _prep.r = reader.Reader(threads=1)
    img = cv2.imread(path)
    if cond != "clean":
        img = wearcam.degrade(img, cond, seed=i)
    lines, boxes = _prep.r.read_regions(img)
    words, _, _ = _prep.r.words(img)
    sid = f"{Path(path).stem}_{cond}"
    frame = path
    if cond != "clean":
        frame = str(D / "frames" / f"{sid}.png")
        cv2.imwrite(frame, img)
    mosaic, kept = reader.foveate(img, boxes) if boxes else (None, [])
    if TAG and boxes:
        # every block scaled to GLYPH_PX print, whatever the sheet costs
        mosaic, kept = reader.foveate(img, boxes, glyph_px=reader.glyphs(lines, boxes, img.shape[0]))
    mpath = None
    if mosaic is not None:
        mpath = str(D / "mosaic" / f"{sid}.png")
        cv2.imwrite(mpath, mosaic)
    hint = [ln for g in kept for ln in lines if ln.group == g]
    return dict(id=sid, idx=i, cond=cond, frame=frame, mosaic=mpath, kept=kept,
                n_regions=len(boxes),
                hint_prompt=Packet(text=tuple(hint)).to_prompt("reader"),
                ocr_words=[(w.text, w.conf) for w in words])


def prep():
    (D / "frames").mkdir(parents=True, exist_ok=True)
    (D / "mosaic").mkdir(parents=True, exist_ok=True)
    items = E.load_icdar("test")
    tasks = [(i, p, c) for i, (p, _) in enumerate(items) for c in ("clean", "outdoor_walk")]
    with Pool(12) as pool:
        rows = pool.map(_prep, tasks, chunksize=4)
    json.dump(rows, open(D / "manifest.json", "w"))
    print(len(rows), "prepared")


def _jsonl(path):
    """Rows of a jsonl file; a line cut short by a killed job is skipped (and redone)."""
    out = []
    for line in open(path):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def order() -> list[int]:
    idx = list(range(500))
    random.Random(SEED).shuffle(idx)
    return idx


def gpu(minutes: float, adapter: str | None = None, hints: bool = True):
    from transformers import AutoProcessor

    from veris.model import MAX_PIXELS, VLM
    from veris.prompts import MODES, build_messages

    t_start = time.time()
    man = {(r["idx"], r["cond"]): r for r in json.load(open(D / "manifest.json"))}
    done = set()
    if RAW_OUT.exists():
        done = {(j["id"], j["path"]) for j in _jsonl(RAW_OUT)}
    ran = done if not TAG else {(j["id"], j["path"]) for j in _jsonl(NATIVE / "gen.jsonl")}
    vlm = VLM(model_id="Qwen/Qwen2.5-VL-3B-Instruct", quantize=True)
    vlm.processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=4 * 28 * 28, max_pixels=MAX_PIXELS)
    if adapter:                 # path e: the trained reader expert, on what it was trained on
        from peft import PeftModel
        vlm.model = PeftModel.from_pretrained(vlm.model, adapter).eval()
    from qwen_vl_utils import process_vision_info
    mode, ex = MODES["text_reading"], EXPERTS["reader"]
    print(f"model loaded in {time.time() - t_start:.0f} s", flush=True)

    def call(msgs, max_new):
        ii, _ = process_vision_info(msgs)
        tmpl = vlm.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        g = vlm.processor(text=[tmpl], images=ii, return_tensors="pt")["image_grid_thw"]
        t0 = time.perf_counter()
        text = "".join(vlm.generate_stream(msgs, max_new)).strip()
        return text, int(g.prod(-1).sum()) // 4, time.perf_counter() - t0, vlm.last_stats.ttft_s

    def mosaic_msgs(r, hints: bool):
        ut = user_text(ex, r["hint_prompt"]) if hints else user_text(ex)
        return [{"role": "system", "content": ex.system},
                {"role": "user", "content": [{"type": "image", "image": r["mosaic"]},
                                             {"type": "text", "text": ut}]}]

    n = 0
    with open(RAW_OUT, "a") as f:
        for k, i in enumerate(order()):
            jobs = [("clean", "a"), ("clean", "c"), ("clean", "d")]
            if k % 2 == 0:
                jobs += [("outdoor_walk", "a"), ("outdoor_walk", "c")]
            if adapter:             # only where the untrained paths already ran
                jobs = [(c, "e" if hints else "f") for c, q in jobs
                        if q == "c" and (man[(i, c)]["id"], "c") in ran]
            if time.time() - t_start > minutes * 60:
                break
            for cond, path in jobs:
                r = man[(i, cond)]
                if (r["id"], path) in done:
                    continue
                if path == "a":
                    out = call(build_messages(mode, r["frame"], None, None),
                               min(MAX_NEW, mode.max_new_tokens))
                elif r["mosaic"] is None:
                    out = ("", 0, 0.0, 0.0)          # nothing detected: no VLM call
                else:
                    out = call(mosaic_msgs(r, path in "ce"), min(MAX_NEW, ex.max_tokens))
                text, toks, dt, ttft = out
                f.write(json.dumps(dict(id=r["id"], idx=i, cond=cond, path=path, text=text,
                                        image_tokens=toks, gen_s=round(dt, 2),
                                        ttft_s=round(ttft, 3))) + "\n")
                f.flush()
            n += 1
            if n % 10 == 0:
                print(f"{n} images, {time.time() - t_start:.0f} s", flush=True)
    print(f"stopped after {n} images in {time.time() - t_start:.0f} s", flush=True)


# -- scoring ------------------------------------------------------------------
QUOTED = re.compile(r'["“”]([^"“”]*)["“”]')


def read_words(text: str) -> list[str]:
    q = QUOTED.findall(text)
    if q:
        ws = [E.norm(w) for s in q for w in s.split()]
    else:
        ws = [E.norm(w) for w in text.split()]
        ws = [w for w in ws if w not in NARRATION]
    return [w for w in ws if w]


def score_bag(pairs):
    """pairs: [(read words, gt words)] -> metrics."""
    n_read = n_gt = correct = invented = 0
    edits = chars = 0
    for rw, gt in pairs:
        gt = [g for g in gt if g not in NARRATION]
        n_read += len(rw)
        n_gt += len(gt)
        bag = set(gt)
        invented += sum(w not in bag for w in rw)
        left_g, left_r = list(gt), []
        for w in rw:
            if w in left_g:
                left_g.remove(w)
                correct += 1
                chars += len(w)
            else:
                left_r.append(w)
        cand = sorted((E.edit(w, g), j, k) for j, w in enumerate(left_r)
                      for k, g in enumerate(left_g) if E.edit(w, g) <= len(g) / 2)
        ur, ug = set(), set()
        for d, j, k in cand:
            if j not in ur and k not in ug:
                ur.add(j)
                ug.add(k)
                edits += d
                chars += len(left_g[k])
    return dict(n_images=len(pairs), gt_words=n_gt, read_words=n_read,
                recall=round(correct / max(1, n_gt), 4),
                precision=round(correct / max(1, n_read), 4),
                invented_rate=round(invented / max(1, n_read), 4),
                cer_matched=round(edits / max(1, chars), 4))


def score():
    items = E.load_icdar("test")
    gt = {i: [E.norm(t) for _, t, dc in g if not dc] for i, (_, g) in enumerate(items)}
    man = {(r["idx"], r["cond"]): r for r in json.load(open(D / "manifest.json"))}
    gen = {}
    for j in _jsonl(RAW_OUT):
        gen[(j["idx"], j["cond"], j["path"])] = j
    res = {}
    for cond, paths in (("clean", "acd"), ("outdoor_walk", "ac")):
        paths = "ef" if TAG else paths
        imgs = sorted({i for (i, c, _) in gen if c == cond
                       and all((i, cond, p) in gen for p in paths)})
        out = {"n_images": len(imgs), "image_idx": imgs}
        for p in dict.fromkeys(paths + "ef"):
            if p in "ef" and any((i, cond, p) not in gen for i in imgs):
                continue            # trained-expert pass not (fully) run
            js = [gen[(i, cond, p)] for i in imgs]
            toks = [j["image_tokens"] for j in js if j["image_tokens"]]
            name = {"a": "a_full_frame_1280", "c": "c_mosaic_hints",
                    "d": "d_mosaic_no_hints",
                    "e": "e_mosaic_hints_trained_reader",
                    "f": "f_mosaic_no_hints_trained_reader"}[p]
            out[name] = score_bag([(read_words(j["text"]), gt[j["idx"]]) for j in js]) | dict(
                image_tokens_mean=round(float(np.mean(toks)), 1) if toks else 0,
                image_tokens_max=max(toks, default=0),
                images_without_vlm_call=sum(1 for j in js if not j["image_tokens"]),
                gen_s_mean_3070=round(float(np.mean([j["gen_s"] for j in js])), 2),
                ttft_s_mean_3070=round(float(np.mean([j["ttft_s"] for j in js])), 3))
        if all((i, cond, p) in gen for i in imgs for p in "ef"):
            # the deployed rule: hints only when OCR could read enough of the lines
            def legible_share(i):
                h = man[(i, cond)]["hint_prompt"]
                n_ok, n_bad = h.count('"') // 2, h.count("[unreadable]")
                return n_ok / max(n_ok + n_bad, 1)
            pick = [gen[(i, cond, "e" if legible_share(i) >= reader.HINT_MIN_LEGIBLE else "f")] for i in imgs]
            out["g_trained_reader_hints_if_mostly_legible"] = score_bag(
                [(read_words(j["text"]), gt[j["idx"]]) for j in pick]) | dict(
                hinted_share=round(float(np.mean([j["path"] == "e" for j in pick])), 3),
                min_legible=reader.HINT_MIN_LEGIBLE)
            sweep = {}
            for t in (0, 0.1, 0.2, 0.3, 0.5, 1.01):     # 0 = always hint, 1.01 = never
                pk = [gen[(i, cond, "e" if legible_share(i) >= t else "f")] for i in imgs]
                s = score_bag([(read_words(j["text"]), gt[j["idx"]]) for j in pk])
                sweep[str(t)] = {k: s[k] for k in ("recall", "precision", "invented_rate")}
            out["g_hint_threshold_sweep"] = sweep
        for t in (0.99, 0.999):
            out[f"b_ocr_conf_{t}"] = score_bag(
                [([E.norm(w) for w, c in man[(i, cond)]["ocr_words"] if c >= t and E.norm(w)],
                  gt[i]) for i in imgs])
        for t in (0.99, 0.999):                     # path b needs no GPU: all 500 images
            out[f"b_ocr_conf_{t}_all_500"] = score_bag(
                [([E.norm(w) for w, c in man[(i, cond)]["ocr_words"] if c >= t and E.norm(w)],
                  gt[i]) for i in range(len(items))])
        regs = [man[(i, cond)] for i in imgs]
        out["regions_detected_mean"] = round(float(np.mean([r["n_regions"] for r in regs])), 2)
        out["regions_kept_mean"] = round(float(np.mean([len(r["kept"]) for r in regs])), 2)
        res[cond] = out
    shown = "ef" if TAG else "acd"
    examples = [dict(id=gen[(i, "clean", shown[0])]["id"], gt=gt[i],
                     **{p: gen[(i, "clean", p)]["text"] for p in shown},
                     hints=man[(i, "clean")]["hint_prompt"])
                for i in res["clean"]["image_idx"][:8]]
    OUT.write_text(json.dumps(dict(
        command=(f"FOVEA_TAG={TAG} GLYPH_PX={reader.GLYPH_PX} evals/reader_fovea.py prep; ... gpu --adapter "
                 "train/experts/out-reader [--no-hints]; ... score") if TAG else
                "evals/reader_fovea.py prep; flock /tmp/veris-gpu.lock "
                "evals/reader_fovea.py gpu --minutes 38 (killed after ~4 min by the "
                "coordinator, resumed with --minutes 34); evals/reader_fovea.py score",
        split="ICDAR2015 ch4 TEST, fixed seeded order (seed 0), prefix that fit the GPU budget",
        model="Qwen/Qwen2.5-VL-3B-Instruct, bnb 4-bit, RTX 3070 (timings are not Orin)",
        max_new_tokens=MAX_NEW, narration_stoplist=sorted(NARRATION),
        method=__doc__.split("Scoring.")[1].strip(),
        results=res, examples=examples), indent=1))
    print(json.dumps({c: {k: v for k, v in r.items() if k != "image_idx"}
                      for c, r in res.items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "gpu", "score"])
    ap.add_argument("--minutes", type=float, default=38)
    ap.add_argument("--no-hints", action="store_true", help="with --adapter: path f")
    ap.add_argument("--adapter", help="LoRA dir (train/experts/out-reader): run path e only")
    a = ap.parse_args()
    {"prep": prep, "gpu": lambda: gpu(a.minutes, a.adapter, not a.no_hints), "score": score}[a.cmd]()
