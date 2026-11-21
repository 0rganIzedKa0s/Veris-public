"""Score the outdoor and indoor experts on held-out COCO val2017.

Every number is a count against ground-truth boxes (train/experts/verify.py);
no model judges another model.

    chair_s    share of descriptions naming at least one object that is not there
    chair_i    share of object mentions that are not there
    coverage   share of an image's annotated categories that get mentioned
    side_acc   share of left/right (or clock) statements the boxes agree with
    fake_echo  share of descriptions repeating an object planted in the packet
               that is not in the image (only for the `fake` packet)

Configs compare the old single model against the experts, with and without
the packet, across image budgets:

    flock /tmp/veris-gpu.lock .venv/bin/python -m evals.experts_eval \
        --configs base@768 v1@768 expert+gt@448 expert+gt@560 expert+gt@768 \
                  expert+none@560 expert+fake@560

A config is  <model>[+<packet>]@<long side px>  where model is base | v1 |
expert, and packet is gt | none | fake | det (det = the reflex detector's own
output, once data/experts/det_packets.json exists).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.experts.build import COCO, OUT, packet_for  # noqa: E402
from train.experts.verify import check, mentions  # noqa: E402
from veris.experts import EXPERTS, user_text  # noqa: E402
from veris.packet import Obj  # noqa: E402
from veris.prompts import MODES  # noqa: E402

ADAPTERS = {"v1": ROOT / "dataset/sft/out-qlora-3b",
            "outdoor": ROOT / "train/experts/out-outdoor",
            "indoor": ROOT / "train/experts/out-indoor"}
OLD_PROMPT = {"outdoor": MODES["outdoor_nav"], "indoor": MODES["indoor_nav"]}
FAKES = ["dog", "bicycle", "bench", "suitcase", "fire hydrant", "chair", "umbrella"]


def load_rows(n: int | None) -> list[dict]:
    rows = [r for r in map(json.loads, open(OUT / "manifest.jsonl")) if r["split"] == "val"]
    return rows if n is None else [r for s in ("outdoor", "indoor")
                                   for r in [x for x in rows if x["scene"] == s][:n]]


def run(configs: list[str], n: int | None, out_name: str) -> None:
    import torch
    from peft import PeftModel
    from PIL import Image
    from veris.model import VLM

    rows = load_rows(n)
    det = OUT / "det_packets.json"
    det_packets = json.load(open(det)) if det.exists() else {}
    det_side = OUT / "det_packets_side.json"
    det_packets_side = json.load(open(det_side)) if det_side.exists() else {}
    det_sets = {"det": det_packets, "detside": det_packets_side}
    for f in OUT.glob("det_packets_side[0-9]*.json"):     # detside60 = packet threshold 0.60
        det_sets["detside" + f.stem[len("det_packets_side"):]] = json.load(open(f))
    vlm = VLM(quantize=True)
    base = vlm.model
    peft = None
    loaded: set[str] = set()

    def use(adapter: str | None) -> None:
        nonlocal peft
        if adapter is None:
            vlm.model = base if peft is None else peft
            if peft is not None:
                peft.disable_adapter_layers()
            return
        if peft is None:
            peft = PeftModel.from_pretrained(base, str(ADAPTERS[adapter]), adapter_name=adapter)
        elif adapter not in loaded:
            peft.load_adapter(str(ADAPTERS[adapter]), adapter_name=adapter)
        loaded.add(adapter)
        peft.enable_adapter_layers()
        peft.set_adapter(adapter)
        vlm.model = peft

    res_path = ROOT / "evals/results" / f"{out_name}.json"
    results = json.load(open(res_path)) if res_path.exists() else {}
    raw_dir = OUT / "eval_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        model_part, side = cfg.split("@")
        model, _, packet = model_part.partition("+")
        side = int(side)
        agg = {s: dict(n=0, hall_caps=0, mentions=0, hall_mentions=0, cov=0.0,
                       side_checked=0, side_wrong=0, banned=0, words=0, fake_echo=0,
                       ttft=[]) for s in ("outdoor", "indoor")}
        raw = open(raw_dir / f"{cfg}.jsonl", "w")
        for i, r in enumerate(rows):
            scene = r["scene"]
            if model == "expert":
                use(scene)
                ex = EXPERTS[scene]
                rng = random.Random(r["image"])
                fake = None
                if packet.startswith("fake"):
                    absent = [c for c in FAKES if c not in {g["cat"] for g in r["gt"]}]
                    fake = Obj(rng.choice(absent), rng.uniform(-30, 30))
                if packet == "none" or not packet:
                    sense = ""
                elif packet in det_sets:
                    sense = det_sets[packet].get(r["image"], "")
                else:   # gt | fake: clock hours; gtside | fakeside: left/ahead/right
                    sense = packet_for(r["gt"], fake=fake).to_prompt(
                        scene, clock_face=not packet.endswith("side"))
                system, text, max_new = ex.system, user_text(ex, sense), ex.max_tokens
            else:
                use("v1" if model == "v1" else None)
                m = OLD_PROMPT[scene]
                system, text, max_new = m.system_prompt, m.user_prompt.format(context=""), \
                    m.max_new_tokens
            img = Image.open(COCO / "images/val" / r["image"]).convert("RGB")
            img.thumbnail((side, side))
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": text}]}]
            with torch.inference_mode():
                out = "".join(vlm.generate_stream(msgs, max_new)).strip()
            c = check(out, r["gt"])
            a = agg[scene]
            a["n"] += 1
            a["hall_caps"] += bool(c["hallucinated"])
            said = mentions(out)
            a["mentions"] += len(said)
            a["hall_mentions"] += sum(s in c["hallucinated"] for s in said)
            a["cov"] += len(c["covered"]) / max(1, c["gt_cats"])
            a["side_checked"] += c["side_checked"]
            a["side_wrong"] += len(c["side_wrong"])
            a["banned"] += bool(c["banned"])
            a["words"] += len(out.split())
            a["ttft"].append(vlm.last_stats.ttft_s)
            if fake is not None:
                a["fake_echo"] += fake.label in said
            raw.write(json.dumps({"image": r["image"], "scene": scene, "out": out,
                                  "check": c}) + "\n")
            if i % 50 == 0:
                print(f"{cfg} [{i}/{len(rows)}]", flush=True)
        raw.close()

        results[cfg] = {}
        for scene, a in agg.items():
            k = max(1, a["n"])
            t = sorted(a["ttft"]) or [0]
            results[cfg][scene] = {
                "n": a["n"],
                "chair_s": round(a["hall_caps"] / k, 4),
                "chair_i": round(a["hall_mentions"] / max(1, a["mentions"]), 4),
                "coverage": round(a["cov"] / k, 4),
                "side_acc": round(1 - a["side_wrong"] / max(1, a["side_checked"]), 4),
                "side_checked": a["side_checked"],
                "banned_rate": round(a["banned"] / k, 4),
                "mean_words": round(a["words"] / k, 1),
                "ttft_3070_median_s": round(t[len(t) // 2], 3),
                **({"fake_echo": round(a["fake_echo"] / k, 4)} if packet.startswith("fake") else {}),
            }
        results["_meta"] = {
            "dataset": "COCO val2017 (held out; training used train2017 only)",
            "command": "python -m evals.experts_eval --configs " + " ".join(configs),
            "scoring": "train/experts/verify.py (CHAIR objects + side agreement)",
            "ttft_note": "RTX 3070, transformers 4-bit; indicative only, not the board",
            "time": time.strftime("%Y-%m-%d %H:%M"),
        }
        json.dump(results, open(res_path, "w"), indent=1)
        print(cfg, json.dumps(results[cfg]), flush=True)


def make_det_packets(thr: float = 0.45) -> None:
    """Run the fine-tuned reflex detector over the held-out images and store
    the packet it would really write, for the `det` configs."""
    import cv2

    from train.nav.build_oi import CLASSES
    from veris.packet import Packet
    from veris.reflex import onnx_detector
    from train.experts.build import HFOV_DEG, OBSTACLES

    onnx = sorted((ROOT / "data/nav").glob("*.onnx"))[-1]
    detect = onnx_detector(str(onnx), CLASSES, thr)
    out, out_side, n_obj = {}, {}, 0
    for r in load_rows(None):
        img = cv2.imread(str(COCO / "images/val" / r["image"]))
        h, w = img.shape[:2]
        dets = sorted(detect(img), key=lambda d: -(d[2] - d[0]) * (d[3] - d[1]))[:6]
        objs = []
        for x1, y1, x2, y2, conf, label in dets:
            cx = (x1 + x2) / 2 / w
            objs.append(Obj(label, (cx - 0.5) * HFOV_DEG, conf=conf,
                            in_path=label in OBSTACLES and 0.38 < cx < 0.62 and y2 / h > 0.7))
        n_obj += len(objs)
        out[r["image"]] = Packet(objects=tuple(objs)).to_prompt(r["scene"])
        out_side[r["image"]] = Packet(objects=tuple(objs)).to_prompt(r["scene"], clock_face=False)
    if thr != 0.45:                 # a stricter packet than the reflex tier's own threshold
        json.dump(out_side, open(OUT / f"det_packets_side{round(thr * 100)}.json", "w"))
    else:
        json.dump(out, open(OUT / "det_packets.json", "w"))
        json.dump(out_side, open(OUT / "det_packets_side.json", "w"))
    print(f"{len(out)} packets from {onnx.name}, {n_obj / len(out):.1f} objects per image")


VEHICLES = {"car", "truck", "bus"}


def rescore_all_boxes(raw: str = "eval_raw", out: str = "experts_all_boxes") -> None:
    """Hallucination against every COCO box. The manifest's `gt` drops boxes
    under 0.4% of the image (right for building packets, wrong for judging:
    a small real person named from the detector's packet is not invented).
    Re-reads data/experts/eval_raw/*.jsonl; no model needed."""
    from train.experts.verify import check
    ann = json.load(open(COCO / "annotations/instances_val2017.json"))
    name = {c["id"]: c["name"] for c in ann["categories"]}
    size = {im["file_name"]: (im["id"], im["width"]) for im in ann["images"]}
    boxes: dict[int, list] = {}
    for a in ann["annotations"]:
        boxes.setdefault(a["image_id"], []).append(a)
    res = {"command": ".venv/bin/python -m evals.experts_eval --rescore",
           "note": "same outputs as experts*.json, checked against all COCO val2017 boxes"}
    for f in sorted((OUT / raw).glob("*.jsonl")):
        agg = {}
        for r in map(json.loads, open(f)):
            iid, w = size[r["image"]]
            gt = [{"cat": name[a["category_id"]], "cx": (a["bbox"][0] + a["bbox"][2] / 2) / w}
                  for a in boxes.get(iid, [])]
            c = check(r["out"], gt)
            g = agg.setdefault(r["scene"], dict(n=0, hall=0, ment=0, hall_m=0, sc=0, sw=0, veh=0))
            # a vehicle named as the wrong kind of vehicle is an error, but not an invented object
            g["veh"] += sum(h in VEHICLES and any(x["cat"] in VEHICLES for x in gt) for h in c["hallucinated"])
            g["n"] += 1; g["hall"] += bool(c["hallucinated"])
            g["ment"] += c["mentioned"]; g["hall_m"] += len(c["hallucinated"])
            g["sc"] += c["side_checked"]; g["sw"] += len(c["side_wrong"])
        res[f.stem] = {s: {"n": g["n"], "chair_s": round(g["hall"] / g["n"], 4),
                           "chair_i": round(g["hall_m"] / max(g["ment"], 1), 4),
                           "side_acc": round(1 - g["sw"] / max(g["sc"], 1), 4),
                           "side_checked": g["sc"], "invented_mentions": g["hall_m"],
                           "wrong_vehicle_type": g["veh"]} for s, g in agg.items()}
        print(f.stem, json.dumps(res[f.stem]), flush=True)
    json.dump(res, open(ROOT / f"evals/results/{out}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-det-packets", action="store_true")
    ap.add_argument("--det-thr", type=float, default=0.45)
    ap.add_argument("--configs", nargs="+", default=[])
    ap.add_argument("--raw-dir", default="eval_raw", help="with --rescore: eval_raw | eval_raw_v1")
    ap.add_argument("--rescore", action="store_true", help="re-check raw outputs against all COCO boxes")
    ap.add_argument("--n", type=int, help="images per scene (default: all)")
    ap.add_argument("--out", default="experts")
    a = ap.parse_args()
    if a.rescore:
        rescore_all_boxes(a.raw_dir, "experts_all_boxes" + a.raw_dir.removeprefix("eval_raw"))
    if a.make_det_packets:
        make_det_packets(a.det_thr)
    if a.configs:
        run(a.configs, a.n, a.out)
