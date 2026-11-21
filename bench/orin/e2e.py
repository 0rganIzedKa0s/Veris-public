"""Every tier together, on the board, with no network.

Run inside a network namespace that has only loopback (`unshare -r -n`), so
the airgap is a property of the kernel, not a promise: there is no interface
to send a packet on. Inside it this script starts llama-server with the base
model and the adapters, loads the gate, detector, OCR and offline planner
beside it, and drives the Conductor through five scenes.

    prep   (dev box)  build data/e2e/ and collect models in data/board_models/
    run    (board)    the measurement; writes evals/results/orin_e2e_airgap.json

    .venv/bin/python -m bench.orin.e2e prep
    bash bench/orin/run_e2e_airgap.sh <adapter-tag>
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
E2E = ROOT / "data/e2e"
MODELS = ROOT / "data/board_models"
HOME = Path.home()
PORT = 8081
HAS_TRT = importlib.util.find_spec("polygraphy") is not None


# ── dev box ──────────────────────────────────────────────────────────
def prep() -> None:
    import shutil

    import cv2

    from sim import gps
    from train.nav.build_oi import CLASSES
    from veris.wearcam import degrade

    E2E.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    street = cv2.imread(str(ROOT / "test_frames/crossing.jpg"))
    h, w = street.shape[:2]
    # an approach, synthesised: zooming on the frame centre grows every box the
    # way walking toward the scene would. 14 frames at 10 fps, 3% per frame.
    frames = []
    for i in range(14):
        z = 1.03 ** i
        cw, ch = int(w / z), int(h / z)
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        f = cv2.resize(street[y0:y0 + ch, x0:x0 + cw], (1280, int(1280 * h / w)))
        cv2.imwrite(str(E2E / f"street_{i:02d}.jpg"), f)
        frames.append(f"street_{i:02d}.jpg")
    for name in ("kitchen", "sign"):
        img = cv2.imread(str(ROOT / f"test_frames/{name}.jpg"))
        s = 1280 / max(img.shape[:2])
        cv2.imwrite(str(E2E / f"{name}.jpg"), cv2.resize(img, None, fx=min(s, 1), fy=min(s, 1)))
    from evals.reader_docs import wildreceipt       # camera photos of receipts, as documents
    for k, (img, _) in enumerate(wildreceipt(8)):
        cv2.imwrite(str(E2E / f"doc_{k}.jpg"), img)
    cv2.imwrite(str(E2E / "blur.jpg"), degrade(cv2.imread(str(E2E / "kitchen.jpg")),
                                               "indoor_worst", seed=1))
    fix = next(r["fix"] for r in gps.simulate("sidewalk_crossings", 0)[30:] if r["fix"])
    json.dump({"fix": fix, "street": frames, "classes": CLASSES},
              open(E2E / "scenes.json", "w"), indent=1)

    shutil.copy(ROOT / "data/gate/gate.onnx", MODELS / "gate.onnx")
    shutil.copy(ROOT / "data/maps/index.pkl", MODELS / "index.pkl")
    for m in ("ch_PP-OCRv5_det_mobile.onnx", "en_PP-OCRv5_rec_mobile.onnx"):
        shutil.copy(ROOT / "data/reader/models" / m, MODELS / m)
    for extra in (ROOT / "data/reader/models").glob("*.txt"):
        shutil.copy(extra, MODELS / extra.name)
    nav = sorted((ROOT / "data/nav").glob("*.onnx"))
    if nav:
        shutil.copy(nav[-1], MODELS / "nav.onnx")
    print("prepared", E2E, "and", MODELS, "| detector:", nav[-1].name if nav else "MISSING")


# ── board ────────────────────────────────────────────────────────────
def network_evidence() -> dict:
    def sh(cmd):
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

    ev = {"interfaces": sh("ip -br addr"), "routes": sh("ip route") or "(none)"}
    try:
        s = socket.socket()
        s.settimeout(3)
        s.connect(("1.1.1.1", 443))
        ev["connect_1.1.1.1:443"] = "CONNECTED (not airgapped!)"
    except OSError as e:
        ev["connect_1.1.1.1:443"] = f"failed: {e}"
    try:
        socket.getaddrinfo("api.openai.com", 443)
        ev["dns_api.openai.com"] = "RESOLVED (not airgapped!)"
    except OSError as e:
        ev["dns_api.openai.com"] = f"failed: {e}"
    ev["airgapped"] = "CONNECTED" not in json.dumps(ev) and "RESOLVED" not in json.dumps(ev)
    return ev


def mem_available_mb() -> int:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024
    return -1


def evict(paths) -> None:
    """Drop these files from the page cache, without root. The GPU allocator
    on this board cannot reclaim page cache, and reading a 2.8 GB model fills
    it, so the server's last and largest buffer fails unless the files it has
    just read are evicted while it loads (the same idea as the systemd unit's
    drop_caches, per file)."""
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except OSError:
            pass


def start_llama(tag: str):
    a = HOME / "models/veris-adapters" / tag
    q = HOME / "models/qwen2.5-vl-3b"
    files = [*q.glob("*.gguf"), *a.glob("*.gguf"), *MODELS.glob("*"), *E2E.glob("*")]
    evict(files)
    cmd = [str(HOME / "llama.cpp/build/bin/llama-server"),
           "-m", str(q / "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"),
           "--mmproj", str(q / "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"),
           "--no-mmap", "-ngl", "99", "--host", "127.0.0.1", "--port", str(PORT),
           "-c", "4096", "-np", "1", "--image-max-tokens", "1600", "--cache-ram", "128",
           "--lora", ",".join(str(a / f"veris-{n}.gguf") for n in ("outdoor", "indoor", "reader")),
           "--lora-init-without-apply"]
    log = open(ROOT / "llama-e2e.log", "w")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 240:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if b"ok" in r.read():
                    # /health says ok even when the vision buffer failed to
                    # allocate; that server dies on its first image
                    if "cudaMalloc failed" in (ROOT / "llama-e2e.log").read_text():
                        p.kill()
                        raise RuntimeError("vision buffer allocation failed at load")
                    return p, round(time.time() - t0, 1)
        except OSError:
            pass
        if p.poll() is not None:
            break
        evict(files)
        time.sleep(0.5)
    p.kill()
    raise RuntimeError("llama-server did not come up; see llama-e2e.log")


def make_detector(classes: list[str], thr: float = 0.5, size: int = 640):
    """The detector as a Det callable: TensorRT engine if one was built,
    otherwise the CPU session from veris.reflex."""
    import cv2
    import numpy as np

    engine = MODELS / "nav.fp16.engine"
    if not (engine.exists() and HAS_TRT):       # an engine only runs on the board that built it
        from veris.reflex import onnx_detector
        return onnx_detector(str(MODELS / "nav.onnx"), classes, thr, size), "onnxruntime-cpu"
    from polygraphy.backend.common import BytesFromPath
    from polygraphy.backend.trt import EngineFromBytes, TrtRunner
    runner = TrtRunner(EngineFromBytes(BytesFromPath(str(engine))))
    runner.activate()
    name = runner.get_input_metadata().keys().__iter__().__next__()

    def detect(frame):
        h, w = frame.shape[:2]
        x = cv2.resize(frame, (size, size))[:, :, ::-1].transpose(2, 0, 1)[None]
        out = runner.infer({name: np.ascontiguousarray(x, np.float32) / 255})
        boxes = next(v for v in out.values() if v.shape[-1] == 4)
        logits = next(v for v in out.values() if v.shape[-1] != 4)
        p = 1 / (1 + np.exp(-logits[0]))
        res = []
        for qi, ci in zip(*np.nonzero(p > thr)):
            cx, cy, bw, bh = boxes[0, qi]
            res.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w,
                        (cy + bh / 2) * h, float(p[qi, ci]), classes[ci]))
        return res
    return detect, "tensorrt-fp16"


class TrtSession:
    """A TensorRT engine behind onnxruntime's `run(None, feed)` call shape."""

    def __init__(self, engine: Path):
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner
        self.r = TrtRunner(EngineFromBytes(BytesFromPath(str(engine))))
        self.r.activate()

    def run(self, _, feed):
        return list(self.r.infer(feed).values())


class Timed:
    """Wrap one method and keep how long each call took."""

    def __init__(self, obj, method: str):
        self.obj, self.ms = obj, []
        self._fn = getattr(obj, method)
        setattr(self, method, self._call)

    def _call(self, *a, **kw):
        t0 = time.perf_counter()
        out = self._fn(*a, **kw)
        self.ms.append((time.perf_counter() - t0) * 1e3)
        return out

    def __getattr__(self, k):
        return getattr(self.obj, k)


def med(xs) -> float | None:
    return round(statistics.median(xs), 2) if xs else None


def run(tag: str) -> None:
    import cv2

    from veris import reader as reader_mod
    from veris.arbiter import Arbiter, Motion
    from veris.conductor import Conductor
    from veris.experts import LlamaClient
    from veris.gate import Gater
    from veris.planner import Planner, locate_guard, locate_prompt
    from veris.ranging import Camera
    from veris.reflex import Reflex

    res: dict = {"time": time.strftime("%Y-%m-%d %H:%M"), "adapters": tag,
                 "network": network_evidence(), "mem_available_mb": {}}
    res["mem_available_mb"]["before_anything"] = mem_available_mb()
    low = [10 ** 9]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.5):
            low[0] = min(low[0], mem_available_mb())
    threading.Thread(target=sample, daemon=True).start()

    llama, load_s = start_llama(tag)
    try:
        res["llama_load_s"] = load_s
        res["mem_available_mb"]["after_llm"] = mem_available_mb()
        sc = json.load(open(E2E / "scenes.json"))
        detect, det_backend = make_detector(sc["classes"])
        g = Gater(MODELS / "gate.onnx", threads=4)
        gate_backend = "onnxruntime-cpu x4"
        if (MODELS / "gate.fp16.engine").exists():
            g.sess, gate_backend = TrtSession(MODELS / "gate.fp16.engine"), "tensorrt-fp16"
        gate = Timed(g, "run")
        reflex = Timed(Reflex(detect, Camera.from_hfov(1280, 853)), "step")
        reader = Timed(reader_mod.Reader(model_dir=MODELS), "read_regions")
        planner = Timed(Planner(MODELS / "index.pkl"), "plan")
        llm = LlamaClient(f"http://127.0.0.1:{PORT}", timeout=300)
        clock = time.monotonic
        arb = Arbiter(now=clock)
        c = Conductor(arb, gate, reflex, reader, planner, llm, now=clock,
                      speak_sure=reader_mod.speak_sure, warm=llm.warm,
                      foveate=reader_mod.mosaic, doc_sheet=reader_mod.document_sheet)
        res["mem_available_mb"]["after_all_models"] = mem_available_mb()
        res["backends"] = {"gate": gate_backend, "detector": det_backend,
                           "ocr": "onnxruntime-cpu", "llm": "llama.cpp CUDA",
                           "planner": "rules + llm (text only)"}

        def describe(name: str, question=None) -> dict:
            t0 = time.perf_counter()
            first, parts, src = None, [], ""
            for say in c.describe(question):
                first = first if first is not None else time.perf_counter() - t0
                parts.append(say.text + " " * (say.source == "ocr"))
                src = src if say.source in src else (src + " then " + say.source).removeprefix(" then ")
            return {"scene": name, "source": src, "first_s": round(first or 0, 3),
                    "total_s": round(time.perf_counter() - t0, 2), "text": "".join(parts).strip()}

        scenes = []
        c.on_fix(sc["fix"])
        res["situation"] = {"situation": c.situation.situation,
                            "suggested_mode": c.situation.suggested_mode,
                            "source": c.situation.source,
                            "class_priors": c.situation.class_priors}

        # 1. walking outdoors toward the scene: the reflex should speak on its own
        arb.user_command("outdoor_nav")
        arb.update_motion(Motion.WALKING)
        hazards = []
        for i, f in enumerate(sc["street"]):
            t0 = time.perf_counter()
            for say in c.on_frame(cv2.imread(str(E2E / f))):
                hazards.append({"frame": i, "text": say.text,
                                "frame_ms": round((time.perf_counter() - t0) * 1e3, 1)})
            time.sleep(0.1)
        res["hazards_spoken"] = hazards
        res["objects_last_frame"] = [o.phrase() for o in c.packet.objects]
        scenes.append(describe("outdoor, walking (first request after boot)"))
        c.on_frame(c._frames[-1][0])      # the same frame again: only recent frames are described
        scenes.append(describe("outdoor, walking (same frame asked again)"))
        c._frames.clear()
        c.on_frame(cv2.imread(str(E2E / sc["street"][4])))
        scenes.append(describe("outdoor, walking (same expert, new frame)"))

        # sustained frame rate of the always-on tiers: no pause between frames,
        # so the CPU governor is not ramping up from idle on every call
        imgs = [cv2.imread(str(E2E / f)) for f in sc["street"]]
        n0, loop = len(gate.ms), []
        for i in range(60):
            t0 = time.perf_counter()
            c.on_frame(imgs[i % len(imgs)])
            loop.append((time.perf_counter() - t0) * 1e3)
        res["sustained"] = {"frames": 60, "frame_ms_median": med(loop),
                            "gate_ms_median": med(gate.ms[n0:]),
                            "reflex_ms_median": med(reflex.ms[n0:]),
                            "fps": round(1000 / statistics.median(loop), 1)}

        # 2. indoors, standing still: describe mode, the gate's vote picks the expert
        arb.update_motion(Motion.STATIONARY)
        arb.user_command("describe")
        for _ in range(3):
            c.on_frame(cv2.imread(str(E2E / "kitchen.jpg")))
        scenes.append(describe("indoor, still (first request after a switch)"))

        # 3. reading a sign
        arb.user_command("text_reading")
        for _ in range(3):
            c.on_frame(cv2.imread(str(E2E / "sign.jpg")))
        scenes.append(describe("sign"))

        # 3b. a printed sign on a street: OCR is sure of the sign and speaks it
        # at once; the stray text behind it is left to the reader expert
        street = cv2.imread(str(E2E / "street_00.jpg"))
        h, w = street.shape[:2]
        cv2.rectangle(street, (w // 4, h // 3), (3 * w // 4, 2 * h // 3), (255, 255, 255), -1)
        cv2.putText(street, "FIRE EXIT", (w // 4 + 30, h // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    w / 420, (0, 0, 0), max(4, w // 160))
        c._frames.clear()
        for _ in range(3):
            c.on_frame(street)
        scenes.append(describe("short sign on a street"))

        # 3c. documents: phone photos of receipts (WildReceipt test), each read once
        docs = []
        for f in sorted(E2E.glob("doc_*.jpg")):
            c._frames.clear()
            for _ in range(2):
                c.on_frame(cv2.imread(str(f)))
            d = describe(f.stem)
            if d["source"] != "gate":           # a frame the gate refuses is not a reading
                docs.append(d | {"request_ttft_s": round(llm.last.ttft_s, 3)})
        res["documents"] = {"n": len(docs), "first_s": [d["first_s"] for d in docs],
                            "first_s_median": med([d["first_s"] for d in docs]),
                            "first_s_max": max([d["first_s"] for d in docs], default=0),
                            "total_s_median": med([d["total_s"] for d in docs]),
                            "readings": docs}

        # 3d. back outdoors by command: the expert is warmed on the first frame after it
        arb.user_command("outdoor_nav")
        for f in sc["street"][:3]:
            c.on_frame(cv2.imread(str(E2E / f)))
            time.sleep(0.1)
        scenes.append(describe("outdoor (first request after a switch)"))

        # 4. a blurred frame only: nothing usable, so the answer is a cue, at once
        c._frames.clear()
        for _ in range(3):
            c.on_frame(cv2.imread(str(E2E / "blur.jpg")))
        scenes.append(describe("blurred frame"))

        # 5. the second LLM: a location question over the offline map
        q = "Where is the nearest crossing?"
        t0 = time.perf_counter()
        ans = llm.run("planner", None, "", locate_prompt(planner.last_facts, q), max_tokens=40)
        scenes.append({"scene": "location question", "source": "planner (base, no adapter)",
                       "question": q, "first_s": round(llm.last.ttft_s, 3),
                       "total_s": round(time.perf_counter() - t0, 2),
                       "text": locate_guard(ans, planner.last_facts, q), "raw": ans})

        res["scenes"] = scenes
        res["tiers_ms"] = {"gate": med(gate.ms), "reflex_detect_track_policy": med(reflex.ms),
                           "ocr_read": med(reader.ms), "planner_rules": med(planner.ms),
                           "n_frames": len(gate.ms)}
        res["mem_available_mb"]["lowest_during_run"] = low[0]
        res["mem_total_mb"] = int(open("/proc/meminfo").readline().split()[1]) // 1024
    finally:
        stop.set()
        llama.terminate()
        llama.wait(30)
    out = ROOT / "evals/results/orin_e2e_airgap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1)[:3000])


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep()
    else:
        run(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("ADAPTERS", "live"))
