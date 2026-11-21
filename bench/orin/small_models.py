"""Time the small models on the board: TensorRT FP16 on the GPU and
onnxruntime on the CPU, while the language model stays loaded beside them.

    bash bench/orin/run_on_board.sh data/board_models bench.orin.small_models \
        gate:data/board_models/gate.onnx:image:1x3x224x224 \
        nav:data/board_models/nav.onnx:images:1x3x640x640

Each argument is name:onnx:input:shape. For a second input (some detectors
take the original image size), append ,input2:shape2:dtype.
Run with PY="env PYTHONPATH=site python3" so the board finds cv2/polygraphy.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


def feeds(spec: str) -> dict[str, np.ndarray]:
    out = {}
    for part in spec.split(","):
        bits = part.split(":")
        name, shape = bits[0], [int(x) for x in bits[1].split("x")]
        dtype = bits[2] if len(bits) > 2 else "float32"
        out[name] = (np.random.rand(*shape).astype(dtype) if dtype.startswith("float")
                     else np.full(shape, 640, dtype))
    return out


def mem_available_mb() -> int:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024
    return -1


def median_ms(fn, runs: int, warm: int = 10) -> float:
    for _ in range(warm):
        fn()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(t), 2)


def bench(name: str, onnx: Path, feed: dict) -> dict:
    import onnxruntime as ort
    from polygraphy.backend.common import BytesFromPath
    from polygraphy.backend.trt import EngineFromBytes, TrtRunner

    res: dict = {"onnx_mb": round(onnx.stat().st_size / 1e6, 2),
                 "inputs": {k: list(v.shape) for k, v in feed.items()}}
    for threads in (1, 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        try:
            s = ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])
            res[f"cpu_{threads}thread_ms"] = median_ms(lambda: s.run(None, feed), 30)
            del s
        except Exception as e:           # e.g. an op the CPU build lacks for this dtype
            res["cpu_error"] = str(e)[-200:]
            break

    engine = onnx.with_suffix(".fp16.engine")
    if not engine.exists():
        shapes = ",".join(f"{k}:{'x'.join(map(str, v.shape))}" for k, v in feed.items())
        base = [TRTEXEC, f"--onnx={onnx}", f"--saveEngine={engine}", "--fp16", "--skipInference"]
        t0 = time.time()
        # a static model rejects --shapes; a dynamic one requires it
        p = subprocess.run(base, capture_output=True, text=True)
        if not engine.exists():
            p = subprocess.run(base + [f"--shapes={shapes}"], capture_output=True, text=True)
        res["engine_build_s"] = round(time.time() - t0, 1)
        if p.returncode != 0 or not engine.exists():
            res["trt_error"] = (p.stdout + p.stderr)[-600:]
            return res
    res["engine_mb"] = round(engine.stat().st_size / 1e6, 2)
    before = mem_available_mb()
    with TrtRunner(EngineFromBytes(BytesFromPath(str(engine)))) as r:
        res["trt_fp16_ms"] = median_ms(lambda: r.infer(feed), 200, warm=20)
        res["mem_taken_mb"] = before - mem_available_mb()
    return res


def main() -> None:
    out = {"_meta": {"time": time.strftime("%Y-%m-%d %H:%M"),
                     "command": " ".join(["bench/orin/" + Path(sys.argv[0]).name, *sys.argv[1:]]),
                     "mem_available_mb_at_start": mem_available_mb(),
                     "note": "llama-server (3B base + adapters) loaded during the run "
                             "unless stated; trt = TensorRT 10.3 FP16 via polygraphy, "
                             "median of 200 after warm-up, host copies included"}}
    for arg in sys.argv[1:]:
        name, onnx, spec = arg.split(":", 2)
        out[name] = bench(name, ROOT / onnx, feeds(spec))
        print(name, json.dumps(out[name]), flush=True)
    p = ROOT / "evals/results/orin_small_models.json"
    old = json.load(open(p)) if p.exists() else {}
    old.update(out)
    json.dump(old, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
