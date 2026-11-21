#!/usr/bin/env bash
# Tier 1 detector on the Orin Nano: build an FP16 TensorRT engine from the
# exported D-FINE ONNX and print its latency. Run ON the Orin (JetPack 6.2,
# TensorRT 10.x). Not run from the desktop; nobody has measured it yet.
#
#   scp data/nav/dfine_s_oi18_640.onnx orin:~/veris/data/nav/
#   bash bench/orin/nav_trt.sh [path/to/model.onnx]
#
# Prints trtexec's latency summary (GPU compute and end-to-end incl. H2D/D2H)
# and saves the full log + engine next to the ONNX. For a stable number set
# the power mode and clocks first: sudo nvpmodel -m 0 && sudo jetson_clocks
set -euo pipefail

ONNX=${1:-data/nav/dfine_s_oi18_640.onnx}
TRTEXEC=/usr/src/tensorrt/bin/trtexec
ENGINE=${ONNX%.onnx}_fp16.engine
LOG=${ONNX%.onnx}_trt.log

[ -f "$ONNX" ] || { echo "missing $ONNX" >&2; exit 1; }
[ -x "$TRTEXEC" ] || { echo "missing $TRTEXEC (JetPack TensorRT)" >&2; exit 1; }

echo "power mode: $(nvpmodel -q 2>/dev/null | head -1 || echo unknown)"
"$TRTEXEC" --onnx="$ONNX" --saveEngine="$ENGINE" --fp16 \
    --memPoolSize=workspace:1024 \
    --warmUp=2000 --iterations=200 --avgRuns=50 --useSpinWait \
    > "$LOG" 2>&1 || { tail -40 "$LOG"; echo "trtexec failed, full log: $LOG" >&2; exit 1; }

grep -E "Throughput|Latency:|GPU Compute Time:|Enqueue Time:|H2D Latency|D2H Latency" "$LOG"
ls -l "$ENGINE"
echo "full log: $LOG"
