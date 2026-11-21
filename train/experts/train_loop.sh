#!/bin/bash
# Train the experts end to end: wait for the teacher, build the sets, train
# each adapter under the GPU lock, convert it to a GGUF adapter for the board.
#   bash train/experts/train_loop.sh outdoor indoor     (default: both)
#   bash train/experts/train_loop.sh reader             (needs data/experts/sft/reader.json)
# Log: train/experts/runs/train-<expert>.log   Adapter: data/experts/gguf/veris-<expert>.gguf
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
LF=$HOME/LLaMA-Factory
BASE=$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/*/ | head -1)
EXPERTS=("${@:-outdoor indoor}")
[ $# -eq 0 ] && EXPERTS=(outdoor indoor)
mkdir -p train/experts/runs data/experts/gguf

if [[ " ${EXPERTS[*]} " =~ " outdoor " || " ${EXPERTS[*]} " =~ " indoor " ]]; then
    while pgrep -f 'teach_loo[p]' > /dev/null; do sleep 30; done
    .venv/bin/python -m train.experts.build sft || exit 1
fi

for e in "${EXPERTS[@]}"; do
    log=train/experts/runs/train-$e.log
    echo "=== $(date '+%F %T') train $e" | tee -a "$log"
    (cd "$LF" && flock /tmp/veris-gpu.lock .venv/bin/llamafactory-cli train \
        "$ROOT/data/experts/sft/qlora_$e.yaml") >> "$log" 2>&1
    rc=$?
    echo "=== $(date '+%F %T') train $e exit=$rc" | tee -a "$log"
    [ $rc -eq 0 ] || continue
    "$LF"/.venv/bin/python "$HOME"/llama.cpp/convert_lora_to_gguf.py --base "$BASE" \
        --outtype f16 --outfile "data/experts/gguf/veris-$e.gguf" "train/experts/out-$e" \
        >> "$log" 2>&1 && echo "=== gguf ok: data/experts/gguf/veris-$e.gguf" | tee -a "$log"
done
echo TRAIN_LOOP_DONE
