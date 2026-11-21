#!/bin/bash
# Everything after the outdoor/indoor experts finish training, unattended:
# reader expert, held-out scoring, then the board with the real adapters.
#   nohup bash train/night_run.sh > train/experts/runs/night.log 2>&1 &
# Each step logs a line starting with "### "; a failed step does not stop the
# ones after it that can still run.
set -uo pipefail
cd "$(dirname "$0")/.."
G=data/experts/gguf
say() { echo "### $(date '+%F %T') $*"; }

say "waiting for outdoor/indoor training (train/experts/train_loop.sh)"
while pgrep -f 'train_loo[p].sh outdoor' > /dev/null; do sleep 60; done
say "outdoor/indoor done: $(ls $G/veris-outdoor.gguf $G/veris-indoor.gguf 2>&1 | tr '\n' ' ')"

if [ -f data/experts/sft/reader.json ]; then
    say "training reader expert"
    bash train/experts/train_loop.sh reader
else
    say "no data/experts/sft/reader.json: reader expert skipped"
fi

say "scoring experts on held-out COCO val2017"
flock /tmp/veris-gpu.lock .venv/bin/python -m evals.experts_eval --n 100 --configs \
    base@768 v1@768 expert+gt@448 expert+gt@560 expert+det@448 expert+none@448 expert+fake@448 \
    || say "experts_eval FAILED"

# the board needs all three adapter slots; a missing one falls back to the
# stand-in so the timing run can still happen (and says so in its note)
NOTE="trained adapters"
declare -A A
for e in outdoor indoor reader; do
    A[$e]=$G/veris-$e.gguf
    if [ ! -f "${A[$e]}" ]; then
        A[$e]=$G/standin-attn.gguf
        NOTE="$NOTE; $e is a STAND-IN"
    fi
done
say "board: loading adapters ($NOTE)"
bash bench/orin/set_adapters.sh "${A[outdoor]}" "${A[indoor]}" "${A[reader]}" live \
    || say "set_adapters FAILED"
say "board: expert timing"
bash bench/orin/run_expert_bench.sh --n 6 --out orin_experts --note "$NOTE, 15W" || say "expert bench FAILED"
say "board: airgapped end-to-end"
bash bench/orin/run_e2e_airgap.sh live || say "e2e FAILED"
say "NIGHT_RUN_DONE"
