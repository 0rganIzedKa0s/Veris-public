#!/bin/bash
# Second round for the outdoor/indoor experts: real-detector packets with
# left/ahead/right directions. The first adapters are kept in */v1.
#   nohup bash train/experts/v2_run.sh > train/experts/runs/v2.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
say() { echo "### $(date '+%F %T') $*"; }
say "waiting for the running evaluations"
until grep -q SIDE_DONE train/experts/runs/eval_side.log && grep -q FOVEA_F_DONE data/reader/fovea/gpu_f.log; do sleep 30; done
cp data/experts/eval_raw/*.jsonl data/experts/eval_raw_v1/
cp evals/results/experts*.json evals/results/experts_v1/
say "training outdoor + indoor (v2 data)"
bash train/experts/train_loop.sh outdoor indoor
say "scoring v2"
flock /tmp/veris-gpu.lock .venv/bin/python -m evals.experts_eval --n 100 --out experts_v2 --configs \
    expert+gtside@448 expert+detside@448 expert+none@448 expert+fakeside@448 expert+detside@560 || say "eval FAILED"
.venv/bin/python -m evals.experts_eval --rescore > train/experts/runs/rescore_v2.log 2>&1
say "V2_DONE"
