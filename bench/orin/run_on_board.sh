#!/bin/bash
# Run a repo module ON the board and bring its data directory back.
#   bash bench/orin/run_on_board.sh <data-dir> <module> [args...]
#   bash bench/orin/run_on_board.sh data/planner_llm evals.planner_llm_eval run
# Syncs veris/, evals/, bench/ sources and <data-dir> to ~/veris-mm, runs
# `python3 -m <module> args` there detached (a long-lived ssh over the USB link
# can hang), polls for completion, then syncs <data-dir> and evals/results back.
# PY=<interpreter> picks the board's interpreter (default python3).
set -euo pipefail
cd "$(dirname "$0")/../.."
O=${ORIN_SSH:-$(grep -E '^ORIN_SSH=' .env | cut -d= -f2- | tr -d '"')}
DATA=$1; shift
PY=${PY:-python3}

ssh "$O" "mkdir -p ~/veris-mm/$DATA ~/veris-mm/evals/results"
rsync -a --include='*/' --include='*.py' --exclude='*' veris evals bench sim "$O":veris-mm/
rsync -a "$DATA/" "$O":veris-mm/"$DATA"/
ssh "$O" "cd ~/veris-mm && rm -f job.done && (nohup sh -c '$PY -m $(printf '%q ' "$@") > job.log 2>&1; echo \$? > job.done' > /dev/null 2>&1 &)"
until ssh -o ConnectTimeout=10 "$O" 'test -f ~/veris-mm/job.done' 2>/dev/null; do sleep 15; done
ssh "$O" 'tail -5 ~/veris-mm/job.log | cut -c1-220; echo "exit=$(cat ~/veris-mm/job.done)"'
rsync -a "$O":veris-mm/"$DATA"/ "$DATA"/
rsync -a "$O":veris-mm/evals/results/ evals/results/
