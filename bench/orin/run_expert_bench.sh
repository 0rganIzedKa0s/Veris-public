#!/bin/bash
# Run the expert timing bench ON the board and bring the result back.
#   bash bench/orin/run_expert_bench.sh --n 8 --out orin_experts [--note "..."]
# Needs: llama-server on the board at :8080 with the adapters loaded, and
# data/bench_frames/ (python -m bench.orin.expert_bench prep).
set -euo pipefail
cd "$(dirname "$0")/../.."
O=${ORIN_SSH:-$(grep -E '^ORIN_SSH=' .env | cut -d= -f2- | tr -d '"')}
OUT=orin_experts
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    [ "${args[$i]}" = "--out" ] && OUT=${args[$((i + 1))]}
done

ssh "$O" 'mkdir -p ~/veris-mm/veris ~/veris-mm/bench/orin ~/veris-mm/data ~/veris-mm/evals/results'
scp -q veris/experts.py veris/packet.py "$O":veris-mm/veris/
ssh "$O" 'touch ~/veris-mm/veris/__init__.py ~/veris-mm/bench/__init__.py ~/veris-mm/bench/orin/__init__.py'
scp -q bench/orin/expert_bench.py "$O":veris-mm/bench/orin/
rsync -a --delete data/bench_frames "$O":veris-mm/data/

# Detach on the board and poll: a long-lived ssh over the USB link can hang
# after the remote command has already finished.
ssh "$O" "cd ~/veris-mm && rm -f bench.done && (nohup sh -c 'python3 -m bench.orin.expert_bench run $(printf '%q ' "$@") > bench.log 2>&1; echo \$? > bench.done' > /dev/null 2>&1 &)"
until ssh -o ConnectTimeout=10 "$O" 'test -f ~/veris-mm/bench.done' 2>/dev/null; do sleep 15; done
ssh "$O" 'tail -4 ~/veris-mm/bench.log | cut -c1-200; echo "exit=$(cat ~/veris-mm/bench.done)"'
scp -q "$O":veris-mm/evals/results/"$OUT".json evals/results/
echo "result: evals/results/$OUT.json"
