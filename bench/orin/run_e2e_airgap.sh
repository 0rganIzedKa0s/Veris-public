#!/bin/bash
# The whole stack on the board with no network: start bench/orin/e2e.py inside
# a network namespace that has only loopback.
#   bash bench/orin/run_e2e_airgap.sh [adapter-tag=live]
# The board's own llama-server is stopped for the run (two copies of the model
# do not fit) and started again afterwards with the same adapters.
set -euo pipefail
cd "$(dirname "$0")/../.."
O=${ORIN_SSH:-$(grep -E '^ORIN_SSH=' .env | cut -d= -f2- | tr -d '"')}
TAG=${1:-live}

ssh "$O" 'mkdir -p ~/veris-mm/data/e2e ~/veris-mm/data/board_models ~/veris-mm/evals/results'
rsync -a --include='*/' --include='*.py' --exclude='*' veris bench sim "$O":veris-mm/
rsync -a data/e2e/ "$O":veris-mm/data/e2e/
rsync -a --exclude='*.engine' data/board_models/ "$O":veris-mm/data/board_models/

# free the GPU for the copy that runs inside the namespace; e2e.py evicts the
# model files from the page cache itself while that copy loads
ssh "$O" 'sudo -n systemctl stop veris-llm; sleep 3; free -m | sed -n 2p'
ssh "$O" "cd ~/veris-mm && rm -f e2e.done && (nohup unshare -r -n sh -c 'ip link set lo up; PYTHONPATH=site python3 -m bench.orin.e2e run $TAG > e2e.log 2>&1; echo \$? > e2e.done' > /dev/null 2>&1 &)"
until ssh -o ConnectTimeout=10 "$O" 'test -f ~/veris-mm/e2e.done' 2>/dev/null; do sleep 15; done
ssh "$O" 'echo "exit=$(cat ~/veris-mm/e2e.done)"; grep -vE "device_discovery|NvMap" ~/veris-mm/e2e.log | tail -5 | cut -c1-300'
scp -q "$O":veris-mm/evals/results/orin_e2e_airgap.json evals/results/ || true
# detached: a long-lived ssh over the USB link can hang after the remote finishes
ssh "$O" "(nohup sh -c 'sudo -n systemctl start veris-llm; for i in \$(seq 1 40); do sleep 3; curl -s -m 2 localhost:8080/health | grep -q ok && break; done; cd ~/veris-mm && python3 -c \"import pathlib; from bench.orin.e2e import evict; evict(pathlib.Path.home().glob(\\\"models/**/*.gguf\\\"))\"' > /dev/null 2>&1 &)"
echo "result: evals/results/orin_e2e_airgap.json"
