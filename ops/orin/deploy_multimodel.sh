#!/bin/bash
# Put the multi-model pipeline behind the public demo on the board.
#   bash ops/orin/deploy_multimodel.sh
# The adapters (~/models/veris-adapters/live) and the small models
# (~/veris-mm/data) must already be on the board: bench/orin/set_adapters.sh
# and bench/orin/run_e2e_airgap.sh put them there.
# Back to the single model: bash bench/orin/restore_demo.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
O=${ORIN_SSH:-$(grep -E '^ORIN_SSH=' .env | cut -d= -f2- | tr -d '"')}

ssh "$O" 'mkdir -p ~/veris-app/bench/orin'
rsync -a --include='*.py' --exclude='*' veris/ "$O":veris-app/veris/
rsync -a demo/app.py demo/pipeline.py demo/local_store.py "$O":veris-app/demo/
rsync -a demo/static/ "$O":veris-app/demo/static/
rsync -a bench/orin/e2e.py "$O":veris-app/bench/orin/

ssh "$O" 'set -e; cd ~/veris-app
[ -e data ] || ln -s ~/veris-mm/data data
[ -f .env.pre-multimodel ] || cp .env .env.pre-multimodel
A=$HOME/models/veris-adapters/live
ls $A/veris-outdoor.gguf $A/veris-indoor.gguf $A/veris-reader.gguf data/board_models/gate.onnx > /dev/null
set_kv() { if grep -q "^$1=" .env; then sed -i "s|^$1=.*|$1=$2|" .env; else echo "$1=$2" >> .env; fi; }
set_kv VERIS_MODEL "$HOME/models/qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
set_kv VERIS_LLM_EXTRA "--image-max-tokens 1600 --cache-ram 128 --lora $A/veris-outdoor.gguf,$A/veris-indoor.gguf,$A/veris-reader.gguf --lora-init-without-apply"
set_kv VERIS_PIPELINE multi
set_kv PYTHONPATH "$HOME/veris-mm/site"
sudo -n systemctl stop veris-watchdog.timer veris-hub
sudo -n systemctl restart veris-llm
for i in $(seq 1 40); do sleep 3; curl -s -m 2 localhost:8080/health | grep -q ok && break; done
sudo -n systemctl start veris-hub
# the hub warms the server, evicts the model files from the page cache and
# loads the gate, detector and OCR; wait for it to say so
for i in $(seq 1 40); do sleep 3; curl -s -m 2 localhost:8000/api/status | grep -q "\"pipeline_ready\": *true" && break; done
sudo -n systemctl start veris-watchdog.timer
curl -s -m 5 localhost:8000/api/status | grep -o "\"pipeline[a-z_]*\": *[a-z\"]*" | tr "\n" " "; echo
systemctl is-active veris-llm veris-hub veris-ctx veris-watchdog.timer | tr "\n" " "; echo
grep MemFree /proc/meminfo
journalctl -u veris-llm --since "-3min" --no-pager | grep -ci "cudaMalloc failed" || true'
