#!/bin/bash
# Point the board's llama-server at the stock base plus three adapters and restart it.
#   bash bench/orin/set_adapters.sh <outdoor.gguf> <indoor.gguf> <reader.gguf> [tag]
# Writes VERIS_MODEL / VERIS_LLM_EXTRA in the board's ~/veris-app/.env (the
# original is kept once as .env.pre-multimodel) and restarts veris-llm, which
# drops the page cache first. Restore the demo with bench/orin/restore_demo.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
O=${ORIN_SSH:-$(grep -E '^ORIN_SSH=' .env | cut -d= -f2- | tr -d '"')}
TAG=${4:-live}
D=models/veris-adapters/$TAG
ssh "$O" "mkdir -p ~/$D"
scp -q "$1" "$O:$D/veris-outdoor.gguf"
scp -q "$2" "$O:$D/veris-indoor.gguf"
scp -q "$3" "$O:$D/veris-reader.gguf"
ssh "$O" "set -e; cd ~/veris-app; [ -f .env.pre-multimodel ] || cp .env .env.pre-multimodel
A=/home/\$USER/$D
sed -i \"s|^VERIS_MODEL=.*|VERIS_MODEL=/home/\$USER/models/qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf|; s|^VERIS_LLM_EXTRA=.*|VERIS_LLM_EXTRA=--image-max-tokens 1600 --cache-ram 128 --lora \$A/veris-outdoor.gguf,\$A/veris-indoor.gguf,\$A/veris-reader.gguf --lora-init-without-apply|\" .env
sudo -n systemctl stop veris-watchdog.timer veris-hub
sudo -n systemctl restart veris-llm
for i in \$(seq 1 40); do sleep 3; curl -s -m 2 localhost:8080/health | grep -q ok && break; done
# the loaded files now sit in page cache, which the GPU allocator cannot reclaim:
# free memory is ~100 MB and a second large image aborts the server. Evict them.
cd ~/veris-mm && python3 -c \"import pathlib; from bench.orin.e2e import evict; evict(pathlib.Path.home().glob('models/**/*.gguf'))\"; grep MemFree /proc/meminfo
curl -s localhost:8080/health; echo; curl -s localhost:8080/lora-adapters | grep -o '\"path\":\"[^\"]*\"'"
