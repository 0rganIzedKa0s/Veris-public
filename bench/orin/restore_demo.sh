#!/bin/bash
# Put the board's public demo back exactly as it was before the multi-model
# benchmarks: original .env (Veris-3B-v1, no adapters), hub and watchdog on.
#   bash bench/orin/restore_demo.sh
# Forward again: bash ops/orin/deploy_multimodel.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
O=${ORIN_SSH:-$(grep -E '^ORIN_SSH=' .env | cut -d= -f2- | tr -d '"')}
ssh "$O" 'set -e; cd ~/veris-app
test -f .env.pre-multimodel || { echo "no .env.pre-multimodel: nothing to restore"; exit 1; }
cp .env.pre-multimodel .env
grep -E "^VERIS_(MODEL|LLM_EXTRA)=" .env
sudo -n systemctl restart veris-llm
for i in $(seq 1 40); do sleep 3; curl -s -m 2 localhost:8080/health | grep -q ok && break; done
sudo -n systemctl restart veris-hub      # a hub started with VERIS_PIPELINE=multi must not outlive the adapters
sudo -n systemctl start veris-watchdog.timer
sleep 3
systemctl is-active veris-llm veris-hub veris-ctx veris-watchdog.timer | tr "\n" " "; echo
# /health can say ok while every image request crashes: check the log too
journalctl -u veris-llm --since "-2min" --no-pager | grep -ci "cudaMalloc failed" || true'
