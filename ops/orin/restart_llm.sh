#!/usr/bin/env bash
# Usage: [VERIS_MODEL=<gguf path>] restart_llm.sh "<extra llama-server flags>"
# The server itself is owned by systemd (veris-llm.service). This persists the
# model choice + extra flags to .env so they survive power loss, then restarts
# the unit and waits for health.
ENVF="$HOME/veris-app/.env"

set_env() {  # set_env KEY VALUE — upsert into .env
  if grep -q "^$1=" "$ENVF" 2>/dev/null; then
    sed -i "s|^$1=.*|$1=$2|" "$ENVF"
  else
    echo "$1=$2" >> "$ENVF"
  fi
}

[ -n "${VERIS_MODEL:-}" ] && set_env VERIS_MODEL "$VERIS_MODEL"
set_env VERIS_LLM_EXTRA "${1:-}"

sudo systemctl restart veris-llm
for i in $(seq 1 40); do
  sleep 3
  curl -s -m 2 http://localhost:8080/health | grep -q ok && { echo SERVER_OK; exit 0; }
done
echo SERVER_FAIL
journalctl -u veris-llm -n 5 --no-pager
