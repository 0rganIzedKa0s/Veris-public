#!/usr/bin/env bash
# Remaining 7B configs (MAXN, 15W, 7W) with a reboot before EACH — one 7B
# server lifecycle per boot avoids NvMap fragmentation on the 8GB Orin.
set -uo pipefail
_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -z "${ORIN_SSH:-}" ] && [ -f "$_ROOT/.env" ] && ORIN_SSH=$(grep -m1 "^ORIN_SSH=" "$_ROOT/.env" | cut -d= -f2-)
NANO="${ORIN_SSH:-user@192.168.55.1}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PW_FILE="${PW_FILE:-$REPO/.device-credentials}"
BASE=http://192.168.55.1:8080
# Board sudo password is read at runtime from a gitignored local file
# (dev-bench convenience; the deployed stack uses scoped NOPASSWD rules
# instead — see ops/orin/veris-sudoers).
pw() { grep 'password:' "$PW_FILE" | cut -d' ' -f2; }

wait_ssh() {
  for i in $(seq 1 40); do
    sleep 10
    if timeout 3 bash -c 'exec 3<>/dev/tcp/192.168.55.1/22' 2>/dev/null; then sleep 8; return 0; fi
  done
  return 1
}

for pm in "2:MAXN" "0:15W" "3:7W"; do
  pm_id="${pm%%:*}"; pm_name="${pm##*:}"
  tag="7B-Q4_K_M@${pm_name}"
  echo "=== $tag ==="
  # set mode (may self-reboot on YES); then force our own reboot regardless
  timeout 40 ssh $NANO 'sudo -S -p "" bash -c "echo YES | nvpmodel -m '"$pm_id"' 2>&1 | tail -1; true"' < <(pw) || true
  sleep 5
  timeout 20 ssh $NANO 'sudo -S -p "" reboot' < <(pw) 2>/dev/null || true
  wait_ssh || { echo "BOARD DID NOT RETURN for $tag"; exit 1; }
  mode=$(timeout 10 ssh $NANO 'nvpmodel -q 2>/dev/null | head -1' || echo unknown)
  echo "mode after boot: $mode"
  echo "$mode" | grep -q "$pm_name" || { echo "MODE MISMATCH for $tag — got: $mode"; continue; }
  ssh $NANO 'sudo -S -p "" bash -c "jetson_clocks; bash ~/veris-bench/prep_mem.sh"' < <(pw)
  ssh $NANO "nohup python3 ~/veris-bench/bench/tegrastats_logger.py --out ~/veris-bench/tegra-${tag}.jsonl > /dev/null 2>&1 & \
    nohup ~/llama.cpp/build/bin/llama-server \
      -m \$(ls ~/models/qwen2.5-vl-7b/*Q4_K_M*.gguf) \
      --mmproj \$(ls ~/models/qwen2.5-vl-7b/*mmproj*Q8_0*.gguf) \
      --no-mmap --host 0.0.0.0 --port 8080 -c 2048 -np 1 > ~/veris-bench/server-${tag}.log 2>&1 & \
    echo LAUNCHED"
  ok=0
  for i in $(seq 1 60); do
    if curl -s -m 3 "$BASE/health" | grep -q ok; then ok=1; break; fi
    sleep 5
  done
  [ "$ok" = "1" ] || { echo "SERVER FAILED for $tag"; ssh $NANO "tail -3 ~/veris-bench/server-${tag}.log"; continue; }
  (cd $REPO && uv run python bench/client.py \
      --base-url "$BASE" --tag "$tag" --runs 2 --warmup 1 --out bench/results.jsonl) || echo "CLIENT ERROR for $tag"
  ssh $NANO 'pkill -f "[l]lama-server"; pkill -f "[t]egrastats_logger"; true'
  scp -q $NANO:"~/veris-bench/tegra-${tag}.jsonl" $REPO/bench/ || true
done
echo REMAINING_DONE