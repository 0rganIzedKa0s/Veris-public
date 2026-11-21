#!/usr/bin/env bash
# Drives the full benchmark matrix against the Nano from the dev box.
# Usage: bash bench/run_matrix.sh
# Produces bench/results.jsonl (client metrics) + bench/tegra-<tag>.jsonl (power).
set -uo pipefail

_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -z "${ORIN_SSH:-}" ] && [ -f "$_ROOT/.env" ] && ORIN_SSH=$(grep -m1 "^ORIN_SSH=" "$_ROOT/.env" | cut -d= -f2-)
NANO="${ORIN_SSH:-user@192.168.55.1}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PW_FILE="${PW_FILE:-$REPO/.device-credentials}"
BASE=http://192.168.55.1:8080
RUNS=${RUNS:-2}

# Board sudo password is read at runtime from a gitignored local file
# (dev-bench convenience; the deployed stack uses scoped NOPASSWD rules
# instead — see ops/orin/veris-sudoers).
pw() { grep 'password:' "$PW_FILE" | cut -d' ' -f2; }

# mode_id:mode_name pairs (per nvpmodel -p on JetPack 6.2 super profile).
# Ordered to minimize reboots: 7W last (different CPU core count -> reboot).
POWER_MODES=("1:25W" "2:MAXN" "0:15W" "3:7W")
MODELS=("7b")

set_power_mode() {  # $1=id $2=name — reboot-safe: never trusts the ssh session
  # timeout guards against the connection dying mid-reboot without close
  timeout 40 ssh $NANO 'sudo -S -p "" bash -c "echo YES | nvpmodel -m '"$1"' 2>&1 | tail -2; true"' < <(pw) || true
  # poll until the board answers AND reports the requested mode
  for i in $(seq 1 40); do
    mode=$(timeout 8 ssh -o ConnectTimeout=5 $NANO 'nvpmodel -q 2>/dev/null | head -1' 2>/dev/null || true)
    if echo "$mode" | grep -q "$2"; then echo "mode confirmed: $mode"; break; fi
    sleep 10
  done
  ssh $NANO 'sudo -S -p "" bash -c "jetson_clocks; true"' < <(pw)
}

for pm in "${POWER_MODES[@]}"; do
  pm_id="${pm%%:*}"; pm_name="${pm##*:}"
  echo "=== power mode $pm_name (id $pm_id) ==="
  set_power_mode "$pm_id" "$pm_name"

  for m in "${MODELS[@]}"; do
    tag="${m^^}-Q4_K_M@${pm_name}"
    echo "--- $tag ---"
    # clean slate ([b]racket pattern so pkill -f can't match its own cmdline);
    # drop page cache — NvMap on Tegra can't reclaim it and cudaMalloc OOMs
    ssh $NANO 'pkill -f "[l]lama-server"; pkill -f "[t]egrastats_logger"; sleep 1; true'
    # memory prep lives in a script ON the board — no quote-nesting roulette
    ssh $NANO 'sudo -S -p "" bash ~/veris-bench/prep_mem.sh' < <(pw)
    # start tegrastats logger + server, detached
    ssh $NANO 'sudo -S -p "" bash -c "nohup tegrastats --interval 500 > /dev/null 2>&1 & true"' < <(pw) >/dev/null 2>&1 || true
    ssh $NANO "nohup python3 ~/veris-bench/bench/tegrastats_logger.py --out ~/veris-bench/tegra-${tag}.jsonl > /dev/null 2>&1 & \
      nohup ~/llama.cpp/build/bin/llama-server \
        -m \$(ls ~/models/qwen2.5-vl-${m}/*Q4_K_M*.gguf) \
        --mmproj \$(ls ~/models/qwen2.5-vl-${m}/*mmproj*Q8_0*.gguf) \
        --no-mmap --host 0.0.0.0 --port 8080 -c 2048 -np 1 > ~/veris-bench/server-${tag}.log 2>&1 & \
      echo LAUNCHED"
    # wait for server health (model load from NVMe)
    ok=0
    for i in $(seq 1 60); do
      if curl -s -m 3 "$BASE/health" | grep -q '"ok"\|ok'; then ok=1; break; fi
      sleep 5
    done
    if [ "$ok" != "1" ]; then
      echo "SERVER FAILED to become healthy for $tag — skipping. Last log:"
      ssh $NANO "tail -5 ~/veris-bench/server-${tag}.log"
      continue
    fi
    # run the benchmark
    (cd $REPO && uv run python bench/client.py \
        --base-url "$BASE" --tag "$tag" --runs "$RUNS" --warmup 1 \
        --out bench/results.jsonl)
    # stop server + logger, pull power log
    ssh $NANO 'pkill -f "[l]lama-server"; pkill -f "[t]egrastats_logger"; sudo -S -p "" pkill -x tegrastats; true' < <(pw)
    scp -q $NANO:~/veris-bench/tegra-${tag}.jsonl $REPO/bench/ || true
  done
done
echo MATRIX_DONE