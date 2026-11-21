#!/usr/bin/env bash
# Closed-loop teacher generation: 25W+reboot -> 7B -> generate -> RESTORE demo.
# The restore runs even if generation fails (trap).
set -uo pipefail
_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -z "${ORIN_SSH:-}" ] && [ -f "$_ROOT/.env" ] && ORIN_SSH=$(grep -m1 "^ORIN_SSH=" "$_ROOT/.env" | cut -d= -f2-)
NANO="${ORIN_SSH:-user@192.168.55.1}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PW_FILE="${PW_FILE:-$REPO/.device-credentials}"
# Board sudo password is read at runtime from a gitignored local file
# (dev-bench convenience; the deployed stack uses scoped NOPASSWD rules
# instead — see ops/orin/veris-sudoers).
pw() { grep 'password:' "$PW_FILE" | cut -d' ' -f2; }

wait_ssh() {
  for i in $(seq 1 40); do sleep 10
    timeout 3 bash -c 'exec 3<>/dev/tcp/192.168.55.1/22' 2>/dev/null && { sleep 8; return 0; }
  done; return 1
}

restore_demo() {
  echo "== RESTORE: 15W + 3B + demo stack =="
  timeout 40 ssh $NANO 'sudo -S -p "" bash -c "systemctl start docker containerd tailscaled 2>/dev/null; modprobe uvcvideo 2>/dev/null; true"' < <(pw) || true
  timeout 40 ssh $NANO 'sudo -S -p "" bash -c "echo YES | nvpmodel -m 0 2>&1 | tail -1; true"' < <(pw) || true
  sleep 5
  timeout 20 ssh $NANO 'sudo -S -p "" reboot' < <(pw) 2>/dev/null || true
  wait_ssh || { echo "RESTORE FAILED: board did not return"; return 1; }
  ssh $NANO 'sudo -S -p "" bash ~/veris-bench/prep_mem.sh > /dev/null' < <(pw)
  ssh $NANO 'bash ~/veris-app/restart_llm.sh "" && bash ~/veris-app/start_services.sh'
  # tailscale serve config persists across reboots; verify HTTPS answers
  # (set VERIS_PUBLIC_URL to your node's https://<node>.<tailnet>.ts.net)
  if [ -n "${VERIS_PUBLIC_URL:-}" ]; then
    ssh $NANO "curl -s -m 20 $VERIS_PUBLIC_URL/api/status > /dev/null && echo HTTPS_OK || echo HTTPS_CHECK_FAILED"
  fi
  echo "== RESTORE DONE =="
}
trap restore_demo EXIT

echo "== phase 1: 25W + fresh boot =="
timeout 40 ssh $NANO 'sudo -S -p "" bash -c "echo YES | nvpmodel -m 1 2>&1 | tail -1; true"' < <(pw) || true
sleep 5
timeout 20 ssh $NANO 'sudo -S -p "" reboot' < <(pw) 2>/dev/null || true
wait_ssh || { echo "board did not return"; exit 1; }
mode=$(timeout 10 ssh $NANO 'nvpmodel -q 2>/dev/null | head -1')
echo "mode: $mode"
echo "$mode" | grep -q 25W || { echo "MODE MISMATCH"; exit 1; }

echo "== phase 2: memory prep + 7B server =="
# the 7B has ~95MB of headroom; park every noncritical daemon during the run.
# CRITICAL: unload the USB camera driver — uvcvideo's kernel buffers block the
# GPU's large contiguous allocations (every 7B failure since the C920 was
# plugged in traces to this; restore modprobes it back for the demo).
ssh $NANO 'sudo -S -p "" bash -c "systemctl stop docker containerd tailscaled 2>/dev/null; rmmod uvcvideo 2>/dev/null; true"' < <(pw)
# SINGLE cache drop — the repeating dropper loop correlated with every
# first-request crash (drop_caches churn during large NvMap allocations)
ssh $NANO 'sudo -S -p "" bash -c "sync; echo 3 > /proc/sys/vm/drop_caches"' < <(pw)
ssh $NANO 'bash ~/veris-app/run_7b_server.sh' | tail -1 | grep -q SERVER_OK || { echo "7B FAILED"; exit 1; }
# belt-and-braces: ensure no dropper loop survives into generation
ssh $NANO 'sudo -S -p "" bash -c "pkill -f \"seq 1 12[0]\" 2>/dev/null; true"' < <(pw)
sleep 3

echo "== phase 3: teacher generation (long) =="
cd $REPO
export PATH="$HOME/.local/bin:$PATH"
uv run python dataset/generate_teacher.py --limit 300
echo "== generation complete: $(wc -l < dataset/teacher_raw.jsonl) rows =="