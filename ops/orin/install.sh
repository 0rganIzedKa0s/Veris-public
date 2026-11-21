#!/usr/bin/env bash
# Install the Veris boot-resilience stack on the Orin. Run as root from this
# directory: sudo bash install.sh
# Idempotent — safe to re-run after editing any unit or the watchdog.
set -euo pipefail
cd "$(dirname "$0")"

# Units are templated on @VERIS_USER@ — the login that owns ~/veris-app.
# Defaults to whoever invoked sudo; override with VERIS_USER=<login>.
VUSER="${VERIS_USER:-${SUDO_USER:-}}"
[ -z "$VUSER" ] && { echo "Set VERIS_USER to the login that runs the app."; exit 1; }
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
for f in veris-llm.service veris-hub.service veris-ctx.service \
         veris-watchdog.service veris-watchdog.timer veris-sudoers; do
  sed "s/@VERIS_USER@/$VUSER/g" "$f" > "$TMP/$f"
done

install -m 644 "$TMP"/veris-llm.service "$TMP"/veris-hub.service "$TMP"/veris-ctx.service \
               "$TMP"/veris-watchdog.service "$TMP"/veris-watchdog.timer /etc/systemd/system/
install -d /usr/local/lib/veris
install -m 755 veris-watchdog.sh /usr/local/lib/veris/veris-watchdog.sh

install -m 440 "$TMP"/veris-sudoers /etc/sudoers.d/veris-ops
visudo -c -f /etc/sudoers.d/veris-ops

systemctl daemon-reload
systemctl enable veris-llm veris-hub veris-ctx veris-watchdog.timer

# take the ports over from any legacy nohup-launched processes
pkill -f "llama-server" 2>/dev/null || true
pkill -f "uvicorn demo.app" 2>/dev/null || true
pkill -f "uvicorn context_engine" 2>/dev/null || true
sleep 2

systemctl restart veris-llm veris-hub veris-ctx
systemctl start veris-watchdog.timer

echo "installed. journalctl -u veris-llm -u veris-hub -u veris-ctx -u veris-watchdog -f to watch."
