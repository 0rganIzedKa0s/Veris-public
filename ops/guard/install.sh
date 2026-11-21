#!/usr/bin/env bash
# Install the Veris front door on an always-on tailnet node.
#
# Run this AFTER the hostname swap described in ops/guard/README.md — this node
# must already own the funnel hostname the resume link points at.
#
#   sudo TAILNET=tail68ae71 bash install.sh
set -euo pipefail

TAILNET="${TAILNET:?set TAILNET, e.g. TAILNET=tail68ae71}"
GUARD_USER="${SUDO_USER:-$USER}"
SRC="$(cd "$(dirname "$0")" && pwd)"

install -d /opt/veris-guard
install -m 755 "$SRC/veris_guard.py" /opt/veris-guard/veris_guard.py
sed -e "s|@GUARD_USER@|$GUARD_USER|g" -e "s|@TAILNET@|$TAILNET|g" \
    "$SRC/veris-guard.service" > /etc/systemd/system/veris-guard.service

systemctl daemon-reload
systemctl enable --now veris-guard
sleep 2
systemctl --no-pager --lines=5 status veris-guard || true

echo
echo "Now expose it publicly on the hostname the resume link uses:"
echo "  tailscale funnel --bg --https=443 http://localhost:8000"
echo "Then verify:"
echo "  curl -4 -sI https://veris-orin.$TAILNET.ts.net/ | head -3   # expect 302"
echo "  curl -4 -s  https://veris-orin.$TAILNET.ts.net/_guard/health"
