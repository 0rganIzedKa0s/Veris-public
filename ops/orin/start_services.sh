#!/usr/bin/env bash
# Start/restart the Veris demo hub + context engine. Both are owned by systemd
# now (veris-hub.service / veris-ctx.service) so they survive power loss; this
# wrapper keeps the old entry point working.
sudo systemctl restart veris-hub veris-ctx
sleep 5
curl -s -m 5 http://localhost:8000/api/status > /dev/null && echo HUB_OK || echo HUB_FAIL
curl -s -m 5 http://localhost:8090/health > /dev/null && echo CTX_OK || echo CTX_FAIL
