#!/usr/bin/env bash
# Verify the public tester site from the dev box (off-tailnet path).
# IPv6 to *.ts.net is unreachable from WSL2 — always force IPv4.
URL="https://veris-orin.tail68ae71.ts.net"
code=$(curl -4 -s -m 15 -o /dev/null -w '%{http_code}' "$URL/")
# The demo is public by design (throttled, not gated), so 200 is the healthy
# answer. 429 means the throttle is doing its job and the site is up.
case "$code" in
  200) echo "SITE_OK (public, HTTP 200)";;
  429) echo "SITE_OK (up, but throttling right now — HTTP 429)";;
  401) echo "SITE_GATED (HTTP 401 — an ACCESS_CODE gate is set; the resume link expects none)";;
  000) echo "SITE_DOWN (no response — funnel/tailscaled/hub down, or Orin offline)";;
  *)   echo "SITE_UNEXPECTED (HTTP $code)";;
esac
