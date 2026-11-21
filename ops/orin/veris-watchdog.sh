#!/usr/bin/env bash
# Veris watchdog — runs from veris-watchdog.timer as root every 2 min.
# Restarts a service only when it is ACTIVE (or failed) but its HTTP endpoint
# is dead. Units that were stopped by hand (e.g. for a 7B teacher run) stay
# stopped — `systemctl stop` leaves is-active=inactive and we skip those.
set -u

log() { echo "$*"; }   # journald adds the unit tag + timestamp

# Seconds since the unit entered active state (monotonic clocks on both sides).
unit_age() {
  local started now
  started=$(systemctl show -p ActiveEnterTimestampMonotonic --value "$1")
  [ -z "$started" ] || [ "$started" = "0" ] && { echo 999999; return; }
  now=$(awk '{printf "%d", $1 * 1000000}' /proc/uptime)
  echo $(( (now - started) / 1000000 ))
}

# check_unit <unit> <health-url> <curl-timeout-s> <startup-grace-s>
# Any HTTP status counts as alive (the hub answers 429 when throttling,
# llama-server answers 503 while the model is loading). Only a total
# non-response (curl code 000) after the grace period triggers a restart.
check_unit() {
  local unit=$1 url=$2 tmo=$3 grace=$4 state code age
  state=$(systemctl is-active "$unit")
  case "$state" in
    failed)
      log "$unit: failed — restarting"
      systemctl restart "$unit"
      return ;;
    active) ;;
    *)
      log "$unit: $state — leaving alone (manual stop?)"
      return ;;
  esac
  code=$(curl -s -m "$tmo" -o /dev/null -w '%{http_code}' "$url" 2>/dev/null)
  if [ "$code" = "000" ]; then
    age=$(unit_age "$unit")
    if [ "$age" -lt "$grace" ]; then
      log "$unit: no HTTP response but only ${age}s old (grace ${grace}s) — waiting"
    else
      log "$unit: active but $url unresponsive (${age}s old) — restarting"
      systemctl restart "$unit"
    fi
  else
    log "$unit: ok (HTTP $code)"
  fi
}

check_unit veris-llm http://localhost:8080/health      15 240
check_unit veris-hub http://localhost:8000/api/status   5  60
check_unit veris-ctx http://localhost:8090/health       5  60

# Funnel state is persisted by tailscaled and may be off on purpose — report
# it, never toggle it.
if tailscale funnel status 2>/dev/null | grep -q "Funnel on"; then
  log "funnel: on"
else
  log "funnel: OFF (site is tailnet-only; enable with: tailscale funnel --bg --https=443 http://localhost:8000)"
fi
