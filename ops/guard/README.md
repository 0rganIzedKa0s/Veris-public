# Veris front door

Keeps the resume link working when the Orin is unplugged.

## The problem

`https://veris-orin.<tailnet>.ts.net/` is served **by the Orin itself** through
Tailscale Funnel. Tailscale routes that hostname to exactly one node, so when
the board loses power the hostname has no server behind it and a visitor gets
Tailscale's relay error. Nothing that runs on the board can fix this — the
board is the thing that is missing. The demo's in-page offline banner only
covers the case where the hub answers but the model is down.

## The fix

Give the funnel hostname to a node that is always on, and let it decide where
to send people:

```
  https://veris-orin.<tailnet>.ts.net/     <- the printed resume link
                  |
          [ this guard, always on ]
                  |
      board answering? --yes--> https://veris-board.<tailnet>.ts.net/   (live demo)
                  |
                  no
                  v
      https://0rganizedka0s.github.io/Veris-public/   (recorded copy)
```

It redirects rather than reverse-proxies. The live demo streams SSE and uploads
multipart frames; a proxy bug here would break the exact case this exists to
protect, and a 302 cannot fail that way. The visitor's query string is carried
across, so the historical `?k=...` on already-printed links still lands
correctly.

## Setup

You need any always-on Linux host — a free-tier VM is fine; it does nothing but
answer a few bytes per visit.

**1. Rename the board, so its current name is free.** On the Orin:

```bash
sudo tailscale set --hostname=veris-board
tailscale funnel --bg --https=443 http://localhost:8000    # re-arm under the new name
curl -4 -sI https://veris-board.<tailnet>.ts.net/ | head -1
```

Confirm in the admin console that the machine now shows as `veris-board` and
that nothing still holds `veris-orin`. Tailscale frees a released name, but it
will append a suffix (`veris-orin-1`) if it thinks the name is taken — check
before continuing, because the whole point is inheriting the exact hostname.

**2. Join the always-on host to the tailnet as `veris-orin`:**

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --hostname=veris-orin
tailscale status | grep veris-orin     # verify it took the name unsuffixed
```

**3. Install the guard and expose it:**

```bash
sudo TAILNET=<tailnet> bash install.sh
tailscale funnel --bg --https=443 http://localhost:8000
```

**4. Verify both states:**

```bash
curl -4 -sI https://veris-orin.<tailnet>.ts.net/ | head -3     # 302
curl -4 -s  https://veris-orin.<tailnet>.ts.net/_guard/health  # {"board_up": ...}
# then power the board down and re-check: the target must switch to the
# recorded copy within ~40s (two 20s probes)
```

## Rolling it back

Stop the guard, rename the board back to `veris-orin`, re-arm its funnel. The
guard holds no state, so nothing is lost.

## Tuning

Environment variables, all optional: `VERIS_BOARD_HEALTH`, `VERIS_LIVE_URL`,
`VERIS_RECORDED_URL`, `VERIS_CHECK_EVERY` (default 20s),
`VERIS_CHECK_TIMEOUT` (6s), `VERIS_FLAPS` (2 agreeing probes before switching).
