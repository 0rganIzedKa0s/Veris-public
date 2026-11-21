#!/usr/bin/env python3
"""Always-on front door for the Veris demo link.

The demo URL is printed on a resume, and it is served BY the Orin over a
Tailscale funnel — so when the board is unplugged the hostname has no server at
all and a visitor gets Tailscale's relay error. Nothing running on the board
can fix that, because the board is the thing that is gone.

This is the piece that stays up. It runs on an always-on tailnet node that owns
the funnel hostname, checks whether the board is answering, and sends the
visitor to the live demo or to the recorded copy accordingly.

It REDIRECTS rather than reverse-proxies on purpose: the live demo streams SSE
and uploads multipart frames, and a proxy bug in this process would break the
working case it exists to protect. A 302 has no such failure mode.

Standard library only — no venv, no wheels, nothing to break on a fresh VM.

  VERIS_BOARD_HEALTH=http://veris-board:8000/api/status \
  VERIS_LIVE_URL=https://veris-board.tail68ae71.ts.net/ \
  VERIS_RECORDED_URL=https://0rganizedka0s.github.io/Veris-public/ \
  python3 veris_guard.py 8000
"""

import http.server
import json
import os
import socketserver
import sys
import threading
import time
import urllib.request

BOARD_HEALTH = os.environ.get("VERIS_BOARD_HEALTH", "http://veris-board:8000/api/status")
LIVE_URL = os.environ.get("VERIS_LIVE_URL", "https://veris-board.tail68ae71.ts.net/")
RECORDED_URL = os.environ.get("VERIS_RECORDED_URL",
                              "https://0rganizedka0s.github.io/Veris-public/")
CHECK_EVERY = float(os.environ.get("VERIS_CHECK_EVERY", "20"))
TIMEOUT = float(os.environ.get("VERIS_CHECK_TIMEOUT", "6"))
# Two agreeing checks before flipping. A single dropped packet must not bounce a
# visitor to the recorded copy while the board is fine, and one lucky response
# from a board that is rebooting must not send the next visitor into a 502.
FLAPS = int(os.environ.get("VERIS_FLAPS", "2"))


class BoardWatch:
    """Polls the board on its own thread so no request ever waits on the network."""

    def __init__(self):
        self.up = False          # start pessimistic: recorded copy always works
        self._streak = 0
        self._last = None
        self.checked_at = 0.0

    def probe(self) -> bool:
        try:
            with urllib.request.urlopen(BOARD_HEALTH, timeout=TIMEOUT) as r:
                if r.status != 200:
                    return False
                body = json.load(r)
        except Exception:
            return False
        # The hub answers even when llama-server is down; that state still has a
        # working page (it replays recordings locally), so treat the HUB being
        # up as "live demo is worth visiting".
        return isinstance(body, dict) and "orin" in body

    def run(self):
        while True:
            seen = self.probe()
            self.checked_at = time.time()
            self._streak = self._streak + 1 if seen == self._last else 1
            self._last = seen
            if self._streak >= FLAPS and seen != self.up:
                self.up = seen
                print(f"[guard] board {'UP' if seen else 'DOWN'} "
                      f"-> sending visitors to {self.target()}", flush=True)
            time.sleep(CHECK_EVERY)

    def target(self) -> str:
        return LIVE_URL if self.up else RECORDED_URL


WATCH = BoardWatch()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "veris-guard"

    def log_message(self, fmt, *args):  # quieter than the default per-request line
        pass

    def _send(self, code, body: bytes, ctype="text/html; charset=utf-8", headers=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self):
        if self.path == "/_guard/health":
            body = json.dumps({"board_up": WATCH.up, "target": WATCH.target(),
                               "checked_at": WATCH.checked_at}).encode()
            return self._send(200, body, "application/json")

        dest = WATCH.target()
        # Carry the visitor's query string through, so the historical ?k=... on
        # the resume link survives the hop and anything else keeps working.
        q = self.path.split("?", 1)
        if len(q) == 2 and q[1]:
            dest += ("&" if "?" in dest else "?") + q[1]
        # 302, never 301: a permanent redirect would be cached by the visitor's
        # browser and would keep pointing at the recorded copy long after the
        # board came back.
        note = ("the live demo on the board" if WATCH.up
                else "the recorded copy — the board is powered down right now")
        body = (f'<!doctype html><meta charset="utf-8">'
                f'<meta http-equiv="refresh" content="0;url={dest}">'
                f'<title>Veris</title>'
                f'<p>Taking you to {note}: <a href="{dest}">{dest}</a>').encode()
        self._send(302, body, headers=[("Location", dest),
                                       ("Cache-Control", "no-store")])

    do_HEAD = do_GET

    def do_POST(self):
        # The live demo POSTs to /api/*; if a client lands here it is talking to
        # the wrong host, so say where the demo actually is instead of hanging.
        self._send(503, json.dumps(
            {"error": "This is the Veris front door, not the demo hub.",
             "demo": WATCH.target()}).encode(), "application/json")


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    WATCH.up = WATCH.probe()   # answer correctly from the very first request
    WATCH.checked_at = time.time()
    threading.Thread(target=WATCH.run, daemon=True).start()
    print(f"[guard] :{port} — board={'UP' if WATCH.up else 'DOWN'} "
          f"live={LIVE_URL} recorded={RECORDED_URL}", flush=True)
    Server(("0.0.0.0", port), Handler).serve_forever()
