"""Serve the training dashboard on localhost.

    python scripts/serve_dashboard.py --port 8765

`/data.json` calls build_dashboard.collect() on every request, so the page
reflects the training run without a restart. Everything is read-only; nothing
here writes to results/ or output/.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
WEB = os.path.join(ROOT, "web")

import build_dashboard  # noqa: E402

CONTENT = {".html": "text/html; charset=utf-8", ".js": "application/javascript",
           ".css": "text/css", ".json": "application/json", ".png": "image/png",
           ".svg": "image/svg+xml"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):        # quieter console
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            path = "/index.html"
        if path == "/data.json":
            try:
                # Reload the collector each request: this module gets edited while
                # the dashboard is live, and a cached import would silently keep
                # serving stale logic.
                importlib.reload(build_dashboard)
                body = json.dumps(build_dashboard.collect()).encode("utf-8")
            except Exception as e:
                body = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode("utf-8")
            return self._send(200, body, "application/json")
        rel = path.lstrip("/").replace("/", os.sep)
        fp = os.path.normpath(os.path.join(WEB, rel))
        if not fp.startswith(WEB) or not os.path.isfile(fp):
            return self._send(404, b"not found", "text/plain")
        with open(fp, "rb") as f:
            body = f.read()
        self._send(200, body, CONTENT.get(os.path.splitext(fp)[1],
                                          "application/octet-stream"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"[web] serving http://{a.host}:{a.port}/  (Ctrl-C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
