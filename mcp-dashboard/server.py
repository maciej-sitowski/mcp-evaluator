"""Tiny dashboard server (stdlib only).

    GET /                 -> index.html
    GET /api/runs         -> run summaries, oldest first
    GET /api/runs/<id>    -> one full run (all cases)
    DELETE /api/runs/<id> -> delete one run; needs header "X-Confirm: delete"
    POST /api/runs/clear  -> delete ALL runs; body must be {"confirm": "delete"}

Reads ``<RESULTS_DIR>/runs/*.json`` on every request, so new runs show up on refresh.
"""

import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/results"))
PORT = int(os.environ.get("PORT", "8080"))
INDEX = Path(__file__).parent / "index.html"
RUN_ID = re.compile(r"[A-Za-z0-9_\-]+")


def load_run(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None  # skip unreadable / corrupt files


def list_runs():
    runs = []
    for path in (RESULTS_DIR / "runs").glob("*.json"):
        run = load_run(path)
        if run and "summary" in run:
            runs.append({k: v for k, v in run.items() if k != "cases"})
    runs.sort(key=lambda r: r.get("timestamp", ""))
    return runs


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200):
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/runs":
            self._json(list_runs())
        elif path.startswith("/api/runs/"):
            run_id = path[len("/api/runs/"):]
            if not RUN_ID.fullmatch(run_id):
                return self._json({"error": "bad run id"}, 400)
            run = load_run(RESULTS_DIR / "runs" / ("%s.json" % run_id))
            if run is None:
                return self._json({"error": "not found"}, 404)
            self._json(run)
        elif path == "/health":
            self._json({"status": "ok"})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/runs/clear":
            return self._json({"error": "not found"}, 404)
        # Destructive: require an explicit confirmation token in the body, not just the URL.
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._json({"error": "bad request"}, 400)
        if not isinstance(body, dict) or body.get("confirm") != "delete":
            return self._json({"error": 'confirmation required: {"confirm": "delete"}'}, 400)
        deleted = 0
        try:
            for p in (RESULTS_DIR / "runs").glob("*.json"):
                p.unlink()
                deleted += 1
        except OSError as e:
            return self._json({"error": "could not delete runs (is %s writable?): %s" % (RESULTS_DIR, e), "deleted": deleted}, 500)
        self._json({"deleted": deleted})

    def do_DELETE(self):
        path = urlparse(self.path).path
        if not path.startswith("/api/runs/"):
            return self._json({"error": "not found"}, 404)
        run_id = path[len("/api/runs/"):]
        if not RUN_ID.fullmatch(run_id):
            return self._json({"error": "bad run id"}, 400)
        # Destructive: a custom header can't be sent cross-site without a CORS preflight, so this doubles as a guard.
        if self.headers.get("X-Confirm") != "delete":
            return self._json({"error": "confirmation required: header X-Confirm: delete"}, 400)
        target = RESULTS_DIR / "runs" / ("%s.json" % run_id)
        if not target.is_file():
            return self._json({"error": "not found"}, 404)
        try:
            target.unlink()
        except OSError as e:
            return self._json({"error": "could not delete run (is %s writable?): %s" % (RESULTS_DIR, e)}, 500)
        self._json({"deleted": 1})

    def log_message(self, fmt, *args):  # keep container logs quiet
        pass


if __name__ == "__main__":
    print("Dashboard on http://0.0.0.0:%d (results: %s)" % (PORT, RESULTS_DIR), flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
