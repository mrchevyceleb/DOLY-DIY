"""Announcements: anything on Matt's network can hand Spark a line to say.

    curl -X POST http://spark:8765/announce?mood=good \
         -H "Authorization: Bearer $(cat ~/.spark-token)" -d "The build finished."

The listener only queues. The voice loop speaks the line from the main
thread between turns, so an announcement never talks over a conversation.
Overnight, asleep or hushed, it waits unless it is marked urgent.
"""
import collections
import hmac
import json
import os
import re
import secrets
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MOODS = ("info", "good", "bad")
_MAX_BODY = 4096
_MAX_TEXT = 280
_MAX_QUEUE = 8


def _log(msg):
    print(f"[inbox] {msg}", file=sys.stderr)


def _truthy(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _clean(text):
    text = re.sub(r"[^\x20-\x7e -￿]+", " ", str(text or ""))
    return re.sub(r"\s+", " ", text).strip()[:_MAX_TEXT]


class Inbox:
    def __init__(self, cfg):
        icfg = cfg.get("inbox", {}) or {}
        self.enabled = bool(icfg.get("enabled", True))
        self.port = int(icfg.get("port", 8765))
        self._queue = collections.deque()
        self._lock = threading.Lock()
        self._token_path = os.path.join(cfg.get("state_dir", "."), "announce-token")
        self._token = None
        self._server = None

    # ---------------------------------------------------------------- queue
    def post(self, text, mood="info", urgent=False):
        """Queue one line. False when it is empty or the queue is full."""
        text = _clean(text)
        if not text:
            return False
        with self._lock:
            if len(self._queue) >= _MAX_QUEUE:
                # a full queue of held lines must not block an urgent one
                held = next((i for i in self._queue if not i["urgent"]), None)
                if not urgent or held is None:
                    return False
                self._queue.remove(held)
            self._queue.append({"text": text, "mood": mood if mood in MOODS else "info",
                                "urgent": bool(urgent), "at": time.time()})
        _log(f"queued ({mood}{', urgent' if urgent else ''}): {text[:60]}")
        return True

    @property
    def waiting(self):
        return bool(self._queue)

    def ready(self, quiet=False):
        """Something to say now? Quiet hours hold everything but urgent lines."""
        with self._lock:
            return any(item["urgent"] or not quiet for item in self._queue)

    def pop(self, quiet=False):
        with self._lock:
            for item in self._queue:
                if item["urgent"] or not quiet:
                    self._queue.remove(item)
                    return item
        return None

    # --------------------------------------------------------------- server
    def _load_token(self):
        try:
            with open(self._token_path, encoding="utf-8") as f:
                token = f.read().strip()
            if token:
                os.chmod(self._token_path, 0o600)
                return token
        except OSError:
            pass
        token = secrets.token_urlsafe(24)
        fd = os.open(self._token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(token + "\n")
        return token

    def start(self):
        if not self.enabled or self._server is not None:
            return False
        inbox = self

        class Handler(BaseHTTPRequestHandler):
            timeout = 5   # a stalled client cannot hold a thread open
            def log_message(self, *args):
                pass

            def _reply(self, code, payload):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._reply(200, {"ok": True}) if self.path == "/health" else self._reply(404, {"error": "not found"})

            def do_POST(self):
                url = urllib.parse.urlsplit(self.path)
                if url.path != "/announce":
                    return self._reply(404, {"error": "not found"})
                given = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
                if not hmac.compare_digest(given.encode(), inbox._token.encode()):
                    return self._reply(401, {"error": "bad token"})
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = -1
                if not 0 < length <= _MAX_BODY:
                    return self._reply(400, {"error": f"body must be 1-{_MAX_BODY} bytes"})
                raw = self.rfile.read(length).decode("utf-8", "replace")
                query = dict(urllib.parse.parse_qsl(url.query))
                text, mood, urgent = raw, query.get("mood", "info"), _truthy(query.get("urgent"))
                if "json" in (self.headers.get("Content-Type") or ""):
                    try:
                        data = json.loads(raw)
                        text = data.get("text")
                        mood = data.get("mood", mood)
                        urgent = _truthy(data.get("urgent", urgent))
                    except (ValueError, AttributeError):
                        return self._reply(400, {"error": "invalid JSON"})
                if mood not in MOODS:
                    return self._reply(400, {"error": f"mood must be one of {', '.join(MOODS)}"})
                if not _clean(text):
                    return self._reply(400, {"error": "no text"})
                if not inbox.post(text, mood, urgent):
                    return self._reply(429, {"error": "queue full"})
                self._reply(202, {"ok": True})

        # bind before touching the token: a second process that cannot have
        # the port must not rewrite the token the first one is using
        try:
            server = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        except OSError as e:
            _log(f"cannot listen on {self.port}: {e}")
            return False
        try:
            self._token = self._load_token()
        except OSError as e:
            server.server_close()
            _log(f"no token file, announcements off: {e}")
            return False
        self._server = server
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        _log(f"listening on :{self.port}")
        return True
