"""Piper TTS HTTP server — runs on Moria (192.168.50.204:8398).

POST plain text -> WAV audio. ~0.4s per sentence on Grace CPU vs 5-15s
for the same voice on the Pi. Optional ?voice=<name> selects the model
by substring match against /opt/piper-moria/voices/*.onnx.

Managed by systemd: piper-tts.service (unit in this directory).
"""
import glob
import http.server
import os
import subprocess
import tempfile
import threading

PIPER = "/opt/piper-moria/piper/piper"
ESPEAK = "/opt/piper-moria/piper/espeak-ng-data"
VOICE_DIR = "/opt/piper-moria/voices"
DEFAULT_VOICE = os.environ.get("PIPER_VOICE", "hfc")
PORT = int(os.environ.get("PIPER_PORT", "8398"))
BIND = os.environ.get("PIPER_BIND", "0.0.0.0")
LOCK = threading.Lock()
MAX_BODY = 16 * 1024  # she speaks sentences, not novels
PERMITS = threading.BoundedSemaphore(8)
READ_TIMEOUT_S = 15


def resolve_voice(name):
    if not name:
        name = DEFAULT_VOICE
    hits = sorted(glob.glob(os.path.join(VOICE_DIR, "*" + name + "*.onnx")))
    return hits[0] if hits else None


class Handler(http.server.BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(READ_TIMEOUT_S)

    def do_GET(self):
        if self.path.startswith("/voices"):
            body = "\n".join(sorted(os.path.basename(p)
                      for p in glob.glob(os.path.join(VOICE_DIR, "*.onnx")))).encode()
        elif self.path.startswith("/health"):
            body = b"ok"
        else:
            body = b"POST text to /?voice=<name> for WAV"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not PERMITS.acquire(blocking=False):
            self.send_response(503)
            self.end_headers()
            return
        try:
            self._handle()
        finally:
            PERMITS.release()

    def _handle(self):
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(self.path).query)
        voice = (q.get("voice") or [None])[0]
        model = resolve_voice(voice)
        if not model:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(("no voice matching " + str(voice)).encode())
            return
        raw_len = self.headers.get("Content-Length")
        try:
            n = int(raw_len)
            if n < 0 or n > MAX_BODY:
                raise ValueError
        except (TypeError, ValueError):
            self.send_response(413)
            self.end_headers()
            return
        text = self.rfile.read(n).decode("utf-8", "replace").strip()
        if not text:
            self.send_response(400)
            self.end_headers()
            return
        with LOCK:
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                proc = subprocess.run(
                    [PIPER, "--model", model, "--espeak_data", ESPEAK,
                     "--output_file", path, "-q"],
                    input=text, capture_output=True, text=True, timeout=30)
                if proc.returncode != 0:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(proc.stderr[:200].encode())
                    return
                with open(path, "rb") as f:
                    data = f.read()
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print("piper-tts on", BIND, PORT, "default voice:", DEFAULT_VOICE, flush=True)
    http.server.ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()
