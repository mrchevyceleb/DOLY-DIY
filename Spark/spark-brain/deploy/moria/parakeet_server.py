"""Parakeet ASR HTTP server — runs on Moria (192.168.50.204:8399).

Drop-in replacement for whisper.cpp's whisper-server /inference contract
(multipart file upload -> plain text), but backed by NVIDIA Parakeet-TDT
0.6B v3 via whisper.cpp's parakeet-cli: ~0.5s per utterance on CPU with
better-than-large-v3 English accuracy. The whisper-server build on Moria
has no GPU backend; large-v3-turbo took ~8.8s per clip there, parakeet
takes ~0.55s.

Managed by systemd: parakeet-server.service (unit in this directory).
Rollback: systemctl disable --now parakeet-server; enable --now whisper-server.
"""
import http.server
import os
import subprocess
import tempfile
import threading

BIN = "/opt/whisper-moria/build/bin/parakeet-cli"
MODEL = "/opt/whisper-moria/models/ggml-parakeet-tdt-0.6b-v3-q8_0.bin"
PORT = int(os.environ.get("PARAKEET_PORT", "8399"))
BIND = os.environ.get("PARAKEET_BIND", "0.0.0.0")
THREADS = os.environ.get("PARAKEET_THREADS", "16")
LOCK = threading.Lock()
MAX_BODY = 2 * 1024 * 1024   # robot caps utterances at 5s (~160KB); 2MB is generous
PERMITS = threading.BoundedSemaphore(8)  # no unbounded thread pileup
READ_TIMEOUT_S = 15


def valid_wav(data):
    """Structural check before handing bytes to the native decoder."""
    import io
    import wave
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            return (w.getnchannels() == 1 and w.getsampwidth() == 2
                    and 8000 <= w.getframerate() <= 48000 and w.getnframes() > 0)
    except Exception:
        return False


def extract_wav(body, ctype):
    """curl-style multipart: pull the part named 'file' out of the body."""
    boundary = None
    for tok in ctype.split(";"):
        tok = tok.strip()
        if tok.startswith("boundary="):
            boundary = tok[len("boundary="):].encode()
    if not boundary:
        return None
    for part in body.split(b"--" + boundary):
        head, _, payload = part.partition(b"\r\n\r\n")
        if b'name="file"' not in head:
            continue
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        return payload
    return None


class Handler(http.server.BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(READ_TIMEOUT_S)

    def do_GET(self):
        body = b"parakeet asr server - POST multipart file to /inference"
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
        raw_len = self.headers.get("Content-Length")
        try:
            n = int(raw_len)
            if n < 0 or n > MAX_BODY:
                raise ValueError
        except (TypeError, ValueError):
            self.send_response(413)
            self.end_headers()
            return
        body = self.rfile.read(n)
        wav = extract_wav(body, self.headers.get("Content-Type", ""))
        if not wav:
            self.send_response(400)
            self.end_headers()
            return
        if not valid_wav(wav):
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"not a valid 16-bit mono wav")
            return
        with LOCK:
            fd, path = tempfile.mkstemp(suffix=".wav")
            try:
                os.write(fd, wav)
                os.close(fd)
                proc = subprocess.run(
                    [BIN, "-t", THREADS, "-m", MODEL, "-f", path, "-np"],
                    capture_output=True, text=True, timeout=30)
                text = " ".join(proc.stdout.split())
            except subprocess.TimeoutExpired:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"asr timeout")
                return
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        if proc.returncode != 0:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(proc.stderr[:200].encode())
            return
        data = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print("parakeet-asr on", BIND, PORT, flush=True)
    http.server.ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()
