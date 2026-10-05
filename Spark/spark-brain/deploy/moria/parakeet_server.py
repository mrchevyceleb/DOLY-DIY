"""Parakeet ASR HTTP server — runs on Moria (192.168.50.204:8399).

Drop-in whisper-server /inference contract (multipart WAV -> plain text).
NVIDIA Parakeet-TDT 0.6B v3 stays resident on CPU in a restartable worker.
The native shim is compiled against the installed parakeet headers.

Managed by systemd: parakeet-server.service (unit in this directory).
Rollback: systemctl disable --now parakeet-server; enable --now whisper-server.
"""
import http.server
import os
import threading
from parakeet_engine import PersistentEngine

MODEL = "/opt/whisper-moria/models/ggml-parakeet-tdt-0.6b-v3-q8_0.bin"
PORT = int(os.environ.get("PARAKEET_PORT", "8399"))
BIND = os.environ.get("PARAKEET_BIND", "0.0.0.0")
THREADS = os.environ.get("PARAKEET_THREADS", "16")
LOCK = threading.Lock()
MAX_BODY = 2 * 1024 * 1024  # WAVs also capped at 30s before native decoding
PERMITS = threading.BoundedSemaphore(8)  # no unbounded thread pileup
READ_TIMEOUT_S = 15
ENGINE = None  # loaded once before accepting requests


class BoundedHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def process_request(self, request, client_address):
        if not PERMITS.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            PERMITS.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            PERMITS.release()


def valid_wav(data):
    """Structural check before handing bytes to the native decoder."""
    import io
    import wave
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            return (w.getnchannels() == 1 and w.getsampwidth() == 2
                    and 8000 <= w.getframerate() <= 48000
                    and 0 < w.getnframes() <= 30 * w.getframerate())
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
        self._handle()

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
        # Leave 1.5s for normal decoding inside the client's 3.5s deadline.
        if not LOCK.acquire(timeout=2):
            self.send_response(503)
            self.end_headers()
            return
        try:
            text = ENGINE.transcribe(wav)
        except Exception as error:
            print("[asr]", error, flush=True)
            self.send_response(500)
            self.end_headers()
            return
        finally:
            LOCK.release()
        data = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ENGINE = PersistentEngine(MODEL, THREADS)
    print("parakeet-asr on", BIND, PORT, flush=True)
    BoundedHTTPServer((BIND, PORT), Handler).serve_forever()
