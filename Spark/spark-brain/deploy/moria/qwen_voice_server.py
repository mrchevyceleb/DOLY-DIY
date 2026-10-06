"""Persistent native/GPU Qwen voice clone. POST text -> raw mono PCM16 WAV.

The Pi applies the selected pitch/robot effects once. Piper stays independent
as the fallback. The saved reference, not a new design prompt, fixes identity.
"""
import collections
import hashlib
import http.server
import io
import json
import os
import struct
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlsplit

MAX_BODY = 4096
MAX_TEXT = 600


class BoundedHTTPServer(http.server.ThreadingHTTPServer):
    """Cap request readers as well as GPU generation; excess sockets close."""
    def __init__(self, *args, **kwargs):
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class Voice:
    def __init__(self, profile_path):
        import torch
        import soundfile as sf
        from faster_qwen3_tts import FasterQwen3TTS
        self.sf = sf
        profile_path = Path(profile_path)
        self.profile = json.loads(profile_path.read_text(encoding="utf-8"))
        self.reference = profile_path.parent / self.profile["reference_file"]
        if hashlib.sha256(self.reference.read_bytes()).hexdigest() != self.profile["reference_sha256"]:
            raise ValueError("voice reference checksum mismatch")
        torch.set_num_threads(4)
        torch.manual_seed(42)
        self.backend = os.environ.get("QWEN_BACKEND", "torch")
        self.model = FasterQwen3TTS.from_pretrained(
            self.profile["model"], device="cuda", dtype=torch.bfloat16,
            attn_implementation="sdpa", max_seq_len=2048,
            backend=self.backend, quant="BF16")
        self.model.warmup(prefill_len=256)
        self.cache = collections.OrderedDict()
        self.permit = threading.Lock()
        self.last_generation_s = None
        self.fixed_cache = {}
        self.generate("Hi Matt. I'm Spark.")  # warm model and reusable reference prompt
        line = "I'm here, Matt. I'm listening."
        self.fixed_cache[line] = self.generate(line)[0]

    def generate(self, text):
        if text in self.fixed_cache:
            return self.fixed_cache[text], 0.0
        if text in self.cache:
            self.cache.move_to_end(text)
            return self.cache[text], 0.0
        started = time.perf_counter()
        audios, rate = self.model.generate_voice_clone(
            text=text, language=self.profile["language"],
            ref_audio=str(self.reference), ref_text=self.profile["reference_text"],
            xvec_only=False, max_new_tokens=1024)
        output = io.BytesIO()
        self.sf.write(output, audios[0], rate, format="WAV", subtype="PCM_16")
        data = output.getvalue()
        elapsed = time.perf_counter() - started
        self.last_generation_s = round(elapsed, 3)
        self.cache[text] = data
        if len(self.cache) > 32:
            self.cache.popitem(last=False)
        return data, elapsed

    def stream(self, text):
        """Yield PCM16 packets; a completed take also populates the WAV cache."""
        if text in self.fixed_cache or text in self.cache:
            import wave
            with wave.open(io.BytesIO(self.fixed_cache.get(text) or self.cache[text]), "rb") as wav:
                while True:
                    pcm = wav.readframes(15360)
                    if not pcm:
                        return
                    yield pcm
        import numpy as np
        started = time.perf_counter()
        chunks, size = [], 0
        generator = self.model.generate_voice_clone_streaming(
            text=text, language=self.profile["language"],
            ref_audio=str(self.reference), ref_text=self.profile["reference_text"],
            xvec_only=False, max_new_tokens=1024, chunk_size=8)
        try:
            for audio, rate, _timing in generator:
                if rate != 24000:
                    raise ValueError("unexpected streaming sample rate")
                pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
                if not pcm:
                    continue
                size += len(pcm)
                if size > 4 * 1024 * 1024:
                    raise ValueError("stream exceeds audio limit")
                chunks.append(pcm)
                yield pcm
        finally:
            generator.close()
        if not chunks:
            raise ValueError("empty voice stream")
        import wave
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            wav.writeframes(b"".join(chunks))
        self.cache[text] = output.getvalue()
        if len(self.cache) > 32:
            self.cache.popitem(last=False)
        self.last_generation_s = round(time.perf_counter() - started, 3)


def handler_for(voice):
    class Handler(http.server.BaseHTTPRequestHandler):
        timeout = 10

        def reply(self, code, data, content_type="text/plain", elapsed=None):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Voice-Id", voice.profile["id"])
            if elapsed is not None:
                self.send_header("X-Generation-Seconds", f"{elapsed:.3f}")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/health":
                data = {"ok": True, "voice": voice.profile["id"],
                        "model": voice.profile["model"], "engine": "faster-qwen3-tts",
                        "backend": voice.backend,
                        "reference_sha256": voice.profile["reference_sha256"],
                        "busy": voice.permit.locked(), "fx": "client",
                        "last_generation_s": voice.last_generation_s}
                self.reply(200, json.dumps(data).encode(), "application/json")
            elif path == "/voices":
                self.reply(200, voice.profile["id"].encode())
            else:
                self.reply(404, b"not found")

        def do_POST(self):
            url = urlsplit(self.path)
            if url.path not in ("/", "/stream"):
                return self.reply(404, b"not found")
            name = parse_qs(url.query).get("voice", [voice.profile["id"]])[0]
            if name != voice.profile["id"]:
                return self.reply(404, b"unknown voice")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if not 0 < length <= MAX_BODY:
                return self.reply(413, b"invalid body length")
            try:
                raw = self.rfile.read(length)
                if len(raw) != length:
                    return self.reply(400, b"incomplete body")
                text = raw.decode("utf-8").strip()
            except UnicodeError:
                return self.reply(400, b"invalid UTF-8")
            if not text or len(text) > MAX_TEXT:
                return self.reply(400, b"text must be 1-600 characters")
            if not voice.permit.acquire(blocking=False):
                return self.reply(503, b"voice busy; use fallback")
            streaming_started = False
            try:
                if url.path == "/stream":
                    started = time.perf_counter()
                    packets = voice.stream(text)
                    try:
                        first = next(packets)  # errors before audio get a normal 500
                        self.send_response(200)
                        self.send_header("Content-Type", "application/x-spark-pcm")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.close_connection = True
                        streaming_started = True
                        self.wfile.write(b"SPK1" + struct.pack("<I", 24000))
                        self.wfile.write(struct.pack("<I", len(first)) + first)
                        self.wfile.flush()
                        print(f"qwen stream first PCM {time.perf_counter()-started:.3f}s", flush=True)
                        for pcm in packets:
                            self.wfile.write(struct.pack("<I", len(pcm)) + pcm)
                            self.wfile.flush()
                        self.wfile.write(struct.pack("<I", 0))  # explicit successful EOS
                        self.wfile.flush()
                        print(f"qwen stream finished {time.perf_counter()-started:.3f}s", flush=True)
                    finally:
                        packets.close()
                else:
                    data, elapsed = voice.generate(text)
                    self.reply(200, data, "audio/wav", elapsed)
                    print(f"qwen voice {elapsed:.3f}s, {len(data)} bytes", flush=True)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as error:
                print(f"qwen generation failed: {error}", flush=True)
                if not streaming_started:
                    self.reply(500, b"synthesis failed; use fallback")
            finally:
                voice.permit.release()

        def log_message(self, *args):
            pass

    return Handler


if __name__ == "__main__":
    voice = Voice(os.environ["QWEN_VOICE_PROFILE"])
    bind = os.environ.get("QWEN_BIND", "127.0.0.1")
    port = int(os.environ.get("QWEN_PORT", "8400"))
    print(f"qwen voice ready on {bind}:{port}", flush=True)
    BoundedHTTPServer((bind, port), handler_for(voice)).serve_forever()
