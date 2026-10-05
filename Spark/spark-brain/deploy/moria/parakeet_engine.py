"""Resident CPU ASR, isolated in a restartable worker with a hard deadline."""
import ctypes as C
import io
import os
from pathlib import Path
import select
import struct
import subprocess
import sys
import time
import wave

MAX_WAV = 2 * 1024 * 1024
MAX_REPLY = 65536


def read_exact(pipe, count, deadline=None):
    result = bytearray()
    while len(result) < count:
        if deadline is not None:
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([pipe], [], [], left)[0]:
                raise TimeoutError("ASR worker timeout")
        chunk = os.read(pipe.fileno(), count - len(result))
        if not chunk:
            raise RuntimeError("ASR worker exited")
        result.extend(chunk)
    return bytes(result)


class PersistentEngine:
    """Caller serializes access. A stuck native decoder cannot strand HTTP."""
    def __init__(self, model, threads):
        self.model, self.threads = model, threads
        self.process = None
        self.start()

    def close(self):
        p, self.process = self.process, None
        if p is not None:
            if p.poll() is None:
                p.kill()
            p.wait(timeout=2)
            p.stdin.close()
            p.stdout.close()

    def start(self):
        self.close()
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", self.model, str(self.threads)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        os.set_blocking(self.process.stdin.fileno(), False)
        try:
            if read_exact(self.process.stdout, 5, time.monotonic() + 20) != b"READY":
                raise RuntimeError("ASR worker failed to load")
        except Exception:
            self.close()
            raise

    def transcribe(self, wav):
        if not self.process or self.process.poll() is not None:
            self.start()
        try:
            # Native worker drains input before inference. Unbuffered writes
            # can be short; write all bytes rather than truncating the WAV.
            packet = memoryview(struct.pack("<I", len(wav)) + wav)
            deadline = time.monotonic() + 6
            while packet:
                left = deadline - time.monotonic()
                if left <= 0 or not select.select([], [self.process.stdin], [], left)[1]:
                    raise TimeoutError("ASR worker input timeout")
                try:
                    n = os.write(self.process.stdin.fileno(), packet)
                except BlockingIOError:
                    continue
                if not n:
                    raise RuntimeError("ASR worker input closed")
                packet = packet[n:]
            size = struct.unpack("<I", read_exact(self.process.stdout, 4, deadline))[0]
            if not 1 <= size <= MAX_REPLY:
                raise RuntimeError("invalid ASR worker response")
            reply = read_exact(self.process.stdout, size, deadline)
            if reply[0] != 0:
                raise RuntimeError(reply[1:].decode("utf-8", errors="replace"))
            return reply[1:].decode("utf-8")
        except Exception:
            self.close()  # reload only after failure, never on a normal request
            raise


def worker(model, threads):
    import audioop
    lib = C.CDLL(str(Path(__file__).with_name("libspark_asr.so")))
    lib.spark_asr_open.argtypes = [C.c_char_p]
    lib.spark_asr_open.restype = C.c_void_p
    lib.spark_asr_decode.argtypes = [C.c_void_p, C.POINTER(C.c_int16), C.c_int, C.c_int, C.c_void_p, C.c_int]
    handle = lib.spark_asr_open(os.fsencode(model))
    if not handle:
        raise RuntimeError("cannot load Parakeet")
    os.write(sys.stdout.fileno(), b"READY")
    while True:
        size = struct.unpack("<I", read_exact(sys.stdin.buffer, 4))[0]
        if not 1 <= size <= MAX_WAV:
            raise RuntimeError("invalid ASR input size")
        wav = read_exact(sys.stdin.buffer, size)
        try:
            with wave.open(io.BytesIO(wav), "rb") as w:
                rate, frames = w.getframerate(), w.getnframes()
                if w.getnchannels() != 1 or w.getsampwidth() != 2 or not 8000 <= rate <= 48000 or frames > rate * 30:
                    raise ValueError("unsupported ASR audio")
                pcm = w.readframes(frames)
                if len(pcm) != frames * 2:
                    raise ValueError("truncated ASR audio")
            if rate != 16000:
                pcm, _ = audioop.ratecv(pcm, 2, 1, rate, 16000, None)
            samples = (C.c_int16 * (len(pcm) // 2)).from_buffer_copy(pcm)
            output = C.create_string_buffer(MAX_REPLY - 1)
            result = lib.spark_asr_decode(handle, samples, len(samples), int(threads), output, len(output))
            if result != 0:
                raise RuntimeError(f"ASR decode failed ({result})")
            reply = b"\0" + b" ".join(output.value.split())
        except Exception as error:
            reply = b"\1" + str(error).encode("utf-8")[:1000]
        packet = memoryview(struct.pack("<I", len(reply)) + reply)
        while packet:
            packet = packet[os.write(sys.stdout.fileno(), packet):]


if __name__ == "__main__":
    worker(sys.argv[2], sys.argv[3])
