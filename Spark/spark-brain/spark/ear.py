"""Ear — microphone capture + energy VAD, via arecord over ALSA.

Pure stdlib (no PortAudio / no build deps). Frames are 20 ms mono 16-bit
at the configured sample rate — directly feedable to Vosk.
"""
import math
import selectors
import struct
import sys
import subprocess
import time

try:
    import audioop
except ImportError:  # removed in py3.13+
    audioop = None


class MicStream:
    """Context-managed raw capture from ALSA."""

    FRAME_MS = 20

    def __init__(self, cfg):
        a = cfg["audio"]
        self.device = a["input_device"]
        self.rate = a["sample_rate"]
        self.bytes_per_frame = int(self.rate * 2 * self.FRAME_MS / 1000)
        self.proc = None

    def __enter__(self):
        self.proc = subprocess.Popen(
            [
                "arecord", "-q", "-D", self.device,
                "-f", "S16_LE", "-r", str(self.rate), "-c", "1", "-t", "raw",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return self

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def frames(self):
        while True:
            chunk = self.proc.stdout.read(self.bytes_per_frame)
            if not chunk or len(chunk) < self.bytes_per_frame:
                if self.proc.poll() is not None:
                    raise RuntimeError("arecord died")
                continue
            yield chunk

    def probe(self, frames=5, timeout=3.0):
        """Verify the mic actually delivers audio (arecord can die at open)."""
        import selectors
        sel = selectors.DefaultSelector()
        sel.register(self.proc.stdout, selectors.EVENT_READ)
        got = 0
        try:
            for _ in range(frames):
                if not sel.select(timeout):
                    return False
                chunk = self.proc.stdout.read(self.bytes_per_frame)
                if not chunk or len(chunk) < self.bytes_per_frame:
                    return False
                got += 1
        finally:
            sel.unregister(self.proc.stdout)
            sel.close()
        return got == frames


def _rms(pcm):
    if audioop is not None:
        return audioop.rms(pcm, 2)
    # pure-python fallback: signed little-endian 16-bit samples
    n = len(pcm) // 2
    if n == 0:
        return 0
    samples = struct.unpack("<%dh" % n, pcm[: n * 2])
    return int(math.sqrt(sum(s * s for s in samples) / n))


def record_utterance(mic, cfg, on_frame=None, should_stop=None):
    """Capture one utterance with energy VAD.

    Returns raw PCM bytes (empty string if nothing captured / aborted).
    - waits for speech (rms > start_rms)
    - ends after silence_ms of quiet, or max_utterance_ms
    """
    a = cfg["audio"]
    silence_frames_needed = a["silence_ms"] // MicStream.FRAME_MS
    max_frames = a["max_utterance_ms"] // MicStream.FRAME_MS

    frames = []
    spoke = False
    silent_run = 0
    total = 0
    peak_rms = 0

    for frame in mic.frames():
        if should_stop is not None and should_stop():
            return b""
        total += 1
        if total > max_frames * 4 and not spoke:
            # no speech after a long wait — keep waiting silently (touch-driven)
            total = 0
            continue
        rms = _rms(frame)
        peak_rms = max(peak_rms, rms)
        if on_frame:
            on_frame(frame)
        if not spoke:
            if rms >= a["start_rms"]:
                spoke = True
                frames.append(frame)
                silent_run = 0
                started_at = time.time()
            continue
        frames.append(frame)
        if rms < a["stop_rms"]:
            silent_run += 1
            if silent_run >= silence_frames_needed:
                break
        else:
            silent_run = 0
        if len(frames) >= max_frames:
            break

    if not spoke:
        print(f"[ear] no speech detected (peak rms={peak_rms}, start threshold={a['start_rms']})",
              file=sys.stderr)
    return b"".join(frames) if spoke else b""
