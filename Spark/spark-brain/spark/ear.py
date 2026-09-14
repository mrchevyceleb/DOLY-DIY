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


def record_utterance(mic, cfg, on_frame=None, should_stop=None, wait_timeout_s=None):
    """Capture one utterance. Ends on:
    - adaptive energy silence (~1.2s below a floor that LEARNS the room —
      fixed thresholds die to fan/room noise and burn the full cap), or
    - a Vosk speech endpoint (on_frame returns finalized text) followed by
      ~0.7s of quiet — natural turn-taking, typically ~1.5s total, or
    - the safety cap (8s).
    """
    a = cfg["audio"]
    silence_needed = a["silence_ms"] // MicStream.FRAME_MS
    max_frames = min(a["max_utterance_ms"], 8000) // MicStream.FRAME_MS
    deadline = time.time() + wait_timeout_s if wait_timeout_s else None

    frames = []
    spoke = False
    silent_run = 0
    peak_rms = 0
    floor = 260.0          # learned ambient noise floor (EMA)
    last_final_at = None

    for frame in mic.frames():
        if should_stop is not None and should_stop():
            return b""
        if deadline is not None and not spoke and time.time() > deadline:
            return b""
        rms = _rms(frame)
        peak_rms = max(peak_rms, rms)

        # learn the room: quiet-ish frames pull the floor toward themselves
        if rms < floor * 2.5:
            floor = floor * 0.97 + rms * 0.03
        eff_stop = max(300, int(floor * 2.2))

        final = None
        if on_frame is not None:
            final = on_frame(frame)   # wrapper returns finalized text, if any

        if not spoke:
            if rms >= max(a["start_rms"], eff_stop * 1.6):
                spoke = True
                frames.append(frame)
                silent_run = 0
            continue

        frames.append(frame)
        if final:
            last_final_at = time.time()
        quiet = rms < eff_stop
        if quiet:
            silent_run += 1
        else:
            silent_run = 0
        # end conditions
        if silent_run >= silence_needed:
            break
        if last_final_at is not None and time.time() - last_final_at > 0.7 and quiet:
            break
        if len(frames) >= max_frames:
            break

    if not spoke:
        print(f"[ear] no speech (peak={peak_rms} floor={int(floor)} eff_stop={int(floor*2.2)})",
              file=sys.stderr, flush=True)
    else:
        print(f"[ear] utterance {len(frames)*20}ms (peak={peak_rms} floor={int(floor)})",
              file=sys.stderr, flush=True)
    return b"".join(frames) if spoke else b""


def listen_for_wake(frames, recognizer, cfg, wake_words, tap_check=None):
    """Listen for the wake word on ONE frame iterator.

    `frames` is a single iterator/generator of 20ms PCM frames (NOT a
    stream object — calling .frames() per-frame would restart the
    generator every time; that bug hung the listener forever). Arms on
    sound onset, feeds Vosk a continuous stream, checks both partials
    and finalized text for the wake word, ends the session on ~1.2s of
    quiet. Returns True on wake; False if tap_check fires.
    """
    import json as _json
    a = cfg["audio"]
    silence_limit = a["silence_ms"] // MicStream.FRAME_MS
    arm_rms = a.get("wake_arm_rms", 400)
    # only SINGLE-word entries can wake alone — "hey spark" must arrive whole
    wake_first = {w for w in wake_words if " " not in w}
    wake_pairs = {tuple(w.split()[:2]) for w in wake_words if len(w.split()) >= 2}

    # vosk-small's realistic transcription set for the spoken wake word
    _STRONG = {"spark", "sparks", "sparked", "sparkle"}
    _WEAK = {"clark", "clarks", "stark", "starks", "park", "mark",
             "dark", "dock", "spar", "spork", "shark"}

    def _is_wake(tokens):
        if not tokens:
            return False
        first = tokens[0]
        if first in wake_first or first.startswith("spark") or first in _STRONG:
            return True
        pair = tuple(tokens[:2])
        if pair in wake_pairs or pair == ("hey", "spark"):
            return True
        # acoustic confusions: 'clark'/'stark'/'the park' — accept only on
        # SHORT utterances (a lone word = a wake attempt), so conversation
        # mentioning them mid-sentence doesn't false-wake.
        if first in _WEAK and len(tokens) <= 2:
            return True
        if pair in {("the", "park"), ("a", "spark"), ("hey", "clark"), ("hey", "stark")} and len(tokens) <= 3:
            return True
        return False


    recognizer.begin()
    while True:
        armed = False
        silence_run = 0
        while True:
            frame = next(frames)
            if tap_check is not None and tap_check():
                return False
            rms = _rms(frame)
            if not armed:
                if rms >= arm_rms:
                    recognizer.begin()  # fresh decode session at onset
                    final = recognizer.feed(frame)
                    armed = True
                    silence_run = 0
                    print(f"[ear] armed (rms={rms})", file=sys.stderr, flush=True)
                    if final and _is_wake(final.lower().split()):
                        return final
                continue
            final = recognizer.feed(frame)  # continuous; returns text at endpoints
            if final:
                print(f"[ear] final: '{final}'", file=sys.stderr, flush=True)
                if _is_wake(final.lower().split()):
                    print(f"[ear] WAKE via final: '{final}'", file=sys.stderr, flush=True)
                    return final
            if rms < a["stop_rms"]:
                silence_run += 1
                if silence_run >= silence_limit:
                    # session ended — check the session's final tail text
                    # ("Spark!" + pause lands here, never in partials)
                    tail = recognizer.finish().strip().lower()
                    if tail:
                        print(f"[ear] session tail: '{tail}'", file=sys.stderr, flush=True)
                        if _is_wake(tail.split()):
                            print(f"[ear] WAKE via tail: '{tail}'", file=sys.stderr, flush=True)
                            return tail
                    break
            else:
                silence_run = 0
