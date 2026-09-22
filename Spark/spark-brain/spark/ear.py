"""Ear — microphone capture + energy VAD, via arecord over ALSA.

Pure stdlib (no PortAudio / no build deps). Frames are 20 ms mono 16-bit
at the configured sample rate — directly feedable to Vosk.
"""
import math
import re
from collections import deque
from dataclasses import dataclass
import itertools
import struct
import sys
import subprocess
import time
import threading

try:
    import audioop
except ImportError:  # removed in py3.13+
    audioop = None


class MicStream:
    """Always drain ALSA; keep a bounded buffer sized for the current activity."""

    FRAME_MS = 20

    def __init__(self, cfg):
        a = cfg["audio"]
        self.device = a["input_device"]
        self.rate = a["sample_rate"]
        self.bytes_per_frame = int(self.rate * 2 * self.FRAME_MS / 1000)
        self.proc = None
        self._ready = threading.Condition()
        self._queue = deque(maxlen=1000 // self.FRAME_MS)
        self._retention_s = 1.0
        self._levels = deque(maxlen=250)
        self._learn_noise = True
        self._closed = False
        self._error = None

    def __enter__(self):
        self.proc = subprocess.Popen(
            [
                "arecord", "-q", "-D", self.device,
                "-f", "S16_LE", "-r", str(self.rate), "-c", "1", "-t", "raw",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._reader = threading.Thread(target=self._capture, daemon=True)
        self._reader.start()
        return self

    def __exit__(self, *exc):
        with self._ready:
            self._closed = True
            self._ready.notify_all()
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2)
            self._reader.join(timeout=2)
            self.proc.stdout.close()
        self.proc = None

    def _capture(self):
        try:
            while not self._closed:
                chunk = self.proc.stdout.read(self.bytes_per_frame)
                if len(chunk) != self.bytes_per_frame:
                    raise RuntimeError("arecord stopped delivering audio")
                with self._ready:
                    self._queue.append((time.monotonic(), chunk))
                    if self._learn_noise:
                        self._levels.append(_rms(chunk))
                    self._ready.notify_all()
        except Exception as exc:
            with self._ready:
                self._error = exc
                self._ready.notify_all()

    @property
    def noise_floor(self):
        with self._ready:
            levels = sorted(self._levels)
        # Lower quintile rejects speech peaks. Bound it for speech-only startup.
        return min(1500, max(260, levels[len(levels)//5])) if levels else 500

    def discard(self):
        """Drop captured playback/idle audio before opening a new listen."""
        with self._ready:
            self._queue.clear()

    def learn_noise(self, enabled):
        """Keep the ambient baseline while command speech and TTS are active."""
        with self._ready:
            self._learn_noise = enabled

    def retain(self, seconds):
        """Preserve command starts while a name-only segment is transcribed."""
        with self._ready:
            self._retention_s = max(1.0, seconds)
            self._queue = deque(self._queue, maxlen=int(self._retention_s * 1000 / self.FRAME_MS))

    def _next(self, timeout=3.0):
        with self._ready:
            if not self._ready.wait_for(
                    lambda: self._queue or self._closed or self._error, timeout):
                raise RuntimeError("microphone stalled")
            if self._error and not self._closed:
                raise RuntimeError("microphone capture failed") from self._error
            if self._closed:
                return None
            # Even after a consumer stalls, never replay old microphone audio.
            now = time.monotonic()
            while len(self._queue) > 1 and now - self._queue[0][0] > self._retention_s:
                self._queue.popleft()
            return self._queue.popleft()[1]

    def frames(self):
        while True:
            chunk = self._next()
            if chunk is None:
                return
            yield chunk

    def probe(self, frames=5, timeout=3.0):
        """Verify the mic actually delivers audio (arecord can die at open)."""
        try:
            for _ in range(frames):
                if self._next(timeout) is None:
                    return False
        except RuntimeError:
            return False
        return True


def _rms(pcm):
    if audioop is not None:
        return audioop.rms(pcm, 2)
    # pure-python fallback: signed little-endian 16-bit samples
    n = len(pcm) // 2
    if n == 0:
        return 0
    samples = struct.unpack("<%dh" % n, pcm[: n * 2])
    return int(math.sqrt(sum(s * s for s in samples) / n))


class CommandAudio:
    """Retain consumed wake audio, even across a pause after just the name."""

    def __init__(self, mic, prefix_pcm=b"", sample_rate=16000):
        self.noise_floor = mic.noise_floor
        frame_bytes = int(sample_rate * 2 * MicStream.FRAME_MS / 1000)
        self.prefix_frames = len(prefix_pcm) // frame_bytes
        prefix = (prefix_pcm[i:i+frame_bytes] for i in range(0, len(prefix_pcm), frame_bytes))
        self._frames = itertools.chain(prefix, mic.frames())

    def frames(self):
        return self._frames


def record_utterance(mic, cfg, on_frame=None, should_stop=None, wait_timeout_s=None):
    """Capture speech with onset pre-roll and a room-relative silence endpoint.

    CommandAudio can replay an early wake's audio before the live stream.
    """
    a = cfg["audio"]
    silence_needed = a["silence_ms"] // MicStream.FRAME_MS
    max_frames = min(a["max_utterance_ms"], 8000) // MicStream.FRAME_MS
    max_frames += getattr(mic, "prefix_frames", 0)
    max_frames = min(max_frames, 8000 // MicStream.FRAME_MS)
    deadline = time.monotonic() + wait_timeout_s if wait_timeout_s is not None else None
    started = time.monotonic()

    frames = []
    spoke = False
    silent_run = 0
    peak_rms = 0
    floor = float(getattr(mic, "noise_floor", 500))
    preroll = deque(maxlen=10)  # include the initial consonant before onset
    end_reason = "source ended"

    for frame in mic.frames():
        if should_stop is not None and should_stop():
            return b""
        if deadline is not None and not spoke and time.monotonic() >= deadline:
            return b""
        rms = _rms(frame)
        peak_rms = max(peak_rms, rms)

        # learn the room: quiet-ish frames pull the floor toward themselves
        if not spoke and rms < floor * 1.5:
            floor = floor * 0.97 + rms * 0.03
        eff_stop = max(a.get("stop_rms", 500), int(floor * 1.5))

        final = None
        if on_frame is not None:
            final = on_frame(frame)   # wrapper returns finalized text, if any

        if not spoke:
            preroll.append(frame)
            if rms >= max(a["start_rms"], eff_stop * 1.3):
                spoke = True
                frames.extend(preroll)
                silent_run = 0
            continue

        frames.append(frame)
        quiet = rms < eff_stop
        if quiet:
            silent_run += 1
        else:
            silent_run = 0
        # end conditions
        if silent_run >= silence_needed:
            end_reason = "silence"
            break
        if final and quiet:
            end_reason = "recognizer endpoint"
            break
        if len(frames) >= max_frames:
            end_reason = "length cap"
            break

    if not spoke:
        print(f"[ear] no speech (peak={peak_rms} floor={int(floor)})",
              file=sys.stderr, flush=True)
    else:
        print(f"[ear] utterance {len(frames)*20}ms in {time.monotonic()-started:.2f}s "
              f"({end_reason}, peak={peak_rms} floor={int(floor)})",
              file=sys.stderr, flush=True)
    return b"".join(frames) if spoke else b""


@dataclass
class WakeResult:
    text: str
    prefix_pcm: bytes = b""


def strip_wake_prefix(text, wake_text=""):
    """Remove an optional name, including the alias that actually woke us."""
    words = list(re.finditer(r"[\w']+", text.lower()))
    heard = wake_text.lower().split()
    if not words:
        return ""
    tokens = [w.group() for w in words]
    count = 0
    if tokens[0] == "hey" and len(tokens) > 1 and tokens[1].startswith("spark"):
        count = 2
    elif tokens[0].startswith("spark"):
        count = 1
    elif heard:
        count = 2 if heard[0] in {"hey", "the", "a"} and len(heard) > 1 else 1
        if tokens[:count] != heard[:count]:
            count = 0
    return text[words[count-1].end():].lstrip(" ,.!?:;- ") if count else text


def listen_for_wake(frames, recognizer, cfg, wake_words, tap_check=None):
    """Listen for the wake word on ONE frame iterator.

    `frames` is a single iterator/generator of 20ms PCM frames (NOT a
    stream object — calling .frames() per-frame would restart the
    generator every time; that bug hung the listener forever). Arms on
    sound onset, feeds Vosk a continuous stream, checks both partials
    and finalized text for the wake word, ends the session on ~1.2s of
    quiet. Returns WakeResult on wake; False if tap_check fires.
    """
    a = cfg["audio"]
    silence_limit = a["silence_ms"] // MicStream.FRAME_MS
    arm_rms = a.get("wake_arm_rms", 400)
    # acoustic-confusion wake words ('clark', 'park', ...) only count when
    # the utterance was LOUD — real wake attempts are near-field speech;
    # TV/music/chatter whispering a lookalike word stays below this.
    weak_min_peak = a.get("wake_weak_rms", 1400)
    # only SINGLE-word entries can wake alone — "hey spark" must arrive whole
    wake_first = {w for w in wake_words if " " not in w}
    wake_pairs = {tuple(w.split()[:2]) for w in wake_words if len(w.split()) >= 2}

    # vosk-small's realistic transcription set for the spoken wake word
    _STRONG = {"spark", "sparks", "sparked", "sparkle"}
    _WEAK = {"clark", "clarks", "stark", "starks", "park", "mark",
             "dark", "dock", "spar", "spork", "shark", "spec", "speck",
             "spock", "spa", "step", "steps"}

    def _is_wake(tokens, peak=0):
        if not tokens:
            return False
        first = tokens[0]
        if first in wake_first or first.startswith("spark") or first in _STRONG:
            return True
        pair = tuple(tokens[:2])
        if pair in wake_pairs:
            return True
        # acoustic confusions: 'clark'/'stark'/'the park' — accept only on
        # SHORT utterances (a lone word = a wake attempt), so conversation
        # mentioning them mid-sentence doesn't false-wake.
        if peak < weak_min_peak:
            return False  # too quiet to be a real wake attempt
        if first in _WEAK and len(tokens) <= 2:
            return True
        if pair in {("the", "park"), ("a", "spark"), ("hey", "clark"),
                    ("hey", "stark"), ("hey", "spa"), ("hey", "heart"),
                    ("hey", "hart")} and len(tokens) <= 3:
            return True
        return False


    recognizer.begin()
    preroll = deque(maxlen=10)
    while True:
        armed = False
        silence_run = 0
        peak = 0
        audio = deque(maxlen=200)
        partial_candidate = None
        partial_count = 0
        frame_count = 0
        while True:
            frame = next(frames, None)
            if frame is None:
                return False
            if tap_check is not None and tap_check():
                return False
            rms = _rms(frame)
            if not armed:
                preroll.append(frame)
                if rms >= arm_rms:
                    recognizer.begin()  # fresh decode session at onset
                    audio.extend(preroll)
                    final = None
                    for lead in preroll:
                        final = recognizer.feed(lead) or final
                    preroll.clear()
                    armed = True
                    silence_run = 0
                    peak = rms
                    print(f"[ear] armed (rms={rms})", file=sys.stderr, flush=True)
                    if final and _is_wake(final.lower().split(), peak):
                        return WakeResult(final)
                continue
            audio.append(frame)
            final = recognizer.feed(frame)  # continuous; returns text at endpoints
            peak = max(peak, rms)
            if final:
                print(f"[ear] final: '{final}'", file=sys.stderr, flush=True)
                if _is_wake(final.lower().split(), peak):
                    print(f"[ear] WAKE via final: '{final}'", file=sys.stderr, flush=True)
                    return WakeResult(final)
                # not a wake: new utterance segment — loudness from the last
                # one must NOT authorize a later quiet weak-word hallucination
                peak = 0
                recognizer.begin()
                audio.clear()
                partial_candidate = None
                partial_count = 0
            frame_count += 1
            if not final and frame_count % 5 == 0:
                partial = recognizer.partial().lower()
                # Only strong names can wake early. Confusions still need a
                # completed short utterance and the existing loudness gate.
                if _is_wake(partial.split(), peak=0):
                    name = tuple(partial.split()[:2]) if partial.startswith("hey ") else partial.split()[0]
                    partial_count = partial_count + 1 if name == partial_candidate else 1
                    partial_candidate = name
                    if partial_count >= 2:
                        print(f"[ear] WAKE via partial: '{partial}'", file=sys.stderr, flush=True)
                        return WakeResult(partial, b"".join(audio))
                else:
                    partial_candidate = None
                    partial_count = 0
            if rms < a["stop_rms"]:
                silence_run += 1
                if silence_run >= silence_limit:
                    # session ended — check the session's final tail text
                    # ("Spark!" + pause lands here, never in partials)
                    tail = recognizer.finish().strip().lower()
                    if tail:
                        print(f"[ear] session tail: '{tail}'", file=sys.stderr, flush=True)
                        if _is_wake(tail.split(), peak):
                            print(f"[ear] WAKE via tail: '{tail}'", file=sys.stderr, flush=True)
                            return WakeResult(tail)
                    break
            else:
                silence_run = 0
