"""Ear — microphone capture + speech VAD, via arecord over ALSA.

No PortAudio. Frames are 20 ms mono 16-bit
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
        cutoff = a.get("highpass_hz", 150)
        self._highpass = SpeechHighPass(self.rate, cutoff) if cutoff else None
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
                if self._highpass:
                    chunk = self._highpass.process(chunk)
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

    def drain_pending(self):
        """Nonblocking audio for stop recognition while motion polls sensors."""
        with self._ready:
            if self._error or self._closed:
                raise RuntimeError("Microphone unavailable during movement")
            now = time.monotonic()
            frames = [pcm for stamp, pcm in self._queue if now-stamp <= 1]
            self._queue.clear()
            return frames

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


class SpeechHighPass:
    """Streaming second-order Butterworth filter for DC and mains hum.

    Keep state across frames: resetting every 20 ms creates new transients
    that look like speech. Filter once at capture so VAD and ASR agree.
    """

    def __init__(self, sample_rate, cutoff_hz=150):
        if not 0 < cutoff_hz < sample_rate / 2:
            raise ValueError("highpass_hz must be below the Nyquist frequency")
        omega = 2 * math.pi * cutoff_hz / sample_rate
        cosine = math.cos(omega)
        alpha = math.sin(omega) / math.sqrt(2)
        self.b0 = (1 + cosine) / (2 * (1 + alpha))
        self.b1 = -2 * self.b0
        self.a1 = -2 * cosine / (1 + alpha)
        self.a2 = (1 - alpha) / (1 + alpha)
        self.z1 = self.z2 = 0.0

    def process(self, pcm):
        samples = struct.unpack("<%dh" % (len(pcm) // 2), pcm)
        result = []
        z1, z2 = self.z1, self.z2
        b0, b1, a1, a2 = self.b0, self.b1, self.a1, self.a2
        for sample in samples:
            out = b0 * sample + z1
            z1 = b1 * sample - a1 * out + z2
            z2 = b0 * sample - a2 * out
            result.append(max(-32768, min(32767, int(out))))
        self.z1, self.z2 = z1, z2
        return struct.pack("<%dh" % len(result), *result)


def _speech_detector(cfg):
    """Reject fan noise by spectrum, rather than assuming loud means speech."""
    try:
        import webrtcvad
    except ImportError:
        print("[ear] WebRTC VAD unavailable; using energy fallback", file=sys.stderr, flush=True)
        return None
    return webrtcvad.Vad(cfg["audio"].get("vad_mode", 2))


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
    vad = _speech_detector(cfg)
    voiced_run = 0
    energy = deque(maxlen=5)
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
        energy.append(rms)
        # VAD can classify steady electrical/fan noise as voiced. The median
        # rejects isolated noise spikes without clipping a consonant's onset.
        median_rms = sorted(energy)[len(energy)//2]
        voiced = (vad.is_speech(frame, a["sample_rate"])
                  and median_rms >= max(a.get("stop_rms", 500), floor*1.8)) if vad else rms >= eff_stop

        final = None
        if on_frame is not None:
            final = on_frame(frame)   # wrapper returns finalized text, if any

        if not spoke:
            preroll.append(frame)
            # Three frames avoid opening a follow-up on a tap or fan spike.
            onset = voiced and rms >= (a["start_rms"] if vad else max(a["start_rms"], eff_stop * 1.3))
            voiced_run = voiced_run + 1 if onset else 0
            if voiced_run >= (3 if vad else 1):
                spoke = True
                frames.extend(preroll)
                silent_run = 0
            continue

        frames.append(frame)
        quiet = not voiced if vad else rms < eff_stop
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
    if tokens[0] in {"hey", "okay", "ok"} and len(tokens) > 1 and tokens[1] in {"spark", "sparky"}:
        count = 2
    elif tokens[0] in {"spark", "sparky"}:
        count = 1
    elif heard:
        count = 2 if heard[0] in {"hey", "the", "a"} and len(heard) > 1 else 1
        if tokens[:count] != heard[:count]:
            count = 0
    return text[words[count-1].end():].lstrip(" ,.!?:;- ") if count else text


def _lev(a, b, cap=3):
    """Bounded edit distance; returns cap+1 when already too far."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def _near_wake(token, heads):
    """Acoustic neighbor of a wake head ('bark'/'bart' for 'spark').

    Vosk-small transcribes the spoken name as 'bar', 'bark', 'barkley',
    'bart' — the leading s-cluster of 'spark' is what gets swallowed, so
    compare against the head AND its cluster-dropped form ('park').
    Common unrelated words ('what', 'right', 'go') stay three or more away.
    """
    if len(token) < 3:
        return False
    variants = set(heads)
    for head in heads:
        if len(head) > 2 and head[0] == "s" and head[1] not in "aeiou":
            variants.add(head[1:])
    return min(_lev(token, v) for v in variants) <= 2


def has_wake_name(text, wake_words):
    """True when the text leads with her name (or a near-name garble).

    'bark'/'barks' are how a muffled or distant mic renders 'spark' —
    the fricative s drops first. The Vosk wake grammar already emits
    them as legal confusables; accepting them here keeps a dull mic
    usable. A stray 'bark' wake just opens a 6s listening window.
    """
    tokens = re.findall(r"[\w']+", text.lower())
    if tokens and tokens[0] in {"okay", "ok"}:
        tokens = tokens[1:]
    if not tokens:
        return False
    return (tokens[0] in {"spark", "sparky", "bark", "barks"} or any(
        tokens[:len(name.split())] == name.lower().split() for name in wake_words if name.strip()))


def listen_for_wake(frames, recognizer, cfg, wake_words, tap_check=None,
                    noise_floor=None, verify_wake=None, idle_check=None, allow_weak=True):
    """Listen for the wake word on ONE frame iterator.

    `frames` is a single iterator/generator of 20ms PCM frames (NOT a
    stream object — calling .frames() per-frame would restart the
    generator every time; that bug hung the listener forever). Arms on
    sound onset, feeds Vosk a continuous stream, checks both partials
    and finalized text for the wake word, ends the session on room-relative
    quiet. verify_wake can recover an omitted or garbled name using better ASR;
    its transcript must contain the wake name before it can start a turn.
    Returns WakeResult on wake; False if tap_check fires.
    """
    a = cfg["audio"]
    silence_limit = a["silence_ms"] // MicStream.FRAME_MS
    arm_rms = a.get("wake_arm_rms", 400)
    # Acoustic confusions are grammar decoys, NOT local authorization.
    # Loud room chatter has already produced false wakes on 'the park'
    # and 'bar'; only a separate ASR hearing her actual name may approve.
    weak_min_peak = a.get("wake_weak_rms", 1400)
    # Vosk-small's realistic transcription set for the spoken wake word.
    _WEAK = {"bar", "bars", "bark", "barks", "bart", "barkley",
             "clark", "clarks", "stark", "starks", "park", "mark",
             "dark", "dock", "spar", "spork", "shark", "spec", "speck",
             "spock", "spa", "step", "steps"}
    _HEADS = {"spark", "sparky"} | {
        name.lower().split()[-1] for name in wake_words
        if name.strip() and name.lower().split()[-1] not in {"hey"}}
    # Wake mode is keyword spotting, not open dictation. This prevents loud,
    # overlapping family speech from turning "Spark" into arbitrary English.
    wake_grammar = sorted({name.lower().strip() for name in wake_words if name.strip()}
                          | {"spark", "sparky", "hey spark", "hey sparky"}
                          | _WEAK | {"the park", "a spark", "hey clark", "hey stark"})
    wake_grammar.append("[unk]")
    keyword_min_peak = a.get("wake_keyword_rms", 2500)
    keyword_max_ms = a.get("wake_keyword_max_ms", 1400)

    def _is_wake(tokens, peak=0, exact_only=False, speech_ms=0):
        if not tokens:
            return False
        short_enough = not speech_ms or speech_ms <= keyword_max_ms
        return (has_wake_name(" ".join(tokens), wake_words)
                and short_enough and peak >= keyword_min_peak)

    next_verify = 0.0
    vad = _speech_detector(cfg)
    energy = deque(maxlen=5)

    def meaningful(tokens, articles=False):
        fillers = {"ok", "okay", "hey"} | ({"a", "the"} if articles else set())
        return next((word for word in tokens if word not in fillers), "")

    def resolve(text):
        nonlocal next_verify
        tokens = text.lower().split()
        if _is_wake(tokens, peak, speech_ms=voiced_frames*20):
            # Retain original audio: constrained KWS only knows the name;
            # command ASR must still hear "go home" in the same breath.
            return WakeResult(text, b"".join(audio))
        # Vosk sometimes drops Spark entirely ('or how much battery...').
        # Check real speech even when its local transcript is empty. Only
        # an explicit wake name from the original audio can authorize a turn.
        # Ambiguous local names need near-field loudness AND independent
        # verification; never let the constrained decoder's 'bar'/'park'
        # win merely because background conversation is loud.
        head = meaningful(tokens, articles=True)
        ambiguous = bool(head and not has_wake_name(text, wake_words)
                         and (head in _WEAK or _near_wake(head, _HEADS)))
        verify_floor = weak_min_peak if ambiguous else a.get("start_rms", 900)
        if (verify_wake and voiced_frames >= 5 and peak >= verify_floor
                and time.monotonic() >= next_verify):
            next_verify = time.monotonic() + 1
            check_started = time.monotonic()
            verified = verify_wake(b"".join(audio))
            print(f"[ear] wake check {time.monotonic()-check_started:.2f}s: '{text}' -> '{verified}' "
                  f"(peak={peak} floor={floor:.0f} speech={voiced_frames*20}ms)",
                  file=sys.stderr, flush=True)
            if verified:
                if has_wake_name(verified, wake_words):
                    return WakeResult(verified)
                # Parakeet's known 'Bart' garble remains usable only when
                # Vosk independently heard a rarer near-name. Common 'bar',
                # 'park', 'mark' and 'dark' may never authorize fuzzily.
                v_tokens = re.findall(r"[\w']+", verified.lower())
                v_head = meaningful(v_tokens)
                if (allow_weak and head in {"bart", "barkley", "bark"}
                        and v_head in {"bart", "barkley"}
                        and peak >= a.get("wake_verify_fuzzy_rms", 4500)):
                    return WakeResult(verified)
            # A rejected earlier segment must not suppress a name in the
            # next completed segment inside the same one-second interval.
            next_verify = 0.0
        return None

    recognizer.begin(wake_grammar)
    preroll = deque(maxlen=10)
    idle_silence = 0
    while True:
        armed = False
        silence_run = 0
        peak = 0
        audio = deque(maxlen=400)
        floor = float(noise_floor()) if noise_floor else 500
        stop_rms = max(a["stop_rms"], floor * 1.3)
        onset_rms = max(arm_rms, stop_rms * 1.1)
        partial_candidate = None
        partial_count = 0
        frame_count = 0
        voiced_run = 0
        voiced_frames = 0
        while True:
            frame = next(frames, None)
            if frame is None:
                return False
            if tap_check is not None and tap_check():
                return False
            rms = _rms(frame)
            energy.append(rms)
            median_rms = sorted(energy)[len(energy)//2]
            voiced = (vad.is_speech(frame, a["sample_rate"])
                      and median_rms >= max(arm_rms, floor*1.8)) if vad else rms >= onset_rms
            idle_silence = 0 if voiced else idle_silence + 1
            if not armed:
                preroll.append(frame)
                onset = voiced and rms >= (arm_rms if vad else onset_rms)
                voiced_run = voiced_run + 1 if onset else 0
                if voiced_run >= (3 if vad else 1):
                    recognizer.begin(wake_grammar)  # constrained keyword decode
                    audio.extend(preroll)
                    final = None
                    for lead in preroll:
                        final = recognizer.feed(lead) or final
                    preroll.clear()
                    armed = True
                    silence_run = 0
                    peak = rms
                    voiced_frames = voiced_run
                    print(f"[ear] armed (rms={rms})", file=sys.stderr, flush=True)
                    if final:
                        result = resolve(final)
                        if result:
                            return result
                elif idle_check and idle_silence >= silence_limit and idle_check():
                    # A timer must never discard a wake already being decoded.
                    # Taps still interrupt immediately through tap_check above.
                    return False
                continue
            audio.append(frame)
            final = recognizer.feed(frame)  # continuous; returns text at endpoints
            peak = max(peak, rms)
            voiced_frames += int(voiced)
            if final:
                print(f"[ear] final: '{final}'", file=sys.stderr, flush=True)
                result = resolve(final)
                if result:
                    print(f"[ear] WAKE via final: '{final}'", file=sys.stderr, flush=True)
                    return result
                # not a wake: new utterance segment — loudness from the last
                # one must NOT authorize a later quiet weak-word hallucination
                peak = 0
                voiced_frames = 0
                recognizer.begin(wake_grammar)
                audio.clear()
                partial_candidate = None
                partial_count = 0
            frame_count += 1
            if not final and frame_count % 5 == 0:
                partial = recognizer.partial().lower()
                # Only strong names can wake early. Confusions still need a
                # completed short utterance and the existing loudness gate.
                if _is_wake(partial.split(), peak=peak, exact_only=True,
                            speech_ms=voiced_frames*20):
                    name = tuple(partial.split()[:2]) if partial.startswith("hey ") else partial.split()[0]
                    partial_count = partial_count + 1 if name == partial_candidate else 1
                    partial_candidate = name
                    if partial_count >= 3:
                        print(f"[ear] WAKE via partial: '{partial}'", file=sys.stderr, flush=True)
                        return WakeResult(partial, b"".join(audio))
                else:
                    partial_candidate = None
                    partial_count = 0
            quiet = not voiced if vad else rms < stop_rms
            if quiet:
                silence_run += 1
            else:
                silence_run = 0
            if silence_run >= silence_limit or len(audio) >= 400:
                # "Spark!" followed by a pause can arrive only in the tail.
                tail = recognizer.finish().strip().lower()
                if tail:
                    print(f"[ear] session tail: '{tail}'", file=sys.stderr, flush=True)
                result = resolve(tail)
                if result:
                    print(f"[ear] WAKE via tail: '{result.text}'", file=sys.stderr, flush=True)
                    return result
                break
