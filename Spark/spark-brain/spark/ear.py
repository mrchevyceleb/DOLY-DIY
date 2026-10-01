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
        self._mute = (0.0, 0.0)  # capture window of our own playback

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

    def mute(self, seconds):
        """Blank audio captured during our own playback from now on. The
        speaker sits inches from the mic: the wake chirp recorded as the
        command ('Oh.') and its endpoint closed before the user spoke."""
        with self._ready:
            now = time.monotonic()
            self._mute = (now, now + seconds)

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
            stamp, chunk = self._queue.popleft()
            # stamp = end of the chunk's capture: blank any overlap.
            if self._mute[0] <= stamp <= self._mute[1] + self.FRAME_MS/1000:
                return bytes(len(chunk))
            return chunk

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
    command: str = ""  # verified speech that is all command (name heard earlier)
    clip_pcm: bytes = b""  # the verified clip, to re-decode with its continuation


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


# Strong name-family tokens: speech that Vosk's constrained wake grammar
# decodes to these sounds like her name being said, not common words.
# Not 'stark': it is how Vosk hears 'start', and every stark wake in the logs
# was background talk ('Okay, start the battery on fire.'). A stark segment
# still wakes her when the verifier hears her name.
_STRONG_NAME_FAMILY = {"spark", "sparky", "spar", "spork", "spock", "spec",
                       "speck", "bark", "barks", "barkley", "bart"}
# How the verifier (Parakeet) renders a spoken 'Spark' when it drops the soft
# 's': 'Park, what's the weather today?' (peak 2842), 'Bark, ...' (3887).
_VERIFIER_NAMES = {"park", "bark", "barks", "bart", "barkley", "sparks", "fark", "farks"}


class RoomTalk:
    """Speech the verifier heard without her name: a video, a call, people
    talking to each other. While the room is busy only her actual name (or a
    vocative 'Park,') wakes her, and her replies stop chaining on the talk.
    Live: a YouTube review ('only loses on quality.') woke her via 'barkley'
    after seven unaddressed segments in 30 s."""

    def __init__(self, window_s=30.0, segments=3):
        self.window_s, self.segments = window_s, segments
        self._heard = deque(maxlen=32)

    def note(self, now=None):
        self._heard.append(time.monotonic() if now is None else now)

    def count(self, now=None):
        now = time.monotonic() if now is None else now
        return sum(1 for t in self._heard if now - t <= self.window_s)

    def busy(self, now=None):
        return self.count(now) >= self.segments


def room_talk(cfg):
    a = cfg.get("audio", {})
    return RoomTalk(a.get("room_busy_window_s", 30), a.get("room_busy_segments", 3))


def requires_hey(wake_words):
    phrases = [name.lower().split() for name in wake_words if name.strip()]
    return bool(phrases) and all(len(words) > 1 and words[0] == "hey" for words in phrases)


def has_wake_name(text, wake_words):
    tokens = re.findall(r"[\w']+", text.lower())
    if tokens and tokens[0] in {"okay", "ok"}:
        tokens = tokens[1:]
    if not tokens:
        return False
    return ((not requires_hey(wake_words) and tokens[0] in {"spark", "sparky"}) or any(
        tokens[:len(name.split())] == name.lower().split() for name in wake_words if name.strip()))


_NAME = r"(?:(?:hey|okay|ok)[,\s]+)?spark(?:y)?"
# 'Spark, what's...' opening a later sentence, or 'Spark.' said on its own.
_SENTENCE_NAME_RE = re.compile(rf"[.!?]\s+(?P<wake>{_NAME})\b\s*(?:[,.!?:;-]+\s*(?P<command>.*)|$)", re.I | re.S)
# '..., Spark?' closing a sentence. The comma marks it as said TO her; talk
# ABOUT her ('have you guys met Spark?') has none.
_TRAILING_NAME_RE = re.compile(rf"^(?P<command>.*?\w)\s*,\s*(?P<wake>{_NAME})\s*(?P<punct>[.!?]*)\s*$", re.I | re.S)
# 'This is my robot, Spark.' introduces her on a call; it is not a command.
_DESCRIBES_HER = {"robot", "bot", "assistant", "friend", "buddy", "pet", "girl", "dog",
                  "companion", "helper", "sidekick", "named", "called", "name", "meet"}
_SENTENCE_END_RE = re.compile(r"(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bSt\.)(?<!\bMrs\.)"
                              r"(?<=[.!?])\s+")


def addressed_command(text, wake_words=None):
    """The command when her name is said TO her somewhere other than the
    start; None when it is not. '' means the name alone: listen for more.
    Live, on a call: 'What's the weather today, Spark?' and 'He's going
    well. Spark, what's the weather today?' were both dropped as room talk."""
    text = (text or "").strip()
    trailing = _TRAILING_NAME_RE.match(text)
    if trailing and (wake_words is None or has_wake_name(trailing.group("wake"), wake_words)):
        last = _SENTENCE_END_RE.split(trailing.group("command"))[-1].strip()
        if last.lower().split()[-1] in _DESCRIBES_HER:
            return None
        return f"{last}{(trailing.group('punct') or '.')[:1]}"
    opening = _SENTENCE_NAME_RE.search(text)
    if opening and (wake_words is None or has_wake_name(opening.group("wake"), wake_words)):
        return (opening.group("command") or "").strip()
    return None


def listen_for_wake(frames, recognizer, cfg, wake_words, tap_check=None,
                    noise_floor=None, verify_wake=None, idle_check=None, allow_weak=True,
                    room=None):
    """Listen for the wake word on ONE frame iterator.

    `frames` is a single iterator/generator of 20ms PCM frames (NOT a
    stream object — calling .frames() per-frame would restart the
    generator every time; that bug hung the listener forever). Arms on
    sound onset, feeds Vosk a continuous stream, checks both partials
    and finalized text for the wake word, ends the session on room-relative
    quiet. verify_wake can recover an omitted or garbled name using better ASR.
    Hey-only phrases disable bare-name and fuzzy recovery. A clear local
    phrase still works when the verification service is silent or unavailable.
    Returns WakeResult on wake; False if tap_check fires.
    """
    a = cfg["audio"]
    require_hey = requires_hey(wake_words)
    allow_weak = allow_weak and not require_hey
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
    last_family_final_at = 0.0   # a strong name-family token finalized recently
    vad = _speech_detector(cfg)
    energy = deque(maxlen=5)

    room = room or room_talk(cfg)

    def meaningful(tokens, articles=False):
        fillers = {"ok", "okay", "hey"} | ({"a", "the"} if articles else set())
        return next((word for word in tokens if word not in fillers), "")

    def vocative(verified):
        """The command after the verifier's own rendering of her name said TO
        her ('Park, what's...'); the comma separates it from 'Park the car'."""
        v_head = meaningful(re.findall(r"[\w']+", verified.lower()))
        if not v_head or v_head.removesuffix("'s") not in _VERIFIER_NAMES:
            return None
        named = re.search(rf"\b{re.escape(v_head)}\b\s*[,.!?:;-]+\s*(\w.*)", verified, re.I)
        return named.group(1).strip() if named else None

    def named_elsewhere(verified, clip):
        """Her name said to her mid-sentence or at the end (see
        addressed_command). Busy rooms included: it is her actual name."""
        command = addressed_command(verified, wake_words)
        if command is None:
            return None
        if not command:
            return WakeResult("Spark", clip_pcm=clip)  # name alone: listen on
        return WakeResult(verified, command=command, clip_pcm=clip)

    def real_talk(verified):
        return len([w for w in re.findall(r"[\w']+", (verified or "").lower())
                    if w not in {"ok", "okay", "hey", "um", "uh", "oh"}]) >= 2

    def confirm_exact(text, clip, final=True):
        """Vosk's grammar knows little besides her name, so it can force loud
        talk into an exact 'sparky' (live: 'sparky' -> 'Fucking background.').
        The verifier vetoes the hit when it heard real words without her
        name; when it is silent or unavailable the local hit stands."""
        if not verify_wake:
            return WakeResult(text, clip)
        started = time.monotonic()
        verified = verify_wake(clip, keep=True) or ""
        print(f"[ear] exact check {time.monotonic()-started:.2f}s: '{text}' -> '{verified}'",
              file=sys.stderr, flush=True)
        if has_wake_name(verified, wake_words):
            return WakeResult(verified, clip_pcm=clip)
        # Live: both deliberate 'Hey Spark' attempts became 'A spark.' on
        # Parakeet. The local decoder must have the complete configured
        # phrase; only this name-only 'a'/'hey' disagreement is recoverable.
        local_words = text.lower().split()
        if (require_hey and local_words in [name.lower().split() for name in wake_words]
                and re.findall(r"[\w']+", verified.lower()) == ["a", *local_words[1:]]):
            if not final:
                return False  # defer this ambiguity until the full utterance
            print(f"[ear] wake phrase corroborated: '{text}' -> '{verified}'",
                  file=sys.stderr, flush=True)
            return WakeResult(text, clip_pcm=clip)
        addressed = named_elsewhere(verified, clip)
        if addressed:
            return addressed
        command = vocative(verified) if not require_hey else None
        if command:
            return WakeResult(verified, command=command, clip_pcm=clip)
        if real_talk(verified) or (require_hey and verified.strip()):
            room.note()
            print(f"[ear] exact '{text}' vetoed: no name in '{verified}'",
                  file=sys.stderr, flush=True)
            return None
        return WakeResult(text, clip)

    def resolve(text):
        nonlocal next_verify
        nonlocal last_family_final_at
        tokens = text.lower().split()
        if _is_wake(tokens, peak, speech_ms=voiced_frames*20):
            if partial_vetoed:
                return None  # already verified and noted as room talk
            # Retain original audio: constrained KWS only knows the name;
            # command ASR must still hear "go home" in the same breath.
            return confirm_exact(text, b"".join(audio))
        # Vosk sometimes drops Spark entirely ('or how much battery...').
        # Check real speech even when its local transcript is empty. Only
        # an explicit wake name from the original audio can authorize a turn.
        # Ambiguous local names need near-field loudness AND independent
        # verification; never let the constrained decoder's 'bar'/'park'
        # win merely because background conversation is loud.
        head = meaningful(tokens, articles=True)
        if head in _STRONG_NAME_FAMILY:
            # Her name was (garbled but) spoken: the command often lands in
            # the NEXT segment after the endpoint splits the utterance.
            last_family_final_at = time.monotonic()
        ambiguous = bool(head and not has_wake_name(text, wake_words)
                         and (head in _WEAK or _near_wake(head, _HEADS)))
        # Every name-like segment is checked at speaking level: only the
        # verifier hearing her name (or a vocative 'Park,') can wake her, so
        # a loud floor only made her deaf. Live misses: 'barkley' at 2715,
        # 'bark [unk]' at 2601 and 'bar' at 2943, all under the old 4000.
        family_head = head in _STRONG_NAME_FAMILY
        # Quiet speech the verifier heard WITHOUT her name never rides the
        # family hint, even when Vosk's grammar said 'sparky' exactly: live
        # 'sparky' -> 'More softly.' (1717) and 'spark' -> 'That's fine.' were
        # background talk.
        quiet_family = family_head and peak < weak_min_peak
        verify_floor = a.get("start_rms", 900)
        if (family_head and not (voiced_frames >= 5 and peak >= verify_floor)):
            print(f"[ear] name-like '{text}' not checked (peak={peak} "
                  f"speech={voiced_frames*20}ms)", file=sys.stderr, flush=True)
        if (verify_wake and voiced_frames >= 5 and peak >= verify_floor
                and time.monotonic() >= next_verify):
            next_verify = time.monotonic() + 1
            check_started = time.monotonic()
            clip = b"".join(audio)
            # A name-like segment may still wake her on the family hint below
            # even when the verifier drops the name: keep the speech queued
            # during the check, or the middle of the question is lost
            # ('what's the address of Shepherd ... Allentown').
            name_like = (family_head or head in _WEAK
                         or (last_family_final_at > 0
                             and time.monotonic() - last_family_final_at < 2.5))
            verified = verify_wake(clip, keep=True) if name_like else verify_wake(clip)
            print(f"[ear] wake check {time.monotonic()-check_started:.2f}s: '{text}' -> '{verified}' "
                  f"(peak={peak} floor={floor:.0f} speech={voiced_frames*20}ms)",
                  file=sys.stderr, flush=True)
            if verified:
                if has_wake_name(verified, wake_words):
                    return WakeResult(verified, clip_pcm=clip)
                addressed = named_elsewhere(verified, clip)
                if addressed:
                    return addressed
                v_tokens = re.findall(r"[\w']+", verified.lower())
                v_head = meaningful(v_tokens)
                # Vosk heard her name family AND the verifier heard its own
                # rendering of it said TO her ('Park, what's...').
                # Vosk's weak 'park' counts too when the verifier heard a
                # vocative name ('park [unk]' -> 'Fark, go home.' at 9415):
                # every such pair in the logs was Matt talking to her.
                command = vocative(verified)
                if (allow_weak and command
                        and (family_head or (head in _WEAK and peak >= weak_min_peak))):
                    return WakeResult(verified, command=command, clip_pcm=clip)
                # Parakeet's known 'Bart' garble remains usable only when
                # Vosk independently heard a rarer near-name. Common 'bar',
                # 'park', 'mark' and 'dark' may never authorize fuzzily.
                if (allow_weak and head in {"bart", "barkley", "bark"}
                        and v_head in {"bart", "barkley"}
                        and peak >= a.get("wake_verify_fuzzy_rms", 4500)):
                    return WakeResult(verified, clip_pcm=clip)
                # A muffled mic can strip the name from even the verified
                # transcript (Vosk: 'barks [unk]', whisper: 'Ten seconds.').
                # When Vosk independently heard a STRONG name-family token
                # and the verifier confirms real command speech, accept at
                # normal arming loudness. A stray wake just opens a short
                # listening window; total deafness is the worse failure.
                family_hint = (head in _STRONG_NAME_FAMILY
                               or (last_family_final_at > 0
                                   and time.monotonic() - last_family_final_at < 2.5))
                # A quiet name-like segment needs the verifier to hear her
                # actual name (above); only loud ones or a later command
                # segment may ride the looser family hint. So does a busy
                # room: a video or a call forces its talk into 'barkley'.
                busy = room.busy()
                if busy and family_hint:
                    print(f"[ear] room busy ({room.count()} unaddressed): "
                          f"'{verified}' needs her name", file=sys.stderr, flush=True)
                if (allow_weak and family_hint and not quiet_family and not busy
                        and len(meaningful(v_tokens)) >= 2
                        and peak >= a.get("start_rms", 900)):
                    # Name in the previous segment ('barkley', then
                    # 'Weather today.'): nothing here is her name, so
                    # stripping the first word turned it into 'today.'
                    # Exact aliases only: fuzzy matching would strip a real
                    # command word ('start ...') as if it were her name.
                    named = v_head.removesuffix("'s") in _STRONG_NAME_FAMILY | {"stark", "starks"}
                    return WakeResult(verified, command="" if named else verified,
                                      clip_pcm=clip)
                if real_talk(verified):
                    room.note()  # real talk that was not for her
            elif (allow_weak and not partial_vetoed
                  and any(name.strip().lower() in {"spark", "sparky", "hey spark", "hey sparky"}
                          for name in wake_words)
                  and head in {"bark", "barks", "barkley", "bart"}
                  and [w for w in tokens if w not in {"ok", "okay", "hey"}] == [head]
                  and peak >= max(weak_min_peak, a.get("wake_verify_fuzzy_rms", 4500))
                  and 100 <= voiced_frames * 20 <= keyword_max_ms
                  and len(clip) / (a["sample_rate"] * 2) <= keyword_max_ms / 1000 + .7):
                # Live: a 600ms 'Spark!' became 'bark' at RMS 8303, while
                # Parakeet returned nothing. Its later weather question then
                # had no name and was ignored. Open the normal listening
                # window, preserving queued speech, just as an exact name
                # does on a silent verifier. Common decoys (bar/park/stark),
                # long clips and ANY nonempty verifier transcript still fail.
                print(f"[ear] name-only '{text}' accepted: verifier silent",
                      file=sys.stderr, flush=True)
                return WakeResult("Spark", clip_pcm=clip)
            # A rejected earlier segment must not suppress a name in the
            # next completed segment inside the same one-second interval.
            next_verify = 0.0
        return None

    recognizer.begin(wake_grammar)
    # 600ms pre-roll: a soft 'Spark,' that sits below the arming gate must
    # still ride into the clip when the loud command after it triggers the
    # onset — live miss: '[unk]' -> 'Alexa, bright lights.' (520ms speech,
    # the name was never captured, so she slept through the govee command).
    preroll = deque(maxlen=30)
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
        partial_vetoed = False  # the verifier already heard this segment's talk
        partial_deferred = False  # a/hey disagreement needs the completed clip
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
                partial_vetoed = False
                partial_deferred = False
            frame_count += 1
            if not final and frame_count % 5 == 0:
                partial = recognizer.partial().lower()
                # Only strong names can wake early. Confusions still need a
                # completed short utterance and the existing loudness gate.
                if not partial_vetoed and not partial_deferred and _is_wake(partial.split(), peak=peak, exact_only=True,
                                                   speech_ms=voiced_frames*20):
                    name = tuple(partial.split()[:2]) if partial.startswith("hey ") else partial.split()[0]
                    partial_count = partial_count + 1 if name == partial_candidate else 1
                    partial_candidate = name
                    if partial_count >= 3:
                        result = confirm_exact(partial, b"".join(audio), final=False)
                        if result:
                            print(f"[ear] WAKE via partial: '{partial}'", file=sys.stderr, flush=True)
                            return result
                        partial_vetoed = result is None
                        partial_deferred = result is False
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
