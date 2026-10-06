"""Continuous echo removal and a cancellable spoken turn, without owning SDKs.

The microphone thread runs only DSP/VAD. A separate, bounded ASR check
confirms an interruption; robot actions still run on the main voice thread.
"""
from collections import deque
import contextlib
import ctypes as C
import ctypes.util
import difflib
import queue
import re
import sys
import socket
import threading
import time

from .voicefx import audioop


class TurnInterrupted(Exception):
    pass


@contextlib.contextmanager
def closing_on_cancel(response, cancel):
    """Wake a blocking urllib read without waiting for the network timeout."""
    done = threading.Event()
    try:
        sock = response.fp.raw._sock
    except AttributeError:  # in-memory/test responses
        sock = None

    def watch():
        while not done.wait(.03):
            if cancel.is_set():
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                return

    threading.Thread(target=watch, daemon=True).start()
    try:
        yield
    finally:
        done.set()


def interruptible(source, cancel):
    """Keep network reads off the voice thread so a tap can hand back the turn."""
    items = queue.Queue(maxsize=2)
    done = threading.Event()

    def produce():
        try:
            for item in source:
                while not done.is_set() and not cancel.is_set():
                    try:
                        items.put((True, item), timeout=.05)
                        break
                    except queue.Full:
                        pass
                if done.is_set() or cancel.is_set():
                    break
        except Exception as error:
            while not done.is_set() and not cancel.is_set():
                try:
                    items.put((False, error), timeout=.05)
                    break
                except queue.Full:
                    pass
        finally:
            close = getattr(source, "close", None)
            if close:
                close()
            done.set()

    threading.Thread(target=produce, daemon=True).start()
    try:
        while True:
            if cancel.is_set():
                raise TurnInterrupted()
            try:
                ok, item = items.get(timeout=.03)
            except queue.Empty:
                if done.is_set():
                    return
                continue
            if not ok:
                raise item
            yield item
    finally:
        done.set()


class EchoCanceller:
    """Speex adaptive filter, with a timed reference AFTER voice FX and gain."""
    def __init__(self, rate=16000, tail_ms=300, capture_latency_ms=40):
        if audioop is None:
            raise RuntimeError("PCM resampling unavailable")
        self.rate, self.samples = rate, rate // 50
        self.frame_bytes = self.samples * 2
        self.capture_latency = capture_latency_ms / 1000
        self.lock = threading.Lock()
        self.references = deque(maxlen=250)
        self.resample_state = None
        self.lib = C.CDLL(ctypes.util.find_library("speexdsp") or "libspeexdsp.so.1")
        lib = self.lib
        sample_ptr = C.POINTER(C.c_int16)
        lib.speex_echo_state_init.argtypes = [C.c_int, C.c_int]
        lib.speex_echo_state_init.restype = C.c_void_p
        lib.speex_echo_ctl.argtypes = [C.c_void_p, C.c_int, C.c_void_p]
        lib.speex_echo_cancellation.argtypes = [C.c_void_p, sample_ptr, sample_ptr, sample_ptr]
        lib.speex_echo_state_destroy.argtypes = [C.c_void_p]
        lib.speex_preprocess_state_init.argtypes = [C.c_int, C.c_int]
        lib.speex_preprocess_state_init.restype = C.c_void_p
        lib.speex_preprocess_ctl.argtypes = [C.c_void_p, C.c_int, C.c_void_p]
        lib.speex_preprocess_run.argtypes = [C.c_void_p, sample_ptr]
        lib.speex_preprocess_state_destroy.argtypes = [C.c_void_p]
        self.state = lib.speex_echo_state_init(self.samples, rate * tail_ms // 1000)
        if not self.state:
            raise RuntimeError("cannot allocate echo canceller")
        sample_rate = C.c_int(rate)
        lib.speex_echo_ctl(self.state, 24, C.byref(sample_rate))
        self.preprocess = lib.speex_preprocess_state_init(self.samples, rate)
        if not self.preprocess:
            lib.speex_echo_state_destroy(self.state)
            self.state = None
            raise RuntimeError("cannot allocate residual echo suppressor")
        lib.speex_preprocess_ctl(self.preprocess, 24, C.c_void_p(self.state))
        for request, number in ((0, 1), (18, -12), (20, -35), (22, -8)):
            value = C.c_int(number)
            lib.speex_preprocess_ctl(self.preprocess, request, C.byref(value))
        self.array = C.c_int16 * self.samples
        self.last_render = 0.0
        self.last_reference = bytes(self.frame_bytes)
        self.rendered_frames = 0

    def render(self, pcm, rate, when):
        with self.lock:
            converted, self.resample_state = audioop.ratecv(
                pcm, 2, 1, rate, self.rate, self.resample_state)
            # PCMPlayer delivers at most one 20ms frame. Pad the one-sample
            # resampler startup and final short frame, without shifting time.
            self.references.append((when, converted[:self.frame_bytes].ljust(self.frame_bytes, b"\0")))
            self.rendered_frames += 1
            self.last_render = when + len(converted) / (2 * self.rate) + .25

    def process(self, pcm, stamp):
        target = stamp - self.capture_latency
        reference = bytes(self.frame_bytes)
        with self.lock:
            # The caller paces writes. Never pair capture with future sound.
            while self.references and self.references[0][0] <= target + .01:
                when, reference = self.references.popleft()
                if target - when > .04:
                    reference = bytes(self.frame_bytes)
            if self.state:
                self.last_reference = reference
                out = self.array()
                self.lib.speex_echo_cancellation(self.state,
                    self.array.from_buffer_copy(pcm), self.array.from_buffer_copy(reference), out)
                self.lib.speex_preprocess_run(self.preprocess, out)
                return bytes(out)
        return pcm

    def close(self):
        with self.lock:
            if self.state:
                self.lib.speex_preprocess_state_destroy(self.preprocess)
                self.lib.speex_echo_state_destroy(self.state)
                self.state = None


def is_stop_request(text):
    normalized = " ".join(re.findall(r"[a-z']+", text.casefold()))
    normalized = re.sub(r"^(?:(?:please|can you|could you|would you|will you) )+", "", normalized)
    return bool(re.fullmatch(
        r"(?:stop(?: (?:it|that|talking|moving|playing|dancing|singing|everything|the music|the dance|all))?"
        r"|wait(?: (?:a|one) (?:second|moment|minute))?|hold on|pause)(?: now| please)*",
        normalized))


def _echo_words(text):
    from .volume import _ONES, _TENS
    tens = {v: k for k, v in _TENS.items()}

    def number(match):
        value = int(match[0])
        if value >= 1000:
            return match[0]  # leave large identifiers exact
        if value >= 100:
            head = _ONES[value // 100] + " hundred "
            value %= 100
        else:
            head = ""
        if value >= 20:
            return head + tens[value // 10 * 10] + (" " + _ONES[value % 10] if value % 10 else "")
        return head + (_ONES[value] if value or not head else "")

    text = text.casefold().replace("°", " degrees ")
    text = re.sub(r"\b(?:it's|it’s)\b", "it is", text)
    text = re.sub(r"\b\d{1,3}\b", number, text)
    return re.findall(r"[a-z']+|\d+", text)


def is_echo(text, expected):
    """Do not let a plausible transcription of her own reply interrupt it."""
    words = _echo_words(text)
    spoken = _echo_words(expected)
    if is_stop_request(text):
        return False  # a verified safety interruption always has priority
    if not words or not spoken:
        return not words
    joined = " ".join(words)
    if joined in " ".join(spoken):
        return True
    # Approximate contiguous phrases tolerate ASR errors without treating
    # every shared word (e.g. 'lights') as echo.
    for n in range(max(1, len(words)-1), len(words)+2):
        for i in range(max(0, len(spoken)-n+1)):
            if difflib.SequenceMatcher(None, joined, " ".join(spoken[i:i+n])).ratio() >= .78:
                return True
    return False


class DuplexAudio:
    def __init__(self, cfg, transcribe):
        from .ear import _speech_detector
        a = cfg["audio"]
        self.rate = a["sample_rate"]
        self.settings = cfg.get("conversation", {})
        self.cancel = threading.Event()
        self.pause = threading.Event()
        self.lock = threading.RLock()
        self.active = False
        self.pending = False
        self.epoch = 0
        self.expected = ""
        self.transcribe = transcribe
        self.vad = _speech_detector(cfg)
        self.floor = 500
        self.history = deque(maxlen=300)  # onset plus the bounded ASR wait
        self.verification_context = deque(maxlen=100)  # two seconds, never handed off
        self.raw_verification_context = deque(maxlen=100)
        self.voiced = 0
        self.voiced_run = 0
        self.pause_run = 0
        self.quiet = 0
        self.checking = False
        self.last_check = 0.0
        self.blocked_until = 0.0
        self.echo = None
        if a.get("echo_cancellation", True):
            try:
                self.echo = EchoCanceller(self.rate, a.get("echo_tail_ms", 300),
                                           a.get("capture_latency_ms", 40))
            except Exception as error:
                self.log(f"AEC unavailable: {error}; voice interruption disabled, tap still works")
        self.log(f"AEC={'Speex' if self.echo else 'off'}, verified voice interruption="
                 f"{bool(self.echo and self.vad and self.settings.get('barge_in', True))}")

    @staticmethod
    def log(text):
        print(f"[duplex] {text}", file=sys.stderr, flush=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def begin(self, floor=500):
        with self.lock:
            self.epoch += 1
            self.active = True
            self.cancel = threading.Event()  # old workers keep their cancelled event
            self.pause = threading.Event()
            self.pending = False
            self.expected = ""
            self.history.clear()
            self.verification_context.clear()
            self.raw_verification_context.clear()
            self.voiced = 0
            self.voiced_run = 0
            self.pause_run = 0
            self.quiet = 0
            self.checking = False
            self.floor = floor

    def end(self):
        with self.lock:
            self.active = False
            if not self.checking:
                self.pause.clear()

    def interrupt(self, reason="tap"):
        with self.lock:
            if not (self.active or self.checking) or self.cancel.is_set():
                return
            self.pending = True
            if reason == "tap":
                self.history.clear()  # open a fresh listen, never replay robot audio
            self.cancel.set()
            self.log(f"interrupted: {reason}")

    def take_pending(self):
        with self.lock:
            if not self.pending:
                return None
            self.pending = False
            frames = list(self.history)
            self.history.clear()
            return (b"".join(pcm for _, pcm in frames), frames[-1][0] if frames else time.monotonic())

    @contextlib.contextmanager
    def output(self, text):
        owns_turn = not self.active
        if owns_turn:
            self.begin(self.floor)
        with self.lock:
            self.expected += " " + text
        try:
            if self.cancel.is_set():
                raise TurnInterrupted()
            yield self.cancel
        finally:
            if owns_turn:
                self.end()

    def render(self, pcm, rate, when):
        if self.echo:
            self.echo.render(pcm, rate, when)

    def block_stock_sound(self, seconds):
        # Music/SFX have their own stock stop/motion handling. They are not
        # a speech reference and must not trigger conversational barge-in.
        self.blocked_until = max(self.blocked_until, time.monotonic()) + seconds + .25

    def capture(self, pcm, stamp):
        raw_pcm = pcm
        raw_level = audioop.rms(pcm, 2)
        if self.echo:
            pcm = self.echo.process(pcm, stamp)
        with self.lock:
            if self.pending:
                self.history.append((stamp, pcm))
                return pcm
            if self.checking and not self.active:
                self.history.append((stamp, pcm))
                return pcm
            if not (self.active and self.echo and self.vad
                    and stamp >= self.blocked_until
                    and self.settings.get("barge_in", True)):
                self.history.clear()
                self.verification_context.clear()
                self.raw_verification_context.clear()
                self.voiced = 0
                self.voiced_run = 0
                self.pause_run = 0
                self.pause.clear()
                return pcm
            self.history.append((stamp, pcm))
            self.verification_context.append(pcm)
            self.raw_verification_context.append(raw_pcm)
            if not self.checking and self.voiced == 0:
                while len(self.history) > 10:
                    self.history.popleft()
            level = audioop.rms(pcm, 2)
            reference_active = stamp < self.echo.last_render
            # Do not repeatedly pause the initial speaker frames before the
            # adaptive filter can learn the acoustic path (four seconds of
            # rendered audio, once after startup). Thereafter residual
            # echo is much smaller than the raw mic; genuine double-talk must
            # contribute at least half its amplitude to the cleaned signal.
            settled = self.echo.rendered_frames >= 200 or not reference_active
            near_end = not reference_active or level >= raw_level * .50
            speech = settled and near_end and level >= max(self.settings.get("barge_in_min_rms", 600), self.floor * 1.8) and self.vad.is_speech(pcm, self.rate)
            if speech:
                self.voiced += 1
                self.voiced_run += 1
                self.quiet = 0
            else:
                self.quiet += 1
                self.voiced_run = 0
            strong_voice = speech and (not reference_active or level >= raw_level * .75)
            self.pause_run = self.pause_run + 1 if strong_voice else 0
            # Pause after 160ms of coherent near-end voice. This is reversible:
            # echo/noise resumes the same buffered PCM, never executes actions.
            if (self.settings.get('barge_in_early_pause', False)
                    and self.pause_run >= 8 and stamp - self.last_check >= .75):
                self.pause.set()
            ready = self.voiced >= 25 or (self.voiced >= 6 and self.quiet >= 5)
            # Keep capture alive through the pause (and during brain thinking),
            # even after the final speaker reference has drained.
            if ready and not self.checking and stamp - self.last_check >= .75:
                self.checking = True
                self.last_check = stamp
                # ASR needs context to identify echo, while handoff retains
                # only the user's onset, without preceding speaker speech.
                clip = b"".join(self.verification_context)
                epoch, expected = self.epoch, self.expected
                raw_clip = b"".join(self.raw_verification_context) if expected else None
                threading.Thread(target=self._verify, args=(clip, epoch, expected, raw_clip), daemon=True).start()
            elif self.quiet >= 20 and not self.checking:
                self.voiced = 0
                self.pause.clear()
        return pcm

    def _verify(self, clip, epoch, expected, raw_clip=None):
        try:
            started = time.monotonic()
            raw_done, raw_result = threading.Event(), {}
            if raw_clip is not None:
                def check_raw():
                    try:
                        raw_result['text'] = self.transcribe(raw_clip).strip()
                    except Exception:
                        pass  # unavailable corroboration cannot authorize a stop
                    finally:
                        raw_done.set()
                threading.Thread(target=check_raw, daemon=True).start()
            # Production callback is WhisperASR.transcribe_wake_pcm: curl
            # deadline 3.5s, subprocess hard timeout 4.5s, no local fallback.
            text = self.transcribe(clip).strip()
            words = re.findall(r"[a-z']+", text.casefold())
            valid = is_stop_request(text) or len(words) >= 2 or words in (["spark"], ["sparky"])
            corroborated = True
            if raw_clip is not None:
                # Check the same audio before echo suppression. Distorted
                # residuals hallucinated "Wait" during her own weather line.
                # Both ASR reads run concurrently within one 4.5s budget.
                raw_done.wait(max(0, 4.5 - (time.monotonic() - started)))
                raw_text = raw_result.get('text', '')
                raw_words = re.findall(r"[a-z']+", raw_text.casefold())
                # During double-talk raw ASR can include her preceding words:
                # "Hmm. Stop talking." still corroborates the user's stop.
                # A quoted stop inside her own reply must not authorize it.
                with self.lock:
                    raw_echo = is_echo(raw_text, self.expected or expected)
                raw_stop = is_stop_request(raw_text) or (not raw_echo and any(
                    is_stop_request(" ".join(raw_words[-n:]))
                    for n in range(1, min(12, len(raw_words)) + 1)))
                corroborated = bool(raw_text) and (
                    raw_stop if is_stop_request(text) else is_echo(text, raw_text))
            with self.lock:
                if self.epoch != epoch or not (self.active or self.checking):
                    return
                # Another streamed clause can start while ASR is checking.
                # Include its reference rather than the stale launch snapshot.
                echo = is_echo(text, self.expected or expected)
                if valid:
                    self.log(f"voice check: {text!r}; echo={echo}; mic_agrees={corroborated}")
                if valid and not echo and corroborated:
                    # Preserve onset through the ASR wait; hand it back to
                    # the normal full-command decoder, never execute here.
                    self.interrupt("speech")
                else:
                    self.voiced = 0
                    self.history.clear()
        except Exception as error:
            self.log(f"interruption check failed: {error}")
        finally:
            with self.lock:
                if self.epoch == epoch:
                    self.checking = False
                    self.voiced = 0
                    self.voiced_run = 0
                    self.pause_run = 0
                    self.pause.clear()

    def close(self):
        self.cancel.set()
        self.pause.clear()
        self.end()
        if self.echo:
            self.echo.close()
