"""Bounded local PCM transport and continuous ALSA playback.

SPK1 + uint32 rate, then uint32 byte length + mono PCM16 per packet.
A zero length explicitly signals successful completion; EOF is failure.
"""
import struct
import subprocess
import time
import threading
import queue
import re
import urllib.parse
import urllib.request
import urllib.error

from .voicefx import PCMEffects, audioop, speech_gain
from .duplex import TurnInterrupted, interruptible, closing_on_cancel


class StreamSpeechError(RuntimeError):
    def __init__(self, message, played=False):
        super().__init__(message)
        self.played = played


def _read_exact(response, size):
    data = response.read(size)
    if len(data) != size:
        raise ValueError("voice stream ended without completion")
    return data


def buffered_packets(source, rate, milliseconds, cancel):
    """Reserve a bounded start of speech to absorb bursty synthesis arrivals."""
    target = int(rate * 2 * max(0, min(3000, float(milliseconds))) / 1000)
    pending, size = [], 0
    try:
        for packet in source:
            if cancel.is_set():
                raise TurnInterrupted()
            pending.append(packet)
            size += len(packet)
            if size >= target:
                break
        for packet in pending:
            if cancel.is_set():
                raise TurnInterrupted()
            yield packet
        yield from source
    finally:
        source.close()


def play(text, cfg, on_audio, cancel=None, on_reference=None, pause=None, prepared=None):
    """Return first packet time, or fail with whether playback has started.

    The SDK remains initialized for tricks/sound effects. aplay uses the
    robot's default dmix ALSA route, which shares that speaker safely.
    """
    url = cfg["server_url"].rstrip("/") + "/stream?" + urllib.parse.urlencode(
        {"voice": cfg["voice_name"]})
    request = urllib.request.Request(url, data=text.encode("utf-8"))
    timeout = max(1, min(15, float(cfg.get("stream_timeout_s", 8))))
    started = time.perf_counter()
    total = 0
    gain = max(0, min(100, float(cfg.get("stream_volume", 90)))) / 100
    cancel = cancel if cancel is not None else threading.Event()

    def packets():
        nonlocal total
        with urllib.request.urlopen(request, timeout=timeout) as response, closing_on_cancel(response, cancel):
            if response.headers.get_content_type() != "application/x-spark-pcm":
                raise ValueError("unexpected voice stream format")
            if _read_exact(response, 8) != b"SPK1" + struct.pack("<I", 24000):
                raise ValueError("unsupported voice stream header")
            while not cancel.is_set():
                size = struct.unpack("<I", _read_exact(response, 4))[0]
                if size == 0:
                    return
                total += size
                if size % 2 or size > 128 * 1024 or total > 4 * 1024 * 1024:
                    raise ValueError("voice stream exceeds PCM limits")
                if time.perf_counter() - started > 60:
                    raise ValueError("voice stream exceeded time limit")
                yield _read_exact(response, size)

    incoming = prepared.packets(cancel) if prepared is not None else interruptible(packets(), cancel)
    incoming = buffered_packets(incoming, 24000, cfg.get("stream_prebuffer_ms", 960), cancel)
    player = PCMPlayer(24000, cfg, on_audio, cancel, on_reference, pause)
    try:
        effects = PCMEffects(24000, float(cfg.get("pitch_semitones", 0) or 0),
                             float(cfg.get("robot_mix", 0) or 0))
        for packet in incoming:
            pcm = speech_gain(effects.process(packet), cfg.get("speech_gain_db", 0))
            if audioop is not None and gain != 1:
                pcm = audioop.mul(pcm, 2, gain)
            player.write(pcm)
        player.finish()
        return player.first - started
    except TurnInterrupted:
        raise
    except Exception as error:
        raise StreamSpeechError(str(error), player.played) from error
    finally:
        incoming.close()
        player.close()


class PreparedPCM:
    """One bounded take, fetched ahead while the previous take is audible."""
    def __init__(self, text, cfg, cancel):
        self.text, self.cfg, self.cancel = text, cfg, cancel
        self.data = queue.Queue()  # protocol total is capped at 4 MiB
        self.done = threading.Event()
        self.error = None

    def fetch(self):
        total, started = 0, time.monotonic()
        try:
            url = self.cfg["server_url"].rstrip("/") + "/stream?" + urllib.parse.urlencode({"voice": self.cfg["voice_name"]})
            req = urllib.request.Request(url, data=self.text.encode("utf-8"))
            timeout = min(15, max(1, float(self.cfg.get("stream_timeout_s", 8))))
            for attempt in range(3):
                try:
                    response = urllib.request.urlopen(req, timeout=timeout)
                    break
                except urllib.error.HTTPError as error:
                    error.close()
                    if error.code != 503 or attempt == 2:
                        raise
                    if self.cancel.wait(.05):
                        raise TurnInterrupted()
            with response as resp, closing_on_cancel(resp, self.cancel):
                if resp.headers.get_content_type() != "application/x-spark-pcm" or _read_exact(resp, 8) != b"SPK1" + struct.pack("<I", 24000):
                    raise ValueError("unexpected voice stream format")
                while not self.cancel.is_set():
                    size = struct.unpack("<I", _read_exact(resp, 4))[0]
                    if size == 0:
                        return
                    total += size
                    if size % 2 or size > 128 * 1024 or total > 4 * 1024 * 1024 or time.monotonic()-started > 60:
                        raise ValueError("voice stream exceeds PCM limits")
                    self.data.put(_read_exact(resp, size))
                raise TurnInterrupted()
        except Exception as error:
            self.error = error
        finally:
            self.done.set()

    def packets(self, cancel):
        while True:
            if cancel.is_set():
                raise TurnInterrupted()
            try:
                yield self.data.get(timeout=.02)
            except queue.Empty:
                if self.done.is_set():
                    if self.error:
                        raise self.error
                    return


def prefetch(sentences, cfg, cancel):
    """Read/generate at most two future takes. SDK/playback stay on the caller."""
    items, done, stop = queue.Queue(maxsize=2), threading.Event(), threading.Event()

    class LinkedCancel:
        def is_set(self):
            return stop.is_set() or cancel.is_set()
        def wait(self, seconds):
            deadline = time.monotonic() + seconds
            while not self.is_set():
                if stop.wait(min(.02, max(0, deadline-time.monotonic()))):
                    break
                if time.monotonic() >= deadline:
                    break
            return self.is_set()

    linked = LinkedCancel()

    def produce():
        primary_error = None
        try:
            for sentence in sentences:
                text = re.sub(r"[*_`#>]+", "", (sentence or "").strip())
                if not text:
                    continue
                prepared = PreparedPCM(text, cfg, linked)
                while not linked.is_set():
                    try:
                        items.put((text, prepared, None), timeout=.03)
                        break
                    except queue.Full:
                        pass
                if linked.is_set():
                    return
                if primary_error is not None:
                    prepared.error = primary_error
                    prepared.done.set()
                else:
                    prepared.fetch()  # immediately start N+1 after remote EOS of N
                    primary_error = prepared.error
        except Exception as error:
            while not linked.is_set():
                try:
                    items.put((None, None, error), timeout=.03)
                    break
                except queue.Full:
                    pass
        finally:
            try:
                close = getattr(sentences, "close", None)
                if close:
                    close()
            finally:
                done.set()

    threading.Thread(target=produce, daemon=True).start()
    try:
        while True:
            if cancel.is_set():
                raise TurnInterrupted()
            try:
                text, prepared, error = items.get(timeout=.02)
            except queue.Empty:
                if done.is_set():
                    return
                continue
            if error:
                raise error
            yield text, prepared
    finally:
        stop.set()


class PCMPlayer:
    """Bound the output queue to 60ms; share pacing for remote and local TTS."""
    def __init__(self, rate, cfg, on_audio, cancel, on_reference, pause=None):
        self.rate, self.cfg = rate, cfg
        self.on_audio, self.cancel, self.on_reference = on_audio, cancel, on_reference
        self.process = None
        self.played = False
        self.first = None
        self.render_until = 0.0
        self.tail = b""
        self.pause = pause

    def _wait_for_listener(self):
        while self.pause is not None and self.pause.is_set():
            if self.cancel.wait(.02):
                raise TurnInterrupted()
        if self.cancel.is_set():
            raise TurnInterrupted()

    def write(self, pcm, final=False):
        pcm = self.tail + pcm
        size = self.rate // 50 * 2
        complete = len(pcm) if final else len(pcm)//size*size
        self.tail = pcm[complete:]
        for offset in range(0, complete, size):
            frame = pcm[offset:min(offset+size, complete)]
            self._wait_for_listener()
            # render_until includes the 60ms device startup estimate. Allow
            # another 60ms of queued PCM, rather than pacing with no headroom.
            if self.cancel.wait(max(0, self.render_until - time.monotonic() - .12)):
                raise TurnInterrupted()
            if self.process is None:
                self.process = subprocess.Popen(
                    ["aplay", "-q", "-D", self.cfg.get("stream_output_device", "default"),
                     "-t", "raw", "-f", "S16_LE", "-r", str(self.rate), "-c", "1",
                     "--buffer-time=100000", "--period-time=20000"],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=None)
            if self.process.poll() is not None:
                raise RuntimeError("ALSA playback exited")
            now = time.monotonic()
            # Preserve the speaker sample clock across scheduler jitter.
            # Re-anchor only after a real synthesis/network underrun.
            render_start = now + .06 if self.render_until < now else self.render_until
            self.render_until = render_start + len(frame)/(2*self.rate)
            self.on_audio(time.time() + max(0, self.render_until-time.monotonic()) + .25)
            if not self.played:
                self.first = time.perf_counter()
            # The first write can fail AFTER audible bytes: never replay it.
            self.played = True
            self.process.stdin.write(frame)
            self.process.stdin.flush()
            if self.on_reference:
                self.on_reference(frame, self.rate, render_start)

    def finish(self):
        self.write(b"", final=True)
        self._wait_for_listener()
        if self.cancel.is_set():
            raise TurnInterrupted()
        if not self.played:
            raise ValueError("empty voice stream")
        self.process.stdin.close()
        deadline = max(time.monotonic(), self.render_until) + 3
        while self.process.poll() is None:
            if self.cancel.wait(.02):
                raise TurnInterrupted()
            if time.monotonic() >= deadline:
                raise RuntimeError("ALSA playback did not drain")
        if self.process.wait() != 0:
            raise RuntimeError("ALSA playback failed")

    def close(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1)
        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        self.on_audio(time.time() + .25)


def play_wav(path, cfg, on_audio, cancel, on_reference, pause=None):
    """Piper/stock spoken fallback uses the same cancellable echo reference."""
    import wave
    with wave.open(path, "rb") as source:
        if source.getsampwidth() != 2 or source.getnchannels() not in (1, 2):
            raise ValueError("unsupported speech WAV")
        rate = source.getframerate()
        player = PCMPlayer(rate, cfg, on_audio, cancel, on_reference, pause)
        gain = max(0, min(100, float(cfg.get("stream_volume", 90)))) / 100
        try:
            while True:
                pcm = source.readframes(rate // 50)
                if not pcm:
                    break
                if source.getnchannels() == 2:
                    pcm = audioop.tomono(pcm, 2, .5, .5)
                player.write(audioop.mul(speech_gain(pcm, cfg.get("speech_gain_db", 0)), 2, gain))
            player.finish()
        finally:
            player.close()
