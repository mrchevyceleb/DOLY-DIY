"""ASR — Vosk streaming recognizer (runs on the Pi).

Accumulates finalized segments across mid-utterance pauses so nothing
the user said is lost (Vosk finalizes on every silence).
"""
import json
import time
import sys


class Recognizer:
    def __init__(self, cfg):
        from vosk import Model, KaldiRecognizer  # deferred: heavy import

        path = cfg["asr"]["model_path"]
        self.model = Model(path)
        self.sample_rate = cfg["asr"]["sample_rate"]
        self._kaldi_cls = KaldiRecognizer
        self._finalized = []

    def begin(self):
        self.rec = self._kaldi_cls(self.model, self.sample_rate)
        self._finalized = []

    def feed(self, pcm):
        """Feed a frame chunk. Returns finalized text so far, or None."""
        if self.rec.AcceptWaveform(pcm):
            text = self._clean(self.rec.Result())
            if text:
                self._finalized.append(text)
                return text
        return None

    def finish(self):
        tail = self._clean(self.rec.FinalResult())
        if tail:
            self._finalized.append(tail)
        return " ".join(self._finalized).strip()

    def partial(self):
        """Current hypothesis, without waiting for a silence endpoint."""
        return json.loads(self.rec.PartialResult()).get("partial", "").strip()

    @staticmethod
    def _clean(result_json):
        try:
            text = json.loads(result_json).get("text", "").strip()
        except Exception:
            return ""
        return " ".join(text.split())


class WhisperASR:
    """Command transcription via whisper.cpp (the wake word stays on Vosk).

    whisper.cpp is a single self-contained binary tuned for Pi-class ARM —
    no torch, no CUDA, no 1GB of python deps.
    """

    def __init__(self, cfg):
        import os
        self.cfg = cfg
        base = "/opt/spark/whisper.cpp"
        self.bin = os.path.join(base, "build", "bin", "whisper-cli")
        if not os.path.exists(self.bin):
            self.bin = os.path.join(base, "build", "bin", "main")  # older name
        candidates = [
            cfg["asr"].get("whisper_model"),
            os.path.join(base, "models", "ggml-base.en.bin"),
            os.path.join(base, "models", "ggml-tiny.en.bin"),
        ]
        self.model = next((c for c in candidates if c and os.path.exists(c)), None)

    @staticmethod
    def _trim_silence(pcm, frame_bytes=640, thresh=600, pad_frames=5):
        """Drop leading/trailing quiet frames — whisper transcribes only
        actual speech, cutting wall-clock time substantially."""
        try:
            import struct as _st
            n = len(pcm) // frame_bytes
            if n == 0:
                return pcm
            rms = []
            for i in range(n):
                seg = pcm[i * frame_bytes:(i + 1) * frame_bytes]
                samples = _st.unpack("<%dh" % (len(seg) // 2), seg)
                rms.append(int((sum(x * x for x in samples) / max(1, len(samples))) ** 0.5))
            loud = [i for i, r in enumerate(rms) if r >= thresh]
            if not loud:
                return pcm
            a = max(0, loud[0] - pad_frames)
            b = min(n, loud[-1] + 1 + pad_frames)
            return pcm[a * frame_bytes:b * frame_bytes]
        except Exception:
            return pcm

    @property
    def available(self):
        import os
        return bool(self.model) and os.path.exists(self.bin)

    def _transcribe_http(self, pcm, sample_rate=16000):
        """Moria's whisper server (~0.3s) via curl (bulletproof multipart)."""
        import os
        import subprocess
        import tempfile
        import wave
        fd, path = tempfile.mkstemp(suffix=".wav", dir="/tmp")
        os.close(fd)
        try:
            w = wave.open(path, "wb")
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate)
            w.writeframes(pcm); w.close()
            url = self.cfg.get("asr", {}).get("server_url",
                                             "http://192.168.50.204:8399/inference")
            out = subprocess.run(
                ["curl", "--fail", "--silent", "--show-error", "--connect-timeout", "1", "-m", "8", "-X", "POST", url,
                 "-F", "file=" + chr(64) + path, "-F", "response_format=text"],
                capture_output=True, text=True, timeout=12, check=True)
            return " ".join(out.stdout.split())
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    def transcribe_pcm(self, pcm, sample_rate=16000):
        """Raw 16-bit mono PCM -> text. Moria first, local whisper fallback."""
        import os
        import subprocess
        import tempfile
        import wave
        if not pcm:
            return ""
        if self.cfg.get("asr", {}).get("server_url"):
            try:
                t0 = time.time()
                text = self._transcribe_http(self._trim_silence(pcm), sample_rate)
                print(f"[asr] moria {time.time()-t0:.2f}s: '{text}'", file=sys.stderr, flush=True)
                # The server ANSWERED: empty means "no speech", not "try
                # harder". Falling through to local tiny.en on empty cost
                # 6.7s and hallucinated '(buzzing)'/'(applause)' over real
                # speech — the local fallback exists only for server-down.
                return text
            except Exception as e:
                print(f"[asr] moria server failed: {e}", file=sys.stderr, flush=True)
        if not self.available:
            return ""
        fd, path = tempfile.mkstemp(suffix=".wav", dir="/tmp")
        os.close(fd)
        try:
            pcm = self._trim_silence(pcm)
            w = wave.open(path, "wb")
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm)
            w.close()
            cmd = [self.bin, "-m", self.model, "-f", path, "-nt",
                   "--prompt", "Spark robot voice commands: dance, forward, "
                   "backward, spin, stop, come here, fist bump, high five, "
                   "timer, weather, photo, sleep, wake up."]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return " ".join(out.stdout.strip().split())
        except Exception as e:
            print(f"[asr] whisper failed: {e}", file=sys.stderr, flush=True)
            return ""
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass
