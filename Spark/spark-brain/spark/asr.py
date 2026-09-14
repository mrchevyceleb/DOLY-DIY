"""ASR — Vosk streaming recognizer (runs on the Pi).

Accumulates finalized segments across mid-utterance pauses so nothing
the user said is lost (Vosk finalizes on every silence).
"""
import json
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

    @staticmethod
    def _clean(result_json):
        try:
            text = json.loads(result_json).get("text", "").strip()
        except Exception:
            return ""
        return " ".join(text.split())
