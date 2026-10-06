"""Bounded, opt-in local evidence for wake misses; no audio device ownership."""
from collections import deque
import io
import json
import os
from pathlib import Path
import threading
import time
import wave


class WakeDiagnostics:
    def __init__(self, rate, until, folder):
        self.rate = rate
        self.until = min(float(until), time.time() + 600)
        self.folder = Path(folder)
        self.frames = deque(maxlen=750)  # 15s covers the clip plus ASR delay
        self.lock = threading.Lock()
        self.count = 0

    def append(self, raw, clean):
        if time.time() < self.until:
            with self.lock:
                self.frames.append((raw, clean))
        else:
            with self.lock:
                self.frames.clear()

    def save(self, clip, primary, secondary):
        if time.time() >= self.until or not clip or len(clip) > self.rate * 2 * 8:
            return
        with self.lock:
            clean = b''.join(pair[1] for pair in self.frames)
            offset = clean.find(clip)
            raw = b''.join(pair[0] for pair in self.frames)[offset:offset+len(clip)] if offset >= 0 else None
            slot = self.count % 20  # keep only the most recent 20 checks
            self.count += 1
        try:
            self.folder.mkdir(mode=0o700, parents=True, exist_ok=True)
            for name, pcm in (('clean', clip), ('raw', raw)):
                path = self.folder / f'{slot:02d}-{name}.wav'
                if pcm is None:
                    path.unlink(missing_ok=True)  # remove a previous slot's raw take
                    continue
                out = io.BytesIO()
                with wave.open(out, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(self.rate)
                    wav.writeframes(pcm)
                self._write(path, out.getvalue())
            self._write(self.folder / f'{slot:02d}.json', json.dumps({
                'captured_at': time.time(), 'primary': primary, 'secondary': secondary,
                'raw_matched': raw is not None, 'raw_stage': 'after high-pass, before AEC',
            }).encode())
        except OSError as error:
            print(f'[ear] wake diagnostic unavailable: {error}', flush=True)

    @staticmethod
    def _write(path, data):
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'wb') as out:
            out.write(data)
