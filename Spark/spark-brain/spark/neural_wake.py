"""Streaming local wake classifier with openWakeWord speech embeddings.

The 80ms feature geometry follows David Scripka's Apache-2.0 openWakeWord
AudioFeatures pipeline: https://github.com/dscripka/openWakeWord.
ONNX inference is CPU-only and owns no microphone, SDK, or robot actions.
"""
from collections import deque
import hashlib
import json
from pathlib import Path
import time


class NeuralWake:
    def __init__(self, cfg):
        import numpy as np
        import onnxruntime as ort
        wake = cfg['wake']
        if cfg['audio']['sample_rate'] != 16000:
            raise ValueError('Neural wake requires 16kHz mono PCM')
        folder = Path(wake['neural_model_dir'])
        manifest = json.loads((folder/'manifest.json').read_text())
        if set(wake['words']) != set(manifest['phrases']):
            raise ValueError('Neural model does not cover configured greetings')
        for name in ('melspectrogram.onnx', 'embedding_model.onnx', 'hey_spark.onnx'):
            if hashlib.sha256((folder/name).read_bytes()).hexdigest() != manifest['sha256'][name]:
                raise ValueError('Neural wake model checksum mismatch: '+name)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        self.mel, self.embedding, self.head = [ort.InferenceSession(str(folder/name),
            sess_options=opts, providers=['CPUExecutionProvider']) for name in
            ('melspectrogram.onnx', 'embedding_model.onnx', 'hey_spark.onnx')]
        self.np = np
        self.threshold = float(manifest['threshold'])
        if not .5 <= self.threshold <= 1:
            raise ValueError('Invalid calibrated neural threshold')
        self.required_hits = manifest['consecutive_frames']
        if type(self.required_hits) is not int or not 1 <= self.required_hits <= 10:
            raise ValueError('Invalid neural consecutive-frame policy')
        self.shadow = wake.get('neural_shadow', True)
        self.phrase = 'hey spark'
        self.last_score = 0.0
        self.last_hit = 0.0
        self.reset()

    def reset(self):
        np = self.np
        self.pending = bytearray()
        self.raw = deque([0]*480, maxlen=1760)
        self.mels = np.ones((76,32), dtype=np.float32)
        self.features = deque(maxlen=16)
        self.hits = 0
        self.last_score = 0.0
        # Seed with deterministic silence, never preceding speaker audio.
        for _ in range(26):
            self._score(bytes(2560))
        self.hits = 0

    def _score(self, pcm):
        np = self.np
        self.raw.extend(np.frombuffer(pcm, dtype='<i2').tolist())
        audio = np.asarray(self.raw, dtype=np.float32)[None,:]
        mel = self.mel.run(None, {self.mel.get_inputs()[0].name: audio})[0]
        mel = np.asarray(mel).reshape(-1,32)/10+2
        self.mels = np.concatenate((self.mels,mel))[-76:].astype(np.float32)
        value = self.embedding.run(None, {self.embedding.get_inputs()[0].name:
            self.mels[None,:,:,None]})[0].reshape(96)
        self.features.append(value)
        if len(self.features) < 16:
            return 0.0
        score = self.head.run(None, {self.head.get_inputs()[0].name:
            np.asarray(self.features,dtype=np.float32)[None,:,:]})[0]
        return float(score.reshape(-1)[0])

    def feed(self, pcm):
        self.pending.extend(pcm)
        if len(self.pending) < 2560:
            return False
        chunk = bytes(self.pending[:2560])
        del self.pending[:2560]
        self.last_score = self._score(chunk)
        self.hits = self.hits+1 if self.last_score >= self.threshold else 0
        if self.hits >= self.required_hits and time.monotonic()-self.last_hit >= 1.5:
            self.last_hit = time.monotonic()
            self.hits = 0
            return True
        return False
