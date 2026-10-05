"""Voice FX — pure-stdlib post-processing for piper WAV output.

Two knobs, both optional and configured via config.json tts.*:

- pitch_semitones: resample shift. Raises pitch AND tempo together
  (classic tape effect) — a couple of semitones turns a flat TTS voice
  into a perky little robot and shaves playback time too.
- robot_mix: subtle 45 Hz ring-modulation mixed under the dry signal
  (0..1). Adds a metallic "robot" sheen without burying clarity.

Everything runs on 16-bit mono PCM via the wave/array modules — no numpy,
no sox, no ffmpeg.
"""
import array
import math
import wave

try:
    import audioop  # deprecated in 3.11, gone in 3.13 — Pi OS bookworm is fine
except ImportError:  # pragma: no cover
    audioop = None


def _read_wav(path):
    with wave.open(path, "rb") as w:
        params = (w.getnchannels(), w.getsampwidth(), w.getframerate())
        frames = w.readframes(w.getnframes())
    return params, frames


def _write_wav(path, params, frames):
    nch, sw, rate = params
    with wave.open(path, "wb") as w:
        w.setnchannels(nch)
        w.setsampwidth(sw)
        w.setframerate(rate)
        w.writeframes(frames)


def _to_mono16(params, frames):
    nch, sw, rate = params
    if sw != 2:
        frames = audioop.lin2lin(frames, sw, 2)
    if nch != 1:
        frames = audioop.tomono(frames, 2, 0.5, 0.5)
    return frames


def _pitch_shift(frames, rate, semitones):
    """Resample so playback at the ORIGINAL rate is higher+faster."""
    if audioop is None or semitones == 0:
        return frames
    factor = 2.0 ** (semitones / 12.0)
    shifted, _ = audioop.ratecv(frames, 2, 1, rate, int(rate / factor), None)
    return shifted


def _ring_mod(frames, rate, mix, freq=45.0):
    """Blend a ring-modulated copy under the dry signal."""
    if mix <= 0:
        return frames
    mix = max(0.0, min(1.0, mix))
    samples = array.array("h", frames)
    n = len(samples)
    step = 2.0 * math.pi * freq / rate
    dry_gain = 1.0 - 0.5 * mix
    for i in range(n):
        mod = math.sin(step * i)
        v = samples[i] * dry_gain + samples[i] * mod * (0.5 * mix)
        samples[i] = max(-32768, min(32767, int(v)))
    return samples.tobytes()


def apply_fx(in_wav, out_wav, pitch_semitones=0.0, robot_mix=0.0):
    """Read in_wav, apply configured FX, write out_wav. Returns out_wav."""
    params, frames = _read_wav(in_wav)
    if audioop is None or (not pitch_semitones and not robot_mix):
        if in_wav != out_wav:
            _write_wav(out_wav, params, frames)
        return out_wav
    frames = _to_mono16(params, frames)
    params = (1, 2, params[2])
    frames = _pitch_shift(frames, params[2], pitch_semitones)
    frames = _ring_mod(frames, params[2], robot_mix)
    _write_wav(out_wav, params, frames)
    return out_wav


class PCMEffects:
    """The same voice effect with continuous resample and ring phase."""
    def __init__(self, rate, pitch_semitones=0.0, robot_mix=0.0):
        self.rate = rate
        self.output_rate = int(rate / (2.0 ** (pitch_semitones / 12.0)))
        self.pitch = pitch_semitones
        self.mix = max(0.0, min(1.0, robot_mix))
        self.state = None
        self.offset = 0

    def process(self, frames):
        if audioop is None:
            return frames
        if self.pitch:
            frames, self.state = audioop.ratecv(
                frames, 2, 1, self.rate, self.output_rate, self.state)
        if self.mix:
            samples = array.array("h", frames)
            step = 2.0 * math.pi * 45.0 / self.rate
            gain = 1.0 - 0.5 * self.mix
            for i in range(len(samples)):
                mod = math.sin(step * (self.offset + i))
                value = samples[i] * gain + samples[i] * mod * (0.5 * self.mix)
                samples[i] = max(-32768, min(32767, int(value)))
            self.offset += len(samples)
            frames = samples.tobytes()
        return frames


def speech_gain(frames, gain_db=0):
    """Boost speech with a smooth peak limiter, before volume and echo reference.

    Stateless per sample: chunk boundaries cannot change loudness or add delay.
    The original signal is preserved when disabled. Above 80% full scale a soft
    knee approaches 95%, avoiding hard PCM clipping on occasional loud syllables.
    """
    gain_db = max(0, min(12, float(gain_db)))
    if not gain_db:
        return frames
    gain = 10 ** (gain_db / 20)
    samples = array.array("h", frames)
    for i, sample in enumerate(samples):
        value = sample / 32768 * gain
        magnitude = abs(value)
        if magnitude > .8:
            magnitude = .8 + .15 * math.tanh((magnitude-.8)/.15)
            value = math.copysign(magnitude, value)
        samples[i] = int(value * 32767)
    return samples.tobytes()
