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
