#!/usr/bin/env python3
"""Generate local voice auditions on Moria without changing Spark's service.

Use an isolated Python environment with kokoro, qwen-tts and CUDA PyTorch.
Each backend writes WAVs and its own manifest. Timings are measured after a
warmup, exclude model loading, and describe synthesis, not robot response time.
"""
import argparse
import io
import json
import platform
import sys
import time
import urllib.request
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spark.voicefx import apply_fx

LINES = [
    "Hey Matt. I'm Spark. What are we working on today?",
    "Hmm, I think we can make that work. Tell me a little more about what you have in mind.",
    "Your twenty-minute timer is set. And yes, I still remember how to dance!",
]
WARMUP = "Hi Matt."
DESIGNS = [
    ("qwen-warm", "Qwen - warm companion",
     "A young adult female voice with a natural American English accent. Warm, clear, "
     "friendly and conversational. Medium-high pitch, gentle smile, relaxed natural "
     "pacing. A helpful companion, with subtle personality and no theatrical delivery."),
    ("qwen-playful", "Qwen - playful Spark",
     "A bright young adult female voice with a natural American English accent. "
     "Playful, curious and affectionate, with a light, slightly higher pitch. "
     "Speak naturally and clearly at a brisk conversational pace, with expressive "
     "intonation. Friendly tiny robot companion; no metallic effects or baby voice."),
]


def write_wav(path, samples, rate):
    import numpy as np
    import soundfile as sf
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError("synthesizer returned empty or nonfinite audio")
    sf.write(str(path), samples, rate, subtype="PCM_16")
    return len(samples) / rate


def piper(text, url):
    request = urllib.request.Request(url, data=text.encode("utf-8"))
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(4 * 1024 * 1024)
    with wave.open(io.BytesIO(data), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise ValueError("expected mono PCM16 Piper WAV")
        import numpy as np
        return np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype("float32") / 32768, audio.getframerate()


def concatenate(paths, target):
    """Join same-format clips with quiet gaps; no pitch/effects on new voices."""
    params, parts = None, []
    for path in paths:
        with wave.open(str(path), "rb") as audio:
            current = (audio.getnchannels(), audio.getsampwidth(), audio.getframerate())
            if params is not None and current != params:
                raise ValueError("clip formats differ")
            params = current
            parts.append(audio.readframes(audio.getnframes()))
            parts.append(bytes(round(current[2] * .35) * current[0] * current[1]))
    with wave.open(str(target), "wb") as audio:
        audio.setnchannels(params[0])
        audio.setsampwidth(params[1])
        audio.setframerate(params[2])
        audio.writeframes(b"".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["piper", "kokoro", "qwen"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--piper-url", default="http://127.0.0.1:8398/?voice=hfc")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    variants = []
    model_id = None
    if args.backend == "piper":
        variants = [("piper-current", "Current Spark - Piper + robot effects", None)]
        synth = lambda text, voice: piper(text, args.piper_url)
    else:
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(42)
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing to silently benchmark CPU")
        if args.backend == "kokoro":
            from kokoro import KPipeline
            model_id = "hexgrad/Kokoro-82M"
            pipeline = KPipeline(lang_code="a", device=args.device, repo_id=model_id)
            variants = [("kokoro-" + voice, "Kokoro - " + label, voice)
                        for voice, label in [("af_heart", "Heart"), ("af_bella", "Bella"), ("af_nicole", "Nicole")]]

            def synth(text, voice):
                import numpy as np
                chunks = [audio.cpu().numpy() for _, _, audio in pipeline(text, voice=voice, speed=1)]
                return np.concatenate(chunks), 24000
        else:
            from qwen_tts import Qwen3TTSModel
            model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
            model = Qwen3TTSModel.from_pretrained(model_id, device_map=args.device,
                                                dtype=torch.bfloat16, attn_implementation="sdpa")
            variants = DESIGNS

            def synth(text, voice):
                wavs, rate = model.generate_voice_design(text=text, language="English", instruct=voice,
                                                        max_new_tokens=1024)
                return wavs[0], rate
    manifest = {"backend": args.backend, "model": model_id, "host": platform.node(),
                "device": "CPU" if args.backend == "piper" else args.device,
                "request_url": args.piper_url if args.backend == "piper" else None,
                "lines": LINES, "timing": "Warm full-clip synthesis, including configured Piper FX; excludes model loading, ASR, LLM and playback.",
                "voices": []}
    for slug, title, voice in variants:
        print("WARMUP", slug, flush=True)
        synth(WARMUP, voice)
        clips, measurements = [], []
        for index, line in enumerate(LINES, 1):
            start = time.perf_counter()
            samples, rate = synth(line, voice)
            if args.backend != "piper" and args.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            clip = args.output / f"{slug}-{index}.wav"
            duration = write_wav(clip, samples, rate)
            if args.backend == "piper":
                raw = clip.with_suffix(".raw.wav")
                clip.replace(raw)
                apply_fx(str(raw), str(clip), pitch_semitones=2, robot_mix=.25)
                with wave.open(str(clip), "rb") as audio:
                    duration = audio.getnframes() / audio.getframerate()
                raw.unlink()
                elapsed = time.perf_counter() - start
            measurements.append({"line": index, "synthesis_s": round(elapsed, 3),
                                 "audio_s": round(duration, 3), "rtf": round(elapsed / duration, 3)})
            clips.append(clip)
            print("CLIP", slug, index, measurements[-1], flush=True)
        target = args.output / f"{slug}.wav"
        # VoiceDesign chooses a new identity on each call. Joining its short
        # benchmark clips would audition several speakers under one label.
        if args.backend == "qwen":
            print("CONTINUOUS TAKE", slug, flush=True)
            samples, rate = synth(" ".join(LINES), voice)
            write_wav(target, samples, rate)
        else:
            concatenate(clips, target)
        manifest["voices"].append({"id": slug, "title": title, "file": target.name,
                                    "voice": voice, "measurements": measurements,
                                    "audition_mode": "continuous voice design" if args.backend == "qwen" else "joined clips"})
        (args.output / f"{args.backend}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("READY", target, flush=True)


if __name__ == "__main__":
    main()
