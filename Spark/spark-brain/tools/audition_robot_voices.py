#!/usr/bin/env python3
"""Add cute Qwen robot auditions; never regenerate the original favorite takes."""
import argparse
import json
import platform
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spark.voicefx import apply_fx, audioop

TUNINGS = [
    ("robot-warm-glow", "Warm - soft robot", "qwen-warm.wav", .75, .16,
     "Original Warm Companion with a gentle lift and light electronic sheen."),
    ("robot-warm-pocket", "Warm - pocket robot", "qwen-warm.wav", 1.4, .30,
     "The same warm voice, smaller and brighter, with more robot texture."),
    ("robot-playful-glow", "Playful - soft robot", "qwen-playful.wav", .65, .14,
     "Original Playful Spark with a subtle, friendly robot shimmer."),
    ("robot-playful-pocket", "Playful - pocket robot", "qwen-playful.wav", 1.25, .26,
     "The same playful voice, extra perky, with a clearer electronic character."),
]
DESIGNS = [
    ("robot-qwen-android", "Qwen - little android", .35, .22,
     "A cute small female robot companion with a clear natural American English accent. "
     "A warm, sweet, medium-high voice, soft rounded vowels and a light crystalline "
     "synthetic timbre. Minimal breathiness. Speak with natural conversational "
     "rhythm, emotional nuance, a gentle smile and curious expressive intonation. "
     "Sound like a lovable intelligent robot with feelings, not a flat monotone. "
     "Keep every word clear; no baby talk, exaggerated cartoon acting or beeps."),
    ("robot-qwen-bubbles", "Qwen - bubbly Spark", .75, .20,
     "A bright, cute female pocket robot with a natural American English accent. "
     "Light, high but not squeaky voice, clear bell-like vowels and a subtle "
     "electronic quality. Cheerful, mischievous, affectionate and curious. "
     "Use a brisk but understandable conversational pace with realistic emotional "
     "expression, natural pauses and lively melodic intonation. Keep the delivery "
     "believable and relaxed, with no baby voice, forced cartoon acting or beeps."),
]
MIXED_DESIGNS = [
    ("robot-neutral-cinder", "Neutral - warm Cinder", 0, .16,
     "An androgynous small robot companion speaking natural American English. "
     "A smooth middle-register voice between an alto and a light tenor, with "
     "balanced resonance, rounded vowels and very little breathiness. Gender "
     "ambiguous, cozy, thoughtful and quietly playful. Use realistic emotional "
     "nuance, a gentle smile, relaxed conversational timing and clear words. "
     "A subtle crystalline electronic quality, like a lovable intelligent robot."),
    ("robot-neutral-pebble", "Neutral - curious Pebble", 0, .20,
     "A cute gender-neutral pocket robot speaking clear American English. "
     "An androgynous light tenor with compact rounded resonance and a clean "
     "slightly buzzy timbre. Friendly, curious and gently mischievous. Speak "
     "naturally with expressive melodic inflection, easy pauses and believable "
     "warmth. A little digital sparkle while staying intelligible and relaxed."),
    ("robot-boyish-milo", "Boyish - gentle Milo", 0, .16,
     "A cute male robot companion with a youthful young-adult masculine voice "
     "and a natural American English accent. A warm light tenor, smooth rounded "
     "male resonance, soft clear consonants and minimal breathiness. Gentle, "
     "affectionate and curious, like a friendly little android. Natural relaxed "
     "conversation with real emotional expression, a smile and subtle electronic "
     "color. Boyish charm without sounding like a baby or a deep announcer."),
    ("robot-boyish-pip", "Boyish - playful Pip", .25, .20,
     "A cheerful small male robot speaking natural American English in a youthful "
     "masculine light tenor. Bright, compact, slightly nasal male resonance with "
     "a soft buzzy digital timbre. Playful, witty, warm and curious. Use lively "
     "but believable conversational expression, natural pauses and clear words. "
     "Cute boyish charm with a relaxed young-adult delivery and a gentle smile."),
]
KID_DESIGNS = [
    ("robot-kid-chip", "Robot kid - gentle Chip", 0, .16,
     "A small boy robot with the speaking voice of a prepubescent boy around "
     "nine years old. Natural American English. A light, clear, youthful high "
     "register, small bright boyish resonance and soft rounded vowels. Warm, "
     "thoughtful and sweet, with a gentle smile. Speak like a real child having "
     "a relaxed conversation: clear words, expressive curiosity and easy pauses. "
     "A subtle crystalline robot timbre. The vocal identity is a little boy, "
     "not a grown man, woman, baby or exaggerated cartoon character."),
    ("robot-kid-pixel", "Robot kid - curious Pixel", .25, .20,
     "A cute little boy android with the voice of an eight-year-old boy. Clear "
     "natural American English. Bright prepubescent child resonance, youthful "
     "high pitch, a little boyish nasal buzz and light clean consonants. Curious, "
     "playful and affectionate. Natural realistic childlike conversation, lively "
     "inflections and spontaneous gentle pauses. Small bell-like digital color. "
     "Sound like a little robot kid with feelings, not an adult tenor or baby."),
    ("robot-kid-widget", "Robot kid - mellow Widget", 0, .24,
     "A tiny robot child with a young boy's prepubescent voice, around ten years "
     "old, soft and light enough to feel gently androgynous. Natural American "
     "English. Compact bright child resonance, a clear youthful register and "
     "rounded clean vowels. Cozy, sincere, curious and slightly mischievous. "
     "Believable emotional expression, relaxed conversational timing and clear "
     "speech, with a subtle soft electronic buzz. A little robot kid rather "
     "than an adult man or woman. No baby talk or exaggerated cartoon acting."),
]


def save_manifest(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--round", choices=("cute", "mixed", "kids"), default="cute")
    args = parser.parse_args()
    if audioop is None:
        parser.error("Robot effects require audioop; use the Python 3.12 audition environment.")
    base = json.loads((args.folder / "qwen.json").read_text(encoding="utf-8"))
    designs = {"cute": DESIGNS, "mixed": MIXED_DESIGNS, "kids": KID_DESIGNS}[args.round]
    tunings = TUNINGS if args.round == "cute" else []
    manifest_path = args.folder / {"cute": "qwen-robots.json", "mixed": "qwen-mixed.json",
                                   "kids": "qwen-kids.json"}[args.round]
    if manifest_path.exists() or any(
            (args.folder / f"{design[0]}{suffix}.wav").exists()
            for design in designs for suffix in ("", "-dry")) or any(
            (args.folder / f"{tuning[0]}.wav").exists() for tuning in tunings):
        parser.error("Auditions already exist; preserve them and use a fresh output directory.")
    manifest = {"backend": "qwen", "model": base["model"], "host": platform.node(),
                "lines": base["lines"], "voices": []}
    for slug, title, source, pitch, mix, description in tunings:
        apply_fx(str(args.folder / source), str(args.folder / f"{slug}.wav"), pitch, mix)
        manifest["voices"].append({"id": slug, "title": title, "file": f"{slug}.wav",
            "source_file": source, "pitch_semitones": pitch, "robot_mix": mix,
            "description": description, "measurements": []})
        print("READY", slug, flush=True)
    if args.round == "cute":
        save_manifest(manifest_path, manifest)
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel
    torch.set_num_threads(4)
    torch.manual_seed(84)
    model = Qwen3TTSModel.from_pretrained(base["model"], device_map="cuda",
                                        dtype=torch.bfloat16, attn_implementation="sdpa")
    model.generate_voice_design(text="Hi Matt.", language="English", instruct=designs[0][4])
    for slug, title, pitch, mix, prompt in designs:
        print("GENERATING", slug, flush=True)
        start = time.perf_counter()
        wavs, rate = model.generate_voice_design(text=" ".join(base["lines"]), language="English",
                                               instruct=prompt, max_new_tokens=1024)
        raw, target = args.folder / f"{slug}-dry.wav", args.folder / f"{slug}.wav"
        sf.write(str(raw), wavs[0], rate, subtype="PCM_16")
        apply_fx(str(raw), str(target), pitch, mix)
        elapsed = time.perf_counter() - start
        with wave.open(str(target), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        manifest["voices"].append({"id": slug, "title": title, "file": target.name,
            "voice": prompt, "pitch_semitones": pitch, "robot_mix": mix,
            "description": "New continuous Qwen voice design with light robot polish.",
            "benchmark_kind": "Full take + robot processing",
            "measurements": [{"synthesis_s": round(elapsed, 3), "audio_s": duration}]})
        save_manifest(manifest_path, manifest)
        print("READY", slug, flush=True)
    save_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
