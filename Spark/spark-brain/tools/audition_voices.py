#!/usr/bin/env python3
"""Voice audition rig — run ON THE ROBOT (or anywhere with piper).

Synthesizes a test sentence through every voice x FX variant and writes
WAVs you can listen to side by side:

    python3 audition_voices.py /tmp/spark_audition

Needs: piper binary + voices (defaults match the robot's layout), and
voicefx.py next to it (or the spark package importable).
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from spark.voicefx import apply_fx  # repo layout: tools/ next to spark/
except ImportError:                    # deployed standalone (e.g. /tmp on the Pi)
    from voicefx import apply_fx  # noqa: E402

PIPER = "/.doly/libs/piper/lib/piper"
ESPEAK = "/.doly/libs/piper/lib/espeak-ng-data"
VOICE_DIR = "/.doly/data/piper"

SENTENCE = ("Hi Matt! I'm Spark, your tiny desk robot. My wheels are ready, "
            "my dance moves are loaded, and I promise not to fall off the table. "
            "Want to hear a joke about batteries?")

# (label, voice file, pitch_semitones, robot_mix)
VARIANTS = [
    ("1-glados-plain",      "glados.onnx",                    0.0, 0.0),
    ("2-hfc-plain",         "en_US-hfc_female-medium.onnx",   0.0, 0.0),
    ("3-hfc-plus2",         "en_US-hfc_female-medium.onnx",   2.0, 0.0),
    ("4-cori-plus2",        "en_GB-cori-high.onnx",           2.0, 0.0),
    ("5-kathleen-plus3",    "en_US-kathleen-low.onnx",        3.0, 0.0),
    ("6-ljspeech-plus2",    "en_US-ljspeech-medium.onnx",     2.0, 0.0),
    ("7-lessac-plus2",      "en_US-lessac-high.onnx",         2.0, 0.0),
    ("8-hfc-plus2-robot",   "en_US-hfc_female-medium.onnx",   2.0, 0.25),
    ("9-kathleen-plus3-robot", "en_US-kathleen-low.onnx",     3.0, 0.25),
]


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/spark_audition"
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/.doly/libs/piper/lib:" + env.get("LD_LIBRARY_PATH", "")
    for label, voice, pitch, robot in VARIANTS:
        model = os.path.join(VOICE_DIR, voice)
        if not os.path.exists(model):
            print(f"SKIP {label}: {voice} missing", flush=True)
            continue
        raw = os.path.join(out_dir, label + ".raw.wav")
        out = os.path.join(out_dir, label + ".wav")
        t0 = time.time()
        proc = subprocess.run(
            [PIPER, "--model", model, "--espeak_data", ESPEAK,
             "--output_file", raw, "-q"],
            input=SENTENCE, capture_output=True, text=True, timeout=60, env=env)
        synth_ms = (time.time() - t0) * 1000
        if proc.returncode != 0:
            print(f"FAIL {label}: rc={proc.returncode} {proc.stderr[:100]}", flush=True)
            continue
        t1 = time.time()
        apply_fx(raw, out, pitch_semitones=pitch, robot_mix=robot)
        fx_ms = (time.time() - t1) * 1000
        os.remove(raw)
        print(f"OK {label}: synth {synth_ms:.0f}ms fx {fx_ms:.0f}ms", flush=True)


if __name__ == "__main__":
    main()
