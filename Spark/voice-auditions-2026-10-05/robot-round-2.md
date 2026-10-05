# Cute robot auditions

Matt's direction: a cute robot voice with the natural expression of Qwen Warm
Companion and Qwen Playful Spark. Both original samples and `qwen.json` are
preserved byte for byte.

Four variations reuse those exact continuous takes with a gentle pitch/tempo
lift and the existing 45 Hz ring-modulation effect. The soft versions use a
lighter effect; the pocket versions sound smaller and more electronic.
Two additional continuous Qwen VoiceDesign takes explore Little Android and
Bubbly Spark, followed by light robot effects. Dry versions are retained too.
All six use the original three-line audition transcript; no robot action runs.

Open `index.html` for the twelve-voice comparison. New voices are added after
the original six. `qwen-robots.json` records prompts, source takes and effects.
Derived takes have no new synthesis benchmark. The two new designs report
full-excerpt generation including effects, excluding model loading and warmup;
these times differ in scope from the original short-line benchmarks.

Reproduce on Moria in the existing Python 3.12 audition environment:

```sh
python tools/audition_robot_voices.py samples
python tools/build_voice_audition_page.py samples
```

Keep `tools/` and `spark/voicefx.py` in their repository layout. Required input:
`samples/qwen.json`, `samples/qwen-warm.wav`, `samples/qwen-playful.wav`.
No robot service, live voice configuration, motion logic or brain was changed.

Verification: all six new samples transcribed correctly through local Parakeet;
all twelve player assets loaded without audio errors. Codex-Fix's direct CLI
review returned `NO FINDINGS`.

Preserved original SHA-256 hashes:

- Warm: `c8ea61431cef8f2da6dacd9b781dd9a10c3f0c95b40dba7fa5987922dc72c4e5`
- Playful: `9ae2807453ac7ec98a752217db9423a7c6b3b97f4ef46f5d55662681cab6f924`
- Manifest: `401da7b04bebe66cba6264e0325a8b2fe8b73c353504f6b332df1b2564c3a1bd`
