# Neutral and boyish auditions

Matt's provisional frontrunner is **Warm - soft robot**, the unchanged
`robot-warm-glow.wav` take. He requested a mix of gender-neutral and boyish
alternatives after the earlier robot candidates all read female.

Four fresh continuous Qwen VoiceDesign takes explore that direction:

- Neutral - warm Cinder: balanced middle register, cozy and thoughtful.
- Neutral - curious Pebble: compact light tenor, playful and curious.
- Boyish - gentle Milo: warm masculine light tenor, relaxed and affectionate.
- Boyish - playful Pip: bright masculine light tenor, lively and witty.

These are prompt directions to audition, not guarantees of perceived gender.
Each uses the same three-line script and subtle robot effects similar to the
frontrunner. The first three have no pitch shift; Pip uses just +0.25 semitone.
All earlier samples remain available. Dry versions of these four are retained.
`qwen-mixed.json` records prompts, effects and full-excerpt generation times,
including WAV writing and robot processing but excluding model loading/warmup.

Reproduce in a fresh directory containing the original `qwen.json`, using the
existing Python 3.12 audition environment on Moria:

```sh
python tools/audition_robot_voices.py samples --round mixed
python tools/build_voice_audition_page.py samples
```

The mixed round refuses to overwrite an existing manifest or take. Repeated
voice design can change identity; preserve these WAVs for later speaker locking.
This round changes only auditions, not Spark's live voice or robot control.

Verification: all four processed clips are nonempty mono PCM16 WAVs and
transcribed the full script correctly through local Parakeet. Transcripts and
durations are retained in `mixed-verification.json`. SHA-256 checks confirm
the original Warm, Playful and frontrunner Warm - soft robot takes are unchanged.

Listening feedback: Matt heard both neutral designs as female and the boyish
designs as men around age 25. These are retained for comparison, but did not
meet his intended character. The next round targets a little robot kid.
