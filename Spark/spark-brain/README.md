# Spark Brain

A bigger brain for Dolly the Doly robot — LLM-powered conversation on
[Moria's LM Studio](http://192.168.50.204:1234), with **every stock behavior
preserved** and instant rollback to stock.

    mic -> wake word / tap -> VAD -> Parakeet ASR (Moria, :8399)
                              |
                         ROUTER
                  +-----------+-----------+
            stock command match      free speech
                  |                        |
            Doly SDK action       qwen3.6-35b-a3b (Moria)
            (dance/drive/time/..) -> streamed reply -> piper voice (Moria, :8398)

## What's preserved

- Stock doly service stays installed. `spark off` restores it in seconds.
- Stock-style voice commands (time, weather, timer, photo, fist bump,
  drive, dance, colors, sleep, battery, go home) run **locally, instantly, offline**.
- Every Moria dependency degrades gracefully: TTS falls back to on-device
  piper, ASR to on-device whisper/vosk, and commands work with no network.

## Her voice

Default: **Robot** — HFC female +2 semitones + a subtle ring-mod sheen,
synthesized on Moria in ~0.4s (the Pi needs 5-15s for the same model — that
synth latency was most of her 'slow to respond').

The curated picks (Matt auditioned 2026-09, samples in `tools/samples/`):

| say | voice | character |
|---|---|---|
| `switch to robot` | HFC +2st + robot sheen | the default — cute, clear, robotic |
| `switch to hfc` | HFC +2st | same voice, no robot effect |
| `switch to lessac` | Lessac +2st | crispest articulation (in the hopper) |
| `switch to kathleen` | Kathleen +3st | lo-fi toy robot, fastest local synth |

Also available: `cori`, `glados`, `wheatley`, `amy`, `stock`.
Audition new tunings on the robot: `python3 tools/audition_voices.py /tmp/aud`
(edit VARIANTS first; needs `voicefx.py` beside it).

FX are pure stdlib (`spark/voicefx.py`): `tts.pitch_semitones` resamples
(pitch+tempo together — robots talk briskly), `tts.robot_mix` blends a 45 Hz
ring-mod under the dry signal.

## Robot access (Spark / Dolly)

- **Host:** `spark` on the tailnet (`100.112.197.4`), LAN `192.168.50.187`, hostname `doly`
- **SSH:** `ssh doly@spark` — agent key installed (trenzalore's `id_ed25519`)
- **Default credentials** (from the official Doly community post — change if stock image is re-flashed):
  - user `doly` / password `morethanarobot`
  - user `root` / same default password
- **sudo:** passwordless for `doly`
- **Environment:** Raspberry Pi OS bookworm aarch64, Python 3.11.2, ~1.8 GB RAM,
  23 GB free disk, Doly SDK at `/.doly/libs/sdk/lib/python3.11/dist-packages`
  (system python sees it — keep venvs `--system-site-packages`)
- **Stock services:** `doly.service`, `doly_recovery.service`, `doly_update.service`
- **Audio device (mic):** `plughw:CARD=tlv320aic31xxso,DEV=0` (Doly's TLV320AIC31xx codec)
- **Brain server:** LM Studio on Moria `192.168.50.204:1234` — same LAN subnet, direct link
- **Moria services:** piper TTS `:8398`, Parakeet ASR `:8399` — see `deploy/moria/`

## Deploy (on the robot)

    scp -r Spark/spark-brain doly@spark:/home/doly/
    ssh doly@spark
    cd ~/spark-brain && sudo bash deploy/install.sh
    spark test        # text REPL first — verify brain pipeline
    spark on          # go live

Requires: robot on the same LAN as Moria (or tailnet), LM Studio server
running with `qwen/qwen3.6-35b-a3b` loaded, context 8192, TTL off.

## Runtime

- **Say 'Spark'** (or tap her) -> she listens -> speak -> silence ends turn.
- Free speech -> Moria -> streamed reply, spoken sentence by sentence.
- **'Go home' / 'go to your charger'** -> she drives back to her dock
  (dead-reckoning + the dock's all-four-void signature, with wrong-edge abort).
- **Battery under 10%** -> she announces and takes herself home to charge.
- **Fully charged (95%+)** -> she hops off the dock and roams the desk;
  edge sensors guard every move, and she self-escapes desk edges
  (backs off, turns away) instead of freezing.
- **'Search for...' / 'look up...'** -> web search -> brain answers from fresh results.
- Moria unreachable -> she says so, **stock commands still work**.

## Files

| file | role |
|---|---|
| `prompt.md` | Spark personality v4 (source of truth) |
| `config.json` | endpoints, model, VAD thresholds, voice + FX, roam/battery |
| `spark/brain.py` | LM Studio streaming client + fallback model |
| `spark/router.py` | stock-command router + voice switching + weather/timer + web search |
| `spark/search.py` | DuckDuckGo web search (ddgs pkg + HTML fallback) |
| `spark/commands.py` | parity table — extend from the Doly app's 'Say' list |
| `spark/body.py` | guarded Doly SDK wrappers (edges, homing, wander, TTS path) |
| `spark/ear.py` | arecord + energy VAD + wake listener (pure stdlib) |
| `spark/asr.py` | vosk (wake) + whisper/parakeet HTTP client (utterances) |
| `spark/voicefx.py` | pitch/ring-mod FX for piper WAVs (pure stdlib) |
| `deploy/moria/` | piper TTS + Parakeet ASR servers and systemd units |
| `tools/audition_voices.py` | voice audition rig (samples in `tools/samples/`) |

## Tuning

- False wake / clipped speech -> `audio.start_rms` / `silence_ms` /
  `wake_weak_rms` (loudness floor for acoustic-confusion wake words).
- Swap brains -> `brain.model` (fallback: `brain.fallback_model`).
- Voice -> say `switch to ...` (persists), or set `tts.voice_name` +
  `tts.pitch_semitones` / `tts.robot_mix` in config.
- Roam/battery -> `idle.roam_*`, `idle.low_battery_pct` (default 10),
  `idle.battery_check_s`.
- Add stock commands -> append to `COMMANDS` in `spark/commands.py`.

## Phase 1 scope (deliberate)

No vision, no app changes. One thing at a time.
