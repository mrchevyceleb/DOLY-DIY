# Spark Brain

A bigger brain for Dolly the Doly robot — LLM-powered conversation on
[Moria's LM Studio](http://192.168.50.204:1234), with **every stock behavior
preserved** and instant rollback to stock.

```
mic → touch-to-talk → VAD → Vosk ASR (on Pi)
                          ↓
                     ROUTER
              ┌──────────┴───────────┐
        stock command match      free speech
              ↓                        ↓
        Doly SDK action       qwen3.6-35b-a3b (Moria)
        (dance/drive/time/…)  → streamed reply → stock TTS voice
```

## What's preserved

- Stock doly service stays installed. `spark off` restores it in seconds.
- Stock-style voice commands (time, weather, timer, photo, fist bump,
  drive, dance, colors, sleep, battery) run **locally, instantly, offline**.
- Her stock TTS voice — she sounds like the same robot.

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

## Deploy (on the robot)

```bash
scp -r Spark/spark-brain doly@spark:/home/doly/
ssh doly@spark
cd ~/spark-brain && sudo bash deploy/install.sh
spark test        # text REPL first — verify brain pipeline
spark on          # go live
```

Requires: robot on the same LAN as Moria (or tailnet), LM Studio server
running with `qwen/qwen3.6-35b-a3b` loaded, context 8192, TTL off.

## Runtime

- **Tap the robot** → she listens (eyes change) → speak → silence ends turn.
- Free speech → Moria → streamed reply, spoken sentence by sentence.
- "Search for…" / "look up…" / "what's the latest on…" → web search →
  brain answers from fresh results (DuckDuckGo, no API key).
- Moria unreachable → she says so, **stock commands still work**.

## Files

| file | role |
|---|---|
| `prompt.md` | Spark personality v4 (source of truth) |
| `config.json` | endpoints, model, VAD thresholds, TTS voice |
| `spark/brain.py` | LM Studio streaming client + fallback model |
| `spark/router.py` | stock-command router + weather/timer + web search |
| `spark/search.py` | DuckDuckGo web search (ddgs pkg + HTML fallback) |
| `spark/commands.py` | parity table — extend from the Doly app's "Say" list |
| `spark/body.py` | guarded Doly SDK wrappers |
| `spark/ear.py` | arecord + energy VAD (pure stdlib) |

## Tuning

- False wake / clipped speech → `audio.start_rms` / `silence_ms` in config.
- Swap brains → `brain.model` (fallback: `brain.fallback_model`).
- Add stock commands → append to `COMMANDS` in `spark/commands.py`.

## Phase 1 scope (deliberate)

No vision, no wake-word (tap-to-talk), no app changes. One thing at a time.
