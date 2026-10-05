# Moria services (192.168.50.204 — DGX Spark, aarch64)

Three local HTTP services back Spark's audio pipeline. The selected Qwen
voice falls back to Moria Piper, then on-device speech when unavailable.
ASR also retains its on-device fallback.

| service | port | purpose | install |
|---|---|---|---|
| `piper-tts` | 8398 | POST text → WAV (piper, ~0.4s) | `/opt/piper-moria/` |
| `parakeet-server` | 8399 | POST wav → text (parakeet-tdt 0.6B v3, ~0.5s) | `/opt/whisper-moria/` |
| `qwen-voice` | 8400 | POST text → raw WAV, saved Warm identity | `~/spark-qwen-voice/` |

## selected Qwen voice setup

Use a separate Python 3.12 environment; the faster runtime uses `qwen-tts-hf`
and must not share the audition environment's upstream `qwen-tts` installation.
Pin Transformers 5.15.1: 5.18.0 failed loading the Mimi tokenizer here.

```sh
mkdir -p ~/spark-qwen-voice/voice
python3 -m venv ~/spark-qwen-voice/venv
~/spark-qwen-voice/venv/bin/pip install -r requirements-qwen-voice.txt
# Copy qwen_voice_server.py to ~/spark-qwen-voice/.
# Copy voices/warm-soft-robot/{reference.wav,voice.json} to its voice/ folder.
sudo cp qwen-voice.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now qwen-voice
curl -f http://192.168.50.204:8400/health
```

The unit's paths/user are specific to Matt's Moria. The native CUDA 13 BF16
backend uses the 1.7B Base model and caches the reference features. Health is
available only after warmup. Concurrent generation returns 503 immediately
so the robot can use Piper. Completed lines have a bounded 32-entry cache.
Request readers are capped at eight; excess connections close immediately.
Pitch +0.75 semitone and robot mix 0.16 are applied on the Pi exactly once;
the server returns unprocessed audio. A clone is conditioned on the chosen
take; new wording need not reproduce every detail of the audition delivery.

With `tts.server_streaming=true`, `/stream` returns framed mono PCM16 at
24 kHz. The Pi applies stateful effects and feeds one continuous `aplay`
process per sentence through its existing default/dmix speaker route.
An explicit zero-length packet marks success; premature EOF is failure.
Before audio starts, failed streams go straight to Piper; after playback
starts and the first PCM write completes, the sentence is not replayed.
A pipe failure during that first write can still trigger a repeated fallback.
`speak(wait=False)` keeps its original
complete-WAV semantics. Set `server_streaming=false` to restore WAV playback.
This does not implement duplex audio, echo cancellation or general speech interruption.
Stock actions still use the existing SDK controller.

## piper-tts setup

    curl -L -o /tmp/piper.tar.gz https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz
    sudo mkdir -p /opt/piper-moria/voices && sudo tar xzf /tmp/piper.tar.gz -C /opt/piper-moria
    # voices from huggingface.co/rhasspy/piper-voices (v1.0.0):
    # hfc_female-medium, kathleen-low, cori-high, lessac-high -> /opt/piper-moria/voices
    sudo cp tts_server.py /opt/piper-moria/
    sudo cp piper-tts.service /etc/systemd/system/
    sudo systemctl enable --now piper-tts

## parakeet-server setup

    curl -L -o /opt/whisper-moria/models/ggml-parakeet-tdt-0.6b-v3-q8_0.bin \
      https://huggingface.co/ggml-org/parakeet-GGUF/resolve/main/ggml-parakeet-tdt-0.6b-v3-q8_0.bin
    sudo cp parakeet_server.py parakeet_engine.py parakeet_bridge.cpp /opt/whisper-moria/
    sudo g++ -O2 -shared -fPIC /opt/whisper-moria/parakeet_bridge.cpp \
      -I/opt/whisper-moria/include -I/opt/whisper-moria/ggml/include \
      -L/opt/whisper-moria/build/bin -lparakeet -lggml -lggml-base \
      -Wl,-rpath,/opt/whisper-moria/build/bin -o /opt/whisper-moria/libspark_asr.so
    sudo cp parakeet-server.service /etc/systemd/system/
    sudo systemctl disable --now whisper-server   # parked, rollback path
    sudo systemctl enable --now parakeet-server

The CPU model now stays loaded in an isolated worker. Compile the shim against
the headers/libraries of this installed parakeet build; rebuild it when upgrading
that build. Normal requests reuse one serialized context with no prior transcript.
Input/output IPC share a six-second hard deadline, native inference has a
five-second abort callback, and a failed worker is killed and reloaded on the next
request. The HTTP endpoint and systemd unit remain the same. WAVs are limited to
30 seconds, mono PCM16, 8–48 kHz, and resampled to 16 kHz before inference.

`whisper-server.service` (whisper.cpp, ggml-large-v3-turbo) stays
installed but disabled — Moria's whisper.cpp build has no GPU backend,
so large models are too slow; parakeet q8_0 on CPU is both faster and
more accurate for English.
