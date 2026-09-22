# Moria services (192.168.50.204 — DGX Spark, aarch64)

Two stdlib-python HTTP services back Spark's realtime audio pipeline.
Both are drop-in: the Pi falls back to on-device synth / ASR when Moria
is unreachable.

| service | port | purpose | install |
|---|---|---|---|
| `piper-tts` | 8398 | POST text → WAV (piper, ~0.4s) | `/opt/piper-moria/` |
| `parakeet-server` | 8399 | POST wav → text (parakeet-tdt 0.6B v3, ~0.5s) | `/opt/whisper-moria/` |

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
    sudo cp parakeet_server.py /opt/whisper-moria/
    sudo cp parakeet-server.service /etc/systemd/system/
    sudo systemctl disable --now whisper-server   # parked, rollback path
    sudo systemctl enable --now parakeet-server

`whisper-server.service` (whisper.cpp, ggml-large-v3-turbo) stays
installed but disabled — Moria's whisper.cpp build has no GPU backend,
so large models are too slow; parakeet q8_0 on CPU is both faster and
more accurate for English.
