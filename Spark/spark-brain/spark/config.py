"""Spark brain service — configuration loader."""
import json
import os
import pathlib

_DEFAULTS = {
    "brain": {
        "lm_studio_url": "http://192.168.50.204:1234",
        "model": "qwen/qwen3.6-35b-a3b",
        "fallback_model": "qwen/qwen3.5-9b",
        "max_tokens": 250,
        "temperature": 0.7,
        "reasoning_effort": "none",
        "history_turns": 12,
        "request_timeout_s": 120,
    },
    "audio": {
        "input_device": "default",
        "sample_rate": 16000,
        "silence_ms": 1200,
        "max_utterance_ms": 12000,
        "start_rms": 900,
        "stop_rms": 500,
    },
    "asr": {"model_path": "/opt/spark/vosk-model-small-en-us-0.15", "sample_rate": 16000},
    "tts": {"voice_model": 1, "volume": 1},
    "location": {"lat": None, "lon": None},
    "state_dir": "/opt/spark/state",
}

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None):
    path = pathlib.Path(path) if path else _ROOT / "config.json"
    data = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    cfg = _merge(_DEFAULTS, data)
    cfg["prompt"] = (_ROOT / "prompt.md").read_text(encoding="utf-8").strip()
    cfg["root"] = str(_ROOT)
    os.makedirs(cfg["state_dir"], exist_ok=True)
    return cfg
