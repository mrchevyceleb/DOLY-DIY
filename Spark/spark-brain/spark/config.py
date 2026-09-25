"""Spark brain service — configuration loader."""
import json
import os
import pathlib
import sys

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
    "web": {
        "enabled": True,       # brain-triggered SEARCH/READ web tools
        "max_hops": 2,         # tool calls per reply (search, then read)
        "page_max_chars": 3500,
        "page_timeout_s": 6,
    },
    "govee": {
        "enabled": True,       # room-light voice control (LAN first)
        "api_key": None,       # Govee Home app: Me -> Apply for API Key
    },
    "alerts": {
        "celebrate": True,       # fired timers/alarms party until stopped
        "celebrate_max_s": 600,  # hard cap so an unattended alarm can't dance forever
        "party_lights": True,    # cycle the Govee room lights too
    },
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
    # robot-local overlay (gitignored): secrets like the Govee API key
    # live here instead of the public repo, and redeploying config.json
    # from the repo can never wipe them.
    local = _ROOT / "config.local.json"
    if local.exists():
        try:
            data = _merge(data, json.loads(local.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"[config] local overlay failed: {e}", file=sys.stderr)
    cfg = _merge(_DEFAULTS, data)
    cfg["prompt"] = (_ROOT / "prompt.md").read_text(encoding="utf-8").strip()
    cfg["root"] = str(_ROOT)
    os.makedirs(cfg["state_dir"], exist_ok=True)
    return cfg
