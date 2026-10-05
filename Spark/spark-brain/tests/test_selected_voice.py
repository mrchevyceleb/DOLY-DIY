"""The selected Qwen voice must preserve speech when its server is unavailable."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if not hasattr(os, "getuid"):
    os.getuid = lambda: 0
from spark.body import Body


class SelectedVoiceTests(unittest.TestCase):
    def test_qwen_failure_uses_moria_piper_and_applies_effects_once(self):
        body = Body({"tts": {"server_url": "qwen", "voice_name": "warm-soft-robot",
                            "server_timeout_s": 30, "fallback_server_url": "piper",
                            "fallback_voice_name": "hfc", "fallback_timeout_s": 6}}, hw=False)
        body._produce_server = Mock(side_effect=[TimeoutError(), None])
        body._produce_piper = Mock()
        body._apply_fx = Mock()
        body._produce_speech("Hello Matt.")
        calls = body._produce_server.call_args_list
        self.assertEqual([call.args[1] for call in calls], ["qwen", "piper"])
        self.assertEqual(calls[1].args[2]["voice_name"], "hfc")
        self.assertEqual(calls[1].args[2]["server_timeout_s"], 6)
        body._produce_piper.assert_not_called()
        body._apply_fx.assert_called_once()

    def test_qwen_success_never_runs_fallback_and_applies_effects_once(self):
        body = Body({"tts": {"server_url": "qwen", "voice_name": "warm-soft-robot",
                            "fallback_server_url": "piper"}}, hw=False)
        body._produce_server = Mock()
        body._apply_fx = Mock()
        body._produce_speech("Hello Matt.")
        body._produce_server.assert_called_once()
        body._apply_fx.assert_called_once()


if __name__ == "__main__":
    unittest.main()
