"""Observed wake-name-only decode must not swallow a light command."""
import os
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if not hasattr(os, "getuid"):
    os.getuid = lambda: 0

from spark.__main__ import Spark
from spark.ear import WakeResult, strip_wake_prefix
from spark.govee import GoveeLights
from spark.router import Router


class LightHandoffTests(unittest.TestCase):
    def listener(self):
        spark = Spark.__new__(Spark)
        spark.cfg = {"audio": {"sample_rate": 16000}, "asr": {"server_url": "test"}}
        spark.whisper = Mock(last_source="server")
        audio = struct.pack("<16000h", *([3000] * 16000))
        return spark, audio, WakeResult("hey spark", prefix_pcm=audio)

    def test_replayed_wake_name_waits_for_and_executes_dim_command(self):
        spark, audio, wake = self.listener()
        spark.whisper.transcribe_pcm.side_effect = ["A spark.", "Dim the lights."]
        with patch("spark.ear.record_utterance", side_effect=[audio, audio]) as capture:
            text, _ = spark._listen_command(Mock(noise_floor=500), Mock(), wake)
        self.assertEqual(text, "Dim the lights.")
        self.assertEqual(capture.call_count, 2)
        router = Router.__new__(Router)
        router.cfg = {"govee": {"enabled": True}}
        router.body, router.brain = Mock(), Mock()
        router.govee = Mock(spec=GoveeLights, enabled=True)
        router.govee.brightness.return_value = "Lights at 30 percent."
        self.assertTrue(router.handle(text))
        router.govee.brightness.assert_called_once_with(30)
        router.body.speak.assert_called_once_with("Lights at 30 percent.")
        router.brain.chat.assert_not_called()

    def test_name_alone_with_no_command_stays_out_of_conversation(self):
        spark, audio, wake = self.listener()
        spark.whisper.transcribe_pcm.return_value = "A Sparky!"
        with patch("spark.ear.record_utterance", side_effect=[audio, b""]):
            text, _ = spark._listen_command(Mock(noise_floor=500), Mock(),
                                            WakeResult("hey sparky", prefix_pcm=audio))
        self.assertEqual(text, "")

    def test_article_alias_is_gated_by_authorized_wake_and_name_only(self):
        self.assertEqual(strip_wake_prefix("A spark.", "hey spark"), "")
        self.assertEqual(strip_wake_prefix("A spark.", "hey bark"), "A spark.")
        self.assertEqual(strip_wake_prefix("A spark."), "A spark.")
        sentence = "A spark started the fire."
        self.assertEqual(strip_wake_prefix(sentence, "hey spark"), sentence)

    def test_light_questions_and_negation_never_act_on_the_lamps(self):
        router = Router.__new__(Router)
        router.cfg = {"govee": {"enabled": True, "shortcuts": {"gaming": {"color": "blue"}}}}
        router.govee = Mock(spec=GoveeLights, enabled=True)
        for text in ("don't dim the lights", "how do i dim the lights", "what are gaming lights",
                     "tell me how to dim the lights to fifty percent", "can you explain how to dim the lights",
                     "tell me if the lights are on", "let me know if the lights off command works"):
            self.assertIsNone(router._govee_lights(text))
            self.assertIsNone(router.light_shortcut(text))
        router.govee.brightness.assert_not_called()
        router.govee.color.assert_not_called()
        router._govee_lights("dim the lights to fifty five percent")
        router.govee.brightness.assert_called_once_with(55)
        router.govee.reset_mock()
        for number in ("one hundred fifty", "two hundred", "two thousand", "150"):
            self.assertIn("zero to one hundred", router._govee_lights("dim the lights to " + number + " percent"))
        router.govee.brightness.assert_not_called()


if __name__ == "__main__":
    unittest.main()
