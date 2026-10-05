"""Voice routing changes both playback paths and survives restart."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if not hasattr(os, "getuid"):
    os.getuid = lambda: 0

from spark.body import Body
from spark.router import Router
from spark import commands, volume


class VolumeTests(unittest.TestCase):
    def router(self, directory):
        router = Router.__new__(Router)
        router.body = Body({"sounds": {"volume": 90}, "state_dir": directory}, hw=False)
        router.body.has["sound"] = True
        router.body._snd = Mock()
        router.body._snd.set_volume.return_value = 0
        router.body.speak = Mock()
        # Volume must take priority even when a game/timer expects a number.
        router.pet = Mock()
        router._pending = {"kind": "timer"}
        return router

    def test_spoken_percentage_updates_sdk_pcm_gain_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.router(directory)
            self.assertTrue(router.handle("Hey Spark, volume forty-five percent."))
            router.body._snd.set_volume.assert_called_once_with(45)
            self.assertEqual(router.body.cfg["sounds"]["volume"], 45)
            router.body.speak.assert_called_once_with("Volume 45 percent.")
            router.pet.handle.assert_not_called()
            restart = Body({"sounds": {"volume": 90}, "state_dir": directory}, hw=False)
            self.assertEqual(restart.volume, 45)

    def test_steps_are_fifteen_points_and_clamp_at_both_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.router(directory)
            for phrase, expected in (("turn it up", 100), ("turn it down", 85),
                                     ("volume 0%", 0), ("turn it down", 0),
                                     ("turn it up", 15)):
                self.assertTrue(router.handle(phrase))
                self.assertEqual(router.body.volume, expected)
            for phrase, expected in (("actually turn down the volume", 0),
                                     ("please turn up your volume", 15),
                                     ("speak louder", 30), ("maximum volume", 100)):
                self.assertTrue(router.handle(phrase))
                self.assertEqual(router.body.volume, expected)

    def test_invalid_unrelated_or_negated_requests_do_not_change_volume(self):
        for phrase in ("turn left", "turn the lights down", "don't turn it up",
                       "what is the volume of a cylinder", "explain volume 50"):
            self.assertIsNone(volume.intent(commands.normalize(phrase)))
        with tempfile.TemporaryDirectory() as directory:
            router = self.router(directory)
            router.handle("volume one hundred fifty percent")
            self.assertEqual(router.body.volume, 90)
            router.body._snd.set_volume.assert_not_called()

    def test_sdk_failure_does_not_save_or_claim_success(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.router(directory)
            router.body._snd.set_volume.return_value = -1
            router.handle("volume 50")
            self.assertEqual(router.body.volume, 90)
            self.assertFalse((Path(directory) / "volume.json").exists())
            self.assertIn("couldn't change", router.body.speak.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
