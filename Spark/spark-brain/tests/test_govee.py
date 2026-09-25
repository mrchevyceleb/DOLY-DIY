"""Govee room-light voice control: intent routing + client basics."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spark import commands as cmds
from spark.govee import GoveeLights, color_rgb
from spark.router import Router

CFG = {"state_dir": None, "govee": {"enabled": True, "api_key": None}}


def _router(govee=None):
    r = Router.__new__(Router)
    r.cfg = dict(CFG)
    r.body = Mock()
    r.brain = Mock()
    r.memory = Mock()
    r.llm_reply = None
    r.last_motion_result = None
    r._last_motion_at = 0
    r._timers = []
    r.govee = govee if govee is not None else Mock(spec=GoveeLights)
    r.govee.enabled = True
    return r


class ColorMapTests(unittest.TestCase):
    def test_body_color_names_map_to_rgb(self):
        for name in cmds.COLORS:
            self.assertIsNotNone(color_rgb(name), name)
        self.assertEqual(color_rgb("warm white"), (255, 180, 100))

    def test_unknown_color(self):
        self.assertIsNone(color_rgb("chartreuse-ish"))


class RouterIntentTests(unittest.TestCase):
    def _say(self, text, govee):
        r = _router(govee)
        self.assertTrue(r.handle(text))
        return r.body.speak.call_args[0][0]

    def test_on_off_and_phrasings(self):
        g = Mock(spec=GoveeLights)
        g.enabled = True
        g.turn.side_effect = lambda on, label="all": "Lights on!" if on else "Lights off."
        self.assertEqual(self._say("turn on my lights", g), "Lights on!")
        self.assertTrue(g.turn.call_args[0][0])
        self.assertEqual(self._say("kill the lights", g), "Lights off.")
        self.assertFalse(g.turn.call_args[0][0])
        self.assertEqual(self._say("lights off", g), "Lights off.")

    def test_color_and_brightness(self):
        g = Mock(spec=GoveeLights)
        g.enabled = True
        g.color.return_value = "Lights blue."
        g.brightness.return_value = "Lights at 30 percent."
        self.assertEqual(self._say("set my lights to blue", g), "Lights blue.")
        g.color.assert_called_once_with("Blue")
        self.assertEqual(self._say("dim my lights to thirty percent", g),
                         "Lights at 30 percent.")
        self.assertEqual(g.brightness.call_args[0][0], 30)

    def test_warmer_cooler(self):
        g = Mock(spec=GoveeLights)
        g.enabled = True
        g.color_temp.return_value = "Warmer light."
        self.assertEqual(self._say("make my lights warmer", g), "Warmer light.")
        g.color_temp.assert_called_once_with(3200)

    def test_her_own_lights_stay_hers(self):
        g = Mock(spec=GoveeLights)
        g.enabled = True
        r = _router(g)
        r.body.eye_color = Mock(return_value=True)
        r.body.led_color = Mock(return_value=True)
        self.assertTrue(r.handle("change your light color to red"))
        g.turn.assert_not_called()
        g.color.assert_not_called()

    def test_disabled_govee_falls_through(self):
        r = _router()
        r.govee.enabled = False
        r.body.eye_color = Mock(return_value=False)
        r.body.led_color = Mock(return_value=False)
        cmd, score = cmds.match_command("dance")
        self.assertTrue(r.handle("dance") or True)  # must not crash
        r.govee.turn.assert_not_called()


class ClientBasicsTests(unittest.TestCase):
    def test_brightness_clamps(self):
        g = GoveeLights({"govee": {"enabled": True}, "state_dir": None})
        g._apply = Mock(return_value=("ok", ""))
        self.assertEqual(g.brightness(250), "Lights at 100 percent.")
        self.assertEqual(g.brightness(0), "Lights at 1 percent.")
        self.assertEqual(g._apply.call_args_list[0][0][1], 100)
        self.assertEqual(g._apply.call_args_list[1][0][1], 1)

    def test_no_devices_message(self):
        g = GoveeLights({"govee": {"enabled": True}, "state_dir": None})
        g.devices = Mock(return_value=[])
        self.assertIn("can't find", g.turn(True))


if __name__ == "__main__":
    unittest.main()
