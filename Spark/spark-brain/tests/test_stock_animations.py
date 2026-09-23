"""Animation regressions using simulated time and strict, inert SDK doubles."""
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if not hasattr(os, "getuid"):
    os.getuid = lambda: 0

from spark.anim import AnimPlayer
from spark.body import Body
from spark.commands import match_command
from spark.router import Router
from spark.__main__ import Spark


class Rig:
    """No hardware imports. Completion takes simulated time, not instant mocks."""
    def __init__(self, directory, held=False):
        self.now, self.arm_end, self.drive_end, self.eye_end = 0., 0., 0., 0.
        self.events = []
        self.held = held
        self.has = dict(arm=True, drive=True, sound=True, eye=True, led=True)
        self._power_lock = threading.RLock()
        self._speaking_until = 0
        states = SimpleNamespace(Running="running", Completed="completed", Error="error")
        self._arm = SimpleNamespace(ArmSide=SimpleNamespace(Both=0, Left=1, Right=2),
                                    ArmState=states, set_angle=self.arm,
                                    get_state=lambda side: "running" if self.now < self.arm_end else "completed")
        self._drive = SimpleNamespace(DriveState=states,
                                      get_state=lambda: "running" if self.now < self.drive_end else "completed")
        self._eye = SimpleNamespace(is_animating=lambda: self.now < self.eye_end)
        self._snd = Mock()
        self.anim = AnimPlayer(self, {"animations": {"dir": str(directory)}})
        self.anim._sound_path = lambda kind, name: name
        self.anim._led_solid = lambda *args, **kwargs: self.events.append(("led", self.now, args))
        self.anim._stop.wait = self.advance

    def advance(self, seconds):
        self.now += seconds

    def _next_id(self):
        return 1

    def actuators_held(self):
        return self.held

    def arm(self, cmd_id, side, speed, angle, with_brake):
        assert type(speed) is int and type(angle) is int
        assert 1 <= speed <= 100 and 0 <= angle <= 200
        self.events.append(("arm", self.now, angle))
        self.arm_end = self.now + .05
        return 0

    def drive_rotate(self, degrees, speed, from_center=False):
        if self.held:
            return False
        self.events.append(("rotate", self.now, degrees))
        self.drive_end = self.now + .05
        return True

    def drive_distance(self, mm, speed):
        if self.held:
            return False
        self.events.append(("distance", self.now, mm))
        self.drive_end = self.now + .05
        return True

    def mood_eyes(self, name):
        self.events.append(("eye", self.now, name))
        self.eye_end = self.now + .05

    def play_sfx(self, path, defer):
        assert threading.current_thread() is threading.main_thread()
        self.events.append(("sound", self.now, path))
        return True

    def _wav_duration(self, path):
        return 2

    def speak(self, text, wait=True):
        self.events.append(("speak", self.now, text))
        self._speaking_until = self.now + .5
        return True

    def speaking_recently(self):
        return self.now < self._speaking_until

    def stop_everything(self):
        self.events.append(("stop", self.now, None))
        self.anim.stop()

    def play(self, name):
        with patch("spark.anim.time.monotonic", lambda: self.now), patch("spark.anim.time.time", lambda: self.now):
            return self.anim.play(name)


class StockAnimationTests(unittest.TestCase):
    def test_repeated_petting_escalates_without_cancelling_or_opening_voice(self):
        body = Body({}, hw=False)
        body.anim = Mock(petting=False)
        body.anim.playing.return_value = False
        body.mood_eyes = Mock()
        body._bump_mood = Mock()
        spark = Spark.__new__(Spark)
        spark.body, spark.talk_trigger, spark.listening = body, Mock(), False
        spark._wire_touch()
        touch = Mock()
        touch.init.return_value = 0
        with patch.dict(sys.modules, {"doly_touch": touch}):
            body._init_touch()
        callback = touch.on_touch.call_args.args[0]
        def event(at, side, state):
            with patch("spark.__main__.time.time", return_value=at):
                callback(side, state)
        # Overlapping touch pads keep independent press durations.
        event(10, "left", "Down")
        event(10.5, "right", "Down")
        event(10.8, "left", "Up")
        self.assertEqual(body._pending_pet, "petting1")
        event(11.3, "right", "Up")
        self.assertEqual(body._pending_pet, "petting2")
        body.anim.playing.return_value = True
        body.anim.petting = True
        for at in (12, 13):
            event(at, "left", "Down")
            event(at+.15, "left", "Up")
        self.assertEqual(body._pending_pet, "petting3")
        body.anim.stop.assert_not_called()
        spark.talk_trigger.set.assert_not_called()
        body.anim.playing.return_value = False
        body.drain_anims()
        body.anim.play.assert_called_once_with("petting3", blocking=True)
        self.assertIsNone(body._pending_pet)
        # A touch during a dance still cancels immediately.
        body.anim.playing.return_value = True
        body.anim.petting = False
        event(15, "left", "Down")
        event(15.2, "left", "Up")
        body.anim.stop.assert_called_once()

    def test_pet_animation_exempts_touch_only_when_it_contains_no_motors(self):
        with tempfile.TemporaryDirectory() as tmp:
            for motor in (False, True):
                self.program(tmp, '<block type="delay_ms"><field name="delay_ms">50</field></block>'
                             + ('<block type="arm_set_angle"/>' if motor else ''))
                Path(tmp, "test.xml").replace(Path(tmp, "petting1.xml"))
                rig = Rig(tmp)
                flags = []
                def wait(seconds):
                    flags.append(rig.anim.petting)
                    rig.advance(seconds)
                rig.anim._stop.wait = wait
                self.assertTrue(rig.play("petting1"))
                self.assertTrue(flags)
                self.assertEqual(set(flags), {not motor})
                self.assertFalse(rig.anim.petting)

    def program(self, directory, xml):
        Path(directory, "test.xml").write_text(
            '<xml xmlns="https://developers.google.com/blockly/xml">'
            '<block type="start_animation"/>' + xml + '</xml>')

    def test_namespaced_music_overlaps_choreography_and_callback_waits(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.program(tmp, '''
              <block type="sound"><field name="name">SALSA</field>
                <statement name="complete_statement"><block type="led">
                  <field name="color_main">#000000</field></block></statement></block>
              <block type="arm_set_angle"><field name="angle">110.0</field><field name="start">0</field></block>
              <block type="repeat"><field name="times">2</field><field name="start">1</field>
                <statement name="repeat_statement"><block type="drive_rotate_left">
                  <field name="driveRotate">90</field><field name="start">1</field></block></statement></block>''')
            rig = Rig(tmp)
            self.assertTrue(rig.play("test"))
            arms = [e for e in rig.events if e[0] == "arm"]
            turns = [e for e in rig.events if e[0] == "rotate"]
            self.assertLess(arms[0][1], 2)  # music must NOT finish before movement
            self.assertEqual([e[2] for e in turns], [-90, -90])
            self.assertGreaterEqual(turns[0][1], arms[0][1] + .05)
            self.assertGreaterEqual([e[1] for e in rig.events if e[0] == "led"][0], 2)
            self.assertFalse(rig.anim.playing())

    def test_dock_allows_show_but_never_arms_or_wheels(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.program(tmp, '''<block type="arm_set_angle"><field name="angle">140</field></block>
                <block type="drive_distance"><field name="distance">100</field></block>
                <block type="eye_animations"><field name="animation">HAPPY</field></block>
                <block type="sound"><field name="name">SALSA</field></block>''')
            rig = Rig(tmp, held=True)
            self.assertTrue(rig.play("test"))
            self.assertEqual({e[0] for e in rig.events}, {"eye", "sound"})

    def test_empty_errors_timeouts_and_cancel_are_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.program(tmp, '')
            rig = Rig(tmp)
            self.assertFalse(rig.play("test"))
            self.program(tmp, '<block type="drive_rotate_left"><field name="driveRotate">3</field></block>')
            rig._drive.get_state = lambda: "running"
            self.assertFalse(rig.play("test"))
            self.assertIn("stop", [e[0] for e in rig.events])
            self.program(tmp, '<block type="sound"><field name="name">missing</field></block>')
            rig.play_sfx = lambda *args, **kwargs: False
            self.assertFalse(rig.play("test"))
            self.program(tmp, '<block type="delay_ms"><field name="delay_ms">5000</field></block>')
            rig.anim._stop.wait = lambda seconds: rig.anim.stop()
            self.assertFalse(rig.play("test"))

    def test_named_dances_negation_and_imagine_dispatch_once_on_main_thread(self):
        for phrase, action in (("do the twist", "dance_twist"), ("salsa dance", "dance_salsa"),
                               ("move left", "left"), ("move right", "right")):
            self.assertEqual(match_command(phrase)[0]["action"], action)
            self.assertIsNone(match_command("don't " + phrase)[0])
        body = Mock()
        self.assertEqual(match_command("No, you're not on a charger. Come here.")[0]["action"], "come_here")
        for text in ("No, you're not on a charger. Don't come here.", "Don't say come here"):
            self.assertIsNone(match_command(text)[0])
        body.dance.side_effect = lambda variant: self.assertIs(threading.current_thread(), threading.main_thread()) or True
        router = Router({}, body, None, None)
        self.assertTrue(router.handle("imagine you are doing exercise"))
        body.dance.assert_called_once_with("workout")
        body.reset_mock()
        router.handle("come to me")
        body.come_here.assert_called_once()
        body.drive_guarded.assert_not_called()
        router.handle("don't imagine you are dancing")
        body.dance.assert_not_called()

    def test_movement_corrections_use_actual_controller_state(self):
        body = Mock(docked=False)
        body.refresh_power.return_value = False
        body.come_here.return_value = "ambiguous"
        router = Router({}, body, None, None)
        router.handle("come here")
        for phrase in ("I'm right here", "There's only one person in front of you"):
            self.assertTrue(router.handle(phrase))
            self.assertIn("camera couldn't pick", body.speak.call_args.args[0])
        self.assertTrue(router.handle("No, you're not on a charger"))
        self.assertIn("off the charger", body.speak.call_args.args[0])
        body.docked = True
        self.assertTrue(router.handle("I'm right here"))
        self.assertIn("parked", body.speak.call_args.args[0])
        body.come_here.assert_called_once()  # explanations never move the robot
        self.assertFalse(router.handle("Tell me about Saturn"))

    def test_replaced_arm_sound_and_speech_completions(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.program(tmp, '''
                <block type="arm_set_angle"><field name="angle">90</field><field name="side">0</field></block>
                <block type="arm_set_angle"><field name="angle">0</field><field name="side">1</field></block>
                <block type="sound"><field name="name">long</field>
                    <statement name="complete_statement"><block type="led"/></statement></block>
                <block type="sound"><field name="name">short</field></block>
                <block type="speak"><field name="say">hello</field><field name="start">1</field></block>
                <block type="eye_animations"><field name="animation">HAPPY</field><field name="start">1</field></block>''')
            rig = Rig(tmp)
            ends = {1: 0, 2: 0}
            def move(cmd_id, side, speed, angle, with_brake):
                for which in ((1, 2) if side == 0 else (side,)):
                    ends[which] = rig.now + (2 if angle == 90 else .05)
                return 0
            rig._arm.set_angle = move
            rig._arm.get_state = lambda side: "running" if rig.now < ends[side] else "completed"
            rig._wav_duration = lambda name: 8 if name == "long" else .2
            self.assertTrue(rig.play("test"))
            self.assertGreaterEqual(rig.now, 2)
            self.assertLess(rig.now, 3)  # cancelled long soundtrack must not hold playback open
            self.assertNotIn("led", [e[0] for e in rig.events])  # no cancelled sound's completion callback
            speech = next(e[1] for e in rig.events if e[0] == "speak")
            eye = next(e[1] for e in rig.events if e[0] == "eye")
            self.assertGreaterEqual(eye, speech + .5)

    def test_gestures_wait_for_contact_and_docked_gestures_do_not_move(self):
        body = Body({}, hw=False)
        body.actuators_held = Mock(return_value=True)
        body.arm_angle = Mock(return_value=True)
        self.assertFalse(body.high_five())
        body.arm_angle.assert_not_called()
        body.actuators_held.return_value = False
        body.anim = Mock()
        def ready_contact(name):
            if name == "fist_ready":
                body._sensor_react("imu", "ShockLight")
                self.assertFalse(body._interaction_contact.is_set())
            return True
        body.anim.play.side_effect = ready_contact
        def contact(timeout):
            self.assertEqual(body._interaction, "fist_bump")
            body._sensor_react("tof", "ObjectComing")
            self.assertFalse(body._interaction_contact.is_set())
            body._sensor_react("imu", "ShockLight")
            return body._interaction_contact.is_set()
        body._interaction_contact.wait = contact
        self.assertTrue(body.fist_bump())
        self.assertEqual([c.args[0] for c in body.anim.play.call_args_list], ["fist_ready", "fist_bump"])
        self.assertEqual(body.arm_angle.call_args_list[0].args[0], 90)
        self.assertIsNone(body._interaction)
        touch = Mock()
        touch.init.return_value = 0
        with patch.dict(sys.modules, {"doly_touch": touch}):
            body._init_touch()
        body._touch_cb = Mock()
        callback = touch.on_touch.call_args.args[0]
        callback(0, "Down")
        callback(0, "Up")
        body.anim.stop.assert_called_once()
        body._touch_cb.assert_not_called()

    def test_requests_and_complaints_are_not_interchangeable(self):
        for phrase in ("Okay, why don't you spin again?", "Please spin", "Can you turn left?",
                       "You didn't spin. Spin again.", "We didn't touch you. Give me another fist bump."):
            self.assertIsNotNone(match_command(phrase)[0], phrase)
        for phrase in ("You didn't spin.", "Why didn't you spin?", "I said spin.",
                       "Don't say spin", "Never turn left", "Don't come here", "I like salsa",
                       "Don't. Spin.", "Don't say. Come here."):
            self.assertIsNone(match_command(phrase)[0], phrase)


if __name__ == "__main__":
    unittest.main()
