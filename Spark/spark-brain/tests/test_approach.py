"""Approach decisions and physical stop conditions with inert hardware."""
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if not hasattr(os, "getuid"):
    os.getuid = lambda: 0
from spark.approach import Approach
from spark.body import Body
from spark.person_vision import Person
from spark.__main__ import Spark


class ApproachTests(unittest.TestCase):
    def rig(self, observations):
        b = Body({}, hw=False)
        b.actuators_held = Mock(return_value=False)
        b.battery_pct = Mock(return_value=50)
        b._edge_gaps = Mock(return_value=[])
        b._hazard_active = Mock(return_value=False)
        b.approach_proximity = Mock(return_value=None)
        b.drive_guarded = Mock(return_value="ok")
        b.drive_stop = Mock(return_value=True)
        b.rotate_guarded = Mock(return_value="ok")
        camera = Mock()
        camera.observe.side_effect = observations
        return b, Approach(b, camera)

    def test_search_only_turns_and_confirmed_person_centers_before_approach(self):
        right = Person(.7, .2, .2, .5, .8)
        center = Person(.388, .2, .2, .5, .8)
        near = Person(.025, .01, .95, .98, .8)
        b, approach = self.rig([[], [right], [right], [center], [center], [near], [near]])
        self.assertEqual(approach.run(), "near")
        self.assertEqual(b.rotate_guarded.call_args_list[0].args, (13.5,))
        self.assertGreater(b.rotate_guarded.call_args_list[1].args[0], 0)
        b.drive_guarded.assert_called_once()
        self.assertEqual(b.drive_guarded.call_args.args, (40,))
        b, approach = self.rig([[]]*12)
        self.assertEqual(approach.run(), "not_found")
        self.assertEqual(b.rotate_guarded.call_count, 11)
        b.drive_guarded.assert_not_called()

        # Observed floor behavior: each requested degree turns about 2.2
        # camera-bearing degrees. Full-error turns oscillated indefinitely.
        import math
        bearing = [21.0]
        b, approach = self.rig([])
        def observe():
            width = .9 if b.drive_guarded.called else .2
            return [Person((312.37+369.45*math.tan(math.radians(bearing[0])))/640-width/2,
                           .01, width, .97 if b.drive_guarded.called else .5, .8)]
        approach.camera.observe.side_effect = observe
        def turn(degrees, **kwargs):
            bearing[0] -= degrees*2.2
            return "ok"
        b.rotate_guarded.side_effect = turn
        self.assertEqual(approach.run(), "near")
        self.assertLessEqual(b.rotate_guarded.call_count, 3)
        self.assertTrue(all(abs(c.args[0]) <= 10 for c in b.rotate_guarded.call_args_list))
        b.drive_guarded.assert_called_once()
        wide = Person(.23, .1, .52, .7, .8)  # observed framing at about three feet
        tiny = Person(.1, .27, .057, .086, .51)  # shelf false-positive in real frame
        b, approach = self.rig([[wide, tiny], [wide, tiny], [near], [near]])
        self.assertEqual(approach.run(), "near")
        b.drive_guarded.assert_called_once()

    def test_no_motion_on_ambiguous_target_loss_edge_power_or_stop(self):
        person = Person(.388, .2, .2, .5, .8)
        for observations, expected in (([[person, person]], "ambiguous"),
                                       ([[person], []], "lost")):
            b, approach = self.rig(observations)
            self.assertEqual(approach.run(), expected)
            b.drive_guarded.assert_not_called()
            b.rotate_guarded.assert_not_called()
        for reason in ("cancelled", "power", "edge", "obstacle"):
            b, approach = self.rig([])
            if reason == "cancelled":
                b._approach_stop.set()
            elif reason == "power":
                b.battery_pct.return_value = 5
            elif reason == "edge":
                b._edge_gaps.return_value = ["Back_Left"]
            else:
                b.approach_proximity.return_value = reason
            self.assertEqual(approach.run(), reason)
            b.drive_guarded.assert_not_called()
            b.rotate_guarded.assert_not_called()

    def test_transient_sensor_noise_brakes_and_reobserves_with_bounded_retry(self):
        person = Person(.388, .2, .2, .5, .8)
        near = Person(.025, .01, .95, .98, .8)
        b, approach = self.rig([[person]]*4 + [[near]]*2)
        b.drive_guarded.side_effect = ["sensor", "ok"]
        with patch("spark.approach.time.sleep"):
            self.assertEqual(approach.run(), "near")
        b.drive_stop.assert_called_once()
        self.assertEqual(b.drive_guarded.call_count, 2)
        self.assertEqual(approach.camera.observe.call_count, 6)
        b, approach = self.rig([])
        b.approach_proximity.return_value = "sensor"
        approach.sensor_retries = 3
        self.assertEqual(approach.run(), "sensor")
        b.drive_guarded.assert_not_called()

    def test_proximity_blocks_missing_stale_faulted_or_close_samples(self):
        b = Body({}, hw=False)
        b.has["tof"] = True
        b._tof = Mock()
        def sensors(error=0, distance=255, stamp=123):
            return [SimpleNamespace(side=s, error=error, range_mm=distance, update_ms=stamp)
                    for s in ("left", "right")]
        b._tof.get_sensors_data.return_value = sensors()
        with patch("spark.body.time.monotonic", return_value=100):
            self.assertIsNone(b.approach_proximity())
        with patch("spark.body.time.monotonic", return_value=100.5):
            self.assertEqual(b.approach_proximity(), "sensor")
        b._tof.get_sensors_data.return_value = sensors(stamp=124, distance=100)
        self.assertEqual(b.approach_proximity(), "obstacle")
        b._tof.get_sensors_data.return_value = sensors(stamp=125, error=18)
        self.assertEqual(b.approach_proximity(), "sensor")
        for error in (6, 7, 13, 15):
            b._approach_sensor_stamps = {}
            b._tof.get_sensors_data.return_value = sensors(stamp=126, error=error, distance=-1)
            with patch("spark.body.time.monotonic", return_value=200):
                self.assertIsNone(b.approach_proximity())
            with patch("spark.body.time.monotonic", return_value=200.5):
                self.assertEqual(b.approach_proximity(), "sensor")
        for error in (1, 5, 12, 14, 16, 18):
            b._tof.get_sensors_data.return_value = sensors(stamp=127, error=error, distance=-1)
            self.assertEqual(b.approach_proximity(), "sensor")
        b._tof.get_sensors_data.return_value = [sensors(stamp=128, error=6, distance=-1)[0],
                                               sensors(stamp=128, distance=70)[1]]
        self.assertEqual(b.approach_proximity(), "obstacle")

    def test_mid_drive_interlock_brakes_and_never_credits_travel(self):
        b = Body({}, hw=False)
        b.has["drive"] = True
        b._drive = Mock()
        b.ensure_mobility = Mock(return_value=True)
        b._motion_allowed = Mock(return_value=True)
        b._hazard_active = Mock(return_value=False)
        b._pose_update = Mock()
        interlock = Mock(side_effect=[None, "cancelled"])
        with patch("spark.body.time.sleep"):
            self.assertEqual(b.drive_guarded(40, interlock=interlock), "cancelled")
        b._drive.go_distance.assert_called_once()
        self.assertEqual(b._drive.free_drive.call_count, 2)
        b._pose_update.assert_not_called()

    def test_edge_hold_reports_edge_not_power(self):
        # two front gaps on a table: the refusal must blame the edge,
        # not claim she needs to charge (regression: 2026-09-25 table test)
        b = Body({}, hw=False)
        b.hw = True
        b.has["drive"] = True
        b.has["edge"] = True
        b.docked = False
        b.actuators_held = Mock(return_value=True)
        b._edge_gaps = Mock(return_value=["Front_Left", "Front_Right"])
        with patch("spark.person_vision.PersonCamera") as camera:
            self.assertEqual(b.come_here(), "edge")
            camera.assert_not_called()
        b._edge_gaps = Mock(return_value=[])
        with patch("spark.person_vision.PersonCamera") as camera:
            self.assertEqual(b.come_here(), "power")
            camera.assert_not_called()

    def test_voice_stop_and_docked_command_never_open_camera(self):
        b = Body({}, hw=False)
        b.hw = True
        b.has["drive"] = True
        b.docked = True
        b.actuators_held = Mock(return_value=True)
        with patch("spark.person_vision.PersonCamera") as camera:
            self.assertEqual(b.come_here(), "docked")
            camera.assert_not_called()
        spark = Spark.__new__(Spark)
        spark.cfg = {"audio": {"sample_rate": 16000}}
        mic, rec = Mock(), Mock()
        mic.drain_pending.return_value = [bytes(640)]
        decoder = rec._kaldi_cls.return_value
        decoder.AcceptWaveform.return_value = False
        decoder.PartialResult.return_value = '{"partial":"stop"}'
        self.assertTrue(spark._motion_stop_listener(mic, rec)())


if __name__ == "__main__":
    unittest.main()
