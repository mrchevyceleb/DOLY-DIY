"""Return-to-charge decisions, cancellation and measured half turns."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if not hasattr(os, "getuid"):
    os.getuid = lambda: 0
from spark.body import Body
from spark.homing import Homing, angle_delta, approach_waypoint, dock_yaw, consistent_plane


class HomingTests(unittest.TestCase):
    def test_drive_initializes_shared_imu_with_factory_offsets(self):
        b = Body({}, hw=False)
        b._helper = Mock()
        b._helper.get_imu_offsets.return_value = (0, 10, -20, 30, -40, 50, -60)
        drive = Mock()
        drive.init.return_value = 0
        with patch.dict(sys.modules, {"doly_drive": drive}):
            b._init_drive()
            drive.init.assert_called_once_with(10, -20, 30, -40, 50, -60)
            drive.init.reset_mock()
            b._helper.get_imu_offsets.return_value = (-1, 0, 0, 0, 0, 0, 0)
            with self.assertRaises(RuntimeError):
                b._init_drive()
            drive.init.assert_not_called()

    def rig(self):
        b = Body({}, hw=False)
        b.refresh_power = Mock(return_value=False)
        b.battery_pct = Mock(return_value=10)
        b.actuators_held = Mock(return_value=False)
        b._edge_gaps = Mock(return_value=[])
        b._hazard_active = Mock(return_value=False)
        b.approach_proximity = Mock(return_value=None)
        b.drive_guarded = Mock(return_value="ok")
        b.rotate_guarded = Mock(return_value="ok")
        b.drive_stop = Mock(return_value=True)
        home = Homing(b)
        home.heading = Mock(return_value=0)
        def plane(camera, observations):
            yaw = dock_yaw(observations)
            return ("alignment", None) if yaw is None else ("ok", yaw)
        home.measure_plane = Mock(side_effect=plane)
        return b, home

    def target(self, x=0, z=220):
        return SimpleNamespace(camera_x_mm=x, camera_z_mm=z,
                               corners_px=[[0,0],[50,0],[50,50],[0,50]],
                               dock_yaw_candidates_deg=[1, -1])

    def test_centered_marker_with_oblique_or_ambiguous_pose_cannot_back_in(self):
        for yaws in ([2, -2], [2, -12], [], [float("nan")]):
            b, h = self.rig()
            target = self.target()
            target.dock_yaw_candidates_deg = yaws
            camera = Mock()
            camera.observe.return_value = target
            result, _ = h.approach(camera)
            self.assertEqual(result, "aligned" if yaws == [2, -2] else "alignment")
            b.drive_guarded.assert_not_called()

    def test_transient_rear_gap_cancels_entry_after_gpio_clears(self):
        from spark.dock_entry import DockEntry
        b, _ = self.rig()
        edge = Mock()
        edge.init.return_value = edge.enable_control.return_value = 0
        b.has["edge"] = True
        b._charging = Mock(charging=False)
        b._charging.healthy.return_value = True
        b._imu_yaw, b._imu_updated_at = 0, 100
        with patch.dict(sys.modules, {"doly_edge": edge}), \
             patch("spark.dock_entry.time.monotonic", return_value=100):
            b._init_edge()
            b._docking_entry = DockEntry(b, 220)
            b._docking_entry.travelled = 160
            # GPIO poll is already clear when the queued event arrives.
            edge.on_gap_detect.call_args.args[0]("Direction.Back_Left")
            self.assertEqual(b._docking_entry.check(), "edge")
            b.drive_stop.assert_called()

    def test_confirmed_visible_dock_required_before_translation(self):
        b, h = self.rig()
        camera = Mock()
        camera.observe.side_effect = [self.target(z=320)]*6+[None]
        self.assertEqual(h.approach(camera), ("lost", None))
        b.drive_guarded.assert_called_once()
        self.assertEqual(b.drive_guarded.call_args.args, (40,))
        b.rotate_guarded.assert_not_called()

    def test_missing_dock_only_searches_in_place_with_a_limit(self):
        b, h = self.rig()
        camera = Mock()
        camera.observe.return_value = None
        heading = [0]
        h.heading.side_effect = lambda: heading[0]
        def rotate(*args):
            heading[0] += 30
            return "ok"
        b.rotate_guarded.side_effect = rotate
        self.assertEqual(h.approach(camera), ("not_found", None))
        self.assertEqual(b.rotate_guarded.call_count, 12)
        b.drive_guarded.assert_not_called()

    def test_half_turn_measures_yaw_in_both_signs_across_wrap(self):
        for gain, clockwise in ((2.2, True), (-2.2, True), (2.2, False), (-2.2, False)):
            b, h = self.rig()
            yaw = [170]
            h.heading.side_effect = lambda: yaw[0]
            def rotate(command, interlock):
                yaw[0] = angle_delta(yaw[0]+command*gain, 0)
                return "ok"
            b.rotate_guarded.side_effect = rotate
            motor = {"remaining": 0, "direction": 1}
            def start(command, **kwargs):
                motor.update(remaining=abs(command), direction=(1 if command*gain > 0 else -1))
                return True
            def state():
                if motor["remaining"] <= 0:
                    return "completed"
                step = min(5, motor["remaining"])
                yaw[0] = angle_delta(yaw[0]+step*motor["direction"], 0)
                motor["remaining"] -= step
                return "running"
            b.drive_rotate = Mock(side_effect=start)
            b._drive = SimpleNamespace(get_state=state, DriveState=SimpleNamespace(
                Running="running", Completed="completed", Error="error"))
            with patch("spark.homing.time.sleep"):
                self.assertEqual(h.half_turn(clockwise=clockwise), "ok")
            self.assertLessEqual(abs(angle_delta(yaw[0], -10)), 3)
            self.assertLessEqual(b.drive_rotate.call_count, 2)

    def test_dock_plane_sets_waypoint_and_ambiguous_normals_stop(self):
        for sign in (-1, 1):
            x, z = approach_waypoint(0, 350, sign*30)
            self.assertAlmostEqual(x, sign*100)
            self.assertGreater(z, 170)
            self.assertLess(z, 180)
        a, b = self.target(), self.target()
        a.dock_yaw_candidates_deg = [0, 18]
        b.dock_yaw_candidates_deg = [1]
        self.assertEqual(dock_yaw([a, b]), 0)
        b.dock_yaw_candidates_deg = [1, 17]
        self.assertIsNone(dock_yaw([a, b]))
        self.assertEqual(dock_yaw([a, b], expected=0), 0)
        self.assertIsNone(dock_yaw([a, b], expected=-30))

    def test_separated_views_resolve_measured_pose_ambiguity(self):
        views = [(-4.38, [-11.66, -1.17]), (-12.46, [1.28, 6.20])]
        self.assertIsNone(consistent_plane(views))
        views.append((4.95, [-10.90]))
        self.assertAlmostEqual(consistent_plane(views), -5.92, delta=.05)
        shifted = [(angle_delta(h+179, 0), yaws) for h, yaws in views]
        self.assertAlmostEqual(angle_delta(consistent_plane(shifted), 173.08), 0, delta=.05)
        self.assertIsNone(consistent_plane([(0, [0]), (10, [20])]))
        self.assertAlmostEqual(consistent_plane([
            (173.3, [6.2]), (-175.4, [-12.1]), (161, [19.4])]), 179.5)
        self.assertIsNone(consistent_plane([(0, [0]), (10, [40]), (-10, [10])]))

    def test_final_entry_needs_fresh_visual_agreement_with_measured_heading(self):
        b, h = self.rig()
        h.measure_plane.side_effect = None
        h.measure_plane.return_value = ("ok", 0)
        target = self.target(z=190)
        target.dock_yaw_candidates_deg = [7]
        camera = Mock()
        camera.observe.return_value = target
        self.assertEqual(h.approach(camera), ("alignment", None))
        self.assertEqual(h.measure_plane.call_count, 3)
        b.drive_guarded.assert_not_called()

    def test_waypoint_step_is_bounded_and_stops_before_drive_on_turn_failure(self):
        b, h = self.rig()
        h.turn_to = Mock(return_value="ok")
        self.assertEqual(h.waypoint_step(100, 30), "ok")
        self.assertEqual(b.drive_guarded.call_args.args, (30,))
        self.assertEqual(h.turn_to.call_count, 2)
        b.drive_guarded.reset_mock()
        h.turn_to.return_value = "edge"
        self.assertEqual(h.waypoint_step(100, 30), "edge")
        b.drive_guarded.assert_not_called()
        self.assertEqual(h.waypoint_step(10, -40), "alignment")

    def test_camera_can_wait_out_sensor_error_but_motion_cannot(self):
        b, h = self.rig()
        b.approach_proximity.return_value = "sensor"
        self.assertFalse(h.camera_interrupted())
        self.assertEqual(h.interlock(), "sensor")
        b._edge_gaps.return_value = ["Back_Left"]
        self.assertTrue(h.camera_interrupted())

    def test_near_standoff_has_no_dead_zone_and_retreats_are_bounded(self):
        b, h = self.rig()
        camera = Mock()
        camera.observe.return_value = self.target(z=179.5)
        self.assertEqual(h.approach(camera)[0], "aligned")
        b.drive_guarded.assert_not_called()
        camera.observe.return_value = self.target(z=164)
        self.assertEqual(h.approach(camera), ("too_close", None))
        self.assertEqual(b.drive_guarded.call_count, 2)
        for call in b.drive_guarded.call_args_list:
            self.assertEqual(call.args, (-40,))
            self.assertIsNotNone(call.kwargs["interlock"])
        b.drive_guarded.reset_mock()
        b.drive_guarded.return_value = "edge"
        self.assertEqual(h.approach(camera), ("edge", None))
        b.drive_guarded.assert_called_once()

    def test_edge_stop_does_not_authorize_more_reverse_steps(self):
        b, h = self.rig()
        h.charge_verified = Mock(return_value=False)
        with patch("spark.dock_entry.DockEntry") as entry:
            entry.return_value.run.return_value = "edge"
            self.assertEqual(h.reverse_entry(self.target()), "edge")
            entry.assert_called_once_with(b, 240, None, allow_front_after=160)
            entry.return_value.run.assert_called_once()
        self.assertIsNone(b._docking_entry)

    def test_controller_completion_is_not_arrival(self):
        b, h = self.rig()
        h.charge_verified = Mock(return_value=False)
        with patch("spark.dock_entry.DockEntry") as entry:
            entry.return_value.run.return_value = "no_contact"
            self.assertEqual(h.reverse_entry(self.target()), "no_contact")
            entry.return_value.run.return_value = "contact"
            self.assertEqual(h.reverse_entry(self.target()), "contact")
            h.charge_verified.return_value = True
            self.assertEqual(h.reverse_entry(self.target()), "arrived")

    def test_return_allowed_at_ten_but_stop_edge_and_empty_battery_win(self):
        b, h = self.rig()
        self.assertIsNone(h.interlock())
        b._approach_stop.set()
        self.assertEqual(h.interlock(), "cancelled")
        b._approach_stop.clear()
        b._edge_gaps.return_value = ["Back_Left"]
        h.reversing = True
        self.assertEqual(h.interlock(), "edge")
        b._edge_gaps.return_value = []
        b.battery_pct.return_value = 2
        self.assertEqual(h.interlock(), "power")

    def test_hardware_proximity_reads_cache_and_stops_when_stale(self):
        b = Body({}, hw=False)
        b.hw = True
        b.has["tof"] = True
        b._tof = Mock()
        b._tof_snapshot = (100, [(0, -1, 13, 1000), (1, -1, 6, 1001)])
        with patch("spark.body.time.monotonic", return_value=100.1):
            self.assertIsNone(b.approach_proximity())
        with patch("spark.body.time.monotonic", return_value=100.4):
            self.assertEqual(b.approach_proximity(), "sensor")
        b._tof.get_sensors_data.assert_not_called()

    def test_range_ignore_statuses_with_no_distance_allow_motion(self):
        b = Body({}, hw=False)
        b.hw = True
        b.has["tof"] = True
        b._tof = Mock()
        # glossy-table crosstalk/sigma rejects: no distance, not a contact
        for status in (8, 9, 10, 11):
            b._tof_snapshot = (100, [(0, -1, status, 1000), (1, -1, status, 1001)])
            with patch("spark.body.time.monotonic", return_value=100.1):
                self.assertIsNone(b.approach_proximity())
        # hardware faults, underflow and a real distance still stop
        for bad in ((0, -1, 4, 1002), (0, -1, 12, 1003), (0, -1, 14, 1004), (0, 45, 0, 1005)):
            b._tof_snapshot = (100, [bad, (1, -1, 6, 1001)])
            with patch("spark.body.time.monotonic", return_value=100.1):
                self.assertEqual(b.approach_proximity(), "sensor" if bad[2] else "obstacle")

    def test_entry_never_waives_leading_gaps_or_early_ramp_gaps(self):
        from spark.dock_entry import DockEntry
        for gap, travelled, expected in (("Front_Left", 0, "edge"),
                                          ("Front_Left", 160, None),
                                          ("Back_Left", 160, "edge"),
                                          ("All", 160, "edge")):
            b, _ = self.rig()
            b.has["edge"] = True
            b._charging = Mock(charging=False)
            b._charging.healthy.return_value = True
            b._imu_yaw, b._imu_updated_at = 0, 100
            b._edge_gaps.return_value = [gap]
            with patch("spark.dock_entry.time.monotonic", return_value=100):
                entry = DockEntry(b, 220)
                entry.travelled = travelled
                self.assertEqual(entry.check(), expected)

    def test_ten_percent_triggers_return_and_eleven_allows_roaming(self):
        from spark.__main__ import Spark
        spark = Spark.__new__(Spark)
        spark.body = Mock()
        spark.body.is_on_dock.return_value = False
        spark.body.actuators_held.return_value = False
        spark.body._return_margin_pct.return_value = 0
        spark.body.go_home.return_value = "arrived"
        for pct in (11, 10):
            spark.body.battery_pct.return_value = pct
            spark._low_battery_check({"low_battery_pct": 10})
        spark.body.go_home.assert_called_once()

        b, _ = self.rig()
        b.cfg = {"homing": {"enabled": True}}
        b._motion_allowed = Mock(return_value=True)
        with patch("spark.roaming.Roaming") as roaming:
            roaming.return_value.run.return_value = "ok"
            b.battery_pct.return_value = 11
            self.assertTrue(b.wander_step())
            b.battery_pct.return_value = 10
            self.assertFalse(b.wander_step())
            roaming.return_value.run.assert_called_once()

    def test_far_from_home_raises_the_return_threshold(self):
        from spark.__main__ import Spark
        spark = Spark.__new__(Spark)
        spark.body = Mock()
        spark.body.is_on_dock.return_value = False
        spark.body.actuators_held.return_value = False
        spark.body._return_margin_pct.return_value = 6
        spark.body.go_home.return_value = "arrived"
        spark.body.battery_pct.return_value = 13  # above 10, below 10+6
        spark._low_battery_check({"low_battery_pct": 10})
        spark.body.go_home.assert_called_once()

    def test_return_margin_scales_with_anchored_distance_and_caps(self):
        b, _ = self.rig()
        b.cfg = {"idle": {"roam_reserve_pct_per_m": 2}}
        b._roam_distance_bound = None
        self.assertEqual(b._return_margin_pct(), 0)
        b._roam_distance_bound = 700
        self.assertEqual(b._return_margin_pct(), 2)
        b._roam_distance_bound = 2600
        self.assertEqual(b._return_margin_pct(), 6)
        b._roam_distance_bound = 9900
        self.assertEqual(b._return_margin_pct(), 10)

    def test_go_home_retries_recoverable_results_until_arrival(self):
        b, _ = self.rig()
        b.hw = Mock()
        b.cfg = {"homing": {"enabled": True, "attempts": 3}}
        b.battery_pct.return_value = 15
        with patch("spark.homing.Homing") as homing, patch("time.sleep"):
            homing.return_value.run.side_effect = ["limit", "alignment", "arrived"]
            self.assertEqual(b.go_home(), "arrived")
        self.assertEqual(homing.return_value.run.call_count, 3)

    def test_go_home_returns_last_result_when_attempts_exhaust(self):
        b, _ = self.rig()
        b.hw = Mock()
        b.cfg = {"homing": {"enabled": True, "attempts": 3}}
        b.battery_pct.return_value = 15
        with patch("spark.homing.Homing") as homing, patch("time.sleep"):
            homing.return_value.run.return_value = "lost"
            self.assertEqual(b.go_home(), "lost")
        self.assertEqual(homing.return_value.run.call_count, 3)

    def test_go_home_does_not_retry_deliberate_stops_or_dead_battery(self):
        b, _ = self.rig()
        b.hw = Mock()
        b.cfg = {"homing": {"enabled": True, "attempts": 3}}
        with patch("spark.homing.Homing") as homing, patch("time.sleep"):
            homing.return_value.run.return_value = "sensor"
            self.assertEqual(b.go_home(), "sensor")
            homing.return_value.run.reset_mock()
            b.battery_pct.return_value = 3
            self.assertEqual(b.go_home(), "power")  # dead pack never starts
        self.assertEqual(homing.return_value.run.call_count, 0)  # dead pack never starts

    def test_reseat_probe_seats_when_charge_appears(self):
        b, _ = self.rig()
        b.hw = Mock()
        b.has = {"drive": True, "edge": True}
        b.docked = False
        b._edge_gaps = Mock(return_value=["Front_Left", "Front_Right"])
        b._charging = Mock()
        b._charging.healthy.return_value = True
        b.speak = Mock()
        drive = Mock()
        drive.DriveState = Mock(Running=1, Completed=2, Error=3)
        drive.get_state.side_effect = [1, 2]  # Running then Completed
        drive.go_distance.return_value = 0
        b._drive = drive
        # drive preflight sees no contact; the settle window after the move finds charge
        b.refresh_power = Mock(side_effect=[False, False, False, True, True])
        self.assertTrue(b.reseat_probe())
        self.assertEqual(drive.go_distance.call_args.args[3], False)  # reverse
        b.speak.assert_not_called()

    def test_reseat_probe_ignores_other_gap_profiles(self):
        for profile in (["Front_Left"], ["Front_Left", "Back_Right"]):
            b, _ = self.rig()
            b.hw = Mock()
            b.has = {"drive": True, "edge": True}
            b.docked = False
            b._edge_gaps = Mock(return_value=profile)
            b._drive = Mock()
            self.assertFalse(b.reseat_probe())
            b._drive.go_distance.assert_not_called()

    def test_roaming_does_not_translate_without_reacquiring_home(self):
        from spark.roaming import Roaming
        b, _ = self.rig()
        roam = Roaming(b)
        roam.locate = Mock(return_value=("not_found", None))
        with patch("spark.dock_camera.DockCamera"):
            self.assertEqual(roam.run(), "not_found")
        b.drive_guarded.assert_not_called()

    def test_roaming_boundary_turns_back_toward_visible_home(self):
        from spark.roaming import Roaming
        b, _ = self.rig()
        b._roam_distance_bound = 2950  # near the 3000mm roam boundary
        roam = Roaming(b)
        roam.locate = Mock(return_value=("found", self.target(z=2900)))
        with patch("spark.dock_camera.DockCamera"):
            self.assertEqual(roam.run(), "ok")
        b.drive_guarded.assert_called_once()
        self.assertEqual(b.drive_guarded.call_args.args, (40,))


if __name__ == "__main__":
    unittest.main()
