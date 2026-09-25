"""Regression checks for the observed charging/movement and ASR failures."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if not hasattr(os, "getuid"):
    os.getuid = lambda: 0  # Body's temp audio paths on Windows test hosts

from spark.body import Body
from spark.anim import AnimPlayer
from spark.asr import WhisperASR
from spark.charging import ChargingMonitor


class ChargingSafetyTests(unittest.TestCase):
    def test_gap_poll_uses_stock_low_polarity_and_fails_closed(self):
        b = Body({"edge": {"gap_gpio_state": "High"}}, hw=False)
        b.has = {"edge": True}
        b._edge = Mock()
        b._edge.GpioState.Low = 0
        b._edge.get_sensors.return_value = []
        self.assertEqual(b._edge_gaps(), [])
        b._edge.get_sensors.assert_called_once_with(0)
        b._edge.get_sensors.return_value = [Mock(id="SensorId.Front_Left")]
        self.assertEqual(b._edge_gaps(), ["Front_Left"])
        b._edge.get_sensors.side_effect = RuntimeError("disconnected")
        self.assertEqual(len(b._edge_gaps()), 4)

    def body(self, charging=True, pct=0, gaps=None):
        b = Body({"state_dir": "/unused", "idle": {}}, hw=False)
        b.hw = True
        b.has = {"drive": True, "arm": True, "edge": True, "battery": True}
        b._drive = Mock()
        b._arm = Mock()
        b._charging = Mock(charging=charging, average=30 if charging else -30,
                           voltage=3.5, error=None)
        b._charging.sample.return_value = charging
        b.battery_pct = lambda: pct
        b._edge_gaps = lambda: gaps if gaps is not None else ["Back_Left", "Back_Right"]
        return b

    def test_charging_probe_and_all_actuator_paths_stay_still(self):
        b = self.body(pct=50)
        b.dock_probe()
        self.assertTrue(b.docked)
        self.assertFalse(b.drive_rotate(12))
        self.assertFalse(b.drive_distance(40))
        self.assertFalse(b.arm_angle(130))
        AnimPlayer(b, {})._arm_to(0, 130, 25, False)
        b._drive.go_distance.assert_not_called()
        b._drive.go_rotate.assert_not_called()
        b._arm.set_angle.assert_not_called()

    def test_zero_battery_cannot_undock_even_on_voice_request(self):
        b = self.body()
        b.dock_probe()
        self.assertFalse(b.drive_guarded(250))
        self.assertFalse(b._undock())
        b._drive.go_distance.assert_not_called()

    def test_airborne_gaps_never_verify_a_dock(self):
        b = self.body(charging=False, pct=50,
                      gaps=["Front_Left", "Front_Right", "Back_Left", "Back_Right"])
        self.assertFalse(b.dock_probe())
        self.assertFalse(b._undock())
        self.assertFalse(b.drive_rotate(12))
        b._drive.go_distance.assert_not_called()
        b._drive.go_rotate.assert_not_called()

    def test_lost_contact_keeps_dock_hold_until_supported_discharge(self):
        gaps = ["Back_Left", "Back_Right"]
        b = self.body(pct=50, gaps=gaps)
        b.refresh_power()
        b._charging.sample.return_value = False
        b._charging.average = -40
        with patch("spark.body.time.monotonic", return_value=100):
            b.refresh_power()
        self.assertTrue(b.docked)
        gaps.clear()
        with patch("spark.body.time.monotonic", return_value=101):
            b.refresh_power()
        self.assertTrue(b.docked)  # one quiet sample alone cannot unlock
        with patch("spark.body.time.monotonic", return_value=107):
            b.refresh_power()
        # Slid off the contacts OR gently picked up onto open ground: the
        # dock face is gone from under the sensors and discharge is real.
        # She must KNOW she is off the dock so roaming/homing can act.
        self.assertFalse(b.docked)
        self.assertIsNone(b._pose)

    def test_dock_monitor_cancels_a_native_operation_that_starts_while_held(self):
        b = self.body(pct=80)
        b.refresh_power()
        b._drive.reset_mock()
        b._arm.reset_mock()
        b._drive.get_state.return_value = b._drive.DriveState.Running
        b._arm.get_state.return_value = b._arm.ArmState.Running
        b.refresh_power()
        b._drive.abort.assert_called_once()
        self.assertEqual([c.args for c in b._drive.free_drive.call_args_list],
                         [(0, False, True), (0, True, True)])
        self.assertEqual(b._arm.abort.call_count, 2)
        b._drive.go_distance.assert_not_called()
        b._arm.set_angle.assert_not_called()

    def test_native_abort_failure_still_attempts_to_zero_both_wheels(self):
        b = self.body(pct=80)
        b._drive.abort.side_effect = RuntimeError("controller unavailable")
        b._drive.free_drive.side_effect = [False, True]
        self.assertFalse(b.drive_stop())
        self.assertEqual(b._drive.free_drive.call_count, 2)

    def test_controlled_departure_requires_clear_ground_and_stationary_discharge(self):
        for failure in (None, "edge", "power", "stop", "motor_load_only", "stalled"):
            with self.subTest(failure=failure):
                gaps = ["Front_Left", "Front_Right"]
                b = self.body(pct=80, gaps=gaps)
                b._charging.healthy.return_value = True
                b._drive.DriveState.Running = "running"
                b._drive.DriveState.Completed = "completed"
                b._drive.DriveState.Error = "error"
                b._drive.get_rpm.return_value = 0.0  # installed SDK returns zero while moving
                clock = [1000.0]
                native = {"active": False, "polls": 0, "stopped": None}

                def dispatch(*args):
                    native.update(active=True, polls=0)
                    # Ordinary commands cannot borrow the departure permit.
                    self.assertFalse(b.drive_rotate(30))
                    self.assertFalse(b.arm_angle(90))
                    return True

                def state():
                    if not native["active"]:
                        return "completed"
                    native["polls"] += 1
                    if native["polls"] == 1:
                        if failure == "edge":
                            gaps.append("Back_Left")
                        elif failure == "power":
                            b._charging.healthy.return_value = False
                        elif failure == "stop":
                            b._approach_stop.set()
                        b._charging.sample.return_value = False
                        b._charging.charging = False
                        return "running"
                    if failure not in ("stalled", "edge"):
                        gaps.clear()
                    # failure == "edge" keeps its Back gap: a real edge is
                    # persistent, unlike one-poll sensor flutter
                    native["active"] = False
                    native["stopped"] = clock[0]
                    if failure == "motor_load_only":
                        b._charging.sample.return_value = True
                        b._charging.charging = True
                    return "completed"

                b._drive.go_distance.side_effect = dispatch
                b._drive.get_state.side_effect = state
                with patch("spark.body.time.monotonic", side_effect=lambda: clock[0]), \
                     patch("spark.body.time.sleep", side_effect=lambda s: clock.__setitem__(0, clock[0]+s)):
                    self.assertEqual(b._undock(), failure is None)
                self.assertEqual(b.docked, failure is not None)
                self.assertFalse(b._leaving_home)
                self.assertIsNone(b._departure)
                # A deep seat keeps the dock-face pair through the whole
                # bounded clearance, so "stalled" (gaps never clear) now
                # also spends all five steps - and still fails closed.
                full_clearance = failure in (None, "stalled")
                self.assertEqual(b._drive.go_distance.call_count, 5 if full_clearance else 1)
                self.assertEqual(b._drive.go_distance.call_args_list[0].args[1:],
                                 (20, 25, True, True))  # stock dock exits forward
                b._drive.go_rotate.assert_not_called()
                b._arm.set_angle.assert_not_called()
                if failure is None:
                    self.assertGreaterEqual(clock[0] - native["stopped"], 1.25)
                if failure == "stalled":
                    self.assertEqual(b.last_departure_result, "edge")
                    self.assertEqual(b._edge_hazard, "forward")
                    self.assertFalse(b._undock())  # no repeat into the same gap
                    self.assertEqual(b._drive.go_distance.call_count, 5)
                b._drive.abort.assert_called()

    def test_failed_voice_exit_below_full_does_not_cancel_later_auto_roam(self):
        b = self.body(pct=93, gaps=["Front_Left", "Front_Right"])
        b.cfg["idle"]["roam_full_pct"] = 96
        b.cfg["homing"] = {"enabled": True}
        b._charging.healthy.return_value = True
        b.refresh_power()
        b._charging.sample.return_value = False  # contact drops during the command
        self.assertFalse(b._undock())
        self.assertEqual(b.last_departure_result, "edge")
        self.assertFalse(b._dock_auto_attempted)
        b.battery_pct = lambda: 96
        b._charging.sample.return_value = True
        with patch("spark.body.time.monotonic", return_value=100):
            b.refresh_power()
        with patch("spark.body.time.monotonic", return_value=161):
            b.refresh_power()
            self.assertTrue(b.dock_roam_ready())

    def test_full_charge_roaming_is_stable_once_per_visit_and_never_contact_loss(self):
        b = self.body(pct=100)
        b.cfg["homing"] = {"enabled": True}
        with patch("spark.body.time.monotonic", return_value=100):
            b.refresh_power()
            self.assertFalse(b.dock_roam_ready())
        # Fully charged current can taper to zero without a sensor fault.
        b._charging.sample.return_value = None
        b._charging.healthy.return_value = True
        with patch("spark.body.time.monotonic", return_value=130):
            b.refresh_power()
            self.assertFalse(b.dock_roam_ready())
        with patch("spark.body.time.monotonic", return_value=161):
            b.refresh_power()
            self.assertTrue(b.dock_roam_ready())
            b.cfg["homing"]["enabled"] = False
            self.assertFalse(b.dock_roam_ready())
            b.cfg["homing"]["enabled"] = True
            with patch.object(b, "_undock", return_value=False) as undock:
                b.wander_step()
                b.wander_step()
                undock.assert_called_once()
        b._dock_auto_attempted = False
        b._charging.sample.return_value = False
        b.refresh_power()
        self.assertFalse(b.dock_roam_ready())
        b._charging.sample.return_value = None
        b._charging.healthy.return_value = False
        b.refresh_power()
        self.assertFalse(b.dock_roam_ready())
        b._charging.sample.return_value = True
        b.sleeping = True
        b.refresh_power()
        self.assertFalse(b.dock_roam_ready())

    def test_departure_cannot_reuse_a_gap_allowance_after_ground_returns(self):
        from spark.departure import Departure
        for forward, trailing in ((True, "Back_Left"), (False, "Front_Left")):
            gaps = [trailing]
            b = self.body(pct=80, gaps=gaps)
            b._leaving_home = True
            b._charging.healthy.return_value = True
            d = Departure(b, forward, gaps, lambda: False, transient_ms=0)
            self.assertIsNone(d.check())
            gaps.clear()
            self.assertIsNone(d.check())
            gaps.append(trailing)
            self.assertEqual(d.check(), "edge")

    def test_deep_seat_tolerates_dock_face_until_ground_returns(self):
        # 2026-09-25: a deeper seat kept the front pair reading the dock
        # base after the 20mm probe - the exit must still complete.
        from spark.departure import Departure
        gaps = ["Front_Left", "Front_Right"]
        b = self.body(pct=80, gaps=gaps)
        b._leaving_home = True
        b._charging.healthy.return_value = True
        d = Departure(b, True, list(gaps), lambda: False, transient_ms=0)
        d.front_probe = False                    # probe step already completed
        self.assertIsNone(d.check())             # post-probe verify tolerates the pair
        d.clearing = True
        self.assertIsNone(d.check())             # bounded clearance tolerates the pair
        gaps.clear()                             # open ground reached under the nose
        self.assertIsNone(d.check())             # allowance shed...
        self.assertFalse({"Front_Left", "Front_Right"} & d.allowed_gaps)
        gaps.append("Front_Left")                # ...so a real edge ahead stops it
        self.assertEqual(d.check(), "edge")

    def test_clearance_can_leave_rear_lip_but_front_and_airborne_events_still_stop(self):
        from spark.departure import Departure
        gaps = []
        b = self.body(pct=80, gaps=gaps)
        b._leaving_home = True
        b._charging.healthy.return_value = True
        d = Departure(b, True, gaps, lambda: False, transient_ms=0)
        d.clearing = True
        gaps.extend(["Back_Left", "Back_Right"])
        self.assertTrue(d.trailing_gap("Back"))
        self.assertFalse(d.trailing_gap("All"))
        gaps.append("Front_Left")
        self.assertFalse(d.trailing_gap("Back"))
        self.assertEqual(d.reason, "edge")

    def test_departure_debounces_sensor_flutter(self):
        # 2026-09-25: a one-poll Back reading as weight shifted aborted
        # the authorized exit. Flutter is judged, not obeyed.
        from spark.departure import Departure
        gaps = ["Front_Left", "Front_Right"]
        b = self.body(pct=80, gaps=gaps)
        b._leaving_home = True
        b._charging.healthy.return_value = True
        d = Departure(b, True, list(gaps), lambda: False, transient_ms=10_000)
        gaps.append("Back_Left")                 # single dirty poll
        self.assertIsNone(d.check())             # not yet a verdict
        self.assertEqual(d.allowed_gaps, {"Front_Left", "Front_Right"})  # allowance intact
        gaps.remove("Back_Left")                 # flutter cleared
        self.assertIsNone(d.check())
        gaps.append("Back_Left")                 # persistent this time
        d.transient_ms = 0
        self.assertEqual(d.check(), "edge")

    def test_only_affirmative_movement_commands_can_leave_dock(self):
        from spark.router import Router
        b = Mock(docked=True, last_departure_result="edge")
        b.battery_pct.return_value = 80
        b._undock.return_value = False
        router = Router({}, b, None, None)
        router.handle("what is your battery")
        router.handle("don't move forward")
        b._undock.assert_not_called()
        router.handle("move forward")
        b._undock.assert_called_once()
        b.drive_guarded.assert_not_called()
        b._undock.side_effect = lambda: setattr(b, "docked", False) or True
        router.handle("move forward")
        b.drive_guarded.assert_called_once_with(150, speed=30)

    def test_front_dock_profile_requires_recent_charging_and_stale_hold_can_clear_at_rest(self):
        b = self.body(charging=False, pct=80, gaps=["Front_Left", "Front_Right"])
        b.docked = True
        b._charging.healthy.return_value = True
        self.assertFalse(b._undock())
        b._drive.go_distance.assert_not_called()
        b._edge_gaps = lambda: []
        clock = [1000.0]
        with patch("spark.body.time.monotonic", side_effect=lambda: clock[0]), \
             patch("spark.body.time.sleep", side_effect=lambda s: clock.__setitem__(0, clock[0]+s)):
            self.assertTrue(b._undock())
        self.assertFalse(b.docked)
        b._drive.go_distance.assert_not_called()

    def test_sensor_outage_holds_even_at_high_battery(self):
        b = self.body(charging=None, pct=90, gaps=[])
        self.assertFalse(b.drive_guarded(60))
        self.assertFalse(b.arm_angle(130))
        b._drive.go_distance.assert_not_called()

    def test_monitor_exception_fails_closed_and_recovers(self):
        b = self.body(charging=False, pct=90, gaps=[])
        b._charging.sample.side_effect = RuntimeError("sensor thread fault")
        self.assertTrue(b.actuators_held())
        self.assertTrue(b._power_fault)
        b._charging.sample.side_effect = None
        self.assertFalse(b.actuators_held())

    def test_current_warmup_stale_reads_and_error_are_not_dock_proof(self):
        m = ChargingMonitor()
        m._read = Mock(return_value=(30, 3.5))
        for now in (10, 10.3, 10.6, 10.9):
            with patch("spark.charging.time.monotonic", return_value=now):
                self.assertIsNone(m.sample())
                self.assertFalse(m.healthy())
        with patch("spark.charging.time.monotonic", return_value=11.2):
            self.assertTrue(m.sample())
            self.assertTrue(m.healthy())
        with patch("spark.charging.time.monotonic", return_value=15):
            self.assertIsNone(m.sample())
            self.assertFalse(m.healthy())
        m._read.side_effect = OSError("bus disconnected")
        with patch("spark.charging.time.monotonic", return_value=16):
            self.assertIsNone(m.sample())
        self.assertIsNone(m.average)

    def test_positive_average_with_zero_sample_does_not_allow_movement(self):
        m = ChargingMonitor()
        m._read = Mock(side_effect=[(30, 3.5)] * 4 + [(0, 3.5)])
        for i in range(5):
            with patch("spark.charging.time.monotonic", return_value=10 + i * .3):
                result = m.sample()
        self.assertEqual(m.average, 24)
        self.assertIsNone(result)
        m.close()

    def test_gap_arriving_after_preflight_prevents_dispatch(self):
        b = self.body(charging=False, pct=80, gaps=[])
        original = b._motion_allowed

        def preflight_then_edge(direction):
            allowed = original(direction)
            if allowed:
                # Simulate the edge event immediately after a clear poll.
                b._gap_lock_until = float("inf")
                b._latch_hazard("forward")
                b.drive_stop()
            return allowed

        b._motion_allowed = preflight_then_edge
        self.assertFalse(b.drive_distance(60))
        b._drive.go_distance.assert_not_called()

    def test_new_confirmed_dock_clears_placement_hazard_but_failed_exit_stays_latched(self):
        for prior in ("forward", "all"):
            b = self.body(pct=80, gaps=["Front_Left", "Front_Right"])
            b._latch_hazard(prior)  # placement events precede current averaging
            b.refresh_power()
            self.assertTrue(b.docked)
            self.assertIsNone(b._edge_hazard)
            b._drive.go_distance.assert_not_called()
            b._latch_hazard("forward")  # a failed probe from the held dock
            b.refresh_power()
            self.assertEqual(b._edge_hazard, "forward")
            self.assertFalse(b._undock())
            b._drive.go_distance.assert_not_called()

    def test_turn_cannot_sweep_a_rear_wheel_over_an_edge(self):
        b = self.body(charging=False, pct=80, gaps=["Back_Left"])
        self.assertFalse(b.drive_rotate(90))
        b._drive.go_rotate.assert_not_called()

    def test_discharge_while_parked_alerts_once_without_releasing_motors(self):
        b = self.body(pct=50)
        b.refresh_power()
        b._charging.sample.return_value = False
        b._charging.average = -40
        with patch("spark.body.time.monotonic", return_value=100):
            b.refresh_power()
        self.assertFalse(b.take_charge_notice())
        with patch("spark.body.time.monotonic", return_value=116):
            b.refresh_power()
            self.assertTrue(b.take_charge_notice())
            b.refresh_power()
            self.assertFalse(b.take_charge_notice())
        self.assertTrue(b.docked)
        self.assertTrue(b.actuators_held())
        b._drive.go_distance.assert_not_called()
        b._charging.sample.return_value = True
        b.refresh_power()
        self.assertFalse(b._charge_notice_sent)

    def test_sleep_holds_all_actuators_without_a_stock_animation(self):
        b = self.body(pct=50)
        b.anim = Mock()
        b.sleep_pose()
        self.assertTrue(b.sleeping)
        self.assertFalse(b.drive_rotate(90))
        self.assertFalse(b.arm_angle(20))
        b.anim.play.assert_not_called()
        b._arm.set_angle.assert_not_called()

    def test_voice_turn_waits_for_completion_and_honors_stop_and_errors(self):
        for states, stop, expected in ((["running", "completed"], False, "ok"),
                                       (["error"], False, "error"),
                                       (["running"], True, "cancelled")):
            b = Body({}, hw=False)
            b.has = {"drive": True}
            b._drive = Mock()
            b._drive.DriveState.Running = "running"
            b._drive.DriveState.Completed = "completed"
            b._drive.DriveState.Error = "error"
            b._drive.get_state.side_effect = states
            b.drive_rotate = Mock(return_value=True)
            b.drive_stop = Mock()
            b._motion_allowed = Mock(return_value=True)
            b.motion_stop_factory = lambda: lambda: stop
            self.assertEqual(b.turn_guarded(360), expected)
            b.drive_stop.assert_called_once()
            self.assertFalse(b._turning)
        b = Body({}, hw=False)
        b.has = {"drive": True}
        b._drive = Mock()
        b._drive.go_rotate.return_value = False
        b._motion_allowed = Mock(return_value=True)
        self.assertFalse(b.drive_rotate(90))

    def test_empty_remote_asr_does_not_launch_local_whisper(self):
        a = WhisperASR({"asr": {"server_url": "http://test"}})
        with patch.object(a, "_transcribe_http", return_value=""), \
                patch("subprocess.run") as run:
            self.assertEqual(a.transcribe_pcm(b"\\0" * 16000), "")
            self.assertEqual(a.last_source, "server")
            run.assert_not_called()

    def test_http_failures_raise_instead_of_becoming_transcripts(self):
        a = WhisperASR({"asr": {"server_url": "http://test"}})
        import subprocess
        with patch("tempfile.mkstemp", return_value=(123, "/tmp/asr-test.wav")), \
                patch("os.close"), patch("os.unlink"), patch("wave.open"), \
                patch("subprocess.run", side_effect=subprocess.CalledProcessError(22, "curl")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                a._transcribe_http(b"\\0" * 16000)
            self.assertTrue(run.call_args.kwargs["check"])
            self.assertIn("--fail", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
