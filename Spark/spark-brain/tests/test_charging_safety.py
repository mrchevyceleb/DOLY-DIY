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
        self.assertTrue(b.docked)
        with patch("spark.body.time.monotonic", return_value=107):
            b.refresh_power()
        self.assertFalse(b.docked)
        self.assertIsNone(b._pose)

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
        with patch("spark.charging.time.monotonic", return_value=11.2):
            self.assertTrue(m.sample())
        with patch("spark.charging.time.monotonic", return_value=15):
            self.assertIsNone(m.sample())
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

    def test_empty_remote_asr_does_not_launch_local_whisper(self):
        a = WhisperASR({"asr": {"server_url": "http://test"}})
        with patch.object(a, "_transcribe_http", return_value=""), \
                patch("subprocess.run") as run:
            self.assertEqual(a.transcribe_pcm(b"\\0" * 16000), "")
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
