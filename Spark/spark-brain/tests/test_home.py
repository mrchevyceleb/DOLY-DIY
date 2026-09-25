"""Stock-style home memory: blind navigation to remembered dock coordinates."""
import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if not hasattr(os, "getuid"):
    os.getuid = lambda: 0
from spark.body import Body
from spark.home import NEAR_MM, STANDOFF_MM, TRAVEL_CAP_MM, navigate_home


class HomeNavTests(unittest.TestCase):
    def rig(self, pose):
        """Real pose tracking; guarded primitives emulate real travel."""
        b = Body({}, hw=False)
        b._pose = pose

        def drive(mm, speed=25, segment_mm=50, interlock=None):
            b._pose_update(dist_mm=mm)
            return "ok"

        def rotate(deg, interlock=None):
            b._pose_update(rot_deg=deg)
            return "ok"

        b.drive_guarded = Mock(side_effect=drive)
        b.rotate_guarded = Mock(side_effect=rotate)
        return b

    def travelled(self, b):
        return sum(c.args[0] for c in b.drive_guarded.call_args_list)

    def test_unknown_pose_skips_blind_navigation(self):
        b = self.rig(None)
        self.assertEqual(navigate_home(b, Mock(return_value=None)), "skip")
        b.drive_guarded.assert_not_called()
        b.rotate_guarded.assert_not_called()

    def test_drives_to_standoff_and_faces_the_dock(self):
        b = self.rig([0.0, 0.0, 0.0])
        self.assertEqual(navigate_home(b, Mock(return_value=None)), "ok")
        x, y, heading = b._pose
        # stopped inside the camera's near range, marker dead ahead
        self.assertLessEqual(math.hypot(STANDOFF_MM - x, y), NEAR_MM + 1)
        self.assertAlmostEqual((heading + 180) % 360 - 180, -180.0, delta=2)
        self.assertGreaterEqual(self.travelled(b), STANDOFF_MM - NEAR_MM - 1)

    def test_sideways_pose_turns_first_then_drives(self):
        b = self.rig([900.0, 600.0, 90.0])
        self.assertEqual(navigate_home(b, Mock(return_value=None)), "ok")
        x, y, _ = b._pose
        self.assertLessEqual(math.hypot(STANDOFF_MM - x, y), NEAR_MM + 1)

    def test_travel_cap_returns_limit_partway(self):
        b = self.rig([5000.0, 0.0, 180.0])
        self.assertEqual(navigate_home(b, Mock(return_value=None)), "limit")
        self.assertLessEqual(self.travelled(b), TRAVEL_CAP_MM)
        self.assertAlmostEqual(b._pose[0], 5000.0 - TRAVEL_CAP_MM, delta=1)

    def test_near_standoff_skips_driving_but_still_faces_dock(self):
        b = self.rig([430.0, 10.0, 90.0])
        self.assertEqual(navigate_home(b, Mock(return_value=None)), "ok")
        b.drive_guarded.assert_not_called()
        target = math.degrees(math.atan2(-10.0, -430.0))
        self.assertAlmostEqual(b._pose[2], (target + 180) % 360 - 180, delta=2)

    def test_edge_stop_aborts_navigation_without_more_travel(self):
        b = self.rig([0.0, 0.0, 0.0])
        b.drive_guarded.side_effect = None
        b.drive_guarded.return_value = "stopped_edge"
        self.assertEqual(navigate_home(b, Mock(return_value=None)), "stopped_edge")
        self.assertEqual(b.drive_guarded.call_count, 1)

    def test_turn_refusal_aborts_before_any_blind_translation(self):
        b = self.rig([900.0, 600.0, 90.0])
        b.rotate_guarded.side_effect = None
        b.rotate_guarded.return_value = "blocked"
        self.assertEqual(navigate_home(b, Mock(return_value=None)), "blocked")
        b.drive_guarded.assert_not_called()

    def test_pose_lost_mid_leg_aborts_as_lost(self):
        b = self.rig([0.0, 0.0, 0.0])

        def pickup(mm, speed=25, segment_mm=50, interlock=None):
            b._pose = None  # airborne event cleared the frame mid-leg
            return "ok"

        b.drive_guarded.side_effect = pickup
        self.assertEqual(navigate_home(b, Mock(return_value=None)), "lost")


class DockFaceRecoveryTests(unittest.TestCase):
    """Failed reseat escalation: retreat only on pose + charging proof."""
    FRONT_PAIR = ["Front_Left", "Front_Right"]

    def rig(self, pose, gaps_holder, state_script=None, charging_at="now"):
        """gaps_holder: one-key dict with the LIVE gap list (flip mid-test).
        state_script: get_state return values, default completes one step."""
        import time as _time
        b = Body({}, hw=False)
        b._pose = pose
        b.has = {"drive": True, "edge": True}
        b.docked = False
        b._edge_gaps = Mock(side_effect=lambda: list(gaps_holder["cur"]))
        b.refresh_power = Mock(return_value=False)
        b._charging = Mock()
        b._charging.charging = False
        b._charging.healthy.return_value = True
        b.battery_pct = Mock(return_value=50)
        b.drive_stop = Mock(return_value=True)
        b._last_charging_at = _time.monotonic() if charging_at == "now" else charging_at
        drive = Mock()
        drive.DriveState = Mock(Running=1, Completed=2, Error=3)
        drive.get_state.side_effect = state_script or [1, 2]
        drive.go_distance.return_value = 0
        b._drive = drive
        b.go_home = Mock(return_value="arrived")
        return b, gaps_holder

    def test_refuses_without_pose_proof(self):
        for pose in (None, [5000.0, 0.0, 0.0]):
            b, _ = self.rig(pose, {"cur": self.FRONT_PAIR})
            self.assertFalse(b.recover_dock_face())
            b._drive.go_distance.assert_not_called()
            b.go_home.assert_not_called()

    def test_refuses_without_recent_charging_proof(self):
        for charging_at in (None, -1.0):  # never, or long past the 30min gate
            b, _ = self.rig([100.0, 0.0, 0.0], {"cur": self.FRONT_PAIR},
                            charging_at=charging_at)
            self.assertFalse(b.recover_dock_face())
            b._drive.go_distance.assert_not_called()

    def test_refuses_on_any_other_gap_profile(self):
        b, _ = self.rig([100.0, 0.0, 0.0], {"cur": ["Front_Left", "Back_Right"]})
        self.assertFalse(b.recover_dock_face())
        b._drive.go_distance.assert_not_called()

    def test_refuses_when_heading_says_wandering_not_slipping(self):
        b, _ = self.rig([100.0, 0.0, 170.0], {"cur": self.FRONT_PAIR})
        self.assertFalse(b.recover_dock_face())
        b._drive.go_distance.assert_not_called()

    def test_retreat_then_full_visual_return(self):
        b, gaps = self.rig([100.0, 0.0, 0.0], {"cur": self.FRONT_PAIR})
        gaps_flip = lambda: gaps.__setitem__("cur", [])  # plate lip cleared
        b._pose_update = Mock(side_effect=lambda dist_mm=0, rot_deg=0:
                              (gaps_flip(), Body._pose_update(b, dist_mm=dist_mm,
                                                              rot_deg=rot_deg))[1])
        self.assertTrue(b.recover_dock_face())
        b.go_home.assert_called_once()
        self.assertAlmostEqual(b._pose[0], 120.0)  # 20mm step credited
        # the 30-minute cadence blocks an immediate second attempt
        self.assertFalse(b.recover_dock_face())
        b.go_home.assert_called_once()

    def test_gap_change_mid_step_stops_before_completion_and_loses_pose(self):
        b, gaps = self.rig([100.0, 0.0, 0.0], {"cur": self.FRONT_PAIR})
        polls = {"n": 0}

        def running_forever():
            polls["n"] += 1
            if polls["n"] >= 3:  # a rear gap appears mid-step
                gaps["cur"] = ["Front_Left", "Front_Right", "Back_Left"]
            return 1  # Running — never completes on its own

        b._drive.get_state.side_effect = running_forever
        self.assertFalse(b.recover_dock_face())
        b.go_home.assert_not_called()
        self.assertIsNone(b._pose)  # unmeasured travel invalidated the frame
        self.assertTrue(b.drive_stop.called)
        # interrupted attempt: short backoff, not the 30-minute cooldown
        import time as _t
        self.assertLessEqual(b._next_dock_face_recovery, _t.monotonic()+181)
    def test_refuses_at_three_percent_where_go_home_would_refuse(self):
        b, _ = self.rig([100.0, 0.0, 0.0], {"cur": self.FRONT_PAIR})
        b.battery_pct = Mock(return_value=3)
        self.assertFalse(b.recover_dock_face())
        b._drive.go_distance.assert_not_called()

    def test_step_timeout_invalidates_pose(self):
        b, gaps = self.rig([100.0, 0.0, 0.0], {"cur": self.FRONT_PAIR})
        b._drive.get_state.side_effect = lambda: 1  # Running until the window expires
        self.assertFalse(b.recover_dock_face())
        b.go_home.assert_not_called()
        self.assertIsNone(b._pose)  # unmeasured travel: frame untrusted


if __name__ == "__main__":
    unittest.main()
