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


if __name__ == "__main__":
    unittest.main()
