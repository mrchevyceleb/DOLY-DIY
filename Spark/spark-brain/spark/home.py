"""Stock-style home memory: drive back to remembered dock coordinates.

Stock Doly keeps the dock pose in HomeControl and navigates to it blind
(AiGoHome.cpp: getPosition + atan2 + SetPointRotate/SetPointXY); the camera
marker is only used for the final approach. Spark mirrors that here.

The seated charging pose anchors a dead-reckoned world frame: [0, 0, 0],
heading 0 = facing away from the dock (the marker faces her). Every guarded
drive/rotate credits the estimate via Body._pose_update, so 'go home' can
first drive the remembered standoff blind and let the visual controller take
over for alignment and entry exactly as before.

This module only chooses WHERE to go, never WHETHER it is safe to move:
every millimetre travels through the existing guarded primitives with live
edge/power interlocks. A lost pose (pickup, airborne, carried) disables
blind navigation entirely and homing falls back to the pure visual search.
"""
import math

STANDOFF_MM = 450.0     # in front of the seated pose; marker in clear view
NEAR_MM = 100.0         # closer than this: the camera alone can finish
TRAVEL_CAP_MM = 1200.0  # per attempt — matches the visual approach budget
LEG_MM = 200.0          # blind legs between full interlock re-checks
DRIVE_SPEED = 25
SEGMENT_MM = 50
HEADING_TOL_DEG = 2.0


def _wrap(deg):
    return (deg + 180) % 360 - 180


def _turn_to(pose_get, body, target, interlock):
    """Guarded center turns (<=30 degrees each) until _pose faces target."""
    while True:
        pose = pose_get()
        if pose is None:
            return "lost"
        delta = _wrap(target - pose[2])
        if abs(delta) <= HEADING_TOL_DEG:
            return "ok"
        result = body.rotate_guarded(max(-30, min(30, delta)), interlock)
        if result != "ok":
            return result


def navigate_home(body, interlock):
    """Drive toward the remembered dock standoff, then face the dock.

    Returns 'ok' (at the standoff, marker should be dead ahead), 'skip'
    (no remembered pose — caller runs the pure visual search), 'limit'
    (travel budget spent mid-floor; caller should still try the camera),
    'lost' (pose invalidated mid-navigation), or an interlock reason from
    the guarded primitives ('stopped_edge', 'cancelled', 'power', ...).
    """
    if body._pose is None:
        return "skip"
    travelled = 0.0
    while True:
        pose = body._pose  # live estimate; guarded moves mutate it in place
        if pose is None:
            return "lost"  # pickup/airborne mid-leg: frame is gone
        dx, dy = STANDOFF_MM - pose[0], 0.0 - pose[1]
        dist = math.hypot(dx, dy)
        if dist <= NEAR_MM:
            break
        result = _turn_to(lambda: body._pose, body,
                          math.degrees(math.atan2(dy, dx)), interlock)
        if result != "ok":
            return result
        leg = min(LEG_MM, dist, TRAVEL_CAP_MM - travelled)
        if leg < 10:
            return "limit"
        result = body.drive_guarded(leg, speed=DRIVE_SPEED,
                                    segment_mm=SEGMENT_MM, interlock=interlock)
        if result != "ok":
            return result
        travelled += leg
    # Face the remembered dock so the marker is dead ahead for the camera.
    pose = body._pose
    if pose is None:
        return "lost"  # cleared between the last leg and the final turn
    dock_bearing = math.degrees(math.atan2(-pose[1], -pose[0]))
    return _turn_to(lambda: body._pose, body, dock_bearing, interlock)
