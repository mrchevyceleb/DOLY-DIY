"""Bounded reverse onto a recently observed dock; rear gaps always stop."""
import math
import time

from .homing import angle_delta


class DockEntry:
    def __init__(self, body, distance, stop=None, allow_front_after=None):
        self.body, self.stop = body, stop
        self.distance = min(240, max(0, int(distance)))
        self.travelled = 0
        self.allow_front_after = (max(0, self.distance-60) if allow_front_after is None
                                  else allow_front_after)
        self.deadline = time.monotonic()+15
        self.heading = body._imu_yaw
        self.reason = None

    def check(self):
        b = self.body
        if self.reason:
            return self.reason
        if b._charging.charging is True:
            self.reason = "contact"
        elif b._approach_stop.is_set() or b.sleeping or (self.stop and self.stop()):
            self.reason = "cancelled"
        elif time.monotonic() >= self.deadline:
            self.reason = "timeout"
        elif (b.battery_pct() is None or b.battery_pct() <= 2
                or not b._charging.healthy() or b._charging.charging is not False):
            self.reason = "power"
        elif (self.heading is None or b._imu_yaw is None
                or not math.isfinite(b._imu_yaw)
                or time.monotonic()-b._imu_updated_at > .3
                or abs(angle_delta(b._imu_yaw, self.heading)) > 5):
            self.reason = "alignment"
        else:
            gaps = set(b._edge_gaps())
            allowed = {"Front_Left", "Front_Right"} if self.travelled >= self.allow_front_after else set()
            if not b.has.get("edge") or not gaps <= allowed:
                self.reason = "edge"
        return self.reason

    def trailing_gap(self, direction):
        return (self.travelled >= self.allow_front_after
                and direction in {"Front", "Front_Left", "Front_Right"}
                and self.check() is None)

    def run(self):
        b = self.body
        try:
            while self.travelled < self.distance:
                with b._power_lock:
                    b.refresh_power()
                    if self.check():
                        return self.reason
                    step = min(20, self.distance-self.travelled)
                    # Installed AiGoHome uses speed 10 for final entry.
                    rc = b._drive.go_distance(b._next_id(), step, 10, False, True)
                    if rc is False or (rc is not None and rc < 0):
                        return "not_started"
                end, running, complete = time.monotonic()+1.5, False, False
                while time.monotonic() < end:
                    with b._power_lock:
                        b.refresh_power()
                        if self.check():
                            return self.reason
                    state = b._drive.get_state()
                    if state == b._drive.DriveState.Running:
                        running = True
                    elif state == b._drive.DriveState.Completed and running:
                        complete = True
                        break
                    elif state == b._drive.DriveState.Error:
                        return "error"
                    time.sleep(.03)
                if not b.drive_stop():
                    return "error"
                if not complete:
                    return "timeout"
                self.travelled += step
            return "no_contact"
        finally:
            b.drive_stop()
