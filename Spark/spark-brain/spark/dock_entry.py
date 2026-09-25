"""Bounded reverse onto a recently observed dock; rear gaps stop immediately."""
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
        # Speed 10 moves ~10mm/s: the full distance needs ~distance/10
        # seconds of motion alone, plus per-step overhead and corrections.
        self.deadline = time.monotonic()+10+self.distance*0.15
        self.heading = body._imu_yaw
        self.initial_heading = self.heading
        self.reason = None
        self.corrections = 0

    def _correct(self):
        """Back off a one-corner ramp miss or small traction yaw drift.

        Never drive into a gap. Front gaps, paired rear gaps, stale IMU,
        contact uncertainty and repeated corrections remain hard stops.
        """
        b = self.body
        gaps = set(b._edge_gaps())
        drift = (angle_delta(b._imu_yaw, self.heading) if self.heading is not None
                 and b._imu_yaw is not None else None)
        rear = gaps in ({"Back_Left"}, {"Back_Right"})
        if (self.corrections >= 2 or time.monotonic() >= self.deadline
                or self.reason not in ("edge", "alignment")
                or (self.reason == "edge" and not rear)
                or (self.reason == "alignment" and (gaps or drift is None or abs(drift) > 10))
                or b._imu_yaw is None or not math.isfinite(b._imu_yaw)
                or time.monotonic()-b._imu_updated_at > .3
                or b._charging.charging is not False or not b._charging.healthy()
                or b._approach_stop.is_set() or b.sleeping or (self.stop and self.stop())):
            return False
        generation = b._hazard_gen
        # Give the cliff sensor's emergency lock its full three seconds.
        wait = getattr(b, "_gap_lock_until", 0)-time.time()
        if wait > 0:
            time.sleep(min(wait+.1, 3.5))
        if b._hazard_gen != generation or time.monotonic() >= self.deadline:
            return False
        previous = b._docking_entry
        b._docking_entry = None  # motion preflights still guard every move
        try:
            b.drive_stop()
            if b.drive_guarded(30, speed=15, segment_mm=20,
                    interlock=lambda: "cancelled" if b._approach_stop.is_set()
                    or (self.stop and self.stop()) else
                    "edge" if b._hazard_gen != generation else None) != "ok":
                return False
            b.drive_stop()
            if b._edge_gaps() or b._hazard_gen != generation:
                return False
            # The former rear cliff latch protected the retreat; after the
            # verified clear, retire it for ONE bounded pivot, generation-safe.
            with b._hazard_lock:
                if b._hazard_gen != generation:
                    return False
                b._edge_hazard = None
                b._hazard_clear_at = None
            if rear:
                command = 4 if "Back_Right" in gaps else -4
            else:
                command = max(-6, min(6, drift))  # SDK command opposes IMU yaw
            before = b._imu_yaw
            if not b.drive_rotate(command, speed=10, from_center=True):
                return False
            if not b._wait_drive_idle(timeout=4, require_running=True):
                return False
            b.drive_stop()
            if (b._edge_gaps() or b._hazard_gen != generation
                    or b._imu_yaw is None or time.monotonic()-b._imu_updated_at > .3
                    or abs(angle_delta(b._imu_yaw, before)) < 2
                    or abs(angle_delta(b._imu_yaw, self.initial_heading)) > 8):
                return False
            self.heading = b._imu_yaw
            self.travelled = max(0, self.travelled-30)
            self.corrections += 1
            self.reason = None
            self.deadline = max(self.deadline, time.monotonic()+8)
            return True
        finally:
            b.drive_stop()
            b._docking_entry = previous

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
                    reason = self.check()
                    if reason:
                        if reason not in ("edge", "alignment"):
                            return reason
                    else:
                        step = min(20, self.distance-self.travelled)
                if reason:
                    if self._correct():
                        continue
                    return self.reason or reason
                with b._power_lock:
                    if self.check():
                        return self.reason
                    # Installed AiGoHome uses speed 10 for final entry.
                    rc = b._drive.go_distance(b._next_id(), step, 10, False, True)
                    if rc is False or (rc is not None and rc < 0):
                        return "not_started"
                # A 20mm step at speed 10 needs ~2s; 1.5s made every step a
                # coin flip against the completion watchdog (real stall loss).
                end, running, complete, corrected = time.monotonic()+3, False, False, False
                while time.monotonic() < end:
                    with b._power_lock:
                        b.refresh_power()
                        reason = self.check()
                    if reason:
                        b.drive_stop()
                        if reason in ("edge", "alignment"):
                            # Native step may have travelled anywhere from 0
                            # to 20mm before the stop. Credit its full bound
                            # so corrections never exceed the entry limit.
                            self.travelled += step
                            if self._correct():
                                corrected = True
                                break
                        return self.reason or reason
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
                if corrected:
                    continue
                if not complete:
                    return "timeout"
                self.travelled += step
            return "no_contact"
        finally:
            b.drive_stop()
