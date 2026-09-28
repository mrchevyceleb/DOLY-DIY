"""Stock-style dock entry, ported from the installed AiGoHome (stages 9-16).

The factory firmware reverses onto the dock in ONE continuous move at speed
20. Inside the observed dock corridor the rear pair losing ground is the
ramp, not a cliff: both rear sensors void means she is on the dock, so the
reverse carries on, then a slow speed-10 push (LoadEnterCharge) seats the
contacts. Only charging counts as arrival. No charge after the push: drive 100mm forward off the
ramp and re-approach from scratch (stage 16). Front gaps trail behind a
reverse and never stop it; an all-void before the ramp still does, and so
does a lone rear gap the other rear sensor never joins (a real cliff).
Without the ramp there is no push: stock only pushes on the dock.

The previous 20mm start/stop steps at speed 10 stalled on the ramp lip
(Sep 24: timeout at 120mm) and treated the ramp's rear voids as cliffs.
"""
import math
import sys
import time

from .homing import angle_delta

REVERSE_SPEED = 20  # AiGoHome stage 9 goXY
PUSH_SPEED = 10     # LoadEnterCharge
PUSH_MM = 60
SEAT_S = .4         # keep pressing briefly after the first charging current
PULL_OUT_MM = 100   # stage 16 retreat before a fresh approach
MAX_YAW_DRIFT = 15
LONE_REAR_S = .15   # ~15mm at speed 20 (130mm took ~1s live): the pair must join
_REAR = {"Back_Left", "Back_Right"}


def _log(message):
    print(f"[entry] {message}", file=sys.stderr, flush=True)


class DockEntry:
    def __init__(self, body, distance, stop=None):
        self.body, self.stop = body, stop
        self.distance = min(260, max(0, int(distance)))
        self.heading = body._imu_yaw
        self.reason = None
        self.ramp = False          # rear pair void inside the dock corridor
        self.pulling_out = False
        self.allowed_front = set()
        self._lone_rear_at = None
        self.moved = False         # a native move was accepted
        self.driving = False       # a native move is live (dock-stop exemption)
        self.deadline = time.monotonic() + 45

    def trailing_gap(self, direction):
        """Gap interrupt filter: True when the event is the dock, not a cliff."""
        if self.reason:
            return False
        if self.pulling_out:
            # Behind travel, or a seated-profile front sensor still clearing.
            return direction.startswith("Back") or direction in self.allowed_front
        if not self.ramp and direction.startswith("Back") and _REAR <= set(self.body._edge_gaps()):
            # Judged here too: the poll may not have run before the next
            # event (the pair's "All" once the front voids over the base).
            self._mark_ramp(direction)
        if direction == "All":
            return self.ramp
        return True

    def check(self):
        b = self.body
        if self.reason:
            return self.reason
        if b._approach_stop.is_set() or b.sleeping or (self.stop and self.stop()):
            self.reason = "cancelled"
        elif time.monotonic() >= self.deadline:
            self.reason = "timeout"
        elif (b.battery_pct() is None or b.battery_pct() <= 2
                or not b._charging.healthy()):
            self.reason = "power"
        elif not b.has.get("edge"):
            self.reason = "edge"
        elif (self.heading is None or b._imu_yaw is None
                or not math.isfinite(b._imu_yaw)
                or time.monotonic()-b._imu_updated_at > .3
                or abs(angle_delta(b._imu_yaw, self.heading)) > MAX_YAW_DRIFT):
            self.reason = "alignment"
        return self.reason

    def _contact(self):
        c = self.body._charging
        return c.charging is True or c.contact() is True

    def _mark_ramp(self, source):
        self.ramp, self._lone_rear_at = True, None
        _log(f"rear pair void: on the ramp ({source})")

    def _drive(self, mm, speed, forward=False):
        """One continuous native move: contact | done | stalled | reason."""
        b = self.body
        with b._power_lock:
            b.refresh_power()
            if not forward and self._contact():
                return "contact"
            if self.check():
                return self.reason
            rc = b._drive.go_distance(b._next_id(), int(mm), speed, forward, True)
            if rc is False or (rc is not None and rc < 0):
                return "not_started"
            self.moved = self.driving = True
        try:
            result = self._watch(mm, speed, forward)
        finally:
            self.driving = False
            stopped = b.drive_stop()
        if not stopped:
            self.reason = self.reason or "error"  # a live native move is never left behind
            return "error"
        return result

    def _watch(self, mm, speed, forward):
        b = self.body
        DS = b._drive.DriveState
        # Stall bound only: live travel is ~5mm/s per speed unit, this
        # allows ~0.5mm/s per unit plus spin-up.
        until = time.monotonic() + 2 + mm/(speed*.5)
        running, contact_at = False, None
        while time.monotonic() < until:
            with b._power_lock:
                b.refresh_power()
                reason = self.check()
                # Pulling out is never arrival: a stale shunt blip after
                # a rejected seat must not abort the retreat.
                contact = not forward and self._contact()
            if reason:
                return reason
            if contact:
                contact_at = contact_at or time.monotonic()
                if time.monotonic()-contact_at >= SEAT_S:
                    return "contact"
            gaps = set(b._edge_gaps())
            if forward:
                front = {g for g in gaps if g.startswith("Front")}
                self.allowed_front &= front  # ground returned: allowance gone
                if front - self.allowed_front:
                    return "edge"
            elif not self.ramp and _REAR <= gaps:
                # Progress, not a cliff: keep the calibrated reverse
                # going (stopping here stalled on the lip at speed 10).
                self._mark_ramp(f"gaps={sorted(gaps)}")
            elif not self.ramp and _REAR & gaps:
                now = time.monotonic()
                self._lone_rear_at = self._lone_rear_at or now
                if now - self._lone_rear_at >= LONE_REAR_S:
                    _log(f"lone rear gap never paired: cliff gaps={sorted(gaps)}")
                    self.reason = "edge"
                    return "edge"
            elif not self.ramp:
                self._lone_rear_at = None
            state = b._drive.get_state()
            if state == DS.Running:
                running = True
            elif state == DS.Completed and running:
                return "contact" if contact_at else "done"
            elif state == DS.Error:
                return "error"  # contact evidence never excuses a fault
            time.sleep(.03)
        return "contact" if contact_at else "stalled"

    def run(self):
        """Reverse, then push. Returns contact | no_contact | stalled | a stop
        reason. A controller error never earns more motion."""
        result = self._drive(self.distance, REVERSE_SPEED)
        _log(f"reverse {self.distance}mm -> {result} ramp={self.ramp}")
        if result in ("done", "stalled") and self.ramp:
            result = self._drive(PUSH_MM, PUSH_SPEED)
            _log(f"push {PUSH_MM}mm -> {result}")
            if result in ("done", "stalled"):
                result = "no_contact"
        elif result == "done":
            result = "no_contact"  # never reached the ramp: no blind push
        return result

    def pull_out(self):
        """Stock stage 16: no charge -> drive forward off the ramp."""
        b = self.body
        self.pulling_out = True
        self.reason = None
        self.deadline = time.monotonic() + 15
        # The seated profile reads the front pair; it may clear, never grow.
        self.allowed_front = {g for g in b._edge_gaps() if g.startswith("Front")}
        result = self._drive(PULL_OUT_MM, REVERSE_SPEED, forward=True)
        _log(f"pull out {PULL_OUT_MM}mm -> {result}")
        if result == "done" and self.ramp:
            # Straight off the dock ramp: re-anchor the home frame so the
            # next attempt drives the stock standoff instead of a stale pose.
            b._pose = [PULL_OUT_MM + 40.0, 0.0, 0.0]
        else:
            b._pose = None  # entry travel is never credited to the frame
        return "ok" if result == "done" else result
