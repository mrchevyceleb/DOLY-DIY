"""A measured 20mm dock exit, then guarded clearance before any turn.

The stock dock's front gap profile cleared within this step in the floor
test. Only a recently confirmed charger permits that initial profile. Once
ground returns, front gaps always cancel; rear gaps are allowed only while
moving straight forward to clear the base.
"""
import sys
import time


class Departure:
    def __init__(self, body, forward, gaps, stop, front_probe=False):
        self.body = body
        self.forward = forward
        self.allowed_gaps = set(gaps)
        self.front_probe = front_probe
        self.clearing = False
        self.stop = stop
        self.deadline = time.monotonic() + 10
        self.reason = None

    def check(self):
        """Called with the body's power lock, including by its monitor."""
        b = self.body
        if self.reason:
            return self.reason
        if not b._leaving_home or b.sleeping or b._approach_stop.is_set():
            self.reason = "cancelled"
        elif time.monotonic() >= self.deadline:
            self.reason = "timeout"
        elif (b.battery_pct() is None or b.battery_pct() <= 2
              or not b._charging.healthy()):
            self.reason = "power"
        else:
            gaps = set(b._edge_gaps())
            leading = "Front" if self.forward else "Back"
            allowed = {"Back_Left", "Back_Right"} if self.clearing else self.allowed_gaps
            if (not b.has.get("edge") or not gaps <= allowed
                    or (not self.front_probe and any(g.startswith(leading) for g in gaps))):
                self.reason = "edge"
            else:
                self.allowed_gaps.intersection_update(gaps)
        return self.reason

    def trailing_gap(self, direction):
        """Forward clearance moves away from the dock under its rear sensors.

        This never applies to the initial probe, turns, a front gap or an
        airborne/all-gap event. The front pair remains the leading guard.
        """
        return (self.clearing and self.forward
                and direction in {"Back", "Back_Left", "Back_Right"}
                and self.check() is None)

    def poll(self):
        b = self.body
        if self.stop():
            b._approach_stop.set()
        with b._power_lock:
            b.refresh_power()
            return self.check()

    def verify_contact_clear(self):
        """Motor load alone cannot prove that the charging contacts cleared."""
        b = self.body
        until = min(self.deadline, time.monotonic() + 3)
        clear_since = None
        while time.monotonic() < until:
            if self.poll():
                return self.reason
            if b._edge_gaps():
                return "edge"
            if b._charging.charging is False:
                if clear_since is None:
                    clear_since = time.monotonic()
                elif time.monotonic() - clear_since >= 1.25:
                    b._pose = None
                    return "ok"
            else:
                clear_since = None
            time.sleep(.05)
        return "still_docked"

    def step(self):
        b = self.body
        DS = b._drive.DriveState
        if self.poll():
            return self.reason
        with b._power_lock:
            if self.check():
                return self.reason
            accepted = b._drive.go_distance(b._next_id(), 20, 25, self.forward, True)
            if accepted is False or (accepted is not None and accepted < 0):
                return "not_started"
        started = time.monotonic()
        running = False
        completed = False
        # A stalled encoder must not leave the dock-lip allowance open.
        while time.monotonic() - started < .8:
            if self.poll():
                return self.reason
            state = b._drive.get_state()
            if state == DS.Running:
                running = True
            elif state == DS.Error:
                return "error"
            elif state == DS.Completed and running:
                completed = True
                break
            time.sleep(.03)
        if not b.drive_stop():
            return "error"
        self.front_probe = False  # never extend or repeat the measured probe
        if not completed:
            return "timeout"
        print(f"[departure] 20mm step completed gaps={b._edge_gaps()}",
              file=sys.stderr, flush=True)
        return "ok"

    def run(self):
        b = self.body
        if self.poll():
            return self.reason
        if not b._edge_gaps() and b._charging.charging is False:
            # Explicitly requested release of a stale hold needs no probe.
            return self.verify_contact_clear()
        result = self.step()
        if result != "ok":
            return result
        # SDK RPM reported zero during verified movement in the floor test.
        # Completion/odometry alone also cannot prove departure. Ground and
        # sustained discharge at rest are the required physical evidence.
        result = self.verify_contact_clear()
        if result != "ok":
            return result
        self.clearing = True
        # Contacts clear before the body clears the base. Move straight
        # another 80mm before allowing any arm action or wheel sweep.
        # Front gaps stop every segment; rear-only gaps are behind travel.
        for _ in range(4):
            result = self.step()
            if result != "ok":
                return result
        self.clearing = False
        self.allowed_gaps.clear()
        return self.verify_contact_clear()
