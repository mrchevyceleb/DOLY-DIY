"""Bounded, camera-guided approach. No translation without a visible target."""
import math
import sys
import time


def _log(message):
    print(f"[approach] {message}", file=sys.stderr, flush=True)


class Approach:
    # Floor trials show center turns change the camera bearing by roughly
    # twice the requested angle. Close the loop gently and re-observe after
    # each small turn instead of alternating full-error corrections.
    TURN_GAIN = .45

    def __init__(self, body, camera, stop_check=None):
        self.body, self.camera, self.stop_check = body, camera, stop_check
        self.deadline = time.monotonic() + 45
        self.sensor_retries = 0

    def _recover_sensor(self):
        """Brake through transient optical noise; never drive on bad data."""
        if not self.body.drive_stop():
            return "blocked"
        if self.sensor_retries >= 3:
            return "sensor"
        self.sensor_retries += 1
        deadline, clear = time.monotonic() + .8, 0
        _log("sensor interrupted motion; waiting stationary for clean readings")
        while time.monotonic() < deadline:
            reason = self.interlock()
            if reason not in (None, "sensor"):
                return reason
            clear = clear + 1 if reason is None else 0
            if clear >= 2:
                _log("sensor readings recovered; rechecking camera before movement")
                return None
            time.sleep(.05)
        return "sensor"

    def _ready(self):
        reason = self.interlock()
        return self._recover_sensor() if reason == "sensor" else reason

    def _observe(self):
        # Nearby people must have usable body detail. A tiny shelf decoration
        # scored as a second person (6% wide, 9% tall) during floor trials.
        return [p for p in self.camera.observe() if p.height >= .20 and p.width >= .08]

    def interlock(self):
        b = self.body
        if b._approach_stop.is_set() or (self.stop_check and self.stop_check()):
            b._approach_stop.set()
            return "cancelled"
        if time.monotonic() >= self.deadline:
            return "limit"
        if b.actuators_held():
            if b.docked:
                return "docked"
            return "edge" if b._edge_gaps() else "power"
        if b.battery_pct() is None or b.battery_pct() <= 10:
            return "power"
        if b._edge_gaps() or b._hazard_active("forward") or b._hazard_active("backward"):
            return "edge"
        return b.approach_proximity()

    def run(self):
        searched, travelled, found = 0, 0, False
        while True:
            reason = self._ready()
            if reason:
                return reason
            targets = self._observe()
            if len(targets) > 1:
                return "ambiguous"
            if not targets:
                if found:
                    return "lost"
                if searched >= 330:
                    return "not_found"
                result = self.body.rotate_guarded(30*self.TURN_GAIN, interlock=self.interlock)
                if result != "ok":
                    if result == "sensor":
                        result = self._recover_sensor()
                        if result is None:
                            continue
                    return result or "blocked"
                searched += 30
                _log(f"search angle={searched}")
                continue
            target = targets[0]
            confirmed = self._observe()
            if len(confirmed) > 1:
                return "ambiguous"
            if len(confirmed) != 1 or target.overlaps(confirmed[0]) < .4:
                return "lost"
            target = confirmed[0]
            found = True
            # Calibration at 640x480: fx=369.45, cx=312.37.
            bearing = math.degrees(math.atan((target.center*640-312.37)/369.45))
            _log(f"person score={target.score:.2f} bearing={bearing:.1f} "
                 f"height={target.height:.2f} travel={travelled}mm")
            reason = self.interlock()  # camera/inference never bypasses new hazards
            if reason == "sensor":
                reason = self._recover_sensor()
                if reason is None:
                    continue  # look again after waiting, before any turn/step
            if reason:
                return reason
            if abs(bearing) > 6:
                turn = max(-10, min(10, bearing*self.TURN_GAIN))
                result = self.body.rotate_guarded(turn, interlock=self.interlock)
            # Width alone depends heavily on pose (e.g. kneeling). A person
            # three feet away filled 45% of the width and was called "near".
            # Only use almost-full framing as a visual stop; the short-range
            # sensors and travel budget independently bound the approach.
            elif target.height >= .95 and target.width >= .85:
                return "near"
            elif travelled >= 600:
                return "limit"
            else:
                result = self.body.drive_guarded(40, speed=20, segment_mm=40,
                                                interlock=self.interlock)
                if result in ("ok", "sensor"):
                    # An interrupted step may have moved partway. Reserve its
                    # full distance so retries cannot exceed the travel limit.
                    travelled += 40
            if result != "ok":
                if result == "sensor":
                    result = self._recover_sensor()
                    if result is None:
                        continue
                return result or "blocked"
