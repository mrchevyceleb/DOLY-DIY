"""Camera-guided dock approach with a bounded, gyro-checked reverse entry.

Approach and turns use ordinary motor interlocks. The bounded final reverse
allows trailing front gaps on the ramp; rear gaps always stop. Charging must
remain confirmed with stopped motors.
"""
import math
import statistics
import sys
import time


def _log(message):
    print(f"[home] {message}", file=sys.stderr, flush=True)


def angle_delta(a, b):
    return (a - b + 180) % 360 - 180


def dock_yaw_candidates(observations):
    """Keep each pose supported by all stationary frames."""
    choices = [getattr(o, "dock_yaw_candidates_deg", []) for o in observations]
    if any(not values or any(not math.isfinite(v) or abs(v) > 70 for v in values)
           for values in choices):
        return []
    agreed = [v for v in choices[0] if all(any(abs(v-w) <= 4 for w in values)
                                         for values in choices[1:])]
    return agreed


def dock_yaw(observations, expected=None):
    """Use only normal angles supported by every stationary observation."""
    agreed = dock_yaw_candidates(observations)
    # Gyro history resolves competing planar solutions. Do not let accumulated
    # heading/traction error override a new, unambiguous visual measurement.
    if agreed and max(agreed)-min(agreed) > 8 and expected is not None:
        agreed = [v for v in agreed if abs(v-expected) <= 6]
    if not agreed or max(agreed)-min(agreed) > 8:
        return None
    return statistics.median(agreed)


def consistent_plane(views):
    """Find a dock orientation supported at separated, measured headings."""
    if len(views) < 2 or any(not angles for _, angles in views):
        return None
    reference = views[0][0]
    choices = [[angle_delta(heading+yaw, reference) for yaw in angles]
               for heading, angles in views]
    matches = []
    for candidate in choices[0]:
        group = [candidate]
        for angles in choices[1:]:
            closest = min(angles, key=lambda value: abs(angle_delta(value, candidate)))
            if abs(angle_delta(closest, candidate)) > 5:
                break
            group.append(candidate+angle_delta(closest, candidate))
        else:
            matches.append(statistics.mean(group))
    if not matches and len(views) == 3 and all(len(v) == 1 for v in choices):
        # Three unique poses can contain one noisy angle. Require a tight
        # pair and a bounded outlier; final entry still checks a fresh pose.
        values = [v[0] for v in choices]
        median = statistics.median(values)
        errors = sorted(abs(v-median) for v in values)
        if errors[1] <= 3 and errors[2] <= 10:
            return reference+median
    if not matches or max(matches)-min(matches) > 6:
        return None
    return reference+statistics.mean(matches)


def approach_waypoint(x, z, yaw):
    """Center on the dock normal, preserving an already adequate standoff."""
    radians = math.radians(yaw)
    distance = z*math.cos(radians)-x*math.sin(radians)
    standoff = max(180, min(200, distance))
    return x+standoff*math.sin(radians), z-standoff*math.cos(radians)


class Homing:
    def __init__(self, body, stop=None):
        self.body, self.stop = body, stop
        self.deadline = time.monotonic() + 120
        self.reversing = False

    def interlock(self):
        b = self.body
        if b.refresh_power() is True:
            return "contact"
        if b._approach_stop.is_set() or b.sleeping or (self.stop and self.stop()):
            return "cancelled"
        if time.monotonic() >= self.deadline:
            return "timeout"
        if b.battery_pct() is None or b.battery_pct() <= 2 or b.actuators_held():
            return "power"
        if (b._edge_gaps() or b._hazard_active("forward")
                or b._hazard_active("backward")):
            return "edge"
        if self.heading() is None:
            return "sensor"
        # The front ToF pair guards forward travel and wheel sweeps. During
        # reverse entry they face away from the dock; edge guards stay active.
        if not self.reversing:
            return b.approach_proximity()
        return None

    def heading(self):
        b = self.body
        value = b._imu_yaw
        if (value is None or not math.isfinite(value)
                or time.monotonic() - b._imu_updated_at > .3):
            return None
        return value

    def ready(self):
        """Camera startup can briefly delay SDK callbacks; wait at rest."""
        until = time.monotonic() + 2
        while True:
            reason = self.interlock()
            if reason not in ("sensor", "power") or time.monotonic() >= until:
                return reason
            self.body.drive_stop()
            time.sleep(.05)

    def camera_interrupted(self):
        # Observe only at rest. A brief invalid ToF sample need not discard
        # the camera session; ready() requires valid data before any motion.
        reason = self.interlock()
        return reason is not None and reason != "sensor"

    def charge_verified(self):
        b = self.body
        b.drive_stop()
        # The charging rail took ~3 seconds to settle in the floor test;
        # leave enough time for the current average plus one steady second.
        until, since = time.monotonic() + 8, None
        while time.monotonic() < until:
            if b.refresh_power() is True:
                since = since if since is not None else time.monotonic()
                if time.monotonic() - since >= 1:
                    return True
            else:
                since = None
            time.sleep(.1)
        return False

    def half_turn(self, clockwise=True):
        """Measure actual yaw; SDK center turns over-rotate on this robot."""
        start = self.heading()
        if start is None:
            return "sensor"
        probe = 5 if clockwise else -5
        result = self.body.rotate_guarded(probe, self.interlock)
        if result != "ok":
            return result
        time.sleep(.1)
        current = self.heading()
        if current is None:
            return "sensor"
        change = angle_delta(current, start)
        if not 2 <= abs(change) <= 20:
            return "turn_unverified"
        direction = math.copysign(1, change/probe)
        target = start + math.copysign(180, change)
        return self.turn_to(target, direction)

    def turn_to(self, target, direction=-1):
        """Turn to an IMU heading; positive SDK turns reduce Doly yaw."""
        # Repeated tiny start/stop turns displaced the robot on the floor.
        # Keep one slow sweep moving, stopping on measured heading or any
        # interlock; corrections keep the original target after a sensor pause.
        for _ in range(4):
            reason = self.ready()
            if reason:
                return reason
            current = self.heading()
            if current is None:
                return "sensor"
            error = angle_delta(target, current)
            _log(f"turn yaw={current:.1f} error={error:.1f}")
            if abs(error) <= 3:
                return "ok"
            command = max(-175, min(175, error/direction))
            if not self.body.drive_rotate(command, speed=10, from_center=True):
                return "blocked"
            started = time.monotonic()
            until, running, result = started+12, False, "timeout"
            swept, last = 0, current
            try:
                while time.monotonic() < until:
                    result = self.interlock()
                    if result:
                        break
                    after = self.heading()
                    if after is None:
                        result = "sensor"
                        break
                    swept += angle_delta(after, last)
                    last = after
                    if (swept*error < -10 or abs(swept) > abs(error)+12
                            or (time.monotonic()-started > 1.5 and abs(swept) < 1)):
                        result = "turn_unverified"
                        break
                    remaining = angle_delta(target, after)
                    if abs(remaining) <= 2 or remaining*error < 0:
                        result = "ok"
                        break
                    state = self.body._drive.get_state()
                    if state == self.body._drive.DriveState.Running:
                        running = True
                    elif state == self.body._drive.DriveState.Completed and running:
                        result = "ok"
                        break
                    elif state == self.body._drive.DriveState.Error:
                        result = "blocked"
                        break
                    time.sleep(.02)
                else:
                    result = "timeout"
            finally:
                self.body.drive_stop()
            if result not in ("ok", "sensor"):
                return result
            time.sleep(.15)
            after = self.heading()
            if result == "ok" and (after is None or abs(angle_delta(after, current)) < .5):
                return "turn_unverified"
        if self.heading() is not None and abs(angle_delta(target, self.heading())) <= 3:
            return "ok"
        return "alignment"

    def waypoint_step(self, x, z):
        """Move at most 30mm toward a freshly observed staging point.

        The dock may leave view while turning sideways. Only the planned
        short step is allowed, then restore the original heading and look
        again. All ordinary forward, turn, edge and power guards still apply.
        """
        start = self.heading()
        observed_at = time.monotonic()
        if start is None or z < -10:
            return "alignment"
        bearing = math.degrees(math.atan2(x, z))
        if abs(bearing) > 120:
            return "alignment"
        result = self.turn_to(start-bearing)
        if result != "ok":
            return result
        if time.monotonic()-observed_at > 8:
            return "lost"
        result = self.body.drive_guarded(min(30, int(math.hypot(x, z))), speed=15,
                                         segment_mm=30, interlock=self.interlock)
        if result != "ok":
            return result
        return self.turn_to(start)

    def measure_plane(self, camera, observations):
        """Resolve planar ambiguity with guarded separated views.

        Close range strengthens the marker's two-pose ambiguity; widen the
        parallax when narrow views disagree, and fall back to plain marker
        yaw (the final entry gate still demands a fresh visual check).
        """
        start = self.heading()
        if start is None:
            return "sensor", None
        angles = dock_yaw_candidates(observations)
        if not angles:
            return "alignment", None
        views = [(start, angles)]
        _log(f"plane view heading={start:.1f} candidates={angles}")
        sign = -1 if observations[-1].camera_x_mm > 0 else 1
        for offset in (sign*10, -sign*10, sign*20, -sign*20):
            result = self.turn_to(start+offset)
            if result != "ok":
                return result, None
            heading = self.heading()
            if heading is None or abs(angle_delta(heading, start)) < 6:
                return "turn_unverified", None
            frames = [camera.observe() for _ in range(3)]
            if any(o is None for o in frames):
                return "lost", None
            if (max(o.camera_x_mm for o in frames)-min(o.camera_x_mm for o in frames) > 15
                    or max(o.camera_z_mm for o in frames)-min(o.camera_z_mm for o in frames) > 25):
                return "lost", None
            views.append((heading, dock_yaw_candidates(frames)))
            _log(f"plane view heading={heading:.1f} candidates={views[-1][1]}")
            plane = consistent_plane(views)
            if plane is not None:
                _log(f"measured dock plane heading={plane:.1f}")
                return "ok", plane
        # Fallback is guidance for reorientation, not docking permission.
        # The final entry gate still insists on a fresh <=5-degree view.
        fallback = dock_yaw(observations)
        if fallback is not None and abs(fallback) <= 60:
            plane = start + fallback
            _log(f"plane unresolved; marker yaw fallback={plane:.1f}")
            return "ok", plane
        return "alignment", None

    def approach(self, camera):
        searched, travelled, seen, retreats, remeasurements = 0, 0, False, 0, 0
        plane_heading = None
        while True:
            reason = self.ready()
            if reason:
                return reason, None
            observation = camera.observe()
            reason = self.ready()
            if reason:
                return reason, None
            if observation is None:
                if seen:
                    return "lost", None
                if searched >= 350:
                    return "not_found", None
                before = self.heading()
                result = self.body.rotate_guarded(13.5, self.interlock)
                if result != "ok":
                    return result, None
                after = self.heading()
                if before is None or after is None or abs(angle_delta(after, before)) < 1:
                    return "turn_unverified", None
                searched += abs(angle_delta(after, before))
                continue
            confirmed = camera.observe()
            reason = self.ready()
            if reason:
                return reason, None
            if (confirmed is None
                    or abs(confirmed.camera_x_mm-observation.camera_x_mm) > 15
                    or abs(confirmed.camera_z_mm-observation.camera_z_mm) > 25):
                return "lost", None
            observation, seen = confirmed, True
            x, z = observation.camera_x_mm, observation.camera_z_mm
            bearing = math.degrees(math.atan2(x, z))
            third = camera.observe()
            if (third is None or abs(third.camera_x_mm-x) > 15
                    or abs(third.camera_z_mm-z) > 25):
                return "lost", None
            heading = self.heading()
            if heading is None:
                return "sensor", None
            if plane_heading is None:
                if abs(bearing) > 10:
                    result = self.turn_to(heading-bearing)
                    if result != "ok":
                        return result, None
                    continue
                result, plane_heading = self.measure_plane(camera, [observation, third])
                if result != "ok":
                    return result, None
                continue  # fresh position after the measured camera turn
            yaw = angle_delta(plane_heading, heading)
            angles = dock_yaw_candidates([observation, third])
            supported = [a for a in angles if abs(angle_delta(a, yaw)) <= 8]
            _log(f"marker x={x:.1f}mm z={z:.1f}mm bearing={bearing:.1f} normal={yaw} visual={angles}")
            if not supported:
                if not angles:
                    # A transient corner-detection dropout (visual=[]) must
                    # not abort a converging approach — recent frames held
                    # valid poses. Re-observe briefly before giving up.
                    misses = getattr(self, "_drop_frames", 0) + 1
                    self._drop_frames = misses
                    if misses <= 3:
                        time.sleep(.4)
                        continue
                    self._drop_frames = 0
                    return "alignment", None
                self._drop_frames = 0
                if abs(bearing) > 8:
                    result = self.body.rotate_guarded(max(-10, min(10, bearing*.45)), self.interlock)
                    if result != "ok":
                        return result, None
                    continue  # resolve the plane with the marker nearer image center
                return "alignment", None
            self._drop_frames = 0
            # Sidling alone preserves the old heading and can never satisfy
            # the <=5-degree entry gate from a 50-degree oblique approach.
            # Retreat to a safe visual standoff, then close the measured
            # heading error in guarded 15-degree increments, re-observing
            # the marker after EACH increment (never a blind 60-degree spin).
            if abs(yaw) > 15:
                plane_distance = z*math.cos(math.radians(yaw))-x*math.sin(math.radians(yaw))
                if z < 320 or plane_distance < 175:
                    if retreats >= 6 or travelled >= 1200:
                        return "too_close", None
                    result = self.body.drive_guarded(-40, speed=15, segment_mm=20,
                                                     interlock=self.interlock)
                    travelled += 40
                    retreats += 1
                    if result != "ok":
                        return result, None
                else:
                    increment = max(-15, min(15, yaw))
                    result = self.turn_to(heading+increment)
                    if result != "ok":
                        return result, None
                continue
            # Keep the measured world orientation through small pose flips.
            # The dock is stationary; a fresh image still must support it.
            if 175 <= z <= 220 and abs(bearing) <= 2 and abs(yaw) <= 5:
                visual_yaw = dock_yaw([observation, third], yaw)
                if visual_yaw is not None and abs(visual_yaw) <= 5:
                    return "aligned", observation
                if remeasurements >= 2:
                    return "alignment", None
                plane_heading = None
                remeasurements += 1
                continue
            plane_distance = z*math.cos(math.radians(yaw))-x*math.sin(math.radians(yaw))
            if plane_distance < 175:
                if retreats >= 2 or travelled >= 1200:
                    return "too_close", None
                result = self.body.drive_guarded(-40, speed=15, segment_mm=20,
                                                 interlock=self.interlock)
                travelled += 40
                retreats += 1
                if result != "ok":
                    return result, None
                continue
            wx, wz = approach_waypoint(x, z, yaw)
            if math.hypot(wx, wz) <= 12:
                steering = bearing
                step = 0
            elif wz >= -10:
                steering = math.degrees(math.atan2(wx, wz))
                if abs(steering) > 20:
                    if travelled >= 1200:
                        return "limit", None
                    result = self.waypoint_step(wx, wz)
                    travelled += 30
                    if result != "ok":
                        return result, None
                    continue
                step = min(40, max(0, int(wz-5)))
            else:
                return "alignment", None
            if abs(steering) > 2:
                result = self.body.rotate_guarded(max(-10, min(10, steering*.45)), self.interlock)
            elif step >= 10:
                if travelled >= 1200:
                    return "limit", None
                result = self.body.drive_guarded(step, speed=20, segment_mm=40,
                                                 interlock=self.interlock)
                travelled += step
            else:
                return "alignment", None
            if result != "ok":
                return result or "blocked", None

    def reverse_entry(self, observation):
        from .dock_entry import DockEntry
        b = self.body
        reason = self.ready()
        if reason:
            return reason
        visual_distance = math.ceil(observation.camera_z_mm)
        # Floor calibration: visual-distance travel left the contacts 20mm
        # short. The single bounded trim established sustained charging.
        entry = DockEntry(b, visual_distance+20, self.stop,
                          allow_front_after=max(0, visual_distance-60))
        b._docking_entry = entry
        try:
            result = entry.run()
            _log(f"entry travelled={entry.travelled}mm result={result}")
            return "arrived" if self.charge_verified() else result
        finally:
            b._docking_entry = None

    def run(self):
        from .dock_camera import DockCamera
        self.body.drive_stop()
        reason = self.ready()
        if reason:
            return reason
        # Installed AiGoHome sets both arms to 0 before approaching.
        if self.body.has.get("arm") and not self.body.arm_angle(0, speed=30):
            return "posture"
        with DockCamera(self.camera_interrupted) as camera:
            result, observation = self.approach(camera)
        if result != "aligned":
            return "arrived" if result == "contact" and self.charge_verified() else result
        aligned_at = time.monotonic()
        # Close the camera before entry to reduce power during rail switching.
        reason = self.ready()
        if reason:
            return reason
        result = self.half_turn(clockwise=False)
        if result != "ok":
            return result
        if time.monotonic()-aligned_at > 25:
            return "lost"
        return self.reverse_entry(observation)
