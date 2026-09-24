"""Short pet-like moves, bounded by the last visual dock observation."""
import math
import random
import time

from .homing import Homing, angle_delta


class Roaming(Homing):
    def interlock(self):
        reason = super().interlock()
        if reason:
            return reason
        low = self.body.cfg.get("idle", {}).get("low_battery_pct", 10)
        if self.body.battery_pct() <= low:
            return "low_battery"
        return None

    def locate(self, camera):
        swept = 0
        for _ in range(24):
            reason = self.ready()
            if reason:
                return reason, None
            observation = camera.observe()
            if observation:
                confirmed = camera.observe()
                if (confirmed and abs(confirmed.camera_z_mm-observation.camera_z_mm) < 25
                        and abs(confirmed.camera_x_mm-observation.camera_x_mm) < 15):
                    return "found", confirmed
                return "lost", None
            if swept >= 350:
                return "not_found", None
            before = self.heading()
            result = self.body.rotate_guarded(10, self.interlock)
            after = self.heading()
            if result != "ok" or before is None or after is None:
                return result or "sensor", None
            swept += abs(angle_delta(after, before))
        return "not_found", None

    def run(self):
        from .dock_camera import DockCamera
        b = self.body
        self.deadline = time.monotonic()+45
        # Boxed in too close to rotate (locate sweeps and moves both blocked):
        # back clear of the proximity zone before anything else.
        if getattr(b, "_roam_blocked_count", 0) >= 5:
            b._roam_blocked_count = 3
            result = b.drive_guarded(-40, speed=15, segment_mm=20, interlock=self.interlock)
            if result != "ok":
                return result
        radius = min(3000, max(300, b.cfg.get("idle", {}).get("roam_radius_mm", 3000)))
        bound = b._roam_distance_bound
        if bound is None or bound+80 >= radius:
            with DockCamera(self.camera_interrupted) as camera:
                result, dock = self.locate(camera)
            if result != "found":
                b._roam_blocked_count = (getattr(b, "_roam_blocked_count", 0) + 1
                                         if result == "obstacle" else 0)
                return result
            b._roam_distance_bound = math.hypot(dock.camera_x_mm, dock.camera_z_mm)+80
            if b._roam_distance_bound+80 >= radius:
                # Facing a visible dock: take a small inward step, then
                # reacquire next time. Do not blindly continue outward.
                return b.drive_guarded(40, speed=20, segment_mm=40, interlock=self.interlock)
            if dock.camera_z_mm < 260:
                result = self.half_turn()
                if result != "ok":
                    return result
        reason = self.ready()
        if reason:
            return reason
        stuck = getattr(b, "_roam_blocked_count", 0)
        if random.random() < .4 or stuck >= 3:
            # repeated obstacles demand a decisive new heading, not a twitch
            turn = random.choice([-15, -10, 10, 15]) if stuck < 3 else random.choice([-45, -30, 30, 45])
            result = b.rotate_guarded(turn, self.interlock)
            b._roam_blocked_count = stuck + 1 if result == "obstacle" else 0
            return result
        result = b.drive_guarded(60, speed=20, segment_mm=30, interlock=self.interlock)
        b._roam_blocked_count = stuck + 1 if result == "obstacle" else 0
        return result
