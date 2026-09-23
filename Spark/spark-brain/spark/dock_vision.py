"""Read-only detection of the stock charging dock. No motor control.

The dock uses marker 2 in Doly's custom dictionary, not a predefined ArUco
dictionary. Native 640x480 capture crops the view on Spark; capture 1280x960
and resize to the dimensions in the installed calibration instead.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DockObservation:
    corners_px: list
    camera_x_mm: float
    camera_z_mm: float
    reprojection_error_px: float
    dock_yaw_candidates_deg: list


class DockDetector:
    def __init__(self, dictionary_path="/.doly/data/doly.dict",
                 calibration_path="/.doly/data/calibration_D8N12K.yml"):
        lines = Path(dictionary_path).read_text().splitlines()
        words = [line.strip() for line in lines[2:]]
        if len(words) != 3 or any(len(w) != 16 or set(w) - {"0", "1"} for w in words):
            raise ValueError("Expected Doly's three 4x4 markers")
        bits = [np.array([int(c) for c in w], dtype=np.uint8).reshape(4, 4)
                for w in words]
        dictionary = cv2.aruco.Dictionary(np.concatenate([
            cv2.aruco.Dictionary_getByteListFromBits(b) for b in bits]), 4, 0)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementWinSize = 3
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        calibration = cv2.FileStorage(str(calibration_path), cv2.FILE_STORAGE_READ)
        try:
            if not calibration.isOpened():
                raise ValueError("Cannot open camera calibration")
            self.size = tuple(int(calibration.getNode(k).real())
                              for k in ("image_width", "image_height"))
            self.matrix = calibration.getNode("camera_matrix").mat()
            self.distortion = calibration.getNode("distortion_coefficients").mat()
        finally:
            calibration.release()
        if (self.size != (640, 480) or self.matrix is None or self.distortion is None
                or self.matrix.shape != (3, 3)
                or not np.isfinite(self.matrix).all()
                or not np.isfinite(self.distortion).all()):
            raise ValueError("Invalid or unsupported stock camera calibration")
        # Installed stock MarkerProcess uses a 30mm side for home marker 2.
        self.object_points = np.array([[-15, 15, 0], [15, 15, 0],
                                       [15, -15, 0], [-15, -15, 0]], dtype=np.float32)

    def detect(self, frame):
        """Return the visible dock's camera-relative pose, or None.

        A visual observation alone does NOT authorize a docking manoeuvre or
        an edge-sensor exemption. Final approach still needs physical testing.
        """
        if frame is None or frame.shape[:2] != (960, 1280):
            raise ValueError("Expected full-field 1280x960 camera image")
        scaled = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        corners, ids, _ = self.detector.detectMarkers(scaled)
        if ids is None:
            return None
        homes = [corner.reshape(4, 2) for corner, ident in zip(corners, ids.flatten())
                 if ident == 2]
        if len(homes) != 1:
            return None
        points = homes[0]
        if min(np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)) < 12:
            return None
        _, rotations, translations, _ = cv2.solvePnPGeneric(
            self.object_points, points, self.matrix, self.distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        candidates = []
        for rotation, translation in zip(rotations, translations):
            # IPPE provides two analytic starting poses. Refine each against
            # the actual corners before comparing them; on Spark's frames
            # this often converges to one pose and halves reprojection error.
            rotation, translation = cv2.solvePnPRefineLM(
                self.object_points, points, self.matrix, self.distortion,
                rotation.copy(), translation.copy())
            x, _, z = translation.reshape(3)
            if not np.isfinite(translation).all() or not 80 <= z <= 1500:
                continue
            projected, _ = cv2.projectPoints(self.object_points, rotation, translation,
                                              self.matrix, self.distortion)
            error = float(np.sqrt(np.mean(np.sum(
                (projected.reshape(4, 2) - points) ** 2, axis=1))))
            if error <= 2:
                normal = cv2.Rodrigues(rotation)[0][:, 2]
                yaw = math.degrees(math.atan2(normal[0], -normal[2]))
                candidates.append((error, float(x), float(z), yaw))
        if not candidates:
            return None
        best = min(candidates)
        # A small planar marker can have two similarly plausible poses.
        # Keep both angles so docking cannot silently choose the convenient one.
        yaws = [c[3] for c in candidates if c[0] <= min(best[0]+.3, max(best[0]+.05, best[0]*2))]
        return DockObservation(points.tolist(), best[1], best[2], best[0], yaws)


def capture():
    """Capture without initializing drives, arms, sound or any other SDK."""
    import doly_camera
    cam = doly_camera.PiCamera()
    cam.options.video_width, cam.options.video_height = 1280, 960
    cam.options.framerate, cam.options.verbose = 15, False
    try:
        if not cam.start_video():
            raise RuntimeError("Camera did not start")
        frame = None
        for _ in range(20):  # let exposure settle, and use the latest frame
            frame = cam.get_video_frame(1500)
            if frame is None:
                raise RuntimeError("Camera frame timed out")
        return frame
    finally:
        cam.stop_video()


def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Analyze a saved 1280x960 image without hardware")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dictionary", default="/.doly/data/doly.dict")
    parser.add_argument("--calibration", default="/.doly/data/calibration_D8N12K.yml")
    args = parser.parse_args()
    detector = DockDetector(args.dictionary, args.calibration)
    if args.worker:
        # The brain owns every actuator/sensor; this process owns ONLY the
        # camera. Do not construct Body or initialize other SDK modules here.
        from .person_vision import PersonCamera
        with PersonCamera(detector) as camera:
            for _ in range(20):
                if camera.cam.get_video_frame(1500) is None:
                    raise RuntimeError("Dock camera warmup timed out")
            for command in sys.stdin:
                if command.strip() == "close":
                    break
                if command.strip() == "observe":
                    result = camera.observe()
                    print("DOCK_OBSERVATION " + json.dumps(asdict(result) if result else None), flush=True)
        return
    if not args.image:
        import subprocess
        status = subprocess.run(["systemctl", "is-active", "spark-brain", "doly"],
                                capture_output=True, text=True, timeout=5)
        states = status.stdout.splitlines()
        if len(states) != 2 or any(s not in ("inactive", "failed") for s in states):
            parser.error("Stop spark-brain and doly before a live camera diagnostic")
    frame = cv2.imread(args.image) if args.image else capture()
    result = detector.detect(frame)
    print(json.dumps(asdict(result) if result else {"dock_visible": False}))


if __name__ == "__main__":
    main()
