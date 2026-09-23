"""Local person detection and short-lived camera access for 'come here'.

NanoDet decoding adapted from OpenCV Zoo (Apache-2.0); see THIRD_PARTY.md.
No face recognition, saved camera frames, or remote image uploads.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Person:
    x: float
    y: float
    width: float
    height: float
    score: float

    @property
    def center(self):
        return self.x + self.width / 2

    def overlaps(self, other):
        intersection = (max(0, min(self.x+self.width, other.x+other.width)-max(self.x, other.x))
                        * max(0, min(self.y+self.height, other.y+other.height)-max(self.y, other.y)))
        union = self.width*self.height + other.width*other.height - intersection
        return intersection / union if union > 0 else 0


class PersonDetector:
    def __init__(self, model_path):
        import cv2
        import numpy as np
        if not Path(model_path).is_file():
            raise RuntimeError("Person detection model is not installed")
        cv2.setNumThreads(2)  # leave CPU time for audio and safety polling
        self.net = cv2.dnn.readNet(str(model_path))
        self.anchors = []
        for stride in (8, 16, 32, 64):
            grid = np.arange(416 // stride) * stride + (stride-1) / 2
            x, y = np.meshgrid(grid, grid)
            self.anchors.append(np.column_stack((x.ravel(), y.ravel())))

    def detect(self, frame):
        import cv2
        import numpy as np
        if frame is None or frame.size == 0:
            raise RuntimeError("Camera returned an empty frame")
        height, width = frame.shape[:2]
        scale = 416 / max(height, width)
        resized = cv2.resize(frame, (round(width*scale), round(height*scale)))
        top, left = (416-resized.shape[0])//2, (416-resized.shape[1])//2
        padded = cv2.copyMakeBorder(resized, top, 416-resized.shape[0]-top,
                                   left, 416-resized.shape[1]-left, cv2.BORDER_CONSTANT)
        normalized = ((padded.astype(np.float32) - [103.53, 116.28, 123.675])
                      / [57.375, 57.12, 58.395]).astype(np.float32)
        self.net.setInput(cv2.dnn.blobFromImage(normalized))
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        boxes, scores = [], []
        for stride, anchors, classes, distances in zip(
                (8, 16, 32, 64), self.anchors, outputs[::2], outputs[1::2]):
            classes = classes.reshape(-1, 80)
            # COCO class 0 is person. Do not relabel a stronger object class.
            keep = (classes.argmax(axis=1) == 0) & (classes[:, 0] >= .5)
            if not keep.any():
                continue
            logits = distances.reshape(-1, 4, 8)[keep]
            probs = np.exp(logits - logits.max(axis=2, keepdims=True))
            distances = (probs / probs.sum(axis=2, keepdims=True)) @ np.arange(8) * stride
            centers = anchors[keep]
            for center, distance, score in zip(centers, distances, classes[keep, 0]):
                x1, y1 = (center-distance[:2]-[left, top]) / scale
                x2, y2 = (center+distance[2:]-[left, top]) / scale
                x1, x2 = np.clip([x1, x2], 0, width)
                y1, y2 = np.clip([y1, y2], 0, height)
                if x2-x1 > 15 and y2-y1 > 30:
                    boxes.append([float(x1), float(y1), float(x2-x1), float(y2-y1)])
                    scores.append(float(score))
        selected = cv2.dnn.NMSBoxes(boxes, scores, .5, .45) if boxes else []
        return [Person(boxes[i][0]/width, boxes[i][1]/height,
                       boxes[i][2]/width, boxes[i][3]/height, scores[i])
                for i in np.asarray(selected).flatten()]


class PersonCamera:
    def __init__(self, detector):
        self.detector = detector
        self.cam = None

    def __enter__(self):
        import doly_camera
        self.cam = doly_camera.PiCamera()
        self.cam.options.video_width, self.cam.options.video_height = 1280, 960
        self.cam.options.framerate, self.cam.options.verbose = 15, False
        if not self.cam.start_video():
            self.cam.stop_video()
            raise RuntimeError("Camera could not start")
        return self

    def __exit__(self, *exc):
        if self.cam:
            self.cam.stop_video()

    def observe(self):
        # Discard frames from the preceding movement; inspect after braking.
        frame = None
        for _ in range(3):
            frame = self.cam.get_video_frame(1500)
            if frame is None:
                raise RuntimeError("Camera frame timed out")
        return self.detector.detect(frame)
