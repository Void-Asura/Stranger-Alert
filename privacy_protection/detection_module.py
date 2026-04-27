"""
Face detection wrapper around MediaPipe Face Detection.

Returns a list of bounding boxes (x, y, w, h) on the *resized* frame
that callers can feed to the recognition module.
"""

from typing import List, Tuple

import cv2
import numpy as np
import mediapipe as mp

from . import config
from .utils import clamp_box


Box = Tuple[int, int, int, int]  # (x, y, w, h)


class FaceDetector:
    """Lightweight wrapper around mp.solutions.face_detection."""

    def __init__(
        self,
        min_confidence: float = config.detection_confidence,
        model_selection: int = config.detection_model_selection,
    ) -> None:
        self._mp_fd = mp.solutions.face_detection
        self._detector = self._mp_fd.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_confidence,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray) -> List[Box]:
        """Detect faces in a BGR frame and return bounding boxes."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]
        # MediaPipe expects RGB.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._detector.process(rgb)

        boxes: List[Box] = []
        if not results.detections:
            return boxes

        for det in results.detections:
            rel = det.location_data.relative_bounding_box
            x = int(rel.xmin * w)
            y = int(rel.ymin * h)
            bw = int(rel.width * w)
            bh = int(rel.height * h)

            # Filter out tiny / invalid boxes early.
            if bw < config.min_face_size or bh < config.min_face_size:
                continue

            boxes.append(clamp_box(x, y, bw, bh, w, h))

        return boxes

    def close(self) -> None:
        """Release the underlying MediaPipe graph."""
        try:
            self._detector.close()
        except Exception:
            pass
