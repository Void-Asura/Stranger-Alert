"""
Face recognition / enrollment using DeepFace embeddings.

- Enrollment: capture N webcam snapshots of the authorized user,
  compute their embeddings, and persist them to disk (pickle).
- Runtime:  load embeddings once, then for each detected face crop
  compute an embedding and compare it to the stored references.
"""

import os
import pickle
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import config
from .utils import (
    best_distance,
    ensure_dir,
)


class FaceRecognizer:
    """Wraps DeepFace embedding generation + reference storage."""

    def __init__(
        self,
        model_name: str = config.recognition_model,
        metric: str = config.recognition_metric,
        threshold: float = config.recognition_threshold,
        embeddings_file: str = config.EMBEDDINGS_FILE,
    ) -> None:
        self.model_name = model_name
        self.metric = metric
        self.threshold = threshold
        self.embeddings_file = embeddings_file
        self._references: List[np.ndarray] = []

        # Lazy import: DeepFace pulls in TensorFlow which is slow to start.
        # We import here (module-level) instead of inside represent() so the
        # weights are downloaded/initialised once at construction time.
        from deepface import DeepFace  # type: ignore
        self._DeepFace = DeepFace

        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def has_references(self) -> bool:
        return len(self._references) > 0

    def load(self) -> None:
        """Load stored authorized embeddings from disk if present."""
        if not os.path.exists(self.embeddings_file):
            self._references = []
            return
        try:
            with open(self.embeddings_file, "rb") as f:
                data = pickle.load(f)
            refs = data.get("embeddings", []) if isinstance(data, dict) else data
            self._references = [np.asarray(r, dtype=np.float32) for r in refs]
        except Exception as exc:
            print(f"[recognition] Failed to load embeddings: {exc}")
            self._references = []

    def save(self) -> None:
        ensure_dir(os.path.dirname(self.embeddings_file))
        payload = {
            "model": self.model_name,
            "metric": self.metric,
            "embeddings": [r.tolist() for r in self._references],
            "saved_at": time.time(),
        }
        with open(self.embeddings_file, "wb") as f:
            pickle.dump(payload, f)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def _embed(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return a single embedding vector for a face crop, or None."""
        if face_bgr is None or face_bgr.size == 0:
            return None
        try:
            # DeepFace expects RGB; we feed BGR converted.
            rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            reps = self._DeepFace.represent(
                img_path=rgb,
                model_name=self.model_name,
                enforce_detection=False,   # we already have a face crop
                detector_backend="skip",
            )
            if not reps:
                return None
            emb = reps[0].get("embedding")
            if emb is None:
                return None
            return np.asarray(emb, dtype=np.float32)
        except Exception as exc:
            # Low-light / blurry / partial faces sometimes raise; skip them.
            print(f"[recognition] embed() failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------
    def add_reference_from_face(self, face_bgr: np.ndarray) -> bool:
        """Compute and store an embedding for one enrollment snapshot."""
        emb = self._embed(face_bgr)
        if emb is None:
            return False
        self._references.append(emb)
        return True

    def reference_count(self) -> int:
        return len(self._references)

    def clear_references(self) -> None:
        self._references = []

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------
    def identify(self, face_bgr: np.ndarray) -> Tuple[bool, float]:
        """
        Decide whether `face_bgr` is the authorized user.

        Returns (is_authorized, distance). distance == inf when the face
        could not be embedded.
        """
        if not self._references:
            return False, float("inf")

        emb = self._embed(face_bgr)
        if emb is None:
            return False, float("inf")

        dist = best_distance(emb, self._references, metric=self.metric)
        return dist <= self.threshold, dist
