"""
Generic helpers shared across modules: distance metrics, drawing,
filesystem helpers, and small numeric utilities.
"""

import os
from typing import Iterable, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    """Create a directory (and parents) if it does not yet exist."""
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2]. 0 means identical direction."""
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(1.0 - np.dot(a, b) / denom)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Plain L2 distance between two embeddings."""
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    return float(np.linalg.norm(a - b))


def best_distance(
    embedding: np.ndarray,
    references: Iterable[np.ndarray],
    metric: str = "cosine",
) -> float:
    """Return the minimum distance between `embedding` and any reference."""
    best = float("inf")
    fn = cosine_distance if metric == "cosine" else euclidean_distance
    for ref in references:
        d = fn(embedding, ref)
        if d < best:
            best = d
    return best


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def clamp_box(
    x: int, y: int, w: int, h: int, frame_w: int, frame_h: int
) -> Tuple[int, int, int, int]:
    """Clip a bounding box so it lies fully inside the frame."""
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return x, y, w, h


def scale_box(
    box: Tuple[int, int, int, int], scale: float
) -> Tuple[int, int, int, int]:
    """Scale a (x, y, w, h) box by `scale` (e.g. to map back to full size)."""
    x, y, w, h = box
    inv = 1.0 / scale
    return int(x * inv), int(y * inv), int(w * inv), int(h * inv)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
# BGR colours (OpenCV convention)
COLOR_AUTHORIZED = (0, 200, 0)     # green
COLOR_UNKNOWN = (0, 0, 230)        # red
COLOR_TEXT = (255, 255, 255)       # white


def draw_label_box(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    is_authorized: bool,
) -> None:
    """Draw a coloured rectangle with a filled label band on `frame`."""
    x, y, w, h = box
    color = COLOR_AUTHORIZED if is_authorized else COLOR_UNKNOWN

    # Main bounding box.
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Label background (filled rectangle above the box, or inside if at top).
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    label_y_top = y - th - baseline - 6
    if label_y_top < 0:
        # Not enough room above the box -> draw inside it at the top.
        label_y_top = y
        text_y = y + th + 4
    else:
        text_y = y - 6

    cv2.rectangle(
        frame,
        (x, label_y_top),
        (x + tw + 8, label_y_top + th + baseline + 6),
        color,
        thickness=cv2.FILLED,
    )
    cv2.putText(
        frame,
        label,
        (x + 4, text_y),
        font,
        font_scale,
        COLOR_TEXT,
        thickness,
        cv2.LINE_AA,
    )


def draw_status_banner(frame: np.ndarray, state: str, fps: float) -> None:
    """Draw a small status banner in the top-left corner."""
    color = COLOR_AUTHORIZED if state == "SAFE" else COLOR_UNKNOWN
    text = f"{state}  |  {fps:5.1f} FPS"
    cv2.rectangle(frame, (0, 0), (260, 32), color, thickness=cv2.FILLED)
    cv2.putText(
        frame,
        text,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )
