"""
Real-Time Laptop Privacy Protection System — entry point.

Behaviour:
  * On startup, if no enrollment exists, the program runs an interactive
    enrollment flow to capture the authorized user's face.
  * Then it enters the runtime loop: detects faces, recognises them,
    tracks SAFE/ALERT state, and fires system notifications when an
    unknown face appears for at least `stability_seconds` continuously.

Run:
    python -m privacy_protection.main
or  python privacy_protection/main.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, Tuple

import cv2
import numpy as np

# Allow running both as a module and as a plain script.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from privacy_protection import config
    from privacy_protection.detection_module import FaceDetector
    from privacy_protection.face_recognition_module import FaceRecognizer
    from privacy_protection.notification_module import Notifier
    from privacy_protection.utils import (
        draw_label_box,
        draw_status_banner,
        ensure_dir,
        scale_box,
    )
else:
    from . import config
    from .detection_module import FaceDetector
    from .face_recognition_module import FaceRecognizer
    from .notification_module import Notifier
    from .utils import (
        draw_label_box,
        draw_status_banner,
        ensure_dir,
        scale_box,
    )


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------
def open_camera() -> cv2.VideoCapture:
    """Open the webcam and apply requested resolution."""
    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {config.camera_index}. "
            "Check that the webcam is connected and not in use."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
    return cap


def resize_for_processing(frame: np.ndarray) -> np.ndarray:
    """Downscale a frame for faster detection / recognition."""
    if config.frame_scale == 1.0:
        return frame
    return cv2.resize(
        frame,
        (0, 0),
        fx=config.frame_scale,
        fy=config.frame_scale,
        interpolation=cv2.INTER_LINEAR,
    )


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------
def run_enrollment(
    cap: cv2.VideoCapture,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
) -> None:
    """
    Capture N snapshots of the authorized user and store embeddings.

    Always shows a window during enrollment (regardless of DEBUG) so the
    user can see themselves being captured.
    """
    print("\n=== ENROLLMENT MODE ===")
    print(f"Capturing {config.enrollment_samples} snapshots of the authorized user.")
    print("Look at the camera and slowly vary your pose / expression.")
    print("Press 'q' to abort.\n")

    ensure_dir(config.ENROLLMENT_IMAGES_DIR)
    recognizer.clear_references()

    captured = 0
    next_capture_at = time.time() + 1.0  # 1s warm-up so the user gets ready
    window = "Enrollment - Privacy Protection"

    while captured < config.enrollment_samples:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[enroll] Camera read failed; retrying...")
            time.sleep(0.05)
            continue

        # Mirror so it feels like a selfie view.
        frame = cv2.flip(frame, 1)
        small = resize_for_processing(frame)
        boxes = detector.detect(small)

        # Map boxes back to full-resolution frame for cropping & display.
        full_boxes = [scale_box(b, config.frame_scale) for b in boxes]

        now = time.time()
        status = f"Captured {captured}/{config.enrollment_samples}"
        instruction = "Hold still..." if len(full_boxes) == 1 else (
            "Show ONE face to the camera"
        )

        # Draw guides.
        for b in full_boxes:
            x, y, w, h = b
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(
            frame, status, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, instruction, (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA,
        )

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[enroll] Aborted by user.")
            cv2.destroyWindow(window)
            return

        # Capture only when exactly one face is visible and interval elapsed.
        if len(full_boxes) == 1 and now >= next_capture_at:
            x, y, w, h = full_boxes[0]
            face_crop = frame[y:y + h, x:x + w].copy()
            if recognizer.add_reference_from_face(face_crop):
                captured += 1
                snap_path = os.path.join(
                    config.ENROLLMENT_IMAGES_DIR, f"sample_{captured:02d}.jpg"
                )
                cv2.imwrite(snap_path, face_crop)
                print(f"[enroll] Captured sample {captured}/{config.enrollment_samples}")
                next_capture_at = now + config.enrollment_capture_interval
            else:
                print("[enroll] Embedding failed for this sample, retrying...")

    recognizer.save()
    print(f"[enroll] Saved {recognizer.reference_count()} embeddings to "
          f"{config.EMBEDDINGS_FILE}")
    cv2.destroyWindow(window)


# ---------------------------------------------------------------------------
# Runtime loop
# ---------------------------------------------------------------------------
def runtime_loop(
    cap: cv2.VideoCapture,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    notifier: Notifier,
) -> None:
    """Main processing loop: detect -> recognise -> alert."""
    print("\n=== RUNTIME ===")
    print(f"DEBUG = {config.DEBUG}")
    print("Press 'q' in the debug window to quit (or Ctrl+C in the terminal).\n")

    state = "SAFE"               # current system state
    unknown_seen_since: float | None = None
    frame_idx = 0
    last_results: List[Tuple[Tuple[int, int, int, int], bool]] = []

    # FPS counter.
    fps = 0.0
    fps_window_start = time.time()
    fps_window_frames = 0

    window = "Privacy Protection (DEBUG)"

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            frame_idx += 1
            run_heavy = (frame_idx % max(1, config.frame_skip)) == 0

            if run_heavy:
                small = resize_for_processing(frame)
                boxes_small = detector.detect(small)

                results: List[Tuple[Tuple[int, int, int, int], bool]] = []
                any_unknown = False

                for box in boxes_small:
                    sx, sy, sw, sh = box
                    face_crop = small[sy:sy + sh, sx:sx + sw]
                    is_auth, _dist = recognizer.identify(face_crop)
                    if not is_auth:
                        any_unknown = True
                    full_box = scale_box(box, config.frame_scale)
                    results.append((full_box, is_auth))

                last_results = results

                # ------------------------------------------------------
                # State machine
                # ------------------------------------------------------
                now = time.time()
                if any_unknown:
                    if unknown_seen_since is None:
                        unknown_seen_since = now
                    elif (now - unknown_seen_since) >= config.stability_seconds:
                        if state == "SAFE":
                            state = "ALERT"
                            fired = notifier.alert()
                            if fired:
                                print("[alert] Unknown person detected -> "
                                      "notification sent.")
                else:
                    unknown_seen_since = None
                    if state == "ALERT":
                        state = "SAFE"
                        print("[state] Returned to SAFE.")

            # ----------------------------------------------------------
            # Debug visualisation
            # ----------------------------------------------------------
            if config.DEBUG:
                for full_box, is_auth in last_results:
                    label = "YOU" if is_auth else "UNKNOWN"
                    draw_label_box(frame, full_box, label, is_auth)
                draw_status_banner(frame, state, fps)
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[runtime] Quit requested by user.")
                    break

            # ----------------------------------------------------------
            # FPS tracking
            # ----------------------------------------------------------
            fps_window_frames += 1
            elapsed = time.time() - fps_window_start
            if elapsed >= 1.0:
                fps = fps_window_frames / elapsed
                fps_window_frames = 0
                fps_window_start = time.time()

    except KeyboardInterrupt:
        print("\n[runtime] Interrupted by user (Ctrl+C).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    ensure_dir(config.DATA_DIR)

    print("Initializing detector...")
    detector = FaceDetector()
    print("Initializing recognizer (this may take a moment on first run)...")
    recognizer = FaceRecognizer()
    notifier = Notifier()

    cap = open_camera()
    try:
        if not recognizer.has_references():
            run_enrollment(cap, detector, recognizer)
            if not recognizer.has_references():
                print("Enrollment did not produce any embeddings. Exiting.")
                return 1

        runtime_loop(cap, detector, recognizer, notifier)
    finally:
        cap.release()
        detector.close()
        if config.DEBUG:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
