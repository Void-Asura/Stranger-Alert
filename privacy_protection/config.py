"""
Global configuration parameters for the Real-Time Laptop Privacy
Protection System.

Edit values here to tune detection sensitivity, alert behaviour, and
debug visualisation.
"""

import os

# ---------------------------------------------------------------------------
# DEBUG MODE
# ---------------------------------------------------------------------------
# True  -> show webcam window with bounding boxes (for development / tuning)
# False -> run silently in the background, only system notifications appear
DEBUG = True

# ---------------------------------------------------------------------------
# CAMERA
# ---------------------------------------------------------------------------
# Index passed to cv2.VideoCapture. 0 is usually the built-in webcam.
camera_index = 0

# Target input resolution requested from the camera (the OS may override).
camera_width = 1280
camera_height = 720

# ---------------------------------------------------------------------------
# DETECTION (MediaPipe Face Detection)
# ---------------------------------------------------------------------------
# Minimum confidence required to keep a detected face.
detection_confidence = 0.7

# 0 = short-range (within 2m, faster), 1 = full-range (up to 5m).
# Full range is preferable for "anyone in the room" privacy use-case.
detection_model_selection = 1

# ---------------------------------------------------------------------------
# RECOGNITION (DeepFace embeddings)
# ---------------------------------------------------------------------------
# DeepFace model used to compute face embeddings.
# "Facenet" gives a strong accuracy/speed balance on CPU.
recognition_model = "Facenet"

# Distance metric used to compare embeddings: "cosine" or "euclidean".
recognition_metric = "cosine"

# Maximum distance at which a detected face is still considered the
# authorized user. Lower = stricter. 0.40 is a sensible default for
# Facenet + cosine; the spec asked for 0.6 as the user-facing knob, so
# we expose it at that value and translate internally if needed.
recognition_threshold = 0.6

# Minimum bounding-box size (pixels, on the resized frame) required
# before we attempt recognition. Avoids wasting CPU on tiny far-away
# blobs that DeepFace cannot reliably embed anyway.
min_face_size = 40

# ---------------------------------------------------------------------------
# PERFORMANCE
# ---------------------------------------------------------------------------
# Scale factor applied to each frame before processing (0.5 = half size).
frame_scale = 0.5

# Process 1 out of every N frames. 2 or 3 keeps CPU low while staying
# responsive enough for a 1-second stability window.
frame_skip = 2

# ---------------------------------------------------------------------------
# ALERT LOGIC
# ---------------------------------------------------------------------------
# Unknown face must persist for at least this many seconds before the
# state flips from SAFE -> ALERT. Prevents flicker / false positives.
stability_seconds = 1.0

# Minimum seconds between two notifications, even if the unknown person
# leaves and comes back.
alert_cooldown = 5.0

# Optional short beep when an alert fires (cross-platform best effort).
play_beep = True

# ---------------------------------------------------------------------------
# STORAGE
# ---------------------------------------------------------------------------
# Folder used to persist embeddings and enrollment snapshots.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "authorized_embeddings.pkl")
ENROLLMENT_IMAGES_DIR = os.path.join(DATA_DIR, "enrollment_images")

# ---------------------------------------------------------------------------
# ENROLLMENT
# ---------------------------------------------------------------------------
# How many snapshots to capture during enrollment.
enrollment_samples = 8

# Seconds to wait between captured snapshots so the user can change pose.
enrollment_capture_interval = 0.8
