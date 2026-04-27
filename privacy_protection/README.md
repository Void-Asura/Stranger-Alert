# Real-Time Laptop Privacy Protection System

Continuously watches your webcam and fires a system notification the moment
*any* person other than the authorized user (you) appears in the frame —
even if they are far away or not looking at the screen.

## Features

- **Face detection** via MediaPipe (fast on CPU).
- **Face recognition** via DeepFace embeddings (Facenet by default).
- **Enrollment mode** captures 8 snapshots of you and stores embeddings on disk.
- **Stability check** — an unknown face must persist for ≥ 1 second before alerting.
- **State-based alerts** — notification fires only on `SAFE → ALERT` transition.
- **Cooldown** — minimum 5 s between notifications.
- **Cross-platform notifications** via `plyer` (Windows / macOS / Linux).
- **Debug mode** with green / red bounding boxes; or fully silent background mode.
- **Performance** — frame downscaling (0.5×) + frame skipping (every 2nd frame).

## Project layout

```
privacy_protection/
├── main.py                       # entry point (enrollment + runtime loop)
├── config.py                     # all tunable parameters
├── detection_module.py           # MediaPipe face detector
├── face_recognition_module.py    # DeepFace embeddings + matching
├── notification_module.py        # plyer notifications + beep
├── utils.py                      # distance metrics, drawing helpers
├── requirements.txt
└── data/                         # created on first run
    ├── authorized_embeddings.pkl
    └── enrollment_images/
```

## Setup

> Requires **Python 3.9 – 3.11** (DeepFace + MediaPipe wheels).

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r privacy_protection/requirements.txt
```

Linux only: install a notification daemon if you don't have one (e.g.
`sudo apt install libnotify-bin`).

## Usage

### 1. Enrollment (first run only)

Just run the program. If no embeddings exist yet it will automatically
enter enrollment mode and walk you through 8 captures:

```bash
python -m privacy_protection.main
```

Look at the camera and slowly vary your pose / expression. The script
writes:

- `privacy_protection/data/authorized_embeddings.pkl` — your embeddings
- `privacy_protection/data/enrollment_images/sample_*.jpg` — the snapshots

To **re-enroll** (e.g. new haircut, new glasses), delete the data folder
and run again:

```bash
rm -rf privacy_protection/data
python -m privacy_protection.main
```

### 2. Run the protection system

After enrollment is done, simply run again:

```bash
python -m privacy_protection.main
```

- With `DEBUG = True` (default) a webcam window opens. Boxes are
  **green / "YOU"** for you and **red / "UNKNOWN"** for anyone else.
  Press **q** in the window to quit.
- Set `DEBUG = False` in `config.py` to run silently in the background.
  No window opens; only system notifications appear when an unknown
  person is detected.

## Configuration

All knobs live in `privacy_protection/config.py`:

| Parameter                 | Default | Meaning                                                  |
| ------------------------- | ------- | -------------------------------------------------------- |
| `DEBUG`                   | `True`  | Show webcam window with boxes vs. silent background      |
| `camera_index`            | `0`     | Which webcam to use                                      |
| `detection_confidence`    | `0.7`   | Min confidence to keep a face                            |
| `recognition_threshold`   | `0.6`   | Max distance to still count as "YOU" (lower = stricter)  |
| `stability_seconds`       | `1.0`   | Unknown face must persist this long before alerting      |
| `alert_cooldown`          | `5.0`   | Min seconds between notifications                        |
| `frame_scale`             | `0.5`   | Downscale factor for processing                          |
| `frame_skip`              | `2`     | Process every Nth frame                                  |
| `enrollment_samples`      | `8`     | Snapshots captured during enrollment                     |
| `play_beep`               | `True`  | Short beep alongside the notification                    |

## Behaviour summary

| Situation                                          | Result    |
| -------------------------------------------------- | --------- |
| No faces in frame                                  | `SAFE`    |
| Only the authorized user                           | `SAFE`    |
| Unknown face for < 1 s (flicker)                   | `SAFE`    |
| Unknown face for ≥ 1 s                             | `ALERT` + notification |
| Multiple unknown faces                             | One alert (state-based) |
| State stays `ALERT` while unknown person remains   | No spam (cooldown + state machine) |
| Unknown person leaves                              | Returns to `SAFE`, ready to alert again after cooldown |

## Troubleshooting

- **"Could not open camera index 0"** — change `camera_index` in `config.py`,
  or close other apps using the webcam.
- **First run is slow** — DeepFace downloads model weights (~90 MB) on first
  use. Subsequent runs reuse the cached weights.
- **Linux: no popup appears** — install a notification daemon (`libnotify-bin`)
  and ensure a desktop session is running.
- **Recognition is too strict / too loose** — tune `recognition_threshold`
  in `config.py`. Lower = stricter (fewer false "YOU"), higher = looser.
- **Performance** — increase `frame_skip` to `3` or lower `frame_scale` to
  `0.4` on slower machines.
