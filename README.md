<div align="center">

# 🔐 Stranger Alert

### Real-Time Laptop Privacy Protection Using Face Recognition

[![Python](https://img.shields.io/badge/Python-3.9%20–%203.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![DeepFace](https://img.shields.io/badge/DeepFace-0.0.93-orange)](https://github.com/serengil/deepface)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-green)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)
[![Status](https://img.shields.io/badge/Status-v1.0%20Initial%20Release-yellow)]()

<br/>

> **Stranger Alert** watches your webcam in real time. The moment anyone other than you appears in the frame — even far in the background — you get an instant system notification. Your screen. Your privacy. Your rules.

<br/>

![Demo Banner](https://img.shields.io/badge/🟢%20YOU%20-%20Safe-brightgreen?style=for-the-badge) &nbsp;
![Demo Banner](https://img.shields.io/badge/🔴%20UNKNOWN%20-%20Alert!-red?style=for-the-badge)

</div>

---

## 🎯 What It Does

Stranger Alert runs silently in the background and uses your webcam to:

1. **Enroll you** — captures 8 photos of your face and stores your unique identity locally (no cloud, no uploads)
2. **Watch continuously** — detects every face in the frame in real time
3. **Recognize you** — compares each face to your stored identity using AI embeddings
4. **Alert immediately** — fires a system notification the moment a stranger appears, even if they're far away or not looking at the screen

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 **AI Face Recognition** | DeepFace + Facenet embeddings — the same tech used in commercial systems |
| ⚡ **Real-Time Detection** | MediaPipe face detection, optimized for CPU — runs at 15+ FPS on a laptop |
| 🔔 **System Notifications** | Native OS popup via `plyer` — works even when the app is minimized |
| 🔒 **100% Local / Offline** | Your face data never leaves your machine — stored as a local `.pkl` file |
| 🎛️ **Debug Mode** | Live webcam window with green (YOU) / red (UNKNOWN) bounding boxes |
| 🔇 **Silent Mode** | No window, no terminal output — just a background process and notifications |
| 🕐 **Anti-Spam Logic** | State machine (SAFE → ALERT) with 1s stability check + 5s cooldown between alerts |
| 🎬 **Frame Skipping** | Processes every 2nd frame at 0.5× scale — keeps CPU usage low |
| 💾 **Persistent Identity** | Your enrollment survives restarts — enroll once, run forever |

---

## 🗂️ Project Structure

```
Stranger-Alert/
│
├── privacy_protection/
│   ├── main.py                    # Entry point — enrollment + runtime loop
│   ├── config.py                  # All configurable parameters
│   ├── detection_module.py        # MediaPipe face detector
│   ├── face_recognition_module.py # DeepFace embeddings + identity matching
│   ├── notification_module.py     # Cross-platform system notifications
│   └── utils.py                   # Distance math, drawing helpers
│
├── requirements.txt               # Pinned dependencies
├── .gitignore                     # Excludes face data & model weights
└── README.md
```

> 📁 A `privacy_protection/data/` folder is created on first run to store your embeddings and enrollment snapshots. This folder is **git-ignored** and stays on your machine only.

---

## 🚀 Quick Start

### Prerequisites
- Python **3.9, 3.10, or 3.11** (not 3.12+)
- A working webcam
- Windows, macOS, or Linux

### 1. Clone the repo
```bash
git clone https://github.com/Void-Asura/Stranger-Alert.git
cd Stranger-Alert
```

### 2. Create a virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> ⏳ First install takes a few minutes — DeepFace downloads model weights (~90 MB) on first run.

### 4. Run the app
```bash
python -m privacy_protection.main
```

**First launch** → Enrollment mode opens automatically. Look at the camera, slowly vary your pose. 8 snapshots are captured and your identity is saved locally.

**All future launches** → Live protection starts immediately.

Press **`q`** in the debug window to quit, or **Ctrl+C** in the terminal.

---

## ⚙️ Configuration

All settings live in `privacy_protection/config.py`:

```python
DEBUG = True               # False = silent background mode (no window)
camera_index = 0           # Which webcam to use (0 = built-in)
recognition_threshold = 0.6  # Lower = stricter identity matching
alert_cooldown = 5.0       # Seconds between repeat notifications
stability_seconds = 1.0    # Unknown face must persist this long before alert
frame_scale = 0.5          # Downscale factor for performance
enrollment_samples = 8     # Snapshots captured during enrollment
```

### Running silently in the background
Change `DEBUG = True` → `DEBUG = False` in `config.py`. No window opens — only system notifications appear.

---

## 🔄 How It Works

```
Webcam Frame
     │
     ▼
Frame Resize (0.5×) ──► MediaPipe Face Detection
                               │
                    ┌──────────┼──────────┐
                    │                     │
               Face Crop #1         Face Crop #2 ...
                    │
                    ▼
            DeepFace Embedding
            (Facenet model)
                    │
                    ▼
         Cosine Similarity vs
         Stored Authorized Embeddings
                    │
            ┌───────┴───────┐
           YOU           UNKNOWN
            │                │
          SAFE          Stability Timer
                         (≥ 1 second)
                              │
                         State: ALERT
                              │
                     System Notification
                      (5s cooldown)
```

---

## ⚠️ Limitations (v1.0)

This is the **initial release** — it works well but has known limitations:

| Limitation | Details |
|---|---|
| 🐢 **Slow first launch** | DeepFace initializes TensorFlow on startup — expect 5–15s delay before the camera opens |
| 👤 **Single authorized user** | Currently supports only one enrolled identity. Multi-user enrollment is not yet implemented |
| 😷 **Mask / heavy occlusion** | If your face is heavily covered, recognition may fail and mark you as UNKNOWN |
| 🌑 **Low light struggles** | Detection accuracy drops in very dark environments |
| 💻 **CPU only** | No GPU acceleration in this version — high-end recognition models are slow on CPU |
| 🐍 **Python 3.12+ not supported** | TensorFlow 2.15 does not have wheels for Python 3.12/3.13 yet |
| 🐧 **Linux notifications need daemon** | Requires `libnotify-bin` and a desktop session on Linux |
| 📸 **No re-enrollment UI** | To re-enroll, manually delete `privacy_protection/data/` and restart |

---

## 🛠️ Troubleshooting

**"Could not open camera index 0"**
→ Close Zoom, Teams, or any browser tab using the webcam. Try `camera_index = 1` in config.

**Marking you as UNKNOWN**
→ Lower `recognition_threshold` (e.g. `0.5`) in config for stricter matching, or re-enroll with better lighting.

**Too many false alerts**
→ Raise `recognition_threshold` (e.g. `0.65`) or increase `stability_seconds`.

**Linux: no notification appears**
```bash
sudo apt install libnotify-bin
```

**Dependency conflict on install**
→ Make sure you're on Python 3.9–3.11 and using the pinned `requirements.txt`.

---

## 🗺️ Roadmap

Things planned for future versions:

- [ ] System tray icon with pause / resume / re-enroll controls
- [ ] Multi-user enrollment support
- [ ] GPU acceleration (CUDA / MPS)
- [ ] Webhook / email alert option
- [ ] Auto-lock screen when alert fires
- [ ] Lightweight installer (no Python setup needed)

---

## 📄 License

MIT — do whatever you want with it.

---

<div align="center">

Built with ❤️ · [Report a Bug](https://github.com/Void-Asura/Stranger-Alert/issues) · [Request a Feature](https://github.com/Void-Asura/Stranger-Alert/issues)

</div>
