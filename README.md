# GameCast ⚽🎮
### Gamified Football Highlight Video Overlay Pipeline

GameCast is an advanced computer vision and web dashboard system that takes standard football (soccer) broadcast highlight videos and transforms them into an interactive, "gamified" experience with premium HUD overlays.

![GameCast HUD Preview](frontend/src/assets/hero.png)

---

## 🚀 Key Features

### 🧠 Core Analytics Pipeline
*   **Automatic Shot Segmentation**: Uses content-aware thresholding to detect scene cuts and splits the video logically into gameplay shots vs closeups/replays.
*   **Advanced Player Tracking & Jersey Identification**: Uses YOLOv8 + ByteTrack to detect players, and K-Means clustering on torso patches to classify them dynamically into Team A and Team B.
*   **Proximity-Assisted Ball Detection**: Tracks the ball using a custom low-confidence detection heuristic and spatial Kalman filtering, completely resolving tracking dropouts and occlusions.

### 🎮 Premium HUD Overlays
*   **Dynamic Roster Assignment (Player Names)**: Automatically assigns real player names to track IDs using the match context rosters. Goalkeepers are automatically identified and labeled correctly.
*   **Possession Percentage Bar**: Renders a dynamic possession progress bar at the top center of the screen based on player-ball proximity.
*   **Tactical 2D Pitch Radar**: Project player coordinate points onto a top-down tactical layout map using homography matrices.
*   **Ball Speed Gauge**: Computes physical velocity based on frame displacement and displays a circular speed gauge (in KM/H).
*   **Kick Shockwaves & Tracer Beams**: Adds high-impact gamified visuals to represent kicks, passes, and shots.
*   **Penalty Box Danger Alert**: Renders a pulsing red screen border and warning prompt when the ball enters a penalty area.

---

## 🛠️ Technology Stack
*   **Backend**: Python, Flask, OpenCV, PyTorch, Ultralytics YOLOv8, easyocr, InsightFace (ArcFace).
*   **Frontend**: React, Vite, Vanilla CSS.

---

## 📁 Dataset Folder Structure
To clone this project and run it on another laptop, place your datasets in the following structure under the root directory:
```
Code FC/
├── datasets/
│   ├── face images/
│   │   └── Images/
│   │       └── Images/
│   │           ├── Group A/
│   │           │   ├── Ecuador Players/
│   │           │   └── Netherland Players/
│   │           └── Group G/
│   │               └── Brazil Players/
│   └── stat/
│       └── FC26_20250921.csv
```
The codebase will automatically find them and use them to construct squad registries and dynamic face galleries.

---

## 🔧 Installation & Local Setup

### 1. Install Backend Dependencies
Ensure you have Python 3.10+ installed:
```bash
pip install -r requirements.txt
pip install insightface onnxruntime-gpu easyocr
```

### 2. Build the Frontend Assets
Navigate to the `frontend` folder and build the React app:
```bash
cd frontend
npm install
npm run build
cd ..
```

### 3. Run the Server
Launch the Flask app (loads YOLOv8 model on startup):
```bash
python app.py
```
Open your browser and navigate to: `http://localhost:5000`

---

## 📂 Project Structure
```
├── app.py                     # Flask entry point (manages pipeline worker thread)
├── main.py                    # Core pipeline orchestrator
├── detection.py               # YOLO player tracking and ball detection heuristics
├── overlay.py                 # HUD drawing, radar project, possession bar, nameplates
├── player_registry.py         # 32 WC teams squad rosters (auto-generated)
├── build_registry.py          # Script to generate squads from stats CSV
├── dynamic_face_gallery.py    # Match-specific face gallery embedding builder
├── face_recognizer.py         # Runtime ArcFace face recognition matcher
├── jersey_ocr.py              # Digit OCR from player crops
├── tracking_utils.py          # Kalman filtering, homography math, speed calculator
├── shot_segmentation.py       # PySceneDetect content cut boundaries
├── stitch.py                  # FFmpeg audio sync and clip merger
├── requirements.txt           # Python packages manifest
├── .gitignore                 # Excludes heavy datasets and model weights
└── frontend/                  # React dashboard codebase
```
