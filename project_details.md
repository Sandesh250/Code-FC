# GameCast — Technical Project Details

This document covers the deep technical architecture, algorithmic decisions, and math behind the GameCast highlight gamification pipeline.

---

## 1. Algorithmic Pipeline Architecture

```
                       [ Broadcast Highlight Video ]
                                     │
                        ( Phase 0.5: Shot Detection )
                                     │
                        ┌────────────┴────────────┐
                        ▼                         ▼
                 [ Replay/Closeup ]          [ Gameplay ]
                  ( Pass-through )          ( Processing )
                        │                         │
                        │           ( Phase 1 & 2: Object Detection )
                        │               YOLOv8 + ByteTrack Tracker
                        │                         │
                        │           ( Phase 3: Player Identification )
                        │           - Cosine similarity matching
                        │           - Dynamic 512-dim ArcFace embeds
                        │           - Torso patch color clustering
                        │           - Deterministic roster mapper
                        │                         │
                        │             ( Phase 4: HUD Drawing )
                        │           - Top-down Radar Projection
                        │           - Possession Tracker & Gauges
                        │           - Penalty Zone Pulsing Alerts
                        │                         │
                        └────────────┬────────────┘
                                     ▼
                      ( Phase 5: FFmpeg Audio Stitch )
                                     │
                        [ Final Gamified Clip ]
```

---

## 2. Dynamic Player Roster Matching

Due to broadcast video resolution limitations, direct optical characters and facial frames can often be too small or blurred. To solve this, GameCast uses a **hybrid name resolution cascade**:

1. **Face Recognition Override**: Cropped player head bounding boxes (top 38% of the box) are passed to an ArcFace (`buffalo_l`) face model. Cosine similarity is computed against a match-specific gallery. A match with similarity $> 0.55$ locks the track ID to that player name.
2. **Jersey OCR Fallback**: Crop upper torso regions, apply CLAHE local contrast normalization and Otsu thresholding, and run `easyocr` on the binary single-channel digit. Confident readings ($> 0.55$) are mapped to the squad database.
3. **Deterministic Roster Heuristic**: If tracking starts and has no face or OCR match yet, the pipeline dynamically assigns names from the team's roster to the active track IDs sequentially. Spain players are given Spain player names, and Austria players get Austrian player names. The goalkeeper is automatically resolved by finding the track closest to the goal line.

---

## 3. Homography Pitch Projection (Radar Map)

To render the 2D Top-down Tactical Radar, screen coordinates $(x_s, y_s)$ are mapped to 2D pitch space $(x_p, y_p)$ using a planar homography matrix $H$:

$$
\begin{bmatrix}
x'_p \\
y'_p \\
w
\end{bmatrix}
= H
\begin{bmatrix}
x_s \\
y_s \\
1
\end{bmatrix}
$$

Where the projected coordinates are:

$$
x_p = \frac{x'_p}{w}, \quad y_p = \frac{y'_p}{w}
$$

The homography matrix $H$ is computed by aligning 4 source points on the broadcast perspective field with 4 corresponding target points on the flat 2D radar diagram:

```python
src = np.array([
    [img_w * 0.15, img_h * 0.35],
    [img_w * 0.85, img_h * 0.35],
    [img_w * 0.02, img_h * 0.95],
    [img_w * 0.98, img_h * 0.95]
], dtype=np.float32)

dst = np.array([
    [0, 0],
    [radar_w, 0],
    [0, radar_h],
    [radar_w, radar_h]
], dtype=np.float32)

H = cv2.getPerspectiveTransform(src, dst)
```

---

## 4. Proximity-Based Possession Tracking

Possession is determined frame-by-frame by identifying which player is closest to the ball. If the minimum Euclidean distance between the ball center and a player's contact point (bottom center of the box) is less than $35$ pixels, that player is marked in possession.

The possession ratio is accumulated frame-by-frame:

$$\text{Possession \% (Team A)} = \frac{\text{Frames with Team A possession}}{\text{Total Possession Frames}} \times 100$$

This is displayed dynamically at the top center of the screen as a colored team bar.

---

## 5. Performance Optimizations
- **Single Capture Handle**: The pipeline shares a single open `cv2.VideoCapture` pointer across relevance checking, tracking, and rendering modules. This avoids resource exhaustion and file access locks on Windows.
- **Dynamic Face Gallery Cache**: Instead of extracting face embeddings for all 41,510 images (taking 8+ hours on CPU), the project builds a cache of embeddings *on the fly* for only the 2 active teams selected in the Match Context (e.g. Spain and Austria). This runs in under 15 seconds!
- **Pre-Loaded YOLOv8 model**: YOLO weights are loaded in Flask's main thread on startup, preventing PyTorch CUDA background thread initialization deadlocks.
