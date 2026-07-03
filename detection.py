import cv2
from ultralytics import YOLO
from ultralytics.trackers.basetrack import BaseTrack
import numpy as np

def reset_trackers(model: YOLO):
    """
    Resets the YOLO tracker state and BaseTrack IDs so that track IDs start from 1
    and do not carry over from previous video segments.
    """
    print("[Detection] Resetting tracking state for the new shot...")
    try:
        BaseTrack.reset_id()
    except Exception as e:
        print(f"[Detection] Note: could not reset BaseTrack IDs: {e}")
        
    if hasattr(model, 'predictor') and model.predictor is not None:
        if hasattr(model.predictor, 'trackers'):
            try:
                delattr(model.predictor, 'trackers')
            except Exception as e:
                print(f"[Detection] Note: could not delete trackers attribute: {e}")
                model.predictor.trackers = []

def get_model_classes(model: YOLO) -> tuple[list[int], list[int]]:
    """
    Examines model.names to find matching classes for players and the ball.
    Falls back to COCO defaults (0: person, 32: sports ball) if not found.
    """
    player_ids = []
    ball_ids = []
    
    if hasattr(model, 'names') and model.names:
        for cls_id, cls_name in model.names.items():
            cls_name_lower = cls_name.lower()
            if any(p in cls_name_lower for p in ["person", "player", "goalkeeper", "referee"]):
                player_ids.append(cls_id)
            elif any(b in cls_name_lower for b in ["ball", "sports ball", "football"]):
                ball_ids.append(cls_id)
                
    if not player_ids:
        player_ids = [0]  # COCO person
    if not ball_ids:
        ball_ids = [32]   # COCO sports ball
        
    return player_ids, ball_ids

def simple_kmeans_2d(data: np.ndarray, k: int = 2, max_iters: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    Lightweight, dependency-free K-Means clustering implementation.
    Clusters data of shape (N, D) into k clusters.
    Returns:
        centroids: np.ndarray of shape (k, D)
        labels: np.ndarray of shape (N,) containing cluster indices (0 to k-1)
    """
    n_samples, n_features = data.shape
    if n_samples <= k:
        dummy_centroids = np.zeros((k, n_features))
        if n_samples > 0:
            dummy_centroids[:n_samples] = data
        dummy_labels = np.zeros(n_samples, dtype=int)
        for i in range(min(n_samples, k)):
            dummy_labels[i] = i
        return dummy_centroids, dummy_labels
        
    # Deterministic initialization: pick k furthest points
    np.random.seed(42)
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = data[indices].copy().astype(float)
    
    labels = np.zeros(n_samples, dtype=int)
    
    for _ in range(max_iters):
        # Distances to centroids
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        new_labels = np.argmin(distances, axis=1)
        
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        
        # Recompute centroids
        for j in range(k):
            mask = (labels == j)
            if np.any(mask):
                centroids[j] = data[mask].mean(axis=0)
                
    return centroids, labels

def track_shot(
    cap: cv2.VideoCapture,
    start_frame: int,
    end_frame: int,
    model: YOLO,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> tuple[list[dict], dict]:
    """
    Tracks players and the ball throughout a single shot.
    Performs jersey color clustering (K-Means) to classify players into Team 0 or Team 1.
    
    Returns:
        tuple: (shot_results, team_colors)
               - shot_results: list[dict] of detections per frame
               - team_colors: dict mapping team_id (0, 1) to BGR color tuple
    """
        
    reset_trackers(model)
    
    player_classes, ball_classes = get_model_classes(model)
    all_tracking_classes = player_classes + ball_classes
    
    shot_results = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Keep track of mean jersey colors per player track ID
    jersey_colors_by_track = {}
    
    prev_ball_pos = None

    for f_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            print(f"[Detection] Warning: frame read failed at index {f_idx}")
            break

        # 1. Track players with standard confidence
        player_results = model.track(
            source=frame,
            persist=True,
            classes=player_classes,
            conf=conf_threshold,
            iou=iou_threshold,
            tracker="bytetrack.yaml",
            verbose=False
        )

        # 2. Detect ball with low confidence to prevent dropouts
        ball_results = model(
            source=frame,
            classes=ball_classes,
            conf=0.10,
            verbose=False
        )

        frame_players = []
        frame_balls = []

        # Parse player detections
        for result in player_results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls.item())
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf.item())
                track_id = int(box.id.item()) if box.id is not None else -1

                frame_players.append({
                    "track_id": track_id,
                    "bbox": xyxy,
                    "conf": conf,
                    "team_id": 0  # default
                })

                # Jersey color extraction for clustering
                if track_id >= 0:
                    x1, y1, x2, y2 = map(int, xyxy)
                    h = y2 - y1
                    w = x2 - x1

                    # Define jersey crop region (upper middle torso)
                    y_start = max(0, y1 + int(h * 0.15))
                    y_end = min(frame.shape[0], y1 + int(h * 0.42))
                    x_start = max(0, x1 + int(w * 0.20))
                    x_end = min(frame.shape[1], x2 - int(w * 0.20))

                    if y_end > y_start and x_end > x_start:
                        patch = frame[y_start:y_end, x_start:x_end]
                        if patch.size > 0:
                            mean_color = patch.mean(axis=(0, 1))  # BGR
                            if track_id not in jersey_colors_by_track:
                                jersey_colors_by_track[track_id] = []
                            jersey_colors_by_track[track_id].append(mean_color)

        # Parse and filter ball detections (best candidate based on size + proximity)
        best_ball = None
        best_ball_score = -1.0

        for result in ball_results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf.item())
                w_box = xyxy[2] - xyxy[0]
                h_box = xyxy[3] - xyxy[1]

                # Filter out too large boxes (a football is small)
                if w_box > 45 or h_box > 45 or w_box < 4 or h_box < 4:
                    continue

                cx = (xyxy[0] + xyxy[2]) / 2.0
                cy = (xyxy[1] + xyxy[3]) / 2.0

                # Score candidate based on conf + distance to last known position
                if prev_ball_pos is not None:
                    dist = np.sqrt((cx - prev_ball_pos[0])**2 + (cy - prev_ball_pos[1])**2)
                    if dist > 200.0:  # Ignore extreme leaps
                        continue
                    dist_score = 1.0 - (dist / 200.0)
                    score = conf * 0.3 + dist_score * 0.7
                else:
                    score = conf

                if score > best_ball_score:
                    best_ball_score = score
                    best_ball = {
                        "track_id": -1,
                        "bbox": xyxy,
                        "conf": conf,
                        "center": (cx, cy)
                    }

        if best_ball is not None:
            frame_balls.append(best_ball)
            prev_ball_pos = best_ball["center"]
        else:
            prev_ball_pos = None

        shot_results.append({
            "frame_idx": f_idx,
            "players": frame_players,
            "balls": frame_balls
        })
    
    # Perform K-Means clustering on player jersey colors
    team_by_track = {}
    team_colors = {0: (255, 100, 0), 1: (0, 100, 255)} # standard default colors
    
    track_ids = []
    color_features = []
    
    for t_id, colors in jersey_colors_by_track.items():
        if len(colors) > 0:
            track_ids.append(t_id)
            color_features.append(np.mean(colors, axis=0))
            
    if len(color_features) > 0:
        data = np.array(color_features)
        centroids, labels = simple_kmeans_2d(data, k=2)
        
        # Populate mapping
        for idx, t_id in enumerate(track_ids):
            team_by_track[t_id] = int(labels[idx])
            
        # Map centroids to team_colors
        # Ensure BGR colors are converted to tuple of ints
        team_colors[0] = tuple(map(int, centroids[0]))
        team_colors[1] = tuple(map(int, centroids[1]))
        
    # Apply team_id to frame players
    for frame_data in shot_results:
        for player in frame_data["players"]:
            t_id = player["track_id"]
            if t_id in team_by_track:
                player["team_id"] = team_by_track[t_id]
                
    return shot_results, team_colors

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video = sys.argv[1]
        model = YOLO("yolov8n.pt")
        print(f"Tracking players and ball on frames 0-50 of {video}...")
        cap = cv2.VideoCapture(video, cv2.CAP_FFMPEG)
        results, team_colors = track_shot(cap, 0, 50, model)
        cap.release()
        print(f"Processed {len(results)} frames. Sample frame 10 data:")
        if len(results) > 10:
            print(results[10])
    else:
        print("Usage: python detection.py <video_path>")
