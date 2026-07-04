import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

def interpolate_ball_positions(
    positions: list[tuple[float, float] | None],
    max_gap: int = 15
) -> list[tuple[float, float] | None]:
    """
    Linearly interpolates missing ball positions (None values) within a shot.
    Gaps of size greater than `max_gap` are left as None.
    """
    n = len(positions)
    result = list(positions)
    
    i = 0
    while i < n:
        if result[i] is not None:
            i += 1
            continue
            
        j = i
        while j < n and result[j] is None:
            j += 1
            
        gap_size = j - i
        start_idx = i - 1
        end_idx = j
        
        if start_idx >= 0 and end_idx < n and gap_size <= max_gap:
            p_start = result[start_idx]
            p_end = result[end_idx]
            
            for k in range(gap_size):
                t = (k + 1) / (gap_size + 1)
                x = p_start[0] + t * (p_end[0] - p_start[0])
                y = p_start[1] + t * (p_end[1] - p_start[1])
                result[start_idx + 1 + k] = (x, y)
                
        i = j
        
    return result

class BallKalmanTracker:
    def __init__(self, init_x: float, init_y: float, dt: float = 1.0):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([init_x, init_y, 0.0, 0.0])
        self.kf.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.kf.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        self.kf.R = np.eye(2) * 9.0
        q = Q_discrete_white_noise(dim=2, dt=dt, var=4.0)
        self.kf.Q = np.zeros((4, 4))
        self.kf.Q[0:2, 0:2] = q
        self.kf.Q[2:4, 2:4] = q
        self.kf.P *= 10.0
        self.consecutive_misses = 0

    def predict(self) -> tuple[float, float]:
        self.kf.predict()
        self.consecutive_misses += 1
        return float(self.kf.x[0]), float(self.kf.x[1])

    def update(self, x: float, y: float) -> tuple[float, float]:
        self.kf.update(np.array([x, y]))
        self.consecutive_misses = 0
        return float(self.kf.x[0]), float(self.kf.x[1])

def filter_ball_trajectory_kalman(
    positions: list[tuple[float, float] | None],
    max_prediction: int = 10
) -> list[tuple[float, float] | None]:
    n = len(positions)
    filtered_positions = [None] * n
    tracker = None
    
    for i in range(n):
        pos = positions[i]
        
        if pos is not None:
            if tracker is None:
                tracker = BallKalmanTracker(pos[0], pos[1])
                filtered_positions[i] = pos
            else:
                tracker.predict()
                cx, cy = tracker.update(pos[0], pos[1])
                filtered_positions[i] = (cx, cy)
        else:
            if tracker is not None:
                if tracker.consecutive_misses < max_prediction:
                    cx, cy = tracker.predict()
                    filtered_positions[i] = (cx, cy)
                else:
                    tracker = None
                    filtered_positions[i] = None
                    
    return filtered_positions

import cv2

def calculate_ball_speeds(
    positions: list[tuple[float, float] | None],
    fps: float,
    scale_factor: float = 0.05
) -> list[float]:
    """
    Computes smoothed physical ball speed in km/h based on frame displacement.
    """
    n = len(positions)
    raw_speeds = [0.0] * n
    for i in range(1, n):
        p_prev = positions[i-1]
        p_curr = positions[i]
        if p_prev is not None and p_curr is not None:
            dx = p_curr[0] - p_prev[0]
            dy = p_curr[1] - p_prev[1]
            dist = np.sqrt(dx*dx + dy*dy)
            # Physical velocity = displacement (px) * fps * scale (m/px) * 3.6 (to km/h)
            raw_speeds[i] = dist * fps * scale_factor * 3.6
        else:
            raw_speeds[i] = raw_speeds[i-1]
            
    # Apply moving average filter to smooth the speeds
    smoothed = [0.0] * n
    window = 5
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        smoothed[i] = float(np.mean(raw_speeds[start:end]))
    return smoothed

def get_perspective_transformer(
    img_w: int,
    img_h: int,
    radar_w: int,
    radar_h: int
) -> np.ndarray:
    """
    Generates a homography matrix H mapping broadcast perspective to top-down 2D radar.
    """
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
    return H

def map_point_homography(H: np.ndarray, px: float, py: float) -> tuple[int, int]:
    """
    Projects screen coordinate (px, py) to top-down coordinates (rx, ry) via H matrix.
    """
    point = np.array([[[px, py]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(point, H)
    rx = int(mapped[0][0][0])
    ry = int(mapped[0][0][1])
    return rx, ry


class CameraMovementEstimator:
    """
    Estimates broadcast camera motion between successive frames using Lucas-Kanade optical flow.
    Tracks sparse background keypoints (excluding players/referee/ball area where possible).
    """
    def __init__(self):
        self.prev_gray = None
        self.prev_pts = None
        # Lucas-Kanade params
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.min_features = 30

    def estimate(self, frame: np.ndarray) -> tuple[float, float]:
        """
        Estimates the (dx, dy) translation of the camera from the last frame.
        Returns: (dx, dy) in pixels
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dx, dy = 0.0, 0.0

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
            return 0.0, 0.0

        if self.prev_pts is not None and len(self.prev_pts) > 0:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_pts, None, **self.lk_params
            )

            if next_pts is not None:
                good_prev = self.prev_pts[status == 1]
                good_next = next_pts[status == 1]

                if len(good_prev) > 0:
                    diff = good_next - good_prev
                    dx = float(np.mean(diff[:, 0]))
                    dy = float(np.mean(diff[:, 1]))

                self.prev_pts = good_next.reshape(-1, 1, 2)

        # Re-initialize keypoints if feature count drops below threshold
        if self.prev_pts is None or len(self.prev_pts) < self.min_features:
            self.prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.3, minDistance=7)

        self.prev_gray = gray
        return dx, dy


def estimate_player_performance(
    radar_pos_prev: tuple[int, int] | None,
    radar_pos_curr: tuple[int, int],
    fps: float,
) -> tuple[float, float]:
    """
    Computes player running speed (km/h) and distance traveled (meters) between frames.
    Conversion constants (mapping 200x130 radar pixels to 105m x 68m pitch):
      - 1 horizontal pixel = 105 / 200 = 0.525 meters
      - 1 vertical pixel = 68 / 130 = 0.523 meters
    """
    if radar_pos_prev is None:
        return 0.0, 0.0

    rx_prev, ry_prev = radar_pos_prev
    rx_curr, ry_curr = radar_pos_curr

    dx_m = (rx_curr - rx_prev) * 0.525
    dy_m = (ry_curr - ry_prev) * 0.523
    dist_m = np.sqrt(dx_m * dx_m + dy_m * dy_m)

    # Filter out sub-pixel coordinates tracker noise
    if dist_m < 0.06:
        return 0.0, 0.0

    # speed (m/s) = distance (m) * fps (1/s)
    speed_ms = dist_m * fps
    speed_kmh = speed_ms * 3.6

    # Clamp human running speed limit (max sprint ~38 km/h)
    if speed_kmh > 38.0:
        speed_kmh = 38.0

    return speed_kmh, dist_m

