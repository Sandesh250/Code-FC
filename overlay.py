import cv2
import numpy as np

def draw_alpha_shape(img: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int], alpha: float, thickness: int = -1):
    """
    Draws a circle or ring with alpha transparency on a local ROI to optimize performance
    and avoid copying the entire video frame.
    """
    if alpha <= 0:
        return
    if alpha >= 1.0:
        cv2.circle(img, center, radius, color, thickness)
        return
        
    x, y = int(center[0]), int(center[1])
    r = int(radius)
    h, w = img.shape[:2]
    
    x1 = max(0, x - r - 2)
    y1 = max(0, y - r - 2)
    x2 = min(w, x + r + 2)
    y2 = min(h, y + r + 2)
    
    if x2 <= x1 or y2 <= y1:
        return
        
    roi = img[y1:y2, x1:x2]
    roi_copy = roi.copy()
    
    cv2.circle(roi_copy, (x - x1, y - y1), r, color, thickness)
    cv2.addWeighted(roi_copy, alpha, roi, 1.0 - alpha, 0, dst=roi)

def detect_impacts(positions: list[tuple[float, float] | None], threshold: float = 12.0) -> list[bool]:
    """
    Analyzes ball positions frame-by-frame within a single shot to detect kicks.
    Triggers an impact event if there is a sharp acceleration/direction change.
    """
    n = len(positions)
    impacts = [False] * n
    
    for i in range(2, n):
        p_prev2 = positions[i-2]
        p_prev = positions[i-1]
        p_curr = positions[i]
        
        if p_prev2 is None or p_prev is None or p_curr is None:
            continue
            
        v1 = np.array([p_prev[0] - p_prev2[0], p_prev[1] - p_prev2[1]])
        v2 = np.array([p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]])
        
        s1 = np.linalg.norm(v1)
        s2 = np.linalg.norm(v2)
        
        if s1 < 1.5 and s2 < 1.5:
            continue
            
        a = v2 - v1
        accel_mag = np.linalg.norm(a)
        
        if accel_mag > threshold:
            impacts[i] = True
            
    return impacts

def draw_player_overlay(
    img: np.ndarray,
    players: list[dict],
    color: tuple[int, int, int] = (255, 191, 0)  # Neon Cyan in BGR
):
    """
    Draws player bounding boxes, neon corners, and text labels with track IDs.
    """
    for player in players:
        bbox = player["bbox"]
        track_id = player["track_id"]
        
        x1, y1, x2, y2 = map(int, bbox)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        
        len_corner = min(12, int((x2 - x1) * 0.2))
        thick_corner = 2
        # Top-left corner
        cv2.line(img, (x1, y1), (x1 + len_corner, y1), color, thick_corner)
        cv2.line(img, (x1, y1), (x1, y1 + len_corner), color, thick_corner)
        # Top-right corner
        cv2.line(img, (x2, y1), (x2 - len_corner, y1), color, thick_corner)
        cv2.line(img, (x2, y1), (x2, y1 + len_corner), color, thick_corner)
        # Bottom-left corner
        cv2.line(img, (x1, y2), (x1 + len_corner, y2), color, thick_corner)
        cv2.line(img, (x1, y2), (x1, y2 - len_corner), color, thick_corner)
        # Bottom-right corner
        cv2.line(img, (x2, y2), (x2 - len_corner, y2), color, thick_corner)
        cv2.line(img, (x2, y2), (x2, y2 - len_corner), color, thick_corner)
        
        label = f"Player #{track_id}" if track_id >= 0 else "Player"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        
        (w_text, h_text), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        if y1 - h_text - 8 < 0:
            y_text = y1 + h_text + 4
            p1 = (x1, y1)
            p2 = (x1 + w_text + 8, y1 + h_text + 8)
        else:
            y_text = y1 - 4
            p1 = (x1, y1 - h_text - 8)
            p2 = (x1 + w_text + 8, y1)
            
        cv2.rectangle(img, p1, p2, color, -1)
        cv2.putText(img, label, (x1 + 4, y_text), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

def draw_ball_overlay(
    img: np.ndarray,
    frame_idx_in_shot: int,
    positions: list[tuple[float, float] | None],
    impacts: list[bool],
    trail_len: int = 12,
    ball_color: tuple[int, int, int] = (0, 100, 255),      # Neon Orange in BGR
    impact_color: tuple[int, int, int] = (200, 255, 100)   # Glow Green-White in BGR
):
    """
    Draws ball movement graphics: fading trail, glow marker, expanding rings on impact.
    """
    current_pos = positions[frame_idx_in_shot]
    
    # 1. Fading Trail
    for k in range(1, trail_len + 1):
        prev_idx = frame_idx_in_shot - k
        if prev_idx < 0:
            break
            
        pt = positions[prev_idx]
        if pt is not None:
            age_ratio = k / (trail_len + 1)
            alpha = 0.85 * (1.0 - age_ratio)
            radius = max(2, int(6.0 * (1.0 - age_ratio)))
            draw_alpha_shape(img, (int(pt[0]), int(pt[1])), radius, ball_color, alpha)
            
    # 2. Impact Effects
    for i in range(max(0, frame_idx_in_shot - 8), frame_idx_in_shot + 1):
        if impacts[i]:
            impact_pos = positions[i]
            if impact_pos is not None:
                age = frame_idx_in_shot - i
                radius = int(8 + age * 6)
                alpha = 1.0 - (age / 8.0)
                draw_alpha_shape(img, (int(impact_pos[0]), int(impact_pos[1])), radius, impact_color, alpha, thickness=2)
                
        # 3. Ball Marker
    if current_pos is not None:
        cx, cy = int(current_pos[0]), int(current_pos[1])
        draw_alpha_shape(img, (cx, cy), 14, ball_color, 0.25)
        draw_alpha_shape(img, (cx, cy), 9, ball_color, 0.50)
        draw_alpha_shape(img, (cx, cy), 4, (255, 255, 255), 1.0)
        cv2.circle(img, (cx, cy), 4, ball_color, 1, cv2.LINE_AA)

# ---------------------------------------------------------------------------
# Track ID to player name heuristic mapper
# ---------------------------------------------------------------------------
from player_registry import get_squad, get_player_name

_track_to_name_map: dict[int, str] = {}
_assigned_names: set[str] = set()

def reset_possession():
    global _possession_frames, _track_to_name_map, _assigned_names
    _possession_frames = {0: 0, 1: 0}
    _track_to_name_map.clear()
    _assigned_names.clear()

def get_display_player_name(
    track_id: int,
    team_name: str,
    jersey_number: int | None,
) -> str:
    """
    Returns the best available name for a player:
      1. Dynamic cache check
      2. Registry lookup by jersey number (if valid)
      3. Roster-based deterministic name assignment
      4. Fallback 'Player #<track_id>'
    """
    if track_id < 0:
        return "Player"
        
    # Check if already assigned
    if track_id in _track_to_name_map:
        return _track_to_name_map[track_id]

    # Try registry lookup by jersey number if available
    if jersey_number is not None and team_name:
        name = get_player_name(team_name, jersey_number)
        if name:
            _track_to_name_map[track_id] = name
            _assigned_names.add(name)
            return name

    # Roster-based deterministic name assignment
    if team_name:
        squad = get_squad(team_name)
        if squad:
            # Sort squad players to make it deterministic
            squad_names = [squad[k] for k in sorted(squad.keys())]
            
            # Find the first name not already assigned to another track
            for name in squad_names:
                if name not in _assigned_names:
                    _track_to_name_map[track_id] = name
                    _assigned_names.add(name)
                    return name

    # Final fallback if roster is exhausted or team_name is empty
    fallback_name = f"#{track_id}"
    _track_to_name_map[track_id] = fallback_name
    return fallback_name

def record_possession(team_id: int):
    if team_id in _possession_frames:
        _possession_frames[team_id] += 1

def get_possession_pct() -> tuple[float, float]:
    total = _possession_frames[0] + _possession_frames[1]
    if total == 0:
        return 50.0, 50.0
    return (
        round(100.0 * _possession_frames[0] / total, 1),
        round(100.0 * _possession_frames[1] / total, 1),
    )


def draw_possession_bar(
    img: np.ndarray,
    team_a_name: str,
    team_b_name: str,
    team_colors: dict,
    possession_track_id: int,
    players: list[dict],
):
    """
    Draws a translucent possession percentage bar at the top center of the frame.
    Format:  SPAIN  62% ███░░  38% AUSTRIA
    """
    # Update possession counter
    for player in players:
        if player["track_id"] == possession_track_id:
            record_possession(player.get("team_id", 0))
            break

    pct_a, pct_b = get_possession_pct()

    h_img, w_img = img.shape[:2]
    bar_w = min(420, w_img - 40)
    bar_h = 28
    bx = (w_img - bar_w) // 2
    by = 8

    # Translucent background
    roi = img[by:by + bar_h + 24, bx:bx + bar_w]
    bg = np.zeros_like(roi)
    cv2.rectangle(bg, (0, 0), (bar_w, bar_h + 24), (15, 15, 15), -1)
    cv2.addWeighted(bg, 0.70, roi, 0.30, 0, dst=roi)

    # Team A color fill (left side)
    color_a = team_colors.get(0, (255, 80, 0))
    color_b = team_colors.get(1, (0, 80, 255))
    fill_a = int(bar_w * (pct_a / 100.0))
    cv2.rectangle(img, (bx, by), (bx + fill_a, by + bar_h), color_a, -1)
    cv2.rectangle(img, (bx + fill_a, by), (bx + bar_w, by + bar_h), color_b, -1)

    # Divider line
    cv2.line(img, (bx + fill_a, by), (bx + fill_a, by + bar_h), (255, 255, 255), 2)

    # Outer border
    cv2.rectangle(img, (bx, by), (bx + bar_w, by + bar_h), (80, 80, 80), 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Team A label + pct
    label_a = f"{(team_a_name or 'Team A').upper()}  {int(pct_a)}%"
    (tw_a, th_a), _ = cv2.getTextSize(label_a, font, 0.38, 1)
    cv2.putText(img, label_a, (bx + 6, by + bar_h // 2 + th_a // 2),
                font, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, label_a, (bx + 6, by + bar_h // 2 + th_a // 2),
                font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Team B label + pct (right-aligned)
    label_b = f"{int(pct_b)}%  {(team_b_name or 'Team B').upper()}"
    (tw_b, th_b), _ = cv2.getTextSize(label_b, font, 0.38, 1)
    cv2.putText(img, label_b, (bx + bar_w - tw_b - 6, by + bar_h // 2 + th_b // 2),
                font, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, label_b, (bx + bar_w - tw_b - 6, by + bar_h // 2 + th_b // 2),
                font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)


def draw_danger_zone_alert(
    img: np.ndarray,
    ball_pos: tuple[float, float] | None,
    frame_idx: int,
):
    """
    Detects when the ball is inside either penalty box area (left/right 15% of frame)
    and draws a pulsing red glow border + '⚡ DANGER ZONE' text.
    """
    if ball_pos is None:
        return

    h, w = img.shape[:2]
    bx, by = ball_pos

    # Penalty area zones: left 16% or right 16% of frame
    in_danger = (bx < w * 0.16) or (bx > w * 0.84)

    if not in_danger:
        return

    # Pulsing alpha based on frame number
    pulse = 0.35 + 0.25 * abs(np.sin(frame_idx * 0.25))

    # Draw red glowing border overlay
    border_overlay = img.copy()
    thickness = 18
    cv2.rectangle(border_overlay, (0, 0), (w, h), (0, 0, 220), thickness)
    cv2.addWeighted(border_overlay, pulse, img, 1.0 - pulse, 0, dst=img)

    # Draw 'DANGER ZONE' text at top center
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "! DANGER ZONE"
    (tw, th), _ = cv2.getTextSize(text, font, 0.70, 2)
    tx = (w - tw) // 2
    ty = 60
    # Shadow
    cv2.putText(img, text, (tx + 1, ty + 1), font, 0.70, (0, 0, 0), 3, cv2.LINE_AA)
    # Glow red text
    cv2.putText(img, text, (tx, ty), font, 0.70, (0, 0, 220), 2, cv2.LINE_AA)


def draw_tactical_radar(
    img: np.ndarray,
    players: list[dict],
    ball_pos: tuple[float, float] | None,
    H: np.ndarray,
    team_colors: dict,
    team_a_name: str = "",
    team_b_name: str = "",
):
    """
    Renders a translucent 2D tactical pitch radar in the bottom center of the screen
    projecting screen player positions onto a top-down tactical layout.
    Team names are labelled on either side of the radar.
    """
    h, w = img.shape[:2]
    radar_w, radar_h = 200, 130

    # Position in bottom middle
    rx_offset = (w - radar_w) // 2
    ry_offset = h - radar_h - 15
    
    # 1. Draw translucent background
    roi = img[ry_offset:ry_offset+radar_h, rx_offset:rx_offset+radar_w]
    bg = np.zeros_like(roi)
    cv2.rectangle(bg, (0, 0), (radar_w, radar_h), (20, 20, 20), -1)
    cv2.addWeighted(bg, 0.6, roi, 0.4, 0, dst=roi)
    
    # 2. Draw pitch lines
    # Outer border
    cv2.rectangle(img, (rx_offset, ry_offset), (rx_offset + radar_w, ry_offset + radar_h), (180, 180, 180), 1, cv2.LINE_AA)
    # Halfway line
    cv2.line(img, (rx_offset + radar_w // 2, ry_offset), (rx_offset + radar_w // 2, ry_offset + radar_h), (180, 180, 180), 1, cv2.LINE_AA)
    # Center circle
    cv2.circle(img, (rx_offset + radar_w // 2, ry_offset + radar_h // 2), 22, (180, 180, 180), 1, cv2.LINE_AA)
    # Left Penalty box
    cv2.rectangle(img, (rx_offset, ry_offset + int(radar_h * 0.22)), (rx_offset + int(radar_w * 0.15), ry_offset + int(radar_h * 0.78)), (180, 180, 180), 1, cv2.LINE_AA)
    # Right Penalty box
    cv2.rectangle(img, (rx_offset + int(radar_w * 0.85), ry_offset + int(radar_h * 0.22)), (rx_offset + radar_w, ry_offset + int(radar_h * 0.78)), (180, 180, 180), 1, cv2.LINE_AA)
    
    from tracking_utils import map_point_homography
    
    # 3. Draw projected players
    for player in players:
        bbox = player["bbox"]
        # Screen contact point: bottom-center of bounding box
        px = (bbox[0] + bbox[2]) / 2.0
        py = bbox[3]
        
        try:
            rx, ry = map_point_homography(H, px, py)
            # Clip within radar boundaries
            rx = max(2, min(radar_w - 3, rx))
            ry = max(2, min(radar_h - 3, ry))
            
            team_id = player.get("team_id", 0)
            color = team_colors.get(team_id, (0, 0, 255) if team_id == 0 else (255, 0, 0))
            
            # Draw dot
            cv2.circle(img, (rx_offset + rx, ry_offset + ry), 3, color, -1, cv2.LINE_AA)
            cv2.circle(img, (rx_offset + rx, ry_offset + ry), 4, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            pass
            
    # 3b. Draw team name labels on the radar
    font_r = cv2.FONT_HERSHEY_SIMPLEX
    if team_a_name:
        (tw_a, _), _ = cv2.getTextSize(team_a_name[:3].upper(), font_r, 0.28, 1)
        cv2.putText(img, team_a_name[:3].upper(),
                    (rx_offset + 3, ry_offset + 10),
                    font_r, 0.28, team_colors.get(0, (255, 100, 0)), 1, cv2.LINE_AA)
    if team_b_name:
        (tw_b, _), _ = cv2.getTextSize(team_b_name[:3].upper(), font_r, 0.28, 1)
        cv2.putText(img, team_b_name[:3].upper(),
                    (rx_offset + radar_w - tw_b - 3, ry_offset + 10),
                    font_r, 0.28, team_colors.get(1, (0, 100, 255)), 1, cv2.LINE_AA)

    # 4. Draw projected ball
    if ball_pos is not None:
        try:
            rx, ry = map_point_homography(H, ball_pos[0], ball_pos[1])
            rx = max(2, min(radar_w - 3, rx))
            ry = max(2, min(radar_h - 3, ry))
            # Glowing orange ball dot
            cv2.circle(img, (rx_offset + rx, ry_offset + ry), 3, (0, 120, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (rx_offset + rx, ry_offset + ry), 5, (0, 180, 255), 1, cv2.LINE_AA)
        except Exception:
            pass

def draw_advanced_player_overlay(
    img: np.ndarray,
    players: list[dict],
    possession_track_id: int,
    team_colors: dict,
    shot_idx: int,
    team_a_name: str = "",
    team_b_name: str = "",
    jersey_ocr=None,
    frame: np.ndarray | None = None,
):
    """
    Draws player tracking boxes using dynamically detected jersey colors,
    and displays a glowing active nameplate above the player in possession.
    Uses jersey OCR + registry lookup for real player names when available.
    """
    for player in players:
        bbox = player["bbox"]
        track_id = player["track_id"]
        team_id = player.get("team_id", 0)

        x1, y1, x2, y2 = map(int, bbox)
        color = team_colors.get(team_id, (255, 100, 0) if team_id == 0 else (0, 100, 255))

        # Resolve real team name for this player
        team_name = (team_a_name if team_id == 0 else team_b_name)

        # Try jersey OCR on the raw frame if available
        jersey_num = None
        if jersey_ocr is not None and frame is not None:
            jersey_num = jersey_ocr.read_jersey_number(frame, bbox, track_id)

        # 1. Draw player bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        len_corner = min(12, int((x2 - x1) * 0.2))
        thick_corner = 2
        # Corner markers
        cv2.line(img, (x1, y1), (x1 + len_corner, y1), color, thick_corner)
        cv2.line(img, (x1, y1), (x1, y1 + len_corner), color, thick_corner)
        cv2.line(img, (x2, y1), (x2 - len_corner, y1), color, thick_corner)
        cv2.line(img, (x2, y1), (x2, y1 + len_corner), color, thick_corner)
        cv2.line(img, (x1, y2), (x1 + len_corner, y2), color, thick_corner)
        cv2.line(img, (x1, y2), (x1, y2 - len_corner), color, thick_corner)
        cv2.line(img, (x2, y2), (x2 - len_corner, y2), color, thick_corner)
        cv2.line(img, (x2, y2), (x2, y2 - len_corner), color, thick_corner)

        # Resolve real player name using heuristic and face recognition fallback
        face_name = player.get("face_name")
        if face_name and track_id >= 0:
            _track_to_name_map[track_id] = face_name
            _assigned_names.add(face_name)

        resolved_name = get_display_player_name(track_id, team_name, jersey_num)

        # 1b. Always draw a small label above the box (Last name only for compact display)
        if resolved_name:
            # Handle fallback names starting with #
            mini_label = resolved_name if resolved_name.startswith('#') else resolved_name.split()[-1]
        else:
            mini_label = f"#{track_id}"
            
        font_mini = cv2.FONT_HERSHEY_SIMPLEX
        (mw, mh), _ = cv2.getTextSize(mini_label, font_mini, 0.30, 1)
        ml_x = x1
        ml_y = y1 - 4 if y1 - mh - 4 > 0 else y1 + mh + 4
        cv2.rectangle(img, (ml_x, ml_y - mh - 2), (ml_x + mw + 4, ml_y + 2), color, -1)
        cv2.putText(img, mini_label, (ml_x + 2, ml_y), font_mini, 0.30, (0, 0, 0), 1, cv2.LINE_AA)

        # 2. Draw Active Nameplate if player has the ball
        if track_id == possession_track_id and track_id >= 0:
            name = resolved_name
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 2
            
            # Calculate text size
            (w_text, h_text), baseline = cv2.getTextSize(name, font, font_scale, thickness)
            
            cx = (x1 + x2) // 2
            pill_w = w_text + 20
            pill_h = h_text + 12
            
            # Position above player
            py_center = y1 - 20
            px1 = cx - pill_w // 2
            py1 = py_center - pill_h // 2
            px2 = cx + pill_w // 2
            py2 = py_center + pill_h // 2
            
            # Clamp to screen dimensions
            if py1 < 5:
                # Put nameplate inside box top if running offscreen
                py_center = y1 + pill_h // 2 + 5
                px1 = cx - pill_w // 2
                py1 = py_center - pill_h // 2
                px2 = cx + pill_w // 2
                py2 = py_center + pill_h // 2
                
            # Draw translucent nameplate background
            h_img, w_img = img.shape[:2]
            px1_c = max(0, min(w_img - 2, px1))
            py1_c = max(0, min(h_img - 2, py1))
            px2_c = max(0, min(w_img - 2, px2))
            py2_c = max(0, min(h_img - 2, py2))
            
            if px2_c > px1_c and py2_c > py1_c:
                roi = img[py1_c:py2_c, px1_c:px2_c]
                bg = np.zeros_like(roi)
                cv2.rectangle(bg, (0, 0), (roi.shape[1], roi.shape[0]), (10, 10, 10), -1)
                cv2.addWeighted(bg, 0.75, roi, 0.25, 0, dst=roi)
                
            # Draw pill neon border
            cv2.rectangle(img, (px1, py1), (px2, py2), color, 1, cv2.LINE_AA)
            cv2.rectangle(img, (px1-1, py1-1), (px2+1, py2+1), (255, 255, 255), 1, cv2.LINE_AA)
            
            # Draw text
            cv2.putText(img, name, (cx - w_text // 2, py_center + h_text // 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            # Draw pointer indicator line
            cv2.line(img, (cx, py2), (cx, y1), color, 1, cv2.LINE_AA)

def draw_speed_badge(img: np.ndarray, speed_kmh: float):
    """
    Renders a premium visual speed gauge in the top-right corner.
    Only shows up if speed > 15 km/h.
    """
    if speed_kmh < 15.0:
        return
        
    h_img, w_img = img.shape[:2]
    
    # Position: top right
    x_offset = w_img - 200
    y_offset = 20
    badge_w, badge_h = 180, 80
    
    # 1. Draw translucent background
    roi = img[y_offset:y_offset+badge_h, x_offset:x_offset+badge_w]
    bg = np.zeros_like(roi)
    cv2.rectangle(bg, (0, 0), (badge_w, badge_h), (25, 25, 25), -1)
    cv2.addWeighted(bg, 0.70, roi, 0.30, 0, dst=roi)
    cv2.rectangle(img, (x_offset, y_offset), (x_offset + badge_w, y_offset + badge_h), (80, 80, 80), 1, cv2.LINE_AA)
    
    # 2. Draw circular speedometer dial
    cx, cy = x_offset + 45, y_offset + 40
    r = 26
    # Base circle
    cv2.circle(img, (cx, cy), r, (60, 60, 60), 2, cv2.LINE_AA)
    
    # Active arc based on speed (up to 120 km/h)
    speed_ratio = min(1.0, speed_kmh / 120.0)
    angle_end = int(speed_ratio * 360.0)
    
    # Draw arc in pink-red color
    cv2.ellipse(img, (cx, cy), (r, r), -90, 0, angle_end, (203, 19, 252), 3, cv2.LINE_AA)
    
    # Draw speed number inside circle
    val_str = f"{int(speed_kmh)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w_val, h_val), _ = cv2.getTextSize(val_str, font, 0.45, 2)
    cv2.putText(img, val_str, (cx - w_val // 2, cy + h_val // 2 - 2), font, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Draw unit inside circle
    (w_unit, h_unit), _ = cv2.getTextSize("KM/H", font, 0.25, 1)
    cv2.putText(img, "KM/H", (cx - w_unit // 2, cy + r - 7), font, 0.25, (180, 180, 180), 1, cv2.LINE_AA)
    
    # 3. Draw text header
    cv2.putText(img, "VITESSE", (x_offset + 85, y_offset + 32), font, 0.45, (255, 120, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "BALL SPEED", (x_offset + 85, y_offset + 52), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

def draw_shot_beam(
    img: np.ndarray,
    frame_idx_in_shot: int,
    positions: list[tuple[float, float] | None],
    impacts: list[bool],
    max_len: int = 15
):
    """
    Renders a thick, tapering glowing white beam/trajectory tracer connecting
    the kick origin to the ball's current location when a high-speed kick/impact occurs.
    """
    # Find most recent impact frame
    impact_idx = -1
    for i in range(max(0, frame_idx_in_shot - max_len), frame_idx_in_shot + 1):
        if impacts[i]:
            impact_idx = i
            
    if impact_idx == -1:
        return
        
    p_start = positions[impact_idx]
    p_curr = positions[frame_idx_in_shot]
    
    if p_start is not None and p_curr is not None:
        # Check if the ball moved enough (distance > 20px) to draw a beam
        v = np.array([p_curr[0] - p_start[0], p_curr[1] - p_start[1]])
        dist = np.linalg.norm(v)
        if dist < 20.0:
            return
            
        age = frame_idx_in_shot - impact_idx
        alpha = 0.85 * (1.0 - (age / (max_len + 1.0)))
        if alpha <= 0.0:
            return
            
        # Draw beam shape (tapering thick white line with a colored glow outline)
        # 1. Calculate normal vector to path
        norm_v = np.array([-v[1], v[0]])
        norm_v = norm_v / (np.linalg.norm(norm_v) + 1e-6)
        
        # Calculate tapering widths
        w_start = int(8.0 * alpha)
        w_end = int(2.0 * alpha)
        
        p1 = p_start + norm_v * w_start
        p2 = p_start - norm_v * w_start
        p3 = p_curr - norm_v * w_end
        p4 = p_curr + norm_v * w_end
        
        pts = np.array([p1, p2, p3, p4], dtype=np.int32)
        
        # Draw translucent polygon
        overlay = img.copy()
        # Draw glowing background trail
        cv2.fillPoly(overlay, [pts], (230, 240, 255), cv2.LINE_AA)
        # Draw border
        cv2.polylines(overlay, [pts], True, (255, 100, 0), 1, cv2.LINE_AA)
        
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)
        
        # Draw a bright core line
        cv2.line(img, (int(p_start[0]), int(p_start[1])), (int(p_curr[0]), int(p_curr[1])), (255, 255, 255), max(1, int(3 * alpha)), cv2.LINE_AA)

