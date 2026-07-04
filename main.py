import os
import sys
import argparse
import cv2
import torch
from ultralytics import YOLO

from shot_segmentation import find_shots
from relevance_check import check_shot_relevance
from detection import track_shot
from tracking_utils import (interpolate_ball_positions, filter_ball_trajectory_kalman,
                            calculate_ball_speeds, get_perspective_transformer,
                            map_point_homography, CameraMovementEstimator,
                            estimate_player_performance)
from overlay import (draw_player_overlay, draw_ball_overlay, detect_impacts,
                     draw_tactical_radar, draw_advanced_player_overlay,
                     draw_speed_badge, draw_shot_beam,
                     draw_possession_bar, draw_danger_zone_alert, reset_possession)
from jersey_ocr import JerseyOCR
from face_recognizer import FaceRecognizer
import numpy as np
from stitch import get_video_writer, finalize_video_with_audio

def parse_args():
    parser = argparse.ArgumentParser(description="GameCast: Football Highlight Gamification Pipeline")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the input football highlight video file (MP4)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path where the final gamified video will be saved"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLOv8 weights (e.g. yolov8n.pt or a custom fine-tuned model path)"
    )
    parser.add_argument(
        "--scene-threshold", "-t",
        type=float,
        default=27.0,
        help="PySceneDetect content threshold (default: 27.0)"
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=5,
        help="Sample interval for relevance check (every Nth frame, default: 5)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO tracking (default: 0.25)"
    )
    parser.add_argument(
        "--interpolate-gap",
        type=int,
        default=15,
        help="Max frames to interpolate ball position over occlusions (default: 15)"
    )
    parser.add_argument(
        "--use-kalman",
        action="store_true",
        help="Apply Kalman filter smoothing to the interpolated ball trajectory"
    )
    return parser.parse_args()

def run_pipeline(
    input_path: str,
    output_path: str,
    model_path: str = "yolov8n.pt",
    scene_threshold: float = 27.0,
    sample_interval: int = 5,
    conf: float = 0.25,
    interpolate_gap: int = 15,
    use_kalman: bool = False,
    team_a_name: str = "",
    team_b_name: str = "",
    team_a_color: tuple = (255, 80, 0),
    team_b_color: tuple = (0, 80, 255),
    progress_callback = None
):
    """
    Runs the full GameCast video processing pipeline from input_path to output_path.
    Can optionally report progress percentage and status text to progress_callback.
    """
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
        
    cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open input video {input_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if progress_callback:
        progress_callback(5, "Initializing YOLOv8 model on device...")
        
    if isinstance(model_path, str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Main] Loading model '{model_path}' on device: {device}")
        model = YOLO(model_path)
        try:
            model.to(device)
        except Exception as e:
            print(f"[Main] Warning: Could not move model to {device}: {e}. Using CPU.")
    else:
        model = model_path
    
    if progress_callback:
        progress_callback(10, "Splitting video into shots...")
        
    shots = find_shots(input_path, threshold=scene_threshold)
    print(f"[Main] Shot segmentation completed. Found {len(shots)} shots.")
    
    if progress_callback:
        progress_callback(15, f"Segmented {len(shots)} shots. Preparing processing...")
        
    # ---- Initialize team color overrides ----
    team_colors_override = {
        0: tuple(int(c) for c in team_a_color),
        1: tuple(int(c) for c in team_b_color),
    }

    # ---- Initialize Jersey OCR + Face Recognizer ----
    jersey_ocr = JerseyOCR()
    active_match_tuple = (team_a_name, team_b_name)
    face_recognizer_a = FaceRecognizer(team_filter=team_a_name, active_match=active_match_tuple)
    face_recognizer_b = FaceRecognizer(team_filter=team_b_name, active_match=active_match_tuple)

    # ---- Telemetry Data Storage ----
    timeline_telemetry = [None] * total_frames
    player_distances = {}
    player_last_radar = {}

    out_dir = os.path.dirname(output_path) or "."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    temp_silent_path = os.path.join(out_dir, f"temp_silent_{os.path.basename(output_path)}")
    
    writer = get_video_writer(temp_silent_path, fps, width, height)
    cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(input_path)
    
    process_shots_count = 0
    passthrough_shots_count = 0
    
    for idx, (start, end) in enumerate(shots):
        percent = 20 + int((idx / len(shots)) * 65)
        status_msg = f"Shot {idx+1}/{len(shots)}: analyzing frame range {start}-{end}..."
        
        if progress_callback:
            progress_callback(percent, status_msg)
            
        print(f"\n[Main] Processing Shot {idx+1}/{len(shots)} (Frames {start} to {end})")
        
        is_relevant = check_shot_relevance(
            cap=cap,
            start_frame=start,
            end_frame=end,
            model=model,
            sample_interval=sample_interval,
            conf_threshold=conf
        )
        
        if is_relevant:
            process_shots_count += 1
            print(f"[Main] Shot {idx+1} is classified as GAMEPLAY. Processing...")
            
            if progress_callback:
                progress_callback(percent, f"Shot {idx+1}/{len(shots)}: tracking players and ball...")
                
            raw_tracking_data, team_colors = track_shot(
                cap=cap,
                start_frame=start,
                end_frame=end,
                model=model,
                conf_threshold=conf
            )

            # Override K-Means colors with user-supplied team colors if provided
            if team_a_name or team_b_name:
                team_colors[0] = team_colors_override[0]
                team_colors[1] = team_colors_override[1]

            # Reset per-shot state
            reset_possession()
            jersey_ocr.reset()
            face_recognizer_a.reset()
            face_recognizer_b.reset()
            
            raw_ball_positions = []
            for frame_data in raw_tracking_data:
                if frame_data["balls"]:
                    raw_ball_positions.append(frame_data["balls"][0]["center"])
                else:
                    raw_ball_positions.append(None)
                    
            ball_positions = interpolate_ball_positions(raw_ball_positions, max_gap=interpolate_gap)
            
            if use_kalman:
                ball_positions = filter_ball_trajectory_kalman(ball_positions)
                
            impacts = detect_impacts(ball_positions)
            ball_speeds = calculate_ball_speeds(ball_positions, fps=fps)
            H = get_perspective_transformer(width, height, radar_w=200, radar_h=130)
            
            # Initialize camera motion estimator for this shot
            camera_estimator = CameraMovementEstimator()

            if progress_callback:
                progress_callback(percent, f"Shot {idx+1}/{len(shots)}: drawing visual effects...")
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for f_offset in range(len(raw_tracking_data)):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_data = raw_tracking_data[f_offset]
                ball_pos = ball_positions[f_offset]
                f_idx = start + f_offset

                # Estimate camera motion translation vector
                cam_dx, cam_dy = camera_estimator.estimate(frame)
                cam_mag = float(np.sqrt(cam_dx*cam_dx + cam_dy*cam_dy))
                
                # Determine player in possession
                possession_track_id = -1
                if ball_pos is not None:
                    min_dist = float('inf')
                    for player in frame_data["players"]:
                        bbox = player["bbox"]
                        px = (bbox[0] + bbox[2]) / 2.0
                        py = bbox[3]  # bottom contact point
                        dist = np.sqrt((px - ball_pos[0])**2 + (py - ball_pos[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            possession_track_id = player["track_id"]
                    if min_dist > 35.0:
                        possession_track_id = -1

                # --- Resolve face recognition names for players ---
                for player in frame_data["players"]:
                    team_id = player.get("team_id", 0)
                    fr = face_recognizer_a if team_id == 0 else face_recognizer_b
                    result = fr.identify(frame, player["bbox"], player["track_id"])
                    if result:
                        player["face_name"] = result["name"]
                        player["face_conf"] = result["confidence"]

                # Draw overlays
                # 0. Possession bar (top center)
                draw_possession_bar(
                    img=frame,
                    team_a_name=team_a_name,
                    team_b_name=team_b_name,
                    team_colors=team_colors,
                    possession_track_id=possession_track_id,
                    players=frame_data["players"],
                )

                # 1. Advanced player overlay with real names, jersey OCR, face recognition
                draw_advanced_player_overlay(
                    img=frame,
                    players=frame_data["players"],
                    possession_track_id=possession_track_id,
                    team_colors=team_colors,
                    shot_idx=idx,
                    team_a_name=team_a_name,
                    team_b_name=team_b_name,
                    jersey_ocr=jersey_ocr,
                    frame=frame,
                )

                # 2. Danger zone alert
                draw_danger_zone_alert(
                    img=frame,
                    ball_pos=ball_pos,
                    frame_idx=f_offset,
                )
                
                # 3. Ball overlay (glow, trail, kick shockwaves)
                draw_ball_overlay(
                    img=frame,
                    frame_idx_in_shot=f_offset,
                    positions=ball_positions,
                    impacts=impacts
                )

                # 4. Glowing shot tracer beam
                draw_shot_beam(
                    img=frame,
                    frame_idx_in_shot=f_offset,
                    positions=ball_positions,
                    impacts=impacts
                )

                # 5. Circular speed dial at the top right
                draw_speed_badge(frame, speed_kmh=ball_speeds[f_offset])

                # 6. Translucent top-down radar mini-map at bottom-center
                draw_tactical_radar(
                    img=frame,
                    players=frame_data["players"],
                    ball_pos=ball_pos,
                    H=H,
                    team_colors=team_colors,
                    team_a_name=team_a_name,
                    team_b_name=team_b_name,
                )
                
                # ---- Collect player metrics telemetry for this frame ----
                from overlay import _track_to_name_map, get_possession_pct
                pct_a, pct_b = get_possession_pct()
                
                frame_players_telemetry = []
                for player in frame_data["players"]:
                    track_id = player["track_id"]
                    team_id = player.get("team_id", 0)
                    bbox = player["bbox"]
                    
                    px = (bbox[0] + bbox[2]) / 2.0
                    py = bbox[3]  # Contact point (bottom center of player box)
                    
                    try:
                        rx, ry = map_point_homography(H, px, py)
                    except Exception:
                        rx, ry = 100, 65 # center fallback
                        
                    # Calculate running speed and accumulate distance
                    speed_kmh, dist_m = estimate_player_performance(
                        player_last_radar.get(track_id), (rx, ry), fps
                    )
                    player_distances[track_id] = player_distances.get(track_id, 0.0) + dist_m
                    player_last_radar[track_id] = (rx, ry)
                    
                    # Roster name lookup
                    name = _track_to_name_map.get(track_id, f"#{track_id}")
                    
                    frame_players_telemetry.append({
                        "name": name,
                        "track_id": track_id,
                        "team_id": team_id,
                        "speed_kmh": round(speed_kmh, 1),
                        "distance_m": round(player_distances[track_id], 1),
                        "radar_coords": [rx, ry]
                    })
                    
                # Save telemetry for this frame
                timeline_telemetry[f_idx] = {
                    "frame_idx": f_idx,
                    "ball_speed": round(ball_speeds[f_offset], 1),
                    "possession_pct_a": pct_a,
                    "possession_pct_b": pct_b,
                    "camera_motion": round(cam_mag, 1),
                    "players": frame_players_telemetry
                }
                
                writer.write(frame)
                
        else:
            passthrough_shots_count += 1
            print(f"[Main] Shot {idx+1} is classified as PASS-THROUGH. Copying original frames...")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for f_idx in range(start, end):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Store default blank stats for pass-through frames
                timeline_telemetry[f_idx] = {
                    "frame_idx": f_idx,
                    "ball_speed": 0.0,
                    "possession_pct_a": 50.0,
                    "possession_pct_b": 50.0,
                    "camera_motion": 0.0,
                    "players": []
                }
                writer.write(frame)
                
    cap.release()
    writer.release()
    
    print("\n" + "=" * 60)
    print("                 Pipeline Processing Complete")
    print("=" * 60)
    
    if progress_callback:
        progress_callback(85, "Stitching shots and restoring sound via FFmpeg...")
        
    finalize_video_with_audio(temp_silent_path, input_path, output_path)

    # Export analytical telemetry for interactive dashboard
    import json
    json_path = output_path + ".json"
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "fps": fps,
                "total_frames": total_frames,
                "team_a_name": team_a_name,
                "team_b_name": team_b_name,
                "team_a_color": team_a_color,
                "team_b_color": team_b_color,
                "timeline": timeline_telemetry
            }, f, indent=2)
        print(f"[Main] Telemetry JSON exported successfully -> {json_path}")
    except Exception as e:
        print(f"[Main] Warning: Could not export telemetry JSON: {e}")
    
    if progress_callback:
        progress_callback(100, "Processing complete!")
        
    return {
        "total_shots": len(shots),
        "gameplay_shots": process_shots_count,
        "passthrough_shots": passthrough_shots_count
    }

def main():
    args = parse_args()
    try:
        res = run_pipeline(
            input_path=args.input,
            output_path=args.output,
            model_path=args.model,
            scene_threshold=args.scene_threshold,
            sample_interval=args.sample_interval,
            conf=args.conf,
            interpolate_gap=args.interpolate_gap,
            use_kalman=args.use_kalman
        )
        print(f"Gameplay Shots Processed:     {res['gameplay_shots']}")
        print(f"Pass-Through Shots Copied:    {res['passthrough_shots']}")
        print(f"Total Shots:                  {res['total_shots']}")
        print("=" * 60)
    except Exception as e:
        print(f"[Main] Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
