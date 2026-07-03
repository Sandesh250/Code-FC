import os
import cv2
from scenedetect import open_video, SceneManager, ContentDetector

def find_shots(video_path: str, threshold: float = 27.0) -> list[tuple[int, int]]:
    """
    Detects shot transitions (cuts) in the input video using PySceneDetect.
    Returns a list of tuples (start_frame, end_frame) where start_frame is inclusive
    and end_frame is exclusive.
    
    If no cuts are detected or an error occurs, falls back to treating the entire
    video as a single shot.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    print(f"[Shot Segmentation] Analyzing video: {video_path}")
    
    # Get total frames using OpenCV as a reliable fallback/reference
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if total_frames <= 0:
        raise ValueError(f"Could not read frame count from video: {video_path}")
        
    print(f"[Shot Segmentation] Video stats: {total_frames} frames, {fps:.2f} FPS")
    
    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        
        # Perform scene detection
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
        
        shots = []
        for scene in scene_list:
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()
            # Clip end_frame to total_frames just in case of discrepancy
            end_frame = min(end_frame, total_frames)
            # Only add valid ranges
            if start_frame < end_frame:
                shots.append((start_frame, end_frame))
                
        if not shots:
            print("[Shot Segmentation] No cuts detected. Treating video as a single shot.")
            return [(0, total_frames)]
            
        # Ensure the entire video is covered
        if shots[0][0] > 0:
            shots[0] = (0, shots[0][1])
            
        if shots[-1][1] < total_frames:
            shots[-1] = (shots[-1][0], total_frames)
            
        print(f"[Shot Segmentation] Detected {len(shots)} shots:")
        for idx, (start, end) in enumerate(shots):
            duration_sec = (end - start) / fps if fps > 0 else 0
            print(f"  Shot {idx+1}: Frames {start} to {end} ({duration_sec:.2f}s)")
            
        return shots
        
    except Exception as e:
        print(f"[Shot Segmentation] Warning: PySceneDetect failed with error: {e}. Falling back to single shot.")
        return [(0, total_frames)]

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
        try:
            find_shots(video_file)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python shot_segmentation.py <video_path>")
