import cv2
from ultralytics import YOLO

def check_shot_relevance(
    cap: cv2.VideoCapture,
    start_frame: int,
    end_frame: int,
    model: YOLO,
    sample_interval: int = 5,
    conf_threshold: float = 0.35
) -> bool:
    """
    Samples frames from the given shot using sequential reading (no seek-in-loop)
    to classify the shot as gameplay (relevant) or closeup/replay (pass-through).
    """
    # Seek once to the start of the shot
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    sampled_frames_count = 0
    ball_detections = 0
    max_persons_in_frame = 0
    frames_with_persons = 0
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Iterate through the frame range sequentially to avoid expensive decoder seek operations
    for f_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Only analyze every sample_interval-th frame
        if (f_idx - start_frame) % sample_interval == 0:
            sampled_frames_count += 1
            results = model(frame, conf=conf_threshold, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
                
            persons_count = 0
            for cls in boxes.cls:
                cls_id = int(cls.item())
                cls_name = model.names.get(cls_id, "").lower()
                
                # Check for person (class 0 in COCO) or equivalent name
                is_person = (cls_id == 0) or ("person" in cls_name) or ("player" in cls_name) or ("goalkeeper" in cls_name) or ("referee" in cls_name)
                # Check for ball (class 32 in COCO) or equivalent name
                is_ball = (cls_id == 32) or ("ball" in cls_name)
                
                if is_person:
                    persons_count += 1
                if is_ball:
                    ball_detections += 1
                    
            if persons_count > 0:
                frames_with_persons += 1
            max_persons_in_frame = max(max_persons_in_frame, persons_count)
            
    # Decision heuristic
    if ball_detections > 0:
        print(f"[Relevance Check] Shot {start_frame}-{end_frame}: RELEVANT (Ball detected {ball_detections} times)")
        return True
        
    if max_persons_in_frame >= 3:
        print(f"[Relevance Check] Shot {start_frame}-{end_frame}: RELEVANT (Wide view: max {max_persons_in_frame} players in frame)")
        return True
        
    if max_persons_in_frame >= 2 and frames_with_persons >= max(1, sampled_frames_count // 3):
        print(f"[Relevance Check] Shot {start_frame}-{end_frame}: RELEVANT (Active player presence: {max_persons_in_frame} players, seen in {frames_with_persons}/{sampled_frames_count} frames)")
        return True
        
    print(f"[Relevance Check] Shot {start_frame}-{end_frame}: PASS-THROUGH (No ball, max players: {max_persons_in_frame})")
    return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        model = YOLO("yolov8n.pt")
        print(f"Testing relevance on the first 100 frames of {video_path}...")
        cap = cv2.VideoCapture(video_path)
        res = check_shot_relevance(cap, 0, 100, model)
        cap.release()
        print(f"Result: {res}")
    else:
        print("Usage: python relevance_check.py <video_path>")
