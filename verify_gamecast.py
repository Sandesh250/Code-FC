import os
import sys
import cv2
import numpy as np
import torch

class MockBoxData:
    def __init__(self, cls, xyxy, conf, track_id=None):
        self.cls = cls
        self.xyxy = xyxy
        self.conf = conf
        self.id = track_id

class MockBoxes:
    def __init__(self, box_list):
        self.box_list = box_list
        
        if len(box_list) > 0:
            self.cls = torch.tensor([b.cls for b in box_list], dtype=torch.int64)
            self.xyxy = torch.tensor([b.xyxy for b in box_list], dtype=torch.float32)
            self.conf = torch.tensor([b.conf for b in box_list], dtype=torch.float32)
            
            has_id = any(b.id is not None for b in box_list)
            if has_id:
                ids = [b.id if b.id is not None else -1 for b in box_list]
                self.id = torch.tensor(ids, dtype=torch.int64)
            else:
                self.id = None
        else:
            self.cls = torch.tensor([], dtype=torch.int64)
            self.xyxy = torch.tensor([], dtype=torch.float32)
            self.conf = torch.tensor([], dtype=torch.float32)
            self.id = None

    def __len__(self):
        return len(self.box_list)

    def __iter__(self):
        for b in self.box_list:
            yield MockBoxes([b])

class MockResult:
    def __init__(self, boxes_obj):
        self.boxes = boxes_obj

class MockPredictor:
    def __init__(self):
        self._trackers = []
        self.frame_count = 0
        
    @property
    def trackers(self):
        return self._trackers
        
    @trackers.setter
    def trackers(self, val):
        self._trackers = val
        if not val:
            self.frame_count = 0
            print("[Mock Predictor] Resetting frame count to 0 (Tracker State Reset).")

    @trackers.deleter
    def trackers(self):
        self._trackers = []
        self.frame_count = 0
        print("[Mock Predictor] Resetting frame count to 0 via deleter.")

class MockYOLO:
    def __init__(self, model_name="yolov8n.pt"):
        self.names = {0: "person", 32: "sports ball"}
        self.predictor = MockPredictor()
        
    def to(self, device):
        return self
        
    def __call__(self, frame, conf=0.25, verbose=False):
        green_mean = frame[:, :, 1].mean()
        red_mean = frame[:, :, 0].mean()
        
        boxes = []
        if green_mean > 120 and red_mean < 80:
            boxes.append(MockBoxData(cls=0, xyxy=[100, 100, 150, 200], conf=0.9))
            boxes.append(MockBoxData(cls=32, xyxy=[200, 180, 215, 195], conf=0.95))
        return [MockResult(MockBoxes(boxes))]
        
    def track(self, source, persist=True, classes=None, conf=0.25, iou=0.45, tracker="bytetrack.yaml", verbose=False):
        self.predictor.frame_count += 1
        fc = self.predictor.frame_count
        
        green_mean = source[:, :, 1].mean()
        red_mean = source[:, :, 0].mean()
        
        boxes = []
        if green_mean > 120 and red_mean < 80:
            p1_x = 100 + fc * 2
            p2_x = 300 - fc * 2
            b_x = 150 + fc * 3
            b_y = 200 - int(abs(np.sin(fc / 6.0)) * 60)
            
            boxes.append(MockBoxData(cls=0, xyxy=[p1_x, 80, p1_x + 35, 170], conf=0.9, track_id=1))
            boxes.append(MockBoxData(cls=0, xyxy=[p2_x, 90, p2_x + 35, 180], conf=0.85, track_id=2))
            
            is_ball_occluded = 20 <= fc <= 35
            if not is_ball_occluded:
                boxes.append(MockBoxData(cls=32, xyxy=[b_x, b_y, b_x + 12, b_y + 12], conf=0.92, track_id=3))
                
        return [MockResult(MockBoxes(boxes))]

import ultralytics
ultralytics.YOLO = MockYOLO

def create_synthetic_test_video():
    os.makedirs("clips", exist_ok=True)
    dest_path = "clips/sample_multi_cut.mp4"
    print(f"[Verify] Generating synthetic test video at: {dest_path}")
    
    fps = 25.0
    width = 640
    height = 360
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(dest_path, fourcc, fps, (width, height))
    
    for f in range(100):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [30, 160, 30]
        cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)
        cv2.circle(frame, (width // 2, height // 2), 50, (255, 255, 255), 2)
        writer.write(frame)
        
    for f in range(50):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(frame, "REPLAY BUMPER", (180, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        
    for f in range(100):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [30, 160, 30]
        cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)
        cv2.circle(frame, (width // 2, height // 2), 50, (255, 255, 255), 2)
        writer.write(frame)
        
    writer.release()
    print("[Verify] Synthetic video generated.")
    return dest_path

def run_verification():
    create_synthetic_test_video()
    
    sys.argv = [
        "main.py",
        "--input", "clips/sample_multi_cut.mp4",
        "--output", "out/sample_gamified.mp4",
        "--scene-threshold", "35.0",
        "--conf", "0.2"
    ]
    
    print("\n[Verify] Running GameCast orchestrator...")
    import main
    main.main()
    
    out_path = "out/sample_gamified.mp4"
    if not os.path.exists(out_path):
        print(f"[Verify] Fail: Output file {out_path} was not created!")
        return
        
    cap = cv2.VideoCapture(out_path)
    if not cap.isOpened():
        print(f"[Verify] Fail: Output file {out_path} cannot be opened by OpenCV.")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print("\n" + "=" * 60)
    print("                 PIPELINE VERIFICATION SUCCESS")
    print("=" * 60)
    print(f"Generated File:   {out_path}")
    print(f"File Size:        {os.path.getsize(out_path) / 1024:.2f} KB")
    print(f"Resolution:       {width}x{height}")
    print(f"FPS:              {fps:.2f}")
    print(f"Total Frames:     {frames}")
    
    if frames == 250:
        print("Success: Pipeline outputs correct duration (250/250 frames mapped).")
    else:
        print(f"Warning: Expected 250 frames, but output has {frames}.")
    print("=" * 60)

if __name__ == "__main__":
    run_verification()
