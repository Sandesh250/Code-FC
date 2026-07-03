import os
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on a Football Ball Detection Dataset from Roboflow")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml file of the downloaded Roboflow dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--device", type=str, default="0", help="Device to train on (e.g. '0' for CUDA GPU, or 'cpu')")
    
    args = parser.parse_args()
    
    data_path = os.path.abspath(args.data)
    if not os.path.exists(data_path):
        print(f"Error: Dataset configuration file not found at: {data_path}")
        print("Please download a dataset and double check the path to data.yaml.")
        return
        
    print("=" * 60)
    print("           GameCast YOLOv8 Fine-Tuning Utility")
    print("=" * 60)
    print(f"Dataset YAML: {data_path}")
    print(f"Epochs:       {args.epochs}")
    print(f"Image Size:   {args.imgsz}")
    print(f"Batch Size:   {args.batch}")
    print(f"Device:       {args.device}")
    print("-" * 60)
    
    print("[Train] Initializing pretrained weights (yolov8n.pt)...")
    model = YOLO("yolov8n.pt")
    
    print("[Train] Starting training run...")
    try:
        model.train(
            data=data_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            name="yolov8n_gamecast_ball",
            workers=4,
            exist_ok=True
        )
        print("=" * 60)
        print("Training successfully completed!")
        print("Your custom ball detector is saved at:")
        print("  runs/detect/yolov8n_gamecast_ball/weights/best.pt")
        print("=" * 60)
    except Exception as e:
        print(f"[Train] Error during training: {e}")

if __name__ == "__main__":
    main()
