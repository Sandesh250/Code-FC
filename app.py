import os
import threading
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from main import run_pipeline

from ultralytics import YOLO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 150 * 1024 * 1024

# Pre-load the YOLO model once in the main thread to avoid thread deadlocks
print("[Server Startup] Pre-loading YOLOv8 model...")
YOLO_MODEL = YOLO("yolov8n.pt")
print("[Server Startup] YOLOv8 model loaded.")

UPLOAD_FOLDER = os.path.abspath("uploads")
PROCESSED_FOLDER = os.path.abspath("processed")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

state_lock = threading.Lock()

processing_state = {
    "status": "idle",
    "percent": 0,
    "message": "Ready to upload.",
    "original_filename": "",
    "processed_filename": "",
    "error": ""
}

def progress_callback(percent: int, message: str):
    with state_lock:
        processing_state["percent"] = percent
        processing_state["message"] = message
    print(f"[Pipeline Progress] {percent}%: {message}")

def run_pipeline_worker(
    input_path: str,
    output_path: str,
    scene_threshold: float,
    conf: float,
    use_kalman: bool,
    orig_name: str,
    proc_name: str,
    team_a_name: str = "",
    team_b_name: str = "",
    team_a_color: tuple = (255, 80, 0),
    team_b_color: tuple = (0, 80, 255),
):
    global processing_state
    try:
        run_pipeline(
            input_path=input_path,
            output_path=output_path,
            model_path=YOLO_MODEL,
            scene_threshold=scene_threshold,
            sample_interval=5,
            conf=conf,
            interpolate_gap=15,
            use_kalman=use_kalman,
            team_a_name=team_a_name,
            team_b_name=team_b_name,
            team_a_color=team_a_color,
            team_b_color=team_b_color,
            progress_callback=progress_callback
        )
        with state_lock:
            processing_state["status"] = "completed"
            processing_state["percent"] = 100
            processing_state["message"] = "Highlight video successfully gamified!"
            processing_state["original_filename"] = orig_name
            processing_state["processed_filename"] = proc_name
    except Exception as e:
        import traceback
        traceback.print_exc()
        with state_lock:
            processing_state["status"] = "failed"
            processing_state["percent"] = 0
            processing_state["message"] = f"Pipeline failed: {str(e)}"
            processing_state["error"] = str(e)

@app.route("/api/upload", methods=["POST"])
def upload_video():
    global processing_state
    with state_lock:
        if processing_state["status"] == "processing":
            return jsonify({"error": "The server is currently busy processing another highlight video. Please wait."}), 409
            
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request payload"}), 400
        
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected video file"}), 400
        
    try:
        scene_threshold = float(request.form.get("scene_threshold", 27.0))
        conf = float(request.form.get("conf", 0.25))
        use_kalman = request.form.get("use_kalman", "false").lower() == "true"
        team_a_name = request.form.get("team_a_name", "").strip()
        team_b_name = request.form.get("team_b_name", "").strip()
        # Colors arrive as hex strings e.g. "#FF0000" → convert to BGR tuple
        def hex_to_bgr(hex_str: str, default: tuple) -> tuple:
            hex_str = hex_str.strip().lstrip('#')
            if len(hex_str) != 6:
                return default
            r, g, b = int(hex_str[0:2],16), int(hex_str[2:4],16), int(hex_str[4:6],16)
            return (b, g, r)  # OpenCV is BGR
        team_a_color = hex_to_bgr(request.form.get("team_a_color", ""), (255, 80, 0))
        team_b_color = hex_to_bgr(request.form.get("team_b_color", ""), (0, 80, 255))
    except ValueError:
        return jsonify({"error": "Invalid config parameter values provided"}), 400
        
    filename = secure_filename(file.filename)
    if not filename:
        filename = "highlight.mp4"
        
    orig_name = f"orig_{filename}"
    proc_name = f"proc_{filename}"
    
    input_path = os.path.join(UPLOAD_FOLDER, orig_name)
    output_path = os.path.join(PROCESSED_FOLDER, proc_name)
    
    try:
        file.save(input_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save uploaded file: {str(e)}"}), 500
        
    with state_lock:
        processing_state["status"] = "processing"
        processing_state["percent"] = 0
        processing_state["message"] = "Video upload completed. Running shot segmenter..."
        processing_state["original_filename"] = ""
        processing_state["processed_filename"] = ""
        processing_state["error"] = ""
        
    t = threading.Thread(
        target=run_pipeline_worker,
        args=(input_path, output_path, scene_threshold, conf, use_kalman,
              orig_name, proc_name, team_a_name, team_b_name,
              team_a_color, team_b_color)
    )
    t.daemon = True
    t.start()
    
    return jsonify({
        "status": "processing",
        "message": "Processing thread started."
    })

@app.route("/api/progress", methods=["GET"])
def check_progress():
    with state_lock:
        return jsonify(processing_state)

@app.route("/api/reset", methods=["POST"])
def reset_state():
    global processing_state
    with state_lock:
        if processing_state["status"] != "processing":
            processing_state["status"] = "idle"
            processing_state["percent"] = 0
            processing_state["message"] = "Ready to upload."
            processing_state["original_filename"] = ""
            processing_state["processed_filename"] = ""
            processing_state["error"] = ""
            return jsonify({"status": "idle"})
        return jsonify({"error": "Cannot reset state while actively processing"}), 400

@app.route("/videos/original/<filename>")
def serve_original_video(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/videos/processed/<filename>")
def serve_processed_video(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react_app(path):
    dist_dir = os.path.abspath("frontend/dist")
    if os.path.exists(dist_dir):
        if path != "" and os.path.exists(os.path.join(dist_dir, path)):
            return send_from_directory(dist_dir, path)
        else:
            return send_from_directory(dist_dir, "index.html")
    else:
        return (
            "<h3>Flask server running.</h3>"
            "<p>React frontend is not compiled yet. Please run the Vite dev server inside "
            "the <b>frontend/</b> directory: <code>npm run dev</code></p>"
        ), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
