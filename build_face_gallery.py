"""
build_face_gallery.py
---------------------
Builds a face embedding gallery from the FIFA WC 2022 face image dataset.
Uses InsightFace (ArcFace) to extract 512-dim embeddings per player.
Saves the gallery as face_gallery.pkl for runtime use.

Run once (takes ~5-15 min for 41k images):
    python build_face_gallery.py

Requirements:
    pip install insightface onnxruntime-gpu opencv-python tqdm
"""

import os, sys, pickle, json
sys.stdout.reconfigure(encoding='utf-8')
import cv2
import numpy as np
from tqdm import tqdm

FACE_IMAGES_BASE = r"C:\Users\sande\Downloads\face images\Images\Images"
OUTPUT_GALLERY   = r"C:\Users\sande\Documents\CODES\Code FC\face_gallery.pkl"
OUTPUT_NAMES     = r"C:\Users\sande\Documents\CODES\Code FC\face_gallery_names.json"

# Team folder name → registry team key mapping
TEAM_MAP = {
    "Argentina Players":   "argentina",
    "Australia Players":   "australia",
    "Belgium Players":     "belgium",
    "Brazil Players":      "brazil",
    "Cameroon Players":    "cameroon",
    "Canada Players":      "canada",
    "Costa Rica Players":  "costa rica",
    "Croatia Players":     "croatia",
    "Denmark Players":     "denmark",
    "Ecuador Players":     "ecuador",
    "England Players":     "england",
    "France Players":      "france",
    "Germany Players":     "germany",
    "Ghana Players":       "ghana",
    "Iran Players":        "iran",
    "Japan Players":       "japan",
    "Mexico Players":      "mexico",
    "Morocco players":     "morocco",
    "Netherland Players":  "netherlands",
    "Poland Players":      "poland",
    "Portugal Players":    "portugal",
    "Qatar Players":       "qatar",
    "Saudi Arabia Players":"saudi arabia",
    "Senegal Players":     "senegal",
    "Serbia Players":      "serbia",
    "South Korea Players": "south korea",
    "Spain Players":       "spain",
    "Switzerland Players": "switzerland",
    "Tunisia Players":     "tunisia",
    "United States Players":"united states",
    "Uruguay Players":     "uruguay",
    "Wales Players":       "wales",
}

def load_insightface():
    """Load InsightFace ArcFace model."""
    try:
        import insightface
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(128, 128))
        print("[FaceGallery] InsightFace loaded (GPU/CPU)")
        return app
    except ImportError:
        print("[FaceGallery] ERROR: insightface not installed.")
        print("  Run: pip install insightface onnxruntime-gpu")
        sys.exit(1)

def extract_embedding(face_app, img_path: str) -> np.ndarray | None:
    """Extract ArcFace embedding from an image file. Returns None on failure."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    # Upscale small images
    h, w = img.shape[:2]
    if h < 64 or w < 64:
        scale = max(64/h, 64/w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    try:
        faces = face_app.get(img)
        if not faces:
            return None
        # Use the largest detected face
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        emb = face.embedding
        return emb / (np.linalg.norm(emb) + 1e-6)
    except Exception:
        return None

def collect_player_paths() -> list[dict]:
    """Walk the dataset directory and collect all players with their images."""
    players = []
    for group in sorted(os.listdir(FACE_IMAGES_BASE)):
        gpath = os.path.join(FACE_IMAGES_BASE, group)
        if not os.path.isdir(gpath):
            continue
        for team_folder in sorted(os.listdir(gpath)):
            tpath = os.path.join(gpath, team_folder)
            if not os.path.isdir(tpath):
                continue
            team_key = TEAM_MAP.get(team_folder, team_folder.lower().replace(" players",""))
            for player_folder in sorted(os.listdir(tpath)):
                ppath = os.path.join(tpath, player_folder)
                if not os.path.isdir(ppath):
                    continue
                # Clean player name: "Images_Neymar" -> "Neymar"
                player_name = player_folder.replace("Images_","").replace("_"," ").strip()
                imgs = [
                    os.path.join(ppath, f)
                    for f in os.listdir(ppath)
                    if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))
                ]
                if imgs:
                    players.append({
                        "name": player_name,
                        "team": team_key,
                        "images": imgs
                    })
    return players

def build_gallery():
    print(f"[FaceGallery] Scanning dataset: {FACE_IMAGES_BASE}")
    players = collect_player_paths()
    print(f"[FaceGallery] Found {len(players)} players across all teams")

    face_app = load_insightface()

    gallery: dict[str, np.ndarray] = {}   # "player_name::team" -> mean embedding
    meta: dict[str, dict] = {}            # for JSON output

    total_imgs = sum(len(p["images"]) for p in players)
    success_count = 0
    fail_count = 0

    with tqdm(total=total_imgs, desc="Extracting embeddings", unit="img") as pbar:
        for player in players:
            embeddings = []
            for img_path in player["images"]:
                emb = extract_embedding(face_app, img_path)
                if emb is not None:
                    embeddings.append(emb)
                    success_count += 1
                else:
                    fail_count += 1
                pbar.update(1)

            if embeddings:
                mean_emb = np.mean(embeddings, axis=0)
                mean_emb /= (np.linalg.norm(mean_emb) + 1e-6)
                key = f"{player['name']}::{player['team']}"
                gallery[key] = mean_emb
                meta[key] = {
                    "name": player["name"],
                    "team": player["team"],
                    "n_images": len(player["images"]),
                    "n_embedded": len(embeddings),
                }

    print(f"\n[FaceGallery] Embedding complete:")
    print(f"  Players with gallery entry: {len(gallery)}")
    print(f"  Images successfully embedded: {success_count}")
    print(f"  Images failed (no face detected): {fail_count}")

    # Save gallery
    with open(OUTPUT_GALLERY, 'wb') as f:
        pickle.dump(gallery, f)
    print(f"[FaceGallery] Gallery saved → {OUTPUT_GALLERY}")

    # Save names metadata
    with open(OUTPUT_NAMES, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[FaceGallery] Names metadata saved → {OUTPUT_NAMES}")

if __name__ == "__main__":
    build_gallery()
