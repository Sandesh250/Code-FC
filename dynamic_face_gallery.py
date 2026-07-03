"""
dynamic_face_gallery.py
-----------------------
Dynamically builds a mini face embedding gallery for the two active teams
playing in the Match Context. This reduces the embedding time from 8 hours
to less than 15-20 seconds on CPU!
"""

import os, sys, pickle, json
import cv2
import numpy as np

FACE_IMAGES_BASE = os.environ.get("FACE_IMAGES_DIR", r"C:\Users\sande\Downloads\face images\Images\Images")
if not os.path.exists(FACE_IMAGES_BASE):
    # Try local datasets folder fallback
    FACE_IMAGES_BASE = os.path.abspath("datasets/face images/Images/Images")

GALLERY_CACHE_DIR = os.path.abspath(".face_cache")

# Team key to folder name mapping
TEAM_FOLDER_MAP = {
    "argentina": "Argentina Players",
    "australia": "Australia Players",
    "belgium": "Belgium Players",
    "brazil": "Brazil Players",
    "cameroon": "Cameroon Players",
    "canada": "Canada Players",
    "costa rica": "Costa Rica Players",
    "croatia": "Croatia Players",
    "denmark": "Denmark Players",
    "ecuador": "Ecuador Players",
    "england": "England Players",
    "france": "France Players",
    "germany": "Germany Players",
    "ghana": "Ghana Players",
    "iran": "Iran Players",
    "japan": "Japan Players",
    "mexico": "Mexico Players",
    "morocco": "Morocco players",
    "netherlands": "Netherland Players",
    "poland": "Poland Players",
    "portugal": "Portugal Players",
    "qatar": "Qatar Players",
    "saudi arabia": "Saudi Arabia Players",
    "senegal": "Senegal Players",
    "serbia": "Serbia Players",
    "south korea": "South Korea Players",
    "spain": "Spain Players",
    "switzerland": "Switzerland Players",
    "tunisia": "Tunisia Players",
    "united states": "United States Players",
    "uruguay": "Uruguay Players",
    "wales": "Wales Players"
}

_face_app = None

def get_face_app():
    global _face_app
    if _face_app is None:
        try:
            import insightface
            from insightface.app import FaceAnalysis
            # Buffalo_l is robust and pre-trained
            _face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            _face_app.prepare(ctx_id=0, det_size=(128, 128))
            print("[DynamicFace] InsightFace loaded successfully.")
        except Exception as e:
            print(f"[DynamicFace] Error loading InsightFace: {e}")
            _face_app = "unavailable"
    return _face_app

def build_team_embeddings(team_name: str, max_imgs_per_player: int = 8) -> dict[str, np.ndarray]:
    """
    Computes face embeddings for a single team.
    Returns: dict { player_name: embedding }
    """
    team_key = team_name.strip().lower()
    folder_name = TEAM_FOLDER_MAP.get(team_key)
    if not folder_name:
        print(f"[DynamicFace] Unknown team: {team_name}")
        return {}

    # Scan groups to find the folder
    team_path = None
    for group in os.listdir(FACE_IMAGES_BASE):
        gpath = os.path.join(FACE_IMAGES_BASE, group)
        if not os.path.isdir(gpath):
            continue
        test_path = os.path.join(gpath, folder_name)
        if os.path.exists(test_path):
            team_path = test_path
            break

    if not team_path:
        print(f"[DynamicFace] Folder not found for team {team_name} in {FACE_IMAGES_BASE}")
        return {}

    face_app = get_face_app()
    if face_app == "unavailable":
        return {}

    print(f"[DynamicFace] Scanning player images for team: {team_name} in {team_path}")
    team_gallery = {}

    for player_folder in os.listdir(team_path):
        ppath = os.path.join(team_path, player_folder)
        if not os.path.isdir(ppath):
            continue

        player_name = player_folder.replace("Images_", "").replace("_", " ").strip()
        imgs = [
            os.path.join(ppath, f)
            for f in os.listdir(ppath)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ]
        
        # Limit the number of images processed per player to keep building fast (8 is plenty!)
        imgs = imgs[:max_imgs_per_player]

        embeddings = []
        for img_path in imgs:
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            if h < 64 or w < 64:
                scale = max(64/h, 64/w)
                img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
            
            try:
                faces = face_app.get(img)
                if faces:
                    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    emb = face.embedding
                    emb /= (np.linalg.norm(emb) + 1e-6)
                    embeddings.append(emb)
            except Exception:
                pass

        if embeddings:
            mean_emb = np.mean(embeddings, axis=0)
            mean_emb /= (np.linalg.norm(mean_emb) + 1e-6)
            team_gallery[player_name] = mean_emb

    print(f"[DynamicFace] Extracted embeddings for {len(team_gallery)} players of {team_name.upper()}")
    return team_gallery

def load_or_build_match_gallery(team_a: str, team_b: str) -> dict[str, np.ndarray]:
    """
    Loads match-specific face embedding gallery from cache or builds it dynamically.
    Returns: dict { "player_name::team": embedding }
    """
    os.makedirs(GALLERY_CACHE_DIR, exist_ok=True)
    
    t_a = team_a.strip().lower()
    t_b = team_b.strip().lower()
    
    match_key = f"{t_a}_vs_{t_b}"
    cache_path = os.path.join(GALLERY_CACHE_DIR, f"{match_key}.pkl")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                gallery = pickle.load(f)
            print(f"[DynamicFace] Loaded cached face gallery for match: {match_key.upper()} ({len(gallery)} players)")
            return gallery
        except Exception:
            pass

    print(f"[DynamicFace] Building dynamic face gallery for match: {match_key.upper()}...")
    
    gallery = {}
    if t_a:
        emb_a = build_team_embeddings(t_a)
        for name, emb in emb_a.items():
            gallery[f"{name}::{t_a}"] = emb
            
    if t_b:
        emb_b = build_team_embeddings(t_b)
        for name, emb in emb_b.items():
            gallery[f"{name}::{t_b}"] = emb

    if gallery:
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(gallery, f)
            print(f"[DynamicFace] Saved face gallery to cache: {cache_path}")
        except Exception as e:
            print(f"[DynamicFace] Warning: failed to cache match gallery: {e}")
            
    return gallery
