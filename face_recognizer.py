"""
face_recognizer.py
------------------
Runtime ArcFace face recognition module.
Loads the pre-built gallery (face_gallery.pkl) and matches player head crops
from the video against it using cosine similarity.

Usage:
    from face_recognizer import FaceRecognizer
    recognizer = FaceRecognizer()
    result = recognizer.identify(frame, bbox, team_name, track_id)
    # result: {"name": "Neymar", "team": "brazil", "confidence": 0.83} or None
"""

import os, sys, pickle
import cv2
import numpy as np

from dynamic_face_gallery import load_or_build_match_gallery, get_face_app

# Confidence threshold — below this we don't assign a name
MATCH_THRESHOLD = 0.55


class FaceRecognizer:
    """
    Identifies players from video frames using ArcFace embeddings.
    Per-track cache ensures each track ID is identified once and locked in.
    """

    def __init__(self, team_filter: str = "", active_match: tuple = None):
        """
        Args:
            team_filter: If set (e.g. "brazil"), only match against players
                         from that team — improves speed and accuracy.
            active_match: (team_a, team_b) tuple to build/load dynamic gallery.
        """
        self.team_filter = team_filter.strip().lower() if team_filter else ""
        self._track_cache: dict[int, dict | None] = {}
        
        # Load or dynamically build match face gallery
        if active_match:
            self._gallery = load_or_build_match_gallery(active_match[0], active_match[1])
        else:
            self._gallery = {}
            
        self._face_app = get_face_app()

        # Build filtered gallery subset
        if self._gallery and self.team_filter:
            self._filtered_keys = [
                k for k in self._gallery
                if k.split("::")[1] == self.team_filter
            ]
        elif self._gallery:
            self._filtered_keys = list(self._gallery.keys())
        else:
            self._filtered_keys = []

    def identify(
        self,
        frame: np.ndarray,
        bbox: list | tuple,
        track_id: int = -1,
    ) -> dict | None:
        """
        Given a full video frame and a player bbox, returns:
            {"name": str, "team": str, "confidence": float}
        or None if unidentifiable.

        Results are cached per track_id.
        """
        if track_id >= 0 and track_id in self._track_cache:
            return self._track_cache[track_id]

        if not self._filtered_keys or self._face_app == "unavailable":
            return None

        result = self._run_match(frame, bbox)

        # Cache result (even None, to avoid re-running on every frame)
        if track_id >= 0:
            self._track_cache[track_id] = result

        return result

    def _run_match(self, frame: np.ndarray, bbox: list | tuple) -> dict | None:
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        w = x2 - x1

        if h < 30 or w < 15:
            return None

        # Crop head region: top 35% of the bbox
        head_y2 = y1 + int(h * 0.38)
        head_crop = frame[y1:head_y2, x1:x2]
        if head_crop.size == 0:
            return None

        # Upscale small crops
        ch, cw = head_crop.shape[:2]
        if ch < 64 or cw < 64:
            scale = max(64/ch, 64/cw)
            head_crop = cv2.resize(head_crop, (int(cw*scale), int(ch*scale)),
                                   interpolation=cv2.INTER_LINEAR)

        # Extract embedding
        try:
            faces = self._face_app.get(head_crop)
            if not faces:
                return None
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            emb = face.embedding
            emb = emb / (np.linalg.norm(emb) + 1e-6)
        except Exception:
            return None

        # Cosine similarity against gallery
        gallery = self._gallery
        best_key = None
        best_sim = -1.0

        for key in self._filtered_keys:
            g_emb = gallery[key]
            sim = float(np.dot(emb, g_emb))
            if sim > best_sim:
                best_sim = sim
                best_key = key

        if best_key is None or best_sim < MATCH_THRESHOLD:
            return None

        name, team = best_key.split("::")
        return {"name": name, "team": team, "confidence": best_sim}

    def set_team_filter(self, team_name: str):
        """Update team filter mid-run (e.g. when team context changes)."""
        self.team_filter = team_name.strip().lower()
        if self._gallery:
            self._filtered_keys = [
                k for k in self._gallery
                if k.split("::")[1] == self.team_filter
            ] if self.team_filter else list(self._gallery.keys())

    def reset(self):
        """Clear per-shot cache."""
        self._track_cache.clear()
