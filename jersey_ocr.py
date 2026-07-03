"""
jersey_ocr.py
-------------
Reads jersey numbers from player bounding-box crops using easyocr.
Falls back gracefully to None when the number is unreadable.

Usage:
    from jersey_ocr import JerseyOCR
    ocr = JerseyOCR()
    number = ocr.read_jersey_number(frame, bbox)  # returns int or None
"""

import cv2
import numpy as np
import re

# Lazy-load easyocr to avoid import errors if not installed
_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        try:
            import easyocr
            _reader = easyocr.Reader(['en'], gpu=True, verbose=False)
        except ImportError:
            print("[JerseyOCR] Warning: easyocr not installed. "
                  "Run: pip install easyocr  — falling back to no-OCR mode.")
            _reader = "unavailable"
    return _reader


class JerseyOCR:
    """
    Lightweight wrapper that crops a jersey patch from a player bounding box
    and uses easyocr to extract the number printed on it.
    """

    def __init__(self):
        # Cache per track_id: once we've seen a reliable number, lock it in
        self._track_cache: dict[int, int] = {}

    def read_jersey_number(
        self,
        frame: np.ndarray,
        bbox: list | tuple,
        track_id: int = -1,
    ) -> int | None:
        """
        Given a full video frame and a player bbox [x1, y1, x2, y2],
        returns the integer jersey number or None if unreadable.

        Results are cached per track_id so we only call OCR until we
        get a confident reading for each player.
        """
        # Return cached value if already locked in
        if track_id >= 0 and track_id in self._track_cache:
            return self._track_cache[track_id]

        reader = _get_reader()
        if reader == "unavailable":
            return None

        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        w = x2 - x1

        if h < 20 or w < 10:
            return None

        # Crop the chest/back area — center 50% horizontally, 
        # middle 25-60% vertically (where the number is printed)
        cx1 = x1 + int(w * 0.15)
        cx2 = x1 + int(w * 0.85)
        cy1 = y1 + int(h * 0.25)
        cy2 = y1 + int(h * 0.65)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        # Preprocess for better OCR accuracy
        crop_prepared = self._preprocess(crop)

        try:
            results = reader.readtext(crop_prepared, allowlist='0123456789', detail=1)
        except Exception:
            return None

        # Parse all candidate numbers from results
        best_num = None
        best_conf = 0.0
        for (_, text, conf) in results:
            # Only accept 1-2 digit strings (jersey numbers 1-99)
            cleaned = re.sub(r'\D', '', text)
            if cleaned and 1 <= len(cleaned) <= 2:
                num = int(cleaned)
                if 1 <= num <= 99 and conf > best_conf:
                    best_num = num
                    best_conf = conf

        # Lock in if confidence is high enough
        if best_num is not None and best_conf >= 0.55 and track_id >= 0:
            self._track_cache[track_id] = best_num

        return best_num

    @staticmethod
    def _preprocess(crop: np.ndarray) -> np.ndarray:
        """
        Enhances the crop for digit OCR:
        1. Upscale 3× so digits are large enough for the OCR model
        2. Grayscale → CLAHE equalization → Otsu threshold
        3. Return single-channel binary image
        """
        scale = 3
        up = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                        interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)

        # CLAHE for local contrast normalization (handles bright / dark jerseys)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        eq = clahe.apply(gray)

        # Otsu threshold — works for both white-on-dark and dark-on-white jerseys
        _, binary = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def reset(self):
        """Clear per-shot cache (call at the start of each new shot)."""
        self._track_cache.clear()
