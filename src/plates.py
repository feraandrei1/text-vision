"""
License Plate Recognition Module
"""

import os
import re
from typing import Dict, List, Optional, Union, Any
import numpy as np
import cv2

# Optional imports
YOLO_AVAILABLE = False
EASYOCR_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except (ImportError, Exception):
    YOLO = None

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except (ImportError, Exception):
    easyocr = None


class PlateRecognizer:
    """License plate detection and recognition."""

    PLATE_PATTERNS = {
        "EU-RO": r"^[A-Z]{1,2}[-\s]?\d{2,3}[-\s]?[A-Z]{3}$",
        "EU-DE": r"^[A-Z]{1,3}[-\s]?[A-Z]{1,2}[-\s]?\d{1,4}$",
        "EU-UK": r"^[A-Z]{2}\d{2}[-\s]?[A-Z]{3}$",
        "US": r"^[A-Z0-9]{5,8}$",
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        languages: List[str] = None,
        confidence_threshold: float = 0.5,
        use_gpu: bool = False
    ):
        self.model_path = model_path
        self.languages = languages or ["en"]
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu

        self._detector = None
        self._ocr = None
        self._init_detector()
        self._init_ocr()

    def _init_detector(self) -> None:
        """Initialize YOLO detector."""
        if YOLO_AVAILABLE:
            model_path = self.model_path or "yolov8n.pt"
            try:
                self._detector = YOLO(model_path)
            except Exception as e:
                print(f"Warning: Could not load YOLO model: {e}")

    def _init_ocr(self) -> None:
        """Initialize OCR engine."""
        if EASYOCR_AVAILABLE:
            try:
                self._ocr = easyocr.Reader(self.languages, gpu=self.use_gpu)
            except Exception as e:
                print(f"Warning: Could not initialize EasyOCR: {e}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(denoised)

    def detect(self, image: Union[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Detect license plates in image."""
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Cannot load image")

        plates = []

        if self._detector is not None:
            results = self._detector(image, conf=self.confidence_threshold)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        plates.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": conf,
                            "crop": image[y1:y2, x1:x2].copy()
                        })
        else:
            plates = self._fallback_detection(image)

        return plates

    def _fallback_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback detection using edge detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plates = []

        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            if 2 < aspect_ratio < 6:
                plates.append({
                    "bbox": [x, y, x + w, y + h],
                    "confidence": 0.5,
                    "crop": image[y:y+h, x:x+w].copy()
                })

        return plates[:5]

    def extract_text(self, plate_image: np.ndarray) -> Dict[str, Any]:
        """Extract text from plate image."""
        processed = self.preprocess(plate_image)
        text = ""
        confidence = 0.0

        if self._ocr is not None:
            results = self._ocr.readtext(processed)
            if results:
                texts = [r[1] for r in results]
                confidences = [r[2] for r in results]
                text = " ".join(texts)
                confidence = sum(confidences) / len(confidences) if confidences else 0

        text = self._clean_text(text)
        return {"text": text, "confidence": confidence}

    def _clean_text(self, text: str) -> str:
        """Clean plate text."""
        text = text.upper()
        text = re.sub(r"[^A-Z0-9\s-]", "", text)
        text = re.sub(r"\s+", "-", text.strip())
        return text

    def validate(self, text: str) -> Dict[str, Any]:
        """Validate plate format."""
        for region, pattern in self.PLATE_PATTERNS.items():
            if re.match(pattern, text):
                return {"valid": True, "region": region}
        return {"valid": False, "region": "UNKNOWN"}

    def recognize(self, image: Union[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Full plate recognition pipeline."""
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError("Cannot load image")

        detected = self.detect(image)
        results = []

        for plate in detected:
            ocr_result = self.extract_text(plate["crop"])
            validation = self.validate(ocr_result["text"])

            results.append({
                "text": ocr_result["text"],
                "confidence": (plate["confidence"] + ocr_result["confidence"]) / 2,
                "bbox": plate["bbox"],
                "region": validation["region"],
                "valid_format": validation["valid"]
            })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results
