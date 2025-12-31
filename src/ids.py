"""
ID Card Recognition Module
Extracts text from ID cards, passports, driver's licenses, etc.
"""

import re
from typing import Dict, List, Optional, Union, Any
import numpy as np
import cv2

# Optional imports
EASYOCR_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except (ImportError, Exception):
    easyocr = None


class IDRecognizer:
    """ID card text extraction and recognition."""

    # Common ID field patterns
    FIELD_PATTERNS = {
        "date": r"\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}",
        "id_number": r"^[A-Z0-9]{6,}$",
        "mrz": r"^[A-Z0-9<]{30,}$",  # Machine Readable Zone
    }

    def __init__(
        self,
        languages: List[str] = None,
        use_gpu: bool = False
    ):
        self.languages = languages or ["en", "ro"]
        self.use_gpu = use_gpu
        self._ocr = None
        self._init_ocr()

    def _init_ocr(self) -> None:
        """Initialize OCR engine."""
        if EASYOCR_AVAILABLE:
            try:
                self._ocr = easyocr.Reader(self.languages, gpu=self.use_gpu)
            except Exception as e:
                print(f"Warning: Could not initialize EasyOCR: {e}")
        else:
            print("Warning: easyocr not installed. Run: pip install easyocr")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(denoised)

    def extract_text(self, image: Union[str, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Extract all text from an ID card image.

        Returns:
            List of dictionaries with text, confidence, and bbox
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError("Cannot load image")

        if self._ocr is None:
            raise RuntimeError("OCR not initialized")

        # Process both original and enhanced
        processed = self.preprocess(image)

        results_original = self._ocr.readtext(image)
        results_processed = self._ocr.readtext(processed)

        # Combine and deduplicate
        all_results = []
        seen_texts = set()

        for result in results_original + results_processed:
            bbox, text, confidence = result
            text_clean = text.strip()

            if text_clean and text_clean.lower() not in seen_texts:
                seen_texts.add(text_clean.lower())
                all_results.append({
                    "text": text_clean,
                    "confidence": confidence,
                    "bbox": bbox
                })

        # Sort by position (top to bottom, left to right)
        all_results.sort(key=lambda x: (x["bbox"][0][1], x["bbox"][0][0]))

        return all_results

    def categorize_fields(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """
        Categorize extracted text into common ID fields.

        Returns:
            Dictionary with categorized fields
        """
        categories = {
            "names": [],
            "dates": [],
            "numbers": [],
            "addresses": [],
            "other": []
        }

        for item in results:
            text = item["text"]

            # Date patterns
            if re.search(self.FIELD_PATTERNS["date"], text):
                categories["dates"].append(item)
            # ID numbers (long alphanumeric)
            elif re.search(self.FIELD_PATTERNS["id_number"], text.replace(" ", "")):
                categories["numbers"].append(item)
            # MRZ (Machine Readable Zone)
            elif re.search(self.FIELD_PATTERNS["mrz"], text.replace(" ", "")):
                categories["numbers"].append(item)
            # Addresses
            elif any(kw in text.lower() for kw in ["str", "street", "road", "ave", "blvd", "nr", "ap", "jud", "mun"]):
                categories["addresses"].append(item)
            # Names (mostly letters)
            elif re.match(r"^[A-Za-zÀ-ÿ\s\-]+$", text) and len(text) > 2:
                categories["names"].append(item)
            else:
                categories["other"].append(item)

        return categories

    def recognize(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Full ID recognition pipeline.

        Returns:
            Dictionary with all_text, categorized fields, and metadata
        """
        results = self.extract_text(image)
        categories = self.categorize_fields(results)

        return {
            "all_text": results,
            "fields": categories,
            "total_items": len(results),
            "avg_confidence": sum(r["confidence"] for r in results) / len(results) if results else 0
        }
