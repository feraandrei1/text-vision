"""
PlateVision - Main ALPR class
"""

import os
import re
from typing import Dict, List, Optional, Union, Any, Iterator, Tuple
import numpy as np
import cv2

# Optional imports
YOLO_AVAILABLE = False
EASYOCR_AVAILABLE = False
PADDLEOCR_AVAILABLE = False

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

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except (ImportError, Exception):
    PaddleOCR = None


class PlateVision:
    """
    Automatic License Plate Recognition system.

    Combines YOLO detection with OCR for complete plate recognition.
    """

    DETECTORS = ["yolov8", "yolov5", "ssd"]
    OCR_BACKENDS = ["easyocr", "paddleocr", "tesseract"]
    REGIONS = ["EU", "US", "ASIA", "AUTO"]

    # Plate format patterns
    PLATE_PATTERNS = {
        "EU-RO": r"^[A-Z]{1,2}[-\s]?\d{2,3}[-\s]?[A-Z]{3}$",
        "EU-DE": r"^[A-Z]{1,3}[-\s]?[A-Z]{1,2}[-\s]?\d{1,4}$",
        "EU-UK": r"^[A-Z]{2}\d{2}[-\s]?[A-Z]{3}$",
        "US": r"^[A-Z0-9]{5,8}$",
    }

    def __init__(
        self,
        detector: str = "yolov8",
        ocr_backend: str = "easyocr",
        languages: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
        ocr_confidence: float = 0.6,
        use_gpu: bool = False,
        plate_region: str = "AUTO",
        model_path: Optional[str] = None
    ):
        """
        Initialize PlateVision.

        Args:
            detector: Detection model (yolov8, yolov5, ssd)
            ocr_backend: OCR engine (easyocr, paddleocr, tesseract)
            languages: OCR languages (default: ["en"])
            confidence_threshold: Detection confidence threshold
            ocr_confidence: OCR confidence threshold
            use_gpu: Enable GPU acceleration
            plate_region: Plate format region (EU, US, ASIA, AUTO)
            model_path: Custom model weights path
        """
        if detector not in self.DETECTORS:
            raise ValueError(f"Invalid detector. Choose from: {self.DETECTORS}")
        if ocr_backend not in self.OCR_BACKENDS:
            raise ValueError(f"Invalid OCR backend. Choose from: {self.OCR_BACKENDS}")
        if plate_region not in self.REGIONS:
            raise ValueError(f"Invalid region. Choose from: {self.REGIONS}")

        self.detector_name = detector
        self.ocr_backend_name = ocr_backend
        self.languages = languages or ["en"]
        self.confidence_threshold = confidence_threshold
        self.ocr_confidence = ocr_confidence
        self.use_gpu = use_gpu
        self.plate_region = plate_region
        self.model_path = model_path

        self._detector = None
        self._ocr = None
        self._init_detector()
        self._init_ocr()

    def _init_detector(self) -> None:
        """Initialize plate detection model."""
        if self.detector_name == "yolov8":
            if YOLO_AVAILABLE:
                model_path = self.model_path or "yolov8n.pt"
                try:
                    self._detector = YOLO(model_path)
                except Exception as e:
                    print(f"Warning: Could not load YOLO model: {e}")
                    self._detector = None
            else:
                print("Warning: ultralytics not installed. Using fallback detector.")
                self._detector = None

    def _init_ocr(self) -> None:
        """Initialize OCR engine."""
        if self.ocr_backend_name == "easyocr":
            if EASYOCR_AVAILABLE:
                try:
                    self._ocr = easyocr.Reader(
                        self.languages,
                        gpu=self.use_gpu
                    )
                except Exception as e:
                    print(f"Warning: Could not initialize EasyOCR: {e}")
                    self._ocr = None
            else:
                print("Warning: easyocr not installed.")
                self._ocr = None
        elif self.ocr_backend_name == "paddleocr":
            if PADDLEOCR_AVAILABLE:
                try:
                    self._ocr = PaddleOCR(
                        use_angle_cls=True,
                        lang=self.languages[0] if self.languages else "en",
                        use_gpu=self.use_gpu
                    )
                except Exception as e:
                    print(f"Warning: Could not initialize PaddleOCR: {e}")
                    self._ocr = None
            else:
                print("Warning: paddleocr not installed.")
                self._ocr = None

    def preprocess_image(
        self,
        image: np.ndarray,
        enhance: bool = True
    ) -> np.ndarray:
        """
        Preprocess image for better detection/OCR.

        Args:
            image: Input image
            enhance: Apply enhancement

        Returns:
            Preprocessed image
        """
        if enhance:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Denoise
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

            # Increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)

            return enhanced
        return image

    def detect_plates(
        self,
        image: Union[str, np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        Detect license plate regions in an image.

        Args:
            image: Image path or numpy array

        Returns:
            List of detected plate regions
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Cannot load image: {image}")

        plates = []

        if self._detector is not None:
            # Use YOLO detector
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
            # Fallback: use cascade classifier or edge detection
            plates = self._fallback_detection(image)

        return plates

    def _fallback_detection(
        self,
        image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Fallback plate detection using traditional CV methods.

        Args:
            image: Input image

        Returns:
            List of detected plates
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plates = []

        # Edge detection
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            # Plate aspect ratio typically between 2 and 6
            if 2 < aspect_ratio < 6:
                plates.append({
                    "bbox": [x, y, x + w, y + h],
                    "confidence": 0.5,
                    "crop": image[y:y+h, x:x+w].copy()
                })

        return plates[:5]  # Return top 5 candidates

    def extract_text(
        self,
        plate_image: np.ndarray,
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """
        Extract text from a cropped plate image.

        Args:
            plate_image: Cropped plate image
            preprocess: Apply preprocessing

        Returns:
            Dictionary with extracted text and confidence
        """
        if preprocess:
            plate_image = self.preprocess_image(plate_image)

        text = ""
        confidence = 0.0

        if self._ocr is not None:
            if self.ocr_backend_name == "easyocr":
                results = self._ocr.readtext(plate_image)
                if results:
                    texts = [r[1] for r in results]
                    confidences = [r[2] for r in results]
                    text = " ".join(texts)
                    confidence = sum(confidences) / len(confidences) if confidences else 0

            elif self.ocr_backend_name == "paddleocr":
                results = self._ocr.ocr(plate_image, cls=True)
                if results and results[0]:
                    texts = [line[1][0] for line in results[0]]
                    confidences = [line[1][1] for line in results[0]]
                    text = " ".join(texts)
                    confidence = sum(confidences) / len(confidences) if confidences else 0

        # Clean up text
        text = self._clean_plate_text(text)

        return {
            "text": text,
            "confidence": confidence,
            "raw_text": text
        }

    def _clean_plate_text(self, text: str) -> str:
        """
        Clean and normalize plate text.

        Args:
            text: Raw OCR text

        Returns:
            Cleaned plate text
        """
        # Remove unwanted characters
        text = text.upper()
        text = re.sub(r"[^A-Z0-9\s-]", "", text)
        text = re.sub(r"\s+", "-", text.strip())
        return text

    def validate_plate(
        self,
        text: str,
        region: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate plate format against known patterns.

        Args:
            text: Plate text
            region: Specific region to validate (optional)

        Returns:
            Validation result
        """
        if region and region in self.PLATE_PATTERNS:
            pattern = self.PLATE_PATTERNS[region]
            is_valid = bool(re.match(pattern, text))
            return {
                "valid": is_valid,
                "region": region,
                "format_matched": is_valid
            }

        # Auto-detect region
        for reg, pattern in self.PLATE_PATTERNS.items():
            if re.match(pattern, text):
                return {
                    "valid": True,
                    "region": reg,
                    "format_matched": True
                }

        return {
            "valid": False,
            "region": "UNKNOWN",
            "format_matched": False
        }

    def recognize(
        self,
        image: Union[str, np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        Full plate recognition pipeline.

        Args:
            image: Image path or numpy array

        Returns:
            List of recognized plates with text and metadata
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Cannot load image")

        # Detect plates
        detected = self.detect_plates(image)
        results = []

        for plate in detected:
            # Extract text
            ocr_result = self.extract_text(plate["crop"])

            # Validate format
            validation = self.validate_plate(ocr_result["text"], self.plate_region)

            results.append({
                "text": ocr_result["text"],
                "confidence": (plate["confidence"] + ocr_result["confidence"]) / 2,
                "detection_confidence": plate["confidence"],
                "ocr_confidence": ocr_result["confidence"],
                "bbox": plate["bbox"],
                "region": validation["region"],
                "valid_format": validation["valid"]
            })

        # Sort by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def process_stream(
        self,
        stream,
        skip_frames: int = 0
    ) -> Iterator[Tuple[np.ndarray, List[Dict]]]:
        """
        Process video stream for plate recognition.

        Args:
            stream: VideoStream object
            skip_frames: Number of frames to skip between processing

        Yields:
            Tuple of (frame, detected_plates)
        """
        frame_count = 0

        for frame in stream:
            frame_count += 1

            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                yield frame, []
                continue

            plates = self.recognize(frame)
            yield frame, plates

    def batch_process(
        self,
        directory: str,
        extensions: Optional[List[str]] = None
    ) -> Dict[str, List[Dict]]:
        """
        Process all images in a directory.

        Args:
            directory: Path to image directory
            extensions: File extensions to process

        Returns:
            Dictionary mapping filenames to results
        """
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        results = {}

        for filename in os.listdir(directory):
            if any(filename.lower().endswith(ext) for ext in extensions):
                filepath = os.path.join(directory, filename)
                try:
                    plates = self.recognize(filepath)
                    results[filename] = plates
                except Exception as e:
                    results[filename] = [{"error": str(e)}]

        return results

    def run_realtime(
        self,
        source: Union[int, str] = 0,
        show: bool = True
    ) -> None:
        """
        Run real-time plate recognition.

        Args:
            source: Video source (0 for webcam)
            show: Display video window
        """
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        print("Starting real-time recognition. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Recognize plates
            plates = self.recognize(frame)

            # Draw results
            for plate in plates:
                x1, y1, x2, y2 = plate["bbox"]
                text = plate["text"]
                conf = plate["confidence"]

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                label = f"{text} ({conf:.0%})"
                cv2.putText(
                    frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

            if show:
                cv2.imshow("PlateVision", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def draw_results(
        image: np.ndarray,
        plates: List[Dict],
        color: tuple = (0, 255, 0)
    ) -> np.ndarray:
        """
        Draw detection results on image.

        Args:
            image: Input image
            plates: List of plate detection results
            color: Box color (BGR)

        Returns:
            Image with drawn results
        """
        result = image.copy()

        for plate in plates:
            x1, y1, x2, y2 = plate["bbox"]
            text = plate.get("text", "")
            conf = plate.get("confidence", 0)

            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            label = f"{text} ({conf:.0%})"
            cv2.putText(
                result, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

        return result
