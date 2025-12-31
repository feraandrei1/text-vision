"""
TextIdentifier - Unified Text Identification System
Combines license plate, ID card, and document recognition.
"""

from typing import Dict, List, Optional, Union, Any
import os

from .plates import PlateRecognizer
from .ids import IDRecognizer
from .documents import DocumentReader


class TextIdentifier:
    """
    Unified text identification system.

    Supports:
    - License plate recognition from vehicle images
    - ID card/passport text extraction
    - PDF and document text extraction
    """

    def __init__(
        self,
        languages: List[str] = None,
        use_gpu: bool = False,
        plate_model_path: Optional[str] = None
    ):
        """
        Initialize TextIdentifier.

        Args:
            languages: OCR languages (default: ["en"])
            use_gpu: Enable GPU acceleration
            plate_model_path: Custom YOLO model for plate detection
        """
        self.languages = languages or ["en"]
        self.use_gpu = use_gpu
        self.plate_model_path = plate_model_path

        # Lazy initialization - modules loaded on first use
        self._plate_recognizer = None
        self._id_recognizer = None
        self._document_reader = None

    @property
    def plates(self) -> PlateRecognizer:
        """Get or initialize plate recognizer."""
        if self._plate_recognizer is None:
            self._plate_recognizer = PlateRecognizer(
                model_path=self.plate_model_path,
                languages=self.languages,
                use_gpu=self.use_gpu
            )
        return self._plate_recognizer

    @property
    def ids(self) -> IDRecognizer:
        """Get or initialize ID recognizer."""
        if self._id_recognizer is None:
            self._id_recognizer = IDRecognizer(
                languages=self.languages,
                use_gpu=self.use_gpu
            )
        return self._id_recognizer

    @property
    def documents(self) -> DocumentReader:
        """Get or initialize document reader."""
        if self._document_reader is None:
            self._document_reader = DocumentReader(
                languages=self.languages,
                use_gpu=self.use_gpu
            )
        return self._document_reader

    def recognize_plate(self, image) -> List[Dict[str, Any]]:
        """
        Recognize license plates in an image.

        Args:
            image: Image path or numpy array

        Returns:
            List of detected plates with text and confidence
        """
        return self.plates.recognize(image)

    def recognize_id(self, image) -> Dict[str, Any]:
        """
        Extract text from an ID card image.

        Args:
            image: Image path or numpy array

        Returns:
            Dictionary with extracted text and categorized fields
        """
        return self.ids.recognize(image)

    def recognize_document(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from a PDF or scanned document.

        Args:
            file_path: Path to PDF or image file

        Returns:
            Dictionary with extracted text and metadata
        """
        return self.documents.recognize(file_path)

    def recognize(self, file_path: str, mode: str = "auto") -> Dict[str, Any]:
        """
        Auto-detect file type and extract text.

        Args:
            file_path: Path to file
            mode: Recognition mode ("auto", "plate", "id", "document")

        Returns:
            Extracted text and metadata
        """
        ext = os.path.splitext(file_path)[1].lower()

        if mode == "auto":
            if ext == ".pdf":
                mode = "document"
            elif ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                # Default to ID recognition for images (most general)
                mode = "id"

        if mode == "plate":
            results = self.recognize_plate(file_path)
            return {
                "mode": "plate",
                "results": results,
                "count": len(results)
            }
        elif mode == "id":
            results = self.recognize_id(file_path)
            return {
                "mode": "id",
                "results": results
            }
        elif mode == "document":
            results = self.recognize_document(file_path)
            return {
                "mode": "document",
                "results": results
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def batch_process(
        self,
        directory: str,
        mode: str = "auto",
        extensions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Process all files in a directory.

        Args:
            directory: Path to directory
            mode: Recognition mode
            extensions: File extensions to process

        Returns:
            Dictionary mapping filenames to results
        """
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".bmp", ".pdf"]

        results = {}

        for filename in os.listdir(directory):
            if any(filename.lower().endswith(ext) for ext in extensions):
                filepath = os.path.join(directory, filename)
                try:
                    results[filename] = self.recognize(filepath, mode)
                except Exception as e:
                    results[filename] = {"error": str(e)}

        return results
