"""
Document/PDF Text Extraction Module
Extracts text from PDF files and scanned documents.
"""

import os
from typing import Dict, List, Optional, Union, Any
import re

# Optional imports
PDF_AVAILABLE = False
EASYOCR_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    fitz = None

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None

try:
    import numpy as np
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class DocumentReader:
    """PDF and document text extraction."""

    def __init__(
        self,
        languages: List[str] = None,
        use_gpu: bool = False,
        ocr_for_images: bool = True
    ):
        """
        Initialize DocumentReader.

        Args:
            languages: OCR languages for scanned pages
            use_gpu: Enable GPU for OCR
            ocr_for_images: Use OCR for image-based PDFs
        """
        self.languages = languages or ["en"]
        self.use_gpu = use_gpu
        self.ocr_for_images = ocr_for_images
        self._ocr = None

        if not PDF_AVAILABLE:
            print("Warning: PyMuPDF not installed. Run: pip install pymupdf")

        if ocr_for_images:
            self._init_ocr()

    def _init_ocr(self) -> None:
        """Initialize OCR for scanned documents."""
        if EASYOCR_AVAILABLE:
            try:
                self._ocr = easyocr.Reader(self.languages, gpu=self.use_gpu)
            except Exception as e:
                print(f"Warning: Could not initialize EasyOCR: {e}")

    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract all text from a PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with pages, text, and metadata
        """
        if not PDF_AVAILABLE:
            raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        result = {
            "filename": os.path.basename(pdf_path),
            "total_pages": len(doc),
            "pages": [],
            "full_text": "",
            "metadata": {}
        }

        # Extract metadata
        result["metadata"] = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", ""),
            "creation_date": doc.metadata.get("creationDate", ""),
        }

        all_text = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()

            # If no text found and OCR enabled, try OCR
            if not page_text.strip() and self.ocr_for_images and self._ocr:
                page_text = self._ocr_page(page)

            page_data = {
                "page_number": page_num + 1,
                "text": page_text,
                "word_count": len(page_text.split()),
                "has_images": len(page.get_images()) > 0
            }

            result["pages"].append(page_data)
            all_text.append(page_text)

        result["full_text"] = "\n\n".join(all_text)
        doc.close()

        return result

    def _ocr_page(self, page) -> str:
        """OCR a PDF page that contains only images."""
        if not CV2_AVAILABLE or not self._ocr:
            return ""

        # Render page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
        img_data = pix.tobytes("png")

        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run OCR
        results = self._ocr.readtext(img)
        texts = [r[1] for r in results]

        return " ".join(texts)

    def extract_from_image(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text from a scanned document image.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with extracted text and metadata
        """
        if not EASYOCR_AVAILABLE or not self._ocr:
            raise RuntimeError("EasyOCR not available for image OCR")

        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV not available")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        # Preprocess
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # OCR
        results = self._ocr.readtext(denoised)

        text_items = []
        for bbox, text, confidence in results:
            text_items.append({
                "text": text,
                "confidence": confidence,
                "bbox": bbox
            })

        # Sort by position
        text_items.sort(key=lambda x: (x["bbox"][0][1], x["bbox"][0][0]))

        full_text = " ".join(item["text"] for item in text_items)

        return {
            "filename": os.path.basename(image_path),
            "text_items": text_items,
            "full_text": full_text,
            "total_items": len(text_items),
            "avg_confidence": sum(r["confidence"] for r in text_items) / len(text_items) if text_items else 0
        }

    def extract_tables(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF (basic implementation).

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of detected tables with their content
        """
        if not PDF_AVAILABLE:
            raise RuntimeError("PyMuPDF not installed")

        doc = fitz.open(pdf_path)
        tables = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Find table-like structures (rows of text with similar x-positions)
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" in block:
                    lines = block["lines"]
                    if len(lines) >= 2:
                        # Check if lines have similar structure (potential table)
                        tables.append({
                            "page": page_num + 1,
                            "lines": len(lines),
                            "content": [[span["text"] for span in line["spans"]] for line in lines]
                        })

        doc.close()
        return tables

    def search_text(self, pdf_path: str, query: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """
        Search for text in PDF.

        Args:
            pdf_path: Path to PDF file
            query: Text to search for
            case_sensitive: Case sensitive search

        Returns:
            List of matches with page numbers and context
        """
        if not PDF_AVAILABLE:
            raise RuntimeError("PyMuPDF not installed")

        doc = fitz.open(pdf_path)
        matches = []

        flags = 0 if case_sensitive else re.IGNORECASE

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            for match in re.finditer(re.escape(query), text, flags):
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].replace("\n", " ")

                matches.append({
                    "page": page_num + 1,
                    "position": match.start(),
                    "context": f"...{context}..."
                })

        doc.close()
        return matches

    def recognize(self, file_path: str) -> Dict[str, Any]:
        """
        Main recognition method - handles both PDFs and images.

        Args:
            file_path: Path to PDF or image file

        Returns:
            Extracted text and metadata
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self.extract_from_pdf(file_path)
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
            return self.extract_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
