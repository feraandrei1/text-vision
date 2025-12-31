# Text Vision

AI-powered text extraction from license plates, ID cards, and documents.

## Features

- **License Plate Recognition** - Detect and read license plates from vehicle images with support for EU, US, and Asian formats
- **ID Card Extraction** - Extract text from ID cards, passports, and driver's licenses with automatic field categorization
- **Document Processing** - Extract text from PDFs and scanned documents with OCR fallback for image-based pages
- **Auto-Rotation** - Automatically detects and corrects document orientation for optimal OCR accuracy
- **GPU Acceleration** - Optional CUDA support for faster processing
- **Multiple OCR Backends** - Supports EasyOCR and PaddleOCR

## Installation

```bash
# Clone the repository
git clone https://github.com/feraandrei1/text-vision.git
cd text-vision

# Install dependencies
pip install -r requirements.txt
```

## Examples
```bash
cd examples

# License plate recognition
python basic_recognition.py

# ID card text extraction
python id_recognition.py

# PDF document processing
python pdf_extraction.py

# Video stream processing
python video_processing.py
```

## Testing

```bash
# Run all tests
pytest tests/ -v
```

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- EasyOCR (for OCR)
- PyMuPDF (for PDFs)
- Ultralytics (optional, for YOLO detection)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
