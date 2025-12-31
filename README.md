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

### Basic Installation

```bash
pip install -r requirements.txt
```

### Minimal Installation (License Plates Only)

```bash
pip install opencv-python numpy easyocr
```

### Full Installation (All Features)

```bash
pip install opencv-python numpy easyocr pymupdf ultralytics torch
```

### GPU Support

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Unified API

```python
from src import TextIdentifier

# Initialize with desired languages
ti = TextIdentifier(languages=["en"])

# Auto-detect file type and extract text
result = ti.recognize("document.pdf")
result = ti.recognize("id_card.jpg", mode="id")
result = ti.recognize("car.jpg", mode="plate")

# Batch process a directory
results = ti.batch_process("./images/", mode="plate")
```

### License Plate Recognition

```python
from src import PlateRecognizer

# Initialize
plates = PlateRecognizer(languages=["en"], use_gpu=False)

# Recognize plates in image
results = plates.recognize("car_image.jpg")

for plate in results:
    print(f"Plate: {plate['text']}")
    print(f"Confidence: {plate['confidence']:.2%}")
    print(f"Region: {plate['region']}")  # EU-RO, EU-UK, US, etc.
```

### ID Card Text Extraction

```python
from src import IDRecognizer

# Initialize with multiple languages
ids = IDRecognizer(languages=["en", "ro"])

# Extract text from ID card
result = ids.recognize("id_card.jpg")

print(f"Found {result['total_items']} text regions")
print(f"Average confidence: {result['avg_confidence']:.2%}")

# Access categorized fields
for name in result['fields']['names']:
    print(f"Name: {name['text']}")
for date in result['fields']['dates']:
    print(f"Date: {date['text']}")
```

### PDF Document Processing

```python
from src import DocumentReader

# Initialize
docs = DocumentReader(languages=["en"])

# Extract text from PDF
result = docs.extract_from_pdf("document.pdf")

print(f"Pages: {result['total_pages']}")
print(f"Title: {result['metadata']['title']}")
print(result['full_text'])

# Search within PDF
matches = docs.search_text("document.pdf", "keyword")
for match in matches:
    print(f"Page {match['page']}: {match['context']}")
```

### Video/Webcam Processing

```python
from src import PlateRecognizer, VideoStream

pv = PlateRecognizer()

# Process webcam feed (press 'q' to quit)
with VideoStream(source=0) as stream:
    for frame in stream:
        plates = pv.recognize(frame)
        # Draw results and display
        stream.show(frame)
```

## Command Line Usage

```bash
# Live webcam demo
python main.py

# Process single image
python main.py --image path/to/car.jpg

# Process folder of images
python main.py --folder path/to/images/
```

## Supported Formats

### License Plate Regions
- `EU-RO` - Romanian (e.g., B-123-ABC)
- `EU-DE` - German (e.g., M-AB-1234)
- `EU-UK` - United Kingdom (e.g., AB12CDE)
- `US` - United States (e.g., ABC1234)
- `AUTO` - Automatic detection

### File Formats
- **Images**: JPG, JPEG, PNG, BMP, TIFF
- **Documents**: PDF
- **Video**: MP4, AVI, MOV, webcam

### Languages
Supports all languages available in EasyOCR including: English, Romanian, German, French, Spanish, Italian, Chinese, Japanese, Korean, and many more.

## Project Structure

```
text-vision/
├── src/
│   ├── __init__.py          # Package exports
│   ├── text_identifier.py   # Unified API
│   ├── plates.py            # License plate recognition
│   ├── ids.py               # ID card extraction
│   ├── documents.py         # PDF processing
│   ├── platevision.py       # Legacy ALPR (full features)
│   └── video_stream.py      # Video utilities
├── tests/                   # Unit tests
├── examples/                # Usage examples
├── models/                  # Custom models (gitignored)
└── storage/                 # Test data (gitignored)
```

## Examples

Run the example scripts:

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

## API Reference

### TextIdentifier

Main unified interface for all recognition modes.

| Method | Description |
|--------|-------------|
| `recognize(file, mode="auto")` | Auto-detect and extract text |
| `recognize_plate(image)` | License plate recognition |
| `recognize_id(image)` | ID card text extraction |
| `recognize_document(file)` | PDF/document processing |
| `batch_process(dir, mode)` | Process all files in directory |

### PlateRecognizer

| Method | Description |
|--------|-------------|
| `recognize(image)` | Full recognition pipeline |
| `detect(image)` | Detect plate regions only |
| `extract_text(plate_image)` | OCR on cropped plate |
| `validate(text)` | Validate plate format |

### IDRecognizer

| Method | Description |
|--------|-------------|
| `recognize(image)` | Full ID recognition |
| `extract_text(image)` | Raw text extraction |
| `categorize_fields(results)` | Categorize into names, dates, etc. |

### DocumentReader

| Method | Description |
|--------|-------------|
| `recognize(file)` | Process PDF or image |
| `extract_from_pdf(path)` | PDF text extraction |
| `extract_from_image(path)` | Image OCR |
| `search_text(pdf, query)` | Search within PDF |
| `extract_tables(pdf)` | Basic table extraction |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=src tests/
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
