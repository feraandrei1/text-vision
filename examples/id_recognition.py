"""
PlateVision - ID Card Text Recognition Example
Extracts all text from ID cards, passports, driver's licenses, etc.
"""

import sys
import os
sys.path.insert(0, '..')

import cv2
import numpy as np

# Check for easyocr
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Error: easyocr not installed. Run: pip install easyocr")
    sys.exit(1)

# Folder for test ID images (not uploaded to GitHub)
TEST_IDS_DIR = "../storage/test_ids"


def preprocess_image(image):
    """Preprocess image for better OCR results."""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    return enhanced


def extract_text_from_id(image_path, reader, languages=['en']):
    """
    Extract all text from an ID card image.

    Args:
        image_path: Path to the ID image
        reader: EasyOCR reader instance
        languages: List of languages to detect

    Returns:
        List of dictionaries with text, confidence, and position
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    # Preprocess for better OCR
    processed = preprocess_image(image)

    # Run OCR on both original and processed
    results_original = reader.readtext(image)
    results_processed = reader.readtext(processed)

    # Combine and deduplicate results
    all_results = []
    seen_texts = set()

    for result in results_original + results_processed:
        bbox, text, confidence = result
        text_clean = text.strip()

        if text_clean and text_clean.lower() not in seen_texts:
            seen_texts.add(text_clean.lower())
            all_results.append({
                'text': text_clean,
                'confidence': confidence,
                'bbox': bbox
            })

    # Sort by vertical position (top to bottom), then horizontal (left to right)
    all_results.sort(key=lambda x: (x['bbox'][0][1], x['bbox'][0][0]))

    return all_results


def categorize_id_fields(results):
    """
    Try to categorize extracted text into common ID fields.

    Args:
        results: List of OCR results

    Returns:
        Dictionary with categorized fields
    """
    categories = {
        'names': [],
        'dates': [],
        'numbers': [],
        'addresses': [],
        'other': []
    }

    import re

    for item in results:
        text = item['text']

        # Date patterns (DD/MM/YYYY, DD.MM.YYYY, DD-MM-YYYY, etc.)
        if re.search(r'\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}', text):
            categories['dates'].append(item)
        # ID/Document numbers (alphanumeric sequences)
        elif re.search(r'^[A-Z0-9]{6,}$', text.replace(' ', '')):
            categories['numbers'].append(item)
        # Addresses (contains street keywords)
        elif any(kw in text.lower() for kw in ['str', 'street', 'road', 'ave', 'blvd', 'nr', 'ap']):
            categories['addresses'].append(item)
        # Names (mostly letters, possibly with spaces)
        elif re.match(r'^[A-Za-zÀ-ÿ\s\-]+$', text) and len(text) > 2:
            categories['names'].append(item)
        else:
            categories['other'].append(item)

    return categories


def main():
    print("=" * 60)
    print("ID Card Text Recognition")
    print("=" * 60)

    # Create test_ids folder if it doesn't exist
    if not os.path.exists(TEST_IDS_DIR):
        os.makedirs(TEST_IDS_DIR)
        print(f"\nCreated folder: {TEST_IDS_DIR}/")

    # Get all images from test_ids folder
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(TEST_IDS_DIR) if f.lower().endswith(extensions)]

    if not images:
        print(f"\nNo images found in {TEST_IDS_DIR}/")
        print("Add your ID card images there and run again.")
        print("\nSupported formats: JPG, JPEG, PNG, BMP")
        return

    print(f"\nFound {len(images)} image(s) to process.")
    print("Initializing OCR engine...")

    # Initialize EasyOCR with multiple languages
    # Add more languages as needed: 'ro' for Romanian, 'de' for German, etc.
    reader = easyocr.Reader(['en', 'ro'], gpu=False)

    for image_name in images:
        image_path = os.path.join(TEST_IDS_DIR, image_name)

        print("\n" + "=" * 60)
        print(f"Processing: {image_name}")
        print("=" * 60)

        try:
            # Extract all text
            results = extract_text_from_id(image_path, reader)

            if not results:
                print("  No text detected in the image.")
                continue

            # Print all extracted text
            print(f"\n  Extracted {len(results)} text regions:\n")
            print("  " + "-" * 50)

            for i, item in enumerate(results, 1):
                conf_pct = item['confidence'] * 100
                print(f"  {i:2}. {item['text']:<40} ({conf_pct:.1f}%)")

            print("  " + "-" * 50)

            # Try to categorize fields
            categories = categorize_id_fields(results)

            print("\n  Categorized fields:")

            if categories['names']:
                print(f"    Names: {', '.join(x['text'] for x in categories['names'])}")
            if categories['dates']:
                print(f"    Dates: {', '.join(x['text'] for x in categories['dates'])}")
            if categories['numbers']:
                print(f"    ID Numbers: {', '.join(x['text'] for x in categories['numbers'])}")
            if categories['addresses']:
                print(f"    Addresses: {', '.join(x['text'] for x in categories['addresses'])}")

        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
