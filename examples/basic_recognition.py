"""
PlateVision - Basic Plate Recognition Example
"""

import sys
import os
sys.path.insert(0, '..')

from src.platevision import PlateVision

# Folder for test images (not uploaded to GitHub)
TEST_IMAGES_DIR = "../storage/test_images"


def main():
    # Create test_images folder if it doesn't exist
    if not os.path.exists(TEST_IMAGES_DIR):
        os.makedirs(TEST_IMAGES_DIR)
        print(f"Created folder: {TEST_IMAGES_DIR}/")

    # Initialize PlateVision with license plate detection model
    pv = PlateVision(
        detector="yolov8",
        ocr_backend="easyocr",
        languages=["en"],
        plate_region="AUTO",
        model_path="../models/license_plate_detector.pt"
    )

    print("=" * 50)
    print("License Plate Recognition Example")
    print("=" * 50)

    # Get all images from test_images folder
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(extensions)] if os.path.exists(TEST_IMAGES_DIR) else []

    if not images:
        print(f"\nNo images found in {TEST_IMAGES_DIR}/")
        print("Add your test images there and run again.")
        return

    print(f"\nFound {len(images)} image(s) to process.\n")

    for image_name in images:
        image_path = os.path.join(TEST_IMAGES_DIR, image_name)
        print("-" * 50)
        print(f"Processing: {image_name}")

        try:
            results = pv.recognize(image_path)

            if results:
                print(f"  Found {len(results)} plate(s):")
                for i, plate in enumerate(results, 1):
                    print(f"    Plate {i}: {plate['text']} ({plate['confidence']:.2%})")
            else:
                print("  No plates detected.")

        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
