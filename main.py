#!/usr/bin/env python3
"""
PlateVision - Live License Plate Recognition Demo
Run: python main.py
     python main.py --image path/to/image.jpg
Press 'q' to quit, 's' for screenshot, 'p' to process OCR
"""

import cv2
import sys
import os
import argparse

sys.path.insert(0, 'src')
from platevision import PlateVision, YOLO_AVAILABLE, EASYOCR_AVAILABLE

# Create screenshots directory
SCREENSHOTS_DIR = "images/screenshots"
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# COCO class names for YOLO (yolov8n.pt is trained on COCO)
# We only want to show car/vehicle related detections
VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}


def detect_plate_regions(frame, pv):
    """
    Detect plate-like regions using edge detection and contour analysis.
    This is more reliable for plates than generic YOLO object detection.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = []

    # Apply bilateral filter to reduce noise while keeping edges sharp
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Edge detection
    edges = cv2.Canny(gray, 30, 200)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

        # License plates are typically rectangles (4 vertices)
        if len(approx) >= 4 and len(approx) <= 6:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # License plate aspect ratios are typically between 2 and 6
            if 1.5 < aspect_ratio < 7 and w > 60 and h > 15:
                plates.append({
                    'bbox': [x, y, x + w, y + h],
                    'confidence': 0.7,
                    'crop': frame[y:y+h, x:x+w].copy()
                })

    return plates[:5]  # Return top 5 candidates


def process_image(image_path, pv):
    """Process a single image file."""
    print(f"\nProcessing image: {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Cannot load image: {image_path}")
        return

    display_frame = frame.copy()

    # Detect plates using contour-based detection
    detected_plates = detect_plate_regions(frame, pv)
    print(f"Detected {len(detected_plates)} potential plate region(s)")

    # Draw detected regions
    for i, plate in enumerate(detected_plates):
        bbox = plate.get('bbox', [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_frame, f"Plate {i+1}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Try OCR on detected regions
    if EASYOCR_AVAILABLE and detected_plates:
        print("\nRunning OCR on detected plates...")
        for i, plate in enumerate(detected_plates):
            try:
                ocr_result = pv.extract_text(plate['crop'])
                text = ocr_result.get('text', '')
                conf = ocr_result.get('confidence', 0)
                if text:
                    print(f"  Plate {i+1}: {text} (confidence: {conf:.0%})")
                    # Update display
                    x1, y1, x2, y2 = plate['bbox']
                    cv2.putText(display_frame, text, (x1, y2+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    print(f"  Plate {i+1}: No text detected")
            except Exception as e:
                print(f"  Plate {i+1}: OCR error - {e}")
    elif not EASYOCR_AVAILABLE:
        print("EasyOCR not available - install with: pip install easyocr")

    # Show result
    cv2.putText(display_frame, f"PlateVision - {os.path.basename(image_path)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('PlateVision - Image Analysis', display_frame)
    print("\nPress any key to continue, 'q' to quit...")

    key = cv2.waitKey(0) & 0xFF
    return key != ord('q')


def run_webcam(pv):
    """Run live webcam detection."""
    print("Starting webcam...")
    print("(For better results, use images with visible license plates)")
    print()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot access webcam!")
        return

    print("Controls:")
    print("  q - Quit")
    print("  s - Screenshot")
    print("  p - Process current frame (full OCR)")
    print()
    print("Webcam started! Show a license plate to the camera...")
    print()

    frame_count = 0
    detected_plates = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display_frame = frame.copy()

        # Detect plates using contour-based detection (every 10 frames for performance)
        if frame_count % 10 == 0:
            try:
                detected_plates = detect_plate_regions(frame, pv)
            except Exception as e:
                detected_plates = []

        # Draw results
        for i, plate in enumerate(detected_plates):
            bbox = plate.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            conf = plate.get('confidence', 0)

            # Green box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label
            label = f"Plate candidate ({conf:.0%})"
            cv2.putText(display_frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Info on screen
        cv2.putText(display_frame, "PlateVision Demo", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Plate candidates: {len(detected_plates)} | OCR: {'OK' if EASYOCR_AVAILABLE else 'NO'}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, "q=quit, s=screenshot, p=process OCR", (10, display_frame.shape[0]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Display
        cv2.imshow('PlateVision - Live Demo', display_frame)

        # Keys
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Exiting...")
            break

        elif key == ord('s'):
            filename = os.path.join(SCREENSHOTS_DIR, f"plate_screenshot_{frame_count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")

        elif key == ord('p'):
            # Full OCR processing
            print("\nProcessing current frame with OCR...")
            try:
                # Re-detect plates
                plates = detect_plate_regions(frame, pv)
                if plates and EASYOCR_AVAILABLE:
                    print(f"Found {len(plates)} plate region(s):")
                    for i, plate in enumerate(plates, 1):
                        ocr_result = pv.extract_text(plate['crop'])
                        text = ocr_result.get('text', 'N/A')
                        conf = ocr_result.get('confidence', 0)
                        print(f"  {i}. Text: {text} (confidence: {conf:.0%})")

                        # Validate format
                        if text:
                            validation = pv.validate_plate(text)
                            print(f"     Region: {validation.get('region', 'N/A')}, Valid format: {validation.get('valid', False)}")
                elif not EASYOCR_AVAILABLE:
                    print("EasyOCR not available. Install with: pip install easyocr")
                else:
                    print("No plate regions detected in frame.")
            except Exception as e:
                print(f"Processing error: {e}")
            print()

    cap.release()
    cv2.destroyAllWindows()
    print("Demo closed.")


def main():
    parser = argparse.ArgumentParser(description='PlateVision - License Plate Recognition')
    parser.add_argument('--image', '-i', type=str, help='Path to image file to process')
    parser.add_argument('--folder', '-f', type=str, help='Path to folder with images to process')
    args = parser.parse_args()

    print("=" * 50)
    print("  PlateVision - License Plate Recognition")
    print("=" * 50)
    print()
    print(f"YOLO available: {YOLO_AVAILABLE}")
    print(f"EasyOCR available: {EASYOCR_AVAILABLE}")
    print()

    if not EASYOCR_AVAILABLE:
        print("WARNING: EasyOCR is not installed!")
        print("Run: pip install easyocr")
        print("Will detect plate regions but without text recognition...")
        print()

    # Initialize
    print("Initializing PlateVision...")
    pv = PlateVision(
        detector='yolov8',
        ocr_backend='easyocr',
        languages=['en'],
        confidence_threshold=0.3
    )
    print(f"  Detector: Contour-based (for plate-like shapes)")
    print(f"  OCR: {pv.ocr_backend_name}")
    print(f"  Screenshots folder: {SCREENSHOTS_DIR}/")
    print()

    if args.image:
        # Process single image
        process_image(args.image, pv)
        cv2.destroyAllWindows()
    elif args.folder:
        # Process folder of images
        print(f"Processing images from: {args.folder}")
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        files = [f for f in os.listdir(args.folder) if f.lower().endswith(extensions)]

        for filename in sorted(files):
            filepath = os.path.join(args.folder, filename)
            if not process_image(filepath, pv):
                break
        cv2.destroyAllWindows()
    else:
        # Run webcam mode
        run_webcam(pv)


if __name__ == "__main__":
    main()
