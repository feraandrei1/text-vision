"""
PlateVision - Video Processing Example
"""

import sys
sys.path.insert(0, '..')

from src.platevision import PlateVision
from src.video_stream import VideoStream


def main():
    # Initialize PlateVision
    pv = PlateVision(
        detector="yolov8",
        ocr_backend="easyocr"
    )

    print("=" * 50)
    print("Video Plate Recognition")
    print("=" * 50)

    # Replace with your video path or use 0 for webcam
    video_source = "traffic_video.mp4"  # or 0 for webcam

    try:
        with VideoStream(source=video_source) as stream:
            print(f"\nProcessing video... Press 'q' to quit.")
            print(f"Resolution: {stream.width}x{stream.height}")
            print(f"FPS: {stream.fps}")

            detected_plates = set()

            for frame, plates in pv.process_stream(stream, skip_frames=2):
                # Draw results on frame
                frame = pv.draw_results(frame, plates)

                # Track unique plates
                for plate in plates:
                    if plate['text'] and plate['confidence'] > 0.7:
                        if plate['text'] not in detected_plates:
                            detected_plates.add(plate['text'])
                            print(f"  New plate: {plate['text']} ({plate['confidence']:.0%})")

                # Show frame
                stream.show(frame)

            print(f"\n\nTotal unique plates detected: {len(detected_plates)}")
            for plate in detected_plates:
                print(f"  - {plate}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
