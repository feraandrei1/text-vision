"""
PlateVision - Video stream utilities
"""

import cv2
import numpy as np
from typing import Iterator, Optional, Union


class VideoStream:
    """
    Video stream handler for ALPR processing.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None
    ):
        """
        Initialize video stream.

        Args:
            source: Video source (0=webcam, path=video file, URL=IP camera)
            width: Frame width (optional)
            height: Frame height (optional)
            fps: Target FPS (optional)
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

        self._window_name = "PlateVision Stream"
        self._running = True

    @property
    def width(self) -> int:
        """Get frame width."""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Get frame height."""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        """Get FPS."""
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        """Get total frames (for video files)."""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self) -> Optional[np.ndarray]:
        """Read single frame."""
        ret, frame = self.cap.read()
        return frame if ret else None

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over frames."""
        while self._running:
            frame = self.read()
            if frame is None:
                break
            yield frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def show(self, frame: np.ndarray, window_name: Optional[str] = None) -> None:
        """Display frame."""
        cv2.imshow(window_name or self._window_name, frame)

    def stop(self) -> None:
        """Stop stream."""
        self._running = False

    def release(self) -> None:
        """Release resources."""
        self._running = False
        self.cap.release()
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def save_frame(self, frame: np.ndarray, path: str) -> bool:
        """Save frame to file."""
        return cv2.imwrite(path, frame)
