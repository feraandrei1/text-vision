"""
Unit tests for PlateVision.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from platevision import PlateVision, YOLO_AVAILABLE, EASYOCR_AVAILABLE


class TestInitialization:
    """Test PlateVision initialization."""

    def test_default_init(self):
        """Test default initialization."""
        pv = PlateVision()
        assert pv.detector_name == 'yolov8'
        assert pv.ocr_backend_name == 'easyocr'
        assert pv.languages == ['en']

    def test_custom_init(self):
        """Test custom initialization."""
        pv = PlateVision(
            detector='yolov8',
            ocr_backend='paddleocr',
            languages=['en', 'ro'],
            confidence_threshold=0.7
        )
        assert pv.ocr_backend_name == 'paddleocr'
        assert pv.languages == ['en', 'ro']
        assert pv.confidence_threshold == 0.7

    def test_invalid_detector(self):
        """Test invalid detector raises error."""
        with pytest.raises(ValueError):
            PlateVision(detector='invalid')

    def test_invalid_ocr_backend(self):
        """Test invalid OCR backend raises error."""
        with pytest.raises(ValueError):
            PlateVision(ocr_backend='invalid')


class TestPreprocessing:
    """Test image preprocessing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.pv = PlateVision()
        self.color_image = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)

    def test_preprocess_converts_to_grayscale(self):
        """Test preprocessing converts to grayscale."""
        result = self.pv.preprocess_image(self.color_image)
        assert len(result.shape) == 2

    def test_preprocess_no_enhance(self):
        """Test preprocessing without enhancement."""
        result = self.pv.preprocess_image(self.color_image, enhance=False)
        assert result.shape == self.color_image.shape


class TestPlateValidation:
    """Test plate format validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.pv = PlateVision()

    def test_valid_romanian_plate(self):
        """Test valid Romanian plate format."""
        result = self.pv.validate_plate('B-123-ABC', 'EU-RO')
        assert result['valid'] is True
        assert result['region'] == 'EU-RO'

    def test_valid_romanian_plate_variant(self):
        """Test Romanian plate with different format."""
        result = self.pv.validate_plate('CJ-01-XYZ', 'EU-RO')
        assert result['valid'] is True

    def test_valid_uk_plate(self):
        """Test valid UK plate format."""
        result = self.pv.validate_plate('AB12CDE', 'EU-UK')
        assert result['valid'] is True

    def test_valid_us_plate(self):
        """Test valid US plate format."""
        result = self.pv.validate_plate('ABC1234', 'US')
        assert result['valid'] is True

    def test_invalid_plate(self):
        """Test invalid plate format."""
        result = self.pv.validate_plate('INVALID!!', 'EU-RO')
        assert result['valid'] is False

    def test_auto_detect_region(self):
        """Test automatic region detection."""
        result = self.pv.validate_plate('B-123-ABC')
        assert result['valid'] is True
        assert result['region'] == 'EU-RO'


class TestTextCleaning:
    """Test plate text cleaning."""

    def setup_method(self):
        """Set up test fixtures."""
        self.pv = PlateVision()

    def test_clean_lowercase(self):
        """Test cleaning converts to uppercase."""
        result = self.pv._clean_plate_text('abc123')
        assert result == 'ABC123'

    def test_clean_removes_special_chars(self):
        """Test cleaning removes special characters."""
        result = self.pv._clean_plate_text('AB!@#123')
        assert result == 'AB123'

    def test_clean_spaces_to_dashes(self):
        """Test cleaning converts spaces to dashes."""
        result = self.pv._clean_plate_text('B 123 ABC')
        assert result == 'B-123-ABC'


class TestDrawResults:
    """Test drawing methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.frame = np.zeros((400, 600, 3), dtype=np.uint8)

    def test_draw_results(self):
        """Test draw_results method."""
        plates = [
            {'bbox': [10, 10, 100, 50], 'text': 'B-123-ABC', 'confidence': 0.95}
        ]
        result = PlateVision.draw_results(self.frame, plates)
        assert result.shape == self.frame.shape
        # Check that something was drawn
        assert np.any(result != 0)

    def test_draw_multiple_plates(self):
        """Test drawing multiple plates."""
        plates = [
            {'bbox': [10, 10, 100, 50], 'text': 'PLATE1', 'confidence': 0.9},
            {'bbox': [200, 100, 350, 150], 'text': 'PLATE2', 'confidence': 0.85},
        ]
        result = PlateVision.draw_results(self.frame, plates)
        assert result.shape == self.frame.shape


class TestFallbackDetection:
    """Test fallback detection method."""

    def test_fallback_detection(self):
        """Test fallback detection finds candidates."""
        pv = PlateVision()
        # Create image with rectangle
        test_img = np.zeros((400, 600, 3), dtype=np.uint8)
        test_img[150:200, 200:400] = 255

        detected = pv._fallback_detection(test_img)
        assert isinstance(detected, list)


class TestConstants:
    """Test class constants."""

    def test_detectors_defined(self):
        """Test detectors list."""
        assert 'yolov8' in PlateVision.DETECTORS
        assert 'yolov5' in PlateVision.DETECTORS

    def test_ocr_backends_defined(self):
        """Test OCR backends list."""
        assert 'easyocr' in PlateVision.OCR_BACKENDS
        assert 'paddleocr' in PlateVision.OCR_BACKENDS

    def test_regions_defined(self):
        """Test regions list."""
        assert 'EU' in PlateVision.REGIONS
        assert 'US' in PlateVision.REGIONS

    def test_plate_patterns_defined(self):
        """Test plate patterns dict."""
        assert 'EU-RO' in PlateVision.PLATE_PATTERNS
        assert 'EU-UK' in PlateVision.PLATE_PATTERNS
        assert 'US' in PlateVision.PLATE_PATTERNS


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
