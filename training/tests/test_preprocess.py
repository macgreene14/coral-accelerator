"""Unit tests for preprocess.py helpers. No datasets or GPU required."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from preprocess import normalize_ir_image


class TestNormalizeIrImage:
    def test_grayscale_input_returns_3channel_uint8(self):
        img = np.random.randint(0, 65535, (480, 640), dtype=np.uint16)
        result = normalize_ir_image(img)
        assert result.shape == (480, 640, 3)
        assert result.dtype == np.uint8

    def test_all_three_channels_are_identical(self):
        img = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        result = normalize_ir_image(img)
        np.testing.assert_array_equal(result[:, :, 0], result[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 1], result[:, :, 2])

    def test_output_pixel_range_is_0_to_255(self):
        img = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        result = normalize_ir_image(img)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_bgr_3channel_input_accepted(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 200
        result = normalize_ir_image(img)
        assert result.shape == (100, 100, 3)

    def test_constant_image_returns_zeros(self):
        # p2 == p98 for constant input → no dynamic range → return zeros
        img = np.full((100, 100), 5000, dtype=np.uint16)
        result = normalize_ir_image(img)
        assert result.max() == 0

    def test_uint8_input_normalizes_correctly(self):
        img = np.array([[0, 127, 255]], dtype=np.uint8)
        result = normalize_ir_image(img)
        assert result.dtype == np.uint8
        # After percentile clip of a 3-element array: min=0, max=255
        assert result.max() == 255
