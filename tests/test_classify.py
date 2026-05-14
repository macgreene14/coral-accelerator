"""Unit tests for classify.py helpers. No EdgeTPU or camera required."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add src/ to path so we can import classify without installing it as a package.
# pycoral is only imported inside main(), so these tests run without libedgetpu.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from classify import draw_classification_overlay, find_coral_usb, get_top_k, load_labels, preprocess_frame


class TestLoadLabels:
    def test_returns_list_of_strings(self, tmp_path):
        labels_file = tmp_path / "labels.txt"
        labels_file.write_text("cat\ndog\nbird\n")
        result = load_labels(labels_file)
        assert result == ["cat", "dog", "bird"]

    def test_strips_whitespace(self, tmp_path):
        labels_file = tmp_path / "labels.txt"
        labels_file.write_text("  cat  \n  dog  \n")
        result = load_labels(labels_file)
        assert result == ["cat", "dog"]

    def test_empty_file_returns_empty_list(self, tmp_path):
        labels_file = tmp_path / "labels.txt"
        labels_file.write_text("")
        result = load_labels(labels_file)
        assert result == []


class TestPreprocessFrame:
    def test_output_shape_matches_target_size(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = preprocess_frame(frame, (224, 224))
        assert result.shape == (224, 224, 3)

    def test_output_dtype_is_uint8(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = preprocess_frame(frame, (224, 224))
        assert result.dtype == np.uint8


class TestGetTopK:
    def test_returns_k_results(self):
        scores = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
        labels = ["a", "b", "c", "d", "e"]
        result = get_top_k(scores, labels, k=3)
        assert len(result) == 3

    def test_sorted_by_confidence_descending(self):
        scores = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
        labels = ["a", "b", "c", "d", "e"]
        result = get_top_k(scores, labels, k=3)
        assert result[0] == ("d", 0.8)
        assert result[1] == ("b", 0.5)
        assert result[2] == ("c", 0.3)

    def test_k_larger_than_scores_returns_all(self):
        scores = np.array([0.1, 0.9])
        labels = ["a", "b"]
        result = get_top_k(scores, labels, k=10)
        assert len(result) == 2


class TestDrawClassificationOverlay:
    def test_returns_array_same_shape_as_input(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        top = [("cat", 0.9), ("dog", 0.5), ("bird", 0.2)]
        result = draw_classification_overlay(frame, top)
        assert result.shape == frame.shape

    def test_returns_uint8(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        top = [("cat", 0.9)]
        result = draw_classification_overlay(frame, top)
        assert result.dtype == np.uint8

    def test_empty_top_returns_unchanged_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100, 100] = [42, 43, 44]
        result = draw_classification_overlay(frame, [])
        assert result[100, 100].tolist() == [42, 43, 44]

    def test_does_not_mutate_input_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        draw_classification_overlay(frame, [("cat", 0.9)])
        np.testing.assert_array_equal(frame, original)


class TestFindCoralUsb:
    def test_returns_true_when_bootloader_vendor_present(self):
        with patch("classify.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="Vendor ID: 0x1a6e")
            assert find_coral_usb() is True

    def test_returns_true_when_runtime_vendor_present(self):
        with patch("classify.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="Vendor ID: 0x18d1")
            assert find_coral_usb() is True

    def test_returns_false_when_no_coral_vendor(self):
        with patch("classify.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="Vendor ID: 0x05ac")
            assert find_coral_usb() is False

    def test_returns_false_on_subprocess_exception(self):
        with patch("classify.subprocess.run", side_effect=Exception("timeout")):
            assert find_coral_usb() is False
