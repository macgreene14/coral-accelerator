"""Unit tests for detect.py helpers. No EdgeTPU or camera required."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from detect import (
    draw_detections,
    find_coral_usb,
    get_top_detections,
    load_labels,
    preprocess_frame,
)


def _make_detection(xmin, ymin, xmax, ymax, class_id, score):
    """Create a mock pycoral detection object."""
    bbox = SimpleNamespace(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
    return SimpleNamespace(bbox=bbox, id=class_id, score=score)


class TestLoadLabels:
    def test_returns_list_of_strings(self, tmp_path):
        f = tmp_path / "labels.txt"
        f.write_text("cat\ndog\nbird\n")
        assert load_labels(f) == ["cat", "dog", "bird"]

    def test_strips_whitespace(self, tmp_path):
        f = tmp_path / "labels.txt"
        f.write_text("  cat  \n  dog  \n")
        assert load_labels(f) == ["cat", "dog"]

    def test_empty_file_returns_empty_list(self, tmp_path):
        f = tmp_path / "labels.txt"
        f.write_text("")
        assert load_labels(f) == []


class TestPreprocessFrame:
    def test_output_shape_matches_target_size(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = preprocess_frame(frame, (300, 300))
        assert result.shape == (300, 300, 3)

    def test_output_dtype_is_uint8(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = preprocess_frame(frame, (300, 300))
        assert result.dtype == np.uint8


class TestFindCoralUsb:
    def test_returns_true_when_bootloader_vendor_present(self):
        with patch("detect.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="Vendor ID: 0x1a6e")
            assert find_coral_usb() is True

    def test_returns_true_when_runtime_vendor_present(self):
        with patch("detect.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="Vendor ID: 0x18d1")
            assert find_coral_usb() is True

    def test_returns_false_when_no_coral_vendor(self):
        with patch("detect.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="Vendor ID: 0x05ac")
            assert find_coral_usb() is False

    def test_returns_false_on_subprocess_exception(self):
        with patch("detect.subprocess.run", side_effect=Exception("timeout")):
            assert find_coral_usb() is False


class TestGetTopDetections:
    def test_filters_by_threshold(self):
        detections = [
            _make_detection(0, 0, 100, 100, 0, 0.9),
            _make_detection(0, 0, 100, 100, 1, 0.3),
            _make_detection(0, 0, 100, 100, 2, 0.6),
        ]
        result = get_top_detections(detections, threshold=0.5)
        assert len(result) == 2
        assert all(d.score >= 0.5 for d in result)

    def test_returns_all_above_threshold(self):
        detections = [_make_detection(0, 0, 10, 10, 0, 0.8)]
        assert len(get_top_detections(detections, threshold=0.5)) == 1

    def test_empty_input_returns_empty(self):
        assert get_top_detections([], threshold=0.5) == []


class TestDrawDetections:
    def test_returns_array_same_shape_as_input(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [_make_detection(10, 10, 100, 100, 0, 0.9)]
        labels = ["cat", "dog"]
        result = draw_detections(frame, detections, labels, input_size=(300, 300))
        assert result.shape == frame.shape

    def test_returns_uint8(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [_make_detection(10, 10, 100, 100, 0, 0.9)]
        labels = ["cat"]
        result = draw_detections(frame, detections, labels, input_size=(300, 300))
        assert result.dtype == np.uint8

    def test_empty_detections_returns_unchanged_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[200, 300] = [10, 20, 30]
        result = draw_detections(frame, [], ["cat"], input_size=(300, 300))
        assert result[200, 300].tolist() == [10, 20, 30]

    def test_does_not_mutate_input_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        detections = [_make_detection(10, 10, 100, 100, 0, 0.9)]
        draw_detections(frame, detections, ["cat"], input_size=(300, 300))
        np.testing.assert_array_equal(frame, original)
