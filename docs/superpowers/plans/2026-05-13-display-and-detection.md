# Display UI & Object Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an OpenCV live-preview window (`--display`) to `classify.py`, implement `detect.py` with SSD MobileNet v2 object detection and the same `--display` flag, and wire both into the Taskfile.

**Architecture:** Both scripts are independent and gain a `--display` argparse flag; the camera loop is unchanged when the flag is absent. Overlay rendering is isolated in pure helper functions (`draw_classification_overlay`, `draw_detections`) so they can be unit-tested without hardware. `detect.py` duplicates the helper pattern from `classify.py` — no shared module.

**Tech Stack:** Python 3.9, OpenCV (`cv2`), pycoral (`pycoral.adapters.detect`, `pycoral.adapters.common`), NumPy, go-task

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `Taskfile.yml` | Modify | Add detection model downloads; add `classify-display`, `detect`, `detect-display` tasks |
| `src/classify.py` | Modify | Add `draw_classification_overlay` helper; add `--display` argparse flag; wire imshow |
| `src/detect.py` | Create | Full SSD MobileNet v2 detection script with `--display` flag |
| `tests/test_classify.py` | Modify | Add `TestDrawClassificationOverlay` |
| `tests/test_detect.py` | Create | Tests for all `detect.py` helpers and `draw_detections` |

---

## Task 1: Extend download-models for detection assets

**Files:**
- Modify: `Taskfile.yml` (download-models task, lines 115–134)

- [ ] **Step 1: Add SSD MobileNet v2 and COCO labels downloads to Taskfile**

Replace the `download-models` task body with:

```yaml
  download-models:
    desc: Download classification and detection models + labels into models/
    cmds:
      - mkdir -p models
      - |
        if [ ! -f "models/mobilenet_v2_1.0_224_quant_edgetpu.tflite" ]; then
          echo "Downloading MobileNet v2 EdgeTPU model..."
          curl -L -o models/mobilenet_v2_1.0_224_quant_edgetpu.tflite \
            "https://github.com/google-coral/test_data/raw/master/mobilenet_v2_1.0_224_quant_edgetpu.tflite"
        else
          echo "Model already present: models/mobilenet_v2_1.0_224_quant_edgetpu.tflite"
        fi
      - |
        if [ ! -f "models/imagenet_labels.txt" ]; then
          echo "Downloading ImageNet labels..."
          curl -L -o models/imagenet_labels.txt \
            "https://raw.githubusercontent.com/google-coral/test_data/master/imagenet_labels.txt"
        else
          echo "Labels already present: models/imagenet_labels.txt"
        fi
      - |
        if [ ! -f "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite" ]; then
          echo "Downloading SSD MobileNet v2 COCO EdgeTPU model..."
          curl -L -o models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
            "https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
        else
          echo "Model already present: models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
        fi
      - |
        if [ ! -f "models/coco_labels.txt" ]; then
          echo "Downloading COCO labels..."
          curl -L -o models/coco_labels.txt \
            "https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt"
        else
          echo "Labels already present: models/coco_labels.txt"
        fi
```

- [ ] **Step 2: Run and verify**

```bash
task download-models
```

Expected: all four files present in `models/`:
```
models/mobilenet_v2_1.0_224_quant_edgetpu.tflite
models/imagenet_labels.txt
models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
models/coco_labels.txt
```

- [ ] **Step 3: Commit**

```bash
git add Taskfile.yml
git commit -m "feat: extend download-models to fetch SSD MobileNet v2 and COCO labels"
```

---

## Task 2: Add draw_classification_overlay to classify.py (TDD)

**Files:**
- Modify: `src/classify.py`
- Modify: `tests/test_classify.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_classify.py` (after the existing imports, add `from classify import ... draw_classification_overlay` to the import line, then add the class):

```python
from classify import draw_classification_overlay, find_coral_usb, get_top_k, load_labels, preprocess_frame


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
```

- [ ] **Step 2: Run to verify it fails**

```bash
task test
```

Expected: `ImportError` — `draw_classification_overlay` not yet defined.

- [ ] **Step 3: Implement draw_classification_overlay in classify.py**

Add after the `get_top_k` function (before `main`):

```python
# BGR color constants for confidence bars
_BAR_GREEN = (0, 200, 0)
_BAR_YELLOW = (0, 200, 200)
_BAR_RED = (0, 0, 200)


def draw_classification_overlay(frame: np.ndarray, top: list) -> np.ndarray:
    """Draw top-k classification results as a semi-transparent panel on the frame.

    Returns a new annotated frame; the input frame is not mutated.
    """
    out = frame.copy()
    if not top:
        return out

    row_h = 28
    pad = 8
    panel_w = 340
    panel_h = len(top) * row_h + pad
    panel_x = 10
    panel_y = frame.shape[0] - panel_h - 10

    overlay = out.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    bar_max_w = 120
    for i, (label, conf) in enumerate(top):
        y = panel_y + pad // 2 + i * row_h
        if conf >= 0.7:
            color = _BAR_GREEN
        elif conf >= 0.4:
            color = _BAR_YELLOW
        else:
            color = _BAR_RED
        bar_w = int(conf * bar_max_w)
        cv2.rectangle(out, (panel_x + 6, y + 4), (panel_x + 6 + bar_w, y + row_h - 4), color, -1)
        cv2.putText(out, label[:22], (panel_x + 132, y + row_h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, f"{conf:.0%}", (panel_x + 300, y + row_h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
task test
```

Expected: all tests pass including the new `TestDrawClassificationOverlay` (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/classify.py tests/test_classify.py
git commit -m "feat: add draw_classification_overlay helper with tests"
```

---

## Task 3: Add --display flag to classify.py and Taskfile

**Files:**
- Modify: `src/classify.py`
- Modify: `Taskfile.yml`

- [ ] **Step 1: Add argparse and wire --display into the camera loop**

Replace the top of `classify.py` imports block and `main()` with the following changes:

Add `import argparse` to the imports at the top of the file (after `import cv2`):

```python
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
```

Replace the start of `main()` through `cap = cv2.VideoCapture(0)`:

```python
def main():
    parser = argparse.ArgumentParser(description="Coral USB image classification")
    parser.add_argument("--display", action="store_true", help="Show live camera window with overlay")
    args = parser.parse_args()

    if not find_coral_usb():
```

Replace the camera loop and cleanup in `main()` (from `print("Running classification")` through `cap.release()`):

```python
    print("Running classification — press Ctrl+C to stop.\n")
    try:
        consecutive_failures = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    print("ERROR: Failed to capture frame.", file=sys.stderr)
                    break
                continue
            consecutive_failures = 0

            resized = preprocess_frame(frame, (width, height))
            common.set_input(interpreter, resized)
            interpreter.invoke()
            scores = coral_classify.get_scores(interpreter)
            top = get_top_k(scores, labels, k=3)

            line = "  |  ".join(f"{label[:28]:28s} {conf:.1%}" for label, conf in top)
            print(f"\r{line}", end="", flush=True)

            if args.display:
                annotated = draw_classification_overlay(frame, top)
                cv2.imshow("Classify", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
```

- [ ] **Step 2: Add classify-display task to Taskfile**

Add after the `classify` task:

```yaml
  classify-display:
    desc: Run classification with live OpenCV preview window
    env:
      DYLD_LIBRARY_PATH: /usr/local/lib
    cmds:
      - "arch -x86_64 {{.PYTHON}} src/classify.py --display"
```

- [ ] **Step 3: Run tests to verify nothing broke**

```bash
task test
```

Expected: all tests pass.

- [ ] **Step 4: Manual smoke test (requires Coral USB)**

```bash
task classify-display
```

Expected: OpenCV window opens showing live camera feed with classification overlay panel at bottom-left. Press `q` to quit.

- [ ] **Step 5: Commit**

```bash
git add src/classify.py Taskfile.yml
git commit -m "feat: add --display flag to classify.py with OpenCV overlay window"
```

---

## Task 4: Implement detect.py helpers + draw_detections (TDD)

**Files:**
- Create: `src/detect.py`
- Create: `tests/test_detect.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_detect.py`:

```python
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
```

- [ ] **Step 2: Run to verify tests fail**

```bash
task test
```

Expected: `ModuleNotFoundError: No module named 'detect'`

- [ ] **Step 3: Create src/detect.py with helpers only (no main yet)**

```python
"""Object detection using Coral USB Accelerator and laptop camera."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODELS_DIR / "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
LABELS_PATH = MODELS_DIR / "coco_labels.txt"

# Deterministic per-class colors in BGR
_DETECTION_COLORS = [
    (0, 200, 0),    # green
    (0, 0, 200),    # red
    (200, 0, 0),    # blue
    (0, 200, 200),  # yellow
    (200, 200, 0),  # cyan
    (200, 0, 200),  # magenta
    (80, 200, 0),   # lime
    (0, 100, 200),  # orange
]


def find_coral_usb() -> bool:
    """Return True if a Coral USB Accelerator is detected via system_profiler."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPUSBDataType"],
            capture_output=True, text=True, timeout=5,
        )
        return "0x1a6e" in result.stdout or "0x18d1" in result.stdout
    except Exception:
        return False


def load_labels(path: Path) -> list:
    """Load label strings from a text file, one label per line, stripped of whitespace."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def preprocess_frame(frame: np.ndarray, size: tuple) -> np.ndarray:
    """Resize frame to (width, height) and return as uint8 array."""
    return cv2.resize(frame, size)


def get_top_detections(detections: list, threshold: float) -> list:
    """Return detections with score >= threshold."""
    return [d for d in detections if d.score >= threshold]


def draw_detections(
    frame: np.ndarray,
    detections: list,
    labels: list,
    input_size: tuple,
) -> np.ndarray:
    """Draw bounding boxes and label pills for detected objects on the frame.

    detections: list of pycoral detection objects with .bbox (xmin/ymin/xmax/ymax),
                .id (class index), and .score attributes. Coordinates are in
                input_size pixel space and will be scaled to the frame dimensions.
    input_size: (width, height) of the model input tensor (e.g. (300, 300)).
    Returns a new annotated frame; the input frame is not mutated.
    """
    out = frame.copy()
    if not detections:
        return out

    input_w, input_h = input_size
    scale_x = frame.shape[1] / input_w
    scale_y = frame.shape[0] / input_h

    for obj in detections:
        color = _DETECTION_COLORS[obj.id % len(_DETECTION_COLORS)]
        xmin = int(obj.bbox.xmin * scale_x)
        ymin = int(obj.bbox.ymin * scale_y)
        xmax = int(obj.bbox.xmax * scale_x)
        ymax = int(obj.bbox.ymax * scale_y)

        cv2.rectangle(out, (xmin, ymin), (xmax, ymax), color, 2)

        label_text = f"{labels[obj.id] if obj.id < len(labels) else str(obj.id)} {obj.score:.0%}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        pill_y = max(ymin - th - 6, 0)
        cv2.rectangle(out, (xmin, pill_y), (xmin + tw + 6, pill_y + th + 6), color, -1)
        cv2.putText(out, label_text, (xmin + 3, pill_y + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
task test
```

Expected: all tests pass including the new `tests/test_detect.py` suite.

- [ ] **Step 5: Commit**

```bash
git add src/detect.py tests/test_detect.py
git commit -m "feat: add detect.py helpers and draw_detections with tests"
```

---

## Task 5: Implement detect.py main() and wire Taskfile

**Files:**
- Modify: `src/detect.py`
- Modify: `Taskfile.yml`

- [ ] **Step 1: Add main() to detect.py**

Append to `src/detect.py`:

```python
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Coral USB object detection")
    parser.add_argument("--display", action="store_true", help="Show live camera window with detections")
    parser.add_argument("--threshold", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    args = parser.parse_args()

    if not find_coral_usb():
        print("ERROR: Coral USB Accelerator not detected.", file=sys.stderr)
        print("Check USB connection. Vendor IDs: 0x1a6e (bootloader) or 0x18d1 (runtime).", file=sys.stderr)
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}", file=sys.stderr)
        print("Run: task download-models", file=sys.stderr)
        sys.exit(1)

    if not LABELS_PATH.exists():
        print(f"ERROR: Labels not found at {LABELS_PATH}", file=sys.stderr)
        print("Run: task download-models", file=sys.stderr)
        sys.exit(1)

    # pycoral imports are deferred so helpers stay testable without libedgetpu.
    import pycoral.utils.edgetpu as _edgetpu_mod
    # Under Rosetta on Apple Silicon, ctypes bare-name dlopen does not search
    # /usr/local/lib. Override with absolute path so load_delegate finds the lib.
    _edgetpu_mod._EDGETPU_SHARED_LIB = "/usr/local/lib/libedgetpu.1.dylib"
    from pycoral.adapters import common
    from pycoral.adapters import detect as coral_detect
    from pycoral.utils.edgetpu import make_interpreter

    print("Loading model on EdgeTPU...")
    interpreter = make_interpreter(str(MODEL_PATH))
    interpreter.allocate_tensors()

    labels = load_labels(LABELS_PATH)
    _, input_height, input_width, _ = interpreter.get_input_details()[0]["shape"]

    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera (index 0).", file=sys.stderr)
        sys.exit(1)

    print(f"Running detection (threshold={args.threshold:.0%}) — press Ctrl+C to stop.\n")
    try:
        consecutive_failures = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    print("ERROR: Failed to capture frame.", file=sys.stderr)
                    break
                continue
            consecutive_failures = 0

            resized = preprocess_frame(frame, (input_width, input_height))
            common.set_input(interpreter, resized)
            interpreter.invoke()
            detections = coral_detect.get_objects(interpreter, args.threshold)
            top = get_top_detections(detections, args.threshold)

            if top:
                line = "  ".join(
                    f"{labels[d.id] if d.id < len(labels) else d.id} {d.score:.0%}"
                    for d in top
                )
            else:
                line = "(no detections)"
            print(f"\r{line:<60}", end="", flush=True)

            if args.display:
                annotated = draw_detections(frame, top, labels, input_size=(input_width, input_height))
                cv2.imshow("Detect", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Replace the detect stub and add detect-display in Taskfile**

Replace the `detect` task and add `detect-display` after `classify-display`:

```yaml
  detect:
    desc: Run object detection using the laptop camera and Coral USB Accelerator
    env:
      DYLD_LIBRARY_PATH: /usr/local/lib
    cmds:
      - "arch -x86_64 {{.PYTHON}} src/detect.py"

  detect-display:
    desc: Run detection with live OpenCV preview window
    env:
      DYLD_LIBRARY_PATH: /usr/local/lib
    cmds:
      - "arch -x86_64 {{.PYTHON}} src/detect.py --display"
```

- [ ] **Step 3: Run tests to verify nothing broke**

```bash
task test
```

Expected: all tests pass.

- [ ] **Step 4: Manual smoke test (requires Coral USB + downloaded models)**

```bash
task detect-display
```

Expected: OpenCV window opens showing live camera feed with bounding boxes drawn around detected objects, label pill above each box showing class name and confidence. Press `q` to quit.

- [ ] **Step 5: Commit**

```bash
git add src/detect.py Taskfile.yml
git commit -m "feat: implement detect.py with SSD MobileNet v2 and --display flag"
```
