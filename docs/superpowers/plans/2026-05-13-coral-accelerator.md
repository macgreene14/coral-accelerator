# Coral USB Accelerator Test Repo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up a go-task + uv managed repo that runs MobileNet v2 image classification on a Coral USB Accelerator using the laptop camera, with a stub for future object detection.

**Architecture:** Host-only Python 3.9 environment managed by uv, with go-task as the canonical command interface. `src/classify.py` defers pycoral imports to inside `main()` so helper functions are unit-testable without the EdgeTPU runtime installed. Models are downloaded on demand, not committed to git.

**Tech Stack:** Python 3.9, uv, go-task 3.48.0, pycoral ~2.0, tflite-runtime, opencv-python, numpy, pytest

---

## File Map

| File | Purpose |
|---|---|
| `pyproject.toml` | uv project config, Python 3.9 pin, dependencies, coral extra-index-url |
| `.python-version` | uv Python version pin (`3.9`) |
| `.gitignore` | Ignore `.venv/`, `models/*.tflite`, `models/*.txt`, `__pycache__/` |
| `Taskfile.yml` | Task definitions: install, download-models, classify, detect, test |
| `models/.gitkeep` | Keeps `models/` in git without committing model files |
| `src/__init__.py` | Marks `src/` as a Python package directory |
| `src/classify.py` | Helper functions + `main()` for camera capture and EdgeTPU classification |
| `src/detect.py` | Stub; prints helpful message directing user to `task classify` |
| `tests/__init__.py` | Marks `tests/` as a package directory |
| `tests/test_classify.py` | Unit tests for classify.py helpers (no EdgeTPU or camera required) |
| `CLAUDE.md` | Dev notes: Python 3.9 constraint, task interface, Apple Silicon caveat |
| `README.md` | Quickstart, prerequisites, expected output, troubleshooting |

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `models/.gitkeep`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "coral-accelerator"
version = "0.1.0"
requires-python = "==3.9.*"
dependencies = [
    "pycoral~=2.0",
    "tflite-runtime",
    "opencv-python",
    "numpy",
]

[project.optional-dependencies]
dev = ["pytest"]

[tool.uv]
extra-index-url = ["https://google-coral.github.io/py-repo/"]
```

- [ ] **Step 2: Create `.python-version`**

```
3.9
```

- [ ] **Step 3: Create `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
*.pyo
models/*.tflite
models/*.txt
uv.lock
```

- [ ] **Step 4: Create directory placeholders**

```bash
mkdir -p models src tests
touch models/.gitkeep src/__init__.py tests/__init__.py
```

- [ ] **Step 5: Commit**

```bash
cd /Users/macgreene/Documents/coral-accelerator
git add pyproject.toml .python-version .gitignore models/.gitkeep src/__init__.py tests/__init__.py
git commit -m "chore: scaffold project structure"
```

---

### Task 2: Taskfile.yml

**Files:**
- Create: `Taskfile.yml`

- [ ] **Step 1: Create `Taskfile.yml`**

```yaml
version: '3'

vars:
  PYTHON: .venv/bin/python

tasks:
  install:
    desc: Install uv (if missing), check libedgetpu, set up Python 3.9 venv and dependencies
    cmds:
      - |
        if ! command -v uv &>/dev/null; then
          if [ ! -f "$HOME/.local/bin/uv" ]; then
            echo "Installing uv..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
          fi
          export PATH="$HOME/.local/bin:$PATH"
        fi
        UV=$(command -v uv 2>/dev/null || echo "$HOME/.local/bin/uv")
        EDGETPU_LIB="/usr/local/lib/libedgetpu.1.dylib"
        if [ ! -f "$EDGETPU_LIB" ]; then
          echo ""
          echo "ERROR: libedgetpu not found at $EDGETPU_LIB"
          echo ""
          echo "Install the Coral EdgeTPU runtime:"
          echo "  https://coral.ai/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime"
          echo ""
          echo "After installing, re-run: task install"
          exit 1
        fi
        echo "libedgetpu found at $EDGETPU_LIB"
        $UV venv --python 3.9
        $UV sync --extra dev

  download-models:
    desc: Download MobileNet v2 EdgeTPU model and ImageNet labels into models/
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

  classify:
    desc: Run image classification using the laptop camera and Coral USB Accelerator
    cmds:
      - "{{.PYTHON}} src/classify.py"

  detect:
    desc: Object detection (not yet implemented)
    cmds:
      - echo "Detection not yet implemented — run 'task classify' instead"

  test:
    desc: Run unit tests (no EdgeTPU or camera required)
    cmds:
      - "{{.PYTHON}} -m pytest tests/ -v"
```

- [ ] **Step 2: Verify tasks are listed correctly**

```bash
/opt/homebrew/bin/task --list
```

Expected output lists: `install`, `download-models`, `classify`, `detect`, `test`

- [ ] **Step 3: Commit**

```bash
git add Taskfile.yml
git commit -m "chore: add Taskfile with install/download-models/classify/detect/test tasks"
```

---

### Task 3: Install environment

**Files:** none (produces `.venv/`)

- [ ] **Step 1: Install uv**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

Expected: `uv 0.x.x`

- [ ] **Step 2: Check libedgetpu**

```bash
ls /usr/local/lib/libedgetpu.1.dylib
```

If the file is missing, install from:
`https://coral.ai/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime`

Download and run the `.pkg` installer with sudo, then re-check.

- [ ] **Step 3: Run task install**

```bash
/opt/homebrew/bin/task install
```

Expected: uv creates `.venv/` with Python 3.9, installs pycoral, tflite-runtime, opencv-python, numpy, pytest.

> **Apple Silicon note:** If pycoral installation fails with a platform or wheel error (pycoral may lack arm64 macOS wheels), try:
> ```bash
> arch -x86_64 /opt/homebrew/bin/task install
> ```
> Then prefix all subsequent task runs with `arch -x86_64`.

- [ ] **Step 4: Verify Python version in venv**

```bash
.venv/bin/python --version
```

Expected: `Python 3.9.x`

---

### Task 4: Write failing tests for classify.py helpers

**Files:**
- Create: `tests/test_classify.py`

- [ ] **Step 1: Create `tests/test_classify.py`**

```python
"""Unit tests for classify.py helpers. No EdgeTPU or camera required."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add src/ to path so we can import classify without installing it as a package.
# pycoral is only imported inside main(), so these tests run without libedgetpu.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from classify import find_coral_usb, get_top_k, load_labels, preprocess_frame


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/opt/homebrew/bin/task test
```

Expected: `ModuleNotFoundError: No module named 'classify'` — confirms `src/classify.py` doesn't exist yet.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_classify.py
git commit -m "test: add failing unit tests for classify.py helpers"
```

---

### Task 5: Implement classify.py helpers

**Files:**
- Create: `src/classify.py`

- [ ] **Step 1: Create `src/classify.py` with helper functions**

```python
"""Image classification using Coral USB Accelerator and laptop camera."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODELS_DIR / "mobilenet_v2_1.0_224_quant_edgetpu.tflite"
LABELS_PATH = MODELS_DIR / "imagenet_labels.txt"


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


def get_top_k(scores: np.ndarray, labels: list, k: int = 3) -> list:
    """Return top-k (label, confidence) pairs sorted by confidence descending."""
    indices = np.argsort(scores)[::-1][:k]
    return [(labels[i], float(scores[i])) for i in indices]
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
/opt/homebrew/bin/task test
```

Expected:
```
tests/test_classify.py ............                          [100%]
12 passed in 0.xxs
```

- [ ] **Step 3: Commit**

```bash
git add src/classify.py
git commit -m "feat: add classify.py helper functions"
```

---

### Task 6: Add main() to classify.py

**Files:**
- Modify: `src/classify.py` (append `main()` after the helpers)

- [ ] **Step 1: Append `main()` to `src/classify.py`**

Add the following after the `get_top_k` function:

```python


def main():
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

    # pycoral imports are deferred here so the helpers above stay testable
    # without libedgetpu installed on the host.
    from pycoral.adapters import classify as coral_classify
    from pycoral.adapters import common
    from pycoral.utils.edgetpu import make_interpreter

    print("Loading model on EdgeTPU...")
    interpreter = make_interpreter(str(MODEL_PATH))
    interpreter.allocate_tensors()

    labels = load_labels(LABELS_PATH)
    _, height, width, _ = interpreter.get_input_details()[0]["shape"]

    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera (index 0).", file=sys.stderr)
        sys.exit(1)

    print("Running classification — press Ctrl+C to stop.\n")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to capture frame.", file=sys.stderr)
                break

            resized = preprocess_frame(frame, (width, height))
            common.set_input(interpreter, resized)
            interpreter.invoke()
            scores = coral_classify.get_scores(interpreter)
            top = get_top_k(scores, labels, k=3)

            line = "  |  ".join(f"{label[:28]:28s} {conf:.1%}" for label, conf in top)
            print(f"\r{line}", end="", flush=True)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run tests to confirm they still pass**

```bash
/opt/homebrew/bin/task test
```

Expected: all 12 tests pass. (pycoral import is inside `main()`, never triggered by tests.)

- [ ] **Step 3: Commit**

```bash
git add src/classify.py
git commit -m "feat: add classify.py main() with camera capture and EdgeTPU inference loop"
```

---

### Task 7: detect.py stub

**Files:**
- Create: `src/detect.py`

- [ ] **Step 1: Create `src/detect.py`**

```python
"""Object detection using Coral USB Accelerator — not yet implemented."""


def main():
    print("Detection not yet implemented — run 'task classify' instead")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add src/detect.py
git commit -m "feat: add detect.py stub"
```

---

### Task 8: CLAUDE.md

**Files:**
- Create: `CLAUDE.md`

- [ ] **Step 1: Create `CLAUDE.md`**

```markdown
# CLAUDE.md

## What this repo is

A test repo for running ML inference on a Coral USB Accelerator (EdgeTPU) connected to a Mac.
Currently implements image classification via laptop camera; object detection is planned next.

## Python version constraint

pycoral only supports Python 3.9 and below. The `.python-version` file pins 3.9 for uv.
Do not upgrade without first checking pycoral wheel availability for the target Python version.

## Task interface

All operations go through go-task. Don't run scripts directly unless debugging.

| Command | What it does |
|---|---|
| `task install` | Install uv (if needed), check libedgetpu, create `.venv` with Python 3.9, install deps |
| `task download-models` | Fetch MobileNet v2 EdgeTPU model and labels into `models/` |
| `task classify` | Run camera classification (requires Coral USB + libedgetpu + downloaded models) |
| `task detect` | Stub — not yet implemented |
| `task test` | Run unit tests (no EdgeTPU or camera required) |

## Apple Silicon note

pycoral may not have arm64 macOS wheels. If `task install` fails with a platform or wheel error, try:

```bash
arch -x86_64 task install
```

Then run all subsequent tasks with `arch -x86_64 task <name>`.

## Models

Models live in `models/` and are not committed to git. Run `task download-models` to fetch them.
The task is idempotent — safe to re-run.

## pycoral import design

In `src/classify.py`, pycoral is imported inside `main()` rather than at module level.
This lets the helper functions (`load_labels`, `preprocess_frame`, `get_top_k`, `find_coral_usb`)
be unit-tested without libedgetpu installed: `task test` works without hardware.

## Next steps

- Implement `src/detect.py` using SSD MobileNet v2 for object detection with bounding boxes
- Add `--display` flag to classify/detect for an OpenCV live preview window
- Add a Dockerfile for non-USB workloads (model conversion, CPU-only TFLite inference, CI)
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add CLAUDE.md"
```

---

### Task 9: README.md

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create `README.md`**

```markdown
# Coral USB Accelerator

Image classification via a [Coral USB Accelerator](https://coral.ai/products/accelerator/)
and laptop camera, running on macOS.

## Prerequisites

- Coral USB Accelerator connected via USB 3
- [Coral EdgeTPU runtime](https://coral.ai/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime) (`libedgetpu`) installed
- [go-task](https://taskfile.dev) (`brew install go-task`)
- macOS (Intel or Apple Silicon — see Troubleshooting for Apple Silicon)

## Quickstart

```bash
task install           # set up Python 3.9 venv and dependencies
task download-models   # fetch MobileNet v2 model and labels
task classify          # point camera and run
```

## Expected output

```
coffee mug               94.1%  |  cup                         3.2%  |  espresso               1.4%
```

Top-3 labels with confidence, updated each frame. Press `Ctrl+C` to stop.

## Commands

| Command | Description |
|---|---|
| `task install` | Install dependencies |
| `task download-models` | Download model files |
| `task classify` | Run camera classification |
| `task test` | Run unit tests (no hardware required) |
| `task detect` | Object detection (coming soon) |

## Troubleshooting

**"Coral USB Accelerator not detected"**
Check USB connection. The device appears as vendor `0x1a6e` (bootloader) or `0x18d1` (after firmware load).

**"libedgetpu not found"**
Install the Coral runtime from the link in Prerequisites. Requires the `.pkg` installer with sudo.

**"Model not found"**
Run `task download-models`.

**pycoral install fails on Apple Silicon**
pycoral may not have arm64 macOS wheels. Try:
```bash
arch -x86_64 task install
```
Then run all tasks as `arch -x86_64 task <name>`.

## Coming soon

- Object detection with bounding box output (`task detect`)
- `--display` flag for a live OpenCV camera preview window
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with quickstart and troubleshooting"
```
