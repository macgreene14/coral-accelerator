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
    import pycoral.utils.edgetpu as _edgetpu_mod
    # Under Rosetta on Apple Silicon, ctypes bare-name dlopen does not search
    # /usr/local/lib, so the default 'libedgetpu.1.dylib' fails. Override with
    # the absolute path so tflite_runtime's load_delegate finds the library.
    _edgetpu_mod._EDGETPU_SHARED_LIB = "/usr/local/lib/libedgetpu.1.dylib"
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

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
