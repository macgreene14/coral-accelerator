"""Object detection using Coral USB Accelerator and laptop camera."""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODELS_DIR / "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
LABELS_PATH = MODELS_DIR / "coco_labels.txt"

_MODEL_CONFIGS = {
    "coco": {
        "model": MODELS_DIR / "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
        "labels": MODELS_DIR / "coco_labels.txt",
        "default_threshold": 0.4,
    },
    "pv": {
        "model": MODELS_DIR / "pv_detector_edgetpu.tflite",
        "labels": MODELS_DIR / "pv_labels.txt",
        "default_threshold": 0.35,
    },
}

# Deterministic per-class colors in BGR
_DETECTION_COLORS = [
    (0, 200, 0),    # green
    (0, 0, 200),    # red
    (200, 0, 0),    # blue
    (0, 200, 200),  # olive-yellow
    (200, 200, 0),  # teal
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
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)
        pill_y = max(ymin - th - 6, 0)
        cv2.rectangle(out, (xmin, pill_y), (xmin + tw + 6, pill_y + th + 6), color, -1)
        cv2.putText(out, label_text, (xmin + 3, pill_y + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def main():
    parser = argparse.ArgumentParser(description="Coral USB object detection")
    parser.add_argument("--display", action="store_true", help="Show live camera window with detections")
    parser.add_argument(
        "--model", choices=["coco", "pv"], default="coco",
        help="Model to use: coco (default COCO SSD) or pv (PV defect detector)"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Confidence threshold (default: 0.4 for coco, 0.35 for pv)"
    )
    args = parser.parse_args()

    cfg = _MODEL_CONFIGS[args.model]
    model_path = cfg["model"]
    labels_path = cfg["labels"]
    threshold = args.threshold if args.threshold is not None else cfg["default_threshold"]

    if not find_coral_usb():
        print("ERROR: Coral USB Accelerator not detected.", file=sys.stderr)
        print("Check USB connection. Vendor IDs: 0x1a6e (bootloader) or 0x18d1 (runtime).", file=sys.stderr)
        sys.exit(1)

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}", file=sys.stderr)
        if args.model == "pv":
            print("Run: task training:export && task training:deploy", file=sys.stderr)
        else:
            print("Run: task download-models", file=sys.stderr)
        sys.exit(1)

    if not labels_path.exists():
        print(f"ERROR: Labels not found at {labels_path}", file=sys.stderr)
        if args.model == "pv":
            print("Run: task training:export && task training:deploy", file=sys.stderr)
        else:
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
    interpreter = make_interpreter(str(model_path))
    interpreter.allocate_tensors()

    labels = load_labels(labels_path)
    _, input_height, input_width, _ = interpreter.get_input_details()[0]["shape"]

    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera (index 0).", file=sys.stderr)
        sys.exit(1)

    print(f"Running detection — model={args.model}, threshold={threshold:.0%} — press Ctrl+C to stop.\n")
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
            top = coral_detect.get_objects(interpreter, threshold)

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
                    print("\nStopped.")
                    break

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
