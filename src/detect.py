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


def main():
    print("Detection not yet implemented — run 'task classify' instead")


if __name__ == "__main__":
    main()
