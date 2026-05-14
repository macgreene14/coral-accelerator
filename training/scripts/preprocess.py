"""Preprocess PV thermal images and unify annotations to COCO JSON format."""
import cv2
import numpy as np


# Unified class map: covers all name variants across all three datasets
CLASS_MAP = {
    # InfraredSolarModules variants
    "hotspot": 0, "Hotspot": 0, "Hot Spot": 0, "HotSpot": 0,
    "bypass_diode_failure": 1, "Bypass Diode": 1, "BypassDiode": 1, "bypass diode": 1,
    "soiling": 2, "Soiling": 2,
    "multi_hotspot": 3, "Multi-Hot Spot": 3, "MultiHotSpot": 3, "multi hotspot": 3,
    "shadowing": 4, "Shadowing": 4,
    "delamination": 5, "Delamination": 5,
    # PVDN generic labels
    "anomaly": 0, "bypass": 1, "disconnected": 5,
}

CATEGORIES = [
    {"id": i, "name": name, "supercategory": "pv_defect"}
    for i, name in enumerate(
        ["hotspot", "bypass_diode_failure", "soiling", "multi_hotspot", "shadowing", "delamination"]
    )
]


def normalize_ir_image(img: np.ndarray) -> np.ndarray:
    """Normalize a thermal IR image to 8-bit 3-channel (R=G=B) format.

    Applies 2nd–98th percentile clipping to maximize anomaly contrast
    across diverse sensor types (FLIR JPEG, 16-bit TIFF, pseudo-color PNG).
    Stacks grayscale result to 3 channels so EfficientDet's pretrained
    ImageNet stem transfers without modification.

    Args:
        img: Input array, any dtype (uint8, uint16, float32), 1 or 3 channels.

    Returns:
        uint8 array of shape (H, W, 3) with all three channels identical.
    """
    # Collapse to single channel
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = img[:, :, 0]

    img = img.astype(np.float32)
    p2 = float(np.percentile(img, 2))
    p98 = float(np.percentile(img, 98))

    if p98 > p2:
        img = np.clip(img, p2, p98)
        img = ((img - p2) / (p98 - p2) * 255.0).astype(np.uint8)
    else:
        img = np.zeros_like(img, dtype=np.uint8)

    return np.stack([img, img, img], axis=2)
