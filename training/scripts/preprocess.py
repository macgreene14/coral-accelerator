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


import xml.etree.ElementTree as ET
from pathlib import Path


def polygon_to_bbox(polygon: list) -> list:
    """Convert a list of [x, y] points to COCO [x_min, y_min, width, height]."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def binary_to_full_bbox(width: int, height: int) -> list:
    """Return a COCO bbox [x, y, w, h] covering the full image."""
    return [0, 0, width, height]


def voc_xml_to_coco_annotations(
    xml_path: Path,
    image_id: int,
    ann_id_start: int,
) -> tuple:
    """Parse a Pascal VOC XML annotation file into COCO-format dicts.

    Args:
        xml_path: Path to VOC .xml file.
        image_id: COCO image id to assign.
        ann_id_start: First annotation id to use (incremented per object).

    Returns:
        (image_info_dict, list_of_annotation_dicts)
        Unknown class names (not in CLASS_MAP) are silently skipped.
        Bounding boxes converted from [xmin, ymin, xmax, ymax] to [x, y, w, h].
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    filename = root.find("filename").text

    image_info = {"id": image_id, "file_name": filename, "width": width, "height": height}

    annotations = []
    ann_id = ann_id_start
    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip()
        if class_name not in CLASS_MAP:
            continue
        bbox_el = obj.find("bndbox")
        xmin = int(float(bbox_el.find("xmin").text))
        ymin = int(float(bbox_el.find("ymin").text))
        xmax = int(float(bbox_el.find("xmax").text))
        ymax = int(float(bbox_el.find("ymax").text))
        w, h = xmax - xmin, ymax - ymin
        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": CLASS_MAP[class_name],
            "bbox": [xmin, ymin, w, h],
            "area": w * h,
            "iscrowd": 0,
        })
        ann_id += 1

    return image_info, annotations


import imagehash
from PIL import Image as PILImage
from sklearn.model_selection import StratifiedGroupKFold


def find_duplicates(image_paths: list, threshold: int = 8) -> set:
    """Return set of indices to remove (keeps first occurrence of near-duplicates).

    Uses perceptual hash (pHash). Two images with hamming distance < threshold
    are considered duplicates. Typical value: threshold=8 for video sequences.
    """
    hashes = []
    for path in image_paths:
        img = PILImage.open(path).convert("L")
        hashes.append(imagehash.phash(img))

    to_remove: set = set()
    for i in range(len(hashes)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(hashes)):
            if j in to_remove:
                continue
            if hashes[i] - hashes[j] < threshold:
                to_remove.add(j)
    return to_remove


def stratified_split(
    ids: list,
    labels: list,
    groups: list,
    ratios: tuple = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple:
    """Split ids into (train, val, test) with stratification and group isolation.

    Stratification ensures rare classes appear in all splits.
    Group isolation prevents leakage between video sequences from the same source.

    Args:
        ids: List of image IDs.
        labels: Primary class label per image (same length as ids).
        groups: Source-group identifier per image (e.g. video sequence name).
        ratios: (train, val, test) fractions summing to 1.0.
        seed: Random seed for reproducibility.

    Returns:
        (train_ids, val_ids, test_ids)
    """
    ids_arr = list(ids)
    labels_arr = list(labels)
    groups_arr = list(groups)
    n = len(ids_arr)

    # First split: separate test set
    splitter = StratifiedGroupKFold(n_splits=max(2, round(1 / ratios[2])), shuffle=True, random_state=seed)
    splits = list(splitter.split(ids_arr, labels_arr, groups_arr))
    # Use last fold's test indices as test set
    _, test_idx = splits[-1]
    trainval_idx = [i for i in range(n) if i not in set(test_idx)]

    # Second split: separate val from train
    trainval_ids = [ids_arr[i] for i in trainval_idx]
    trainval_labels = [labels_arr[i] for i in trainval_idx]
    trainval_groups = [groups_arr[i] for i in trainval_idx]

    val_frac = ratios[1] / (ratios[0] + ratios[1])
    splitter2 = StratifiedGroupKFold(
        n_splits=max(2, round(1 / val_frac)), shuffle=True, random_state=seed
    )
    splits2 = list(splitter2.split(trainval_ids, trainval_labels, trainval_groups))
    _, val_local_idx = splits2[-1]
    train_local_idx = [i for i in range(len(trainval_ids)) if i not in set(val_local_idx)]

    train_ids = [trainval_ids[i] for i in train_local_idx]
    val_ids = [trainval_ids[i] for i in val_local_idx]
    test_ids = [ids_arr[i] for i in test_idx]

    return train_ids, val_ids, test_ids
