"""Preprocess PV thermal images and unify annotations to COCO JSON format."""
import cv2
import numpy as np


# Unified class map: covers all name variants across all three datasets
CLASS_MAP = {
    # InfraredSolarModules (module_metadata.json anomaly_class values)
    "Hot-Spot": 0, "Cell": 0,
    "Hot-Spot-Multi": 3, "Cell-Multi": 3,
    "Diode": 1, "Diode-Multi": 1,
    "Soiling": 2,
    "Shadowing": 4, "Vegetation": 4,
    "Cracking": 5, "Offline-Module": 5,
    # Generic / other dataset variants
    "hotspot": 0, "Hotspot": 0, "Hot Spot": 0, "HotSpot": 0,
    "bypass_diode_failure": 1, "Bypass Diode": 1, "BypassDiode": 1, "bypass diode": 1,
    "soiling": 2, "multi_hotspot": 3, "shadowing": 4,
    "delamination": 5, "Delamination": 5,
    # PVDN generic labels
    "anomaly": 0, "bypass": 1, "disconnected": 5,
}

# ISM classes to skip (no useful detection annotation possible)
_ISM_SKIP = {"No-Anomaly"}

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


import json
import shutil
from collections import Counter


def _primary_class(image_id: int, annotations: list) -> int:
    """Return most common category_id for a given image_id, or 0 if none."""
    cats = [a["category_id"] for a in annotations if a["image_id"] == image_id]
    return Counter(cats).most_common(1)[0][0] if cats else 0


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess PV thermal datasets to COCO JSON")
    parser.add_argument("--raw", type=Path, required=True, help="Raw data directory (from download_data.py)")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for processed data")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    images_out = args.output / "images"
    images_out.mkdir(exist_ok=True)

    all_images: list = []
    all_annotations: list = []
    image_id = 1
    ann_id = 1

    # ── 1. InfraredSolarModules (module_metadata.json classification labels) ────
    # Dataset: 20,000 single-module crops (24×40px grayscale), 12 anomaly classes.
    # No bounding boxes — each anomaly image gets a full-image bbox.
    # No-Anomaly images are skipped (nothing to detect).
    ism_dir = args.raw / "infrared_solar_modules"
    ism_meta = ism_dir / "module_metadata.json"
    if ism_meta.exists():
        print("Processing InfraredSolarModules...")
        ism_start = image_id
        with open(ism_meta) as f:
            ism_data = json.load(f)
        for key in sorted(ism_data.keys(), key=lambda k: int(k)):
            entry = ism_data[key]
            anomaly_class = entry.get("anomaly_class", "No-Anomaly")
            if anomaly_class in _ISM_SKIP:
                continue
            cat_id = CLASS_MAP.get(anomaly_class)
            if cat_id is None:
                continue
            img_path = ism_dir / entry["image_filepath"]
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            normalized = normalize_ir_image(img)
            out_name = f"ism_{image_id:05d}.jpg"
            cv2.imwrite(str(images_out / out_name), normalized)
            h_img, w_img = normalized.shape[:2]
            all_images.append({
                "id": image_id, "file_name": out_name,
                "width": w_img, "height": h_img, "source_group": "ism",
            })
            all_annotations.append({
                "id": ann_id, "image_id": image_id, "category_id": cat_id,
                "bbox": binary_to_full_bbox(w_img, h_img), "area": w_img * h_img, "iscrowd": 0,
            })
            ann_id += 1
            image_id += 1
        print(f"  InfraredSolarModules: {image_id - ism_start} anomaly images (No-Anomaly skipped)")

    # ── 2. PVDN (polygon mask JSON or unannotated frames) ─────────────────────
    pvdn_dir = args.raw / "pvdn"
    if pvdn_dir.exists():
        print("Processing PVDN...")
        pvdn_start = image_id
        for img_path in sorted(pvdn_dir.rglob("*.jpg")):
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            normalized = normalize_ir_image(img)
            out_name = f"pvdn_{image_id:05d}.jpg"
            cv2.imwrite(str(images_out / out_name), normalized)
            h_img, w_img = normalized.shape[:2]

            img_info = {
                "id": image_id, "file_name": out_name,
                "width": w_img, "height": h_img,
                "source_group": img_path.parent.name,
            }
            json_path = img_path.with_suffix(".json")
            anns_for_img = []
            if json_path.exists():
                with open(json_path) as f:
                    mask_data = json.load(f)
                for obj in mask_data.get("objects", []):
                    label = obj.get("label", "anomaly")
                    cat_id = CLASS_MAP.get(label, 0)
                    polygon = obj.get("polygon", [])
                    bbox = polygon_to_bbox(polygon) if polygon else binary_to_full_bbox(w_img, h_img)
                    anns_for_img.append({
                        "id": ann_id, "image_id": image_id, "category_id": cat_id,
                        "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0,
                    })
                    ann_id += 1
            else:
                anns_for_img.append({
                    "id": ann_id, "image_id": image_id, "category_id": 0,
                    "bbox": binary_to_full_bbox(w_img, h_img), "area": w_img * h_img, "iscrowd": 0,
                })
                ann_id += 1

            all_images.append(img_info)
            all_annotations.extend(anns_for_img)
            image_id += 1
        print(f"  PVDN: {image_id - pvdn_start} images")

    # ── 3. Thermographic PV (binary classification, anomaly images only) ──────
    therm_dir = args.raw / "thermographic_pv"
    if therm_dir.exists():
        print("Processing Thermographic PV Systems...")
        therm_start = image_id
        anomaly_dirs = [d for d in therm_dir.rglob("*")
                        if d.is_dir() and "anomal" in d.name.lower()]
        for img_path in sorted(p for d in anomaly_dirs for p in d.rglob("*.jpg")):
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            normalized = normalize_ir_image(img)
            out_name = f"therm_{image_id:05d}.jpg"
            cv2.imwrite(str(images_out / out_name), normalized)
            h_img, w_img = normalized.shape[:2]
            all_images.append({
                "id": image_id, "file_name": out_name,
                "width": w_img, "height": h_img, "source_group": "therm",
            })
            all_annotations.append({
                "id": ann_id, "image_id": image_id, "category_id": 0,
                "bbox": binary_to_full_bbox(w_img, h_img), "area": w_img * h_img, "iscrowd": 0,
            })
            ann_id += 1
            image_id += 1
        print(f"  Thermographic PV: {image_id - therm_start} images")

    print(f"\nTotal images before deduplication: {len(all_images)}")

    # ── 4. Deduplicate ────────────────────────────────────────────────────────
    paths_for_dedup = [images_out / img["file_name"] for img in all_images]
    dupes = find_duplicates(paths_for_dedup, threshold=8)
    print(f"Removing {len(dupes)} near-duplicate images")
    keep_indices = [i for i in range(len(all_images)) if i not in dupes]
    kept_ids = {all_images[i]["id"] for i in keep_indices}
    all_images = [all_images[i] for i in keep_indices]
    all_annotations = [a for a in all_annotations if a["image_id"] in kept_ids]
    print(f"Total images after deduplication: {len(all_images)}")

    # ── 5. Stratified split ───────────────────────────────────────────────────
    img_ids = [img["id"] for img in all_images]
    labels = [_primary_class(iid, all_annotations) for iid in img_ids]
    groups = [img.get("source_group", "default") for img in all_images]
    train_ids, val_ids, test_ids = stratified_split(img_ids, labels, groups, seed=42)

    # ── 6. Write COCO JSON splits ─────────────────────────────────────────────
    id_to_img = {img["id"]: img for img in all_images}
    id_to_anns: dict = {}
    for ann in all_annotations:
        id_to_anns.setdefault(ann["image_id"], []).append(ann)

    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_imgs = [id_to_img[iid] for iid in split_ids]
        split_anns = [a for iid in split_ids for a in id_to_anns.get(iid, [])]
        coco = {"images": split_imgs, "annotations": split_anns, "categories": CATEGORIES}
        out_path = args.output / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(coco, f, indent=2)
        print(f"  {split_name}: {len(split_imgs)} images, {len(split_anns)} annotations → {out_path}")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
