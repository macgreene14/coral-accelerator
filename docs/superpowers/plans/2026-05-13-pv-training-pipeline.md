# PV Thermal Defect Detection Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end pipeline in `training/` that downloads open IR PV datasets, preprocesses and unifies annotations, fine-tunes EfficientDet-Lite0 via TFLite Model Maker on Apple Silicon Metal GPU, exports an EdgeTPU-compiled `.tflite`, and wires it into `task detect-pv` and `task detect-pv-display`.

**Architecture:** Four CLI scripts (`download_data.py`, `preprocess.py`, `train.py`, `export.py`) for the pipeline stages, a Jupyter notebook for evaluation, and a `training:` Taskfile namespace powered by a separate Python 3.9 venv (arm64, no Rosetta needed — standard TF, not pycoral). Preprocessing outputs COCO JSON; `train.py` converts to Pascal VOC directories for Model Maker ingestion. Unit tests live in `training/tests/` and cover preprocessing helpers and the EdgeTPU output parser.

**Tech Stack:** Python 3.9, tensorflow-macos==2.9.2, tensorflow-metal==0.5.1 (Apple Silicon GPU), tflite-model-maker==0.4.2, kagglehub, imagehash, scikit-learn, pycocotools, edgetpu_compiler (already installed)

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `training/Taskfile.yml` | Create | training: task namespace |
| `training/requirements.txt` | Create | Pinned Python dependencies |
| `training/.gitignore` | Create | Ignore data/, models/, venv |
| `training/README.md` | Create | End-to-end usage guide |
| `training/scripts/__init__.py` | Create | Package marker |
| `training/scripts/download_data.py` | Create | Fetch datasets from Kaggle + Zenodo |
| `training/scripts/preprocess.py` | Create | Normalize IR, unify annotations, dedup, split → COCO JSON |
| `training/scripts/train.py` | Create | EfficientDet-Lite0 fine-tuning + TFLite export |
| `training/scripts/export.py` | Create | EdgeTPU compile + verify |
| `training/notebooks/evaluate.ipynb` | Create | mAP, confusion matrix, sample predictions |
| `training/tests/__init__.py` | Create | Package marker |
| `training/tests/test_preprocess.py` | Create | Unit tests for preprocessing helpers |
| `training/tests/test_export.py` | Create | Unit tests for EdgeTPU output parser |
| `Taskfile.yml` | Modify | Add `includes: training` + detect-pv tasks |
| `src/detect.py` | Modify | Add --model flag, PV model config, per-model threshold |

---

## Task 1: Directory scaffold, requirements.txt, training Taskfile

**Files:**
- Create: `training/Taskfile.yml`
- Create: `training/requirements.txt`
- Create: `training/.gitignore`
- Create: `training/scripts/__init__.py`
- Create: `training/tests/__init__.py`
- Create: `training/notebooks/` (empty dir, gitkeep)
- Modify: `Taskfile.yml` (add includes + detect-pv tasks)

- [ ] **Step 1: Create directory structure**

```bash
cd /Users/macgreene/Documents/coral-accelerator
mkdir -p training/scripts training/tests training/notebooks training/data training/models
touch training/scripts/__init__.py training/tests/__init__.py
touch training/notebooks/.gitkeep training/data/.gitkeep training/models/.gitkeep
```

- [ ] **Step 2: Write training/requirements.txt**

```
tensorflow-macos==2.9.2
tensorflow-metal==0.5.1
tflite-model-maker==0.4.2
kagglehub>=0.2.0
ImageHash>=4.3.1
scikit-learn>=1.3.0
pycocotools>=2.0.7
jupyter>=1.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=9.5.0
numpy>=1.23.0,<2.0
requests>=2.31.0
lxml>=4.9.0
opencv-python>=4.8.0
tqdm>=4.65.0
pytest>=7.0.0
```

- [ ] **Step 3: Write training/.gitignore**

```
data/
models/
../.venv-training/
__pycache__/
*.pyc
.ipynb_checkpoints/
```

- [ ] **Step 4: Write training/Taskfile.yml**

```yaml
version: '3'

vars:
  PYTHON: ../.venv-training/bin/python

tasks:
  setup:
    desc: Create Python 3.9 venv and install training dependencies (handles TF/Model Maker conflict)
    cmds:
      - |
        uv venv ../.venv-training --python 3.9
        PIP="../.venv-training/bin/pip"
        $PIP install --upgrade pip
        # Install TF first — tflite-model-maker pins an older TF, we install ours first
        $PIP install tensorflow-macos==2.9.2 tensorflow-metal==0.5.1
        # Install tflite-model-maker without its TF constraint
        $PIP install tflite-model-maker==0.4.2 --no-deps
        # Install remaining dependencies
        $PIP install kagglehub "ImageHash>=4.3.1" "scikit-learn>=1.3.0" \
          "pycocotools>=2.0.7" jupyter "matplotlib>=3.7.0" "seaborn>=0.12.0" \
          "Pillow>=9.5.0" "numpy>=1.23.0,<2.0" requests lxml \
          "opencv-python>=4.8.0" tqdm "pytest>=7.0.0"

  download:
    desc: Download PV thermal datasets from Kaggle and Zenodo (~1.5 GB, requires ~/.kaggle/kaggle.json)
    cmds:
      - "{{.PYTHON}} scripts/download_data.py --output data/raw"

  preprocess:
    desc: Normalize IR images, unify annotations, deduplicate, split 80/10/10
    cmds:
      - "{{.PYTHON}} scripts/preprocess.py --raw data/raw --output data/processed"

  train:
    desc: Fine-tune EfficientDet-Lite0 on PV thermal dataset (50 epochs, Metal GPU)
    cmds:
      - "{{.PYTHON}} scripts/train.py --data data/processed --output models"

  export:
    desc: Compile quantized TFLite model for EdgeTPU and verify compilation quality
    cmds:
      - "{{.PYTHON}} scripts/export.py --quant models/pv_detector_quant.tflite --output models"

  deploy:
    desc: Copy trained PV model into models/ for use with task detect-pv
    cmds:
      - |
        if [ ! -f "models/pv_detector_edgetpu.tflite" ]; then
          echo "ERROR: pv_detector_edgetpu.tflite not found. Run task training:export first."
          exit 1
        fi
        cp models/pv_detector_edgetpu.tflite ../models/
        cp models/pv_labels.txt ../models/
        echo "Deployed pv_detector_edgetpu.tflite and pv_labels.txt to models/"

  test:
    desc: Run training pipeline unit tests (no GPU or data required)
    cmds:
      - "{{.PYTHON}} -m pytest tests/ -v"
```

- [ ] **Step 5: Add includes and detect-pv tasks to root Taskfile.yml**

Add `includes:` block after `version: '3'` and before `vars:`:

```yaml
version: '3'

includes:
  training:
    taskfile: ./training/Taskfile.yml
    dir: ./training
```

Add these tasks after `detect-display:` and before `test:`:

```yaml
  detect-pv:
    desc: Run PV defect detection using thermal camera and Coral USB Accelerator
    env:
      DYLD_LIBRARY_PATH: /usr/local/lib
    cmds:
      - "arch -x86_64 {{.PYTHON}} src/detect.py --model pv"

  detect-pv-display:
    desc: Run PV defect detection with live OpenCV preview window
    env:
      DYLD_LIBRARY_PATH: /usr/local/lib
    cmds:
      - "arch -x86_64 {{.PYTHON}} src/detect.py --model pv --display"
```

- [ ] **Step 6: Verify setup runs**

```bash
cd /Users/macgreene/Documents/coral-accelerator
task training:setup
```

Expected: `.venv-training/` created, `tensorflow`, `tflite_model_maker`, `kagglehub` all importable:

```bash
.venv-training/bin/python -c "import tensorflow as tf; print(tf.__version__)"
```

Expected: `2.9.2`

- [ ] **Step 7: Commit**

```bash
git add training/ Taskfile.yml
git commit -m "feat: scaffold training/ directory, venv setup, Taskfile training: namespace"
```

---

## Task 2: download_data.py

**Files:**
- Create: `training/scripts/download_data.py`

- [ ] **Step 1: Write training/scripts/download_data.py**

```python
"""Download PV thermal datasets for training.

Sources
-------
1. InfraredSolarModules (Kaggle) — ~2,400 images, Pascal VOC bounding boxes
   Classes: hotspot, bypass diode failure, multi-hotspot, shadowing, soiling
2. Thermographic PV Systems (Kaggle) — ~700 images, binary anomaly/normal
3. PV module inspection dataset (Zenodo record 3894823) — ~800 images, polygon masks

Prerequisites
-------------
  ~/.kaggle/kaggle.json  — Kaggle API credentials
  Run: pip install kagglehub
"""

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

import requests


KAGGLE_DATASETS = [
    {
        "id": "afsharshamsi/infrared-solar-modules",
        "dest": "infrared_solar_modules",
    },
    {
        "id": "marcosgabriel/thermographic-images-of-photovoltaic-systems",
        "dest": "thermographic_pv",
    },
]

ZENODO_URL = "https://zenodo.org/record/3894823/files/module_images.tar.gz"


def download_kaggle(dataset_id: str, output_dir: Path) -> Path:
    """Download a Kaggle dataset via kagglehub and copy to output_dir."""
    import kagglehub

    print(f"Downloading Kaggle dataset: {dataset_id}")
    cache_path = kagglehub.dataset_download(dataset_id)
    dest = output_dir / dataset_id.split("/")[1]
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(cache_path, dest)
    img_count = sum(1 for _ in dest.rglob("*.jpg")) + sum(1 for _ in dest.rglob("*.png"))
    print(f"  {dest.name}: {img_count} images")
    return dest


def download_zenodo(url: str, output_dir: Path) -> Path:
    """Download and extract a Zenodo dataset archive."""
    dest = output_dir / "pvdn"
    dest.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "pvdn.tar.gz"

    if not archive.exists():
        print(f"Downloading: {url}")
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(archive, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.1f}%", end="", flush=True)
        print()

    print(f"Extracting to {dest}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest)
    archive.unlink()
    img_count = sum(1 for _ in dest.rglob("*.jpg")) + sum(1 for _ in dest.rglob("*.png"))
    print(f"  pvdn: {img_count} images")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PV thermal training datasets")
    parser.add_argument("--output", type=Path, required=True, help="Directory for raw data")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["kaggle", "zenodo", "all"],
        default=["all"],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    sources = set(args.sources)
    if "all" in sources:
        sources = {"kaggle", "zenodo"}

    if "kaggle" in sources:
        try:
            for ds in KAGGLE_DATASETS:
                download_kaggle(ds["id"], args.output)
        except Exception as exc:
            print(f"ERROR downloading Kaggle datasets: {exc}", file=sys.stderr)
            print("Ensure ~/.kaggle/kaggle.json exists. Get credentials at kaggle.com/account", file=sys.stderr)
            sys.exit(1)

    if "zenodo" in sources:
        try:
            download_zenodo(ZENODO_URL, args.output)
        except Exception as exc:
            print(f"WARNING: Zenodo download failed: {exc}", file=sys.stderr)
            print("Continuing without PVDN dataset.", file=sys.stderr)

    print(f"\nDownload complete. Raw data: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses args without network calls**

```bash
cd /Users/macgreene/Documents/coral-accelerator
.venv-training/bin/python training/scripts/download_data.py --help
```

Expected: usage message printed, no errors.

- [ ] **Step 3: Commit**

```bash
git add training/scripts/download_data.py
git commit -m "feat: add download_data.py for Kaggle and Zenodo PV datasets"
```

---

## Task 3: preprocess.py — IR normalization helper (TDD)

**Files:**
- Create: `training/tests/test_preprocess.py` (normalize_ir_image tests only)
- Create: `training/scripts/preprocess.py` (normalize_ir_image only)

- [ ] **Step 1: Write failing tests for normalize_ir_image**

Create `training/tests/test_preprocess.py`:

```python
"""Unit tests for preprocess.py helpers. No datasets or GPU required."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from preprocess import normalize_ir_image


class TestNormalizeIrImage:
    def test_grayscale_input_returns_3channel_uint8(self):
        img = np.random.randint(0, 65535, (480, 640), dtype=np.uint16)
        result = normalize_ir_image(img)
        assert result.shape == (480, 640, 3)
        assert result.dtype == np.uint8

    def test_all_three_channels_are_identical(self):
        img = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        result = normalize_ir_image(img)
        np.testing.assert_array_equal(result[:, :, 0], result[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 1], result[:, :, 2])

    def test_output_pixel_range_is_0_to_255(self):
        img = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        result = normalize_ir_image(img)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_bgr_3channel_input_accepted(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 200
        result = normalize_ir_image(img)
        assert result.shape == (100, 100, 3)

    def test_constant_image_returns_zeros(self):
        # p2 == p98 for constant input → no dynamic range → return zeros
        img = np.full((100, 100), 5000, dtype=np.uint16)
        result = normalize_ir_image(img)
        assert result.max() == 0

    def test_uint8_input_normalizes_correctly(self):
        img = np.array([[0, 127, 255]], dtype=np.uint8)
        result = normalize_ir_image(img)
        assert result.dtype == np.uint8
        # After percentile clip of a 3-element array: min=0, max=255
        assert result.max() == 255
```

- [ ] **Step 2: Run to verify tests fail**

```bash
cd /Users/macgreene/Documents/coral-accelerator
.venv-training/bin/python -m pytest training/tests/test_preprocess.py -v
```

Expected: `ModuleNotFoundError: No module named 'preprocess'`

- [ ] **Step 3: Implement normalize_ir_image in preprocess.py**

Create `training/scripts/preprocess.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv-training/bin/python -m pytest training/tests/test_preprocess.py::TestNormalizeIrImage -v
```

Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add training/scripts/preprocess.py training/tests/test_preprocess.py
git commit -m "feat: add normalize_ir_image with tests (TDD)"
```

---

## Task 4: preprocess.py — annotation converters (TDD)

**Files:**
- Modify: `training/tests/test_preprocess.py`
- Modify: `training/scripts/preprocess.py`

- [ ] **Step 1: Add failing tests for annotation converters**

Append to `training/tests/test_preprocess.py`:

```python
from preprocess import binary_to_full_bbox, polygon_to_bbox, voc_xml_to_coco_annotations


class TestPolygonToBbox:
    def test_rectangle_polygon_returns_xywh(self):
        polygon = [[10, 20], [50, 20], [50, 80], [10, 80]]
        assert polygon_to_bbox(polygon) == [10, 20, 40, 60]

    def test_diamond_polygon_uses_bounding_box(self):
        polygon = [[0, 10], [20, 0], [40, 10], [20, 30]]
        assert polygon_to_bbox(polygon) == [0, 0, 40, 30]

    def test_single_point_returns_zero_size(self):
        assert polygon_to_bbox([[5, 5]]) == [5, 5, 0, 0]


class TestBinaryToFullBbox:
    def test_returns_full_image_dimensions(self):
        assert binary_to_full_bbox(640, 480) == [0, 0, 640, 480]

    def test_square_image(self):
        assert binary_to_full_bbox(300, 300) == [0, 0, 300, 300]


class TestVocXmlToCocoAnnotations:
    def test_parses_single_hotspot_object(self, tmp_path):
        xml = tmp_path / "test.xml"
        xml.write_text(
            "<annotation>"
            "<filename>test.jpg</filename>"
            "<size><width>640</width><height>480</height><depth>3</depth></size>"
            "<object><name>hotspot</name>"
            "<bndbox><xmin>100</xmin><ymin>150</ymin><xmax>200</xmax><ymax>250</ymax></bndbox>"
            "</object></annotation>"
        )
        img_info, anns = voc_xml_to_coco_annotations(xml, image_id=1, ann_id_start=0)
        assert img_info == {"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}
        assert len(anns) == 1
        assert anns[0]["category_id"] == 0  # hotspot
        assert anns[0]["bbox"] == [100, 150, 100, 100]  # [x, y, w, h]
        assert anns[0]["area"] == 10000

    def test_skips_unknown_class(self, tmp_path):
        xml = tmp_path / "test.xml"
        xml.write_text(
            "<annotation>"
            "<filename>test.jpg</filename>"
            "<size><width>640</width><height>480</height><depth>3</depth></size>"
            "<object><name>unknown_defect</name>"
            "<bndbox><xmin>0</xmin><ymin>0</ymin><xmax>100</xmax><ymax>100</ymax></bndbox>"
            "</object></annotation>"
        )
        _, anns = voc_xml_to_coco_annotations(xml, image_id=1, ann_id_start=0)
        assert len(anns) == 0

    def test_multiple_objects_sequential_ids(self, tmp_path):
        xml = tmp_path / "multi.xml"
        xml.write_text(
            "<annotation>"
            "<filename>multi.jpg</filename>"
            "<size><width>640</width><height>480</height><depth>3</depth></size>"
            "<object><name>hotspot</name>"
            "<bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>"
            "</object>"
            "<object><name>Soiling</name>"
            "<bndbox><xmin>100</xmin><ymin>100</ymin><xmax>200</xmax><ymax>200</ymax></bndbox>"
            "</object></annotation>"
        )
        _, anns = voc_xml_to_coco_annotations(xml, image_id=5, ann_id_start=10)
        assert len(anns) == 2
        assert anns[0]["id"] == 10
        assert anns[1]["id"] == 11
        assert anns[1]["category_id"] == 2  # soiling

    def test_float_bbox_coords_truncated_to_int(self, tmp_path):
        xml = tmp_path / "float.xml"
        xml.write_text(
            "<annotation>"
            "<filename>float.jpg</filename>"
            "<size><width>640</width><height>480</height><depth>3</depth></size>"
            "<object><name>hotspot</name>"
            "<bndbox><xmin>10.7</xmin><ymin>20.3</ymin><xmax>50.9</xmax><ymax>60.1</ymax></bndbox>"
            "</object></annotation>"
        )
        _, anns = voc_xml_to_coco_annotations(xml, image_id=1, ann_id_start=0)
        assert anns[0]["bbox"] == [10, 20, 40, 40]
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv-training/bin/python -m pytest training/tests/test_preprocess.py -v -k "Polygon or Bbox or VocXml"
```

Expected: `ImportError` — functions not yet defined.

- [ ] **Step 3: Implement the three converter functions in preprocess.py**

Append to `training/scripts/preprocess.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify all pass**

```bash
.venv-training/bin/python -m pytest training/tests/test_preprocess.py -v
```

Expected: all tests pass (6 normalize + 3 polygon + 2 binary + 4 voc = 15 tests).

- [ ] **Step 5: Commit**

```bash
git add training/scripts/preprocess.py training/tests/test_preprocess.py
git commit -m "feat: add annotation converter helpers with tests (TDD)"
```

---

## Task 5: preprocess.py — deduplication + stratified split (TDD)

**Files:**
- Modify: `training/tests/test_preprocess.py`
- Modify: `training/scripts/preprocess.py`

- [ ] **Step 1: Add failing tests for find_duplicates and stratified_split**

Append to `training/tests/test_preprocess.py`:

```python
from preprocess import find_duplicates, stratified_split
from PIL import Image as PILImage


class TestFindDuplicates:
    def test_identical_images_all_but_first_removed(self, tmp_path):
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        paths = []
        for i in range(3):
            p = tmp_path / f"img{i}.jpg"
            PILImage.fromarray(arr).save(p)
            paths.append(p)
        dupes = find_duplicates(paths, threshold=8)
        assert len(dupes) == 2  # keep index 0, remove 1 and 2

    def test_different_images_not_flagged(self, tmp_path):
        paths = []
        for i in range(3):
            arr = np.zeros((64, 64, 3), dtype=np.uint8)
            arr[:, :, 0] = i * 80  # clearly different
            p = tmp_path / f"img{i}.jpg"
            PILImage.fromarray(arr).save(p)
            paths.append(p)
        dupes = find_duplicates(paths, threshold=8)
        assert len(dupes) == 0

    def test_returns_set_of_indices(self, tmp_path):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        p = tmp_path / "single.jpg"
        PILImage.fromarray(arr).save(p)
        dupes = find_duplicates([p], threshold=8)
        assert isinstance(dupes, set)
        assert len(dupes) == 0  # single image → no duplicates


class TestStratifiedSplit:
    def test_split_ratios_approximately_correct(self):
        ids = list(range(100))
        labels = [i % 3 for i in range(100)]
        groups = [str(i % 10) for i in range(100)]
        train, val, test = stratified_split(ids, labels, groups, ratios=(0.8, 0.1, 0.1))
        assert abs(len(train) - 80) <= 5
        assert abs(len(val) - 10) <= 5
        assert abs(len(test) - 10) <= 5

    def test_splits_are_disjoint(self):
        ids = list(range(100))
        labels = [i % 3 for i in range(100)]
        groups = [str(i % 10) for i in range(100)]
        train, val, test = stratified_split(ids, labels, groups)
        assert set(train).isdisjoint(set(val))
        assert set(train).isdisjoint(set(test))
        assert set(val).isdisjoint(set(test))

    def test_all_ids_appear_exactly_once(self):
        ids = list(range(100))
        labels = [i % 3 for i in range(100)]
        groups = [str(i % 10) for i in range(100)]
        train, val, test = stratified_split(ids, labels, groups)
        assert sorted(train + val + test) == ids

    def test_seeded_split_is_reproducible(self):
        ids = list(range(50))
        labels = [i % 2 for i in range(50)]
        groups = [str(i % 5) for i in range(50)]
        r1 = stratified_split(ids, labels, groups, seed=42)
        r2 = stratified_split(ids, labels, groups, seed=42)
        assert r1 == r2
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv-training/bin/python -m pytest training/tests/test_preprocess.py -v -k "Duplicates or Split"
```

Expected: `ImportError` — functions not yet defined.

- [ ] **Step 3: Implement find_duplicates and stratified_split**

Append to `training/scripts/preprocess.py`:

```python
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
    import numpy as np as_np

    ids_arr = list(ids)
    labels_arr = list(labels)
    groups_arr = list(groups)
    n = len(ids_arr)

    # First split: separate test set
    n_test = max(1, round(n * ratios[2]))
    n_val = max(1, round(n * ratios[1]))

    splitter = StratifiedGroupKFold(n_splits=max(2, round(1 / ratios[2])), shuffle=True, random_state=seed)
    splits = list(splitter.split(ids_arr, labels_arr, groups_arr))
    # Use last fold's test indices as test set
    _, test_idx = splits[-1]
    trainval_idx = [i for i in range(n) if i not in set(test_idx)]

    # Second split: separate val from train
    trainval_ids = [ids_arr[i] for i in trainval_idx]
    trainval_labels = [labels_arr[i] for i in trainval_idx]
    trainval_groups = [groups_arr[i] for i in trainval_idx]

    n_val_splits = max(2, round(len(trainval_ids) * ratios[1] / (ratios[0] + ratios[1])))
    # Invert: val fraction from trainval
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
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
.venv-training/bin/python -m pytest training/tests/test_preprocess.py -v
```

Expected: all tests pass (15 previous + 7 new = 22 tests).

- [ ] **Step 5: Commit**

```bash
git add training/scripts/preprocess.py training/tests/test_preprocess.py
git commit -m "feat: add find_duplicates and stratified_split with tests (TDD)"
```

---

## Task 6: preprocess.py — orchestrator main()

**Files:**
- Modify: `training/scripts/preprocess.py`

- [ ] **Step 1: Append main() to preprocess.py**

```python
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

    # ── 1. InfraredSolarModules (Pascal VOC) ──────────────────────────────────
    ism_dir = args.raw / "infrared_solar_modules"
    if ism_dir.exists():
        print("Processing InfraredSolarModules...")
        for xml_path in sorted(ism_dir.rglob("*.xml")):
            img_path = next(
                (xml_path.with_suffix(ext) for ext in (".jpg", ".jpeg", ".png")
                 if xml_path.with_suffix(ext).exists()),
                None,
            )
            if img_path is None:
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            normalized = normalize_ir_image(img)
            out_name = f"ism_{image_id:05d}.jpg"
            cv2.imwrite(str(images_out / out_name), normalized)
            h, w = normalized.shape[:2]

            img_info, anns = voc_xml_to_coco_annotations(xml_path, image_id, ann_id)
            img_info["file_name"] = out_name
            img_info["width"] = w
            img_info["height"] = h
            img_info["source_group"] = xml_path.parent.name
            all_images.append(img_info)
            all_annotations.extend(anns)
            ann_id += len(anns)
            image_id += 1
        print(f"  InfraredSolarModules: {image_id - 1} images")

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
```

- [ ] **Step 2: Run existing tests to confirm nothing broke**

```bash
.venv-training/bin/python -m pytest training/tests/test_preprocess.py -v
```

Expected: all 22 tests pass.

- [ ] **Step 3: Commit**

```bash
git add training/scripts/preprocess.py
git commit -m "feat: add preprocess.py orchestrator main() — normalize, unify, dedup, split"
```

---

## Task 7: train.py + training:train task

**Files:**
- Create: `training/scripts/train.py`

- [ ] **Step 1: Write training/scripts/train.py**

```python
"""Fine-tune EfficientDet-Lite0 on PV thermal data using TFLite Model Maker.

Usage
-----
  python train.py --data data/processed --output models
  python train.py --data data/processed --output models --dry-run  # 1 epoch on 20 images

Apple Silicon note
------------------
  tensorflow-metal is loaded automatically when detected. No code changes needed.
  Verify with: python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
  Expected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
"""

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


# Model Maker uses 1-based label IDs; id=0 is background
LABEL_MAP = {
    1: "hotspot",
    2: "bypass_diode_failure",
    3: "soiling",
    4: "multi_hotspot",
    5: "shadowing",
    6: "delamination",
}


def coco_to_pascal_voc(coco_json_path: Path, images_dir: Path, output_dir: Path) -> None:
    """Convert COCO JSON annotations to Pascal VOC directory structure.

    Creates:
      output_dir/images/      — JPEG images (copied from images_dir)
      output_dir/Annotations/ — VOC XML files (one per image)

    Model Maker's DataLoader.from_pascal_voc() expects this layout.
    Category ids in COCO (0-based) are shifted +1 to match LABEL_MAP (1-based).
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    imgs_out = output_dir / "images"
    anns_out = output_dir / "Annotations"
    imgs_out.mkdir(parents=True, exist_ok=True)
    anns_out.mkdir(parents=True, exist_ok=True)

    cat_map = {cat["id"]: cat["name"] for cat in coco["categories"]}
    img_to_anns: dict = {}
    for ann in coco["annotations"]:
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    for img_info in coco["images"]:
        src = images_dir / img_info["file_name"]
        if not src.exists():
            continue
        shutil.copy2(src, imgs_out / img_info["file_name"])

        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = img_info["file_name"]
        sz = ET.SubElement(root, "size")
        ET.SubElement(sz, "width").text = str(img_info["width"])
        ET.SubElement(sz, "height").text = str(img_info["height"])
        ET.SubElement(sz, "depth").text = "3"

        for ann in img_to_anns.get(img_info["id"], []):
            x, y, w, h = ann["bbox"]
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = cat_map[ann["category_id"]]
            bb = ET.SubElement(obj, "bndbox")
            ET.SubElement(bb, "xmin").text = str(int(x))
            ET.SubElement(bb, "ymin").text = str(int(y))
            ET.SubElement(bb, "xmax").text = str(int(x + w))
            ET.SubElement(bb, "ymax").text = str(int(y + h))

        stem = Path(img_info["file_name"]).stem
        ET.ElementTree(root).write(anns_out / f"{stem}.xml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EfficientDet-Lite0 for PV defect detection")
    parser.add_argument("--data", type=Path, required=True, help="Processed data dir (contains train.json, val.json, images/)")
    parser.add_argument("--output", type=Path, required=True, help="Output dir for models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run 1 epoch on a 20-image subset — verifies pipeline without full training"
    )
    args = parser.parse_args()

    import tensorflow as tf
    from tflite_model_maker import object_detector
    from tflite_model_maker.config import QuantizationConfig
    from tflite_model_maker.object_detector import DataLoader

    print(f"TensorFlow: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU devices: {gpus}")
    if not gpus:
        print("WARNING: No Metal GPU detected. Training on CPU (slower).")

    args.output.mkdir(parents=True, exist_ok=True)
    voc_dir = args.data / "voc"

    # Convert COCO → Pascal VOC if not already done
    for split in ("train", "val"):
        voc_split = voc_dir / split
        if not voc_split.exists():
            print(f"Converting {split}.json → Pascal VOC at {voc_split} ...")
            coco_to_pascal_voc(args.data / f"{split}.json", args.data / "images", voc_split)

    print("Loading data...")
    train_data = DataLoader.from_pascal_voc(
        images_dir=str(voc_dir / "train" / "images"),
        annotations_dir=str(voc_dir / "train" / "Annotations"),
        label_map=LABEL_MAP,
    )
    val_data = DataLoader.from_pascal_voc(
        images_dir=str(voc_dir / "val" / "images"),
        annotations_dir=str(voc_dir / "val" / "Annotations"),
        label_map=LABEL_MAP,
    )

    if args.dry_run:
        # Slice to 20 training + 10 val images for a quick pipeline check
        train_data = train_data.split(0.1)[0]
        val_data = val_data.split(0.2)[0]

    print(f"Training images: {len(train_data)}, Validation images: {len(val_data)}")

    spec = object_detector.EfficientDetSpec(
        model_name="efficientdet-lite0",
        uri="https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1",
        hparams={"learning_rate": 0.05, "lr_warmup_init": 0.005},
    )

    epochs = 1 if args.dry_run else args.epochs
    batch_size = 2 if args.dry_run else args.batch_size
    print(f"Training: {epochs} epoch(s), batch size {batch_size}")

    model = object_detector.create(
        train_data,
        model_spec=spec,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        train_whole_model=False,
    )

    if args.dry_run:
        print("Dry run complete — pipeline verified.")
        return

    # Export float TFLite
    float_path = args.output / "pv_detector_float.tflite"
    model.export(export_dir=str(args.output), tflite_filename="pv_detector_float.tflite")
    print(f"Float TFLite: {float_path}")

    # Export int8 quantized TFLite (uses val set as representative dataset)
    quant_config = QuantizationConfig.for_int8(representative_data=val_data)
    model.export(
        export_dir=str(args.output),
        tflite_filename="pv_detector_quant.tflite",
        quantization_config=quant_config,
    )
    print(f"Quantized TFLite: {args.output / 'pv_detector_quant.tflite'}")

    # Write labels file
    labels_path = args.output / "pv_labels.txt"
    labels_path.write_text("\n".join(LABEL_MAP.values()) + "\n")
    print(f"Labels: {labels_path}")
    print("\nTraining complete. Run: task training:export")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script imports and --help work**

```bash
cd /Users/macgreene/Documents/coral-accelerator
.venv-training/bin/python training/scripts/train.py --help
```

Expected: usage message with `--data`, `--output`, `--epochs`, `--batch-size`, `--dry-run`.

- [ ] **Step 3: Run tests to confirm nothing in test suite broke**

```bash
.venv-training/bin/python -m pytest training/tests/ -v
```

Expected: 22 tests pass.

- [ ] **Step 4: Commit**

```bash
git add training/scripts/train.py
git commit -m "feat: add train.py — EfficientDet-Lite0 fine-tuning with Model Maker, float+quant export"
```

---

## Task 8: export.py — EdgeTPU compiler output parser (TDD)

**Files:**
- Create: `training/tests/test_export.py`
- Create: `training/scripts/export.py` (parse_compiler_output only)

- [ ] **Step 1: Write failing tests**

Create `training/tests/test_export.py`:

```python
"""Unit tests for export.py helpers. No GPU or model files required."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from export import parse_compiler_output


_FULL_COMPILE = """
Edge TPU Compiler version 16.0.384591198
INFO: Model compiled successfully in 1942 ms.

Input model: pv_detector_quant.tflite
Output model: pv_detector_quant_edgetpu.tflite
Total number of operations: 191
Operation log: pv_detector_quant_edgetpu.log

Operator                       Count      Status

CONV_2D                        123        Mapped to Edge TPU
DEPTHWISE_CONV_2D              36         Mapped to Edge TPU
MUL                            20         Mapped to Edge TPU
ADD                            12         Mapped to Edge TPU
"""

_PARTIAL_COMPILE = """
Edge TPU Compiler version 16.0.384591198
INFO: Model compiled successfully in 1200 ms.

Total number of operations: 100
Operation log: model_edgetpu.log

Operator                       Count      Status

CONV_2D                        80         Mapped to Edge TPU
RESHAPE                        15         More than one subgraph is not supported
GATHER                         5          Operation not supported
"""

_EMPTY = ""


class TestParseCompilerOutput:
    def test_full_compile_all_ops_mapped(self):
        result = parse_compiler_output(_FULL_COMPILE)
        assert result["total_ops"] == 191
        assert result["compiled_ops"] == 191
        assert result["cpu_ops"] == 0
        assert result["compiled_pct"] == 100.0

    def test_partial_compile_counts_cpu_fallbacks(self):
        result = parse_compiler_output(_PARTIAL_COMPILE)
        assert result["total_ops"] == 100
        assert result["compiled_ops"] == 80
        assert result["cpu_ops"] == 20
        assert abs(result["compiled_pct"] - 80.0) < 0.01

    def test_empty_output_returns_zeros(self):
        result = parse_compiler_output(_EMPTY)
        assert result["total_ops"] == 0
        assert result["compiled_ops"] == 0
        assert result["cpu_ops"] == 0
        assert result["compiled_pct"] == 0.0

    def test_returns_dict_with_all_keys(self):
        result = parse_compiler_output(_FULL_COMPILE)
        assert set(result.keys()) == {"total_ops", "compiled_ops", "cpu_ops", "compiled_pct"}

    def test_compiled_pct_is_float(self):
        result = parse_compiler_output(_FULL_COMPILE)
        assert isinstance(result["compiled_pct"], float)
```

- [ ] **Step 2: Run to verify tests fail**

```bash
cd /Users/macgreene/Documents/coral-accelerator
.venv-training/bin/python -m pytest training/tests/test_export.py -v
```

Expected: `ModuleNotFoundError: No module named 'export'`

- [ ] **Step 3: Create training/scripts/export.py with parse_compiler_output**

```python
"""Export quantized TFLite model to EdgeTPU-compiled format and verify quality."""
import shutil
import subprocess
import sys
from pathlib import Path


def parse_compiler_output(stdout: str) -> dict:
    """Parse edgetpu_compiler stdout and return compilation statistics.

    The compiler emits lines like:
      "Total number of operations: 191"
      "CONV_2D    123    Mapped to Edge TPU"
      "RESHAPE    5      More than one subgraph is not supported"

    Returns:
        dict with keys:
          total_ops (int)   — total ops in model
          compiled_ops (int) — ops mapped to EdgeTPU
          cpu_ops (int)     — ops that will run on host CPU
          compiled_pct (float) — percentage mapped to EdgeTPU
    """
    total_ops = 0
    compiled_ops = 0

    for line in stdout.splitlines():
        if "Total number of operations:" in line:
            try:
                total_ops = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
            continue
        if "Mapped to Edge TPU" in line:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    compiled_ops += int(parts[1])
                except ValueError:
                    pass

    cpu_ops = total_ops - compiled_ops
    compiled_pct = (compiled_ops / total_ops * 100.0) if total_ops > 0 else 0.0

    return {
        "total_ops": total_ops,
        "compiled_ops": compiled_ops,
        "cpu_ops": cpu_ops,
        "compiled_pct": compiled_pct,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv-training/bin/python -m pytest training/tests/test_export.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add training/scripts/export.py training/tests/test_export.py
git commit -m "feat: add parse_compiler_output with tests (TDD)"
```

---

## Task 9: export.py — full pipeline

**Files:**
- Modify: `training/scripts/export.py`

- [ ] **Step 1: Append compile_edgetpu and main() to export.py**

```python
_MIN_EDGETPU_PCT = 95.0  # warn if fewer than this % of ops compile to EdgeTPU


def compile_edgetpu(quant_path: Path, output_dir: Path) -> Path:
    """Compile quantized TFLite model for EdgeTPU.

    Calls edgetpu_compiler, parses output, asserts ≥95% ops compiled.
    Falls back to copying the quantized model if compiler is not found.

    Returns:
        Path to pv_detector_edgetpu.tflite in output_dir.
    """
    compiler = shutil.which("edgetpu_compiler")
    fallback_path = output_dir / "pv_detector_edgetpu.tflite"

    if not compiler:
        print("WARNING: edgetpu_compiler not found in PATH.", file=sys.stderr)
        print("Falling back to quantized model (no EdgeTPU acceleration).", file=sys.stderr)
        shutil.copy2(quant_path, fallback_path)
        return fallback_path

    print(f"Compiling for EdgeTPU: {quant_path.name}")
    result = subprocess.run(
        [compiler, str(quant_path), "-o", str(output_dir)],
        capture_output=True,
        text=True,
    )
    combined = result.stdout + result.stderr
    stats = parse_compiler_output(combined)

    print(
        f"EdgeTPU: {stats['compiled_ops']}/{stats['total_ops']} ops "
        f"({stats['compiled_pct']:.1f}%) mapped to EdgeTPU"
    )

    if stats["total_ops"] > 0 and stats["compiled_pct"] < _MIN_EDGETPU_PCT:
        print(
            f"WARNING: Only {stats['compiled_pct']:.1f}% of ops run on EdgeTPU "
            f"({stats['cpu_ops']} ops fall back to CPU).",
            file=sys.stderr,
        )
        print(
            "EfficientDet-Lite0 should achieve ~100%. "
            "Check for unsupported ops in the compiler log.",
            file=sys.stderr,
        )

    # edgetpu_compiler writes <stem>_edgetpu.tflite; rename to canonical name
    compiled_src = output_dir / (quant_path.stem + "_edgetpu.tflite")
    if compiled_src.exists() and compiled_src != fallback_path:
        compiled_src.rename(fallback_path)
    elif not fallback_path.exists():
        # compiler may have failed silently — fall back
        print("WARNING: compiler produced no output. Using quantized model.", file=sys.stderr)
        shutil.copy2(quant_path, fallback_path)

    return fallback_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compile quantized TFLite model for EdgeTPU")
    parser.add_argument("--quant", type=Path, required=True,
                        help="Path to pv_detector_quant.tflite (from task training:train)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for pv_detector_edgetpu.tflite")
    args = parser.parse_args()

    if not args.quant.exists():
        print(f"ERROR: {args.quant} not found. Run task training:train first.", file=sys.stderr)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    edgetpu_path = compile_edgetpu(args.quant, args.output)

    print(f"\nExport complete:")
    print(f"  EdgeTPU model: {edgetpu_path}")
    labels_src = args.quant.parent / "pv_labels.txt"
    if labels_src.exists() and labels_src != args.output / "pv_labels.txt":
        shutil.copy2(labels_src, args.output / "pv_labels.txt")
        print(f"  Labels:        {args.output / 'pv_labels.txt'}")
    print("\nRun: task training:deploy")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all training tests**

```bash
.venv-training/bin/python -m pytest training/tests/ -v
```

Expected: all tests pass (22 preprocess + 5 export = 27 tests).

- [ ] **Step 3: Verify --help works**

```bash
.venv-training/bin/python training/scripts/export.py --help
```

Expected: usage with `--quant` and `--output`.

- [ ] **Step 4: Commit**

```bash
git add training/scripts/export.py
git commit -m "feat: add export.py compile_edgetpu with ≥95% op verification and fallback"
```

---

## Task 10: detect.py —model flag + detect-pv Taskfile tasks

**Files:**
- Modify: `src/detect.py`
- Modify: `training/tests/test_detect_model_flag.py` (new file — keeps inference test suite clean)

- [ ] **Step 1: Write failing tests for --model flag**

Create `training/tests/test_detect_model_flag.py`:

```python
"""Tests for detect.py --model flag. Run from repo root with .venv/bin/python."""
import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestModelFlag:
    def _make_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", choices=["coco", "pv"], default="coco")
        parser.add_argument("--threshold", type=float, default=None)
        parser.add_argument("--display", action="store_true")
        return parser

    def test_model_defaults_to_coco(self):
        args = self._make_parser().parse_args([])
        assert args.model == "coco"

    def test_pv_model_accepted(self):
        args = self._make_parser().parse_args(["--model", "pv"])
        assert args.model == "pv"

    def test_threshold_defaults_to_none(self):
        args = self._make_parser().parse_args([])
        assert args.threshold is None

    def test_threshold_override_accepted(self):
        args = self._make_parser().parse_args(["--model", "pv", "--threshold", "0.5"])
        assert args.threshold == 0.5

    def test_model_configs_present(self):
        from detect import _MODEL_CONFIGS
        assert "coco" in _MODEL_CONFIGS
        assert "pv" in _MODEL_CONFIGS
        for cfg in _MODEL_CONFIGS.values():
            assert "model" in cfg
            assert "labels" in cfg
            assert "default_threshold" in cfg

    def test_pv_default_threshold_lower_than_coco(self):
        from detect import _MODEL_CONFIGS
        assert _MODEL_CONFIGS["pv"]["default_threshold"] < _MODEL_CONFIGS["coco"]["default_threshold"]
```

- [ ] **Step 2: Run to verify tests fail**

```bash
cd /Users/macgreene/Documents/coral-accelerator
.venv/bin/python -m pytest training/tests/test_detect_model_flag.py -v
```

Expected: `ImportError: cannot import name '_MODEL_CONFIGS' from 'detect'`

- [ ] **Step 3: Add _MODEL_CONFIGS and --model flag to src/detect.py**

Add after the `LABELS_PATH` line at the top of `src/detect.py`:

```python
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
```

In `main()`, replace the argparse block:

```python
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
```

Replace the `MODEL_PATH` and `LABELS_PATH` existence checks with:

```python
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
```

Replace `make_interpreter(str(MODEL_PATH))` with `make_interpreter(str(model_path))`.
Replace `load_labels(LABELS_PATH)` with `load_labels(labels_path)`.
Replace `coral_detect.get_objects(interpreter, args.threshold)` with `coral_detect.get_objects(interpreter, threshold)`.
Replace the `print(f"Running detection (threshold={args.threshold:.0%})` line with:

```python
    print(f"Running detection — model={args.model}, threshold={threshold:.0%} — press Ctrl+C to stop.\n")
```

- [ ] **Step 4: Run inference tests to verify nothing broke**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: all 34 inference tests pass.

- [ ] **Step 5: Run new model flag tests**

```bash
.venv/bin/python -m pytest training/tests/test_detect_model_flag.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/detect.py training/tests/test_detect_model_flag.py
git commit -m "feat: add --model flag to detect.py with PV config and per-model threshold defaults"
```

---

## Task 11: evaluate.ipynb + training/README.md

**Files:**
- Create: `training/notebooks/evaluate.ipynb`
- Create: `training/README.md`

- [ ] **Step 1: Write training/notebooks/evaluate.ipynb**

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# PV Defect Detector — Evaluation\n", "Evaluate the trained model on the held-out test set.\n", "Run `task training:train` and `task training:export` before this notebook."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "from pathlib import Path\n",
    "\n",
    "import cv2\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "from tflite_runtime.interpreter import Interpreter\n",
    "\n",
    "DATA_DIR = Path('../data/processed')\n",
    "MODELS_DIR = Path('../models')\n",
    "\n",
    "LABELS = ['hotspot', 'bypass_diode_failure', 'soiling', 'multi_hotspot', 'shadowing', 'delamination']\n",
    "COLORS = ['#e74c3c', '#e67e22', '#f1c40f', '#c0392b', '#3498db', '#9b59b6']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load test annotations\n",
    "with open(DATA_DIR / 'test.json') as f:\n",
    "    test_coco = json.load(f)\n",
    "\n",
    "print(f\"Test images: {len(test_coco['images'])}\")\n",
    "print(f\"Test annotations: {len(test_coco['annotations'])}\")\n",
    "\n",
    "from collections import Counter\n",
    "class_counts = Counter(a['category_id'] for a in test_coco['annotations'])\n",
    "for cid, count in sorted(class_counts.items()):\n",
    "    print(f\"  {LABELS[cid]}: {count}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load float model for evaluation\n",
    "interpreter = Interpreter(model_path=str(MODELS_DIR / 'pv_detector_float.tflite'))\n",
    "interpreter.allocate_tensors()\n",
    "input_details = interpreter.get_input_details()\n",
    "output_details = interpreter.get_output_details()\n",
    "_, h, w, _ = input_details[0]['shape']\n",
    "print(f'Input shape: {h}x{w}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run inference on all test images\n",
    "from tflite_runtime.interpreter import load_delegate\n",
    "\n",
    "predictions = []\n",
    "images_dir = DATA_DIR / 'images'\n",
    "\n",
    "for img_info in test_coco['images']:\n",
    "    img = cv2.imread(str(images_dir / img_info['file_name']))\n",
    "    if img is None:\n",
    "        continue\n",
    "    resized = cv2.resize(img, (w, h))\n",
    "    inp = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0\n",
    "    interpreter.set_tensor(input_details[0]['index'], inp)\n",
    "    interpreter.invoke()\n",
    "    boxes = interpreter.get_tensor(output_details[0]['index'])[0]\n",
    "    class_ids = interpreter.get_tensor(output_details[1]['index'])[0]\n",
    "    scores = interpreter.get_tensor(output_details[2]['index'])[0]\n",
    "    num = int(interpreter.get_tensor(output_details[3]['index'])[0])\n",
    "    predictions.append({'image_id': img_info['id'], 'boxes': boxes[:num],\n",
    "                         'class_ids': class_ids[:num].astype(int), 'scores': scores[:num]})\n",
    "\n",
    "print(f'Inference complete on {len(predictions)} images')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Precision / Recall per class at threshold=0.35\n",
    "THRESHOLD = 0.35\n",
    "IOU_THRESH = 0.5\n",
    "\n",
    "def iou(boxA, boxB):\n",
    "    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])\n",
    "    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])\n",
    "    inter = max(0, xB - xA) * max(0, yB - yA)\n",
    "    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])\n",
    "    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])\n",
    "    return inter / (areaA + areaB - inter + 1e-6)\n",
    "\n",
    "gt_by_img = {}\n",
    "for ann in test_coco['annotations']:\n",
    "    gt_by_img.setdefault(ann['image_id'], []).append(ann)\n",
    "\n",
    "tp = [0] * 6; fp = [0] * 6; fn_count = [0] * 6\n",
    "\n",
    "for pred in predictions:\n",
    "    gt_anns = gt_by_img.get(pred['image_id'], [])\n",
    "    matched = set()\n",
    "    for i, (score, cls_id) in enumerate(zip(pred['scores'], pred['class_ids'])):\n",
    "        if score < THRESHOLD:\n",
    "            continue\n",
    "        # Convert normalised [y1,x1,y2,x2] to pixel\n",
    "        box = pred['boxes'][i]  # [ymin, xmin, ymax, xmax] normalised\n",
    "        best_iou, best_j = 0, -1\n",
    "        for j, gt in enumerate(gt_anns):\n",
    "            if j in matched or gt['category_id'] != cls_id:\n",
    "                continue\n",
    "            x, y, bw, bh = gt['bbox']\n",
    "            gt_norm = [y/gt.get('h',480), x/gt.get('w',640),\n",
    "                       (y+bh)/gt.get('h',480), (x+bw)/gt.get('w',640)]\n",
    "            this_iou = iou(box, gt_norm)\n",
    "            if this_iou > best_iou:\n",
    "                best_iou, best_j = this_iou, j\n",
    "        if best_iou >= IOU_THRESH:\n",
    "            tp[cls_id] += 1; matched.add(best_j)\n",
    "        else:\n",
    "            fp[cls_id] += 1\n",
    "    for j, gt in enumerate(gt_anns):\n",
    "        if j not in matched:\n",
    "            fn_count[gt['category_id']] += 1\n",
    "\n",
    "print(f'{'Class':<25} {'TP':>5} {'FP':>5} {'FN':>5} {'Precision':>10} {'Recall':>8}')\n",
    "print('-' * 65)\n",
    "for i, label in enumerate(LABELS):\n",
    "    prec = tp[i] / (tp[i] + fp[i] + 1e-6)\n",
    "    rec  = tp[i] / (tp[i] + fn_count[i] + 1e-6)\n",
    "    print(f'{label:<25} {tp[i]:>5} {fp[i]:>5} {fn_count[i]:>5} {prec:>10.1%} {rec:>8.1%}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualise sample predictions (first 6 test images)\n",
    "fig, axes = plt.subplots(2, 3, figsize=(15, 10))\n",
    "for ax, pred in zip(axes.flat, predictions[:6]):\n",
    "    img_info = next(i for i in test_coco['images'] if i['id'] == pred['image_id'])\n",
    "    img = cv2.imread(str(images_dir / img_info['file_name']))\n",
    "    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "    ih, iw = img_rgb.shape[:2]\n",
    "    for score, cls_id, box in zip(pred['scores'], pred['class_ids'], pred['boxes']):\n",
    "        if score < THRESHOLD:\n",
    "            continue\n",
    "        y1, x1, y2, x2 = int(box[0]*ih), int(box[1]*iw), int(box[2]*ih), int(box[3]*iw)\n",
    "        color = tuple(int(c*255) for c in plt.cm.tab10(cls_id)[:3])\n",
    "        cv2.rectangle(img_rgb, (x1,y1), (x2,y2), color, 2)\n",
    "        cv2.putText(img_rgb, f'{LABELS[cls_id]} {score:.0%}', (x1, max(y1-5,0)),\n",
    "                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)\n",
    "    ax.imshow(img_rgb)\n",
    "    ax.axis('off')\n",
    "plt.suptitle('Sample Predictions (float model, threshold=0.35)', fontsize=14)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Confusion matrix\n",
    "conf_matrix = np.zeros((6, 6), dtype=int)\n",
    "for pred in predictions:\n",
    "    gt_anns = gt_by_img.get(pred['image_id'], [])\n",
    "    for j, gt in enumerate(gt_anns):\n",
    "        best_score, best_pred_cls = 0, -1\n",
    "        for i, (score, cls_id) in enumerate(zip(pred['scores'], pred['class_ids'])):\n",
    "            if score >= THRESHOLD and score > best_score:\n",
    "                best_score, best_pred_cls = score, cls_id\n",
    "        if best_pred_cls >= 0:\n",
    "            conf_matrix[gt['category_id'], best_pred_cls] += 1\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 6))\n",
    "sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=LABELS, yticklabels=LABELS,\n",
    "            cmap='Blues', ax=ax)\n",
    "ax.set_xlabel('Predicted'); ax.set_ylabel('Ground Truth')\n",
    "ax.set_title('Confusion Matrix (IoU ≥ 0.5)')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.9.0"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Write training/README.md**

```markdown
# PV Thermal Defect Detection — Training Pipeline

End-to-end pipeline to train an EfficientDet-Lite0 model for detecting PV panel defects
from thermal drone imagery and deploy it to a Coral USB Accelerator.

## Prerequisites

- Apple Silicon Mac (M1/M2/M3) with macOS 13+
- Kaggle account with API credentials at `~/.kaggle/kaggle.json`
  - Get credentials: https://www.kaggle.com/settings → API → Create New Token
- `edgetpu_compiler` installed (run `task install-edgetpu` from repo root if not)
- ~10 GB free disk space

## Quick Start

```bash
# From repo root
task training:setup      # Create .venv-training, install TF + Model Maker
task training:download   # Download ~1.5 GB of IR PV datasets
task training:preprocess # Normalize, unify annotations, deduplicate, split 80/10/10
task training:train      # Fine-tune EfficientDet-Lite0 (~1-3 hours on M-series GPU)
task training:export     # Compile for EdgeTPU (requires edgetpu_compiler)
task training:deploy     # Copy model to models/ → ready for task detect-pv
```

After deploy, run the PV detector:
```bash
task detect-pv           # Headless (prints detections to stdout)
task detect-pv-display   # Live OpenCV window with bounding boxes
```

## Datasets

| Dataset | Source | Images | Format |
|---|---|---|---|
| InfraredSolarModules | Kaggle: `afsharshamsi/infrared-solar-modules` | ~2,400 | Pascal VOC |
| Thermographic PV | Kaggle: `marcosgabriel/thermographic-images-of-photovoltaic-systems` | ~700 | Binary class |
| PVDN | Zenodo record 3894823 | ~800 | Polygon masks |

## Defect Classes

| ID | Label | Description |
|---|---|---|
| 0 | hotspot | Single-cell thermal anomaly |
| 1 | bypass_diode_failure | String-level heating from failed bypass diode |
| 2 | soiling | Panel-level uniform heating from dirt/dust |
| 3 | multi_hotspot | Multiple hotspots in same cell area |
| 4 | shadowing | Shading-induced thermal pattern |
| 5 | delamination | Structural separation causing thermal anomaly |

## Dry Run (no data needed — tests the pipeline with synthetic data)

```bash
# After task training:preprocess, verify training works before waiting 3 hours:
.venv-training/bin/python training/scripts/train.py \
  --data training/data/processed --output training/models --dry-run
```

## Evaluation

After training and export:

```bash
cd training
../.venv-training/bin/jupyter notebook notebooks/evaluate.ipynb
```

The notebook computes per-class precision/recall, mAP@0.5, displays sample predictions,
and shows a confusion matrix.

## Unit Tests

```bash
task training:test
```

Runs 27+ unit tests covering preprocessing helpers (IR normalization, annotation converters,
deduplication, stratified split) and EdgeTPU compiler output parsing. No GPU or datasets required.

## Directory Structure

```
training/
  scripts/          Python pipeline scripts
  notebooks/        evaluate.ipynb
  tests/            Unit tests (no hardware required)
  data/             Downloaded and processed data (gitignored)
  models/           Trained models and exports (gitignored)
  requirements.txt  Pinned Python dependencies
```

## Troubleshooting

**`tflite_model_maker` import error after setup**
Re-run setup — the installation order matters (TF before Model Maker with `--no-deps`):
```bash
task training:setup
```

**Metal GPU not detected during training**
Verify tensorflow-metal is installed: `.venv-training/bin/pip show tensorflow-metal`
Training falls back to CPU automatically — it's slower but correct.

**EdgeTPU compilation <95% ops compiled**
EfficientDet-Lite0 should compile at ~100%. If you see fallbacks, check the compiler log
at `training/models/pv_detector_quant_edgetpu.log` for unsupported op names.

**Kaggle download fails**
Ensure `~/.kaggle/kaggle.json` exists and contains `{"username": "...", "key": "..."}`.
Set permissions: `chmod 600 ~/.kaggle/kaggle.json`
```

- [ ] **Step 3: Run all tests one final time**

```bash
cd /Users/macgreene/Documents/coral-accelerator
.venv/bin/python -m pytest tests/ -v              # inference tests (34)
.venv-training/bin/python -m pytest training/tests/ -v  # training tests (27)
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add training/notebooks/evaluate.ipynb training/README.md
git commit -m "feat: add evaluate.ipynb and training/README.md"
```

---

## Self-Review

**Spec coverage check:**
- ✅ `training/` directory with scripts/, notebooks/, tests/, data/, models/ — Task 1
- ✅ `training/Taskfile.yml` with setup/download/preprocess/train/export/deploy/test — Task 1
- ✅ Root Taskfile includes + detect-pv / detect-pv-display tasks — Task 1
- ✅ `download_data.py` — InfraredSolarModules (Kaggle), Thermographic PV (Kaggle), PVDN (Zenodo) — Task 2
- ✅ IR normalization (percentile clip 2nd-98th, 3-channel) — Task 3
- ✅ Annotation unification: VOC XML, polygon→bbox, binary→full-image bbox — Task 4
- ✅ Deduplication (pHash, hamming < 8) — Task 5
- ✅ Stratified split 80/10/10, grouped by source sequence — Task 5
- ✅ preprocess.py main() orchestrator — Task 6
- ✅ train.py with EfficientDet-Lite0, Model Maker, Metal GPU, hyperparams (50 epochs, batch 8, lr 0.05) — Task 7
- ✅ float TFLite + int8 quantized TFLite exported by train.py — Task 7
- ✅ `parse_compiler_output` with ≥95% threshold warning — Task 8
- ✅ `compile_edgetpu` with fallback if compiler missing — Task 9
- ✅ export.py main() — Task 9
- ✅ `_MODEL_CONFIGS` in detect.py with coco/pv configs — Task 10
- ✅ --model flag, per-model threshold defaults (0.4/0.35) — Task 10
- ✅ Error messages with `task training:deploy` hint for missing PV model — Task 10
- ✅ evaluate.ipynb with precision/recall, sample predictions, confusion matrix — Task 11
- ✅ training/README.md with end-to-end guide — Task 11

**No placeholders found.**

**Type consistency:** `parse_compiler_output` returns `dict` with keys `total_ops`, `compiled_ops`, `cpu_ops`, `compiled_pct` — consistent between test_export.py (Task 8) and export.py usage in Task 9. `_MODEL_CONFIGS` dict shape consistent between test_detect_model_flag.py (Task 10) and detect.py definition.
