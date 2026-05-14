# PV Thermal Defect Detection — Training Pipeline Design

**Date:** 2026-05-13
**Scope:** End-to-end pipeline in `training/` to fine-tune EfficientDet-Lite0 on open IR PV datasets, export an EdgeTPU-compiled `.tflite`, and integrate with `task detect` for drone thermal camera inference.

---

## Goal

Train a model that detects photovoltaic panel defects from thermal drone imagery and deploys on the Coral USB Accelerator. The pipeline runs locally on Apple Silicon (Metal GPU) using TensorFlow Model Maker.

---

## Directory Structure

```
training/
  scripts/
    download_data.py      # fetch datasets from Kaggle + GitHub → data/raw/
    preprocess.py         # normalize IR images, unify annotations → COCO JSON, split
    train.py              # EfficientDet-Lite0 fine-tuning via Model Maker
    export.py             # quantize (int8) + EdgeTPU compile + verify
  notebooks/
    evaluate.ipynb        # mAP, confusion matrix, sample predictions
  data/                   # gitignored — raw + processed datasets
  models/                 # gitignored — checkpoints + exported .tflite files
  requirements.txt        # tensorflow-metal, tflite-model-maker, kagglehub, imagehash, etc.
  README.md               # end-to-end usage guide with commands
```

Root `Taskfile.yml` gains a `training:` namespace:

| Task | What it does |
|---|---|
| `task training:setup` | Create venv, install requirements.txt |
| `task training:download` | Run download_data.py |
| `task training:preprocess` | Run preprocess.py |
| `task training:train` | Run train.py |
| `task training:export` | Run export.py |
| `task training:deploy` | Copy pv_detector_edgetpu.tflite + pv_labels.txt into models/ |

---

## Data Sources

Three open datasets, fetched and unified automatically:

### 1. InfraredSolarModules (primary)
- **Source:** Kaggle — `afsharshamsi/infrared-solar-modules`
- **Fetch:** `kagglehub.dataset_download("afsharshamsi/infrared-solar-modules")`
- **Size:** ~2,400 images
- **Annotations:** PASCAL VOC XML bounding boxes
- **Classes:** hotspot, bypass_diode_failure, multi_hotspot, shadowing, soiling, diode+hotspot (merged → bypass_diode_failure)

### 2. PVDN / PV-Hawk (secondary)
- **Source:** GitHub — `LukasBommes/PV-Hawk` releases
- **Fetch:** curl from GitHub releases API (latest tagged release)
- **Size:** ~1,000 sequences (~3 frames each, deduplicated to ~800 unique)
- **Annotations:** Polygon masks → converted to axis-aligned bounding boxes
- **Classes:** anomaly (mapped to hotspot), bypass (mapped to bypass_diode_failure), disconnected (mapped to delamination)

### 3. Thermographic PV Systems (supplemental)
- **Source:** Kaggle — `marcosgabriel/thermographic-images-of-photovoltaic-systems`
- **Fetch:** `kagglehub.dataset_download("marcosgabriel/thermographic-images-of-photovoltaic-systems")`
- **Size:** ~700 images
- **Annotations:** Binary class (normal/anomaly) — anomaly images get a full-image bounding box labeled `hotspot`

### Unified Class Map

| ID | Label | Source datasets |
|---|---|---|
| 0 | hotspot | all three |
| 1 | bypass_diode_failure | InfraredSolarModules, PVDN |
| 2 | soiling | InfraredSolarModules |
| 3 | multi_hotspot | InfraredSolarModules |
| 4 | shadowing | InfraredSolarModules |
| 5 | delamination | PVDN |

Combined: ~4,100 images, split 80/10/10 (train/val/test), stratified by class, seeded for reproducibility.

---

## Preprocessing (`preprocess.py`)

Steps run in sequence:

1. **IR normalization** — All images converted to 8-bit grayscale via percentile clipping (2nd–98th percentile), then stacked to 3-channel (R=G=B) so EfficientDet's pretrained ImageNet stem transfers without modification. 16-bit TIFF and radiometric JPEG both handled.

2. **Annotation unification** — VOC XML → COCO JSON (InfraredSolarModules); polygon masks → axis-aligned bounding boxes (PVDN); binary classification → full-region bounding box (Thermographic). Output: single `annotations_all.json` in COCO format.

3. **Deduplication** — Perceptual hash (pHash via `imagehash` library) removes near-duplicate frames from PVDN video sequences. Threshold: hamming distance < 8.

4. **Train/val/test split** — Stratified by class using `sklearn.model_selection.StratifiedGroupKFold` (grouped by source video to prevent sequence leakage). Saved as `data/processed/train.json`, `val.json`, `test.json`.

5. **Augmentation** — Handled at training time by Model Maker (horizontal flip, random crop, brightness jitter ±20%). No offline augmentation step needed.

---

## Training (`train.py`)

- **Framework:** TensorFlow Model Maker (`tflite_model_maker.object_detector`)
- **Architecture:** EfficientDet-Lite0 — smallest variant, highest EdgeTPU compatibility
- **Pretrained checkpoint:** COCO (downloaded automatically by Model Maker)
- **GPU:** Apple Metal via `tensorflow-metal` — automatic, zero config
- **Hyperparameters:**
  - Epochs: 50
  - Batch size: 8
  - Learning rate: 0.05, cosine decay
  - Input resolution: 320×320 (EfficientDet-Lite0 native)
- **Checkpoint:** Best val mAP saved to `training/models/checkpoint/`
- **Progress:** per-epoch mAP printed to stdout; full TensorBoard logs in `training/models/logs/`

---

## Export & Verification (`export.py`)

Three outputs:

| File | Description |
|---|---|
| `training/models/pv_detector_float.tflite` | Full-precision, for accuracy baseline |
| `training/models/pv_detector_quant.tflite` | int8 post-training quantization (200 representative val images) |
| `training/models/pv_detector_edgetpu.tflite` | EdgeTPU-compiled via `edgetpu_compiler` |

**EdgeTPU verification step** (critical):
- Parses `edgetpu_compiler` stdout to count ops compiled vs. delegated to CPU
- Asserts ≥95% of ops compile to EdgeTPU (EfficientDet-Lite0 should achieve 100%)
- If <95%: prints warning listing which ops fell back, suggests fixes
- If `edgetpu_compiler` not found: falls back to `pv_detector_quant.tflite` with a clear warning

---

## Evaluation (`notebooks/evaluate.ipynb`)

Loads `test.json`, runs inference using `pv_detector_float.tflite` (for maximum accuracy baseline):

- Per-class precision, recall, F1 table
- mAP@0.5 and mAP@0.5:0.95
- Sample images: predicted boxes vs. ground truth, colour-coded by class
- Confusion matrix (predicted class vs. true class at IoU 0.5)
- Quantization accuracy delta: float vs. int8 mAP comparison

---

## Integration with `task detect`

`detect.py` gains a `--model` flag. New Taskfile tasks:

```yaml
task detect-pv:          # PV model, headless
task detect-pv-display:  # PV model, OpenCV window
```

PV-specific display settings (applied when PV model active):
- Confidence threshold: 0.35 (lower than COCO's 0.4 — thermal anomalies are often low-contrast)
- Per-class colors (BGR):
  - hotspot → (0, 0, 220) red
  - bypass_diode_failure → (0, 100, 255) orange
  - soiling → (0, 220, 255) yellow
  - multi_hotspot → (0, 0, 160) dark red
  - shadowing → (200, 80, 0) blue
  - delamination → (200, 0, 200) magenta

`task training:deploy` copies `pv_detector_edgetpu.tflite` and `pv_labels.txt` into `models/` to make them available to `task detect-pv`.

---

## Python Environment

Separate venv from inference environment — Model Maker requires TF 2.x which conflicts with pycoral's TFLite runtime.

- Python 3.10 (via `uv`)
- Key dependencies: `tensorflow-metal`, `tflite-model-maker`, `kagglehub`, `imagehash`, `scikit-learn`, `pycocotools`, `jupyter`, `matplotlib`, `seaborn`
- `training/requirements.txt` pins all versions for reproducibility

---

## Prerequisites

- Kaggle API credentials (`~/.kaggle/kaggle.json`) for dataset downloads
- `edgetpu_compiler` installed (already present if `task install-edgetpu` was run)
- ~10 GB free disk for raw datasets + processed images + checkpoints
- Apple Silicon Mac with at least 16 GB RAM recommended for batch size 8
