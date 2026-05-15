# PV Thermal Defect Detection — Training Pipeline

End-to-end pipeline to train EfficientDet-Lite0 on open-source IR photovoltaic datasets, export an EdgeTPU-compiled `.tflite`, and deploy it to `task detect-pv`.

---

## Prerequisites

- Apple Silicon Mac (training uses Metal GPU via `tensorflow-metal`)
- [uv](https://github.com/astral-sh/uv) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Kaggle API credentials at `~/.kaggle/kaggle.json` ([get them here](https://www.kaggle.com/account))
- `edgetpu_compiler` installed (`task install-edgetpu` from repo root)
- ~10 GB free disk space

---

## Quick Start

All commands run from the **repo root** directory.

```bash
# 1. Install Python dependencies (separate venv from inference, ~5 min)
task training:setup

# 2. Download datasets from Kaggle + Zenodo (~1.5 GB, ~10 min)
task training:download

# 3. Normalize images, unify annotations, deduplicate, split 80/10/10
task training:preprocess

# 4. Train EfficientDet-Lite0 (50 epochs, Metal GPU, ~2–4 hours)
task training:train

# 5. Compile for EdgeTPU + verify ≥95% ops mapped
task training:export

# 6. Deploy model to inference pipeline
task training:deploy

# 7. Run PV defect detection with live camera
task detect-pv-display
```

---

## Pipeline Steps

### `task training:setup`

Creates `.venv-training/` (Python 3.9, arm64 — **not** Rosetta) and installs:
- `tensorflow-macos==2.9.2` + `tensorflow-metal==0.5.1` (Apple Silicon GPU)
- `tflite-model-maker==0.4.2` (installed `--no-deps` to avoid TF version conflict)
- Data processing: `kagglehub`, `scikit-learn`, `imagehash`, `pycocotools`, `opencv-python`

> **Note:** Do not run `pip install -r requirements.txt` directly — the install order matters for the TF/Model Maker conflict. Always use `task training:setup`.

### `task training:download`

Downloads three open-source datasets to `training/data/raw/`:

| Dataset | Source | Size | Annotations |
|---|---|---|---|
| InfraredSolarModules | Kaggle `afsharshamsi/infrared-solar-modules` | ~2,400 images | Pascal VOC XML bounding boxes |
| Thermographic PV Systems | Kaggle `marcosgabriel/thermographic-images-of-photovoltaic-systems` | ~700 images | Binary anomaly/normal |
| PVDN (Zenodo 3894823) | Zenodo record 3894823 | ~800 images | Polygon masks |

### `task training:preprocess`

Runs `scripts/preprocess.py --raw data/raw --output data/processed`:

1. **IR normalization** — 2nd–98th percentile clipping → 8-bit 3-channel (R=G=B)
2. **Annotation unification** — VOC XML, polygon masks, binary labels all → COCO JSON
3. **Deduplication** — perceptual hash (pHash) removes near-duplicate video frames
4. **Stratified split** — 80/10/10 train/val/test, grouped by source to prevent sequence leakage

Output: `data/processed/{train,val,test}.json` + `data/processed/images/`

### `task training:train`

Runs `scripts/train.py --data data/processed --output models`:

- Architecture: EfficientDet-Lite0 (smallest variant, highest EdgeTPU compatibility)
- Pretrained checkpoint: COCO (downloaded automatically)
- Epochs: 50, batch size: 8, learning rate: 0.05 cosine decay
- Output: `models/pv_detector_float.tflite`, `models/pv_detector_quant.tflite`, `models/pv_labels.txt`

**Dry run** (verifies pipeline in ~2 min, 1 epoch on 20 images):
```bash
.venv-training/bin/python scripts/train.py --data data/processed --output models --dry-run
```

### `task training:export`

Compiles the quantized model with `edgetpu_compiler` and verifies ≥95% of ops run on EdgeTPU.

Output: `models/pv_detector_edgetpu.tflite`

If `edgetpu_compiler` is not found, falls back to the quantized model with a warning.

### `task training:deploy`

Copies `pv_detector_edgetpu.tflite` and `pv_labels.txt` to the repo's `models/` directory, making them available to `task detect-pv`.

### `task training:test`

Runs unit tests (no GPU or dataset required):

```bash
task training:test
```

Tests cover: `normalize_ir_image`, `polygon_to_bbox`, `binary_to_full_bbox`, `voc_xml_to_coco_annotations`, `find_duplicates`, `stratified_split`, `parse_compiler_output`, and the `--model` flag in `detect.py`.

---

## Detection

After `task training:deploy`:

```bash
task detect-pv             # headless (stdout only)
task detect-pv-display     # live OpenCV window
```

Confidence threshold: **0.35** (lower than COCO's 0.4 — thermal anomalies are often low-contrast).

### Classes

| ID | Label | Color |
|---|---|---|
| 0 | hotspot | red |
| 1 | bypass_diode_failure | orange |
| 2 | soiling | yellow |
| 3 | multi_hotspot | dark red |
| 4 | shadowing | blue |
| 5 | delamination | magenta |

---

## Evaluation

Open `notebooks/evaluate.ipynb` (run `task training:setup` first for the Jupyter kernel):

```bash
.venv-training/bin/jupyter notebook training/notebooks/evaluate.ipynb
```

Provides:
- Per-class precision, recall, F1 on the held-out test set
- Sample prediction visualizations (predicted vs. ground truth boxes)
- Quantization accuracy delta (float vs. int8 mAP comparison)

---

## Directory Structure

```
training/
  scripts/
    download_data.py    — fetch datasets from Kaggle + Zenodo → data/raw/
    preprocess.py       — normalize IR images, unify annotations → COCO JSON, split
    train.py            — EfficientDet-Lite0 fine-tuning via Model Maker
    export.py           — quantize (int8) + EdgeTPU compile + verify
  notebooks/
    evaluate.ipynb      — mAP, confusion matrix, sample predictions
  tests/
    test_preprocess.py  — 22 unit tests for preprocessing helpers
    test_export.py      — 5 unit tests for EdgeTPU compiler output parser
    test_detect_model_flag.py — 6 tests for detect.py --model flag
  data/                 — gitignored — raw + processed datasets
  models/               — gitignored — checkpoints + exported .tflite files
  requirements.txt      — pinned dependencies (use task training:setup, not pip install -r)
  README.md             — this file
```
