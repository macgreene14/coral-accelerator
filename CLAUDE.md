# CLAUDE.md

## What this repo is

A repo for running ML inference on a Coral USB Accelerator (EdgeTPU) connected to a Mac.
Implements image classification and PV solar panel defect detection via laptop camera.
Includes a full training pipeline (`training/`) for fine-tuning EfficientDet-Lite0 on IR thermal imagery.

## Python version constraint

pycoral only supports Python 3.9 and below. The `.python-version` file pins 3.9 for uv.
Do not upgrade without first checking pycoral wheel availability for the target Python version.

## Two venvs

| Venv | Purpose |
|---|---|
| `.venv` | Inference: pycoral, tflite-runtime, opencv. x86_64/Rosetta (required for pycoral). |
| `.venv-training` | Training: tensorflow-macos 2.9.2, tflite-model-maker 0.4.2. arm64 Python 3.9. |

Never mix them. Inference tasks run under `arch -x86_64`; training tasks do not.

## Task interface

All operations go through go-task. Don't run scripts directly unless debugging.

### Inference tasks

| Command | What it does |
|---|---|
| `task install-edgetpu` | Download and install the Coral EdgeTPU runtime (requires sudo, idempotent) |
| `task install` | Install x86_64 uv, create `.venv` with Python 3.9, install pycoral + deps |
| `task download-models` | Fetch MobileNet v2 + SSD COCO EdgeTPU models and labels into `models/` |
| `task classify` | Camera classification — MobileNet v2 on EdgeTPU |
| `task classify-display` | Same with live OpenCV window |
| `task detect` | Camera object detection — SSD MobileNet v2 COCO on EdgeTPU |
| `task detect-display` | Same with live OpenCV window |
| `task detect-pv` | Camera PV defect detection — EfficientDet-Lite0 trained model on EdgeTPU |
| `task detect-pv-display` | Same with live OpenCV window |
| `task test` | Run unit tests (no EdgeTPU or camera required) |

### Training tasks (prefix `training:`)

| Command | What it does |
|---|---|
| `task training:setup` | Create `.venv-training` with tensorflow-macos + tflite-model-maker |
| `task training:download` | Download InfraredSolarModules dataset from Kaggle |
| `task training:preprocess` | Normalize IR images, deduplicate, 80/10/10 split → `training/data/processed/` |
| `task training:train` | Fine-tune EfficientDet-Lite0 for 50 epochs (CPU, ~5 hours) |
| `task training:export` | Compile quantized TFLite for EdgeTPU (requires Docker + colima) |
| `task training:deploy` | Copy compiled model + labels to `models/` |
| `task training:test` | Run training pipeline unit tests |

## Apple Silicon / Rosetta note

pycoral has no arm64 macOS wheels. All inference runs under Rosetta via `arch -x86_64`.
libedgetpu must be installed as x86_64 (run `task install-edgetpu`).

EdgeTPU compiler (`edgetpu_compiler`) is Linux-only. Export runs via Docker:
`colima start` required before `task training:export`.

## Models

Models live in `models/` and are not committed to git (gitignored).

| File | How to get it |
|---|---|
| `mobilenet_v2_1.0_224_quant_edgetpu.tflite` | `task download-models` |
| `ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite` | `task download-models` |
| `pv_detector_edgetpu.tflite` | `task training:export && task training:deploy` |
| `pv_labels.txt` | same |

## PV defect detector

Trained on [InfraredSolarModules](https://github.com/RaptorMaps/InfraredSolarModules) (20k IR images, 12 classes collapsed to 6):
`hotspot`, `bypass_diode_failure`, `soiling`, `multi_hotspot`, `shadowing`, `delamination`.

Training: EfficientDet-Lite0, 50 epochs, 7,717 images, final val_loss=0.511.
EdgeTPU: 264/275 ops on-chip (96%), single subgraph, 5.6 MB, fits entirely in on-chip cache.

### EfficientDet output tensor quirk

pycoral's `detect.get_objects()` fails on EfficientDet-Lite0 models from tflite-model-maker:
the quantized `count` tensor returns one more than the scores array size (zero-point rounding).
`src/detect.py` uses `get_objects_safe()` for the `pv` model — reads tensors directly with
`count = min(count, len(scores))` clamp.

## pycoral import design

pycoral is imported inside `main()` in both `src/classify.py` and `src/detect.py`.
This lets all helper functions be unit-tested without libedgetpu installed: `task test` works without hardware.

## Training venv compatibility patches

`tflite-model-maker 0.4.2` requires several patches on arm64 macOS 15 (applied by `task training:setup`
via `training/scripts/patch_venv.py`, idempotent):

- `tflite_model_maker/__init__.py` — non-object-detector sub-modules wrapped in try/except
- `tensorflow_examples/.../model_spec/__init__.py` — same
- `tensorflow_examples/.../model_util.py` — tensorflowjs / tflite_support imports made lazy
- `tflite_support` stub package — no arm64 macOS wheel; stub satisfies the import

tensorflow-metal is excluded (symbol incompatibility on macOS 15); training runs on CPU.
numpy must be pinned to `==1.23.5` (tensorflowjs 3.18 uses `np.object` removed in 1.24).
