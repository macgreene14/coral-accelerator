# CLAUDE.md

## What this repo is

A test repo for running ML inference on a Coral USB Accelerator (EdgeTPU) connected to a Mac.
Currently implements image classification via laptop camera; object detection is planned next.

## Python version constraint

pycoral only supports Python 3.9 and below. The `.python-version` file pins 3.9 for uv.
Do not upgrade without first checking pycoral wheel availability for the target Python version.

## Task interface

All operations go through go-task. Don't run scripts directly unless debugging.

| Command | What it does |
|---|---|
| `task install-edgetpu` | Download and install the Coral EdgeTPU runtime (requires sudo, idempotent) |
| `task install` | Install uv (if needed), check libedgetpu, create `.venv` with Python 3.9, install deps |
| `task download-models` | Fetch MobileNet v2 EdgeTPU model and labels into `models/` |
| `task classify` | Run camera classification (requires Coral USB + libedgetpu + downloaded models) |
| `task detect` | Stub — not yet implemented |
| `task test` | Run unit tests (no EdgeTPU or camera required) |

## Apple Silicon note

pycoral may not have arm64 macOS wheels. If `task install` fails with a platform or wheel error, try:

```bash
arch -x86_64 task install
```

Then run all subsequent tasks with `arch -x86_64 task <name>`.

## Models

Models live in `models/` and are not committed to git. Run `task download-models` to fetch them.
The task is idempotent — safe to re-run.

## pycoral import design

In `src/classify.py`, pycoral is imported inside `main()` rather than at module level.
This lets the helper functions (`load_labels`, `preprocess_frame`, `get_top_k`, `find_coral_usb`)
be unit-tested without libedgetpu installed: `task test` works without hardware.

## Next steps

- Implement `src/detect.py` using SSD MobileNet v2 for object detection with bounding boxes
- Add `--display` flag to classify/detect for an OpenCV live preview window
- Add a Dockerfile for non-USB workloads (model conversion, CPU-only TFLite inference, CI)
