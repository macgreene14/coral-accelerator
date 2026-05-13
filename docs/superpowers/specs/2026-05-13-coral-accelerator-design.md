---
title: Coral USB Accelerator Test Repo
date: 2026-05-13
status: approved
---

# Coral USB Accelerator Test Repo

## Purpose

A local test repo for running inference on a Coral USB Accelerator (EdgeTPU) connected to a Mac. Starts with image classification via laptop camera; detection is planned next. All tooling is task-driven via go-task with uv managing the Python environment.

## Repo Structure

```
coral-accelerator/
├── CLAUDE.md
├── README.md
├── Taskfile.yml
├── pyproject.toml          # uv-managed, pins Python 3.9
├── .python-version         # pins 3.9 for uv
├── src/
│   ├── classify.py         # camera + EdgeTPU classification (working)
│   └── detect.py           # object detection (stub)
└── models/
    └── .gitkeep            # models downloaded at setup, not committed
```

## Tooling

- **go-task** — canonical interface for all operations; already installed at `/opt/homebrew/bin/task`
- **uv** — installed by `task install` if missing; manages Python 3.9 venv at `.venv/`
- **Python 3.9** — required for pycoral compatibility; uv downloads it automatically (no Homebrew needed)
- **Docker** — not used for the core workflow (macOS USB passthrough limitation); reserved for future non-USB workloads (model conversion, CI)

## Taskfile Tasks

| Task | Description |
|---|---|
| `task install` | Installs uv (if missing), checks for libedgetpu (prints install URL if absent), creates Python 3.9 venv, installs dependencies |
| `task download-models` | Fetches MobileNet v2 EdgeTPU `.tflite` and labels from Google Coral model zoo into `models/` |
| `task classify` | Runs `src/classify.py` — camera → EdgeTPU → top-3 labels + confidence in terminal |
| `task detect` | Stub — prints helpful message pointing to `task classify` |

## Dependencies (`pyproject.toml`)

- `pycoral` — Coral Python SDK (Python 3.9 max)
- `tflite-runtime` — lightweight TFLite inference runtime
- `opencv-python` — camera capture and frame preprocessing

## Inference Script (`src/classify.py`)

Linear flow:

1. **Device check** — detect Coral USB via USB vendor ID (`0x1a6e` bootloader or `0x18d1` runtime); exit with clear error if not found
2. **Model load** — load EdgeTPU-compiled `.tflite` from `models/`; initialize EdgeTPU delegate via pycoral
3. **Camera open** — open default camera (index 0) via OpenCV; exit with clear error if unavailable
4. **Capture loop** — grab frame → resize to 224×224 → run inference → print top-3 labels + confidence → repeat until `Ctrl+C`
5. **Cleanup** — release camera handle on exit

Output is terminal-only. No GUI window in MVP. A `--display` flag for OpenCV preview is a natural future extension.

## Stub Script (`src/detect.py`)

Prints: `"Detection not yet implemented — run 'task classify' instead"`. Placeholder for SSD MobileNet object detection once classification is validated.

## Setup Constraints

- **libedgetpu**: Requires the Coral runtime `.pkg` installer (macOS). `task install` detects if it is missing and prints the install URL with instructions rather than attempting a silent install.
- **Python 3.9**: Not available in Homebrew on Apple Silicon by default. uv handles the download and pin automatically via `.python-version`.
- **Models**: Downloaded from Google's Coral model zoo (`storage.googleapis.com`). Not committed to git. `task download-models` is idempotent (skips if files exist).

## Future Work

- `src/detect.py` — SSD MobileNet v2 object detection with bounding box output
- `--display` flag on classify/detect for OpenCV preview window
- Dockerfile for non-USB workloads (model conversion, CPU-only TFLite inference, CI)
