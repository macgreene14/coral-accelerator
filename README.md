# Coral USB Accelerator

Image classification via a [Coral USB Accelerator](https://coral.ai/products/accelerator/)
and laptop camera, running on macOS.

## Prerequisites

- Coral USB Accelerator connected via USB 3
- [go-task](https://taskfile.dev) (`brew install go-task`)
- macOS (Intel or Apple Silicon — see Troubleshooting for Apple Silicon)

## Quickstart

```bash
task install-edgetpu   # download and install Coral EdgeTPU runtime (sudo, one-time)
task install           # set up Python 3.9 venv and dependencies
task download-models   # fetch MobileNet v2 model and labels
task classify          # point camera and run
```

## Expected output

```
coffee mug               94.1%  |  cup                         3.2%  |  espresso               1.4%
```

Top-3 labels with confidence, updated each frame. Press `Ctrl+C` to stop.

## Commands

| Command | Description |
|---|---|
| `task install-edgetpu` | Install Coral EdgeTPU runtime (requires sudo) |
| `task install` | Install dependencies |
| `task download-models` | Download model files |
| `task classify` | Run camera classification |
| `task test` | Run unit tests (no hardware required) |
| `task detect` | Object detection (coming soon) |

## Troubleshooting

**"Coral USB Accelerator not detected"**
Check USB connection. The device appears as vendor `0x1a6e` (bootloader) or `0x18d1` (after firmware load).

**"libedgetpu not found"**
Run `task install-edgetpu`.

**"Model not found"**
Run `task download-models`.

**pycoral install fails on Apple Silicon**
pycoral may not have arm64 macOS wheels. Try:
```bash
arch -x86_64 task install
```
Then run all tasks as `arch -x86_64 task <name>`.

## Coming soon

- Object detection with bounding box output (`task detect`)
- `--display` flag for a live OpenCV camera preview window
