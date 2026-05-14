# Display UI & Object Detection Design

**Date:** 2026-05-13
**Scope:** Add `--display` OpenCV window to `classify.py`; implement `detect.py` with SSD MobileNet v2 and `--display`; extend `task download-models` for detection assets.

---

## Architecture

Two scripts, independent, both gaining a `--display` flag via `argparse`.

### classify.py changes
- Add `argparse` with a single `--display` boolean flag.
- Existing headless loop is unchanged when flag is absent.
- When `--display` is set, each iteration calls `draw_classification_overlay(frame, top)` then `cv2.imshow("Classify", frame)`.
- `cv2.waitKey(1)` checked each frame; `q` key quits (in addition to Ctrl+C).
- `draw_classification_overlay` is a pure helper (takes frame + top-k list, returns annotated frame) — unit-testable without hardware.

### detect.py (new)
- Mirrors `classify.py` structure: pre-import checks, deferred pycoral imports, `_EDGETPU_SHARED_LIB` patch, camera loop.
- Uses `pycoral.adapters.detect` and `pycoral.adapters.common`.
- Confidence threshold: `--threshold` flag, default `0.4`.
- `--display` flag: enables `draw_detections(frame, detections)` helper + `cv2.imshow("Detect", frame)`.
- Headless output (no `--display`): single line per frame, e.g. `person 94.1%  chair 61.3%`.
- `draw_detections` is a pure helper — unit-testable without hardware.

### Taskfile.yml additions
| Task | Command |
|---|---|
| `task classify-display` | `arch -x86_64 .venv/bin/python src/classify.py --display` |
| `task detect` | `arch -x86_64 .venv/bin/python src/detect.py` |
| `task detect-display` | `arch -x86_64 .venv/bin/python src/detect.py --display` |

`task detect` replaces the existing stub.

---

## Display UI

### classify overlay
- Semi-transparent dark panel anchored to bottom-left of the frame.
- Three rows, one per top-k result.
- Each row: label name (left), filled confidence bar (center), percentage (right).
- Bar color: green (≥70%), yellow (40–69%), red (<40%).
- Rendered with `cv2.rectangle` + `cv2.putText`; overlay blended with `cv2.addWeighted`.

### detect overlay
- Bounding boxes drawn directly on the frame using `cv2.rectangle`.
- Each box has a filled label pill above it: dark background, white text, `"label confidence%"`.
- Box and pill color is deterministic per class index (hue derived from class id) so the same object type always gets the same color.
- Rendered with `cv2.rectangle` + `cv2.putText`.

Both windows: press `q` to quit.

---

## Models & Data

### New assets fetched by `task download-models`
| File | Source |
|---|---|
| `models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite` | google-coral/test_data on GitHub |
| `models/coco_labels.txt` | google-coral/test_data on GitHub |

Downloads are idempotent (skip if file exists), consistent with existing download-models behavior.

### detect.py data flow
1. Check Coral USB detected (`find_coral_usb`)
2. Check model + labels files exist
3. Deferred pycoral imports + `_EDGETPU_SHARED_LIB` patch
4. `make_interpreter(model)` → `allocate_tensors()`
5. Open camera (`cv2.VideoCapture(0)`)
6. Loop: `cap.read()` → `preprocess_frame(frame, (300, 300))` → `common.set_input` → `interpreter.invoke()` → `detect.get_objects(interpreter, threshold)` → render or print

---

## Testing

- Unit tests for `draw_classification_overlay`, `draw_detections`, and the detect helpers (`load_labels`, `preprocess_frame`, `get_top_k`) run without hardware via `task test`.
- No hardware-dependent tests; inference loop tested manually with Coral USB attached.
