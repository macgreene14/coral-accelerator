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


def coco_to_pascal_voc(coco_json_path: Path, images_dir: Path, output_dir: Path, max_images: int = 0) -> None:
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

    images = coco["images"]
    if max_images:
        images = images[:max_images]
    for img_info in images:
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
            ET.SubElement(obj, "difficult").text = "0"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "pose").text = "Unspecified"
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
    # For dry-run use a 20-image subset to minimise conversion time
    if args.dry_run:
        from_coco_kwargs = {"max_images": 20}
        from_val_kwargs = {"max_images": 10}
    else:
        from_coco_kwargs = {}
        from_val_kwargs = {}

    for split, kwargs in (("train", from_coco_kwargs), ("val", from_val_kwargs)):
        voc_split = voc_dir / split
        if not voc_split.exists():
            print(f"Converting {split}.json → Pascal VOC at {voc_split} ...")
            coco_to_pascal_voc(args.data / f"{split}.json", args.data / "images", voc_split, **kwargs)

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
