"""Apply compatibility patches to tflite-model-maker for arm64 macOS 15.

Run automatically by `task training:setup`. Safe to re-run.

Patches applied
---------------
1. tflite_model_maker/__init__.py  — wrap non-object-detector sub-modules in
   try/except so their broken deps don't prevent importing object_detector.
2. tensorflow_examples/.../model_spec/__init__.py — same for model specs.
3. tensorflow_examples/.../model_util.py — make tensorflowjs / tflite_support
   imports lazy (both unavailable / broken on arm64 macOS 15).
4. tflite_support stub package — tflite-support has no arm64 macOS wheel;
   we provide a minimal stub that satisfies the import without native libs.
"""

import sys
import textwrap
from pathlib import Path


def apply_patches(venv_root: Path) -> None:
    site = venv_root / "lib" / "python3.9" / "site-packages"

    _patch_mm_init(site)
    _patch_model_spec_init(site)
    _patch_model_util(site)
    _create_tflite_support_stub(site)
    print("All patches applied.")


# ---------------------------------------------------------------------------
# Patch 1: tflite_model_maker/__init__.py
# ---------------------------------------------------------------------------

_MM_INIT = """\
try:
    from tflite_model_maker import audio_classifier
except Exception:
    pass
from tflite_model_maker import config
try:
    from tflite_model_maker import image_classifier
except Exception:
    pass
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
try:
    from tflite_model_maker import question_answer
except Exception:
    pass
try:
    from tflite_model_maker import recommendation
except Exception:
    pass
try:
    from tflite_model_maker import searcher
except Exception:
    pass
try:
    from tflite_model_maker import text_classifier
except Exception:
    pass
"""

_MM_INIT_DEPRECATED = """\
try:
    from tensorflow_examples.lite.model_maker.core.data_util.image_dataloader import ImageClassifierDataLoader
except Exception:
    pass
from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.task import configs
"""


def _patch_mm_init(site: Path) -> None:
    p = site / "tflite_model_maker" / "__init__.py"
    text = p.read_text()
    if "try:" in text:
        return  # already patched

    # Replace the block of sub-module imports
    old_imports = (
        "from tflite_model_maker import audio_classifier\n"
        "from tflite_model_maker import config\n"
        "from tflite_model_maker import image_classifier\n"
        "from tflite_model_maker import model_spec\n"
        "from tflite_model_maker import object_detector\n"
        "from tflite_model_maker import question_answer\n"
        "from tflite_model_maker import recommendation\n"
        "from tflite_model_maker import searcher\n"
        "from tflite_model_maker import text_classifier\n"
    )
    text = text.replace(old_imports, _MM_INIT)

    old_deprecated = (
        "from tensorflow_examples.lite.model_maker.core.data_util.image_dataloader import ImageClassifierDataLoader\n"
        "from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat\n"
        "from tensorflow_examples.lite.model_maker.core.task import configs\n"
    )
    text = text.replace(old_deprecated, _MM_INIT_DEPRECATED)
    p.write_text(text)
    print(f"  patched {p.relative_to(site)}")


# ---------------------------------------------------------------------------
# Patch 2: .../model_spec/__init__.py
# ---------------------------------------------------------------------------

_MODEL_SPEC_IMPORTS = """\
from tensorflow_examples.lite.model_maker.core.task.model_spec import object_detector_spec
try:
    from tensorflow_examples.lite.model_maker.core.task.model_spec import audio_spec
except Exception:
    audio_spec = None
try:
    from tensorflow_examples.lite.model_maker.core.task.model_spec import image_spec
except Exception:
    image_spec = None
try:
    from tensorflow_examples.lite.model_maker.core.task.model_spec import recommendation_spec
except Exception:
    recommendation_spec = None
try:
    from tensorflow_examples.lite.model_maker.core.task.model_spec import text_spec
except Exception:
    text_spec = None
"""

_MODEL_SPECS_DICT = """\
MODEL_SPECS = {}
if image_spec is not None:
    MODEL_SPECS.update({
        'efficientnet_lite0': image_spec.efficientnet_lite0_spec,
        'efficientnet_lite1': image_spec.efficientnet_lite1_spec,
        'efficientnet_lite2': image_spec.efficientnet_lite2_spec,
        'efficientnet_lite3': image_spec.efficientnet_lite3_spec,
        'efficientnet_lite4': image_spec.efficientnet_lite4_spec,
        'mobilenet_v2': image_spec.mobilenet_v2_spec,
        'resnet_50': image_spec.resnet_50_spec,
    })
if text_spec is not None:
    MODEL_SPECS.update({
        'average_word_vec': text_spec.AverageWordVecModelSpec,
        'bert': text_spec.BertModelSpec,
        'bert_classifier': text_spec.BertClassifierModelSpec,
        'mobilebert_classifier': text_spec.mobilebert_classifier_spec,
        'bert_qa': text_spec.BertQAModelSpec,
        'mobilebert_qa': text_spec.mobilebert_qa_spec,
        'mobilebert_qa_squad': text_spec.mobilebert_qa_squad_spec,
    })
if audio_spec is not None:
    MODEL_SPECS.update({
        'audio_browser_fft': audio_spec.BrowserFFTSpec,
        'audio_teachable_machine': audio_spec.BrowserFFTSpec,
        'audio_yamnet': audio_spec.YAMNetSpec,
    })
if recommendation_spec is not None:
    MODEL_SPECS.update({
        'recommendation': recommendation_spec.RecommendationSpec,
    })
MODEL_SPECS.update({
    'efficientdet_lite0': object_detector_spec.efficientdet_lite0_spec,
    'efficientdet_lite1': object_detector_spec.efficientdet_lite1_spec,
    'efficientdet_lite2': object_detector_spec.efficientdet_lite2_spec,
    'efficientdet_lite3': object_detector_spec.efficientdet_lite3_spec,
    'efficientdet_lite4': object_detector_spec.efficientdet_lite4_spec,
})
"""


def _patch_model_spec_init(site: Path) -> None:
    p = (
        site
        / "tensorflow_examples/lite/model_maker/core/task/model_spec/__init__.py"
    )
    text = p.read_text()
    if "audio_spec = None" in text:
        return  # already patched

    old = (
        "from tensorflow_examples.lite.model_maker.core.task.model_spec import audio_spec\n"
        "from tensorflow_examples.lite.model_maker.core.task.model_spec import image_spec\n"
        "from tensorflow_examples.lite.model_maker.core.task.model_spec import object_detector_spec\n"
        "from tensorflow_examples.lite.model_maker.core.task.model_spec import recommendation_spec\n"
        "from tensorflow_examples.lite.model_maker.core.task.model_spec import text_spec\n"
    )
    text = text.replace(old, _MODEL_SPEC_IMPORTS)

    old_dict = (
        "# A dict for model specs to make it accessible by string key.\n"
        "MODEL_SPECS = {\n"
        "    # Image classification\n"
        "    'efficientnet_lite0': image_spec.efficientnet_lite0_spec,\n"
    )
    # Find and replace the entire MODEL_SPECS dict
    start = text.find("# A dict for model specs")
    end = text.find("\n}\n", start) + 3
    if start != -1 and end > start:
        text = text[:start] + _MODEL_SPECS_DICT + text[end:]

    p.write_text(text)
    print(f"  patched {p.relative_to(site)}")


# ---------------------------------------------------------------------------
# Patch 3: model_util.py  — lazy tensorflowjs / tflite_support
# ---------------------------------------------------------------------------

_MODEL_UTIL_LAZY = """\
from tensorflow_examples.lite.model_maker.core import compat

# Lazy imports — tflite_support unavailable on arm64 macOS; tensorflowjs only needed for TFJS export
_metadata = None
_tfjs_converter = None

def _get_metadata():
    global _metadata
    if _metadata is None:
        from tflite_support import metadata as _m
        _metadata = _m
    return _metadata

def _get_tfjs_converter():
    global _tfjs_converter
    if _tfjs_converter is None:
        from tensorflowjs.converters import converter as _c
        _tfjs_converter = _c
    return _tfjs_converter
"""


def _patch_model_util(site: Path) -> None:
    p = site / "tensorflow_examples/lite/model_maker/core/task/model_util.py"
    text = p.read_text()
    if "_get_metadata" in text:
        return  # already patched

    old = (
        "from tensorflow_examples.lite.model_maker.core import compat\n"
        "from tensorflowjs.converters import converter as tfjs_converter\n"
        "from tflite_support import metadata as _metadata\n"
    )
    text = text.replace(old, _MODEL_UTIL_LAZY)
    text = text.replace(
        "    tfjs_converter.dispatch_keras_saved_model_to_tensorflowjs_conversion(\n",
        "    _get_tfjs_converter().dispatch_keras_saved_model_to_tensorflowjs_conversion(\n",
    )
    text = text.replace(
        "  return tfjs_converter.keras_tfjs_loader.load_keras_model(\n",
        "  return _get_tfjs_converter().keras_tfjs_loader.load_keras_model(\n",
    )
    text = text.replace(
        "  displayer = _metadata.MetadataDisplayer.with_model_file(tflite_filepath)\n",
        "  displayer = _get_metadata().MetadataDisplayer.with_model_file(tflite_filepath)\n",
    )
    p.write_text(text)
    print(f"  patched {p.relative_to(site)}")


# ---------------------------------------------------------------------------
# Patch 4: tflite_support stub
# ---------------------------------------------------------------------------

def _create_tflite_support_stub(site: Path) -> None:
    stub = site / "tflite_support"
    if stub.exists() and (stub / "__init__.py").exists():
        return  # already created

    stub.mkdir(exist_ok=True)
    (stub / "__init__.py").write_text('"""Stub tflite_support — no arm64 macOS wheel."""\n')

    mw = stub / "metadata_writers"
    mw.mkdir(exist_ok=True)
    (mw / "__init__.py").write_text('"""Stub"""\n')

    (stub / "metadata.py").write_text(textwrap.dedent("""\
        class _Displayer:
            def get_metadata_json(self): return "{}"
        class MetadataDisplayer:
            @staticmethod
            def with_model_file(path): return _Displayer()
    """))

    (mw / "writer_utils.py").write_text(textwrap.dedent("""\
        def load_file(path):
            with open(path, 'rb') as f: return f.read()
        def save_file(data, path):
            with open(path, 'wb') as f: f.write(data)
    """))

    (mw / "object_detector.py").write_text(textwrap.dedent("""\
        class _Writer:
            def populate(self): return self._data
            def get_populated_metadata_json(self): return "{}"
        class MetadataWriter:
            @staticmethod
            def create_for_inference(data, mean_rgb, stddev_rgb, label_file_paths):
                w = _Writer(); w._data = data; return w
    """))

    for mod in ("schema_py_generated", "metadata_schema_py_generated"):
        (stub / f"{mod}.py").write_text('"""Stub"""\n')

    print(f"  created tflite_support stub at {stub.relative_to(site)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <venv_root>")
        sys.exit(1)
    apply_patches(Path(sys.argv[1]))
