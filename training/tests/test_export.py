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
