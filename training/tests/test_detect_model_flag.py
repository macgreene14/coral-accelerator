"""Tests for detect.py --model flag. Run from repo root with .venv/bin/python."""
import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestModelFlag:
    def _make_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", choices=["coco", "pv"], default="coco")
        parser.add_argument("--threshold", type=float, default=None)
        parser.add_argument("--display", action="store_true")
        return parser

    def test_model_defaults_to_coco(self):
        args = self._make_parser().parse_args([])
        assert args.model == "coco"

    def test_pv_model_accepted(self):
        args = self._make_parser().parse_args(["--model", "pv"])
        assert args.model == "pv"

    def test_threshold_defaults_to_none(self):
        args = self._make_parser().parse_args([])
        assert args.threshold is None

    def test_threshold_override_accepted(self):
        args = self._make_parser().parse_args(["--model", "pv", "--threshold", "0.5"])
        assert args.threshold == 0.5

    def test_model_configs_present(self):
        from detect import _MODEL_CONFIGS
        assert "coco" in _MODEL_CONFIGS
        assert "pv" in _MODEL_CONFIGS
        for cfg in _MODEL_CONFIGS.values():
            assert "model" in cfg
            assert "labels" in cfg
            assert "default_threshold" in cfg

    def test_pv_default_threshold_lower_than_coco(self):
        from detect import _MODEL_CONFIGS
        assert _MODEL_CONFIGS["pv"]["default_threshold"] < _MODEL_CONFIGS["coco"]["default_threshold"]
