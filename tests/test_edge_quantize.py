"""Tests for certus.edge.quantize's dependency-loading helper.

Only `_require` is tested without the `edge` extra installed: it must
resolve dotted module paths correctly (e.g. `onnxruntime.quantization`),
which plain `__import__` does not do — `__import__("a.b")` returns `a`,
not `a.b`, unless `fromlist` is used. The rest of `EdgePackager` needs
torch/onnx/onnxruntime/tensorflow/onnx2tf and is exercised manually with
the `edge` extra installed (see examples/colab_training_template.py).
"""

from __future__ import annotations

import pytest

from certus.edge.quantize import _require


def test_require_resolves_top_level_module():
    module = _require("json")

    assert module.__name__ == "json"
    assert hasattr(module, "dumps")


def test_require_resolves_dotted_submodule():
    # A stdlib stand-in for the onnxruntime.quantization case: plain
    # __import__("xml.etree.ElementTree") would return the `xml` package,
    # not the `ElementTree` submodule.
    module = _require("xml.etree.ElementTree")

    assert module.__name__ == "xml.etree.ElementTree"
    assert hasattr(module, "Element")


def test_require_raises_helpful_error_for_missing_module():
    with pytest.raises(ImportError, match="certus-ai\\[edge\\]"):
        _require("certus_definitely_not_a_real_module")
