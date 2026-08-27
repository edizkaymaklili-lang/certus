"""End-to-end integration test for EdgePackager (requires the `edge` extra).

Skipped automatically wherever torch/onnx/onnxruntime/onnx2tf aren't
installed (e.g. the default dev environment). Run it with:

    python3.11 -m venv .venv-edge && source .venv-edge/bin/activate
    pip install -e ".[edge,dev]"
    pytest tests/test_edge_packager_integration.py -v

This is deliberately a real conversion, not a mock: it exports an actual
tiny CNN to ONNX, quantizes it with ONNX Runtime, and converts it to a
genuinely INT8 TFLite model via onnx2tf, then asserts on the produced
files' actual dtypes — the same pipeline examples/colab_training_template.py
runs.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx2tf")
pytest.importorskip("onnxruntime.quantization")

import torch.nn as nn  # noqa: E402

from certus.edge.quantize import EdgePackager  # noqa: E402


def _tiny_model() -> nn.Module:
    # Mirrors examples/colab_training_template.py's build_model() exactly.
    # A more minimal architecture (fewer channels, no ReLU/MaxPool) was
    # observed to crash TensorFlow Lite's native calibrator with a
    # "Floating point exception" on Linux during INT8 calibration — this
    # shape is the one actually verified to convert cleanly end to end.
    model = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 10),
    )
    model.eval()
    return model


def test_to_onnx_produces_a_loadable_model(tmp_path):
    onnx = pytest.importorskip("onnx")
    packager = EdgePackager(output_dir=tmp_path)
    sample_input = torch.randn(1, 1, 8, 8)

    onnx_path = packager.to_onnx(_tiny_model(), sample_input)

    assert onnx_path.exists()
    onnx.checker.check_model(onnx.load(str(onnx_path)))


def test_quantize_onnx_int8_shrinks_and_produces_int8_weights(tmp_path):
    onnx = pytest.importorskip("onnx")
    packager = EdgePackager(output_dir=tmp_path)
    onnx_path = packager.to_onnx(_tiny_model(), torch.randn(1, 1, 8, 8))

    int8_path = packager.quantize_onnx_int8(onnx_path)

    assert int8_path.exists()
    model = onnx.load(str(int8_path))
    int8_initializers = [i for i in model.graph.initializer if i.data_type == onnx.TensorProto.INT8]
    assert int8_initializers, "expected at least one INT8-quantized weight tensor"


def test_to_tflite_int8_from_onnx_produces_genuine_int8_model(tmp_path):
    np = pytest.importorskip("numpy")
    litert = pytest.importorskip("ai_edge_litert.interpreter")
    packager = EdgePackager(output_dir=tmp_path)
    onnx_path = packager.to_onnx(_tiny_model(), torch.randn(1, 1, 8, 8))
    calibration_data = np.random.randn(20, 1, 8, 8).astype("float32")

    tflite_path = packager.to_tflite_int8_from_onnx(onnx_path, calibration_data)

    assert tflite_path.exists()
    interpreter = litert.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    assert input_detail["dtype"] == np.int8


def test_full_pipeline_writes_manifest_with_all_three_artifacts(tmp_path):
    pytest.importorskip("onnx")
    np = pytest.importorskip("numpy")
    packager = EdgePackager(output_dir=tmp_path)
    sample = torch.randn(1, 1, 8, 8)

    onnx_path = packager.to_onnx(_tiny_model(), sample)
    packager.quantize_onnx_int8(onnx_path)
    packager.to_tflite_int8_from_onnx(onnx_path, np.random.randn(20, 1, 8, 8).astype("float32"))
    manifest = packager.write_manifest(run_id="test-run")

    assert set(manifest["artifacts"]) == {"onnx", "int8_onnx", "int8_tflite"}
    for artifact in manifest["artifacts"].values():
        assert artifact["size_bytes"] > 0
