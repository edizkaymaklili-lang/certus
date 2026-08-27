"""Edge packaging pipeline: PyTorch -> ONNX -> INT8 ONNX / INT8 TFLite.

Heavy ML dependencies (``torch``, ``onnx``, ``onnxruntime``, ``tensorflow``)
are imported lazily inside each function so that ``pip install certus-ai``
(the guardrail SDK) never pulls them in. Install the ``edge`` extra to use
this module::

    pip install "certus-ai[edge]"

Typical usage (see ``examples/colab_training_template.py`` for the full
Colab-side script)::

    from certus.edge.quantize import EdgePackager

    packager = EdgePackager(output_dir="artifacts/run-001")
    onnx_path = packager.to_onnx(model, sample_input)
    int8_onnx_path = packager.quantize_onnx_int8(onnx_path)
    manifest = packager.write_manifest(run_id="run-001")
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _require(module_name: str, extra_hint: str = "edge") -> Any:
    try:
        return __import__(module_name)
    except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
        raise ImportError(
            f"'{module_name}' is required for edge packaging but is not installed. "
            f'Install it with: pip install "certus-ai[{extra_hint}]"'
        ) from exc


class EdgePackager:
    """Converts and quantizes a trained model for constrained edge hardware.

    Args:
        output_dir: Directory where every converted/quantized artifact and
            the final manifest are written. Created if it doesn't exist.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # PyTorch -> ONNX
    # ------------------------------------------------------------------ #

    def to_onnx(
        self,
        model: Any,
        sample_input: Any,
        *,
        filename: str = "model.onnx",
        input_names: Iterable[str] = ("input",),
        output_names: Iterable[str] = ("output",),
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        opset: int = 17,
    ) -> Path:
        """Export a PyTorch ``model`` to ONNX.

        Args:
            model: A ``torch.nn.Module`` in eval mode.
            sample_input: A representative input tensor (or tuple of
                tensors) matching the model's ``forward`` signature, used
                to trace the graph.
            filename: Output filename within ``output_dir``.
            dynamic_axes: Optional ONNX dynamic-axes spec, e.g.
                ``{"input": {0: "batch"}, "output": {0: "batch"}}`` for
                variable batch size.
            opset: ONNX opset version to target.

        Returns:
            Path to the written ``.onnx`` file.
        """
        torch = _require("torch")
        model.eval()
        output_path = self.output_dir / filename
        with torch.no_grad():
            torch.onnx.export(
                model,
                sample_input,
                str(output_path),
                input_names=list(input_names),
                output_names=list(output_names),
                dynamic_axes=dynamic_axes,
                opset_version=opset,
            )
        self._record_artifact("onnx", output_path)
        return output_path

    # ------------------------------------------------------------------ #
    # ONNX -> INT8 ONNX
    # ------------------------------------------------------------------ #

    def quantize_onnx_int8(
        self,
        onnx_path: str | Path,
        *,
        filename: str = "model.int8.onnx",
        per_channel: bool = False,
    ) -> Path:
        """Apply dynamic INT8 quantization to an ONNX model via ONNX Runtime.

        Dynamic quantization is used because it requires no calibration
        dataset, making it suitable as a fast default in an automated
        pipeline; swap in ``quantize_static`` with a calibration reader for
        higher accuracy if a representative dataset is available.

        Args:
            onnx_path: Path to the FP32 ONNX model produced by :meth:`to_onnx`.
            filename: Output filename within ``output_dir``.
            per_channel: Whether to quantize weights per output channel
                (usually more accurate, slightly larger model).

        Returns:
            Path to the written INT8 ``.onnx`` file.
        """
        quantization = _require("onnxruntime.quantization")
        output_path = self.output_dir / filename
        quantization.quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(output_path),
            weight_type=quantization.QuantType.QInt8,
            per_channel=per_channel,
        )
        self._record_artifact("int8_onnx", output_path)
        return output_path

    # ------------------------------------------------------------------ #
    # TensorFlow SavedModel -> INT8 TFLite
    # ------------------------------------------------------------------ #

    def to_tflite_int8(
        self,
        saved_model_dir: str | Path,
        *,
        filename: str = "model.int8.tflite",
        representative_dataset: Any | None = None,
    ) -> Path:
        """Convert a TensorFlow SavedModel to a fully INT8-quantized TFLite model.

        Args:
            saved_model_dir: Directory containing a TensorFlow SavedModel
                (``tf.saved_model.save`` output).
            filename: Output filename within ``output_dir``.
            representative_dataset: A callable yielding representative
                input batches (the standard TFLite calibration generator).
                If omitted, dynamic-range quantization is used instead of
                full integer quantization.

        Returns:
            Path to the written ``.tflite`` file.
        """
        tf = _require("tensorflow")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_dataset is not None:
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        output_path = self.output_dir / filename
        output_path.write_bytes(tflite_model)
        self._record_artifact("int8_tflite", output_path)
        return output_path

    # ------------------------------------------------------------------ #
    # Manifest
    # ------------------------------------------------------------------ #

    def _record_artifact(self, kind: str, path: Path) -> None:
        self._artifacts[kind] = {
            "path": str(path),
            "size_bytes": os.path.getsize(path),
            "created_at": time.time(),
        }

    def write_manifest(self, run_id: str, *, filename: str = "manifest.json") -> dict[str, Any]:
        """Write a ``manifest.json`` summarizing every artifact produced so far.

        Args:
            run_id: Identifier of the training run these artifacts belong to.
            filename: Output filename within ``output_dir``.

        Returns:
            The manifest dict that was written to disk.
        """
        manifest = {
            "run_id": run_id,
            "generated_at": time.time(),
            "artifacts": self._artifacts,
        }
        (self.output_dir / filename).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
