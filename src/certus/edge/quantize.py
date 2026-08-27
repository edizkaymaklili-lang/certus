"""Edge packaging pipeline: PyTorch -> ONNX -> INT8 ONNX / INT8 TFLite.

Heavy ML dependencies (``torch``, ``onnx``, ``onnxruntime``, ``tensorflow``,
``onnx2tf``) are imported lazily inside each function so that
``pip install certus-ai`` (the guardrail SDK) never pulls them in. Install
the ``edge`` extra to use this module::

    pip install "certus-ai[edge]"

Typical usage (see ``examples/colab_training_template.py`` for the full
Colab-side script)::

    from certus.edge.quantize import EdgePackager

    packager = EdgePackager(output_dir="artifacts/run-001")
    onnx_path = packager.to_onnx(model, sample_input)
    int8_onnx_path = packager.quantize_onnx_int8(onnx_path)
    int8_tflite_path = packager.to_tflite_int8_from_onnx(onnx_path, calibration_batch)
    manifest = packager.write_manifest(run_id="run-001")
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import shutil
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _require(module_name: str, extra_hint: str = "edge") -> Any:
    """Import and return ``module_name``, including dotted submodules.

    Uses :func:`importlib.import_module` rather than the ``__import__``
    builtin directly: ``__import__("onnxruntime.quantization")`` returns
    the top-level ``onnxruntime`` package, not the ``quantization``
    submodule — a classic footgun that silently breaks any dotted name.
    """
    try:
        return importlib.import_module(module_name)
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
        opset: int = 18,
        dynamo: bool = False,
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
                variable batch size. Edge/embedded targets almost always run
                one inference at a time, so leaving this ``None`` (a fixed
                shape) is usually both simpler and closer to the deployment
                target.
            opset: ONNX opset version to target.
            dynamo: PyTorch's newer ``torch.export``-based ONNX exporter
                (the default in ``torch.onnx.export`` since 2.9) has been
                observed to emit graphs — using ``ReduceMean``/``Reshape``
                in place of ``GlobalAveragePool``/``Flatten`` for pooling
                layers — that fail ONNX shape inference inside
                :meth:`quantize_onnx_int8` and :meth:`to_tflite_int8_from_onnx`
                with ``InferenceError: Inferred shape and existing shape
                differ``. Defaulting to the legacy TorchScript-based exporter
                (``dynamo=False``) avoids that failure mode; set this to
                True to opt into the newer exporter once your model/PyTorch
                version combination is verified to convert cleanly end to end.

        Returns:
            Path to the written ``.onnx`` file.
        """
        torch = _require("torch")
        model.eval()
        output_path = self.output_dir / filename
        export_kwargs: dict[str, Any] = {
            "input_names": list(input_names),
            "output_names": list(output_names),
            "dynamic_axes": dynamic_axes,
            "opset_version": opset,
        }
        # `dynamo` was added to torch.onnx.export in a later 2.x release; only
        # pass it through on torch versions that actually accept it, so this
        # still works against the `torch>=2.2` floor in the `edge` extra.
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_kwargs["dynamo"] = dynamo
        with torch.no_grad():
            torch.onnx.export(model, sample_input, str(output_path), **export_kwargs)
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
    # ONNX -> INT8 TFLite (direct, via onnx2tf)
    # ------------------------------------------------------------------ #

    def to_tflite_int8_from_onnx(
        self,
        onnx_path: str | Path,
        representative_dataset: Any,
        *,
        input_op_name: str = "input",
        filename: str = "model.int8.tflite",
        keep_intermediate_artifacts: bool = False,
    ) -> Path:
        """Convert an ONNX model directly to a fully INT8-quantized TFLite model.

        Unlike :meth:`to_tflite_int8`, which requires a pre-built TensorFlow
        SavedModel, this converts straight from the ONNX file produced by
        :meth:`to_onnx` via `onnx2tf <https://github.com/PINTO0309/onnx2tf>`_
        — no manual ONNX-to-TensorFlow graph translation step required.

        Args:
            onnx_path: Path to the ONNX model to convert — the FP32 model
                from :meth:`to_onnx`, not the ONNX Runtime-quantized one
                from :meth:`quantize_onnx_int8` (onnx2tf calibrates and
                quantizes for TFLite itself).
            representative_dataset: A small batch of real (or at least
                realistic) inputs, shape ``(N, *input_shape)``, used to
                calibrate the INT8 quantization ranges. Random noise will
                "work" but produces poorly calibrated, inaccurate ranges —
                pass actual validation-set samples whenever possible.
            input_op_name: Name of the model's input tensor; must match the
                ``input_names`` passed to :meth:`to_onnx`.
            filename: Output filename within ``output_dir`` for the
                resulting fully-integer-quantized ``.tflite`` file (onnx2tf's
                ``model_full_integer_quant.tflite`` output — int8 input
                tensor, matching the format most microcontroller/NPU TFLite
                runtimes expect).
            keep_intermediate_artifacts: If True, keep onnx2tf's full working
                directory (the intermediate SavedModel plus every TFLite
                variant it produces: float32, float16, dynamic-range, and
                both integer-quantized flavors) under
                ``output_dir/onnx2tf_out`` instead of deleting it after
                extracting the one artifact described above.

        Returns:
            Path to the fully INT8-quantized ``.tflite`` file.

        Raises:
            FileNotFoundError: If onnx2tf did not produce the expected
                fully-integer-quantized output (e.g. an unsupported op in
                the ONNX graph forced a partial fallback).
        """
        onnx2tf = _require("onnx2tf")
        np = _require("numpy")

        work_dir = self.output_dir / "onnx2tf_out"
        work_dir.mkdir(parents=True, exist_ok=True)
        calibration_path = work_dir / "calibration_data.npy"
        np.save(calibration_path, np.asarray(representative_dataset, dtype="float32"))

        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(work_dir),
            output_integer_quantized_tflite=True,
            # [input_op_name, npy_file_path, mean, std] — mean=0/std=1 leaves
            # the calibration data unnormalized, matching plain feature/pixel
            # inputs rather than the ImageNet-style normalization these
            # defaults are usually meant for.
            custom_input_op_name_np_data_path=[[input_op_name, str(calibration_path), 0.0, 1.0]],
            non_verbose=True,
        )

        produced = work_dir / "model_full_integer_quant.tflite"
        if not produced.exists():
            raise FileNotFoundError(
                f"onnx2tf did not produce {produced.name} in {work_dir}; check the "
                "conversion output above for warnings (e.g. an op in the ONNX graph "
                "that onnx2tf could not fully integer-quantize)."
            )
        output_path = self.output_dir / filename
        output_path.write_bytes(produced.read_bytes())
        self._record_artifact("int8_tflite", output_path)

        if not keep_intermediate_artifacts:
            shutil.rmtree(work_dir, ignore_errors=True)

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
