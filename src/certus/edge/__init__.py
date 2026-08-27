"""Edge AI MLOps pipeline: the Claude-Colab bridge and model packaging.

``certus.edge.quantize`` requires the ``edge`` extra (``torch``, ``onnx``,
``onnxruntime``, ``tensorflow``); it is intentionally not imported here so
that the lightweight guardrail SDK stays free of heavy ML dependencies.
"""

from certus.edge.colab_bridge import (
    EpochMetric,
    TrainingConfig,
    TrainingMetrics,
    load_config,
    load_metrics,
    save_config,
    save_metrics,
    suggest_next_config,
)

__all__ = [
    "EpochMetric",
    "TrainingConfig",
    "TrainingMetrics",
    "load_config",
    "load_metrics",
    "save_config",
    "save_metrics",
    "suggest_next_config",
]
