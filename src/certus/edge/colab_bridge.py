"""Claude-Colab bridge: the file contract between a training run and its orchestrator.

The intended loop is:

1. An orchestrating agent (typically Claude, driving a Colab notebook) writes
   a :class:`TrainingConfig` to ``config.json``.
2. The Colab training script (see ``examples/colab_training_template.py``)
   reads that config, trains, and writes a :class:`TrainingMetrics` history
   to ``metrics.json``, alongside packaged edge artifacts (see
   :mod:`certus.edge.quantize`).
3. The orchestrator reads ``metrics.json`` back, decides whether to accept
   the run or launch another one, and — if iterating — writes a new
   ``config.json``.

This module owns steps 1 and 3's *data contract* plus a deterministic,
rule-based helper (:func:`suggest_next_config`) that proposes the next
hyperparameters from a completed run's metrics. It is a starting heuristic
an orchestrator can call directly, inspect, or override — never a hidden
decision the pipeline makes on its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

EdgeFormat = Literal["onnx", "tflite", "int8_onnx", "int8_tflite"]
_DEFAULT_TARGET_FORMATS: list[EdgeFormat] = ["int8_tflite"]


class TrainingConfig(BaseModel):
    """Hyperparameters and run metadata consumed by the Colab training script."""

    run_id: str = Field(..., description="Unique identifier for this training run.")
    model_name: str = Field(default="edge-model", description="Base model/architecture name.")
    learning_rate: float = Field(default=1e-3, gt=0)
    batch_size: int = Field(default=32, gt=0)
    epochs: int = Field(default=10, gt=0)
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    weight_decay: float = Field(default=0.0, ge=0)
    dropout: float = Field(default=0.0, ge=0, lt=1)
    target_formats: list[EdgeFormat] = Field(default_factory=lambda: list(_DEFAULT_TARGET_FORMATS))
    notes: str | None = Field(default=None, description="Free-text notes from the orchestrator.")


class EpochMetric(BaseModel):
    """Metrics recorded for a single training epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float | None = None
    val_accuracy: float | None = None


class TrainingMetrics(BaseModel):
    """Full result of a completed (or in-progress) training run."""

    run_id: str
    history: list[EpochMetric] = Field(default_factory=list)
    final_model_size_mb: float | None = None
    inference_latency_ms: float | None = None
    exported_formats: list[str] = Field(default_factory=list)
    status: Literal["completed", "failed", "in_progress"] = "completed"
    error: str | None = None


def load_config(path: str | Path) -> TrainingConfig:
    """Read a :class:`TrainingConfig` from ``config.json``."""
    return TrainingConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_config(config: TrainingConfig, path: str | Path) -> None:
    """Write a :class:`TrainingConfig` to ``config.json`` (pretty-printed)."""
    Path(path).write_text(config.model_dump_json(indent=2), encoding="utf-8")


def load_metrics(path: str | Path) -> TrainingMetrics:
    """Read a :class:`TrainingMetrics` from ``metrics.json``."""
    return TrainingMetrics.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_metrics(metrics: TrainingMetrics, path: str | Path) -> None:
    """Write a :class:`TrainingMetrics` to ``metrics.json`` (pretty-printed)."""
    Path(path).write_text(metrics.model_dump_json(indent=2), encoding="utf-8")


def suggest_next_config(
    config: TrainingConfig,
    metrics: TrainingMetrics,
    *,
    next_run_id: str,
    overfit_ratio_threshold: float = 1.2,
    plateau_window: int = 3,
    plateau_epsilon: float = 1e-3,
    lr_decay_factor: float = 0.5,
) -> TrainingConfig:
    """Propose the next run's hyperparameters using deterministic rules.

    This deliberately mirrors the "zero black-box" principle of the rest of
    Certus: the adjustment logic is a fixed decision tree over the metrics
    history, not a learned or probabilistic suggestion. An orchestrating
    agent (e.g. Claude) is expected to call this, inspect the result, and
    is always free to override any field before writing the next
    ``config.json``.

    Rules applied, in order:

    1. **Overfitting** — if the last epoch's ``val_loss`` exceeds
       ``train_loss * overfit_ratio_threshold``, increase regularization
       (bump ``dropout`` by 0.1 capped at 0.5, ``weight_decay`` ×2).
    2. **Plateau** — if ``val_loss`` over the last ``plateau_window`` epochs
       improved by less than ``plateau_epsilon``, decay ``learning_rate`` by
       ``lr_decay_factor``.
    3. **Failed run** — if ``metrics.status == "failed"``, halve the
       ``learning_rate`` and ``batch_size`` as a conservative retry.
    4. Otherwise, keep hyperparameters unchanged (the run is healthy —
       re-running with the same config, or accepting it, is left to the
       orchestrator).

    Args:
        config: The :class:`TrainingConfig` that produced ``metrics``.
        metrics: The completed run's :class:`TrainingMetrics`.
        next_run_id: Identifier to assign to the proposed next run.

    Returns:
        A new :class:`TrainingConfig` (the input is left untouched).
    """
    next_config = config.model_copy(update={"run_id": next_run_id})

    if metrics.status == "failed":
        next_config = next_config.model_copy(
            update={
                "learning_rate": config.learning_rate * 0.5,
                "batch_size": max(1, config.batch_size // 2),
                "notes": f"Auto-retry after failed run '{config.run_id}': {metrics.error}",
            }
        )
        return next_config

    if not metrics.history:
        return next_config

    last = metrics.history[-1]
    notes: list[str] = []

    if last.train_loss > 0 and last.val_loss > last.train_loss * overfit_ratio_threshold:
        next_config = next_config.model_copy(
            update={
                "dropout": min(0.5, config.dropout + 0.1),
                "weight_decay": (config.weight_decay or 1e-4) * 2,
            }
        )
        notes.append(
            f"Increased regularization: overfit detected (val_loss={last.val_loss:.4f} > "
            f"{overfit_ratio_threshold}x train_loss={last.train_loss:.4f})."
        )

    window = metrics.history[-plateau_window:]
    if len(window) == plateau_window:
        improvement = window[0].val_loss - window[-1].val_loss
        if improvement < plateau_epsilon:
            next_config = next_config.model_copy(
                update={"learning_rate": config.learning_rate * lr_decay_factor}
            )
            notes.append(
                f"Decayed learning_rate by {lr_decay_factor}x: val_loss plateaued "
                f"over last {plateau_window} epochs (improvement={improvement:.5f})."
            )

    if notes:
        next_config = next_config.model_copy(update={"notes": " ".join(notes)})

    return next_config
