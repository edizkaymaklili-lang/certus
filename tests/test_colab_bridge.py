"""Tests for certus.edge.colab_bridge (config/metrics contract + heuristic optimizer)."""

from __future__ import annotations

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


def test_config_round_trips_through_json(tmp_path):
    config = TrainingConfig(run_id="run-001", learning_rate=0.01, epochs=3)
    path = tmp_path / "config.json"

    save_config(config, path)
    loaded = load_config(path)

    assert loaded == config


def test_metrics_round_trips_through_json(tmp_path):
    metrics = TrainingMetrics(
        run_id="run-001",
        history=[EpochMetric(epoch=1, train_loss=0.5, val_loss=0.6)],
        status="completed",
    )
    path = tmp_path / "metrics.json"

    save_metrics(metrics, path)
    loaded = load_metrics(path)

    assert loaded == metrics


def test_suggest_next_config_increases_regularization_on_overfit():
    config = TrainingConfig(run_id="run-001", dropout=0.1, weight_decay=1e-4)
    metrics = TrainingMetrics(
        run_id="run-001",
        history=[EpochMetric(epoch=1, train_loss=0.2, val_loss=0.5)],  # val >> train
    )

    next_config = suggest_next_config(config, metrics, next_run_id="run-002")

    assert next_config.run_id == "run-002"
    assert next_config.dropout > config.dropout
    assert next_config.weight_decay > config.weight_decay


def test_suggest_next_config_decays_lr_on_plateau():
    config = TrainingConfig(run_id="run-001", learning_rate=0.01)
    history = [
        EpochMetric(epoch=1, train_loss=0.5, val_loss=0.50),
        EpochMetric(epoch=2, train_loss=0.5, val_loss=0.4995),
        EpochMetric(epoch=3, train_loss=0.5, val_loss=0.4991),
    ]
    metrics = TrainingMetrics(run_id="run-001", history=history)

    next_config = suggest_next_config(config, metrics, next_run_id="run-002", plateau_window=3)

    assert next_config.learning_rate < config.learning_rate


def test_suggest_next_config_retries_conservatively_on_failure():
    config = TrainingConfig(run_id="run-001", learning_rate=0.01, batch_size=64)
    metrics = TrainingMetrics(run_id="run-001", status="failed", error="CUDA OOM")

    next_config = suggest_next_config(config, metrics, next_run_id="run-002")

    assert next_config.learning_rate == config.learning_rate * 0.5
    assert next_config.batch_size == config.batch_size // 2
    assert "run-001" in (next_config.notes or "")


def test_suggest_next_config_keeps_healthy_run_unchanged():
    config = TrainingConfig(run_id="run-001", learning_rate=0.01, dropout=0.1)
    history = [
        EpochMetric(epoch=1, train_loss=0.5, val_loss=0.52),
        EpochMetric(epoch=2, train_loss=0.4, val_loss=0.41),
        EpochMetric(epoch=3, train_loss=0.3, val_loss=0.31),
    ]
    metrics = TrainingMetrics(run_id="run-001", history=history)

    next_config = suggest_next_config(config, metrics, next_run_id="run-002")

    assert next_config.learning_rate == config.learning_rate
    assert next_config.dropout == config.dropout
