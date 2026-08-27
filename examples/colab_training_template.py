"""Colab-side training script for the Claude-Colab edge bridge.

Intended to run as a Colab cell (or any machine with the `edge` extra
installed). It reads a `config.json` written by the orchestrating agent,
trains a small PyTorch model, writes a `metrics.json` history back, and
packages the trained model into INT8 ONNX / TFLite artifacts plus a
`manifest.json` describing them.

Install the extra this script needs:

    pip install "certus-ai[edge]"

Usage:

    python examples/colab_training_template.py \
        --config examples/config.json \
        --output-dir artifacts/run-001
"""

from __future__ import annotations

import argparse
from pathlib import Path

from certus.edge.colab_bridge import (
    EpochMetric,
    TrainingConfig,
    TrainingMetrics,
    load_config,
    save_metrics,
)
from certus.edge.quantize import EdgePackager


def build_model(config: TrainingConfig):
    """A minimal CNN sized for edge deployment (MNIST-shaped input)."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(config.dropout),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 10),
    )


def build_optimizer(config: TrainingConfig, model):
    import torch.optim as optim

    lr, wd = config.learning_rate, config.weight_decay
    if config.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if config.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def synthetic_dataloaders(config: TrainingConfig):
    """Synthetic MNIST-shaped data so this template runs with no external dataset.

    Swap this out for a real `torchvision.datasets.MNIST` (or your own data)
    in an actual Colab run — this exists purely so the template is
    self-contained and fast to smoke-test.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    generator = torch.Generator().manual_seed(0)
    x_train = torch.randn(512, 1, 28, 28, generator=generator)
    y_train = torch.randint(0, 10, (512,), generator=generator)
    x_val = torch.randn(128, 1, 28, 28, generator=generator)
    y_val = torch.randint(0, 10, (128,), generator=generator)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=config.batch_size)
    return train_loader, val_loader


def train(config: TrainingConfig) -> tuple:
    import torch
    import torch.nn as nn

    model = build_model(config)
    optimizer = build_optimizer(config, model)
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader = synthetic_dataloaders(config)

    history: list[EpochMetric] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += x.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
                val_total += x.size(0)

        metric = EpochMetric(
            epoch=epoch,
            train_loss=train_loss / train_total,
            val_loss=val_loss / val_total,
            train_accuracy=train_correct / train_total,
            val_accuracy=val_correct / val_total,
        )
        history.append(metric)
        print(
            f"epoch {epoch}/{config.epochs} "
            f"train_loss={metric.train_loss:.4f} val_loss={metric.val_loss:.4f} "
            f"val_acc={metric.val_accuracy:.4f}"
        )

    return model, history


def package_for_edge(config: TrainingConfig, model, output_dir: Path) -> dict:
    import torch

    packager = EdgePackager(output_dir=output_dir)
    sample_input = torch.randn(1, 1, 28, 28)

    exported_formats = []
    onnx_path = packager.to_onnx(model, sample_input, dynamic_axes={"input": {0: "batch"}})
    exported_formats.append("onnx")

    if "int8_onnx" in config.target_formats:
        packager.quantize_onnx_int8(onnx_path)
        exported_formats.append("int8_onnx")

    if "int8_tflite" in config.target_formats:
        try:
            # Requires an ONNX->TF conversion step (e.g. `onnx2tf`) not bundled
            # here; this call is expected to raise on a plain `edge` install
            # until a SavedModel directory is produced by that extra tool.
            saved_model_dir = output_dir / "saved_model"
            packager.to_tflite_int8(saved_model_dir)
            exported_formats.append("int8_tflite")
        except Exception as exc:  # pragma: no cover - depends on external conversion tooling
            print(f"Skipping TFLite export ({exc}). Run `onnx2tf` on {onnx_path} first.")

    return packager.write_manifest(run_id=config.run_id), exported_formats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/config.json")
    parser.add_argument("--output-dir", default="artifacts/run-001")
    parser.add_argument("--metrics-out", default=None, help="Defaults to <output-dir>/metrics.json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_out = Path(args.metrics_out) if args.metrics_out else output_dir / "metrics.json"

    config = load_config(args.config)
    print(f"Loaded config for run '{config.run_id}': {config.model_dump()}")

    try:
        model, history = train(config)
        manifest, exported_formats = package_for_edge(config, model, output_dir)
        metrics = TrainingMetrics(
            run_id=config.run_id,
            history=history,
            final_model_size_mb=sum(a["size_bytes"] for a in manifest["artifacts"].values()) / 1e6,
            exported_formats=exported_formats,
            status="completed",
        )
    except Exception as exc:
        metrics = TrainingMetrics(run_id=config.run_id, status="failed", error=str(exc))
        raise
    finally:
        save_metrics(metrics, metrics_out)
        print(f"Wrote metrics to {metrics_out}")


if __name__ == "__main__":
    main()
