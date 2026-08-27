"""Colab-side training script for the Claude-Colab edge bridge.

Intended to run as a Colab cell (or any machine with the `edge` extra
installed). It reads a `config.json` written by the orchestrating agent,
trains a small PyTorch model on the real `sklearn.datasets.load_digits`
dataset (1,797 real handwritten-digit images, bundled with scikit-learn —
no network download required, unlike `torchvision.datasets.MNIST`), writes
a `metrics.json` history back, and packages the trained model into INT8
ONNX and a genuinely fully-INT8-quantized TFLite model (via `onnx2tf`,
calibrated on real held-out validation images) plus a `manifest.json`
describing every artifact produced.

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
    """A minimal CNN sized for edge deployment (8x8 grayscale digit input)."""
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


def real_digits_dataloaders(config: TrainingConfig):
    """Load the real, bundled `sklearn.datasets.load_digits` dataset.

    1,797 real 8x8 grayscale handwritten-digit images (10 classes), shipped
    with scikit-learn — genuine data with no network download required,
    unlike `torchvision.datasets.MNIST`. Pixel values (0-16) are scaled to
    [0, 1]. Returns train/val DataLoaders plus the raw validation tensor,
    which doubles as real calibration data for INT8 TFLite export.
    """
    import torch
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset

    images, labels = load_digits(return_X_y=True)
    x_train, x_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=0, stratify=labels
    )

    def to_tensor(x, y):
        x_t = torch.tensor(x, dtype=torch.float32).reshape(-1, 1, 8, 8) / 16.0
        y_t = torch.tensor(y, dtype=torch.long)
        return x_t, y_t

    x_train_t, y_train_t = to_tensor(x_train, y_train)
    x_val_t, y_val_t = to_tensor(x_val, y_val)

    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t), batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(x_val_t, y_val_t), batch_size=config.batch_size)
    return train_loader, val_loader, x_val_t


def train(config: TrainingConfig) -> tuple:
    import torch
    import torch.nn as nn

    model = build_model(config)
    optimizer = build_optimizer(config, model)
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader, calibration_data = real_digits_dataloaders(config)

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

    return model, history, calibration_data


def package_for_edge(config: TrainingConfig, model, calibration_data, output_dir: Path) -> dict:
    packager = EdgePackager(output_dir=output_dir)
    sample_input = calibration_data[:1]  # one real, normalized validation image: (1, 1, 8, 8)

    exported_formats = []
    # Fixed batch size (1): edge/embedded deployments run one inference at a
    # time, and `dynamic_axes` is unreliable with torch's dynamo-based ONNX
    # exporter (see the caveat on EdgePackager.to_onnx) — a fixed shape sidesteps it.
    onnx_path = packager.to_onnx(model, sample_input)
    exported_formats.append("onnx")

    if "int8_onnx" in config.target_formats:
        packager.quantize_onnx_int8(onnx_path)
        exported_formats.append("int8_onnx")

    if "int8_tflite" in config.target_formats:
        # Calibrate on real held-out validation images (not random noise) so
        # the resulting INT8 quantization ranges actually reflect the data
        # this model will see in production.
        packager.to_tflite_int8_from_onnx(onnx_path, calibration_data.numpy())
        exported_formats.append("int8_tflite")

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
        model, history, calibration_data = train(config)
        manifest, exported_formats = package_for_edge(config, model, calibration_data, output_dir)
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
