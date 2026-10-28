"""Evaluation metrics and artifacts for a trained SmartWaste checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from smartwaste.labels import CANONICAL_CLASSES


def evaluate_predictions(
    y_true: list[int], y_pred: list[int], output_dir: Path
) -> dict[str, Any]:
    """Calculate and persist evaluation artifacts for already-collected predictions."""
    metrics = calculate_metrics(y_true, y_pred)
    save_evaluation(metrics, output_dir)
    return metrics


def main() -> int:
    """Evaluate a checkpoint against the held-out test split."""
    parser = argparse.ArgumentParser(description="Evaluate a SmartWaste checkpoint.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/candidate/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    args = parser.parse_args()

    try:
        import torch

        from smartwaste.training import TrainingConfig, build_model, create_dataloaders, get_device
    except ImportError as error:
        raise RuntimeError(
            "Evaluation requires the training dependencies. Install the project first."
        ) from error

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    training_config = TrainingConfig(**checkpoint["training_config"])
    loaders = create_dataloaders(args.processed_dir, training_config)
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(get_device()).eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.inference_mode():
        for images, labels in loaders["test"]:
            logits = model(images.to(get_device()))
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = evaluate_predictions(y_true, y_pred, args.output_dir)
    print(f"Test accuracy: {metrics['accuracy']:.4f}; macro F1: {metrics['macro_f1']:.4f}")
    return 0


def calculate_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    """Calculate aggregate and per-class classification metrics."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CANONICAL_CLASSES))),
        average=None,
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CANONICAL_CLASSES))),
        target_names=list(CANONICAL_CLASSES),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(CANONICAL_CLASSES)
        },
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(CANONICAL_CLASSES)))
        ).tolist(),
    }


def save_evaluation(metrics: dict[str, Any], output_dir: Path) -> None:
    """Save JSON metrics, CSV data, and a confusion-matrix plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    matrix = metrics["confusion_matrix"]
    lines = [",".join(["actual/predicted", *CANONICAL_CLASSES])]
    lines.extend(
        ",".join([label, *(str(value) for value in row)])
        for label, row in zip(CANONICAL_CLASSES, matrix, strict=False)
    )
    (output_dir / "confusion_matrix.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Plot generation requires matplotlib.") from error
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(CANONICAL_CLASSES)),
        yticks=range(len(CANONICAL_CLASSES)),
        xticklabels=CANONICAL_CLASSES,
        yticklabels=CANONICAL_CLASSES,
        xlabel="Predicted label",
        ylabel="True label",
        title="SmartWaste confusion matrix",
    )
    threshold = max(max(row) for row in matrix) / 2 if matrix else 0
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
