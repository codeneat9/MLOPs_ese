"""EfficientNet-B0 training and checkpoint utilities."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from smartwaste.labels import CANONICAL_CLASSES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CanonicalImageFolder(datasets.ImageFolder):
    """ImageFolder that remaps alphabetical directory indices to canonical indices."""

    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        super().__init__(root, transform=transform)
        if set(self.classes) != set(CANONICAL_CLASSES):
            raise ValueError(
                f"Unexpected classes in {root}: {self.classes}; "
                f"expected {list(CANONICAL_CLASSES)}"
            )
        self._canonical_indices = {
            self.class_to_idx[label]: index for index, label in enumerate(CANONICAL_CLASSES)
        }

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, target = super().__getitem__(index)
        return image, self._canonical_indices[target]


@dataclass(frozen=True)
class TrainingConfig:
    """Validated options controlling a training run."""

    image_size: int = 224
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    early_stopping_patience: int = 5
    num_workers: int = 2
    pretrained: bool = True
    seed: int = 42


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Select CUDA, Apple MPS, or CPU in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms(image_size: int, training: bool) -> transforms.Compose:
    """Build training or inference preprocessing with shared normalization."""
    operations: list[Any] = []
    if training:
        operations.extend(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ]
        )
    else:
        operations.extend([transforms.Resize((image_size, image_size))])
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


def build_model(*, num_classes: int = len(CANONICAL_CLASSES), pretrained: bool = True) -> nn.Module:
    """Create EfficientNet-B0 with a five-class classification head."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    classifier_input = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(classifier_input, num_classes)
    return model


def create_dataloaders(
    processed_dir: Path,
    config: TrainingConfig,
) -> dict[str, DataLoader]:
    """Create loaders from the prepared train/validation/test directories."""
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "validation", "test"):
        dataset = CanonicalImageFolder(
            processed_dir / split,
            transform=build_transforms(config.image_size, training=split == "train"),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    if total == 0:
        raise ValueError("Cannot run an epoch with an empty dataset.")
    return total_loss / total, correct / total


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainingConfig,
    *,
    best_epoch: int,
    validation_metrics: dict[str, float],
) -> None:
    """Save model weights and all preprocessing metadata needed for inference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "efficientnet_b0",
            "classes": list(CANONICAL_CLASSES),
            "image_size": config.image_size,
            "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
            "training_config": config.__dict__,
            "best_epoch": best_epoch,
            "validation_metrics": validation_metrics,
        },
        path,
    )


def save_training_plot(history: list[dict[str, float | int]], output_dir: Path) -> None:
    """Save loss and accuracy curves for a completed training run."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Training plots require matplotlib.") from error
    epochs = [int(item["epoch"]) for item in history]
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4))
    loss_axis.plot(epochs, [item["train_loss"] for item in history], label="train")
    loss_axis.plot(epochs, [item["validation_loss"] for item in history], label="validation")
    loss_axis.set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    loss_axis.legend()
    accuracy_axis.plot(epochs, [item["train_accuracy"] for item in history], label="train")
    accuracy_axis.plot(
        epochs, [item["validation_accuracy"] for item in history], label="validation"
    )
    accuracy_axis.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    accuracy_axis.legend()
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(figure)


def train_model(
    processed_dir: Path,
    output_dir: Path,
    config: TrainingConfig,
    *,
    tracking_uri: str | None = None,
    experiment_name: str = "smartwaste-classification",
) -> dict[str, Any]:
    """Train EfficientNet-B0 and return the training history."""
    seed_everything(config.seed)
    device = get_device()
    loaders = create_dataloaders(processed_dir, config)
    model = build_model(pretrained=config.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    patience = 0
    checkpoint_path = output_dir / "best.pt"

    mlflow_run = None
    if tracking_uri:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow_run = mlflow.start_run()
        mlflow.log_params(config.__dict__ | {"device": str(device)})

    try:
        for epoch in range(1, config.epochs + 1):
            train_loss, train_accuracy = _run_epoch(
                model, loaders["train"], criterion, device, optimizer
            )
            validation_loss, validation_accuracy = _run_epoch(
                model, loaders["validation"], criterion, device
            )
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
            history.append(metrics)
            if mlflow_run:
                mlflow.log_metrics(
                    {key: value for key, value in metrics.items() if key != "epoch"},
                    step=epoch,
                )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                patience = 0
                save_checkpoint(
                    checkpoint_path,
                    model,
                    config,
                    best_epoch=epoch,
                    validation_metrics={
                        "loss": validation_loss,
                        "accuracy": validation_accuracy,
                    },
                )
            else:
                patience += 1
                if patience >= config.early_stopping_patience:
                    break
    except Exception:
        if mlflow_run:
            import mlflow

            mlflow.end_run(status="FAILED")
        raise
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    save_training_plot(history, output_dir)
    if mlflow_run:
        import mlflow

        mlflow.log_artifact(str(checkpoint_path))
        mlflow.log_artifact(str(history_path))
        mlflow.log_artifact(str(output_dir / "training_curves.png"))
        mlflow.end_run()
    return {"history": history, "best_epoch": best_epoch, "device": str(device)}


def load_image(path: Path, image_size: int) -> torch.Tensor:
    """Load one image with exactly the inference preprocessing."""
    with Image.open(path) as image:
        return build_transforms(image_size, training=False)(image.convert("RGB")).unsqueeze(0)
