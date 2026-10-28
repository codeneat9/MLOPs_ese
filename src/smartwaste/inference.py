"""Checkpoint-backed image inference for the SmartWaste API."""

from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from smartwaste.labels import CANONICAL_CLASSES


class ModelUnavailableError(RuntimeError):
    """Raised when the approved serving checkpoint cannot be loaded."""


class InvalidImageError(ValueError):
    """Raised when uploaded bytes are not a supported readable image."""


@dataclass(frozen=True)
class Prediction:
    """Prediction result returned by the inference service."""

    predicted_class: str
    probabilities: dict[str, float]
    model_version: str
    latency_ms: float


class ModelPredictor:
    """Lazy-loading predictor that uses checkpoint-stored preprocessing metadata."""

    def __init__(self, checkpoint_path: Path, model_version: str | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        self._configured_model_version = model_version
        self._model: Any = None
        self._device: Any = None
        self._image_size: int | None = None
        self._model_version: str | None = None
        self._inference_lock = threading.Lock()

    @property
    def model_version(self) -> str:
        """Return a stable version identifier based on checkpoint metadata."""
        self._ensure_loaded()
        assert self._model_version is not None
        return self._model_version

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.checkpoint_path.is_file():
            raise ModelUnavailableError(
                f"Model checkpoint is unavailable: {self.checkpoint_path}"
            )
        try:
            import torch
            from smartwaste.training import build_model, get_device
        except ImportError as error:
            raise ModelUnavailableError(
                "Model serving dependencies are not installed."
            ) from error
        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            classes = checkpoint.get("classes")
            if classes != list(CANONICAL_CLASSES):
                raise ValueError("checkpoint class ordering does not match canonical classes")
            self._image_size = int(checkpoint["image_size"])
            self._model_version = self._configured_model_version or (
                f"checkpoint-{self.checkpoint_path.stat().st_mtime_ns}"
            )
            model = build_model(pretrained=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            self._device = get_device()
            self._model = model.to(self._device).eval()
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            raise ModelUnavailableError("Model checkpoint could not be loaded.") from error

    def predict(self, image_bytes: bytes) -> Prediction:
        """Validate image bytes and return a probability distribution."""
        self._ensure_loaded()
        assert self._image_size is not None
        try:
            from smartwaste.training import build_transforms
            import torch

            with Image.open(io.BytesIO(image_bytes)) as image:
                image.verify()
            with Image.open(io.BytesIO(image_bytes)) as image:
                tensor = build_transforms(self._image_size, training=False)(
                    image.convert("RGB")
                ).unsqueeze(0)
        except (OSError, UnidentifiedImageError, ValueError) as error:
            raise InvalidImageError("The uploaded file is not a readable image.") from error

        started = time.perf_counter()
        with torch.inference_mode():
            with self._inference_lock:
                logits = self._model(tensor.to(self._device))
            probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
        latency_ms = (time.perf_counter() - started) * 1000
        probability_map = {
            label: round(float(probabilities[index]), 6)
            for index, label in enumerate(CANONICAL_CLASSES)
        }
        predicted_index = int(max(range(len(probabilities)), key=probabilities.__getitem__))
        return Prediction(
            predicted_class=CANONICAL_CLASSES[predicted_index],
            probabilities=probability_map,
            model_version=self.model_version,
            latency_ms=round(latency_ms, 3),
        )

    def predict_bytes(self, data: bytes, shape: tuple[int, ...]) -> Prediction:
        """Predict an RGB ndarray payload supplied by a webcam callback."""
        try:
            import numpy as np

            image = Image.fromarray(np.frombuffer(data, dtype=np.uint8).reshape(shape))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
        except (ValueError, OSError) as error:
            raise InvalidImageError("The camera frame could not be decoded.") from error
        return self.predict(buffer.getvalue())
