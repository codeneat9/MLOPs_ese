"""Native laptop-camera inference without browser or WebRTC dependencies."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from smartwaste.config import get_settings
from smartwaste.inference import ModelPredictor


def _checkpoint_from_settings() -> tuple[Path, str | None]:
    settings = get_settings()
    checkpoint = settings.model_checkpoint
    version = None
    if settings.approved_model_manifest.is_file():
        manifest = json.loads(
            settings.approved_model_manifest.read_text(encoding="utf-8")
        )
        checkpoint = settings.approved_model_manifest.parent / manifest["checkpoint"]
        version = manifest.get("version")
    return checkpoint, version


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show live waste predictions from a local camera."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index reported by macOS (default: 0).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.35,
        help="Minimum seconds between predictions (default: 0.35).",
    )
    return parser.parse_args()


def main() -> None:
    """Open the selected camera and overlay predictions until Q or Escape."""
    args = _parse_args()
    if args.interval <= 0:
        raise SystemExit("--interval must be greater than zero.")

    try:
        import cv2
    except ImportError as error:
        raise SystemExit(
            "OpenCV is not installed. Run `./.venv/bin/pip install -e .` first."
        ) from error

    checkpoint, model_version = _checkpoint_from_settings()
    predictor = ModelPredictor(checkpoint, model_version=model_version)
    camera = cv2.VideoCapture(args.camera_index, cv2.CAP_AVFOUNDATION)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        camera.release()
        raise SystemExit(
            f"Camera index {args.camera_index} could not be opened. "
            "Allow camera access in System Settings > Privacy & Security > Camera, "
            "then close other camera apps and try again."
        )

    label = "Place one waste item in view"
    confidence = 0.0
    last_prediction = 0.0
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("The camera stopped returning frames.")

            now = time.monotonic()
            if now - last_prediction >= args.interval:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = predictor.predict_bytes(rgb_frame.tobytes(), rgb_frame.shape)
                label = result.predicted_class.title()
                confidence = result.probabilities[result.predicted_class]
                last_prediction = now

            text = f"{label}  {confidence:.1%}" if confidence else label
            cv2.rectangle(frame, (12, 12), (390, 62), (0, 0, 0), -1)
            cv2.putText(
                frame,
                text,
                (24, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("SmartWaste Live Camera - press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
