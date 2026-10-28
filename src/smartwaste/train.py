"""Command-line entry point for candidate model training."""

from __future__ import annotations

import argparse
import ssl
from pathlib import Path
from urllib.error import URLError

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a SmartWaste EfficientNet-B0 model.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/candidate"))
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use cached/downloaded ImageNet weights (default: params.yaml value).",
    )
    args = parser.parse_args()

    try:
        from smartwaste.training import TrainingConfig, train_model
    except ImportError as error:
        raise RuntimeError(
            "Training requires the project dependencies. Install with "
            '`python -m pip install -e ".[dev,dvc]"`.'
        ) from error

    parameters = yaml.safe_load(args.params.read_text(encoding="utf-8"))
    dataset_params = parameters["dataset"]
    training_params = parameters["training"]
    config = TrainingConfig(
        image_size=dataset_params["image_size"],
        seed=parameters["seed"],
        **training_params
        | (
            {"pretrained": args.pretrained}
            if args.pretrained is not None
            else {}
        ),
    )
    try:
        result = train_model(
            args.processed_dir,
            args.output_dir,
            config,
            tracking_uri=args.tracking_uri,
        )
    except (URLError, ssl.SSLError) as error:
        raise RuntimeError(
            "The pretrained EfficientNet weights could not be downloaded because "
            "the Python SSL certificate could not be verified. Install the macOS "
            "Python certificates, or rerun with --no-pretrained for an offline "
            "baseline. The offline run is not pretrained transfer learning."
        ) from error
    print(f"Training finished on {result['device']}; best epoch: {result['best_epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
