"""Explicit model approval, promotion, and rollback workflow."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from smartwaste.labels import CANONICAL_CLASSES


class ModelGovernanceError(ValueError):
    """Raised when a model cannot pass governance checks."""


def load_metrics(path: Path) -> dict[str, Any]:
    """Load and validate required evaluation metrics."""
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelGovernanceError(f"Unable to read metrics: {path}") from error
    required = ("accuracy", "macro_f1", "macro_precision", "macro_recall")
    if any(key not in metrics for key in required):
        raise ModelGovernanceError(f"Metrics file is missing required fields: {path}")
    return metrics


def approve_candidate(
    checkpoint_path: Path,
    metrics_path: Path,
    approved_dir: Path,
    *,
    minimum_accuracy: float = 0.80,
    minimum_macro_f1: float = 0.80,
    baseline_metrics_path: Path | None = None,
) -> dict[str, Any]:
    """Promote a validated candidate without overwriting an approved model."""
    if not checkpoint_path.is_file():
        raise ModelGovernanceError(f"Candidate checkpoint does not exist: {checkpoint_path}")
    metrics = load_metrics(metrics_path)
    if metrics["accuracy"] < minimum_accuracy or metrics["macro_f1"] < minimum_macro_f1:
        raise ModelGovernanceError("Candidate failed configured approval thresholds.")
    if baseline_metrics_path:
        baseline = load_metrics(baseline_metrics_path)
        if metrics["accuracy"] < baseline["accuracy"] - 0.02:
            raise ModelGovernanceError(
                "Candidate accuracy regressed more than 2 percentage points."
            )
        if metrics["macro_f1"] < baseline["macro_f1"] - 0.02:
            raise ModelGovernanceError(
                "Candidate macro F1 regressed more than 2 percentage points."
            )

    approved_dir.mkdir(parents=True, exist_ok=True)
    version = f"model-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    destination = approved_dir / f"{version}.pt"
    shutil.copy2(checkpoint_path, destination)
    manifest = {
        "version": version,
        "checkpoint": destination.name,
        "source_checkpoint": str(checkpoint_path),
        "metrics": metrics,
        "classes": list(CANONICAL_CLASSES),
        "approved_at": datetime.now(UTC).isoformat(),
    }
    manifest_path = approved_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def rollback(approved_dir: Path) -> dict[str, Any]:
    """Restore the previous approved model from the promotion history."""
    manifest_path = approved_dir / "manifest.json"
    history_path = approved_dir / "history.jsonl"
    if not history_path.is_file():
        raise ModelGovernanceError("No previous approved model is available for rollback.")
    entries = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    if len(entries) < 2:
        raise ModelGovernanceError("No previous approved model is available for rollback.")
    previous = entries[-2]
    manifest_path.write_text(
        json.dumps(previous, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return previous


def record_promotion(manifest: dict[str, Any], approved_dir: Path) -> None:
    """Append a promotion to an auditable local history."""
    with (approved_dir / "history.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(manifest, sort_keys=True) + "\n")


def _main() -> int:
    parser = argparse.ArgumentParser(description="Govern SmartWaste model promotion.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    approve = subparsers.add_parser("approve")
    approve.add_argument("--checkpoint", type=Path, default=Path("models/candidate/best.pt"))
    approve.add_argument("--metrics", type=Path, default=Path("artifacts/evaluation/metrics.json"))
    approve.add_argument("--approved-dir", type=Path, default=Path("models/approved"))
    approve.add_argument("--minimum-accuracy", type=float, default=0.80)
    approve.add_argument("--minimum-macro-f1", type=float, default=0.80)
    approve.add_argument("--baseline-metrics", type=Path)
    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("--approved-dir", type=Path, default=Path("models/approved"))
    args = parser.parse_args()
    if args.command == "approve":
        manifest = approve_candidate(
            args.checkpoint,
            args.metrics,
            args.approved_dir,
            minimum_accuracy=args.minimum_accuracy,
            minimum_macro_f1=args.minimum_macro_f1,
            baseline_metrics_path=args.baseline_metrics,
        )
        record_promotion(manifest, args.approved_dir)
        print(f"Approved {manifest['version']}")
    else:
        print(f"Rolled back to {rollback(args.approved_dir)['version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
