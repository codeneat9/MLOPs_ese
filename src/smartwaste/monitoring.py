"""Local prediction monitoring and drift reporting."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from smartwaste.database import PredictionRecord, get_session_factory
from smartwaste.labels import CANONICAL_CLASSES


def _distribution(values: pd.Series, labels: tuple[str, ...]) -> dict[str, float]:
    counts = values.value_counts(normalize=True)
    return {label: float(counts.get(label, 0.0)) for label in labels}


def _psi(reference: dict[str, float], current: dict[str, float]) -> float:
    score = 0.0
    for label in reference:
        expected = max(reference[label], 1e-6)
        actual = max(current.get(label, 0.0), 1e-6)
        score += (actual - expected) * math.log(actual / expected)
    return float(score)


def build_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    psi_threshold: float = 0.20,
    confidence_threshold: float = 0.05,
) -> dict[str, Any]:
    """Compare reference and current predictions using transparent local metrics."""
    required = {"predicted_class", "confidence"}
    if not required.issubset(reference.columns) or not required.issubset(current.columns):
        raise ValueError(f"Both datasets must contain {sorted(required)}.")
    reference_distribution = _distribution(reference["predicted_class"], CANONICAL_CLASSES)
    current_distribution = _distribution(current["predicted_class"], CANONICAL_CLASSES)
    reference_confidence = float(reference["confidence"].mean()) if len(reference) else 0.0
    current_confidence = float(current["confidence"].mean()) if len(current) else 0.0
    confidence_delta = current_confidence - reference_confidence
    psi = _psi(reference_distribution, current_distribution)
    warnings = []
    if psi >= psi_threshold:
        warnings.append(f"Prediction distribution drift exceeded PSI threshold ({psi:.3f}).")
    if confidence_delta <= -confidence_threshold:
        warnings.append(
            "Mean confidence dropped by "
            f"{-confidence_delta:.3f}, exceeding the configured threshold."
        )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "reference_count": int(len(reference)),
        "current_count": int(len(current)),
        "metrics": {
            "population_stability_index": psi,
            "reference_mean_confidence": reference_confidence,
            "current_mean_confidence": current_confidence,
            "confidence_delta": confidence_delta,
        },
        "thresholds": {
            "population_stability_index": psi_threshold,
            "confidence_drop": confidence_threshold,
        },
        "reference_distribution": reference_distribution,
        "current_distribution": current_distribution,
        "drift_detected": bool(warnings),
        "warnings": warnings,
    }


def load_prediction_frames(
    database_url: str,
    reference_path: Path,
    *,
    current_limit: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a reference CSV and the latest prediction records from SQLite."""
    reference = pd.read_csv(reference_path)
    session = get_session_factory(database_url)()
    try:
        records = (
            session.query(PredictionRecord)
            .order_by(PredictionRecord.created_at.desc())
            .limit(current_limit)
            .all()
        )
        current = pd.DataFrame(
            [
                {"predicted_class": row.predicted_class, "confidence": row.confidence}
                for row in records
            ]
        )
    finally:
        session.close()
    return reference, current


def write_drift_report(
    database_url: str,
    reference_path: Path,
    output_path: Path,
    *,
    current_limit: int = 1000,
    psi_threshold: float = 0.20,
    confidence_threshold: float = 0.05,
) -> dict[str, Any]:
    """Generate and persist a prediction drift report."""
    reference, current = load_prediction_frames(
        database_url, reference_path, current_limit=current_limit
    )
    report = build_drift_report(
        reference,
        current,
        psi_threshold=psi_threshold,
        confidence_threshold=confidence_threshold,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a local SmartWaste drift report.")
    parser.add_argument("--database-url", default="sqlite:///./smartwaste.db")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/monitoring/drift.json"))
    parser.add_argument("--current-limit", type=int, default=1000)
    parser.add_argument("--psi-threshold", type=float, default=0.20)
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    args = parser.parse_args()
    report = write_drift_report(
        args.database_url,
        args.reference,
        args.output,
        current_limit=args.current_limit,
        psi_threshold=args.psi_threshold,
        confidence_threshold=args.confidence_threshold,
    )
    print(json.dumps({"drift_detected": report["drift_detected"], "warnings": report["warnings"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
