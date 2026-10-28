"""Feedback export and controlled candidate retraining gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from smartwaste.database import FeedbackRecord, PredictionRecord, get_session_factory
from smartwaste.governance import (
    ModelGovernanceError,
    approve_candidate,
    load_metrics,
    record_promotion,
)


def export_reviewed_feedback(database_url: str, output_path: Path) -> int:
    """Export only reviewed feedback accepted for training."""
    session = get_session_factory(database_url)()
    try:
        rows = (
            session.query(FeedbackRecord, PredictionRecord)
            .join(PredictionRecord, FeedbackRecord.prediction_id == PredictionRecord.id)
            .filter(
                FeedbackRecord.reviewed.is_(True),
                FeedbackRecord.accepted_for_training.is_(True),
            )
            .all()
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            for feedback, prediction in rows:
                file.write(
                    json.dumps(
                        {
                            "request_id": prediction.request_id,
                            "predicted_class": prediction.predicted_class,
                            "corrected_label": feedback.corrected_label,
                            "model_version": prediction.model_version,
                        }
                    )
                    + "\n"
                )
        return len(rows)
    finally:
        session.close()


def validate_candidate(
    candidate_metrics_path: Path,
    baseline_metrics_path: Path,
    *,
    minimum_accuracy: float = 0.80,
    minimum_macro_f1: float = 0.80,
    maximum_regression: float = 0.02,
) -> dict[str, Any]:
    """Require absolute quality and bounded regression against the approved baseline."""
    candidate = load_metrics(candidate_metrics_path)
    baseline = load_metrics(baseline_metrics_path)
    if candidate["accuracy"] < minimum_accuracy or candidate["macro_f1"] < minimum_macro_f1:
        raise ModelGovernanceError("Candidate failed minimum accuracy or macro-F1 thresholds.")
    for metric in ("accuracy", "macro_f1"):
        if candidate[metric] < baseline[metric] - maximum_regression:
            raise ModelGovernanceError(f"Candidate {metric} regressed beyond the allowed gate.")
    return {
        "candidate": {metric: candidate[metric] for metric in ("accuracy", "macro_f1")},
        "baseline": {metric: baseline[metric] for metric in ("accuracy", "macro_f1")},
        "approved": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controlled SmartWaste retraining gates.")
    parser.add_argument("--database-url", default="sqlite:///./smartwaste.db")
    parser.add_argument(
        "--feedback-output",
        type=Path,
        default=Path("data/feedback/accepted.jsonl"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--approved-dir", type=Path, default=Path("models/approved"))
    parser.add_argument(
        "--minimum-feedback",
        type=int,
        default=1,
        help="Minimum number of reviewed feedback records required to retrain.",
    )
    args = parser.parse_args()
    count = export_reviewed_feedback(args.database_url, args.feedback_output)
    if count < args.minimum_feedback:
        raise ModelGovernanceError(
            f"Retraining trigger not met: {count} accepted records; "
            f"{args.minimum_feedback} required."
        )
    gate = validate_candidate(args.metrics, args.baseline_metrics)
    manifest = approve_candidate(
        args.checkpoint,
        args.metrics,
        args.approved_dir,
        baseline_metrics_path=args.baseline_metrics,
    )
    record_promotion(manifest, args.approved_dir)
    print(json.dumps({"accepted_feedback": count, "gate": gate, "version": manifest["version"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
