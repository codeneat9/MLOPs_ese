import pandas as pd

from smartwaste.monitoring import build_drift_report


def test_monitoring_reports_distribution_drift() -> None:
    reference = pd.DataFrame(
        {"predicted_class": ["plastic", "paper", "glass", "metal", "organic"] * 4,
         "confidence": [0.9] * 20}
    )
    current = pd.DataFrame(
        {"predicted_class": ["plastic"] * 20, "confidence": [0.7] * 20}
    )

    report = build_drift_report(reference, current)

    assert report["drift_detected"] is True
    assert report["metrics"]["population_stability_index"] > 0.20
    assert report["warnings"]


def test_monitoring_accepts_stable_predictions() -> None:
    frame = pd.DataFrame(
        {"predicted_class": ["plastic", "paper", "glass", "metal", "organic"] * 4,
         "confidence": [0.9] * 20}
    )

    report = build_drift_report(frame, frame)

    assert report["drift_detected"] is False
    assert report["warnings"] == []
