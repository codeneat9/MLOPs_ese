from pathlib import Path

import json

from smartwaste.evaluate import calculate_metrics, save_evaluation


def test_calculate_metrics_contains_required_aggregates() -> None:
    metrics = calculate_metrics(
        [0, 1, 2, 3, 4],
        [0, 1, 1, 3, 4],
    )

    assert metrics["accuracy"] == 0.8
    assert set(metrics["per_class"]) == {"plastic", "paper", "glass", "metal", "organic"}
    assert len(metrics["confusion_matrix"]) == 5


def test_save_evaluation_writes_json_and_confusion_matrix(tmp_path: Path) -> None:
    metrics = calculate_metrics([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])

    save_evaluation(metrics, tmp_path)

    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["macro_f1"] == 1.0
    assert (tmp_path / "confusion_matrix.csv").is_file()
