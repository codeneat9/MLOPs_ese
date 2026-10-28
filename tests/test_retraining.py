import json
from pathlib import Path

import pytest

from smartwaste.governance import ModelGovernanceError
from smartwaste.retraining import validate_candidate


def _metrics(path: Path, accuracy: float, macro_f1: float) -> None:
    path.write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "macro_precision": macro_f1,
                "macro_recall": macro_f1,
            }
        ),
        encoding="utf-8",
    )


def test_retraining_gate_accepts_candidate_with_bounded_regression(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _metrics(candidate, 0.91, 0.90)
    _metrics(baseline, 0.90, 0.89)

    result = validate_candidate(candidate, baseline)

    assert result["approved"] is True


def test_retraining_gate_rejects_regression(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _metrics(candidate, 0.87, 0.86)
    _metrics(baseline, 0.90, 0.90)

    with pytest.raises(ModelGovernanceError, match="regressed"):
        validate_candidate(candidate, baseline)
