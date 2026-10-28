import json
from pathlib import Path

import pytest

from smartwaste.governance import ModelGovernanceError, approve_candidate, rollback


def _fixtures(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "accuracy": 0.9,
                "macro_f1": 0.88,
                "macro_precision": 0.89,
                "macro_recall": 0.87,
            }
        ),
        encoding="utf-8",
    )
    return checkpoint, metrics


def test_approval_preserves_candidate_and_writes_manifest(tmp_path: Path) -> None:
    checkpoint, metrics = _fixtures(tmp_path)

    manifest = approve_candidate(checkpoint, metrics, tmp_path / "approved")

    assert manifest["version"].startswith("model-")
    assert (tmp_path / "approved" / manifest["checkpoint"]).read_bytes() == b"checkpoint"
    assert checkpoint.read_bytes() == b"checkpoint"


def test_approval_rejects_below_threshold(tmp_path: Path) -> None:
    checkpoint, metrics = _fixtures(tmp_path)
    data = json.loads(metrics.read_text(encoding="utf-8"))
    data["macro_f1"] = 0.2
    metrics.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ModelGovernanceError, match="thresholds"):
        approve_candidate(checkpoint, metrics, tmp_path / "approved")


def test_rollback_requires_history(tmp_path: Path) -> None:
    with pytest.raises(ModelGovernanceError, match="previous"):
        rollback(tmp_path / "approved")
