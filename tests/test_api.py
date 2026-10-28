import json
from pathlib import Path

from fastapi.testclient import TestClient

from smartwaste.api import create_app
from smartwaste.config import Settings
from smartwaste.inference import InvalidImageError, Prediction


class FakePredictor:
    model_version = "test-model"

    def predict(self, image_bytes: bytes) -> Prediction:
        if image_bytes == b"invalid":
            raise InvalidImageError("The uploaded file is not a readable image.")
        return Prediction(
            predicted_class="plastic",
            probabilities={
                "plastic": 0.8,
                "paper": 0.1,
                "glass": 0.05,
                "metal": 0.03,
                "organic": 0.02,
            },
            model_version=self.model_version,
            latency_ms=1.25,
        )


def _client(tmp_path: Path) -> TestClient:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"accuracy": 0.9}), encoding="utf-8")
    settings = Settings(
        max_upload_bytes=10,
        evaluation_metrics_path=metrics_path,
        model_checkpoint=tmp_path / "missing.pt",
        database_url=f"sqlite:///{tmp_path / 'smartwaste.db'}",
    )
    return TestClient(create_app(settings, FakePredictor()))


def test_health_classes_and_metrics(tmp_path: Path) -> None:
    client = _client(tmp_path)

    assert client.get("/health").json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": "test-model",
    }
    assert client.get("/classes").json() == ["plastic", "paper", "glass", "metal", "organic"]
    assert client.get("/metrics/summary").json() == {"accuracy": 0.9}


def test_monitoring_summary_is_available_when_report_exists(tmp_path: Path) -> None:
    report_path = tmp_path / "drift.json"
    report_path.write_text(json.dumps({"drift_detected": False}), encoding="utf-8")
    settings = Settings(
        monitoring_report_path=report_path,
        model_checkpoint=tmp_path / "missing.pt",
        database_url=f"sqlite:///{tmp_path / 'monitoring.db'}",
    )

    response = TestClient(create_app(settings, FakePredictor())).get("/monitoring/summary")

    assert response.status_code == 200
    assert response.json() == {"drift_detected": False}


def test_predict_returns_probabilities_and_request_id(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/predict",
        files={"file": ("sample.jpg", b"valid", "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "plastic"
    assert body["model_version"] == "test-model"
    assert len(body["request_id"]) == 36
    assert set(body["probabilities"]) == {"plastic", "paper", "glass", "metal", "organic"}


def test_feedback_requires_existing_prediction_and_valid_label(tmp_path: Path) -> None:
    client = _client(tmp_path)

    missing = client.post(
        "/feedback",
        json={"request_id": "missing", "corrected_label": "plastic"},
    )
    invalid = client.post(
        "/feedback",
        json={"request_id": "missing", "corrected_label": "trash"},
    )

    assert missing.status_code == 404
    assert invalid.status_code == 422


def test_prediction_history_and_feedback_review(tmp_path: Path) -> None:
    client = _client(tmp_path)
    prediction = client.post(
        "/predict",
        files={"file": ("sample.jpg", b"valid", "image/jpeg")},
    ).json()
    feedback = client.post(
        "/feedback",
        json={"request_id": prediction["request_id"], "corrected_label": "plastic"},
    ).json()

    history = client.get("/predictions").json()
    reviewed = client.post(
        f"/feedback/{feedback['feedback_id']}/review",
        json={"accepted_for_training": True},
    )

    assert history[0]["request_id"] == prediction["request_id"]
    assert reviewed.status_code == 200
    assert reviewed.json()["accepted_for_training"] is True


def test_predict_rejects_invalid_images_and_oversized_uploads(tmp_path: Path) -> None:
    client = _client(tmp_path)

    invalid = client.post(
        "/predict",
        files={"file": ("sample.jpg", b"invalid", "image/jpeg")},
    )
    oversized = client.post(
        "/predict",
        files={"file": ("sample.jpg", b"12345678901", "image/jpeg")},
    )

    assert invalid.status_code == 415
    assert oversized.status_code == 413
