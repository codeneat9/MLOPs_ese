"""FastAPI inference service for the approved SmartWaste model."""

from __future__ import annotations

import json
import uuid
from collections.abc import Generator
from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from smartwaste.config import Settings, get_settings
from smartwaste.database import FeedbackRecord, PredictionRecord, get_db
from smartwaste.inference import (
    InvalidImageError,
    ModelPredictor,
    ModelUnavailableError,
    Prediction,
)
from smartwaste.labels import CANONICAL_CLASSES


class Predictor(Protocol):
    """Protocol allowing API tests to inject a deterministic predictor."""

    @property
    def model_version(self) -> str: ...

    def predict(self, image_bytes: bytes) -> Prediction: ...


class ProbabilityResponse(BaseModel):
    """Probability response with a fixed five-class schema."""

    plastic: float = Field(ge=0, le=1)
    paper: float = Field(ge=0, le=1)
    glass: float = Field(ge=0, le=1)
    metal: float = Field(ge=0, le=1)
    organic: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    """Public prediction response."""

    request_id: str
    predicted_class: str
    probabilities: ProbabilityResponse
    model_version: str
    latency_ms: float = Field(ge=0)


class HealthResponse(BaseModel):
    """Service health response."""

    status: str
    model_loaded: bool
    model_version: str | None = None


class FeedbackRequest(BaseModel):
    """Reviewed-label submission associated with a prediction request."""

    request_id: str
    corrected_label: str


class FeedbackResponse(BaseModel):
    """Accepted feedback response."""

    feedback_id: int
    request_id: str
    corrected_label: str
    reviewed: bool
    accepted_for_training: bool


class ReviewFeedbackRequest(BaseModel):
    """Explicit review decision for submitted feedback."""

    accepted_for_training: bool


def _prediction_response(request_id: str, prediction: Prediction) -> PredictionResponse:
    return PredictionResponse(
        request_id=request_id,
        predicted_class=prediction.predicted_class,
        probabilities=ProbabilityResponse(**prediction.probabilities),
        model_version=prediction.model_version,
        latency_ms=prediction.latency_ms,
    )


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> FastAPI:
    """Create the API application, optionally with injected settings/predictor."""
    app = FastAPI(
        title="SmartWaste Inference API",
        version="0.1.0",
        description="Waste image classification using the approved SmartWaste checkpoint.",
    )
    runtime_settings = settings or get_settings()
    checkpoint = runtime_settings.model_checkpoint
    model_version: str | None = None
    if runtime_settings.approved_model_manifest.is_file():
        try:
            manifest = json.loads(
                runtime_settings.approved_model_manifest.read_text(encoding="utf-8")
            )
            checkpoint = runtime_settings.approved_model_manifest.parent / manifest["checkpoint"]
            model_version = manifest["version"]
        except (OSError, json.JSONDecodeError, KeyError):
            checkpoint = runtime_settings.model_checkpoint
    runtime_predictor = predictor or ModelPredictor(checkpoint, model_version=model_version)
    def request_db() -> Generator[Session, None, None]:
        yield from get_db(runtime_settings.database_url)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        try:
            version = runtime_predictor.model_version
        except ModelUnavailableError:
            return HealthResponse(status="degraded", model_loaded=False)
        return HealthResponse(status="ok", model_loaded=True, model_version=version)

    @app.get("/classes", response_model=list[str])
    def classes() -> list[str]:
        return list(CANONICAL_CLASSES)

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        file: Annotated[UploadFile, File(...)],
        session: Session = Depends(request_db),
    ) -> PredictionResponse:
        request_id = str(uuid.uuid4())
        content_length = file.headers.get("content-length")
        if content_length:
            try:
                declared_length = int(content_length)
            except ValueError as error:
                raise HTTPException(status_code=400, detail="Invalid content length.") from error
            if declared_length > runtime_settings.max_upload_bytes:
                raise HTTPException(status_code=413, detail="Uploaded file exceeds the size limit.")
        image_bytes = await file.read(runtime_settings.max_upload_bytes + 1)
        if len(image_bytes) > runtime_settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Uploaded file exceeds the size limit.")
        try:
            prediction = runtime_predictor.predict(image_bytes)
        except InvalidImageError as error:
            raise HTTPException(status_code=415, detail=str(error)) from error
        except ModelUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        session.add(
            PredictionRecord(
                request_id=request_id,
                predicted_class=prediction.predicted_class,
                confidence=max(prediction.probabilities.values()),
                model_version=prediction.model_version,
                latency_ms=prediction.latency_ms,
            )
        )
        session.commit()
        return _prediction_response(request_id, prediction)

    @app.post("/feedback", response_model=FeedbackResponse, status_code=201)
    def feedback(
        payload: FeedbackRequest,
        session: Session = Depends(request_db),
    ) -> FeedbackResponse:
        corrected_label = payload.corrected_label.strip().lower()
        if corrected_label not in CANONICAL_CLASSES:
            raise HTTPException(status_code=422, detail="Corrected label is not a valid class.")
        prediction = (
            session.query(PredictionRecord)
            .filter(PredictionRecord.request_id == payload.request_id)
            .one_or_none()
        )
        if prediction is None:
            raise HTTPException(status_code=404, detail="Prediction request was not found.")
        record = FeedbackRecord(
            prediction_id=prediction.id,
            corrected_label=corrected_label,
            reviewed=False,
            accepted_for_training=False,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return FeedbackResponse(
            feedback_id=record.id,
            request_id=payload.request_id,
            corrected_label=record.corrected_label,
            reviewed=record.reviewed,
            accepted_for_training=record.accepted_for_training,
        )

    @app.get("/predictions")
    def prediction_history(
        limit: int = 50,
        session: Session = Depends(request_db),
    ) -> list[dict[str, object]]:
        if not 1 <= limit <= 100:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 100.")
        records = (
            session.query(PredictionRecord)
            .order_by(PredictionRecord.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "request_id": record.request_id,
                "predicted_class": record.predicted_class,
                "confidence": record.confidence,
                "model_version": record.model_version,
                "latency_ms": record.latency_ms,
                "created_at": record.created_at,
            }
            for record in records
        ]

    @app.post("/feedback/{feedback_id}/review", response_model=FeedbackResponse)
    def review_feedback(
        feedback_id: int,
        payload: ReviewFeedbackRequest,
        session: Session = Depends(request_db),
    ) -> FeedbackResponse:
        record = session.get(FeedbackRecord, feedback_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Feedback record was not found.")
        prediction = session.get(PredictionRecord, record.prediction_id)
        if prediction is None:
            raise HTTPException(status_code=500, detail="Feedback has no linked prediction.")
        record.reviewed = True
        record.accepted_for_training = payload.accepted_for_training
        session.commit()
        return FeedbackResponse(
            feedback_id=record.id,
            request_id=prediction.request_id,
            corrected_label=record.corrected_label,
            reviewed=record.reviewed,
            accepted_for_training=record.accepted_for_training,
        )

    @app.get("/metrics/summary")
    def metrics_summary() -> dict[str, object]:
        metrics_path = runtime_settings.evaluation_metrics_path
        if not metrics_path.is_file():
            raise HTTPException(status_code=404, detail="Evaluation metrics are unavailable.")
        try:
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise HTTPException(
                status_code=500, detail="Evaluation metrics could not be read."
            ) from error

    @app.get("/monitoring/summary")
    def monitoring_summary() -> dict[str, object]:
        report_path = runtime_settings.monitoring_report_path
        if not report_path.is_file():
            raise HTTPException(status_code=404, detail="Monitoring report is unavailable.")
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise HTTPException(
                status_code=500, detail="Monitoring report could not be read."
            ) from error

    return app


app = create_app()
