from pathlib import Path

from smartwaste.database import FeedbackRecord, PredictionRecord, create_session_factory


def test_database_initializes_and_links_feedback(tmp_path: Path) -> None:
    factory = create_session_factory(f"sqlite:///{tmp_path / 'smartwaste.db'}")
    with factory() as session:
        prediction = PredictionRecord(
            request_id="request-1",
            predicted_class="paper",
            confidence=0.9,
            model_version="test-model",
            latency_ms=4.2,
        )
        session.add(prediction)
        session.commit()
        session.refresh(prediction)
        feedback = FeedbackRecord(
            prediction_id=prediction.id,
            corrected_label="plastic",
            reviewed=False,
            accepted_for_training=False,
        )
        session.add(feedback)
        session.commit()

        assert session.query(PredictionRecord).count() == 1
        assert session.query(FeedbackRecord).one().prediction_id == prediction.id
