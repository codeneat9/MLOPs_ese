"""Streamlit dashboard for SmartWaste classification and feedback."""

from __future__ import annotations

import io
import os
import threading
import time
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from PIL import Image

from smartwaste.labels import CANONICAL_CLASSES

DEFAULT_API_URL = "http://127.0.0.1:8000"


def _api_url() -> str:
    return os.getenv("SMARTWASTE_API_URL", DEFAULT_API_URL).rstrip("/")


def _request(method: str, path: str, **kwargs: Any) -> requests.Response:
    try:
        response = requests.request(method, f"{_api_url()}{path}", timeout=30, **kwargs)
        response.raise_for_status()
        return response
    except requests.RequestException as error:
        raise RuntimeError(f"SmartWaste API request failed: {error}") from error


def _render_prediction(result: dict[str, Any]) -> None:
    st.subheader(f"Prediction: {result['predicted_class'].title()}")
    probabilities = result["probabilities"]
    confidence = probabilities[result["predicted_class"]]
    st.metric("Confidence", f"{confidence:.2%}")
    st.caption(
        f"Model: {result['model_version']} | "
        f"Inference latency: {result['latency_ms']:.1f} ms | "
        f"Request: {result['request_id']}"
    )
    chart_data = pd.DataFrame(
        {"class": list(probabilities), "probability": list(probabilities.values())}
    )
    st.plotly_chart(
        px.bar(chart_data, x="class", y="probability", range_y=[0, 1]),
        use_container_width=True,
    )
    st.session_state["last_prediction"] = result


def _predict_image(image_bytes: bytes, filename: str) -> None:
    try:
        result = _request(
            "POST",
            "/predict",
            files={"file": (filename, io.BytesIO(image_bytes), "image/jpeg")},
        ).json()
    except RuntimeError as error:
        st.error(str(error))
        return
    _render_prediction(result)


def _render_feedback() -> None:
    prediction = st.session_state.get("last_prediction")
    if not prediction:
        return
    st.subheader("Submit correction")
    corrected = st.selectbox("Correct class", CANONICAL_CLASSES)
    if st.button("Submit feedback"):
        try:
            response = _request(
                "POST",
                "/feedback",
                json={
                    "request_id": prediction["request_id"],
                    "corrected_label": corrected,
                },
            )
            st.success(f"Feedback saved for review: {response.json()['feedback_id']}")
        except RuntimeError as error:
            st.error(str(error))


def _render_history() -> None:
    st.header("Prediction history")
    try:
        records = _request("GET", "/predictions?limit=100").json()
    except RuntimeError as error:
        st.error(str(error))
        return
    if not records:
        st.info("No predictions have been logged yet.")
        return
    frame = pd.DataFrame(records)
    frame["created_at"] = pd.to_datetime(frame["created_at"])
    st.dataframe(frame, use_container_width=True, hide_index=True)
    counts = frame["predicted_class"].value_counts().reindex(CANONICAL_CLASSES, fill_value=0)
    st.plotly_chart(
        px.bar(
            counts.reset_index(name="count"),
            x="predicted_class",
            y="count",
            title="Prediction distribution",
        ),
        use_container_width=True,
    )


def _render_metrics() -> None:
    st.header("Model performance")
    try:
        metrics = _request("GET", "/metrics/summary").json()
    except RuntimeError as error:
        st.error(str(error))
        return
    columns = st.columns(4)
    for column, key in zip(
        columns,
        ("accuracy", "macro_precision", "macro_recall", "macro_f1"),
    ):
        column.metric(key.replace("_", " ").title(), f"{metrics[key]:.2%}")
    per_class = pd.DataFrame(metrics["per_class"]).T.reset_index(names="class")
    st.dataframe(per_class, use_container_width=True, hide_index=True)
    matrix = metrics["confusion_matrix"]
    st.plotly_chart(
        px.imshow(
            matrix,
            x=CANONICAL_CLASSES,
            y=CANONICAL_CLASSES,
            labels={"x": "Predicted", "y": "Actual", "color": "Count"},
            title="Confusion matrix",
        ),
        use_container_width=True,
    )


def _render_upload_page() -> None:
    st.title("SmartWaste")
    st.write("Upload a waste image to classify it with the approved candidate model.")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        image_bytes = uploaded.getvalue()
        st.image(Image.open(io.BytesIO(image_bytes)), caption=uploaded.name, width=360)
        if st.button("Predict uploaded image", type="primary"):
            _predict_image(image_bytes, uploaded.name)
    _render_feedback()


def _render_camera_page() -> None:
    st.title("Live Camera")
    st.info(
        "Choose a camera with SELECT DEVICE, then start the stream. You can use "
        "your iPhone through macOS Continuity Camera."
    )
    try:
        from av import VideoFrame
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
        from smartwaste.config import get_settings
        from smartwaste.inference import ModelPredictor
    except ImportError:
        st.error(
            "Live camera support is unavailable because streamlit-webrtc is not "
            "installed. Install the project dependencies and restart Streamlit."
        )
        return

    @st.cache_resource
    def get_camera_predictor() -> ModelPredictor:
        settings = get_settings()
        checkpoint = settings.model_checkpoint
        if settings.approved_model_manifest.is_file():
            import json

            manifest = json.loads(settings.approved_model_manifest.read_text(encoding="utf-8"))
            checkpoint = settings.approved_model_manifest.parent / manifest["checkpoint"]
        return ModelPredictor(checkpoint)

    predictor = get_camera_predictor()
    inference_lock = threading.Lock()
    state = {
        "last_inference": 0.0,
        "label": "Place one waste item in view",
        "confidence": 0.0,
    }

    def process_frame(frame: VideoFrame) -> VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        now = time.monotonic()
        if now - state["last_inference"] >= 0.35 and inference_lock.acquire(
            blocking=False
        ):
            try:
                result = predictor.predict_bytes(image.tobytes(), image.shape)
                state["label"] = result.predicted_class.title()
                state["confidence"] = result.probabilities[result.predicted_class]
                state["last_inference"] = now
            finally:
                inference_lock.release()
        image = _overlay_prediction(image, state["label"], state["confidence"])
        return VideoFrame.from_ndarray(image, format="bgr24")

    context = webrtc_streamer(
        key="smartwaste-live-camera",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=process_frame,
        media_stream_constraints={"video": True, "audio": False},
        video_html_attrs={
            "autoPlay": True,
            "controls": False,
            "muted": True,
            "playsInline": True,
        },
        async_processing=True,
    )
    processor = context.video_processor if context and context.video_processor else None
    if processor and processor.error:
        st.error(f"Live inference error: {processor.error}")
    st.caption(
        "Live frames are inferred locally and are not written to SQLite history. "
        "Select the desired camera, allow permission, and start the stream. "
        "The selected camera must be available to your browser."
    )


def _overlay_prediction(image: Any, label: str, confidence: float) -> Any:
    """Render the current live prediction directly onto a video frame."""
    from PIL import ImageDraw, ImageFont

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    text = f"{label}  {confidence:.1%}" if confidence else label
    draw.rectangle((12, 12, 360, 56), fill=(0, 0, 0))
    draw.text((24, 22), text, fill=(255, 255, 255))
    return __import__("numpy").array(pil_image)


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(page_title="SmartWaste", page_icon="♻️", layout="wide")
    page = st.sidebar.radio("Page", ["Classify image", "Live Camera", "History", "Metrics"])
    if page == "Classify image":
        _render_upload_page()
    elif page == "Live Camera":
        _render_camera_page()
    elif page == "History":
        _render_history()
    else:
        _render_metrics()


if __name__ == "__main__":
    main()
