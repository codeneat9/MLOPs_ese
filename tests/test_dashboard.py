from smartwaste.dashboard import DEFAULT_API_URL, _api_url, _overlay_prediction


def test_dashboard_uses_local_api_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SMARTWASTE_API_URL", raising=False)

    assert _api_url() == DEFAULT_API_URL


def test_dashboard_strips_api_url_trailing_slash(monkeypatch) -> None:
    monkeypatch.setenv("SMARTWASTE_API_URL", "http://example.test///")

    assert _api_url() == "http://example.test"


def test_overlay_prediction_preserves_frame_shape() -> None:
    import numpy as np

    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    result = _overlay_prediction(frame, "Plastic", 0.95)

    assert result.shape == frame.shape
