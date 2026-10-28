from pathlib import Path

from smartwaste.config import Settings
from smartwaste.labels import CANONICAL_CLASSES


def test_default_settings_use_expected_project_conventions() -> None:
    settings = Settings()

    assert settings.data_root == Path("data")
    assert settings.max_upload_bytes == 10 * 1024 * 1024
    assert settings.database_url.startswith("sqlite:///")


def test_environment_variable_overrides_environment(monkeypatch) -> None:
    monkeypatch.setenv("SMARTWASTE_ENVIRONMENT", "test")

    assert Settings().environment == "test"


def test_canonical_classes_are_fixed_and_ordered() -> None:
    assert CANONICAL_CLASSES == ("plastic", "paper", "glass", "metal", "organic")
