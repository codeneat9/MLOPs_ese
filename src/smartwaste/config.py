"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings shared by training, serving, and monitoring."""

    model_config = SettingsConfigDict(
        env_prefix="SMARTWASTE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "development"
    log_level: str = "INFO"
    data_root: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
    model_dir: Path = Path("models")
    model_checkpoint: Path = Path("models/candidate/best.pt")
    approved_model_dir: Path = Path("models/approved")
    approved_model_manifest: Path = Path("models/approved/manifest.json")
    evaluation_metrics_path: Path = Path("artifacts/evaluation/metrics.json")
    database_url: str = "sqlite:///./smartwaste.db"
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment: str = "smartwaste-classification"
    max_upload_bytes: int = 10 * 1024 * 1024

    def create_runtime_directories(self) -> None:
        """Create directories used for generated local state."""
        for directory in (
            self.data_root,
            self.raw_data_dir,
            self.processed_data_dir,
            self.artifacts_dir,
            self.model_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide validated settings instance."""
    return Settings()
