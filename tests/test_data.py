import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from smartwaste.data import DatasetValidationError, prepare_dataset, validate_dataset
from smartwaste.labels import CANONICAL_CLASSES


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def _create_dataset(root: Path, images_per_class: int = 4) -> None:
    for class_index, label in enumerate(CANONICAL_CLASSES):
        for image_index in range(images_per_class):
            _write_image(
                root / label / f"{image_index}.png",
                (class_index * 40, image_index * 30, 100),
            )


def test_validate_dataset_reports_classes_and_readable_images(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _create_dataset(raw_dir)

    records, report = validate_dataset(raw_dir)

    assert len(records) == 20
    assert report.is_valid
    assert report.class_counts == dict.fromkeys(CANONICAL_CLASSES, 4)


def test_duplicate_images_are_rejected_before_split(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _create_dataset(raw_dir)
    duplicate = raw_dir / "plastic" / "0.png"
    (raw_dir / "paper" / "duplicate.png").write_bytes(duplicate.read_bytes())

    _, report = validate_dataset(raw_dir)

    assert not report.is_valid
    assert len(report.duplicate_groups) == 1


def test_prepare_dataset_is_reproducible_and_writes_manifest(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    _create_dataset(raw_dir, images_per_class=10)

    prepare_dataset(raw_dir, output_dir, validation_fraction=0.2, test_fraction=0.2, seed=7)
    first_manifest = (output_dir / "manifest.csv").read_text(encoding="utf-8")
    report = json.loads((output_dir / "validation_report.json").read_text(encoding="utf-8"))

    prepare_dataset(raw_dir, output_dir, validation_fraction=0.2, test_fraction=0.2, seed=7)
    second_manifest = (output_dir / "manifest.csv").read_text(encoding="utf-8")

    assert first_manifest == second_manifest
    assert report["image_count"] == 50
    with (output_dir / "manifest.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert {row["split"] for row in rows} == {"train", "validation", "test"}


def test_small_classes_fail_with_actionable_error(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _create_dataset(raw_dir, images_per_class=2)
    records, _ = validate_dataset(raw_dir)

    with pytest.raises(DatasetValidationError, match="at least 3"):
        from smartwaste.data import split_records

        split_records(records, validation_fraction=0.2, test_fraction=0.2, seed=42)
