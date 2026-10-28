"""Dataset validation, duplicate detection, and reproducible preparation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from smartwaste.labels import CANONICAL_CLASSES

SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
REALWASTE_CLASS_MAPPING = {
    "Cardboard": "paper",
    "Food Organics": "organic",
    "Glass": "glass",
    "Metal": "metal",
    "Paper": "paper",
    "Plastic": "plastic",
    "Vegetation": "organic",
}
REALWASTE_EXCLUDED_CLASSES = frozenset({"Miscellaneous Trash", "Textile Trash"})
FINAL_MERGED_CLASS_MAPPING = {
    "cardboard": "paper",
    "glass": "glass",
    "metal": "metal",
    "organic": "organic",
    "paper": "paper",
    "plastic": "plastic",
}
FINAL_MERGED_EXCLUDED_CLASSES = frozenset({"battery", "textile", "trash"})


@dataclass(frozen=True)
class DatasetRecord:
    """A validated image and its canonical class."""

    source_path: Path
    relative_path: str
    label: str
    sha256: str
    width: int
    height: int


@dataclass(frozen=True)
class ValidationReport:
    """Serializable summary of dataset validation."""

    image_count: int
    class_counts: dict[str, int]
    duplicate_groups: list[list[str]]
    invalid_files: list[dict[str, str]]
    missing_classes: list[str]
    duplicate_groups_excluded: int = 0

    @property
    def is_valid(self) -> bool:
        """Return whether the dataset can safely proceed to splitting."""
        return (
            self.image_count > 0
            and not self.missing_classes
            and not self.duplicate_groups
            and not self.invalid_files
        )


class DatasetValidationError(ValueError):
    """Raised when the dataset cannot be used for preparation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_candidate_files(raw_dir: Path) -> Iterable[tuple[Path, str]]:
    for label in CANONICAL_CLASSES:
        class_dir = raw_dir / label
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path, label


def validate_dataset(raw_dir: Path) -> tuple[list[DatasetRecord], ValidationReport]:
    """Validate image files and return records plus a machine-readable report."""
    raw_dir = raw_dir.resolve()
    if not raw_dir.is_dir():
        raise DatasetValidationError(f"Dataset directory does not exist: {raw_dir}")

    missing_classes = [
        label for label in CANONICAL_CLASSES if not (raw_dir / label).is_dir()
    ]
    records: list[DatasetRecord] = []
    invalid_files: list[dict[str, str]] = []
    hashes: defaultdict[str, list[str]] = defaultdict(list)

    for path, label in _iter_candidate_files(raw_dir):
        relative_path = path.relative_to(raw_dir).as_posix()
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
            if width < 1 or height < 1:
                raise ValueError("image has invalid dimensions")
            file_hash = _sha256(path)
        except (OSError, UnidentifiedImageError, ValueError) as error:
            invalid_files.append({"path": relative_path, "error": str(error)})
            continue

        record = DatasetRecord(
            source_path=path,
            relative_path=relative_path,
            label=label,
            sha256=file_hash,
            width=width,
            height=height,
        )
        records.append(record)
        hashes[file_hash].append(relative_path)

    duplicate_groups = sorted(
        sorted(paths) for paths in hashes.values() if len(paths) > 1
    )
    class_counts = dict(
        sorted(Counter(record.label for record in records).items())
    )
    for label in CANONICAL_CLASSES:
        class_counts.setdefault(label, 0)

    missing_classes = sorted(
        set(missing_classes)
        | {label for label in CANONICAL_CLASSES if class_counts[label] == 0}
    )
    report = ValidationReport(
        image_count=len(records),
        class_counts=class_counts,
        duplicate_groups=duplicate_groups,
        invalid_files=invalid_files,
        missing_classes=missing_classes,
    )
    return records, report


def validate_mapped_dataset(
    source_dir: Path,
    class_mapping: dict[str, str],
    *,
    deduplicate: bool = False,
) -> tuple[list[DatasetRecord], ValidationReport]:
    """Validate a source dataset after applying an explicit class mapping."""
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise DatasetValidationError(f"Dataset directory does not exist: {source_dir}")
    if set(class_mapping.values()) != set(CANONICAL_CLASSES):
        raise DatasetValidationError(
            "The mapping must produce exactly the canonical SmartWaste classes."
        )

    records: list[DatasetRecord] = []
    invalid_files: list[dict[str, str]] = []
    hashes: defaultdict[str, list[str]] = defaultdict(list)
    for source_label, target_label in sorted(class_mapping.items()):
        class_dir = source_dir / source_label
        if not class_dir.is_dir():
            raise DatasetValidationError(
                f"Mapped source class directory does not exist: {class_dir}"
            )
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            relative_path = path.relative_to(source_dir).as_posix()
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                if width < 1 or height < 1:
                    raise ValueError("image has invalid dimensions")
                file_hash = _sha256(path)
            except (OSError, UnidentifiedImageError, ValueError) as error:
                invalid_files.append({"path": relative_path, "error": str(error)})
                continue
            record = DatasetRecord(
                source_path=path,
                relative_path=relative_path,
                label=target_label,
                sha256=file_hash,
                width=width,
                height=height,
            )
            records.append(record)
            hashes[file_hash].append(relative_path)

    duplicate_groups = sorted(sorted(paths) for paths in hashes.values() if len(paths) > 1)
    duplicate_groups_excluded = 0
    if deduplicate and duplicate_groups:
        retained_hashes = {file_hash for file_hash, paths in hashes.items() if paths}
        retained_paths = {
            paths[0] for file_hash, paths in hashes.items() if file_hash in retained_hashes
        }
        duplicate_groups_excluded = len(duplicate_groups)
        records = [record for record in records if record.relative_path in retained_paths]
        duplicate_groups = []
    class_counts = {label: 0 for label in CANONICAL_CLASSES}
    class_counts.update(Counter(record.label for record in records))
    missing_classes = sorted(label for label, count in class_counts.items() if count == 0)
    report = ValidationReport(
        image_count=len(records),
        class_counts=class_counts,
        duplicate_groups=duplicate_groups,
        invalid_files=invalid_files,
        missing_classes=missing_classes,
        duplicate_groups_excluded=duplicate_groups_excluded,
    )
    return records, report


def split_records(
    records: list[DatasetRecord],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[DatasetRecord]]:
    """Create reproducible stratified train, validation, and test partitions."""
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise DatasetValidationError("Split fractions must be between 0 and 1.")
    if validation_fraction + test_fraction >= 1:
        raise DatasetValidationError("Validation and test fractions must sum to less than 1.")

    counts = Counter(record.label for record in records)
    insufficient = sorted(label for label in CANONICAL_CLASSES if counts[label] < 3)
    if insufficient:
        raise DatasetValidationError(
            "Each class needs at least 3 valid images for stratified splitting; "
            f"insufficient classes: {', '.join(insufficient)}"
        )

    try:
        train_and_validation, test = train_test_split(
            records,
            test_size=test_fraction,
            random_state=seed,
            stratify=[record.label for record in records],
        )
        validation_relative_fraction = validation_fraction / (1 - test_fraction)
        train, validation = train_test_split(
            train_and_validation,
            test_size=validation_relative_fraction,
            random_state=seed,
            stratify=[record.label for record in train_and_validation],
        )
    except ValueError as error:
        raise DatasetValidationError(
            "The requested split fractions cannot produce a stratified split "
            f"for this dataset: {error}"
        ) from error
    return {"train": sorted(train, key=lambda item: item.relative_path),
            "validation": sorted(validation, key=lambda item: item.relative_path),
            "test": sorted(test, key=lambda item: item.relative_path)}


def prepare_dataset(
    raw_dir: Path,
    output_dir: Path,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> ValidationReport:
    """Validate, split, and copy a dataset into deterministic split folders."""
    records, report = validate_dataset(raw_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_report.json").write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not report.is_valid:
        raise DatasetValidationError(
            "Dataset validation failed; see "
            f"{output_dir / 'validation_report.json'} for details."
        )

    for split_name in ("train", "validation", "test"):
        split_dir = output_dir / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)

    splits = split_records(
        records,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    manifest_rows: list[dict[str, str | int]] = []
    for split_name, split_records_for_name in splits.items():
        for record in split_records_for_name:
            destination_name = f"{record.sha256[:12]}_{Path(record.relative_path).name}"
            destination = output_dir / split_name / record.label / destination_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record.source_path, destination)
            manifest_rows.append(
                {
                    "split": split_name,
                    "label": record.label,
                    "path": destination.relative_to(output_dir).as_posix(),
                    "source_path": record.relative_path,
                    "sha256": record.sha256,
                    "width": record.width,
                    "height": record.height,
                }
            )

    manifest_rows.sort(key=lambda row: (str(row["split"]), str(row["path"])))
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return report


def prepare_mapped_dataset(
    source_dir: Path,
    output_dir: Path,
    *,
    class_mapping: dict[str, str],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    deduplicate: bool = False,
) -> ValidationReport:
    """Prepare a mapped source dataset using the standard split and artifacts."""
    records, report = validate_mapped_dataset(
        source_dir, class_mapping, deduplicate=deduplicate
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_payload = asdict(report) | {
        "source_directory": str(source_dir.resolve()),
        "class_mapping": class_mapping,
        "excluded_source_classes": sorted(
            path.name for path in source_dir.iterdir()
            if path.is_dir() and path.name not in class_mapping
        ),
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not report.is_valid:
        raise DatasetValidationError(
            "Mapped dataset validation failed; see "
            f"{output_dir / 'validation_report.json'} for details."
        )
    for split_name in ("train", "validation", "test"):
        split_dir = output_dir / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
    splits = split_records(
        records,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    manifest_rows: list[dict[str, str | int]] = []
    for split_name, split_records_for_name in splits.items():
        for record in split_records_for_name:
            destination_name = f"{record.sha256[:12]}_{Path(record.relative_path).name}"
            destination = output_dir / split_name / record.label / destination_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record.source_path, destination)
            manifest_rows.append(
                {
                    "split": split_name,
                    "label": record.label,
                    "path": destination.relative_to(output_dir).as_posix(),
                    "source_path": record.relative_path,
                    "source_label": Path(record.relative_path).parts[0],
                    "sha256": record.sha256,
                    "width": record.width,
                    "height": record.height,
                }
            )
    manifest_rows.sort(key=lambda row: (str(row["split"]), str(row["path"])))
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and prepare SmartWaste images.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="RealWaste-style source directory with original class names.",
    )
    parser.add_argument(
        "--source-format",
        choices=("realwaste", "final-merged"),
        default="realwaste",
        help="Mapping to apply when --source-dir is supplied.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.source_dir:
            mapping = (
                FINAL_MERGED_CLASS_MAPPING
                if args.source_format == "final-merged"
                else REALWASTE_CLASS_MAPPING
            )
            report = prepare_mapped_dataset(
                args.source_dir,
                args.output_dir,
                class_mapping=mapping,
                validation_fraction=args.validation_fraction,
                test_fraction=args.test_fraction,
                seed=args.seed,
                deduplicate=args.source_format == "final-merged",
            )
        else:
            report = prepare_dataset(
                args.raw_dir,
                args.output_dir,
                validation_fraction=args.validation_fraction,
                test_fraction=args.test_fraction,
                seed=args.seed,
            )
    except DatasetValidationError as error:
        print(f"Dataset preparation failed: {error}")
        return 1
    print(f"Prepared {report.image_count} validated images in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
