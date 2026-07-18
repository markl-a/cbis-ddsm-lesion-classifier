# -*- coding: utf-8 -*-
"""CBIS-DDSM metadata, leakage-safe splitting, and image loading.

The awsaf49 Kaggle repack stores JPEG files as
``jpeg/<SeriesInstanceUID>/<generated-name>.jpg`` while the case-description
CSVs reference the original DICOM hierarchy.  The only reliable bridge is the
SeriesInstanceUID plus ``dicom_info.csv`` and its ``SeriesDescription``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

LABEL_MAP = {
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT": 1,
}
CLASS_NAMES = ["benign", "malignant"]

DESC_CSVS = {
    ("mass", "train"): "mass_case_description_train_set.csv",
    ("mass", "test"): "mass_case_description_test_set.csv",
    ("calc", "train"): "calc_case_description_train_set.csv",
    ("calc", "test"): "calc_case_description_test_set.csv",
}


def _series_uid_from_dcm_path(value: Any) -> str | None:
    """Extract the SeriesInstanceUID (the path component before the file)."""
    if not isinstance(value, str) or not value.strip():
        return None
    parts = value.strip().replace("\\", "/").rstrip("/").split("/")
    return parts[-2] if len(parts) >= 2 else None


def _canonical_id(value: Any) -> str:
    """Render CSV numeric identifiers without an accidental ``.0`` suffix."""
    if pd.isna(value):
        return "unknown"
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _resolve_dirs(data_dir: str | os.PathLike) -> tuple[Path, Path, Path]:
    """Resolve common flat, Kaggle, and CBIS-DDSM nested extraction layouts."""
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    csv_candidates = (root / "csv", root, root / "CBIS-DDSM" / "csv")
    csv_dir = next(
        (candidate for candidate in csv_candidates if (candidate / "dicom_info.csv").is_file()),
        None,
    )
    if csv_dir is None:
        raise FileNotFoundError(
            f"Could not find dicom_info.csv under supported layouts in {root}"
        )

    jpeg_candidates = (root / "jpeg", root / "CBIS-DDSM" / "jpeg")
    jpeg_dir = next((candidate for candidate in jpeg_candidates if candidate.is_dir()), None)
    if jpeg_dir is None:
        # Metadata-only preparation remains possible with require_files=False.
        jpeg_dir = root / "jpeg"
    return root, csv_dir, jpeg_dir


def _local_jpeg_path(image_path: str, jpeg_dir: Path) -> Path:
    normalized = str(image_path).replace("\\", "/")
    marker = "jpeg/"
    marker_index = normalized.lower().find(marker)
    if marker_index >= 0:
        normalized = normalized[marker_index + len(marker) :]
    return jpeg_dir.joinpath(*Path(normalized).parts)


def _case_id(abnormality: str, row: pd.Series) -> str:
    return "|".join(
        [
            abnormality,
            str(row.get("patient_id", "unknown")).strip(),
            str(row.get("left or right breast", "unknown")).strip().upper(),
            _canonical_id(row.get("abnormality id")),
        ]
    )


def build_manifest(
    data_dir: str | os.PathLike,
    use: str = "cropped",
    abnormality: str = "both",
    require_files: bool = True,
    min_mapping_coverage: float = 0.995,
    verbose: bool = True,
) -> pd.DataFrame:
    """Join case-description rows to the intended JPEG images.

    The returned frame retains patient, side, view, abnormality, official split,
    and case identifiers required for group-disjoint splitting and case-level
    evaluation.  Audit details are available in ``manifest.attrs``.
    """
    if use not in {"cropped", "full"}:
        raise ValueError("use must be 'cropped' or 'full'")
    if abnormality not in {"both", "mass", "calc"}:
        raise ValueError("abnormality must be 'both', 'mass', or 'calc'")
    if not 0 <= min_mapping_coverage <= 1:
        raise ValueError("min_mapping_coverage must be between 0 and 1")

    _, csv_dir, jpeg_dir = _resolve_dirs(data_dir)
    dicom = pd.read_csv(csv_dir / "dicom_info.csv", low_memory=False)
    required_dicom_columns = {"SeriesInstanceUID", "SeriesDescription", "image_path"}
    missing_columns = required_dicom_columns.difference(dicom.columns)
    if missing_columns:
        raise ValueError(f"dicom_info.csv missing columns: {sorted(missing_columns)}")

    wanted_description = "cropped images" if use == "cropped" else "full mammogram images"
    descriptions = dicom["SeriesDescription"].fillna("").astype(str).str.strip().str.casefold()
    selected = dicom.loc[descriptions.eq(wanted_description)].copy()
    selected = selected.dropna(subset=["SeriesInstanceUID", "image_path"])

    ambiguous_uids = (
        selected.groupby("SeriesInstanceUID")["image_path"].nunique().loc[lambda values: values > 1]
    )
    if not ambiguous_uids.empty:
        raise ValueError(
            f"Found {len(ambiguous_uids)} {wanted_description!r} series with multiple JPEG paths"
        )
    uid_to_path = selected.set_index("SeriesInstanceUID")["image_path"].to_dict()

    path_column = "cropped image file path" if use == "cropped" else "image file path"
    records: list[dict[str, Any]] = []
    missing_csvs: list[str] = []
    for (lesion_type, official_split), filename in DESC_CSVS.items():
        if abnormality != "both" and lesion_type != abnormality:
            continue
        csv_path = csv_dir / filename
        if not csv_path.is_file():
            missing_csvs.append(filename)
            continue

        cases = pd.read_csv(csv_path)
        required_case_columns = {
            "patient_id",
            "left or right breast",
            "image view",
            "abnormality id",
            "pathology",
            path_column,
        }
        absent = required_case_columns.difference(cases.columns)
        if absent:
            raise ValueError(f"{filename} missing columns: {sorted(absent)}")

        for source_row, row in cases.iterrows():
            pathology = str(row["pathology"]).strip().upper()
            if pathology not in LABEL_MAP:
                continue
            series_uid = _series_uid_from_dcm_path(row[path_column])
            records.append(
                {
                    "patient_id": str(row["patient_id"]).strip(),
                    "abnormality": lesion_type,
                    "official_split": official_split,
                    # Backward-compatible alias for the earlier local scripts.
                    "split": official_split,
                    "pathology": pathology,
                    "label": LABEL_MAP[pathology],
                    "breast_side": str(row["left or right breast"]).strip().upper(),
                    "image_view": str(row["image view"]).strip().upper(),
                    "abnormality_id": _canonical_id(row["abnormality id"]),
                    "case_id": _case_id(lesion_type, row),
                    "series_uid": series_uid,
                    "source_csv": filename,
                    "source_row": int(source_row),
                }
            )

    manifest = pd.DataFrame.from_records(records)
    if manifest.empty:
        raise RuntimeError(f"No labelled case rows were found in {csv_dir}")

    manifest["image_path"] = manifest["series_uid"].map(uid_to_path)
    mapped_mask = manifest["image_path"].notna()
    mapping_coverage = float(mapped_mask.mean())
    unmatched_uids = sorted(
        manifest.loc[~mapped_mask, "series_uid"].dropna().astype(str).unique().tolist()
    )
    if mapping_coverage < min_mapping_coverage:
        raise RuntimeError(
            f"Only {mapping_coverage:.2%} of description rows mapped to "
            f"{wanted_description!r}; required {min_mapping_coverage:.2%}"
        )

    manifest = manifest.loc[mapped_mask].copy()
    manifest["filepath"] = manifest["image_path"].map(
        lambda value: str(_local_jpeg_path(value, jpeg_dir))
    )
    exists_mask = manifest["filepath"].map(os.path.isfile)
    file_coverage = float(exists_mask.mean()) if len(manifest) else 0.0
    missing_filepaths = manifest.loc[~exists_mask, "filepath"].astype(str).tolist()
    if require_files:
        manifest = manifest.loc[exists_mask].copy()

    manifest = manifest.reset_index(drop=True)
    manifest.insert(0, "manifest_index", np.arange(len(manifest), dtype=np.int64))
    audit = {
        "description_rows": len(records),
        "mapped_rows": int(mapped_mask.sum()),
        "retained_rows": len(manifest),
        "mapping_coverage": mapping_coverage,
        "file_coverage": file_coverage,
        "missing_csvs": missing_csvs,
        "unmatched_series_uids": unmatched_uids,
        "missing_filepaths": missing_filepaths,
        "wanted_series_description": wanted_description,
        "csv_dir": str(csv_dir),
        "jpeg_dir": str(jpeg_dir),
    }
    manifest.attrs.update(audit)

    if verbose:
        print(
            f"Description rows: {audit['description_rows']:,} | "
            f"UID mapped: {audit['mapped_rows']:,} ({mapping_coverage:.2%}) | "
            f"retained: {audit['retained_rows']:,}"
        )
        print(manifest.groupby(["official_split", "pathology"]).size().to_string())
        if unmatched_uids:
            print(f"Unmatched series UIDs: {len(unmatched_uids)}")
        if require_files and missing_filepaths:
            print(f"Missing JPEG files: {len(missing_filepaths)}")
    return manifest


def make_patient_splits(
    manifest: pd.DataFrame,
    n_splits: int = 5,
    fold: int = 0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create patient-disjoint train/validation and untouched official test.

    Because mass and calcification were split independently by the dataset
    authors, the union contains cross-task patient overlap.  Every official-test
    patient is purged from the combined training pool before grouped CV.
    """
    required = {"patient_id", "label", "official_split"}
    absent = required.difference(manifest.columns)
    if absent:
        raise ValueError(f"manifest missing split columns: {sorted(absent)}")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if not 0 <= fold < n_splits:
        raise ValueError(f"fold must be in [0, {n_splits - 1}]")

    official_train = manifest.loc[manifest["official_split"].eq("train")].copy()
    official_test = manifest.loc[manifest["official_split"].eq("test")].copy()
    if official_train.empty or official_test.empty:
        raise ValueError("manifest must contain non-empty official train and test rows")

    test_patients = set(official_test["patient_id"].astype(str))
    train_patients_before = set(official_train["patient_id"].astype(str))
    overlap = sorted(train_patients_before.intersection(test_patients))
    train_pool = official_train.loc[~official_train["patient_id"].isin(test_patients)].copy()

    if train_pool["patient_id"].nunique() < n_splits:
        raise ValueError("not enough distinct training patients for grouped folds")
    if train_pool["label"].nunique() != 2 or official_test["label"].nunique() != 2:
        raise ValueError("training pool and official test must each contain both labels")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    folds = list(
        splitter.split(
            X=np.zeros(len(train_pool), dtype=np.uint8),
            y=train_pool["label"].to_numpy(),
            groups=train_pool["patient_id"].to_numpy(),
        )
    )
    train_positions, val_positions = folds[fold]
    train = train_pool.iloc[train_positions].copy()
    val = train_pool.iloc[val_positions].copy()
    test = official_test.copy()

    split_frames = (("train", train), ("validation", val), ("test", test))
    for split_name, frame in split_frames:
        if frame.empty or frame["label"].nunique() != 2:
            raise RuntimeError(f"{split_name} split must be non-empty and contain both labels")
        frame["assigned_split"] = split_name

    patient_sets = {
        "train": set(train["patient_id"].astype(str)),
        "validation": set(val["patient_id"].astype(str)),
        "test": set(test["patient_id"].astype(str)),
    }
    if not (
        patient_sets["train"].isdisjoint(patient_sets["validation"])
        and patient_sets["train"].isdisjoint(patient_sets["test"])
        and patient_sets["validation"].isdisjoint(patient_sets["test"])
    ):
        raise RuntimeError("patient leakage detected after grouped splitting")

    audit = {
        "seed": seed,
        "fold": fold,
        "n_splits": n_splits,
        "purged_train_patients": overlap,
        "purged_train_rows": int(official_train["patient_id"].isin(test_patients).sum()),
        "rows": {name: len(frame) for name, frame in split_frames},
        "patients": {name: len(values) for name, values in patient_sets.items()},
        "positive_rate": {
            name: float(frame["label"].mean()) for name, frame in split_frames
        },
    }
    return train, val, test, audit


def manifest_fingerprint(*frames: pd.DataFrame) -> str:
    """Stable SHA-256 over split membership and labels for checkpoint auditing."""
    columns = [
        column
        for column in ("assigned_split", "patient_id", "case_id", "series_uid", "label")
        if all(column in frame.columns for frame in frames)
    ]
    if not columns:
        raise ValueError("frames do not share fingerprint columns")
    combined = pd.concat([frame[columns] for frame in frames], ignore_index=True)
    combined = combined.sort_values(columns, kind="stable").reset_index(drop=True)
    payload = combined.to_json(orient="records", date_format="iso").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class CBISDataset(Dataset):
    """Load grayscale mammography JPEGs as three-channel model inputs."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        transform=None,
        return_index: bool = False,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True).copy()
        self.paths = self.manifest["filepath"].tolist()
        self.labels = self.manifest["label"].astype(int).tolist()
        self.transform = transform
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[index]
        return (image, label, index) if self.return_index else (image, label)


class SquarePad:
    """Pad a PIL image to square without distorting lesion geometry."""

    def __init__(self, fill: int = 0) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        right = side - width - left
        bottom = side - height - top
        return ImageOps.expand(image, border=(left, top, right, bottom), fill=self.fill)


def get_transforms(train: bool = True, img_size: int = 224):
    """ImageNet-compatible preprocessing with conservative mammography augments."""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    operations: list[Any] = [SquarePad()]
    if train:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=7,
                    translate=(0.03, 0.03),
                    scale=(0.95, 1.05),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                ),
                transforms.ColorJitter(brightness=0.08, contrast=0.12),
            ]
        )
    operations.extend(
        [
            transforms.Resize(
                (img_size, img_size), interpolation=InterpolationMode.BICUBIC, antialias=True
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(operations)


def split_audit_json(audit: dict[str, Any]) -> str:
    """Pretty, deterministic JSON for notebook display and artifact metadata."""
    return json.dumps(audit, indent=2, sort_keys=True)
