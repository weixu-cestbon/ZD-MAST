"""Validation for a de-identified ZD-MAST feature release."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReleaseValidation:
    status: str
    sample_rows: int
    spectrum_rows: int
    label_rows: int
    task_n: int
    site_n: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _find_feature_directory(root: Path) -> Path:
    candidates = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("feature-release"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one feature-release directory, found {len(candidates)}")
    return candidates[0]


def validate_release(root: Path) -> ReleaseValidation:
    """Validate public identifiers, table keys, and matrix dimensions."""

    feature = _find_feature_directory(root)
    sample = pd.read_csv(feature / "zd_mast_sample_metadata_public_v1.0.0.csv")
    spectrum = pd.read_csv(feature / "zd_mast_spectrum_metadata_public_v1.0.0.csv")
    labels = pd.read_parquet(feature / "zd_mast_ast_labels_historical_v1.0.0.parquet")
    tasks = pd.read_csv(feature / "zd_mast_task_dictionary_v1.0.0.csv")
    splits = pd.read_csv(feature / "zd_mast_split_assignments_public_v1.0.0.csv")

    required_sample = {"site_id", "public_sample_id", "feature_row", "feature_schema"}
    required_label = {"site_id", "public_sample_id", "task_id", "historical_sir", "binary_s_vs_ir"}
    required_split = {"analysis_id", "protocol", "site_id", "task_id", "public_sample_id", "split"}
    for name, frame, required in (
        ("sample metadata", sample, required_sample),
        ("labels", labels, required_label),
        ("splits", splits, required_split),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    if sample["public_sample_id"].duplicated().any():
        raise ValueError("Duplicate public sample identifiers")
    if labels.duplicated(["site_id", "public_sample_id", "task_id"]).any():
        raise ValueError("Duplicate sample-task labels")
    if splits.duplicated(["analysis_id", "protocol", "site_id", "task_id", "public_sample_id"]).any():
        raise ValueError("Duplicate split assignments")
    if not labels["historical_sir"].isin(["S", "I", "R"]).all():
        raise ValueError("Unsupported historical S/I/R value")
    if not labels["binary_s_vs_ir"].dropna().isin([0, 1]).all():
        raise ValueError("Invalid binary label")
    if not set(labels["public_sample_id"]).issubset(set(sample["public_sample_id"])):
        raise ValueError("Label table references unknown public sample identifiers")
    if not set(splits["public_sample_id"]).issubset(set(sample["public_sample_id"])):
        raise ValueError("Split table references unknown public sample identifiers")
    if not set(labels["task_id"]).issubset(set(tasks["task_id"])):
        raise ValueError("Label table references unknown tasks")

    for site_id, group in sample.groupby("site_id"):
        token = "a" if site_id == "ZD-MAST-A" else "b"
        matrix_path = feature / f"zd_mast_{token}_sample_level_intensity6000_v1.0.0.npy"
        matrix = np.load(matrix_path, mmap_mode="r")
        if matrix.shape != (len(group), 6000):
            raise ValueError(f"{site_id} sample matrix shape {matrix.shape} does not match metadata")

    return ReleaseValidation(
        status="PASS",
        sample_rows=len(sample),
        spectrum_rows=len(spectrum),
        label_rows=len(labels),
        task_n=labels["task_id"].nunique(),
        site_n=labels["site_id"].nunique(),
    )

