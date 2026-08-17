"""Checksum-backed provenance validation for manuscript source artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ProvenanceValidation:
    status: str
    artifact_n: int
    missing_n: int
    mismatch_n: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_manifest(root: Path, manifest_path: Path) -> ProvenanceValidation:
    """Verify that every frozen manuscript source exists and retains its checksum."""

    root = root.resolve()
    manifest = pd.read_csv(manifest_path)
    required = {"relative_path", "sha256"}
    missing_columns = required - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Source manifest missing columns: {sorted(missing_columns)}")
    if manifest.empty or manifest["relative_path"].duplicated().any():
        raise ValueError("Source manifest must contain unique artifact paths")

    missing_n = 0
    mismatch_n = 0
    for row in manifest.itertuples(index=False):
        path = root / str(row.relative_path)
        if not path.is_file():
            missing_n += 1
        elif sha256(path) != str(row.sha256):
            mismatch_n += 1
    status = "PASS" if missing_n == 0 and mismatch_n == 0 else "FAIL"
    return ProvenanceValidation(status, len(manifest), missing_n, mismatch_n)
