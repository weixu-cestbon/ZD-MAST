from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from zd_mast.provenance import validate_source_manifest


def test_source_manifest_passes_and_detects_changes(tmp_path: Path) -> None:
    artifact = tmp_path / "result.csv"
    artifact.write_text("metric,value\nAUROC,0.8\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"relative_path": artifact.name, "sha256": digest}]).to_csv(manifest, index=False)
    assert validate_source_manifest(tmp_path, manifest).status == "PASS"
    artifact.write_text("metric,value\nAUROC,0.9\n", encoding="utf-8")
    result = validate_source_manifest(tmp_path, manifest)
    assert result.status == "FAIL"
    assert result.mismatch_n == 1
