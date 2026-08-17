#!/usr/bin/env python3
"""Validate the release-candidate schema without accessing private source data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PRIVATE_TOKENS = ("private", "patient_id", "sample_id_private", "event_id", "absolute_path")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.release_root
    feature_candidates = sorted(root.glob("feature-release-*"))
    feature = feature_candidates[0] if feature_candidates else root / "feature-release"
    required = [
        feature / "zd_mast_a_sample_level_intensity6000_v1.0.0.npy",
        feature / "zd_mast_b_sample_level_intensity6000_v1.0.0.npy",
        feature / "zd_mast_a_sample_level_peak_presence6000_v1.0.0.npy",
        feature / "zd_mast_b_sample_level_peak_presence6000_v1.0.0.npy",
        feature / "zd_mast_ast_labels_historical_v1.0.0.parquet",
        feature / "zd_mast_feature_axis_6000_v1.0.0.csv",
        feature / "zd_mast_split_assignments_public_v1.0.0.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("missing release files: " + ", ".join(missing))
    for site in ("a", "b"):
        x = np.load(feature / f"zd_mast_{site}_sample_level_intensity6000_v1.0.0.npy", mmap_mode="r")
        p = np.load(feature / f"zd_mast_{site}_sample_level_peak_presence6000_v1.0.0.npy", mmap_mode="r")
        if x.ndim != 2 or x.shape[1] != 6000:
            raise ValueError(f"invalid intensity shape for site {site}: {x.shape}")
        if p.shape != x.shape or p.dtype != np.uint8:
            raise ValueError(f"invalid presence shape/dtype for site {site}: {p.shape} {p.dtype}")
        if not np.isfinite(x).all():
            raise ValueError(f"non-finite intensity values for site {site}")
    labels = pd.read_parquet(feature / "zd_mast_ast_labels_historical_v1.0.0.parquet")
    forbidden_columns = [c for c in labels.columns if any(token in c.lower() for token in PRIVATE_TOKENS)]
    if forbidden_columns:
        raise ValueError(f"private-looking label columns: {forbidden_columns}")
    required_columns = {"site_id", "public_sample_id", "task_id", "historical_sir", "binary_s_vs_ir", "year"}
    if not required_columns.issubset(labels.columns):
        raise ValueError(f"missing label columns: {sorted(required_columns - set(labels.columns))}")
    splits = pd.read_csv(feature / "zd_mast_split_assignments_public_v1.0.0.csv")
    required_split_columns = {"analysis_id", "protocol", "site_id", "task_id", "public_sample_id", "split"}
    if not required_split_columns.issubset(splits.columns):
        raise ValueError(f"missing split columns: {sorted(required_split_columns - set(splits.columns))}")
    if splits.empty or splits[["task_id", "public_sample_id", "protocol"]].duplicated().any():
        raise ValueError("invalid or duplicate public split assignments")
    if not set(splits["split"].astype(str)).issubset({"train", "validation", "test", "early_70", "late_30"}):
        raise ValueError("unknown public split label")
    allowed_prefixes = ("ZDMA_SMP_", "ZDMB_SMP_")
    if (~labels["public_sample_id"].astype(str).str.startswith(allowed_prefixes)).any():
        raise ValueError("sample IDs contain a non-release public identifier")
    print("public release validation: PASS")
    print(f"labels={len(labels)} sites={labels.site_id.nunique()} tasks={labels.task_id.nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
