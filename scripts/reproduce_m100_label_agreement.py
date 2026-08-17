#!/usr/bin/env python3
"""Recompute M100 label agreement from the de-identified public label table."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from zd_mast.harmonization import compare_m100_agreement, summarize_m100_agreement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frozen-agreement", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    feature_dirs = sorted(args.release_root.glob("feature-release-*/"))
    if len(feature_dirs) != 1:
        raise ValueError(f"Expected one feature release directory, found {feature_dirs}")
    labels_path = feature_dirs[0] / "zd_mast_ast_labels_m100_v1.0.0.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    labels = pd.read_parquet(labels_path)
    agreement = summarize_m100_agreement(labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    agreement_path = args.output_dir / "zd_mast_m100_label_agreement_reproduced_v1.0.0.csv"
    agreement.to_csv(agreement_path, index=False)

    regression_status = "NOT_REQUESTED"
    compared_fields = 0
    if args.frozen_agreement:
        frozen = pd.read_csv(args.frozen_agreement)
        regression = compare_m100_agreement(agreement, frozen, args.tolerance)
        regression.to_csv(
            args.output_dir / "zd_mast_m100_label_agreement_regression_v1.0.0.csv",
            index=False,
        )
        compared_fields = len(regression)
        regression_status = "PASS" if regression["status"].eq("PASS").all() else "FAIL"

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_id": "m100_harmonization_label_agreement",
        "release_candidate": args.release_root.resolve().name,
        "input_label_rows": len(labels),
        "task_n": int(labels["task_id"].nunique()),
        "historical_labels_overwritten": False,
        "model_training_performed": False,
        "compared_fields": compared_fields,
        "regression_status": regression_status,
    }
    (args.output_dir / "zd_mast_m100_label_agreement_manifest_v1.0.0.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        f"tasks={manifest['task_n']} labels={manifest['input_label_rows']} "
        f"regression_status={regression_status}",
        flush=True,
    )
    if regression_status == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
