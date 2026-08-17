#!/usr/bin/env python3
"""Compare Site A training histories on one frozen 2026 test cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from zd_mast.cross_platform import TASK_IDS
from zd_mast.training_history import (
    ANALYSIS_ID,
    DEFAULT_SEED,
    build_history_cohorts,
    load_training_history_inputs,
    paired_history_deltas,
    run_history_task,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--temporal-bridge", required=True, type=Path)
    parser.add_argument("--primary-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--write-predictions", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output}")
    output.mkdir(parents=True)
    inputs = load_training_history_inputs(
        args.release_root.resolve(),
        args.temporal_bridge.resolve(),
        args.primary_metrics.resolve(),
    )
    result_parts = []
    bootstrap_parts = []
    prediction_parts = []
    fold_parts = []
    cohort_rows = []
    for index, task_id in enumerate(TASK_IDS, start=1):
        print(f"[{index}/{len(TASK_IDS)}] {task_id}", flush=True)
        cohorts = build_history_cohorts(inputs, task_id)
        fold_parts.append(cohorts.fold_audit)
        cohort_rows.append(
            {
                "task_id": task_id,
                **{
                    f"{regime}_n": len(frame)
                    for regime, frame in cohorts.development_by_regime.items()
                },
                "test_all_samples_n": len(cohorts.test_all_samples),
                "test_patient_disjoint_common_n": len(cohorts.test_patient_disjoint_common),
                "test_removed_patient_overlap_n": cohorts.test_purge_audit[
                    "removed_patient_overlap_n"
                ],
                "test_removed_missing_patient_cluster_n": cohorts.test_purge_audit[
                    "removed_missing_patient_cluster_n"
                ],
            }
        )
        results, bootstrap, predictions = run_history_task(
            inputs,
            cohorts,
            threads=args.threads,
            bootstrap_count=args.bootstrap_count,
            seed=args.seed,
        )
        result_parts.append(results)
        bootstrap_parts.append(bootstrap)
        prediction_parts.append(predictions)
    results = pd.concat(result_parts, ignore_index=True)
    bootstraps = pd.concat(bootstrap_parts, ignore_index=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    deltas = paired_history_deltas(
        predictions,
        bootstrap_count=args.bootstrap_count,
        seed=args.seed,
    )
    results.to_csv(output / "zd_mast_same_test_training_history_metrics_v1.csv", index=False)
    pd.DataFrame(cohort_rows).to_csv(
        output / "zd_mast_same_test_training_history_cohort_counts_v1.csv", index=False
    )
    pd.concat(fold_parts, ignore_index=True).to_csv(
        output / "zd_mast_same_test_training_history_fold_audit_v1.csv", index=False
    )
    bootstraps.to_csv(
        output / "zd_mast_same_test_training_history_bootstrap_intervals_v1.csv", index=False
    )
    deltas.to_csv(output / "zd_mast_same_test_training_history_paired_deltas_v1.csv", index=False)
    if args.write_predictions:
        predictions.to_parquet(
            output / "zd_mast_same_test_training_history_predictions_private_review_v1.parquet",
            index=False,
        )
    primary = results.loc[
        results["analysis_variant"].eq("patient_disjoint_common_test_primary")
    ]
    public_labels = pd.read_parquet(
        inputs.feature_root / "zd_mast_ast_labels_historical_v1.0.0.parquet"
    )
    public_label_n = int(
        (
            public_labels["site_id"].eq("ZD-MAST-A")
            & public_labels["task_id"].isin(TASK_IDS)
        ).sum()
    )
    exact_date_missing_n = public_label_n - len(inputs.bridge)
    medians = primary.groupby("training_regime")["raw_auroc"].median().to_dict()
    report = [
        "# Same-test training-history comparison v1",
        "",
        "All three LightGBM models are evaluated on the identical 2026-03-01 through ",
        "2026-06-09 Site A test rows. The primary test removes any patient seen in the ",
        "union of pre-marker and current-workflow development data, so the test cohort is ",
        "identical across training regimes. Hyperparameters are fixed from the frozen ",
        "current-workflow Protocol B task model; only training history changes.",
        f"The exact-date bridge covers {len(inputs.bridge):,} of {public_label_n:,} Site A task labels; ",
        f"{exact_date_missing_n:,} labels without an exact date are excluded from this comparison.",
        "",
        "## Median raw AUROC",
        "",
        *[f"- {regime}: {value:.3f}" for regime, value in sorted(medians.items())],
        "",
        "The comparison estimates training-history effects on one common test. It does not ",
        "identify a causal drug-card effect.",
    ]
    (output / "same_test_training_history_report_v1.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    files = sorted(path for path in output.iterdir() if path.is_file())
    manifest = {
        "analysis_id": ANALYSIS_ID,
        "analysis_version": "major-revision-v1",
        "parent_analysis_version": "v2026.07.17.3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task_ids": list(TASK_IDS),
        "endpoint": "historical_S_vs_IR",
        "feature_representation": "intensity6000",
        "test_window": ["2026-03-01", "2026-06-09"],
        "training_regimes": [
            "pre_marker_history_only",
            "current_workflow_only",
            "pooled_pre_and_current",
        ],
        "fixed_hyperparameter_source": str(args.primary_metrics.resolve()),
        "threads": args.threads,
        "bootstrap_count": args.bootstrap_count,
        "seed": args.seed,
        "test_labels_used_for_tuning": False,
        "shared_model_seed_across_training_regimes": True,
        "site_a_public_task_label_rows": public_label_n,
        "exact_date_bridge_label_rows": len(inputs.bridge),
        "exact_date_missing_label_rows": exact_date_missing_n,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "pyarrow")
        },
        "outputs": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in files
        },
    }
    (output / "run_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(pd.Series(medians).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
