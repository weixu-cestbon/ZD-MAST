#!/usr/bin/env python3
"""Run the reviewer-grade ZD-MAST calendar-year temporal benchmark."""

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

from zd_mast.temporal_revision import (
    ANALYSIS_ID,
    DEFAULT_SEED,
    TEST_YEARS,
    build_annual_cohort,
    load_annual_inputs,
    run_annual_cell,
    selected_tasks,
)
from zd_mast.cross_platform import TASK_IDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tasks", default=",".join(TASK_IDS))
    parser.add_argument("--years", default=",".join(str(year) for year in TEST_YEARS))
    parser.add_argument("--write-predictions", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(path: Path, primary: pd.DataFrame, support: pd.DataFrame) -> None:
    status_counts = primary["status"].value_counts().to_dict()
    ok = primary.loc[primary["status"].eq("ok")]
    lines = [
        "# Calendar-year rolling temporal benchmark v1",
        "",
        "## Design",
        "",
        "Each named calendar test year uses only earlier years for development. Hyperparameter ",
        "selection, Platt calibration and operating thresholds use patient-purged rolling-origin ",
        "development folds only. The primary test excludes missing patient groups and patients ",
        "seen in development; the original all-sample workload is retained as a sensitivity.",
        "",
        "## Completion",
        "",
        f"- Complete task-year grid: {len(primary)} cells.",
        f"- Status counts: {status_counts}.",
        f"- Median primary raw AUROC among evaluable cells: {ok['raw_auroc'].median():.3f}."
        if len(ok)
        else "- No evaluable cells.",
        "",
        "## Interpretation boundary",
        "",
        "Annual differences estimate total temporal transportability. They combine epidemiology, ",
        "laboratory workflow, data-route and other time-associated changes and are not interpreted ",
        "as a causal card effect.",
        "",
        "The machine-readable cohort and purge audit is the authoritative denominator record.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output}")
    output.mkdir(parents=True)
    tasks = selected_tasks(args.tasks.split(","))
    years = tuple(int(value) for value in args.years.split(",") if value.strip())
    if not set(years).issubset(TEST_YEARS):
        raise ValueError(f"Unsupported test years: {sorted(set(years) - set(TEST_YEARS))}")
    inputs = load_annual_inputs(args.release_root.resolve())

    all_results: list[dict[str, object]] = []
    tuning_parts: list[pd.DataFrame] = []
    bootstrap_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    fold_parts: list[pd.DataFrame] = []
    support_rows: list[dict[str, object]] = []
    for test_year in years:
        for task_index, task_id in enumerate(tasks, start=1):
            print(f"[{test_year} {task_index}/{len(tasks)}] {task_id}", flush=True)
            cohorts = build_annual_cohort(inputs, task_id, test_year)
            fold = cohorts.fold_audit.copy()
            if not fold.empty:
                fold.insert(0, "test_year", test_year)
                fold.insert(0, "task_id", task_id)
                fold_parts.append(fold)
            support_rows.append(
                {
                    "task_id": task_id,
                    "test_year": test_year,
                    "n_development": len(cohorts.development),
                    "development_positive_n": int(cohorts.development["y"].sum()),
                    "development_negative_n": int(cohorts.development["y"].eq(0).sum()),
                    "n_test_all_samples": len(cohorts.test_all_samples),
                    "test_all_positive_n": int(cohorts.test_all_samples["y"].sum()),
                    "test_all_negative_n": int(cohorts.test_all_samples["y"].eq(0).sum()),
                    "removed_test_patient_overlap_n": cohorts.test_purge_audit[
                        "removed_patient_overlap_n"
                    ],
                    "removed_test_missing_patient_cluster_n": cohorts.test_purge_audit[
                        "removed_missing_patient_cluster_n"
                    ],
                    "n_test_patient_disjoint": len(cohorts.test_patient_disjoint),
                    "test_disjoint_positive_n": int(cohorts.test_patient_disjoint["y"].sum()),
                    "test_disjoint_negative_n": int(cohorts.test_patient_disjoint["y"].eq(0).sum()),
                    "valid_fold_n": len(cohorts.folds),
                }
            )
            results, tuning, bootstrap, predictions = run_annual_cell(
                inputs,
                cohorts,
                threads=args.threads,
                bootstrap_count=args.bootstrap_count,
                seed=args.seed,
            )
            all_results.extend(results)
            if not tuning.empty:
                tuning_parts.append(tuning)
            if not bootstrap.empty:
                bootstrap_parts.append(bootstrap)
            if not predictions.empty:
                prediction_parts.append(predictions)

    results = pd.DataFrame(all_results)
    primary = results.loc[results["analysis_variant"].eq("patient_disjoint_primary")].copy()
    sensitivity = results.loc[results["analysis_variant"].eq("all_sample_sensitivity")].copy()
    support = pd.DataFrame(support_rows)
    primary.to_csv(output / "zd_mast_calendar_year_rolling_patient_disjoint_metrics_v1.csv", index=False)
    sensitivity.to_csv(output / "zd_mast_calendar_year_rolling_all_sample_sensitivity_v1.csv", index=False)
    support.to_csv(output / "zd_mast_calendar_year_rolling_cohort_support_v1.csv", index=False)
    (pd.concat(fold_parts, ignore_index=True) if fold_parts else pd.DataFrame()).to_csv(
        output / "zd_mast_calendar_year_rolling_fold_purge_audit_v1.csv", index=False
    )
    (pd.concat(tuning_parts, ignore_index=True) if tuning_parts else pd.DataFrame()).to_csv(
        output / "zd_mast_calendar_year_rolling_tuning_audit_v1.csv", index=False
    )
    (pd.concat(bootstrap_parts, ignore_index=True) if bootstrap_parts else pd.DataFrame()).to_csv(
        output / "zd_mast_calendar_year_rolling_bootstrap_intervals_v1.csv", index=False
    )
    if args.write_predictions and prediction_parts:
        pd.concat(prediction_parts, ignore_index=True).to_parquet(
            output / "zd_mast_calendar_year_rolling_predictions_private_review_v1.parquet",
            index=False,
        )
    write_report(output / "calendar_year_rolling_report_v1.md", primary, support)
    output_files = sorted(path for path in output.iterdir() if path.is_file())
    manifest = {
        "analysis_id": ANALYSIS_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_version": "major-revision-v1",
        "parent_analysis_version": "v2026.07.17.3",
        "task_ids": list(tasks),
        "test_years": list(years),
        "endpoint": "historical_S_vs_IR",
        "feature_representation": "intensity6000",
        "patient_disjoint_primary": True,
        "threads": args.threads,
        "bootstrap_count": args.bootstrap_count,
        "seed": args.seed,
        "test_labels_used_for_tuning": False,
        "release_root": str(args.release_root.resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "pyarrow")
        },
        "outputs": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in output_files
        },
    }
    (output / "run_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(primary["status"].value_counts(dropna=False).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
