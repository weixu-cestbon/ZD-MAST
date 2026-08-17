#!/usr/bin/env python3
"""Compare intensity6000 and peak_presence6000 on identical Site A cohorts.

This reviewer-requested sensitivity changes only the feature representation.
Tasks, source development membership, rolling-origin folds, patient-disjoint
test rows, tuning grid, calibration domain, and random seeds are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from zd_mast.cross_platform import (
    SITE_A,
    TASK_IDS,
    build_task_cohorts,
    deterministic_seed,
    evaluate_fitted_task,
    fit_source_task,
    load_analysis_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--target-date-table", type=Path, required=True)
    parser.add_argument("--peak-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resample_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = pd.unique(groups)
    positions = {group: np.flatnonzero(groups == group) for group in unique}
    selected = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([positions[group] for group in selected])


def paired_bootstrap(
    merged: pd.DataFrame,
    *,
    task_id: str,
    cohort_id: str,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    y = merged["y"].to_numpy(dtype=np.int8)
    intensity = merged["raw_probability_intensity"].to_numpy(dtype=float)
    peak = merged["raw_probability_peak"].to_numpy(dtype=float)
    patient = merged["public_patient_cluster_id"].astype("string")
    fallback = "sample:" + merged["public_sample_id"].astype(str)
    groups = patient.where(patient.notna() & patient.str.strip().ne(""), fallback).to_numpy(str)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for replicate in range(n_boot):
        index = resample_indices(groups, rng)
        if np.unique(y[index]).size < 2:
            continue
        rows.append(
            {
                "task_id": task_id,
                "cohort_id": cohort_id,
                "bootstrap_replicate": replicate,
                "intensity_auroc": roc_auc_score(y[index], intensity[index]),
                "peak_presence_auroc": roc_auc_score(y[index], peak[index]),
                "delta_peak_minus_intensity_auroc": roc_auc_score(y[index], peak[index])
                - roc_auc_score(y[index], intensity[index]),
                "intensity_auprc": average_precision_score(y[index], intensity[index]),
                "peak_presence_auprc": average_precision_score(y[index], peak[index]),
                "delta_peak_minus_intensity_auprc": average_precision_score(y[index], peak[index])
                - average_precision_score(y[index], intensity[index]),
            }
        )
    draws = pd.DataFrame(rows)
    summary: dict[str, object] = {
        "task_id": task_id,
        "cohort_id": cohort_id,
        "n_test": len(merged),
        "positive_n": int(y.sum()),
        "negative_n": int((y == 0).sum()),
        "patient_cluster_n": int(pd.Series(groups).nunique()),
        "intensity_raw_auroc": roc_auc_score(y, intensity),
        "peak_presence_raw_auroc": roc_auc_score(y, peak),
        "delta_peak_minus_intensity_raw_auroc": roc_auc_score(y, peak)
        - roc_auc_score(y, intensity),
        "intensity_raw_auprc": average_precision_score(y, intensity),
        "peak_presence_raw_auprc": average_precision_score(y, peak),
        "delta_peak_minus_intensity_raw_auprc": average_precision_score(y, peak)
        - average_precision_score(y, intensity),
        "bootstrap_valid_n": len(draws),
    }
    for metric in ("delta_peak_minus_intensity_auroc", "delta_peak_minus_intensity_auprc"):
        summary[f"{metric}_ci_low"] = draws[metric].quantile(0.025)
        summary[f"{metric}_ci_high"] = draws[metric].quantile(0.975)
    return draws, summary


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)

    inputs = load_analysis_inputs(
        args.release_root,
        args.target_date_table,
        task_ids=TASK_IDS,
        validate_binary_matrices=False,
    )
    intensity_path = inputs.feature_root / "zd_mast_a_sample_level_intensity6000_v1.0.0.npy"
    if not intensity_path.is_file():
        raise FileNotFoundError(intensity_path)
    intensity_matrix = np.load(intensity_path, mmap_mode="r")
    if intensity_matrix.shape != inputs.matrices[SITE_A].shape:
        raise ValueError(
            f"Site A intensity/peak matrix shape mismatch: {intensity_matrix.shape} vs "
            f"{inputs.matrices[SITE_A].shape}"
        )
    if not np.isfinite(intensity_matrix).all():
        raise ValueError("Site A intensity matrix contains NaN or infinity")

    peak_predictions = pd.read_parquet(args.peak_predictions)
    required = {
        "task_id",
        "cohort_id",
        "site_id",
        "public_sample_id",
        "public_patient_cluster_id",
        "y",
        "raw_probability",
    }
    missing = required - set(peak_predictions.columns)
    if missing:
        raise ValueError(f"Peak prediction table missing columns: {sorted(missing)}")
    peak_predictions = peak_predictions[
        peak_predictions["site_id"].eq(SITE_A)
        & peak_predictions["cohort_id"].isin(
            ["site_a_test_patient_disjoint", "site_a_test_all_samples"]
        )
    ].copy()

    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    tuning: list[pd.DataFrame] = []
    bootstraps: list[pd.DataFrame] = []
    paired_summaries: list[dict[str, object]] = []
    paired_draws: list[pd.DataFrame] = []
    for index, task_id in enumerate(TASK_IDS, start=1):
        print(f"[{index}/{len(TASK_IDS)}] {task_id}", flush=True)
        cohorts = build_task_cohorts(inputs, task_id)
        fitted = fit_source_task(
            intensity_matrix,
            cohorts,
            threads=args.threads,
            seed=deterministic_seed(args.seed, task_id, "intensity6000"),
        )
        tuning.append(fitted.tuning.assign(feature_representation="intensity6000"))
        for cohort_id, frame in (
            ("site_a_test_patient_disjoint", cohorts.source_test_patient_disjoint),
            ("site_a_test_all_samples", cohorts.source_test_all_samples),
        ):
            row, bootstrap, prediction = evaluate_fitted_task(
                fitted,
                intensity_matrix,
                frame,
                cohort_id=cohort_id,
                site_id=SITE_A,
                date_start="2026-03-01",
                date_end="2026-06-09",
                n_boot=args.bootstrap,
                seed=deterministic_seed(args.seed, task_id, cohort_id, "intensity_bootstrap"),
            )
            row["analysis_id"] = "site_a_representation_comparison"
            row["feature_representation"] = "intensity6000"
            metrics.append(row)
            prediction["feature_representation"] = "intensity6000"
            predictions.append(prediction)
            if not bootstrap.empty:
                bootstrap["feature_representation"] = "intensity6000"
                bootstraps.append(bootstrap)

            peak = peak_predictions[
                peak_predictions["task_id"].eq(task_id)
                & peak_predictions["cohort_id"].eq(cohort_id)
            ].copy()
            merged = prediction.merge(
                peak[
                    [
                        "public_sample_id",
                        "y",
                        "raw_probability",
                    ]
                ],
                on="public_sample_id",
                how="inner",
                validate="one_to_one",
                suffixes=("_intensity", "_peak"),
            )
            if len(merged) != len(prediction) or len(merged) != len(peak):
                raise ValueError(f"{task_id} {cohort_id}: representation cohorts differ")
            if not merged["y_intensity"].eq(merged["y_peak"]).all():
                raise ValueError(f"{task_id} {cohort_id}: labels differ by representation")
            merged = merged.rename(columns={"y_intensity": "y"}).drop(columns="y_peak")
            draws, summary = paired_bootstrap(
                merged,
                task_id=task_id,
                cohort_id=cohort_id,
                n_boot=args.bootstrap,
                seed=deterministic_seed(args.seed, task_id, cohort_id, "paired_representation"),
            )
            paired_draws.append(draws)
            paired_summaries.append(summary)

    pd.DataFrame(metrics).to_csv(output / "site_a_intensity6000_metrics_v1.csv", index=False)
    pd.concat(tuning, ignore_index=True).to_csv(output / "site_a_intensity6000_tuning_v1.csv", index=False)
    pd.concat(bootstraps, ignore_index=True).to_csv(
        output / "site_a_intensity6000_bootstrap_v1.csv", index=False
    )
    pd.DataFrame(paired_summaries).to_csv(
        output / "site_a_representation_paired_summary_v1.csv", index=False
    )
    pd.concat(paired_draws, ignore_index=True).to_csv(
        output / "site_a_representation_paired_bootstrap_v1.csv", index=False
    )
    pd.concat(predictions, ignore_index=True).to_parquet(
        output / "site_a_intensity6000_predictions_private_review_v1.parquet", index=False
    )

    manifest = {
        "analysis_id": "site_a_representation_comparison_v1",
        "status": "COMPLETE",
        "purpose": "Reviewer-requested like-for-like representation degradation audit",
        "only_changed_factor": "feature_representation",
        "representations": ["intensity6000", "peak_presence6000"],
        "site": SITE_A,
        "task_ids": list(TASK_IDS),
        "model": "lightgbm",
        "endpoint": "historical_S_vs_IR",
        "seed": args.seed,
        "threads": args.threads,
        "bootstrap_replicates": args.bootstrap,
        "target_labels_used": False,
        "python": sys.version,
        "platform": platform.platform(),
        "inputs": {
            "intensity_matrix": {"path": str(intensity_path), "sha256": sha256(intensity_path)},
            "peak_predictions": {
                "path": str(args.peak_predictions.resolve()),
                "sha256": sha256(args.peak_predictions.resolve()),
            },
        },
    }
    (output / "run_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
