#!/usr/bin/env python3
"""Run current-contract L2 logistic baselines on frozen ZD-MAST cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# Allow a clean checkout to run this script without an editable installation.
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from zd_mast.cross_platform import (
    COHORT_WINDOWS,
    SITE_A,
    SITE_B,
    TASK_IDS,
    build_task_cohorts,
    deterministic_seed,
    load_analysis_inputs,
    validate_task_ids,
)
from zd_mast.logistic_sensitivity import (
    ANALYSIS_ID,
    DEFAULT_SEED,
    cohort_audit_row,
    evaluate_logistic_fit,
    fit_development_logistic,
    fold_membership_sha256,
    logistic_grid,
    membership_sha256,
)
from zd_mast.temporal_revision import (
    TEST_YEARS,
    build_annual_cohort,
    load_annual_inputs,
    support_status,
)
from zd_mast.training_history import (
    build_history_cohorts,
    load_training_history_inputs,
    paired_history_deltas,
)


PROTOCOL_ANNUAL = "annual_patient_purged"
PROTOCOL_HISTORY = "same_test_training_history"
PROTOCOL_CROSS = "cross_platform_source_only"


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--task-ids",
        default=",".join(TASK_IDS),
        help="Comma-separated subset of the frozen historical ten-task panel.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--bootstrap-count",
        type=int,
        default=2000,
        help="Patient-cluster bootstrap replicates; use 0 to disable.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--write-predictions",
        action="store_true",
        help="Write public-ID sample probabilities for private review.",
    )
    parser.add_argument(
        "--skip-input-hashes",
        action="store_true",
        help="Record size/mtime but skip SHA-256 of runtime inputs.",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="protocol_family", required=True)

    annual = subparsers.add_parser(
        "annual",
        help="Calendar-year rolling patient-purged sensitivity.",
    )
    _add_common_arguments(annual)
    annual.add_argument(
        "--test-years",
        default=",".join(str(year) for year in TEST_YEARS),
        help="Comma-separated subset of 2022,2023,2024,2025.",
    )

    history = subparsers.add_parser(
        "training-history",
        help="Same-test pre/current/pooled training-history sensitivity.",
    )
    _add_common_arguments(history)
    history.add_argument("--temporal-bridge", required=True, type=Path)
    history.add_argument(
        "--primary-metrics",
        required=True,
        type=Path,
        help=(
            "Existing primary metrics required by the frozen history loader. "
            "Its LightGBM parameters are validated by that loader but ignored here."
        ),
    )

    cross = subparsers.add_parser(
        "cross-platform",
        help="One Site A logistic model applied unchanged to both sites.",
    )
    _add_common_arguments(cross)
    cross.add_argument("--target-date-table", required=True, type=Path)
    cross.add_argument(
        "--skip-full-matrix-value-scan",
        action="store_true",
        help="Skip the chunked 0/1 scan while retaining shape and row checks.",
    )

    args = parser.parse_args(argv)
    if args.bootstrap_count < 0:
        parser.error("--bootstrap-count must be non-negative")
    args.task_ids = validate_task_ids(str(args.task_ids).split(","))
    if args.protocol_family == "annual":
        try:
            years = tuple(dict.fromkeys(int(value) for value in args.test_years.split(",")))
        except ValueError as exc:
            parser.error(f"Invalid --test-years: {exc}")
        unknown = set(years) - set(TEST_YEARS)
        if unknown or not years:
            parser.error(f"--test-years must be a non-empty subset of {TEST_YEARS}")
        args.test_years = tuple(year for year in TEST_YEARS if year in set(years))
    return args


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, hash_content: bool) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "modified_time_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(resolved) if hash_content else None,
        "sha256_status": "computed" if hash_content else "skipped_by_cli",
    }


def _package_versions() -> dict[str, str]:
    output: dict[str, str] = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow"):
        try:
            output[package] = version(package)
        except PackageNotFoundError:
            output[package] = "not_installed"
    return output


def _output_prefix(protocol_family: str) -> str:
    return f"zd_mast_{protocol_family}_logistic_sensitivity_v1"


def _base_manifest(args: argparse.Namespace) -> dict[str, object]:
    return {
        "analysis_id": ANALYSIS_ID,
        "analysis_version": "reviewer-r9-v1",
        "protocol_family": args.protocol_family,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "task_ids": list(args.task_ids),
        "endpoint": "historical_S_vs_IR",
        "model": "logistic_regression",
        "penalty": "l2",
        "hyperparameter_grid": logistic_grid(),
        "selection_objective": "median_AUROC + 0.05 * median_AUPRC",
        "standardization": "StandardScaler(with_mean=False) fitted within each fold",
        "calibration": "none_raw_probability_only",
        "metrics": ["raw_AUROC", "raw_AUPRC", "raw_Brier", "raw_ECE"],
        "bootstrap_count": int(args.bootstrap_count),
        "seed": int(args.seed),
        "preflight_only": bool(args.preflight_only),
        "write_predictions": bool(args.write_predictions),
        "guardrails": {
            "cohorts_and_splits_loaded_from_existing_frozen_contracts": True,
            "evaluation_labels_used_for_hyperparameter_tuning": False,
            "target_labels_used_for_training": False,
            "target_labels_used_for_hyperparameter_tuning": False,
            "target_labels_used_for_calibration": False,
            "source_only_calibration_performed": False,
            "sparse_centering_disabled": True,
            "refuse_existing_output_directory": True,
        },
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "command": [str(value) for value in sys.argv],
    }


def _write_manifest(output_dir: Path, prefix: str, manifest: Mapping[str, object]) -> Path:
    path = output_dir / f"{prefix}_run_manifest.json"
    path.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def _cohort_support(frame: pd.DataFrame) -> tuple[str, str]:
    if frame.empty:
        return "insufficient", "no_samples"
    counts = frame["y"].value_counts()
    if counts.size < 2:
        return "insufficient", "single_class"
    if len(frame) < 100:
        return "insufficient", "n<100"
    if int(counts.min()) < 20:
        return "insufficient", "min_class<20"
    return "adequate", ""


def _placeholder_row(
    *,
    protocol_family: str,
    task_id: str,
    cohort_id: str,
    site_id: str,
    frame: pd.DataFrame,
    feature_representation: str,
    reason: str,
    extra_fields: Mapping[str, object] | None = None,
) -> dict[str, object]:
    y = frame.get("y", pd.Series(dtype=np.int8)).to_numpy(dtype=np.int8)
    row: dict[str, object] = {
        "analysis_id": ANALYSIS_ID,
        "protocol_family": protocol_family,
        "task_id": task_id,
        "model": "logistic_regression",
        "endpoint": "historical_S_vs_IR",
        "feature_representation": feature_representation,
        "cohort_id": cohort_id,
        "site_id": site_id,
        "status": "insufficient",
        "insufficient_reason": reason,
        "n_test": int(len(frame)),
        "test_positive_n": int(y.sum()) if len(y) else 0,
        "test_negative_n": int(y.size - y.sum()) if len(y) else 0,
        "evaluation_membership_sha256": membership_sha256(frame),
        "target_labels_used_for_training": False,
        "target_labels_used_for_hyperparameter_tuning": False,
        "target_labels_used_for_calibration": False,
    }
    if extra_fields:
        row.update(dict(extra_fields))
    return row


def _annotate_tuning(
    tuning: pd.DataFrame,
    *,
    protocol_family: str,
    task_id: str,
    annotations: Mapping[str, object],
) -> pd.DataFrame:
    output = tuning.copy()
    output.insert(0, "task_id", task_id)
    output.insert(0, "protocol_family", protocol_family)
    for column, value in reversed(list(annotations.items())):
        output.insert(2, column, value)
    return output


def _run_annual(
    args: argparse.Namespace,
) -> tuple[dict[str, list[pd.DataFrame] | list[dict[str, object]]], list[Path], dict[str, object]]:
    inputs = load_annual_inputs(args.release_root.resolve())
    collected: dict[str, list] = {
        "metrics": [],
        "tuning": [],
        "bootstrap": [],
        "predictions": [],
        "cohort_audit": [],
        "fold_audit": [],
        "paired_deltas": [],
    }
    for task_id in args.task_ids:
        for test_year in args.test_years:
            print(f"[annual] {task_id} test={test_year}", flush=True)
            cohorts = build_annual_cohort(inputs, task_id, test_year)
            for cohort_id, frame, role in (
                ("development", cohorts.development, "development"),
                ("patient_disjoint_primary", cohorts.test_patient_disjoint, "primary_test"),
                ("all_sample_sensitivity", cohorts.test_all_samples, "sensitivity_test"),
            ):
                collected["cohort_audit"].append(
                    {
                        **cohort_audit_row(
                            frame,
                            protocol_family=PROTOCOL_ANNUAL,
                            task_id=task_id,
                            cohort_id=cohort_id,
                            site_id=SITE_A,
                            role=role,
                        ),
                        "test_year": test_year,
                    }
                )
            if not cohorts.fold_audit.empty:
                audit = cohorts.fold_audit.copy()
                audit.insert(0, "test_year", test_year)
                audit.insert(0, "task_id", task_id)
                audit.insert(0, "protocol_family", PROTOCOL_ANNUAL)
                audit["fold_membership_sha256"] = fold_membership_sha256(cohorts.folds)
                collected["fold_audit"].append(audit)
            if args.preflight_only:
                continue
            development_status, development_reason = support_status(cohorts.development, 100, 20)
            primary_status, primary_reason = support_status(
                cohorts.test_patient_disjoint, 100, 20
            )
            if development_status != "adequate" or primary_status != "adequate" or len(cohorts.folds) < 2:
                reason = ";".join(
                    value
                    for value in (
                        development_reason,
                        primary_reason,
                        "valid_folds<2" if len(cohorts.folds) < 2 else "",
                    )
                    if value
                )
                for cohort_id, frame in (
                    ("patient_disjoint_primary", cohorts.test_patient_disjoint),
                    ("all_sample_sensitivity", cohorts.test_all_samples),
                ):
                    collected["metrics"].append(
                        _placeholder_row(
                            protocol_family=PROTOCOL_ANNUAL,
                            task_id=task_id,
                            cohort_id=cohort_id,
                            site_id=SITE_A,
                            frame=frame,
                            feature_representation="intensity6000",
                            reason=reason,
                            extra_fields={"test_year": test_year},
                        )
                    )
                continue
            fitted = fit_development_logistic(
                inputs.matrix,
                task_id,
                cohorts.development,
                cohorts.folds,
                seed=deterministic_seed(args.seed, PROTOCOL_ANNUAL, test_year),
                evaluation_frames_for_overlap_audit=(cohorts.test_all_samples,),
            )
            collected["tuning"].append(
                _annotate_tuning(
                    fitted.tuning,
                    protocol_family=PROTOCOL_ANNUAL,
                    task_id=task_id,
                    annotations={"test_year": test_year},
                )
            )
            for cohort_id, frame in (
                ("patient_disjoint_primary", cohorts.test_patient_disjoint),
                ("all_sample_sensitivity", cohorts.test_all_samples),
            ):
                row, bootstrap, predictions = evaluate_logistic_fit(
                    fitted,
                    inputs.matrix,
                    frame,
                    protocol_family=PROTOCOL_ANNUAL,
                    cohort_id=cohort_id,
                    site_id=SITE_A,
                    feature_representation="intensity6000",
                    bootstrap_count=args.bootstrap_count,
                    seed=deterministic_seed(args.seed, PROTOCOL_ANNUAL, task_id, test_year, cohort_id),
                    extra_fields={
                        "test_year": test_year,
                        "n_development": len(cohorts.development),
                        "development_positive_n": int(cohorts.development["y"].sum()),
                        "development_negative_n": int(cohorts.development["y"].eq(0).sum()),
                    },
                )
                predictions["test_year"] = test_year
                collected["metrics"].append(row)
                if not bootstrap.empty:
                    bootstrap["test_year"] = test_year
                    collected["bootstrap"].append(bootstrap)
                collected["predictions"].append(predictions)
    input_paths = [
        inputs.feature_root / "zd_mast_a_sample_level_intensity6000_v1.0.0.npy",
        inputs.feature_root / "zd_mast_sample_metadata_public_v1.0.0.csv",
        inputs.feature_root / "zd_mast_ast_labels_historical_v1.0.0.parquet",
        inputs.feature_root / "zd_mast_patient_episode_groups_public_v1.0.0.parquet",
        inputs.feature_root / "zd_mast_split_assignments_public_v1.0.0.csv",
    ]
    contract = {
        "loader": "zd_mast.temporal_revision.load_annual_inputs/build_annual_cohort",
        "analysis_variants": ["patient_disjoint_primary", "all_sample_sensitivity"],
        "test_years": list(args.test_years),
        "feature_representation": "intensity6000",
    }
    return collected, input_paths, contract


def _run_history(
    args: argparse.Namespace,
) -> tuple[dict[str, list[pd.DataFrame] | list[dict[str, object]]], list[Path], dict[str, object]]:
    inputs = load_training_history_inputs(
        args.release_root.resolve(),
        args.temporal_bridge.resolve(),
        args.primary_metrics.resolve(),
    )
    collected: dict[str, list] = {
        "metrics": [],
        "tuning": [],
        "bootstrap": [],
        "predictions": [],
        "cohort_audit": [],
        "fold_audit": [],
        "paired_deltas": [],
    }
    for task_id in args.task_ids:
        print(f"[training-history] {task_id}", flush=True)
        cohorts = build_history_cohorts(inputs, task_id)
        if not cohorts.fold_audit.empty:
            audit = cohorts.fold_audit.copy()
            audit.insert(0, "protocol_family", PROTOCOL_HISTORY)
            collected["fold_audit"].append(audit)
        for regime, development in cohorts.development_by_regime.items():
            collected["cohort_audit"].append(
                cohort_audit_row(
                    development,
                    protocol_family=PROTOCOL_HISTORY,
                    task_id=task_id,
                    cohort_id=regime,
                    site_id=SITE_A,
                    role="development",
                )
            )
        for cohort_id, frame, role in (
            (
                "patient_disjoint_common_test_primary",
                cohorts.test_patient_disjoint_common,
                "primary_common_test",
            ),
            ("all_sample_common_test_sensitivity", cohorts.test_all_samples, "sensitivity_test"),
        ):
            collected["cohort_audit"].append(
                cohort_audit_row(
                    frame,
                    protocol_family=PROTOCOL_HISTORY,
                    task_id=task_id,
                    cohort_id=cohort_id,
                    site_id=SITE_A,
                    role=role,
                )
            )
        if args.preflight_only:
            continue
        for regime, development in cohorts.development_by_regime.items():
            folds = cohorts.folds_by_regime[regime]
            development_status, development_reason = _cohort_support(development)
            if development_status != "adequate" or len(folds) < 2:
                reason = ";".join(
                    value
                    for value in (development_reason, "valid_folds<2" if len(folds) < 2 else "")
                    if value
                )
                for cohort_id, frame in (
                    (
                        "patient_disjoint_common_test_primary",
                        cohorts.test_patient_disjoint_common,
                    ),
                    ("all_sample_common_test_sensitivity", cohorts.test_all_samples),
                ):
                    collected["metrics"].append(
                        _placeholder_row(
                            protocol_family=PROTOCOL_HISTORY,
                            task_id=task_id,
                            cohort_id=cohort_id,
                            site_id=SITE_A,
                            frame=frame,
                            feature_representation="intensity6000",
                            reason=reason,
                            extra_fields={"training_regime": regime},
                        )
                    )
                continue
            fitted = fit_development_logistic(
                inputs.matrix,
                task_id,
                development,
                folds,
                seed=deterministic_seed(args.seed, PROTOCOL_HISTORY, regime),
                evaluation_frames_for_overlap_audit=(cohorts.test_all_samples,),
            )
            collected["tuning"].append(
                _annotate_tuning(
                    fitted.tuning,
                    protocol_family=PROTOCOL_HISTORY,
                    task_id=task_id,
                    annotations={"training_regime": regime},
                )
            )
            for cohort_id, frame in (
                (
                    "patient_disjoint_common_test_primary",
                    cohorts.test_patient_disjoint_common,
                ),
                ("all_sample_common_test_sensitivity", cohorts.test_all_samples),
            ):
                row, bootstrap, predictions = evaluate_logistic_fit(
                    fitted,
                    inputs.matrix,
                    frame,
                    protocol_family=PROTOCOL_HISTORY,
                    cohort_id=cohort_id,
                    site_id=SITE_A,
                    feature_representation="intensity6000",
                    bootstrap_count=args.bootstrap_count,
                    seed=deterministic_seed(args.seed, PROTOCOL_HISTORY, task_id, regime, cohort_id),
                    extra_fields={
                        "training_regime": regime,
                        "n_development": len(development),
                        "development_positive_n": int(development["y"].sum()),
                        "development_negative_n": int(development["y"].eq(0).sum()),
                    },
                )
                predictions["training_regime"] = regime
                predictions["analysis_variant"] = cohort_id
                collected["metrics"].append(row)
                if not bootstrap.empty:
                    bootstrap["training_regime"] = regime
                    collected["bootstrap"].append(bootstrap)
                collected["predictions"].append(predictions)
    if not args.preflight_only and collected["predictions"] and args.bootstrap_count > 0:
        aligned = pd.concat(collected["predictions"], ignore_index=True)
        collected["paired_deltas"].append(
            paired_history_deltas(
                aligned,
                bootstrap_count=args.bootstrap_count,
                seed=args.seed,
            )
        )
    input_paths = [
        inputs.feature_root / "zd_mast_a_sample_level_intensity6000_v1.0.0.npy",
        inputs.feature_root / "zd_mast_protocol_b_rolling_origin_folds_public_v1.0.0.csv",
        args.temporal_bridge.resolve(),
        args.primary_metrics.resolve(),
    ]
    contract = {
        "loader": "zd_mast.training_history.load_training_history_inputs/build_history_cohorts",
        "training_regimes": [
            "pre_marker_history_only",
            "current_workflow_only",
            "pooled_pre_and_current",
        ],
        "feature_representation": "intensity6000",
        "existing_primary_metrics_role": (
            "required and validated by the frozen loader; LightGBM parameters ignored by this analysis"
        ),
    }
    return collected, input_paths, contract


def _run_cross_platform(
    args: argparse.Namespace,
) -> tuple[dict[str, list[pd.DataFrame] | list[dict[str, object]]], list[Path], dict[str, object]]:
    inputs = load_analysis_inputs(
        args.release_root.resolve(),
        args.target_date_table.resolve(),
        task_ids=args.task_ids,
        validate_binary_matrices=not args.skip_full_matrix_value_scan,
    )
    collected: dict[str, list] = {
        "metrics": [],
        "tuning": [],
        "bootstrap": [],
        "predictions": [],
        "cohort_audit": [],
        "fold_audit": [],
        "paired_deltas": [],
    }
    for task_id in args.task_ids:
        print(f"[cross-platform] {task_id}", flush=True)
        cohorts = build_task_cohorts(inputs, task_id)
        for cohort_id, frame, site_id, role in (
            ("source_development", cohorts.source_development, SITE_A, "development"),
            (
                "site_a_test_patient_disjoint",
                cohorts.source_test_patient_disjoint,
                SITE_A,
                "primary_source_test",
            ),
            (
                "site_a_test_all_samples",
                cohorts.source_test_all_samples,
                SITE_A,
                "source_sensitivity_test",
            ),
            ("site_b_primary", cohorts.target_primary, SITE_B, "primary_target_test"),
            (
                "site_b_full_period_sensitivity",
                cohorts.target_full_period,
                SITE_B,
                "target_sensitivity_test",
            ),
        ):
            collected["cohort_audit"].append(
                cohort_audit_row(
                    frame,
                    protocol_family=PROTOCOL_CROSS,
                    task_id=task_id,
                    cohort_id=cohort_id,
                    site_id=site_id,
                    role=role,
                )
            )
        if not cohorts.fold_purge_audit.empty:
            audit = cohorts.fold_purge_audit.copy()
            audit.insert(0, "protocol_family", PROTOCOL_CROSS)
            collected["fold_audit"].append(audit)
        if args.preflight_only:
            continue
        development_status, development_reason = _cohort_support(cohorts.source_development)
        if development_status != "adequate" or len(cohorts.source_folds) < 2:
            reason = ";".join(
                value
                for value in (
                    development_reason,
                    "valid_folds<2" if len(cohorts.source_folds) < 2 else "",
                )
                if value
            )
            for cohort_id, frame, site_id in (
                ("site_a_test_patient_disjoint", cohorts.source_test_patient_disjoint, SITE_A),
                ("site_a_test_all_samples", cohorts.source_test_all_samples, SITE_A),
                ("site_b_primary", cohorts.target_primary, SITE_B),
                ("site_b_full_period_sensitivity", cohorts.target_full_period, SITE_B),
            ):
                collected["metrics"].append(
                    _placeholder_row(
                        protocol_family=PROTOCOL_CROSS,
                        task_id=task_id,
                        cohort_id=cohort_id,
                        site_id=site_id,
                        frame=frame,
                        feature_representation="peak_presence6000",
                        reason=reason,
                    )
                )
            continue

        # Only Site A development/folds enter this fitting call. Target frames
        # are not accepted by the estimator boundary.
        fitted = fit_development_logistic(
            inputs.matrices[SITE_A],
            task_id,
            cohorts.source_development,
            cohorts.source_folds,
            seed=deterministic_seed(args.seed, PROTOCOL_CROSS),
            evaluation_frames_for_overlap_audit=(cohorts.source_test_all_samples,),
        )
        collected["tuning"].append(
            _annotate_tuning(
                fitted.tuning,
                protocol_family=PROTOCOL_CROSS,
                task_id=task_id,
                annotations={"training_site": SITE_A},
            )
        )
        for cohort_id, frame, site_id in (
            ("site_a_test_patient_disjoint", cohorts.source_test_patient_disjoint, SITE_A),
            ("site_a_test_all_samples", cohorts.source_test_all_samples, SITE_A),
            ("site_b_primary", cohorts.target_primary, SITE_B),
            ("site_b_full_period_sensitivity", cohorts.target_full_period, SITE_B),
        ):
            date_start, date_end, _ = COHORT_WINDOWS[cohort_id]
            row, bootstrap, predictions = evaluate_logistic_fit(
                fitted,
                inputs.matrices[site_id],
                frame,
                protocol_family=PROTOCOL_CROSS,
                cohort_id=cohort_id,
                site_id=site_id,
                feature_representation="peak_presence6000",
                bootstrap_count=args.bootstrap_count,
                seed=deterministic_seed(args.seed, PROTOCOL_CROSS, task_id, cohort_id),
                extra_fields={
                    "training_site": SITE_A,
                    "date_start": date_start,
                    "date_end": date_end,
                    "same_fitted_source_model": True,
                    "n_development": len(cohorts.source_development),
                    "development_positive_n": int(cohorts.source_development["y"].sum()),
                    "development_negative_n": int(cohorts.source_development["y"].eq(0).sum()),
                },
            )
            predictions["calibrated_probability"] = np.nan
            collected["metrics"].append(row)
            if not bootstrap.empty:
                collected["bootstrap"].append(bootstrap)
            collected["predictions"].append(predictions)
    input_paths = list(inputs.input_paths.values())
    contract = {
        "loader": "zd_mast.cross_platform.load_analysis_inputs/build_task_cohorts",
        "feature_representation": "peak_presence6000",
        "training_site": SITE_A,
        "target_site": SITE_B,
        "same_fitted_source_model_applied_to_both_sites": True,
        "target_labels_role": "final_evaluation_only",
        "target_labels_used_for_training_tuning_or_calibration": False,
        "matrix_value_scan": "skipped" if args.skip_full_matrix_value_scan else "complete",
    }
    return collected, input_paths, contract


def _combine(parts: Iterable[pd.DataFrame | dict[str, object]]) -> pd.DataFrame:
    items = list(parts)
    if not items:
        return pd.DataFrame()
    frames = [item if isinstance(item, pd.DataFrame) else pd.DataFrame([item]) for item in items]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _write_report(
    output_dir: Path,
    prefix: str,
    *,
    protocol_family: str,
    metrics: pd.DataFrame,
    cohort_audit: pd.DataFrame,
    preflight_only: bool,
) -> Path:
    lines = [
        "# ZD-MAST current-contract logistic sensitivity v1",
        "",
        f"- Protocol family: `{protocol_family}`",
        f"- Status: `{'PREFLIGHT_COMPLETE' if preflight_only else 'COMPLETE'}`",
        "- Model: L2 logistic regression with sparse-safe fold-local standardization",
        "- Hyperparameters: C in 0.01, 0.1, 1, 10; class weight none or balanced",
        "- Selection domain: frozen development folds only",
        "- Reported probabilities: raw, without post-hoc calibration",
        "- Target labels used for source-only training, tuning, or calibration: no",
        "",
        "## Cohort audit",
        "",
        f"Cohort rows: {len(cohort_audit):,}",
    ]
    if preflight_only:
        lines.extend(["", "No estimators were fitted in preflight mode."])
    elif metrics.empty:
        lines.extend(["", "No evaluable model rows were produced."])
    else:
        successful = metrics.loc[metrics["status"].eq("ok")]
        lines.extend(
            [
                "",
                "## Raw discrimination summary",
                "",
                f"Successful evaluation rows: {len(successful):,} / {len(metrics):,}",
            ]
        )
        if not successful.empty:
            summary = (
                successful.groupby(["cohort_id", "site_id"], observed=True)
                .agg(task_n=("task_id", "nunique"), median_raw_auroc=("raw_auroc", "median"), median_raw_auprc=("raw_auprc", "median"))
                .reset_index()
            )
            for row in summary.itertuples(index=False):
                lines.append(
                    f"- {row.cohort_id} ({row.site_id}): tasks={row.task_n}, "
                    f"median AUROC={row.median_raw_auroc:.3f}, "
                    f"median AUPRC={row.median_raw_auprc:.3f}"
                )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is a prespecified linear-model sensitivity analysis on existing frozen "
            "cohorts and splits. It does not replace the primary LightGBM analyses and "
            "does not use evaluation performance to alter task membership or model settings.",
        ]
    )
    path = output_dir / f"{prefix}_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_mapping = {
        "annual": PROTOCOL_ANNUAL,
        "training-history": PROTOCOL_HISTORY,
        "cross-platform": PROTOCOL_CROSS,
    }
    protocol_family = protocol_mapping[args.protocol_family]
    args.protocol_family = protocol_family
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    prefix = _output_prefix(protocol_family)
    manifest = _base_manifest(args)
    _write_manifest(output_dir, prefix, manifest)
    try:
        if protocol_family == PROTOCOL_ANNUAL:
            collected, input_paths, contract = _run_annual(args)
        elif protocol_family == PROTOCOL_HISTORY:
            collected, input_paths, contract = _run_history(args)
        else:
            collected, input_paths, contract = _run_cross_platform(args)

        tables = {name: _combine(values) for name, values in collected.items()}
        written: list[Path] = []
        for name in ("cohort_audit", "fold_audit", "metrics", "tuning", "bootstrap", "paired_deltas"):
            table = tables[name]
            if table.empty and name not in {"cohort_audit", "metrics"}:
                continue
            path = output_dir / f"{prefix}_{name}.csv"
            table.to_csv(path, index=False)
            written.append(path)
        if args.write_predictions and not tables["predictions"].empty:
            prediction_path = output_dir / f"{prefix}_predictions_private_review.parquet"
            tables["predictions"].to_parquet(prediction_path, index=False)
            written.append(prediction_path)
        report_path = _write_report(
            output_dir,
            prefix,
            protocol_family=protocol_family,
            metrics=tables["metrics"],
            cohort_audit=tables["cohort_audit"],
            preflight_only=args.preflight_only,
        )
        written.append(report_path)
        unique_inputs = sorted({Path(path).resolve() for path in input_paths})
        manifest.update(
            {
                "status": "PREFLIGHT_COMPLETE" if args.preflight_only else "COMPLETE",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "data_contract": contract,
                "input_files": [
                    _file_record(path, hash_content=not args.skip_input_hashes)
                    for path in unique_inputs
                ],
                "output_files": {
                    path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                    for path in written
                },
                "output_row_counts": {
                    name: int(len(table)) for name, table in tables.items() if name != "predictions"
                },
            }
        )
        _write_manifest(output_dir, prefix, manifest)
        print(f"status={manifest['status']} output={output_dir}", flush=True)
        return 0
    except Exception as exc:
        manifest.update(
            {
                "status": "FAILED",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        _write_manifest(output_dir, prefix, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
