#!/usr/bin/env python3
"""Run the prespecified R2 size-matched training-history sensitivity."""

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

from zd_mast.cross_platform import TASK_IDS, validate_task_ids
from zd_mast.training_history import build_history_cohorts, load_training_history_inputs
from zd_mast.training_history_sensitivity import (
    ANALYSIS_ID,
    DEFAULT_SEED,
    REQUIRED_LEARNING_FRACTIONS,
    paired_size_matched_deltas,
    run_size_matched_task,
    summarize_paired_deltas,
    validate_learning_fractions,
)


MINIMUM_FULL_REPEATS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--release-root",
        required=True,
        type=Path,
        help="Feature-release root containing Site A intensity6000 and split files.",
    )
    parser.add_argument(
        "--temporal-bridge",
        required=True,
        type=Path,
        help="Private exact-date Site A temporal bridge used by the existing history analysis.",
    )
    parser.add_argument(
        "--primary-metrics",
        required=True,
        type=Path,
        help="Frozen primary task metrics containing best_hyperparameters JSON.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New output directory; existing directories are never overwritten.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_IDS),
        help="Frozen task IDs to analyze.",
    )
    parser.add_argument(
        "--learning-fractions",
        nargs="+",
        type=float,
        default=list(REQUIRED_LEARNING_FRACTIONS),
        help="Prespecified nested learning-curve fractions.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=MINIMUM_FULL_REPEATS,
        help="Repeated deterministic source-sampling runs; must be at least 20.",
    )
    parser.add_argument(
        "--minimum-train-class-n",
        type=int,
        default=10,
        help="Minimum sampled count in each class required to fit a cell.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--write-predictions",
        action="store_true",
        help="Write patient-level predictions to a clearly private-review parquet file.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(
    output: Path,
    metrics: pd.DataFrame,
    delta_summary: pd.DataFrame,
    *,
    repeats: int,
    fractions: tuple[float, ...],
) -> None:
    successful = metrics.loc[metrics["status"].eq("ok")].copy()
    performance = (
        successful.groupby(["training_regime", "learning_fraction"])[
            ["raw_auroc", "raw_auprc"]
        ]
        .median()
        .reset_index()
    )
    pooled_pre = (
        delta_summary.loc[
            delta_summary["comparison_id"].eq(
                "pooled_era_balanced_minus_pre_marker_only"
            )
            & delta_summary["metric"].eq("raw_auroc")
            & delta_summary["learning_fraction"].eq(1.0)
        ].copy()
        if not delta_summary.empty
        else delta_summary.copy()
    )
    lines = [
        "# Size-matched training-history sensitivity v2",
        "",
        "This reviewer-R2 sensitivity keeps one identical patient-disjoint future test ",
        "cohort for every training regime. Within each task and learning fraction, ",
        "current-only, pre-marker-only, and era-balanced pooled training use identical ",
        "negative and positive sample counts. Pooled samples are divided as evenly as ",
        "possible between eras within each class.",
        "",
        f"- Repeats: {repeats}",
        f"- Learning fractions: {', '.join(f'{value:.2f}' for value in fractions)}",
        "- Hyperparameters: frozen task-specific values read from the primary metrics table",
        "- Test labels used for sampling or tuning: no",
        "- Calibration or threshold selection: none",
        "- Reported performance: raw AUROC and raw AUPRC",
        "",
        "## Median raw performance across task-repeat cells",
        "",
    ]
    if performance.empty:
        lines.append("No cells met the prespecified training support requirement.")
    else:
        for row in performance.itertuples(index=False):
            lines.append(
                f"- {row.training_regime}, fraction={row.learning_fraction:.2f}: "
                f"AUROC {row.raw_auroc:.3f}; AUPRC {row.raw_auprc:.3f}"
            )
    lines.extend(
        [
            "",
            "## Full-fraction pooled-minus-pre paired delta",
            "",
        ]
    )
    if pooled_pre.empty:
        lines.append("No full-fraction pooled-versus-pre comparison was estimable.")
    else:
        for row in pooled_pre.sort_values("task_id").itertuples(index=False):
            lines.append(
                f"- {row.task_id}: median delta AUROC {row.delta_median:+.3f} "
                f"(repeat-distribution q2.5-q97.5 "
                f"{row.repeat_distribution_q025:+.3f} to "
                f"{row.repeat_distribution_q975:+.3f})"
            )
    lines.extend(
        [
            "",
            "These repeat-distribution quantiles describe sensitivity to source sampling; ",
            "they are not inferential confidence intervals. The analysis estimates a ",
            "training-history association under matched size and class composition and ",
            "does not identify a causal workflow-card effect.",
        ]
    )
    (output / "training_history_size_matched_report_v2.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if args.repeats < MINIMUM_FULL_REPEATS:
        raise ValueError(
            f"Full analysis requires at least {MINIMUM_FULL_REPEATS} repeats"
        )
    if args.minimum_train_class_n < 1:
        raise ValueError("--minimum-train-class-n must be at least 1")
    fractions = validate_learning_fractions(args.learning_fractions)
    task_ids = validate_task_ids(args.tasks)

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output}")
    output.mkdir(parents=True)

    release_root = args.release_root.resolve()
    temporal_bridge = args.temporal_bridge.resolve()
    primary_metrics = args.primary_metrics.resolve()
    inputs = load_training_history_inputs(
        release_root,
        temporal_bridge,
        primary_metrics,
    )

    metric_parts: list[pd.DataFrame] = []
    sampling_parts: list[pd.DataFrame] = []
    cap_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    cohort_rows: list[dict[str, object]] = []
    for task_index, task_id in enumerate(task_ids, start=1):
        print(f"[{task_index}/{len(task_ids)}] {task_id}", flush=True)
        cohorts = build_history_cohorts(inputs, task_id)
        metrics, sampling, caps, predictions = run_size_matched_task(
            inputs,
            cohorts,
            learning_fractions=fractions,
            repeats=args.repeats,
            threads=args.threads,
            seed=args.seed,
            minimum_train_class_n=args.minimum_train_class_n,
            return_predictions=args.write_predictions,
        )
        metric_parts.append(metrics)
        sampling_parts.append(sampling)
        cap_parts.append(caps)
        if not predictions.empty:
            prediction_parts.append(predictions)
        cohort_rows.append(
            {
                "task_id": task_id,
                "pre_marker_available_n": len(
                    cohorts.development_by_regime["pre_marker_history_only"]
                ),
                "current_workflow_available_n": len(
                    cohorts.development_by_regime["current_workflow_only"]
                ),
                "pre_marker_missing_patient_excluded_n": int(
                    cohorts.development_by_regime["pre_marker_history_only"]
                    ["public_patient_cluster_id"]
                    .astype("string")
                    .str.strip()
                    .eq("")
                    .fillna(True)
                    .sum()
                ),
                "current_workflow_missing_patient_excluded_n": int(
                    cohorts.development_by_regime["current_workflow_only"]
                    ["public_patient_cluster_id"]
                    .astype("string")
                    .str.strip()
                    .eq("")
                    .fillna(True)
                    .sum()
                ),
                "fixed_patient_disjoint_future_test_n": len(
                    cohorts.test_patient_disjoint_common
                ),
                "future_test_positive_n": int(
                    cohorts.test_patient_disjoint_common["y"].sum()
                ),
                "future_test_negative_n": int(
                    cohorts.test_patient_disjoint_common["y"].eq(0).sum()
                ),
            }
        )

    metrics = pd.concat(metric_parts, ignore_index=True)
    sampling = pd.concat(sampling_parts, ignore_index=True)
    caps = pd.concat(cap_parts, ignore_index=True)
    deltas = paired_size_matched_deltas(metrics)
    delta_summary = summarize_paired_deltas(deltas)
    cohort_counts = pd.DataFrame(cohort_rows)

    metrics.to_csv(
        output / "zd_mast_training_history_size_matched_metrics_v2.csv", index=False
    )
    sampling.to_csv(
        output / "zd_mast_training_history_size_matched_sampling_audit_v2.csv",
        index=False,
    )
    caps.to_csv(
        output / "zd_mast_training_history_size_matched_class_caps_v2.csv",
        index=False,
    )
    cohort_counts.to_csv(
        output / "zd_mast_training_history_size_matched_cohort_counts_v2.csv",
        index=False,
    )
    deltas.to_csv(
        output / "zd_mast_training_history_size_matched_paired_deltas_v2.csv",
        index=False,
    )
    delta_summary.to_csv(
        output / "zd_mast_training_history_size_matched_delta_summary_v2.csv",
        index=False,
    )
    if args.write_predictions:
        if not prediction_parts:
            raise ValueError("Prediction output requested but no model cell was estimable")
        pd.concat(prediction_parts, ignore_index=True).to_parquet(
            output
            / "zd_mast_training_history_size_matched_predictions_private_review_v2.parquet",
            index=False,
        )

    _write_report(
        output,
        metrics,
        delta_summary,
        repeats=args.repeats,
        fractions=fractions,
    )
    files_before_manifest = sorted(path for path in output.iterdir() if path.is_file())
    manifest = {
        "analysis_id": ANALYSIS_ID,
        "analysis_version": "reviewer-r2-size-matched-v2",
        "parent_analysis": "same_test_training_history",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": list(task_ids),
        "training_regimes": [
            "current_only",
            "pre_marker_only",
            "pooled_era_balanced",
        ],
        "learning_fractions": list(fractions),
        "repeats": args.repeats,
        "minimum_train_class_n": args.minimum_train_class_n,
        "seed": args.seed,
        "threads": args.threads,
        "model": "lightgbm",
        "model_parameters": "frozen task-specific best_hyperparameters from primary metrics",
        "endpoint": "historical_S_vs_IR",
        "feature_representation": "intensity6000",
        "fixed_test": "common patient-disjoint 2026-03-01 through 2026-06-09",
        "sampling_rule": (
            "common per-class cap=min(pre-marker available,current-workflow available); "
            "nested deterministic prefixes; pooled class counts split across eras within one"
        ),
        "test_labels_used_for_sampling_or_tuning": False,
        "calibration_applied": False,
        "threshold_selection_applied": False,
        "inputs": {
            "release_root": str(release_root),
            "temporal_bridge": {
                "path": str(temporal_bridge),
                "sha256": sha256_file(temporal_bridge),
            },
            "primary_metrics": {
                "path": str(primary_metrics),
                "sha256": sha256_file(primary_metrics),
            },
        },
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "lightgbm",
                "pyarrow",
            )
        },
        "outputs": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in files_before_manifest
        },
    }
    (output / "run_manifest_v2.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    printable = (
        delta_summary.loc[
            delta_summary["comparison_id"].eq(
                "pooled_era_balanced_minus_pre_marker_only"
            )
            & delta_summary["metric"].eq("raw_auroc")
            & delta_summary["learning_fraction"].eq(1.0),
            [
                "task_id",
                "delta_median",
                "repeat_distribution_q025",
                "repeat_distribution_q975",
            ],
        ]
        if not delta_summary.empty
        else delta_summary
    )
    print(printable.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
