#!/usr/bin/env python3
"""Build reviewer-requested calibration and precision addenda from frozen predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


SOURCE_COHORT = "site_a_test_patient_disjoint"
TARGET_COHORT = "site_b_primary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ece(y: np.ndarray, probability: np.ndarray, *, bins: int, strategy: str) -> float:
    y = np.asarray(y, dtype=float)
    probability = np.asarray(probability, dtype=float)
    if not len(y) or not np.isfinite(probability).all():
        return float("nan")
    if strategy == "equal_width":
        edges = np.linspace(0.0, 1.0, bins + 1)
    elif strategy == "equal_frequency":
        edges = np.quantile(probability, np.linspace(0.0, 1.0, bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
        if len(edges) < 2:
            return float(abs(y.mean() - probability.mean()))
    else:
        raise ValueError(strategy)
    assignment = np.clip(np.digitize(probability, edges[1:-1], right=True), 0, len(edges) - 2)
    value = 0.0
    for index in range(len(edges) - 1):
        mask = assignment == index
        if mask.any():
            value += mask.mean() * abs(y[mask].mean() - probability[mask].mean())
    return float(value)


def calibration_ece_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = predictions[predictions["cohort_id"].isin([SOURCE_COHORT, TARGET_COHORT])]
    for keys, group in selected.groupby(["task_id", "site_id", "cohort_id"], sort=True):
        task_id, site_id, cohort_id = keys
        y = group["y"].to_numpy(dtype=np.int8)
        for probability_type, column in (
            ("raw", "raw_probability"),
            ("calibrated", "calibrated_probability"),
        ):
            probability = group[column].to_numpy(dtype=float)
            if not np.isfinite(probability).all():
                continue
            for bins in (5, 10):
                for strategy in ("equal_width", "equal_frequency"):
                    rows.append(
                        {
                            "task_id": task_id,
                            "site_id": site_id,
                            "cohort_id": cohort_id,
                            "probability_type": probability_type,
                            "bin_n": bins,
                            "binning_strategy": strategy,
                            "n": len(group),
                            "positive_n": int(y.sum()),
                            "negative_n": int((y == 0).sum()),
                            "ece": ece(y, probability, bins=bins, strategy=strategy),
                        }
                    )
    return pd.DataFrame(rows)


def support_precision_table(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics[metrics["cohort_id"].isin([SOURCE_COHORT, TARGET_COHORT])].copy()
    selected["raw_auroc_ci_width"] = selected["raw_auroc_ci_high"] - selected["raw_auroc_ci_low"]
    selected["minimum_support_met"] = (
        selected["n_test"].ge(100)
        & selected[["positive_n", "negative_n"]].min(axis=1).ge(20)
    )
    selected["support_classification"] = np.where(
        selected["minimum_support_met"], "prespecified_support_adequate", "insufficient_support"
    )
    selected["precision_flag"] = np.select(
        [
            ~selected["minimum_support_met"],
            selected["raw_auroc_ci_width"].gt(0.20),
        ],
        ["insufficient_class_support", "wide_95pct_interval_over_0.20"],
        default="interval_width_at_or_below_0.20",
    )
    columns = [
        "task_id",
        "site_id",
        "cohort_id",
        "n_test",
        "positive_n",
        "negative_n",
        "positive_rate",
        "patient_cluster_n",
        "raw_auroc",
        "raw_auroc_ci_low",
        "raw_auroc_ci_high",
        "raw_auroc_ci_width",
        "minimum_support_met",
        "support_classification",
        "precision_flag",
    ]
    return selected[columns].sort_values(["task_id", "site_id"])


def calibration_parameter_table(metrics: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    selected = metrics[metrics["cohort_id"].isin([SOURCE_COHORT, TARGET_COHORT])].copy()
    threshold_columns = [
        "task_id",
        "platt_slope",
        "platt_intercept",
        "calibration_status",
        "calibration_failure_reason",
    ]
    threshold_part = thresholds[threshold_columns].drop_duplicates("task_id")
    selected = selected.merge(
        threshold_part,
        on="task_id",
        how="left",
        suffixes=("_evaluation", "_source_oof"),
        validate="many_to_one",
    )
    columns = [
        "task_id",
        "site_id",
        "cohort_id",
        "n_test",
        "positive_rate",
        "platt_slope",
        "platt_intercept",
        "calibration_status_source_oof",
        "calibration_failure_reason_source_oof",
        "raw_calibration_slope",
        "raw_calibration_intercept",
        "calibrated_calibration_slope",
        "calibrated_calibration_intercept",
        "raw_brier",
        "calibrated_brier",
        "raw_ece",
        "calibrated_ece",
    ]
    available = [column for column in columns if column in selected.columns]
    return selected[available].sort_values(["task_id", "site_id"])


def with_bootstrap_group(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    patient = output["public_patient_cluster_id"].astype("string")
    fallback = "sample:" + output["public_sample_id"].astype(str)
    output["bootstrap_group"] = patient.where(
        patient.notna() & patient.str.strip().ne(""), fallback
    )
    return output


def resample_site(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    groups = pd.unique(frame["bootstrap_group"])
    selected = rng.choice(groups, size=len(groups), replace=True)
    blocks = [frame[frame["bootstrap_group"].eq(group)] for group in selected]
    return pd.concat(blocks, ignore_index=True)


def median_transport_bootstrap(
    predictions: pd.DataFrame,
    *,
    task_ids: list[str],
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = with_bootstrap_group(
        predictions[
            predictions["cohort_id"].eq(SOURCE_COHORT)
            & predictions["task_id"].isin(task_ids)
        ]
    )
    target = with_bootstrap_group(
        predictions[
            predictions["cohort_id"].eq(TARGET_COHORT)
            & predictions["task_id"].isin(task_ids)
        ]
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for replicate in range(n_boot):
        sampled_source = resample_site(source, rng)
        sampled_target = resample_site(target, rng)
        source_values: list[float] = []
        target_values: list[float] = []
        for task_id in task_ids:
            a = sampled_source[sampled_source["task_id"].eq(task_id)]
            b = sampled_target[sampled_target["task_id"].eq(task_id)]
            if a["y"].nunique() < 2 or b["y"].nunique() < 2:
                continue
            source_values.append(roc_auc_score(a["y"], a["raw_probability"]))
            target_values.append(roc_auc_score(b["y"], b["raw_probability"]))
        if len(source_values) != len(task_ids) or len(target_values) != len(task_ids):
            continue
        source_median = float(np.median(source_values))
        target_median = float(np.median(target_values))
        rows.append(
            {
                "bootstrap_replicate": replicate,
                "task_n": len(task_ids),
                "source_median_raw_auroc": source_median,
                "target_median_raw_auroc": target_median,
                "delta_target_minus_source_median_raw_auroc": target_median - source_median,
            }
        )
    draws = pd.DataFrame(rows)
    observed_source = [
        roc_auc_score(
            source.loc[source["task_id"].eq(task), "y"],
            source.loc[source["task_id"].eq(task), "raw_probability"],
        )
        for task in task_ids
    ]
    observed_target = [
        roc_auc_score(
            target.loc[target["task_id"].eq(task), "y"],
            target.loc[target["task_id"].eq(task), "raw_probability"],
        )
        for task in task_ids
    ]
    summary = pd.DataFrame(
        [
            {
                "task_ids": "|".join(task_ids),
                "task_n": len(task_ids),
                "source_median_raw_auroc": np.median(observed_source),
                "source_median_raw_auroc_ci_low": draws["source_median_raw_auroc"].quantile(0.025),
                "source_median_raw_auroc_ci_high": draws["source_median_raw_auroc"].quantile(0.975),
                "target_median_raw_auroc": np.median(observed_target),
                "target_median_raw_auroc_ci_low": draws["target_median_raw_auroc"].quantile(0.025),
                "target_median_raw_auroc_ci_high": draws["target_median_raw_auroc"].quantile(0.975),
                "delta_target_minus_source_median_raw_auroc": np.median(observed_target)
                - np.median(observed_source),
                "delta_ci_low": draws["delta_target_minus_source_median_raw_auroc"].quantile(0.025),
                "delta_ci_high": draws["delta_target_minus_source_median_raw_auroc"].quantile(0.975),
                "bootstrap_requested_n": n_boot,
                "bootstrap_valid_n": len(draws),
                "bootstrap_unit": "site-level patient cluster with sample fallback",
            }
        ]
    )
    return draws, summary


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    predictions = pd.read_parquet(args.predictions)
    metrics = pd.read_csv(args.metrics)
    thresholds = pd.read_csv(args.thresholds)

    ece_table = calibration_ece_table(predictions)
    precision = support_precision_table(metrics)
    calibration = calibration_parameter_table(metrics, thresholds)
    eligible = precision[
        precision["site_id"].eq("ZD-MAST-B") & precision["minimum_support_met"]
    ]["task_id"].tolist()
    if len(eligible) != 8:
        raise ValueError(f"Expected eight prespecified supported target tasks, found {eligible}")
    draws, summary = median_transport_bootstrap(
        predictions,
        task_ids=eligible,
        n_boot=args.bootstrap,
        seed=args.seed,
    )

    ece_table.to_csv(output / "cross_platform_ece_sensitivity_v1.csv", index=False)
    precision.to_csv(output / "cross_platform_support_precision_v1.csv", index=False)
    calibration.to_csv(output / "cross_platform_calibration_parameters_v1.csv", index=False)
    draws.to_csv(output / "cross_platform_median_auroc_bootstrap_draws_v1.csv", index=False)
    summary.to_csv(output / "cross_platform_median_auroc_bootstrap_summary_v1.csv", index=False)

    platt_failures = thresholds[thresholds["calibration_status"].ne("ok")]
    report = [
        "# Revision v6 statistical addendum",
        "",
        f"- Supported target tasks: {len(eligible)} ({', '.join(eligible)}).",
        f"- Source OOF Platt guard failures: {len(platt_failures)} of {thresholds['task_id'].nunique()} tasks.",
        f"- Median AUROC bootstrap valid replicates: {int(summary.iloc[0]['bootstrap_valid_n'])}.",
        "- ECE is reported for 5 and 10 bins under equal-width and equal-frequency binning.",
        "- Target labels were used only for evaluation and uncertainty estimation.",
        "",
    ]
    (output / "revision_v6_statistical_addendum_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    manifest = {
        "analysis_id": "revision_v6_statistical_addendum",
        "status": "COMPLETE",
        "seed": args.seed,
        "bootstrap_requested_n": args.bootstrap,
        "supported_task_ids": eligible,
        "target_labels_used_for_model_fitting_or_selection": False,
        "inputs": {
            "predictions": {"path": str(args.predictions.resolve()), "sha256": sha256(args.predictions)},
            "metrics": {"path": str(args.metrics.resolve()), "sha256": sha256(args.metrics)},
            "thresholds": {"path": str(args.thresholds.resolve()), "sha256": sha256(args.thresholds)},
        },
    }
    (output / "run_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
