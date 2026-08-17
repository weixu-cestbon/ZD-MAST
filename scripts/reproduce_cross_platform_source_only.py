#!/usr/bin/env python3
"""Reproduce the frozen Site A to Site B peak-presence stress test from public data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


SEED = 20260803
SHORT_TO_LEGACY = {
    "sa_oxa": "sa_oxacillin",
    "sa_lvx": "sa_levofloxacin",
    "sa_gen": "sa_gentamicin",
    "kp_fep": "kp_cefepime",
    "kp_cro": "kp_ceftriaxone",
    "kp_caz": "kp_ceftazidime",
    "kp_cip": "kp_ciprofloxacin",
    "ec_cro": "ec_ceftriaxone",
    "ec_cip": "ec_ciprofloxacin",
    "ec_fep": "ec_cefepime",
}


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    total = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        include = (probability >= left) & (
            probability <= right if index == bins - 1 else probability < right
        )
        if include.any():
            total += float(include.mean()) * abs(
                float(y[include].mean()) - float(probability[include].mean())
            )
    return total


def feature_release(root: Path) -> Path:
    candidates = sorted(root.glob("feature-release-*"))
    if len(candidates) != 1:
        raise ValueError("release root must contain exactly one feature release")
    return candidates[0]


def fit_source_models(
    feature: Path,
    labels: pd.DataFrame,
    metadata: pd.DataFrame,
    splits: pd.DataFrame,
    groups: pd.DataFrame,
) -> dict[str, tuple[lgb.LGBMClassifier, LogisticRegression, dict[str, int]]]:
    matrix = np.load(
        feature / "zd_mast_a_sample_level_peak_presence6000_v1.0.0.npy",
        mmap_mode="r",
    )
    row_map = metadata.loc[metadata["site_id"].eq("ZD-MAST-A")].set_index(
        "public_sample_id"
    )["feature_row"].astype(int)
    strict = splits.loc[
        splits["analysis_id"].eq("cross_platform_source_model")
        & splits["protocol"].eq("source_temporal_latest20_patient_disjoint")
        & splits["site_id"].eq("ZD-MAST-A")
    ].copy()
    group_map = groups.set_index(["public_sample_id", "task_id"])[
        "public_patient_cluster_id"
    ]
    source = labels.loc[
        labels["site_id"].eq("ZD-MAST-A")
        & labels["year"].le(2025)
        & labels["binary_s_vs_ir"].isin([0, 1])
    ].copy()
    source = source.merge(
        strict[["task_id", "public_sample_id", "split"]],
        on=["task_id", "public_sample_id"],
        how="inner",
        validate="one_to_one",
    )
    source["patient"] = [
        group_map.get((sample, task), pd.NA)
        for sample, task in zip(source["public_sample_id"], source["task_id"], strict=True)
    ]
    fitted: dict[str, tuple[lgb.LGBMClassifier, LogisticRegression, dict[str, int]]] = {}
    for task_id in sorted(SHORT_TO_LEGACY):
        task = source.loc[source["task_id"].eq(task_id)].copy()
        validation = task.loc[task["split"].eq("validation")].copy()
        train = task.loc[task["split"].eq("train")].copy()
        if min(train["binary_s_vs_ir"].value_counts().reindex([0, 1], fill_value=0)) < 20:
            raise ValueError(f"{task_id}: source train min_class <20")
        if min(validation["binary_s_vs_ir"].value_counts().reindex([0, 1], fill_value=0)) < 10:
            raise ValueError(f"{task_id}: source validation min_class <10")
        train_rows = row_map.loc[train["public_sample_id"]].to_numpy(dtype=np.int64)
        validation_rows = row_map.loc[validation["public_sample_id"]].to_numpy(dtype=np.int64)
        y_train = train["binary_s_vs_ir"].to_numpy(dtype=np.int8)
        y_validation = validation["binary_s_vs_ir"].to_numpy(dtype=np.int8)
        positive_weight = float((y_train == 0).sum() / (y_train == 1).sum())
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=positive_weight,
            random_state=SEED,
            n_jobs=8,
            verbosity=-1,
        )
        model.fit(matrix[train_rows], y_train)
        raw_validation = model.predict_proba(matrix[validation_rows])[:, 1]
        platt = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs", random_state=SEED)
        platt.fit(raw_validation.reshape(-1, 1), y_validation)
        fitted[task_id] = (
            model,
            platt,
            {"n_train": int(len(train)), "n_validation": int(len(validation))},
        )
    return fitted


def evaluate_target(
    feature: Path,
    labels: pd.DataFrame,
    metadata: pd.DataFrame,
    fitted: dict[str, tuple[lgb.LGBMClassifier, LogisticRegression, dict[str, int]]],
) -> pd.DataFrame:
    matrix = np.load(
        feature / "zd_mast_b_sample_level_peak_presence6000_v1.0.0.npy",
        mmap_mode="r",
    )
    row_map = metadata.loc[metadata["site_id"].eq("ZD-MAST-B")].set_index(
        "public_sample_id"
    )["feature_row"].astype(int)
    target = labels.loc[
        labels["site_id"].eq("ZD-MAST-B")
        & labels["year"].eq(2026)
        & labels["binary_s_vs_ir"].isin([0, 1])
    ].copy()
    rows: list[dict[str, object]] = []
    for task_id, (model, platt, source_counts) in fitted.items():
        task = target.loc[target["task_id"].eq(task_id)].drop_duplicates(
            "public_sample_id", keep="last"
        )
        y = task["binary_s_vs_ir"].to_numpy(dtype=np.int8)
        feature_rows = row_map.loc[task["public_sample_id"]].to_numpy(dtype=np.int64)
        raw = model.predict_proba(matrix[feature_rows])[:, 1]
        probability = platt.predict_proba(raw.reshape(-1, 1))[:, 1]
        positive_n, negative_n = int(y.sum()), int((y == 0).sum())
        row: dict[str, object] = {
            "task_id": SHORT_TO_LEGACY[task_id],
            "public_task_id": task_id,
            "model": "lightgbm",
            "target_subset": "primary_contemporaneous_2026",
            "n_test": int(len(y)),
            "positive_n": positive_n,
            "negative_n": negative_n,
            "positive_rate": float(y.mean()),
            "target_labels_used_for_training_or_selection": False,
            **source_counts,
        }
        if len(y) < 100 or min(positive_n, negative_n) < 20:
            row.update(
                {
                    "status": "exploratory_or_insufficient",
                    "insufficient_reason": "n<100 or min_class<20",
                }
            )
        else:
            baseline = float(y.mean())
            auprc = float(average_precision_score(y, probability))
            row.update(
                {
                    "status": "ok",
                    "insufficient_reason": "",
                    "AUROC": float(roc_auc_score(y, probability)),
                    "AUPRC": auprc,
                    "AUPRC_baseline": baseline,
                    "AUPRC_lift": auprc / baseline,
                    "Brier": float(brier_score_loss(y, probability)),
                    "ECE": expected_calibration_error(y, probability),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("public_task_id").reset_index(drop=True)


def compare_frozen(reproduced: pd.DataFrame, frozen_path: Path) -> pd.DataFrame:
    frozen = pd.read_csv(frozen_path)
    frozen = frozen.loc[
        frozen["model"].eq("lightgbm")
        & frozen["target_subset"].eq("primary_contemporaneous_2026")
    ].set_index("task_id")
    reproduced = reproduced.set_index("task_id")
    rows: list[dict[str, object]] = []
    integer_fields = ["n_test", "positive_n", "negative_n"]
    metric_fields = ["AUROC", "AUPRC", "AUPRC_baseline", "AUPRC_lift", "Brier", "ECE"]
    for task_id in frozen.index:
        for field in integer_fields + metric_fields:
            frozen_value = frozen.at[task_id, field] if field in frozen else np.nan
            reproduced_value = reproduced.at[task_id, field] if field in reproduced else np.nan
            both_missing = pd.isna(frozen_value) and pd.isna(reproduced_value)
            tolerance = 0.0 if field in integer_fields else 1e-10
            difference = (
                0.0
                if both_missing
                else abs(float(frozen_value) - float(reproduced_value))
                if pd.notna(frozen_value) and pd.notna(reproduced_value)
                else np.inf
            )
            rows.append(
                {
                    "task_id": task_id,
                    "field": field,
                    "frozen_value": frozen_value,
                    "reproduced_value": reproduced_value,
                    "absolute_difference": difference,
                    "tolerance": tolerance,
                    "status": "PASS" if difference <= tolerance else "FAIL",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--frozen-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, output = args.release_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    output.mkdir(parents=True)
    feature = feature_release(root)
    labels = pd.read_parquet(feature / "zd_mast_ast_labels_historical_v1.0.0.parquet")
    metadata = pd.read_csv(feature / "zd_mast_sample_metadata_public_v1.0.0.csv")
    splits = pd.read_csv(feature / "zd_mast_split_assignments_public_v1.0.0.csv")
    groups = pd.read_parquet(feature / "zd_mast_patient_episode_groups_public_v1.0.0.parquet")
    fitted = fit_source_models(feature, labels, metadata, splits, groups)
    results = evaluate_target(feature, labels, metadata, fitted)
    regression = compare_frozen(results, args.frozen_results)
    results.to_csv(output / "cross_platform_source_only_reproduced_results_v1.csv", index=False)
    regression.to_csv(output / "cross_platform_source_only_result_regression_v1.csv", index=False)
    summary = {
        "status": "PASS" if regression["status"].eq("PASS").all() else "FAIL",
        "task_n": int(len(results)),
        "field_n": int(len(regression)),
        "pass_n": int(regression["status"].eq("PASS").sum()),
        "fail_n": int(regression["status"].eq("FAIL").sum()),
        "target_labels_used_for_training_or_selection": False,
        "feature_representation": "peak_presence6000",
        "source": "ZD-MAST-A development through 2025",
        "target": "ZD-MAST-B 2026",
    }
    (output / "run_manifest_v1.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
