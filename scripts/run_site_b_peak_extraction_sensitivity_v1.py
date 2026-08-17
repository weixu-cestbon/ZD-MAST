#!/usr/bin/env python3
"""Prespecified Site B peak-extraction sensitivity and source-only evaluation.

The variants are defined without AST labels. One unchanged Site A model per
task is evaluated against every Site B representation. Target labels are used
only after feature construction for final performance estimation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from zd_mast.cross_platform import (  # noqa: E402
    DEFAULT_SEED,
    SITE_A,
    SITE_B,
    TASK_IDS,
    _bootstrap_groups,
    _resample_cluster_indices,
    build_task_cohorts,
    classify_support,
    deterministic_seed,
    load_analysis_inputs,
)
from zd_mast.features import DensePeakParameters, dense_profile_peak_presence  # noqa: E402
from zd_mast.modeling import fit_lightgbm, matrix_rows, predict  # noqa: E402


VARIANTS = {
    "public_rebuild_default": DensePeakParameters(),
    "baseline_sigma_100": DensePeakParameters(baseline_sigma_points=100.0),
    "baseline_sigma_200": DensePeakParameters(baseline_sigma_points=200.0),
    "prominence_2p5": DensePeakParameters(prominence_noise_multiplier=2.5),
    "prominence_4p0": DensePeakParameters(prominence_noise_multiplier=4.0),
    "distance_3": DensePeakParameters(min_peak_distance_points=3),
    "distance_7": DensePeakParameters(min_peak_distance_points=7),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_release_dirs(release_root: Path) -> tuple[Path, Path]:
    feature = sorted(release_root.glob("feature-release-*"))
    peak = sorted(release_root.glob("peak-table-release-*"))
    if len(feature) != 1 or len(peak) != 1:
        raise ValueError("Expected one feature release and one peak-table release")
    return feature[0], peak[0]


def read_profile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, delimiter="\t", skiprows=1, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected two-column profile: {path}")
    return values[:, 0], values[:, 1]


def build_site_b_variant_matrices(
    release_root: Path,
    output_dir: Path,
    workers: int,
) -> tuple[dict[str, Path], pd.DataFrame]:
    feature, peak = resolve_release_dirs(release_root)
    spectrum = pd.read_csv(feature / "zd_mast_spectrum_metadata_public_v1.0.0.csv")
    spectrum = spectrum.loc[spectrum["site_id"].eq(SITE_B)].sort_values(
        "feature_row"
    ).reset_index(drop=True)
    sample = pd.read_csv(feature / "zd_mast_sample_metadata_public_v1.0.0.csv")
    sample = sample.loc[sample["site_id"].eq(SITE_B)].sort_values(
        "feature_row"
    ).reset_index(drop=True)
    if not np.array_equal(
        sample["feature_row"].to_numpy(dtype=int), np.arange(len(sample))
    ):
        raise ValueError("Site B sample feature rows are not contiguous")
    sample_row = sample.set_index("public_sample_id")["feature_row"].astype(int)
    table_dir = peak / "zd_mast_b_open_peak_tables"
    paths = [table_dir / f"{value}.tsv" for value in spectrum["public_spectrum_id"]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Site B profiles: {missing[:5]}")

    matrix_dir = output_dir / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    final_paths = {
        name: matrix_dir / f"site_b_peak_presence6000_{name}.npy"
        for name in VARIANTS
    }
    if all(path.is_file() for path in final_paths.values()):
        matrices = {name: np.load(path, mmap_mode="r") for name, path in final_paths.items()}
    else:
        partial_paths = {
            name: path.with_name(path.stem + ".partial.npy")
            for name, path in final_paths.items()
        }
        for path in partial_paths.values():
            if path.exists():
                path.unlink()
        matrices = {
            name: np.lib.format.open_memmap(
                partial_paths[name],
                mode="w+",
                dtype=np.uint8,
                shape=(len(sample), 6000),
            )
            for name in VARIANTS
        }
        for matrix in matrices.values():
            matrix[:] = 0

        def vectorize(path: Path) -> dict[str, np.ndarray]:
            mz, intensity = read_profile(path)
            return {
                name: dense_profile_peak_presence(
                    mz, intensity, parameters=parameters
                )
                for name, parameters in VARIANTS.items()
            }

        counts = np.zeros(len(sample), dtype=np.uint32)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for spectrum_index, vectors in enumerate(executor.map(vectorize, paths)):
                public_sample_id = str(spectrum.loc[spectrum_index, "public_sample_id"])
                row = int(sample_row.loc[public_sample_id])
                for name, vector in vectors.items():
                    matrices[name][row] |= vector
                counts[row] += 1
        if np.any(counts == 0):
            raise ValueError(
                f"Site B samples without profiles: {int(np.count_nonzero(counts == 0))}"
            )
        for name, matrix in matrices.items():
            matrix.flush()
            os.replace(partial_paths[name], final_paths[name])
        matrices = {name: np.load(path, mmap_mode="r") for name, path in final_paths.items()}

    released_path = feature / "zd_mast_b_sample_level_peak_presence6000_v1.0.0.npy"
    released = np.load(released_path, mmap_mode="r")
    rows = []
    frozen = np.asarray(released)
    comparison_matrices = {"frozen_release": released, **matrices}
    comparison_paths = {"frozen_release": released_path, **final_paths}
    for name, matrix in comparison_matrices.items():
        values = np.asarray(matrix)
        if values.shape != released.shape:
            raise ValueError(f"Variant {name} has unexpected shape {values.shape}")
        different_rows_from_frozen = int(np.count_nonzero(np.any(values != frozen, axis=1)))
        union = np.logical_or(values, frozen).sum(axis=1)
        intersection = np.logical_and(values, frozen).sum(axis=1)
        jaccard = np.divide(
            intersection,
            union,
            out=np.ones_like(intersection, dtype=float),
            where=union > 0,
        )
        rows.append(
            {
                "variant": name,
                **(
                    asdict(VARIANTS[name])
                    if name in VARIANTS
                    else {
                        "savgol_window": np.nan,
                        "savgol_polyorder": np.nan,
                        "baseline_sigma_points": np.nan,
                        "prominence_noise_multiplier": np.nan,
                        "min_peak_distance_points": np.nan,
                    }
                ),
                "sample_n": len(values),
                "median_detected_bins": float(np.median(values.sum(axis=1))),
                "q25_detected_bins": float(np.quantile(values.sum(axis=1), 0.25)),
                "q75_detected_bins": float(np.quantile(values.sum(axis=1), 0.75)),
                "different_rows_from_frozen": different_rows_from_frozen,
                "median_jaccard_with_frozen": float(np.median(jaccard)),
                "frozen_release_exact_match": bool(np.array_equal(values, released)),
                "matrix_sha256": sha256(comparison_paths[name]),
            }
        )
    return comparison_paths, pd.DataFrame(rows)


def bootstrap_raw_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    if n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    draws_auroc = []
    draws_auprc = []
    for _ in range(n_boot):
        index = _resample_cluster_indices(groups, rng)
        if np.unique(y[index]).size < 2:
            continue
        draws_auroc.append(float(roc_auc_score(y[index], probability[index])))
        draws_auprc.append(float(average_precision_score(y[index], probability[index])))
    output: dict[str, float] = {"bootstrap_valid_n": float(len(draws_auroc))}
    for name, values in (("raw_auroc", draws_auroc), ("raw_auprc", draws_auprc)):
        array = np.asarray(values, dtype=float)
        output[f"{name}_ci_low"] = float(np.quantile(array, 0.025))
        output[f"{name}_ci_high"] = float(np.quantile(array, 0.975))
    return output


def evaluate_variants(
    release_root: Path,
    target_date_table: Path,
    primary_metrics: Path,
    matrix_paths: dict[str, Path],
    *,
    threads: int,
    bootstrap_count: int,
    seed: int,
) -> pd.DataFrame:
    inputs = load_analysis_inputs(release_root, target_date_table)
    existing = pd.read_csv(primary_metrics)
    parameters = {}
    for task_id, group in existing.groupby("task_id"):
        values = group["best_hyperparameters"].dropna().unique()
        if len(values) != 1:
            raise ValueError(f"{task_id}: expected one frozen hyperparameter set")
        parameters[str(task_id)] = json.loads(values[0])
    rows = []
    for task_id in TASK_IDS:
        cohorts = build_task_cohorts(inputs, task_id)
        params = parameters[task_id]
        task_seed = deterministic_seed(seed, task_id)
        x_development, y_development = matrix_rows(
            inputs.matrices[SITE_A], cohorts.source_development
        )
        model = fit_lightgbm(
            params, x_development, y_development, task_seed + 20_000, threads
        )
        for variant, path in matrix_paths.items():
            matrix = np.load(path, mmap_mode="r")
            x_target, y_target = matrix_rows(matrix, cohorts.target_primary)
            probability = predict(model, x_target)
            positive_n = int(y_target.sum())
            negative_n = int(y_target.size - positive_n)
            support = classify_support(len(y_target), positive_n, negative_n)
            groups, group_source = _bootstrap_groups(cohorts.target_primary)
            row = {
                "analysis_id": "site_b_peak_extraction_sensitivity",
                "task_id": task_id,
                "variant": variant,
                "site_id": SITE_B,
                "cohort_id": "site_b_primary",
                "n_test": len(y_target),
                "positive_n": positive_n,
                "negative_n": negative_n,
                "positive_rate": float(y_target.mean()),
                "support_status": support.status,
                "insufficient_reason": support.reason,
                "bootstrap_group_source": group_source,
                "raw_auroc": float(roc_auc_score(y_target, probability)),
                "raw_auprc": float(average_precision_score(y_target, probability)),
                "auprc_baseline": float(y_target.mean()),
                "raw_auprc_lift": float(
                    average_precision_score(y_target, probability) / y_target.mean()
                ),
                "same_source_model_across_variants": True,
                "target_labels_used_for_feature_selection": False,
                "target_labels_used_for_model_fitting": False,
                "best_hyperparameters": json.dumps(params, sort_keys=True),
            }
            if support.eligible_for_discrimination:
                row.update(
                    bootstrap_raw_metrics(
                        y_target,
                        probability,
                        groups,
                        n_boot=bootstrap_count,
                        seed=deterministic_seed(
                            task_seed, variant, "peak_extraction_bootstrap"
                        ),
                    )
                )
            rows.append(row)
    return pd.DataFrame(rows)


def write_report(metrics: pd.DataFrame, qc: pd.DataFrame, output: Path) -> None:
    supported = metrics[metrics["support_status"].eq("adequate")].copy()
    summary = (
        supported.groupby("variant", as_index=False)
        .agg(
            task_n=("task_id", "nunique"),
            median_raw_auroc=("raw_auroc", "median"),
            median_raw_auprc=("raw_auprc", "median"),
            median_raw_auprc_lift=("raw_auprc_lift", "median"),
        )
        .merge(
            qc[["variant", "median_detected_bins", "median_jaccard_with_frozen"]],
            on="variant",
            how="left",
        )
    )
    summary.to_csv(output / "site_b_peak_extraction_sensitivity_summary_v1.csv", index=False)
    lines = [
        "# Site B peak-extraction sensitivity",
        "",
        "All variants were specified without AST labels. One unchanged Site A model per task was applied to each target representation; target labels were used only for final evaluation.",
        "",
        "| variant | tasks | median AUROC | median AUPRC | median AUPRC lift | median detected bins | median Jaccard vs frozen |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.task_n} | {row.median_raw_auroc:.3f} | "
            f"{row.median_raw_auprc:.3f} | {row.median_raw_auprc_lift:.3f} | "
            f"{row.median_detected_bins:.1f} | {row.median_jaccard_with_frozen:.3f} |"
        )
    (output / "site_b_peak_extraction_sensitivity_report_v1.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--target-date-table", type=Path, required=True)
    parser.add_argument("--primary-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--bootstrap-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths, qc = build_site_b_variant_matrices(
        args.release_root, args.output_dir, args.workers
    )
    qc.to_csv(
        args.output_dir / "site_b_peak_extraction_feature_qc_v1.csv", index=False
    )
    metrics = evaluate_variants(
        args.release_root,
        args.target_date_table,
        args.primary_metrics,
        paths,
        threads=args.threads,
        bootstrap_count=args.bootstrap_count,
        seed=args.seed,
    )
    metrics.to_csv(
        args.output_dir / "site_b_peak_extraction_sensitivity_metrics_v1.csv",
        index=False,
    )
    write_report(metrics, qc, args.output_dir)
    manifest = {
        "analysis_id": "site_b_peak_extraction_sensitivity",
        "variants": {name: asdict(parameters) for name, parameters in VARIANTS.items()},
        "feature_variants_selected_without_ast_labels": True,
        "frozen_release_matrix_preserved_as_primary_reference": True,
        "public_default_rebuild_reported_as_separate_sensitivity": True,
        "source_model_unchanged_across_target_variants": True,
        "bootstrap_count": args.bootstrap_count,
        "seed": args.seed,
        "input_release_root": str(args.release_root),
        "target_date_table": str(args.target_date_table),
        "primary_metrics": str(args.primary_metrics),
    }
    (args.output_dir / "run_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
