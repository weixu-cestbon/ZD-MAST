#!/usr/bin/env python3
"""Reproduce the ten-task ZD-MAST primary temporal analysis from public IDs."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from zd_mast.modeling import PUBLIC_TO_LEGACY_TASK, compare_to_frozen, run_primary_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--tasks", default=",".join(PUBLIC_TO_LEGACY_TASK))
    parser.add_argument("--frozen-metrics", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument(
        "--write-predictions",
        action="store_true",
        help="Write public-ID sample predictions for private review; disabled by default.",
    )
    args = parser.parse_args()

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unknown = set(tasks) - set(PUBLIC_TO_LEGACY_TASK)
    if unknown:
        raise ValueError(f"Unknown task IDs: {sorted(unknown)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    tuning = []
    bootstrap = []
    predictions = []
    for index, task_id in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] {task_id}", flush=True)
        result, task_tuning, task_bootstrap, task_predictions = run_primary_task(
            args.release_root,
            task_id,
            threads=args.threads,
            n_boot=args.bootstrap,
        )
        results.append(result)
        tuning.append(task_tuning)
        bootstrap.append(task_bootstrap)
        predictions.append(task_predictions)

    result_table = pd.DataFrame(results)
    result_table.to_csv(args.output_dir / "zd_mast_primary_protocol_b_reproduced_metrics_v1.0.0.csv", index=False)
    pd.concat(tuning, ignore_index=True).to_csv(
        args.output_dir / "zd_mast_primary_protocol_b_tuning_audit_v1.0.0.csv", index=False
    )
    pd.concat(bootstrap, ignore_index=True).to_csv(
        args.output_dir / "zd_mast_primary_protocol_b_bootstrap_intervals_v1.0.0.csv", index=False
    )
    if args.write_predictions:
        pd.concat(predictions, ignore_index=True).to_parquet(
            args.output_dir / "zd_mast_primary_protocol_b_predictions_private_review_v1.0.0.parquet",
            index=False,
        )

    regression_status = "NOT_REQUESTED"
    if args.frozen_metrics:
        regression = compare_to_frozen(result_table, args.frozen_metrics, args.tolerance)
        regression.to_csv(
            args.output_dir / "zd_mast_primary_protocol_b_result_regression_v1.0.0.csv",
            index=False,
        )
        regression_status = "PASS" if regression["status"].eq("PASS").all() else "FAIL"

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_version": "v2026.07.17.3",
        "public_analysis_version": "analysis-v1.0.0",
        "protocol": "current_workflow_protocol_b",
        "tasks": tasks,
        "threads": args.threads,
        "bootstrap": args.bootstrap,
        "test_used_for_tuning": False,
        "regression_status": regression_status,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "pyarrow")
        },
    }
    (args.output_dir / "zd_mast_primary_protocol_b_run_manifest_v1.0.0.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"result_regression_status={regression_status}", flush=True)
    if regression_status == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
