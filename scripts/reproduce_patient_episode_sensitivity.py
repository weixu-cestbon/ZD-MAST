#!/usr/bin/env python3
"""Reproduce Protocol B patient-disjoint and episode-first sensitivities."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from zd_mast.modeling import (
    PUBLIC_TO_LEGACY_TASK,
    compare_sensitivities_to_frozen,
    run_patient_episode_sensitivity_task,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--primary-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frozen-sensitivity", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--tasks", default=",".join(PUBLIC_TO_LEGACY_TASK))
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unknown = set(tasks) - set(PUBLIC_TO_LEGACY_TASK)
    if unknown:
        raise ValueError(f"Unknown task IDs: {sorted(unknown)}")
    primary = pd.read_csv(args.primary_metrics).set_index("task_id")
    missing = set(tasks) - set(primary.index)
    if missing:
        raise ValueError(f"Primary metrics missing task IDs: {sorted(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[pd.DataFrame] = []
    bootstraps: list[pd.DataFrame] = []
    for index, task_id in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] {task_id}", flush=True)
        params = json.loads(primary.loc[task_id, "best_hyperparameters"])
        result, bootstrap = run_patient_episode_sensitivity_task(
            args.release_root,
            task_id,
            params,
            threads=args.threads,
            n_boot=args.bootstrap,
        )
        results.append(result)
        bootstraps.append(bootstrap)

    result_table = pd.concat(results, ignore_index=True)
    result_table.to_csv(
        args.output_dir / "zd_mast_patient_episode_sensitivity_metrics_v1.0.0.csv",
        index=False,
    )
    pd.concat(bootstraps, ignore_index=True).to_csv(
        args.output_dir / "zd_mast_patient_episode_sensitivity_bootstrap_v1.0.0.csv",
        index=False,
    )

    regression_status = "NOT_REQUESTED"
    if args.frozen_sensitivity:
        regression = compare_sensitivities_to_frozen(
            result_table,
            args.frozen_sensitivity,
            args.tolerance,
        )
        regression.to_csv(
            args.output_dir / "zd_mast_patient_episode_sensitivity_regression_v1.0.0.csv",
            index=False,
        )
        regression_status = (
            "PASS_WITH_DOCUMENTED_METADATA_CORRECTION"
            if regression["status"].isin(["PASS", "CORRECTED_FROZEN_METADATA"]).all()
            and regression["status"].eq("CORRECTED_FROZEN_METADATA").any()
            else "PASS"
            if regression["status"].eq("PASS").all()
            else "FAIL"
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_id": "patient_episode_sensitivity",
        "analysis_version": "v2026.07.17.3",
        "public_analysis_version": "analysis-v1.0.0",
        "protocol": "current_workflow_protocol_b",
        "tasks": tasks,
        "threads": args.threads,
        "bootstrap": args.bootstrap,
        "hyperparameters_fixed_from_primary": True,
        "test_used_for_tuning": False,
        "regression_status": regression_status,
    }
    (args.output_dir / "zd_mast_patient_episode_sensitivity_manifest_v1.0.0.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"result_regression_status={regression_status}", flush=True)
    if regression_status == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
