#!/usr/bin/env python3
"""Build the reviewer-R1 performance-independent task-universe support audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from zd_mast.task_universe import SupportThresholds, run_task_universe_audit


DEFAULT_CONFIG = (
    Path(__file__).parents[1]
    / "configs"
    / "analyses"
    / "task_universe_selection_audit_v1.yaml"
)


def read_table(path: Path) -> pd.DataFrame:
    """Read a supported tabular input without mutating it."""

    suffixes = "".join(path.suffixes).casefold()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path)
    if suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported table format: {path}")


def load_config(path: Path) -> dict[str, Any]:
    """Load the JSON-compatible YAML config without adding a YAML dependency."""

    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            "The task-universe config must remain JSON-compatible YAML so the "
            "public tool has no undeclared PyYAML dependency"
        ) from exc
    if not isinstance(config, dict):
        raise ValueError("Task-universe config root must be an object")
    return config


def _optional_int(cli_value: int | None, config_value: object) -> int | None:
    if cli_value is not None:
        return cli_value
    if config_value is None or str(config_value).strip() == "":
        return None
    return int(config_value)


def _sites(cli_sites: str | None, configured: object) -> list[str] | None:
    if cli_sites is not None:
        values = [value.strip() for value in cli_sites.split(",") if value.strip()]
        return values or None
    if configured is None:
        return None
    if not isinstance(configured, list):
        raise ValueError("scope.site_ids must be a list")
    values = [str(value).strip() for value in configured if str(value).strip()]
    return values or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ast-labels", required=True, type=Path)
    parser.add_argument("--sample-metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--min-total", type=int)
    parser.add_argument("--min-class", type=int)
    parser.add_argument("--min-years", type=int)
    parser.add_argument("--sites", help="Comma-separated site IDs; overrides config")
    parser.add_argument("--year-min", type=int)
    parser.add_argument("--year-max", type=int)
    parser.add_argument("--sir-column")
    parser.add_argument("--feature-availability-column")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")

    config = load_config(args.config.resolve())
    scope = config.get("scope", {})
    threshold_config = config.get("support_thresholds", {})
    columns = config.get("columns", {})
    if not isinstance(scope, dict) or not isinstance(threshold_config, dict):
        raise ValueError("scope and support_thresholds config entries must be objects")
    if not isinstance(columns, dict):
        raise ValueError("columns config entry must be an object")
    core_tasks = config.get("historical_core_panel")
    if not isinstance(core_tasks, list):
        raise ValueError("historical_core_panel config entry must be a list")
    author_policy = config.get("author_input_policy", {})
    if not isinstance(author_policy, dict):
        raise ValueError("author_input_policy config entry must be an object")

    thresholds = SupportThresholds(
        min_total=int(
            args.min_total
            if args.min_total is not None
            else threshold_config.get("min_total", 300)
        ),
        min_class=int(
            args.min_class
            if args.min_class is not None
            else threshold_config.get("min_class", 50)
        ),
        min_years=int(
            args.min_years
            if args.min_years is not None
            else threshold_config.get("min_years", 2)
        ),
    )
    ast_labels = read_table(args.ast_labels.resolve())
    sample_metadata = read_table(args.sample_metadata.resolve())
    feature_availability_column = (
        args.feature_availability_column
        if args.feature_availability_column is not None
        else columns.get("feature_availability")
    )
    audit = run_task_universe_audit(
        ast_labels,
        sample_metadata,
        thresholds=thresholds,
        core_tasks=core_tasks,
        default_author_input_reason=str(author_policy.get("default_reason", "")).strip(),
        sample_column=str(columns.get("sample", "public_sample_id")),
        site_column=str(columns.get("site", "site_id")),
        year_column=str(columns.get("year", "year")),
        organism_column=str(columns.get("organism", "organism_reported")),
        antimicrobial_code_column=str(
            columns.get("antimicrobial_code", "antimicrobial_code_reported")
        ),
        antimicrobial_name_column=str(
            columns.get("antimicrobial_name", "antimicrobial_name_reported")
        ),
        sir_column=str(
            args.sir_column
            if args.sir_column is not None
            else columns.get("sir", "reported_sir")
        ),
        conflict_column=str(columns.get("conflict", "conflict_flag")),
        core_task_column=str(columns.get("core_task", "task_id_if_core")),
        feature_availability_column=(
            str(feature_availability_column) if feature_availability_column else None
        ),
        sites=_sites(args.sites, scope.get("site_ids")),
        year_min=_optional_int(args.year_min, scope.get("year_min")),
        year_max=_optional_int(args.year_max, scope.get("year_max")),
    )

    output_dir.mkdir(parents=True)
    inventory_path = output_dir / "zd_mast_task_universe_combination_inventory_v1.csv"
    funnel_path = output_dir / "zd_mast_task_universe_support_funnel_v1.csv"
    core_path = output_dir / "zd_mast_historical_core_ten_mapping_v1.csv"
    audit.combination_inventory.to_csv(inventory_path, index=False)
    audit.funnel.to_csv(funnel_path, index=False)
    audit.core_ten_mapping.to_csv(core_path, index=False)
    manifest = {
        "analysis_id": str(config.get("analysis_id", "task_universe_selection_audit")),
        "analysis_version": str(config.get("analysis_version", "reviewer-r1-v1")),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": {
            "ast_labels": args.ast_labels.name,
            "sample_metadata": args.sample_metadata.name,
            "config": args.config.name,
        },
        "thresholds": {
            "min_total": thresholds.min_total,
            "min_class": thresholds.min_class,
            "min_years": thresholds.min_years,
        },
        "output_rows": {
            "combination_inventory": len(audit.combination_inventory),
            "support_funnel": len(audit.funnel),
            "core_ten_mapping": len(audit.core_ten_mapping),
        },
        "guardrails": {
            "model_training_performed": False,
            "performance_used_for_selection": False,
            "prospective_selection_claim_supported": False,
            "identity_matching": "exact_lexical_no_alias_or_substring_mapping",
        },
        "outputs": [inventory_path.name, funnel_path.name, core_path.name],
    }
    (output_dir / "zd_mast_task_universe_selection_audit_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
