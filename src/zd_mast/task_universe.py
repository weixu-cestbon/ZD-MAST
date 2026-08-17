"""Performance-independent support audit for the historical ZD-MAST task panel.

The task universe is constructed from exact reported organism and antimicrobial
identities.  Normalization is deliberately limited to Unicode width, case and
whitespace; it never applies substring matching or biological synonym mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import unicodedata
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_SAMPLE_COLUMN = "public_sample_id"
DEFAULT_SITE_COLUMN = "site_id"
DEFAULT_YEAR_COLUMN = "year"
DEFAULT_ORGANISM_COLUMN = "organism_reported"
DEFAULT_ANTIMICROBIAL_CODE_COLUMN = "antimicrobial_code_reported"
DEFAULT_ANTIMICROBIAL_NAME_COLUMN = "antimicrobial_name_reported"
DEFAULT_SIR_COLUMN = "reported_sir"
DEFAULT_CONFLICT_COLUMN = "conflict_flag"
DEFAULT_CORE_TASK_COLUMN = "task_id_if_core"


@dataclass(frozen=True)
class SupportThresholds:
    """Minimum support required after conflict and feature-linkage filtering."""

    min_total: int = 300
    min_class: int = 50
    min_years: int = 2

    def __post_init__(self) -> None:
        for name, value in (
            ("min_total", self.min_total),
            ("min_class", self.min_class),
            ("min_years", self.min_years),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")


@dataclass(frozen=True)
class TaskUniverseAudit:
    """Machine-readable outputs of the support-only task-universe audit."""

    combination_inventory: pd.DataFrame
    funnel: pd.DataFrame
    core_ten_mapping: pd.DataFrame
    sample_combination_labels: pd.DataFrame


def normalize_exact_identity(value: object) -> str:
    """Return a lexical identity key without biological alias expansion."""

    if value is None or pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKC", str(value))
    return " ".join(normalized.split()).strip().casefold()


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _truthy_feature_availability(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        return values.notna() & values.ge(0)
    normalized = values.fillna("").astype(str).map(normalize_exact_identity)
    unavailable = {"", "0", "false", "no", "n", "missing", "unavailable"}
    return ~normalized.isin(unavailable)


def feature_available_samples(
    sample_metadata: pd.DataFrame,
    *,
    sample_column: str = DEFAULT_SAMPLE_COLUMN,
    availability_column: str | None = None,
) -> set[str]:
    """Return sample identifiers with a released or otherwise available feature row."""

    _require_columns(sample_metadata, [sample_column], "sample metadata")
    metadata = sample_metadata.copy()
    metadata[sample_column] = metadata[sample_column].fillna("").astype(str).str.strip()
    if metadata[sample_column].eq("").any():
        raise ValueError("sample metadata contains blank sample identifiers")

    selected_column = availability_column
    if selected_column is None and "feature_row" in metadata.columns:
        selected_column = "feature_row"
    if selected_column is None:
        available = pd.Series(True, index=metadata.index)
    else:
        _require_columns(metadata, [selected_column], "sample metadata")
        available = _truthy_feature_availability(metadata[selected_column])
    return set(metadata.loc[available, sample_column].astype(str))


def _combination_id(
    organism_key: str,
    antimicrobial_code_key: str,
    antimicrobial_name_key: str,
) -> str:
    payload = "\x1f".join(
        [organism_key, antimicrobial_code_key, antimicrobial_name_key]
    ).encode("utf-8")
    return "cmb_" + hashlib.sha256(payload).hexdigest()[:16]


def _joined_unique(values: pd.Series) -> str:
    cleaned = {
        str(value).strip()
        for value in values
        if value is not None and not pd.isna(value) and str(value).strip()
    }
    return "|".join(sorted(cleaned, key=lambda item: (item.casefold(), item)))


def _coerce_conflict(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(0).ne(0)
    normalized = values.fillna("").astype(str).map(normalize_exact_identity)
    return normalized.isin({"1", "true", "yes", "y", "conflict"})


def _scope_ast_rows(
    ast_labels: pd.DataFrame,
    *,
    site_column: str,
    year_column: str,
    sites: Sequence[str] | None,
    year_min: int | None,
    year_max: int | None,
) -> pd.DataFrame:
    scoped = ast_labels.copy()
    scoped[year_column] = pd.to_numeric(scoped[year_column], errors="coerce")
    if scoped[year_column].isna().any():
        raise ValueError("AST label table contains missing or non-numeric years")
    scoped[year_column] = scoped[year_column].astype(int)
    if sites:
        requested = {str(value) for value in sites}
        scoped = scoped.loc[scoped[site_column].astype(str).isin(requested)].copy()
    if year_min is not None:
        scoped = scoped.loc[scoped[year_column].ge(year_min)].copy()
    if year_max is not None:
        scoped = scoped.loc[scoped[year_column].le(year_max)].copy()
    return scoped.reset_index(drop=True)


def build_sample_combination_labels(
    ast_labels: pd.DataFrame,
    feature_samples: set[str],
    *,
    sample_column: str = DEFAULT_SAMPLE_COLUMN,
    site_column: str = DEFAULT_SITE_COLUMN,
    year_column: str = DEFAULT_YEAR_COLUMN,
    organism_column: str = DEFAULT_ORGANISM_COLUMN,
    antimicrobial_code_column: str = DEFAULT_ANTIMICROBIAL_CODE_COLUMN,
    antimicrobial_name_column: str = DEFAULT_ANTIMICROBIAL_NAME_COLUMN,
    sir_column: str = DEFAULT_SIR_COLUMN,
    conflict_column: str = DEFAULT_CONFLICT_COLUMN,
    core_task_column: str = DEFAULT_CORE_TASK_COLUMN,
    sites: Sequence[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse AST rows to one auditable sample-by-exact-combination label.

    Returns the scoped input rows with identity keys and the collapsed table.
    A source conflict, multiple S/I/R categories, or multiple years for the same
    site/sample/combination is treated as a conflict and excluded downstream.
    """

    required = [
        sample_column,
        site_column,
        year_column,
        organism_column,
        antimicrobial_code_column,
        antimicrobial_name_column,
        sir_column,
    ]
    _require_columns(ast_labels, required, "AST label table")
    scoped = _scope_ast_rows(
        ast_labels,
        site_column=site_column,
        year_column=year_column,
        sites=sites,
        year_min=year_min,
        year_max=year_max,
    )
    if scoped.empty:
        raise ValueError("No AST rows remain after applying site/year scope")

    scoped[sample_column] = scoped[sample_column].fillna("").astype(str).str.strip()
    if scoped[sample_column].eq("").any():
        raise ValueError("AST label table contains blank sample identifiers")
    scoped["organism_exact_key"] = scoped[organism_column].map(normalize_exact_identity)
    scoped["antimicrobial_code_exact_key"] = scoped[antimicrobial_code_column].map(
        normalize_exact_identity
    )
    scoped["antimicrobial_name_exact_key"] = scoped[antimicrobial_name_column].map(
        normalize_exact_identity
    )
    scoped["identity_complete"] = scoped["organism_exact_key"].ne("") & (
        scoped["antimicrobial_code_exact_key"].ne("")
        | scoped["antimicrobial_name_exact_key"].ne("")
    )
    scoped["combination_id"] = ""
    complete = scoped["identity_complete"]
    scoped.loc[complete, "combination_id"] = [
        _combination_id(organism, code, name)
        for organism, code, name in scoped.loc[
            complete,
            [
                "organism_exact_key",
                "antimicrobial_code_exact_key",
                "antimicrobial_name_exact_key",
            ],
        ].itertuples(index=False, name=None)
    ]
    scoped["sir_normalized"] = (
        scoped[sir_column].fillna("").astype(str).str.strip().str.upper()
    )
    if conflict_column in scoped.columns:
        scoped["source_conflict"] = _coerce_conflict(scoped[conflict_column])
    else:
        scoped["source_conflict"] = False
    if core_task_column in scoped.columns:
        scoped["core_task_id"] = (
            scoped[core_task_column].fillna("").astype(str).str.strip()
        )
    else:
        scoped["core_task_id"] = ""

    eligible_identity = scoped.loc[scoped["identity_complete"]].copy()
    group_key = [site_column, sample_column, "combination_id"]
    eligible_identity["valid_sir"] = eligible_identity["sir_normalized"].where(
        eligible_identity["sir_normalized"].isin(["S", "I", "R"])
    )
    eligible_identity["nonblank_core_task"] = eligible_identity["core_task_id"].replace(
        "", np.nan
    )
    # The released full-AST extension is already close to this unit, but the
    # vectorized collapse also handles duplicate export rows without iterating
    # over hundreds of thousands of sample groups in Python.
    collapsed = (
        eligible_identity.groupby(group_key, sort=False, dropna=False)
        .agg(
            **{
                year_column: (year_column, "first"),
                "year_n": (year_column, "nunique"),
                "organism_exact_key": ("organism_exact_key", "first"),
                "organism_reported_values": (organism_column, "first"),
                "antimicrobial_code_exact_key": (
                    "antimicrobial_code_exact_key",
                    "first",
                ),
                "antimicrobial_code_reported_values": (
                    antimicrobial_code_column,
                    "first",
                ),
                "antimicrobial_name_exact_key": (
                    "antimicrobial_name_exact_key",
                    "first",
                ),
                "antimicrobial_name_reported_values": (
                    antimicrobial_name_column,
                    "first",
                ),
                "source_row_n": (sample_column, "size"),
                "source_conflict_flag": ("source_conflict", "max"),
                "sir_n": ("valid_sir", "nunique"),
                "resolved_sir": ("valid_sir", "first"),
                "core_task_n": ("nonblank_core_task", "nunique"),
                "core_task_ids": ("nonblank_core_task", "first"),
            }
        )
        .reset_index()
    )
    if collapsed["core_task_n"].gt(1).any():
        raise ValueError("One exact sample-combination maps to multiple core task IDs")
    collapsed["resolved_sir"] = collapsed["resolved_sir"].fillna("")
    collapsed["core_task_ids"] = collapsed["core_task_ids"].fillna("")
    collapsed["sample_combination_conflict_flag"] = (
        collapsed["source_conflict_flag"].astype(bool)
        | collapsed["sir_n"].gt(1)
        | collapsed["year_n"].ne(1)
    )
    collapsed.loc[collapsed["sample_combination_conflict_flag"], "resolved_sir"] = ""
    collapsed["binary_s_vs_ir"] = collapsed["resolved_sir"].map(
        {"S": 0, "I": 1, "R": 1}
    )
    collapsed["feature_available"] = collapsed[sample_column].astype(str).isin(
        feature_samples
    )
    collapsed = collapsed.drop(columns=["year_n", "sir_n", "core_task_n"])
    if collapsed.empty:
        raise ValueError("No rows have complete exact organism/drug identity")
    if collapsed.duplicated(group_key).any():
        raise ValueError("Collapsed sample-combination table is not unique")
    return scoped, collapsed.sort_values(group_key).reset_index(drop=True)


def _inventory_for_combination(
    combination_id: str,
    group: pd.DataFrame,
    thresholds: SupportThresholds,
    *,
    sample_column: str,
    site_column: str,
    year_column: str,
) -> dict[str, object]:
    conflict_free = group.loc[~group["sample_combination_conflict_flag"]]
    labelled = conflict_free.loc[conflict_free["resolved_sir"].isin(["S", "I", "R"])]
    feature_linked = labelled.loc[labelled["feature_available"]].copy()
    positive_n = int(feature_linked["binary_s_vs_ir"].eq(1).sum())
    negative_n = int(feature_linked["binary_s_vs_ir"].eq(0).sum())
    year_values = sorted(set(feature_linked[year_column].dropna().astype(int)))
    total_n = int(len(feature_linked))
    min_class_n = min(positive_n, negative_n) if total_n else 0
    passes_years = len(year_values) >= thresholds.min_years
    passes_total = total_n >= thresholds.min_total
    passes_class = min_class_n >= thresholds.min_class
    failed: list[str] = []
    if not passes_years:
        failed.append("min_years")
    if not passes_total:
        failed.append("min_total")
    if not passes_class:
        failed.append("min_class")
    return {
        "combination_id": combination_id,
        "organism_exact_key": group["organism_exact_key"].iloc[0],
        "organism_reported_values": _joined_unique(group["organism_reported_values"]),
        "antimicrobial_code_exact_key": group[
            "antimicrobial_code_exact_key"
        ].iloc[0],
        "antimicrobial_code_reported_values": _joined_unique(
            group["antimicrobial_code_reported_values"]
        ),
        "antimicrobial_name_exact_key": group[
            "antimicrobial_name_exact_key"
        ].iloc[0],
        "antimicrobial_name_reported_values": _joined_unique(
            group["antimicrobial_name_reported_values"]
        ),
        "site_n": int(group[site_column].nunique()),
        "site_ids": _joined_unique(group[site_column]),
        "raw_sample_combination_n": int(len(group)),
        "raw_unique_sample_n": int(group[sample_column].nunique()),
        "conflict_n": int(group["sample_combination_conflict_flag"].sum()),
        "conflict_free_sir_n": int(len(labelled)),
        "feature_linked_label_n": total_n,
        "total_n": total_n,
        "feature_linked_unique_sample_n": int(feature_linked[sample_column].nunique()),
        "year_n": int(len(year_values)),
        "years": "|".join(str(value) for value in year_values),
        "positive_n_s_vs_ir": positive_n,
        "negative_n_s_vs_ir": negative_n,
        "min_class_n": min_class_n,
        "positive_rate_s_vs_ir": positive_n / total_n if total_n else np.nan,
        "min_years_threshold": thresholds.min_years,
        "min_total_threshold": thresholds.min_total,
        "min_class_threshold": thresholds.min_class,
        "passes_min_years": passes_years,
        "passes_min_total": passes_total,
        "passes_min_class": passes_class,
        "support_eligible": passes_years and passes_total and passes_class,
        "support_exclusion_reasons": "|".join(failed),
        "core_task_ids": _joined_unique(group["core_task_ids"]),
        "selection_basis": "performance_independent_support_only",
    }


def build_combination_inventory(
    collapsed: pd.DataFrame,
    thresholds: SupportThresholds,
    *,
    sample_column: str = DEFAULT_SAMPLE_COLUMN,
    site_column: str = DEFAULT_SITE_COLUMN,
    year_column: str = DEFAULT_YEAR_COLUMN,
) -> pd.DataFrame:
    """Summarize support for every exact organism-antimicrobial combination."""

    rows = [
        _inventory_for_combination(
            str(combination_id),
            group,
            thresholds,
            sample_column=sample_column,
            site_column=site_column,
            year_column=year_column,
        )
        for combination_id, group in collapsed.groupby("combination_id", sort=False)
    ]
    inventory = pd.DataFrame(rows)
    return inventory.sort_values(
        ["support_eligible", "feature_linked_label_n", "combination_id"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _funnel_row(
    order: int,
    step: str,
    description: str,
    frame: pd.DataFrame,
    *,
    sample_column: str,
    combination_column: str | None = "combination_id",
    threshold_value: int | None = None,
    threshold_unit: str = "",
) -> dict[str, object]:
    combination_n: int | float
    if combination_column is None:
        combination_n = np.nan
    else:
        combination_n = int(frame[combination_column].replace("", np.nan).nunique())
    sir_column = "resolved_sir" if "resolved_sir" in frame.columns else "sir_normalized"
    if sir_column in frame.columns:
        labelled = frame.loc[frame[sir_column].isin(["S", "I", "R"])].copy()
        if "binary_s_vs_ir" not in labelled.columns:
            labelled["binary_s_vs_ir"] = labelled[sir_column].map(
                {"S": 0, "I": 1, "R": 1}
            )
    else:
        labelled = frame.iloc[0:0].copy()
    return {
        "step_order": order,
        "step": step,
        "description": description,
        "combination_n": combination_n,
        "label_rows": int(len(frame)),
        "unique_sample_n": int(frame[sample_column].nunique()),
        "positive_n_s_vs_ir": int(labelled.get("binary_s_vs_ir", pd.Series(dtype=float)).eq(1).sum()),
        "negative_n_s_vs_ir": int(labelled.get("binary_s_vs_ir", pd.Series(dtype=float)).eq(0).sum()),
        "threshold_value": threshold_value if threshold_value is not None else np.nan,
        "threshold_unit": threshold_unit,
    }


def build_support_funnel(
    scoped: pd.DataFrame,
    collapsed: pd.DataFrame,
    inventory: pd.DataFrame,
    thresholds: SupportThresholds,
    *,
    sample_column: str = DEFAULT_SAMPLE_COLUMN,
) -> pd.DataFrame:
    """Build a deterministic row/sample/combination support funnel."""

    complete_rows = scoped.loc[scoped["identity_complete"]].copy()
    conflict_free = collapsed.loc[~collapsed["sample_combination_conflict_flag"]].copy()
    labelled = conflict_free.loc[conflict_free["resolved_sir"].isin(["S", "I", "R"])]
    feature_linked = labelled.loc[labelled["feature_available"]].copy()
    years_pass = set(inventory.loc[inventory["passes_min_years"], "combination_id"])
    total_pass = set(
        inventory.loc[
            inventory["passes_min_years"] & inventory["passes_min_total"],
            "combination_id",
        ]
    )
    final_pass = set(inventory.loc[inventory["support_eligible"], "combination_id"])
    rows = [
        _funnel_row(
            1,
            "scoped_linked_ast_rows",
            "All de-identified linked AST rows in the requested site/year scope.",
            scoped,
            sample_column=sample_column,
            combination_column=None,
        ),
        _funnel_row(
            2,
            "complete_exact_identity",
            "Rows with nonblank exact organism identity and antimicrobial code or name.",
            complete_rows,
            sample_column=sample_column,
        ),
        _funnel_row(
            3,
            "unique_sample_combination",
            "One row per site, sample and exact organism-antimicrobial identity.",
            collapsed,
            sample_column=sample_column,
        ),
        _funnel_row(
            4,
            "conflict_free",
            "Sample-combination rows without source, category or year conflicts.",
            conflict_free,
            sample_column=sample_column,
        ),
        _funnel_row(
            5,
            "binary_s_vs_ir_available",
            "Conflict-free rows with S, I or R and S versus I/R label available.",
            labelled,
            sample_column=sample_column,
        ),
        _funnel_row(
            6,
            "feature_linked",
            "Label rows whose public sample ID has an available feature row.",
            feature_linked,
            sample_column=sample_column,
        ),
        _funnel_row(
            7,
            "minimum_year_support",
            "Feature-linked combinations meeting the configured year-count threshold.",
            feature_linked.loc[feature_linked["combination_id"].isin(years_pass)],
            sample_column=sample_column,
            threshold_value=thresholds.min_years,
            threshold_unit="years",
        ),
        _funnel_row(
            8,
            "minimum_total_support",
            "Year-supported combinations meeting the configured total-label threshold.",
            feature_linked.loc[feature_linked["combination_id"].isin(total_pass)],
            sample_column=sample_column,
            threshold_value=thresholds.min_total,
            threshold_unit="sample_combination_labels",
        ),
        _funnel_row(
            9,
            "minimum_class_support",
            "Total-supported combinations meeting the configured minimum-class threshold.",
            feature_linked.loc[feature_linked["combination_id"].isin(final_pass)],
            sample_column=sample_column,
            threshold_value=thresholds.min_class,
            threshold_unit="minimum_binary_class",
        ),
    ]
    return pd.DataFrame(rows)


def validate_core_task_config(core_tasks: Sequence[Mapping[str, object]]) -> None:
    """Validate the fixed 3-anchor/7-extension provenance contract."""

    if len(core_tasks) != 10:
        raise ValueError(f"Historical panel config must contain 10 tasks, found {len(core_tasks)}")
    task_ids = [str(task.get("task_id", "")).strip() for task in core_tasks]
    if any(not task_id for task_id in task_ids) or len(task_ids) != len(set(task_ids)):
        raise ValueError("Historical panel task IDs must be nonblank and unique")
    categories = pd.Series(
        [str(task.get("historical_provenance_category", "")).strip() for task in core_tasks]
    ).value_counts()
    expected = {"literature_anchor": 3, "local_extension": 7}
    if categories.to_dict() != expected:
        raise ValueError(
            "Historical provenance config must contain exactly 3 literature_anchor "
            f"and 7 local_extension tasks; found {categories.to_dict()}"
        )


def build_core_ten_mapping(
    inventory: pd.DataFrame,
    core_tasks: Sequence[Mapping[str, object]],
    *,
    default_author_input_reason: str,
) -> pd.DataFrame:
    """Map the historical ten-task panel onto exact task-universe combinations."""

    validate_core_task_config(core_tasks)
    rows: list[dict[str, object]] = []
    for task in core_tasks:
        task_id = str(task["task_id"]).strip()
        task_inventory = inventory.loc[
            inventory["core_task_ids"].str.split("|").map(lambda values: task_id in values)
        ].copy()
        evidence = str(task.get("original_human_selection_evidence", "")).strip()
        author_input_needed = not evidence
        exact_ids = sorted(task_inventory["combination_id"].astype(str))
        rows.append(
            {
                "task_id": task_id,
                "organism": str(task.get("organism", "")).strip(),
                "antimicrobial": str(task.get("antimicrobial", "")).strip(),
                "historical_provenance_category": str(
                    task["historical_provenance_category"]
                ).strip(),
                "historical_provenance_note": str(
                    task.get("historical_provenance_note", "")
                ).strip(),
                "exact_combination_n": int(len(task_inventory)),
                "exact_combination_ids": "|".join(exact_ids),
                "feature_linked_label_n": int(task_inventory["feature_linked_label_n"].sum()),
                "positive_n_s_vs_ir": int(task_inventory["positive_n_s_vs_ir"].sum()),
                "negative_n_s_vs_ir": int(task_inventory["negative_n_s_vs_ir"].sum()),
                "support_eligible_exact_combination_n": int(
                    task_inventory["support_eligible"].sum()
                ),
                "observed_in_task_universe": bool(len(task_inventory)),
                "original_human_selection_evidence": evidence,
                "author_input_needed": author_input_needed,
                "author_input_needed_reason": (
                    default_author_input_reason if author_input_needed else ""
                ),
                "selection_status": "historical_panel_member_not_prospectively_selected",
                "interpretation_boundary": (
                    "Retrospective provenance mapping; support counts do not establish "
                    "the original human selection rationale or prospective task selection."
                ),
            }
        )
    return pd.DataFrame(rows)


def run_task_universe_audit(
    ast_labels: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    *,
    thresholds: SupportThresholds,
    core_tasks: Sequence[Mapping[str, object]],
    default_author_input_reason: str,
    sample_column: str = DEFAULT_SAMPLE_COLUMN,
    site_column: str = DEFAULT_SITE_COLUMN,
    year_column: str = DEFAULT_YEAR_COLUMN,
    organism_column: str = DEFAULT_ORGANISM_COLUMN,
    antimicrobial_code_column: str = DEFAULT_ANTIMICROBIAL_CODE_COLUMN,
    antimicrobial_name_column: str = DEFAULT_ANTIMICROBIAL_NAME_COLUMN,
    sir_column: str = DEFAULT_SIR_COLUMN,
    conflict_column: str = DEFAULT_CONFLICT_COLUMN,
    core_task_column: str = DEFAULT_CORE_TASK_COLUMN,
    feature_availability_column: str | None = None,
    sites: Sequence[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> TaskUniverseAudit:
    """Run the exact-identity support audit without fitting predictive models."""

    feature_samples = feature_available_samples(
        sample_metadata,
        sample_column=sample_column,
        availability_column=feature_availability_column,
    )
    scoped, collapsed = build_sample_combination_labels(
        ast_labels,
        feature_samples,
        sample_column=sample_column,
        site_column=site_column,
        year_column=year_column,
        organism_column=organism_column,
        antimicrobial_code_column=antimicrobial_code_column,
        antimicrobial_name_column=antimicrobial_name_column,
        sir_column=sir_column,
        conflict_column=conflict_column,
        core_task_column=core_task_column,
        sites=sites,
        year_min=year_min,
        year_max=year_max,
    )
    inventory = build_combination_inventory(
        collapsed,
        thresholds,
        sample_column=sample_column,
        site_column=site_column,
        year_column=year_column,
    )
    funnel = build_support_funnel(
        scoped,
        collapsed,
        inventory,
        thresholds,
        sample_column=sample_column,
    )
    core_mapping = build_core_ten_mapping(
        inventory,
        core_tasks,
        default_author_input_reason=default_author_input_reason,
    )
    return TaskUniverseAudit(
        combination_inventory=inventory,
        funnel=funnel,
        core_ten_mapping=core_mapping,
        sample_combination_labels=collapsed,
    )
