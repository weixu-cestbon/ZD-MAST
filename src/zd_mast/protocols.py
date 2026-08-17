"""Frozen cohort and rolling-origin contracts for the primary analysis."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ProtocolWindows:
    """Calendar boundaries used by analysis version ``v2026.07.17.3``."""

    transition_start: str = "2025-06-23"
    transition_end: str = "2025-06-30"
    current_start: str = "2025-07-01"
    current_train_end: str = "2026-02-28"
    late_test_start: str = "2026-03-01"
    frozen_end: str = "2026-06-09"


@dataclass(frozen=True)
class TemporalCohort:
    """One development/test cohort generated from a frozen date contract."""

    protocol: str
    role: str
    development: pd.DataFrame
    test: pd.DataFrame
    development_definition: str
    test_definition: str


def _dated(frame: pd.DataFrame, date_column: str) -> pd.DataFrame:
    if date_column not in frame:
        raise ValueError(f"Missing date column: {date_column}")
    output = frame.copy()
    output[date_column] = pd.to_datetime(output[date_column], errors="raise")
    return output


def build_temporal_cohorts(
    frame: pd.DataFrame,
    date_column: str = "date",
    windows: ProtocolWindows | None = None,
) -> dict[str, TemporalCohort]:
    """Build the three frozen temporal protocols without using outcomes.

    Protocol B is the manuscript-primary current-workflow analysis. Protocols A
    and C are workflow and pooled-history sensitivities. The transition interval
    is excluded from Protocol C exactly as in the archived analysis script.
    """

    windows = windows or ProtocolWindows()
    dated = _dated(frame, date_column)
    transition_start = pd.Timestamp(windows.transition_start)
    transition_end = pd.Timestamp(windows.transition_end)
    current_start = pd.Timestamp(windows.current_start)
    current_train_end = pd.Timestamp(windows.current_train_end)
    late_test_start = pd.Timestamp(windows.late_test_start)
    frozen_end = pd.Timestamp(windows.frozen_end)
    transition = dated[date_column].between(transition_start, transition_end, inclusive="both")

    return {
        "A_pre_marker_to_post_marker": TemporalCohort(
            protocol="A_pre_marker_to_post_marker",
            role="workflow_transportability",
            development=dated[dated[date_column] < transition_start].copy(),
            test=dated[dated[date_column].between(current_start, frozen_end, inclusive="both")].copy(),
            development_definition="date < 2025-06-23",
            test_definition="2025-07-01 through 2026-06-09",
        ),
        "B_post_marker_current_temporal": TemporalCohort(
            protocol="B_post_marker_current_temporal",
            role="manuscript_primary_current_workflow",
            development=dated[
                dated[date_column].between(current_start, current_train_end, inclusive="both")
            ].copy(),
            test=dated[
                dated[date_column].between(late_test_start, frozen_end, inclusive="both")
            ].copy(),
            development_definition="2025-07-01 through 2026-02-28",
            test_definition="2026-03-01 through 2026-06-09",
        ),
        "C_pooled_history_temporal": TemporalCohort(
            protocol="C_pooled_history_temporal",
            role="pooled_history_sensitivity_not_causal",
            development=dated[(dated[date_column] < late_test_start) & ~transition].copy(),
            test=dated[
                dated[date_column].between(late_test_start, frozen_end, inclusive="both")
            ].copy(),
            development_definition=(
                "all dates before 2026-03-01 excluding 2025-06-23 through 2025-06-30"
            ),
            test_definition="2026-03-01 through 2026-06-09",
        ),
    }


def rolling_origin_folds(
    frame: pd.DataFrame,
    date_column: str = "date",
    sample_column: str = "sample_key",
    label_column: str = "y",
) -> list[tuple[pd.DataFrame, pd.DataFrame, str]]:
    """Reproduce the development-only rolling-origin fold construction.

    The function mirrors the archived `v2026.07.17.3` fold boundaries and
    adequacy checks. It never sees the frozen temporal test block.
    """

    required = {date_column, sample_column, label_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing rolling-origin columns: {sorted(missing)}")
    dated = _dated(frame, date_column)
    dated = dated.sort_values([date_column, sample_column]).drop_duplicates(sample_column, keep="last")
    month = dated[date_column].dt.to_period("M")
    months = sorted(month.dropna().unique())
    if len(months) < 4:
        return []

    import math

    boundaries = sorted(
        set([max(2, math.floor(len(months) * fraction)) for fraction in (0.50, 0.67, 0.84)] + [len(months)])
    )
    folds: list[tuple[pd.DataFrame, pd.DataFrame, str]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        train_months = set(months[:start])
        validation_months = set(months[start:end])
        train = dated[month.isin(train_months)].copy()
        validation = dated[month.isin(validation_months)].copy()
        if len(train) < 50 or len(validation) < 20:
            continue
        if train[label_column].nunique() < 2 or validation[label_column].nunique() < 2:
            continue
        if train[label_column].value_counts().min() < 10 or validation[label_column].value_counts().min() < 5:
            continue
        note = (
            f"train_through={max(train_months)};"
            f"validation={min(validation_months)}..{max(validation_months)}"
        )
        folds.append((train, validation, note))
    return folds


def patient_disjoint_test(
    development: pd.DataFrame,
    test: pd.DataFrame,
    patient_column: str = "patient_cluster_id",
) -> pd.DataFrame:
    """Return test rows whose non-missing patient cluster is absent from development."""

    if patient_column not in development or patient_column not in test:
        raise ValueError(f"Missing patient grouping column: {patient_column}")
    development_groups = set(development[patient_column].dropna().astype(str))
    group = test[patient_column]
    keep = group.notna() & ~group.astype(str).isin(development_groups)
    return test.loc[keep].copy()


def episode_first_rows(
    frame: pd.DataFrame,
    patient_column: str = "patient_cluster_id",
    flag_column: str = "episode_first_sample_flag",
) -> pd.DataFrame:
    """Return the prespecified patient-species 30-day episode-first rows."""

    missing = {patient_column, flag_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing episode columns: {sorted(missing)}")
    keep = frame[patient_column].notna() & frame[flag_column].fillna(False).astype(bool)
    return frame.loc[keep].copy()
