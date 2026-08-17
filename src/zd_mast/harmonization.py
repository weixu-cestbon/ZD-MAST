"""Aggregate and validate parallel historical and harmonized AST labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


M100_REQUIRED_COLUMNS = {
    "task_id",
    "historical_sir",
    "historical_binary_s_vs_ir",
    "harmonized_sir",
    "harmonized_binary_s_vs_ir",
    "reclassified_flag",
}


def summarize_m100_agreement(labels: pd.DataFrame) -> pd.DataFrame:
    """Summarize task-level agreement without modifying either label system."""
    missing = M100_REQUIRED_COLUMNS - set(labels.columns)
    if missing:
        raise ValueError(f"M100 label table is missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for task_id, group in labels.groupby("task_id", sort=True):
        interpreted = group[group["harmonized_sir"].isin(["S", "I", "R"])].copy()
        historical_binary = interpreted["historical_binary_s_vs_ir"].astype(float)
        harmonized_binary = interpreted["harmonized_binary_s_vs_ir"].astype(float)
        rows.append(
            {
                "task_id": task_id,
                "total_labels": len(group),
                "interpreted_n": len(interpreted),
                "unsupported_n": len(group) - len(interpreted),
                "historical_s_n": int(group["historical_sir"].eq("S").sum()),
                "historical_i_n": int(group["historical_sir"].eq("I").sum()),
                "historical_r_n": int(group["historical_sir"].eq("R").sum()),
                "harmonized_s_n": int(interpreted["harmonized_sir"].eq("S").sum()),
                "harmonized_i_n": int(interpreted["harmonized_sir"].eq("I").sum()),
                "harmonized_r_n": int(interpreted["harmonized_sir"].eq("R").sum()),
                "binary_s_vs_ir_agreement": float(
                    historical_binary.eq(harmonized_binary).mean()
                )
                if len(interpreted)
                else np.nan,
                "exact_sir_agreement": float(
                    interpreted["historical_sir"].eq(interpreted["harmonized_sir"]).mean()
                )
                if len(interpreted)
                else np.nan,
                "reclassified_n": int(
                    interpreted["reclassified_flag"].fillna(False).astype(bool).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def compare_m100_agreement(
    reproduced: pd.DataFrame,
    frozen: pd.DataFrame,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Return field-level regression results for task-level agreement tables."""
    key = "task_id"
    if reproduced[key].duplicated().any() or frozen[key].duplicated().any():
        raise ValueError("Agreement tables must contain one row per task_id")
    if set(reproduced[key]) != set(frozen[key]):
        raise ValueError("Reproduced and frozen agreement task sets differ")

    common = [column for column in reproduced.columns if column in frozen.columns]
    fields = [column for column in common if column != key]
    merged = reproduced.merge(frozen, on=key, suffixes=("_reproduced", "_frozen"))
    rows: list[dict[str, object]] = []
    for row in merged.to_dict(orient="records"):
        for field in fields:
            observed = row[f"{field}_reproduced"]
            expected = row[f"{field}_frozen"]
            if pd.isna(observed) and pd.isna(expected):
                difference = 0.0
                status = "PASS"
            elif pd.api.types.is_number(observed) and pd.api.types.is_number(expected):
                difference = abs(float(observed) - float(expected))
                status = "PASS" if difference <= tolerance else "FAIL"
            else:
                difference = np.nan
                status = "PASS" if observed == expected else "FAIL"
            rows.append(
                {
                    key: row[key],
                    "field": field,
                    "reproduced_value": observed,
                    "frozen_value": expected,
                    "absolute_difference": difference,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)
