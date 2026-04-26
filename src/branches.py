"""Downstream branches for anomaly handling: removal and repair."""

from __future__ import annotations

import numpy as np
import pandas as pd


def removal_branch(
    labeled_df: pd.DataFrame,
    anomaly_column: str = "is_anomaly",
) -> pd.DataFrame:
    """Removal branch: drop rows marked as anomalous."""
    if anomaly_column not in labeled_df.columns:
        raise KeyError(f"'{anomaly_column}' column is required for removal branch.")

    # Keep only normal observations.
    return labeled_df[labeled_df[anomaly_column] == 0].copy()


def repair_branch(
    labeled_df: pd.DataFrame,
    anomaly_column: str = "is_anomaly",
    strategy: str = "median",
    protected_columns: tuple[str, ...] = ("Date", "Time", "ocsvm_raw_label", "ocsvm_score"),
) -> pd.DataFrame:
    """Repair branch: replace anomalous numeric values with robust estimates.

    Strategies:
    - median: replace each anomalous numeric feature with median of normal rows.
    - mean: replace with mean of normal rows.

    Non-numeric columns are left unchanged.
    """
    if anomaly_column not in labeled_df.columns:
        raise KeyError(f"'{anomaly_column}' column is required for repair branch.")
    if strategy not in {"median", "mean"}:
        raise ValueError("strategy must be one of: {'median', 'mean'}")

    repaired = labeled_df.copy()
    normal_mask = repaired[anomaly_column] == 0
    anomaly_mask = repaired[anomaly_column] == 1

    # Select numeric feature columns only, excluding labels/metadata.
    numeric_cols = repaired.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_repair = [
        c for c in numeric_cols if c not in set(protected_columns + (anomaly_column,))
    ]

    # Fit replacement statistics on normal rows only.
    if strategy == "median":
        replacement_values = repaired.loc[normal_mask, cols_to_repair].median(axis=0)
    else:
        replacement_values = repaired.loc[normal_mask, cols_to_repair].mean(axis=0)

    # Apply replacement stats to anomalous rows.
    repaired.loc[anomaly_mask, cols_to_repair] = replacement_values.values
    return repaired
