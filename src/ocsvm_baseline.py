"""OC-SVM baseline utilities for air-quality anomaly detection.

This module provides:
1) Dataset cleaning helpers tailored to the Kaggle AirQuality.csv format.
2) Feature preparation.
3) A One-Class SVM baseline trainer with standard scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


@dataclass
class OCSVMBaseline:
    """Container for a fitted OC-SVM baseline and metadata."""

    scaler: StandardScaler
    model: OneClassSVM
    feature_columns: list[str]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict labels using OC-SVM output convention.

        Returns:
            np.ndarray: +1 for inliers (normal), -1 for outliers (anomalies).
        """
        x = features[self.feature_columns].to_numpy(dtype=float)
        x_scaled = self.scaler.transform(x)
        return self.model.predict(x_scaled)

    def decision_function(self, features: pd.DataFrame) -> np.ndarray:
        """Return OC-SVM decision score (higher means more normal)."""
        x = features[self.feature_columns].to_numpy(dtype=float)
        x_scaled = self.scaler.transform(x)
        return self.model.decision_function(x_scaled)


def load_and_clean_air_quality_csv(csv_path: str, sep: str = ";") -> pd.DataFrame:
    """Load the Kaggle air-quality CSV and apply standard cleaning steps.

    Cleaning assumptions (based on dataset format):
    - Decimal values may be strings with comma decimal separator.
    - Missing values are often represented as -200.
    - Extra unnamed columns can exist and should be removed.
    """
    df = pd.read_csv(csv_path, sep=sep)

    # Drop unnamed columns that are empty artifacts in this dataset.
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    # Convert every non-date/time column to numeric when possible.
    non_datetime_cols = [c for c in df.columns if c not in {"Date", "Time"}]
    for col in non_datetime_cols:
        # Convert comma-decimals to dot-decimals then parse as float.
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .replace({"nan": np.nan})
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace -200 sentinel values with NaN for consistent missing-value handling.
    df = df.replace(-200, np.nan)
    return df


def load_preprocessed_air_quality_csv(
    csv_path: str,
    datetime_column: str = "Datetime",
) -> pd.DataFrame:
    """Load a notebook-preprocessed CSV for downstream OC-SVM stages."""
    df = pd.read_csv(csv_path)

    if datetime_column in df.columns:
        parsed_dt = pd.to_datetime(df[datetime_column], errors="coerce")
        if parsed_dt.notna().any():
            df = df.copy()
            df[datetime_column] = parsed_dt
            df = df.set_index(datetime_column)

    return df


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_columns: Optional[Iterable[str]] = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Prepare a numeric feature matrix for OC-SVM training/inference."""
    if feature_columns is None:
        # Use all numeric columns by default.
        features = df.select_dtypes(include=[np.number]).copy()
    else:
        features = df[list(feature_columns)].copy()

    if drop_na:
        # OC-SVM cannot consume NaNs directly; drop incomplete rows by default.
        features = features.dropna(axis=0)
    return features


def train_ocsvm_baseline(
    features: pd.DataFrame,
    nu: float = 0.05,
    kernel: str = "rbf",
    gamma: str | float = "scale",
) -> OCSVMBaseline:
    """Train a standard OC-SVM baseline on prepared feature data."""
    if features.empty:
        raise ValueError("Cannot train OC-SVM baseline on an empty feature matrix.")

    x = features.to_numpy(dtype=float)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(x_scaled)

    return OCSVMBaseline(
        scaler=scaler,
        model=model,
        feature_columns=list(features.columns),
    )
