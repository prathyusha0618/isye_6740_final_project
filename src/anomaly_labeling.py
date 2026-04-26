"""Anomaly labeling utilities using a trained OC-SVM baseline."""

from __future__ import annotations

import pandas as pd

from .ocsvm_baseline import OCSVMBaseline


def generate_anomaly_labels(
    baseline: OCSVMBaseline,
    features: pd.DataFrame,
    anomaly_column: str = "is_anomaly",
    svm_output_column: str = "ocsvm_raw_label",
    score_column: str = "ocsvm_score",
) -> pd.DataFrame:
    """Generate anomaly labels from a trained baseline.

    Output columns:
    - ocsvm_raw_label: +1 (normal), -1 (anomaly) from OC-SVM.
    - is_anomaly: 0 (normal), 1 (anomaly), easier for downstream processing.
    - ocsvm_score: decision function score (lower values imply more anomalous).
    """
    raw_labels = baseline.predict(features)
    scores = baseline.decision_function(features)

    labeled = features.copy()
    labeled[svm_output_column] = raw_labels
    labeled[anomaly_column] = (raw_labels == -1).astype(int)
    labeled[score_column] = scores
    return labeled
