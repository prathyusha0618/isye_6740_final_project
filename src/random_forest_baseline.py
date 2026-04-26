"""Random Forest helpers for branch-level supervised evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass
class RandomForestRunResult:
    """Summary of a Random Forest run on one dataset variant."""

    dataset_name: str
    rows_used: int
    feature_count: int
    class_count: int
    skipped: bool
    reason: str | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    roc_auc: float | None = None


def train_random_forest_on_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    target_column: str = "is_anomaly",
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 300,
    drop_columns: tuple[str, ...] = ("ocsvm_raw_label", "ocsvm_score"),
) -> RandomForestRunResult:
    """Train/evaluate Random Forest on one dataset variant.

    This function selects numeric features only, excludes known OC-SVM artifact
    columns, and reports held-out metrics.
    """
    if target_column not in df.columns:
        raise KeyError(f"'{target_column}' column is required for Random Forest.")

    y = pd.to_numeric(df[target_column], errors="coerce")
    feature_frame = df.drop(columns=[target_column], errors="ignore").drop(
        columns=list(drop_columns),
        errors="ignore",
    )
    x = feature_frame.select_dtypes(include=[np.number]).copy()

    # Keep rows where target and all selected features are available.
    valid_mask = y.notna()
    if not x.empty:
        valid_mask &= x.notna().all(axis=1)

    x = x.loc[valid_mask]
    y = y.loc[valid_mask].astype(int)

    if x.empty:
        return RandomForestRunResult(
            dataset_name=dataset_name,
            rows_used=0,
            feature_count=0,
            class_count=0,
            skipped=True,
            reason="No numeric feature rows available after cleaning.",
        )

    unique_classes = np.unique(y.to_numpy())
    if unique_classes.size < 2:
        return RandomForestRunResult(
            dataset_name=dataset_name,
            rows_used=len(x),
            feature_count=x.shape[1],
            class_count=int(unique_classes.size),
            skipped=True,
            reason="Only one target class present; classifier training skipped.",
        )

    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError as exc:
        return RandomForestRunResult(
            dataset_name=dataset_name,
            rows_used=len(x),
            feature_count=x.shape[1],
            class_count=int(unique_classes.size),
            skipped=True,
            reason=f"Train/test split failed: {exc}",
        )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="binary",
        zero_division=0,
    )

    roc_auc = None
    if hasattr(model, "predict_proba"):
        class_to_index = {int(cls): idx for idx, cls in enumerate(model.classes_)}
        if 1 in class_to_index:
            y_proba = model.predict_proba(x_test)[:, class_to_index[1]]
            roc_auc = roc_auc_score(y_test, y_proba)

    return RandomForestRunResult(
        dataset_name=dataset_name,
        rows_used=len(x),
        feature_count=x.shape[1],
        class_count=int(unique_classes.size),
        skipped=False,
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=float(roc_auc) if roc_auc is not None else None,
    )
