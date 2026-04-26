"""Run the OC-SVM anomaly pipeline from the command line.

This script reads AirQuality.csv, trains an OC-SVM baseline, generates anomaly
labels, and writes three outputs:
1) labeled data
2) anomaly-removed data
3) anomaly-repaired data
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from src import (
    generate_anomaly_labels,
    load_and_clean_air_quality_csv,
    load_preprocessed_air_quality_csv,
    prepare_feature_matrix,
    removal_branch,
    repair_branch,
    train_random_forest_on_dataframe,
    train_ocsvm_baseline,
)

# Candidate delimiters used when inferring CSV format from input samples.
COMMON_CSV_DELIMITERS = ",;\t|"


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run OC-SVM anomaly pipeline on AirQuality.csv."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to AirQuality.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for output CSVs (default: current directory).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="air_quality",
        help="Prefix for generated output files.",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default=";",
        help="CSV separator for input file (default: ';').",
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=0.05,
        help="OC-SVM nu parameter (default: 0.05).",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        help="OC-SVM kernel (default: rbf).",
    )
    parser.add_argument(
        "--gamma",
        type=str,
        default="scale",
        help="OC-SVM gamma (default: scale). Use numeric strings like '0.1' if needed.",
    )
    parser.add_argument(
        "--repair-strategy",
        choices=("median", "mean"),
        default="median",
        help="Repair strategy for anomaly rows (default: median).",
    )
    parser.add_argument(
        "--train-rf",
        action="store_true",
        help="Train Random Forest on labeled/removed/repaired outputs.",
    )
    parser.add_argument(
        "--rf-target-column",
        type=str,
        default="is_anomaly",
        help="Target column for Random Forest training (default: is_anomaly).",
    )
    parser.add_argument(
        "--rf-test-size",
        type=float,
        default=0.2,
        help="Test split ratio for Random Forest evaluation (default: 0.2).",
    )
    parser.add_argument(
        "--rf-random-state",
        type=int,
        default=42,
        help="Random seed for Random Forest split/model (default: 42).",
    )
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=300,
        help="Number of trees for Random Forest (default: 300).",
    )
    return parser


def _parse_gamma(gamma_raw: str) -> str | float:
    """Parse gamma argument as float when possible; fallback to original string."""
    try:
        return float(gamma_raw)
    except ValueError:
        return gamma_raw


def _detect_separator(csv_path: Path, fallback_sep: str) -> str:
    """Infer CSV separator from header; fallback to CLI-provided separator."""
    with csv_path.open("r", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
    if not sample.strip():
        return fallback_sep

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=COMMON_CSV_DELIMITERS)
        return dialect.delimiter
    except csv.Error:
        return fallback_sep


def _load_pipeline_input(csv_path: Path, sep: str) -> pd.DataFrame:
    """Load raw or notebook-preprocessed input CSVs.

    Heuristic: Kaggle raw files contain both `Date` and `Time`; notebook outputs
    store cleaned rows (typically with `Datetime`) and are treated as preprocessed.
    """
    inferred_sep = _detect_separator(csv_path, fallback_sep=sep)
    header = pd.read_csv(csv_path, sep=inferred_sep, nrows=0)
    header_columns = set(header.columns)

    if {"Date", "Time"}.issubset(header_columns):
        return load_and_clean_air_quality_csv(str(csv_path), sep=inferred_sep)
    return load_preprocessed_air_quality_csv(str(csv_path), sep=inferred_sep)


def main() -> None:
    """Execute full pipeline and write labeled/removed/repaired outputs."""
    parser = build_parser()
    args = parser.parse_args()

    # Validate input path early for clear user feedback.
    if not args.csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv_path}")
    if args.train_rf and not 0.0 < args.rf_test_size < 1.0:
        raise ValueError("--rf-test-size must be between 0 and 1.")
    if args.train_rf and args.rf_n_estimators <= 0:
        raise ValueError("--rf-n-estimators must be a positive integer.")

    # Ensure output directory exists before writing any files.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline: load -> feature prep -> train -> label -> branch processing.
    df = _load_pipeline_input(args.csv_path, sep=args.sep)
    features = prepare_feature_matrix(df)
    baseline = train_ocsvm_baseline(
        features,
        nu=args.nu,
        kernel=args.kernel,
        gamma=_parse_gamma(args.gamma),
    )
    labeled_df = generate_anomaly_labels(baseline, features)
    removed_df = removal_branch(labeled_df)
    repaired_df = repair_branch(labeled_df, strategy=args.repair_strategy)

    # Standardized output file names for downstream project workflow.
    labeled_path = args.output_dir / f"{args.output_prefix}_labeled.csv"
    removed_path = args.output_dir / f"{args.output_prefix}_removed.csv"
    repaired_path = args.output_dir / f"{args.output_prefix}_repaired.csv"

    labeled_df.to_csv(labeled_path, index=False)
    removed_df.to_csv(removed_path, index=False)
    repaired_df.to_csv(repaired_path, index=False)

    print("Pipeline completed successfully.")
    print(f"Labeled output : {labeled_path}")
    print(f"Removed output : {removed_path}")
    print(f"Repaired output: {repaired_path}")

    if args.train_rf:
        print("\nRandom Forest results:")
        runs = [
            train_random_forest_on_dataframe(
                labeled_df,
                dataset_name="labeled",
                target_column=args.rf_target_column,
                test_size=args.rf_test_size,
                random_state=args.rf_random_state,
                n_estimators=args.rf_n_estimators,
            ),
            train_random_forest_on_dataframe(
                removed_df,
                dataset_name="removed",
                target_column=args.rf_target_column,
                test_size=args.rf_test_size,
                random_state=args.rf_random_state,
                n_estimators=args.rf_n_estimators,
            ),
            train_random_forest_on_dataframe(
                repaired_df,
                dataset_name="repaired",
                target_column=args.rf_target_column,
                test_size=args.rf_test_size,
                random_state=args.rf_random_state,
                n_estimators=args.rf_n_estimators,
            ),
        ]

        for run in runs:
            if run.skipped:
                print(f"- {run.dataset_name}: skipped ({run.reason})")
                continue

            roc_auc_str = f"{run.roc_auc:.4f}" if run.roc_auc is not None else "N/A"
            print(
                f"- {run.dataset_name}: "
                f"accuracy={run.accuracy:.4f}, "
                f"precision={run.precision:.4f}, "
                f"recall={run.recall:.4f}, "
                f"f1={run.f1:.4f}, "
                f"roc_auc={roc_auc_str}"
            )


if __name__ == "__main__":
    main()
