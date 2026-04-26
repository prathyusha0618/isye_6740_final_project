"""Run the OC-SVM anomaly pipeline from the command line.

This script reads AirQuality.csv, trains an OC-SVM baseline, generates anomaly
labels, and writes three outputs:
1) labeled data
2) anomaly-removed data
3) anomaly-repaired data
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import (
    generate_anomaly_labels,
    load_and_clean_air_quality_csv,
    prepare_feature_matrix,
    removal_branch,
    repair_branch,
    train_ocsvm_baseline,
)


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
    return parser


def _parse_gamma(gamma_raw: str) -> str | float:
    """Parse gamma argument as float when possible; fallback to original string."""
    try:
        return float(gamma_raw)
    except ValueError:
        return gamma_raw


def main() -> None:
    """Execute full pipeline and write labeled/removed/repaired outputs."""
    parser = build_parser()
    args = parser.parse_args()

    # Validate input path early for clear user feedback.
    if not args.csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv_path}")

    # Ensure output directory exists before writing any files.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline: load/clean -> feature prep -> train -> label -> branch processing.
    df = load_and_clean_air_quality_csv(str(args.csv_path), sep=args.sep)
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


if __name__ == "__main__":
    main()
