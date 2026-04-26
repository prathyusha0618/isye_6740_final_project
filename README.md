# isye_6740_final_project
isye 6740 final project

## Implemented Python modules

The following modules are added under `/src`:

- `ocsvm_baseline.py`
  - Load and clean `AirQuality.csv`
  - Load notebook-preprocessed CSV (`AirQuality_preprocessed.csv`)
  - Prepare numeric feature matrix
  - Train OC-SVM baseline
- `anomaly_labeling.py`
  - Generate anomaly labels and OC-SVM scores
- `branches.py`
  - `removal_branch`: remove anomalous rows
  - `repair_branch`: repair anomalous rows via median/mean replacement
- `random_forest_baseline.py`
  - Train/evaluate Random Forest on labeled/removed/repaired outputs

### Quick usage

```python
from src import (
    load_and_clean_air_quality_csv,
    load_preprocessed_air_quality_csv,
    prepare_feature_matrix,
    train_ocsvm_baseline,
    generate_anomaly_labels,
    removal_branch,
    repair_branch,
)

df = load_and_clean_air_quality_csv("AirQuality.csv")
# or: df = load_preprocessed_air_quality_csv("AirQuality_preprocessed.csv")
X = prepare_feature_matrix(df)
baseline = train_ocsvm_baseline(X)
labeled = generate_anomaly_labels(baseline, X)

removed_df = removal_branch(labeled)
repaired_df = repair_branch(labeled, strategy="median")
```

### Run full pipeline from CLI

`final project.ipynb` now writes `AirQuality_preprocessed.csv` after preprocessing.

Use `run_pipeline.py` to load either raw `AirQuality.csv` or notebook-preprocessed
`AirQuality_preprocessed.csv`, then generate three files:
- labeled output
- anomaly-removed output
- anomaly-repaired output

```bash
python run_pipeline.py /path/to/AirQuality_preprocessed.csv --output-dir /path/to/out
```

Optional parameters:
- `--output-prefix` (default: `air_quality`)
- `--nu` (default: `0.05`)
- `--kernel` (default: `rbf`)
- `--gamma` (default: `scale`)
- `--repair-strategy` (`median` or `mean`, default: `median`)

To train Random Forest on all three outputs after pipeline generation:

```bash
python run_pipeline.py /path/to/AirQuality_preprocessed.csv --output-dir /path/to/out --train-rf
```

Random Forest options:
- `--rf-target-column` (default: `is_anomaly`)
- `--rf-test-size` (default: `0.2`)
- `--rf-random-state` (default: `42`)
- `--rf-n-estimators` (default: `300`)
