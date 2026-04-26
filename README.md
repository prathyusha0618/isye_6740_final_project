# isye_6740_final_project
isye 6740 final project

## Implemented Python modules

The following modules are added under `/src`:

- `ocsvm_baseline.py`
  - Load and clean `AirQuality.csv`
  - Prepare numeric feature matrix
  - Train OC-SVM baseline
- `anomaly_labeling.py`
  - Generate anomaly labels and OC-SVM scores
- `branches.py`
  - `removal_branch`: remove anomalous rows
  - `repair_branch`: repair anomalous rows via median/mean replacement

### Quick usage

```python
from src import (
    load_and_clean_air_quality_csv,
    prepare_feature_matrix,
    train_ocsvm_baseline,
    generate_anomaly_labels,
    removal_branch,
    repair_branch,
)

df = load_and_clean_air_quality_csv("AirQuality.csv")
X = prepare_feature_matrix(df)
baseline = train_ocsvm_baseline(X)
labeled = generate_anomaly_labels(baseline, X)

removed_df = removal_branch(labeled)
repaired_df = repair_branch(labeled, strategy="median")
```

### Run full pipeline from CLI

Use `run_pipeline.py` to load `AirQuality.csv` and generate three files:
- labeled output
- anomaly-removed output
- anomaly-repaired output

```bash
python run_pipeline.py /path/to/AirQuality.csv --output-dir /path/to/out
```

Optional parameters:
- `--output-prefix` (default: `air_quality`)
- `--nu` (default: `0.05`)
- `--kernel` (default: `rbf`)
- `--gamma` (default: `scale`)
- `--repair-strategy` (`median` or `mean`, default: `median`)
