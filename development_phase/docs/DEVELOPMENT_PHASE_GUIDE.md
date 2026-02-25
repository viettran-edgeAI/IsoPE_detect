# Development Phase Guide: PE Malware Detection

Version: 2.4
Date: 2026-02-21

## Overview
The development phase produces the final feature set and optimized Isolation Forest model for static PE anomaly detection. The pipeline now uses three programs in development_phase/src:
- feature_extraction.py: heavy one-time extraction of raw PE features.
- feature_selection.py: filtering/selection from raw datasets to cleaned datasets (including variance_threshold).
- model_optimization.py: model tuning, feature ranking (top-k), final artifact export, and malware val/test separation analysis.

All datasets at each stage (raw, cleaned, optimized) share the same feature order and columns.

## Sealed-Test Policy
- Test splits (`benign_test`, `malware_test`) are holdout-only for final evaluation.
- Stage 1 schema building is derived from non-test splits only.
- Stage 2 feature filtering/selection is derived from `benign_train` only.
- Stage 3 grid search/model ranking is validation-driven. Test exposure during search is disabled by default (`thresholding.expose_test_during_search=false`).

## Required Dataset Layout
These directories must exist at the workspace root:
- BENIGN_TRAIN_DATASET
- BENIGN_VALIDATION_DATASET
- BENIGN_TEST_DATASET
- MALWARE_VALIDATION_DATASET
- MALWARE_TEST_DATASET

## Directory Structure
```
development_phase/
  src/
    feature_extraction.py
    feature_selection.py
    feature_selection_config.json
    feature_extraction_config.json
    model_optimization.py
    model_config.json
  data/
    raw/
    cleaned/
    optimized/
  reports/
  results/
  schemas/
```

## Environment Setup
Recommended: Python 3.11+

Install dependencies:
```
pip install lief scikit-learn pandas numpy pyarrow scipy matplotlib umap-learn
```

## Stage 1: Raw Feature Extraction
Script: development_phase/src/feature_extraction.py

Run:
```
cd /home/viettran/Documents/visual_code/EDR_AGENT/development_phase/src
python feature_extraction.py
```

CLI parameters:
- --workers: number of parallel workers (default: cpu_count - 1).

Config: development_phase/src/feature_extraction_config.json

Outputs:
- Raw datasets: development_phase/data/raw/*_raw.parquet
- Feature schema: development_phase/schemas/feature_schema.json
- Note: canonical schema is derived from non-test splits to avoid test leakage.

## Stage 2: Filtering and Selection (Raw -> Cleaned)
Script: development_phase/src/feature_selection.py

Run:
```
cd /home/viettran/Documents/visual_code/EDR_AGENT/development_phase/src
python feature_selection.py --config feature_selection_config.json
```

Config: development_phase/src/feature_selection_config.json

Key filtering parameters:
- variance_threshold: base variance gate for low-information features.
- corr_threshold: correlation pruning threshold.
- norm_var_threshold: normalized variance gate for continuous features.

Outputs:
- Cleaned datasets: development_phase/data/cleaned/*_clean.parquet
- Selected schema: development_phase/schemas/feature_schema_selected.json
- Report: development_phase/reports/feature_selection_report.md
- Note: feature filtering/selection runs on benign train only; test splits are transform-only outputs.  
- Stage-2 independence manifest is written to development_phase/reports/malware_val_test_independence_manifest_stage2.json.

## Stage 3: Optimization + Malware Split Analysis
Script: development_phase/src/model_optimization.py

Run:
```
cd /home/viettran/Documents/visual_code/EDR_AGENT/development_phase/src
python model_optimization.py --config model_config.json
```

Useful Stage-3 audit CLI overrides:
- `--audit-similarity-profile-threshold` (default: 0.995)
- `--audit-distribution-diagnostic-k` (default: 5)
- `--audit-split-auc-classifier-folds` (default: 3)

This script loads the cleaned datasets, selects top-k features using Cohen's d, runs grid search over Isolation Forest parameters, and exports split embedding artifacts in `results/`: model config, scaler params, feature schema, and optimized feature list. If raw malware parquet files are present, it computes imphash overlap as a family proxy.  
The Stage-3 manifest records dataset counts and audit parameters.

### Config File: development_phase/src/model_config.json

#### data
Paths to cleaned datasets used for training and evaluation.
- train_benign_path
- val_benign_path
- test_benign_path
- test_malware_path
- val_malware_path

#### feature_selection
Controls feature selection before model training.
- top_k_list: list of top-k values to evaluate using Cohen's d ranking.

#### model
Isolation Forest hyperparameters.
- n_estimators: number of trees (int or list).
- max_samples: subsample size per tree (float, int, or "auto").
- contamination: expected anomaly rate (float or list).
- max_features: fraction of features per tree (float).
- bootstrap: use bootstrapping (bool).
- random_state: RNG seed (int).
- n_jobs: parallelism for scikit-learn (int).

#### thresholding
Controls how decision thresholds are selected.
- fpr_threshold: max allowed FPR on validation data.
- strategy: one of fpr | f1 | tpr | youden | model.
- f_beta: beta for F-score when strategy is f1.
- val_fpr_target: optional explicit FPR target on validation.
- val_fpr_delta: required if val_fpr_target is not set; uses fpr_threshold - val_fpr_delta.
- expose_test_during_search: if false (default), test metrics are not computed or shown during grid search.

#### independence_audit
Simplified Stage-3 independence configuration.
- stage2_manifest_path
- max_cross_similarity
- mmd_permutations
- require_min_malware_val_samples
- max_malware_val_percent_of_benign_train (supports percent-style values such as 5 or 8)
- independence_metrics_json
- independence_assessment_md

Stage-3 diagnostic sampling behavior:
- kNN diagnostic sample size per split is automatic: `min(len(malware_val), 2000)`.
- kNN diagnostic seed is fixed at `42`.
- MMD sample size per split is automatic: `len(malware_val)`.

#### outputs
Where artifacts and reports are written.
- report_dir
- results_dir
- output_prefix
- optimized_config_json
- scaler_params_json
- feature_schema_json
- optimized_feature_set_json
- optimized_data_dir
- results_csv
- optimized_params_json
- optimized_features_csv
- roc_curve_svg
- pr_curve_svg
- score_distribution_svg
- score_distribution_val_svg

### Outputs
- Optimized datasets: development_phase/data/optimized/*_optimized.parquet
- Embedding artifacts:
  - development_phase/results/<model_name>_optimized_config.json
  - development_phase/results/<model_name>_scaler_params.json
  - development_phase/results/<model_name>_feature_schema.json
  - development_phase/results/<model_name>_optimized_features.json
- Optimization report: development_phase/reports/optimization_results.csv
- Plots: development_phase/reports/roc_curve.svg and pr_curve.svg
- Malware split analysis:
  - development_phase/reports/malware_val_test_umap.png
  - development_phase/reports/malware_val_test_tsne.png
  - development_phase/reports/malware_val_test_embedding.csv
  - development_phase/reports/malware_val_test_cluster_report.md

## Pipeline Run Order
1) Raw extraction:
```
python feature_extraction.py
```

2) Filtering/selection:
```
python feature_selection.py --config feature_selection_config.json
```

3) Optimization:
```
python model_optimization.py --config model_config.json
```

## Troubleshooting
- If extraction fails on many files, verify LIEF can parse your PE samples.
- If malware split plots fail, ensure matplotlib and umap-learn are installed.
- If no configuration meets the FPR constraint, relax thresholding.fpr_threshold or expand the grid in model_config.json.
- If sealed-test checks fail at startup in Stage 3, verify `val_*` and `test_*` paths point to different files.
