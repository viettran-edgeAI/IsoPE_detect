# Results Directory Overview

The `development_phase/results/` directory stores Stage-3 export artifacts produced by `development_phase/src/model_optimization.py`.

## Current artifact contract

Stage-3 exports a **split artifact set** (model-name prefixed):

- `<model_name>_optimized_config.json`
  - Model hyperparameters, thresholding config, evaluation metrics, and references to split preprocessing artifacts.

- `<model_name>_scaler_params.json`
  - Scaler parameters (`mean`, `scale`) for parity/audit workflows.

- `<model_name>_feature_schema.json`
  - Feature transform metadata (for example, indices/features requiring log transform).

- `<model_name>_optimized_features.json`
  - Ordered optimized feature names and feature-group mapping.

## Notes

- Files in this directory are overwritten on each optimization run.
- Embedded and development paths should consume the same model-name-prefixed artifacts.
