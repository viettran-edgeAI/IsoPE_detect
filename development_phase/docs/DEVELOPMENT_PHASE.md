# Development Phase (Quick Reference)

See the full guide in `DEVELOPMENT_PHASE_GUIDE.md`.

## Pipeline
1. `feature_extraction.py` + `feature_extraction_config.json`
   - One-time heavy PE extraction.
   - Canonical feature schema is derived from non-test splits.
   - Output: `data/raw/*_raw.parquet`.

2. `feature_selection.py` + `feature_selection_config.json`
   - Filtering from raw to cleaned datasets.
   - Includes `variance_threshold`, correlation pruning, and stability filtering.
   - Selection is driven by `benign_train`; test splits are transform-only outputs.
   - Output: `data/cleaned/*_clean.parquet` and selection schema/report.

3. `model_optimization.py` + `model_config.json`
   - Model optimization on cleaned datasets.
   - Grid search/model ranking is validation-driven; keep `thresholding.expose_test_during_search=false` to seal test data.
    - Output: `data/optimized/*`, `reports/*`, and split embedding artifacts in `results/`:
       - `<model_name>_optimized_config.json`
       - `<model_name>_scaler_params.json`
       - `<model_name>_feature_schema.json`
       - `<model_name>_optimized_features.json`
