# Development Phase (Quick Reference)

See the full guide in `DEVELOPMENT_PHASE_GUIDE.md`.

## Pipeline
1. `feature_extraction.py` + `feature_extraction_config.json`
   - One-time heavy PE extraction.
   - Output: `data/raw/*_raw.parquet`.

2. `feature_selection.py` + `feature_selection_config.json`
   - Filtering from raw to cleaned datasets.
   - Includes `variance_threshold`, correlation pruning, and stability filtering.
   - Output: `data/cleaned/*_clean.parquet` and selection schema/report.

3. `model_optimization.py` + `model_config.json`
   - Model optimization on cleaned datasets.
   - Output: `data/optimized/*`, `reports/*`, and `results/*`.
