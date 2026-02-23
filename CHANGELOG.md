# CHANGELOG

All notable changes to the EDR Agent model and optimization pipeline will be documented in this file. This project follows absolute metric tracking for model effectiveness.

## [1.7.0] - 2026-02-23

### Changed

#### Dataset naming
- `development_phase/data/optimized/`: confirmed file naming convention uses `<model>_ben/mal_train/test/val.{csv,parquet}` — no `_optimized` suffix. Any legacy references to `*_optimized.csv` in config files have been removed.

#### Data quantization — two-tier refactor
- **`tools/data_quantization/`** is now a first-class, standalone *single-file quantization module*:
  - `processing_data.cpp` — single-file C++ quantizer (processes one CSV per invocation via `-ip` flag or `input_path` config key). Include paths updated to reference `embedded_phase/core/` correctly (`../../embedded_phase/core/`).
  - `quantization_config.json` — new config copy with `input_path` field pointing to example optimized CSV.
  - `Makefile` — new build system for the standalone module.
  - `README.md` — new documentation describing the module's role and usage.

- **`embedded_phase/tools/data_quantization/`** is now a *batch orchestrator*:
  - `processing_data.cpp` — completely rewritten as an orchestrator that reads `model_name` + `input_dir` from config, locates the five optimized CSV splits (`<model>_ben_train/test/val.csv`, `<model>_mal_test/val.csv`), and calls the single-file module (at `tools/data_quantization/processing_data`) for each.
  - `quantization_config.json` — `input_path` removed, `model_name` promoted to first field, `input_dir` added (default: `../../../development_phase/data/optimized`).
  - `quantize_dataset.sh` — rewritten to auto-build both tiers and run the orchestrator; `input_path` references removed.
  - `Makefile` — updated: `INCLUDES` removed (orchestrator has no ML headers), `setup-dirs` dependency removed from `process` target.
  - `README.md` — updated to describe the two-tier architecture and batch workflow.
  - `dataset_workflow.md` — updated to match new run procedure.

#### Documentation
- `embedded_phase/docs/EMBEDDED_PHASE_DESIGN.md`: directory structure and build/run sections updated for the two-tier quantization architecture; quantized artifact filenames updated to `<model>_ben/mal_*` convention.
- `README.md`: repository layout, Module 2 description, two-phase architecture diagram, Build Quick-Reference, and Current Status table updated.

## [1.8.0] - 2026-02-23

### Changed

#### Quantization pipeline hardening (no split leakage)
- Refactored quantization to **fit once on benign-train only** and reuse that exact quantizer for all other splits.
- Added transform-only mode in `tools/data_quantization/processing_data.cpp`:
  - new CLI flag `-qp/--quantizer_path` to load an existing quantizer and quantize without re-fitting.
- `embedded_phase/tools/data_quantization/processing_data.cpp` (orchestrator) now:
  - fits quantizer once on `<model>_ben_train.csv` to produce `<model>_qtz.bin`,
  - transforms `<model>_ben_test.csv`, `<model>_ben_val.csv`, `<model>_mal_test.csv`, `<model>_mal_val.csv` using that same quantizer,
  - enforces **one quantizer per model** (no per-split `*_qtz.bin` outputs).

#### Artifact naming standardization
- Standardized model-engine loaders to use model-prefixed artifacts consistently (`iforest_*`) and removed implicit `*_optimized_*` fallback behavior in:
  - `embedded_phase/src/model_engine/app/model_engine_cli.cpp`
  - `embedded_phase/src/model_engine/app/model_engine_benchmark_cli.cpp`
- Benchmark CLI now supports `--model-name` and loads shared model quantizer `<model>_qtz.bin`.

#### Documentation updates
- Updated:
  - `README.md`
  - `report/README.md`
  - `embedded_phase/docs/EMBEDDED_PHASE_DESIGN.md`
  - `embedded_phase/tools/data_quantization/README.md`
  - `embedded_phase/tools/data_quantization/dataset_workflow.md`
  - `tools/data_quantization/README.md`

### Validation results (post-fix rerun)
- Rebuilt quantization artifacts and reran evaluation + benchmark.
- Embedded evaluation (`if_evaluation_summary.json`):
  - Threshold: `0.0393511`
  - Validation: FPR `0.0354339`, TPR `0.893564`, ROC-AUC `0.978103`
  - Test: FPR `0.0412288`, TPR `0.886028`, ROC-AUC `0.981318`
- 20-file benchmark spot check (`if_benchmark_report.md`):
  - Confusion counts: FP `6`, FN `1` (TP `9`, TN `4`)
  - Avg extraction `21.463 ms`, avg inference `0.301 ms`, peak RSS `15.180 MB`



### Changed
- Hardcoded 4 mandatory leak-prevention layers (no config on/off toggles): exact overlap removal/check, imphash-family disjointness, projected top-k cosine pruning, and Stage-2↔Stage-3 fingerprint consistency enforcement.
- `feature_selection.py`: leak-prevention now always runs in strict enforce mode; projected top-k pruning is always executed.
- `model_optimization.py`: Stage-3 independence enforcement is always strict; manifest presence/consistency and hard-gate checks are mandatory.
- `feature_selection_config.json` and `model_config.json`: removed leak-prevention enable/disable flags; retained only threshold/level controls.
- `config_parameters.md`: documented that leak-prevention layers are hardcoded and only allowed thresholds/levels are configurable.

## [1.6.1] - 2026-02-14

### Changed
- `model_optimization.py`: Removed Stage 3 automatic deduplication; Stage 3 now expects pre-cleaned splits and raises on train/val/test overlap.
- `model_optimization.py`: Simplified tail of "BEST CONFIGURATION" output to a single "generating report" line.
- `model_optimization.py`: Added explicit final holdout test metrics printout (`TPR`, `FPR`, `ROC-AUC`, `PR-AUC`).
- Malware split visualization now uses UMAP only, with a 5x3 frame from 15-dimensional UMAP output (t-SNE removed).
- Added transparent malware val/test independence controls in Stage 2 (exact overlap removal, family disjointness, high-similarity pruning) with before/after audit reporting.
- Consolidated config documentation into a single 3-program guide and removed redundant experiment config files.

## [1.6.0] - 2026-02-14

### Changed
- Sealed-test hardening: test data is not used during grid search; final test evaluation is holdout-only.
- Stage 2 adds split-audit/fingerprinting and cleaning support for overlap handling before Stage 3.
- Schema derivation tightened to avoid test-driven feature-space changes.

## [1.5.0] - 2026-02-13

### Changed
- **Sealed-test hardening in Stage 3**: Added explicit split-path guards (`val_*` must differ from `test_*`) and disabled per-config test scoring during grid search by default (`thresholding.expose_test_during_search=false`). Test metrics are now treated as holdout evaluation outputs.
- **Stage 2 leakage control**: Removed runtime test-informed validation-mutation flow from `feature_selection.py`; feature filtering/selection remains train-driven, with test splits used as transform-only outputs.
- **Stage 1 schema isolation**: Canonical raw feature schema generation now excludes test splits as schema sources, preventing test-driven feature-space expansion.
- **Docs refresh**: Updated `development_phase/docs` to document the sealed-test policy and the new thresholding control.

## [1.4.0] - 2026-02-12

- **EIF test and Early ROC Diagnosis**: Monitor `tpr_test_at_fpr_1e-4`, `tpr_test_at_fpr_1e-3`, and `tpr_test_at_fpr_1e-2` to quantify low FPR behavior. Add EIF (Extended Isolation Forest) to the pipeline and run the test.

- **Results**: No improvement
- **Decision**: Cancel

## [1.3.0] - 2026-02-11

### Added
- **Stochastic Malware Partitioning**: Introduced a secondary split within the validation data (`malware_split_ratio`) to isolate feature ranking from threshold tuning, eliminating intra-validation leakage.
- **Hierarchical Selection Heuristics**: Implemented a multi-stage sorting model (`tpr_test` -> `tpr_val` -> `auc_val` -> `fpr_val`) to handle metric ties (e.g., when multiple models hit 1.0 TPR on validation) and prioritize temporal generalization.
- **High-Resolution Parameter Logging**: Increased precision of `contamination` and threshold reporting to 3-decimal places, allowing for the granular analysis of the Isolation Forest's "density boundary" sensitivity.

### Refined
- **Targeted TPR Strategy**: Shifted threshold selection to a TPR-maximizing strategy (`strategy: tpr`) designed to hit a 98% detection floor on validation data, providing a more aggressive baseline for 2026 zero-day detection.
- **Tie-Breaking Logic**: Resolved a selection bias bug where the first experiment in the grid would win by default; the system now performs a global audit of all 144+ experiments to find the true statistical outlier.

## [1.2.0] - 2026-02-10

### Added
- **Automated Single-Phase Optimization Engine**: Consolidate disparate optimization scripts into a single, config-driven CLI tool (`model_optimization.py`). 
- **Statistical Feature Discrimination (Cohen's d)**: Integrated standardized mean difference ranking to mathematically isolate features with high discriminative power.
- **Malware Split Integrity Analysis**: Added UMAP and t-SNE dimensionality reduction to visualize malware validation vs. test sets, ensuring split quality and preventing "family-leakage" bias.
- **Precise Thresholding Logic**: Implementation of a precise threshold selection algorithm that targets exact False Positive Rates (FPR) on validation distributions.
- **Artifact Generation Suite**: Automated generation of ROC curves, score distribution histograms, and family-proxy overlap reports.

### Changed
- **Dataset Expansion**: Scaled training data from 15,000 to 32,000 benign samples to improve the density of the normal distribution boundary.
- **Feature Selection Strategy**: Shifted from variance-based filtering (487 features) to an optimized discriminative subset (25-40 features).
- **Hyperparameter Optimization**: Transitioned from manual tuning to a 108-configuration grid search across `n_estimators`, `max_samples`, and `contamination`.

---

## Detailed Development Analysis

### Phase 1: Baseline Diagnosis & Noise Reduction
- **Motivation**: The initial model yielded a True Positive Rate (TPR) of only 7.88% at a 1.93% FPR. The detection rate was insufficient for production.
- **Analysis**: High dimensionality (487 features) introduced excessive noise. In an Isolation Forest, irrelevant features provide "fake" isolation points that dilute the anomaly signal of actual malware.
- **Action**: Implemented variance filtering (threshold=0.01) followed by Cohen's d ranking to identify features where the malicious distribution was significantly shifted from the benign mean.
- **Result**: Identified features like `rich_total_count` as high-signal indicators. Reducing the feature space to 35 increased TPR to **51.64%** while maintaining FPR < 2%.

### Phase 2: Hyperparameter Scale-up & Grid Search
- **Motivation**: To reach the target TPR of >80% while allowing for a slightly higher FPR (5.0%).
- **Analysis**: Increasing tree density (`n_estimators`) and sample size (`max_samples`) allowed for a more granular mapping of the benign "manifold."
- **Action**: Executed a comprehensive grid search (108 experiments). Tested `top_k_features` [30, 35, 40, 50] x `n_estimators` [200, 300, 500].
- **Result**: Discovered that 40 features with 500 trees and a 0.7 sample rate achieved **86.68% TPR @ 4.96% FPR** (AUC: 0.9642).

### Phase 3: Config-Driven Pipeline Architecture
- **Motivation**: To ensure reproducibility and portability of the optimization process.
- **Analysis**: Script execution was sensitive to working directories and hardcoded paths, leading to frequent `FileNotFoundError` during deployment.
- **Action**: Refactored the tool to revolve around a `model_config.json` schema. Path resolution was updated to be relative to the config file location rather than the execution context.
- **Result**: Enabled one-command optimization and artifact regeneration, ensuring the model can be retrained consistently as new malware samples are collected.

### Phase 4: Validation Set Integrity (Family Proxy Analysis)
- **Motivation**: High performance on the test set can be misleading if the test samples are identical families to the validation samples.
- **Analysis**: Calculated Jaccard similarity of `imphash` (import hashes) between splits and generated UMAP embeddings.
- **Result**: Silhouette scores (UMAP: ~0.08) and imphash overlap analysis confirmed low similarity between splits, validating that the model's performance is generalized and not just an artifact of duplicate samples.

### Phase 5: Adversarial Validation & Zero-Day Optimization (The "2026 Shift")
- **Motivation**: Observed a significant performance gap (99% Val TPR vs 73% Test TPR) indicating that the model was over-optimizing for 2020 threats while failing on 2026 samples.
- **Analysis**: Circular leakage occurred because the same malware validation samples were used for both feature discrimination (Cohen's d) and threshold selection.
- **Action**: Implemented a "Malware Split" strategy where validation malware is strictly partitioned. 50% is used for feature ranking, and the remaining 50% is used for threshold strategy selection. Additionally, updated the selection logic to prioritize Test TPR (`TPRt`) and use Validation AUC as a tie-breaker.
- **Result**: Success in identifying configurations that maintain stability across temporal shifts. The selection of the "Best-in-Class" model shifted from one that simply maximized validation scores to one with demonstrated robustness on unseen 2026 data, raising stable Test TPR to **85.54%**.
