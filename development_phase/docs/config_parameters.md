# Configuration Parameters

This document centralizes configuration for the 3 development programs:
- `feature_extraction.py` → `development_phase/src/feature_extraction_config.json`
- `feature_selection.py` → `development_phase/src/feature_selection_config.json`
- `model_optimization.py` → `development_phase/src/model_config.json`

For each parameter: purpose, impact, and how to adjust.

## 1) Feature Extraction (`feature_extraction_config.json`)

### Core
| Parameter | Purpose | Impact on process | How to adjust |
|---|---|---|---|
| `version` | Config schema/version marker | Reproducibility and config tracking | Keep aligned with code/schema changes |
| `workers` | Parallel extraction worker count | Throughput vs CPU/RAM usage | Increase on high-core hosts; reduce if memory pressure or instability |
| `parse_timeout` | Max seconds per sample parse | Prevents hangs on malformed binaries; can skip slow files | Increase only if many valid files timeout |
| `max_sections` | Max PE sections examined | More detail/features vs slower extraction | Raise for complex/packed binaries; lower for speed on endpoints |

### Hash bucket sizes
| Parameter | Purpose | Impact on process | How to adjust |
|---|---|---|---|
| `hash_sizes.imp_func` | Import-function hashing dimension | Collision rate vs feature dimensionality | Increase if collisions hurt discrimination; decrease for memory budget |
| `hash_sizes.imp_dll` | Import-DLL hashing dimension | Same tradeoff as above | Tune with `imp_func`; keep moderate for embedded targets |
| `hash_sizes.exp_name` | Export-name hashing dimension | Export signal granularity vs sparsity | Increase only if export signals are useful in your corpus |
| `hash_sizes.sec_name` | Section-name hashing dimension | Captures section naming patterns | Increase for diverse packer families; lower for compact models |

### Data sources
| Parameter | Purpose | Impact on process | How to adjust |
|---|---|---|---|
| `dataset_dirs.benign_train` | Benign training source | Defines normal profile learning base | Keep clean and broad across endpoint software types |
| `dataset_dirs.benign_val` | Benign validation source | Drives threshold false-positive control | Keep independent from train/test |
| `dataset_dirs.benign_test` | Benign test source | Final benign holdout performance | Keep sealed and independent |
| `dataset_dirs.malware_val` | Malware validation source | Stage-2 separation + Stage-3 ranking/thresholding | Keep independent from malware test by sample/family/similarity |
| `dataset_dirs.malware_test` | Malware test source | Final holdout malware detection metric | Keep sealed; never used in training/tuning |

## 2) Feature Selection (`feature_selection_config.json`)

### Input datasets
| Parameter | Purpose | Impact on process | How to adjust |
|---|---|---|---|
| `data.train_benign_path` | Raw benign training features | Source for variance/correlation/stability selection | Must match Stage-1 output and remain clean |
| `data.val_benign_path` | Raw benign validation features | Used in split hygiene and downstream tuning datasets | Keep independent from train/test |
| `data.test_benign_path` | Raw benign test features | Holdout benign set, cleaned/propagated to Stage-3 | Keep sealed |
| `data.val_malware_path` | Raw malware validation features | Split hygiene + Stage-3 validation malware | Keep independent from malware test |
| `data.test_malware_path` | Raw malware test features | Holdout malware set | Keep sealed |

### Filtering
| Parameter | Purpose | Impact on process | How to adjust |
|---|---|---|---|
| `filtering.variance_threshold` | Remove low-variance features | Higher value prunes aggressively, may remove weak signals | Start around current default; increase if noisy, decrease if underfitting |
| `filtering.norm_var_threshold` | Remove near-constant continuous features | Stabilizes continuous features and scale behavior | Increase for stronger cleanup; lower if useful low-variance features are dropped |
| `filtering.corr_threshold` | Correlation pruning threshold | Lower value removes more redundancy | Use 0.90–0.98; lower for compact/robust models |

### Separation / transparency controls
Hardcoded policy (not configurable on/off):
1. Exact overlap removal and sealed-test overlap check
2. Imphash family disjoint enforcement
3. Projected top-k cosine pruning against malware_test
4. Stage-2/Stage-3 fingerprint consistency verification

Only threshold/level values are configurable below.

| Parameter | Purpose | Impact on process | How to adjust |
|---|---|---|---|
| `separation.malware_val_ratio_to_benign_val` | Caps malware validation size relative to benign val | Controls prevalence realism vs tuning stability | Keep moderate (e.g., 0.1–0.5) for stable thresholding |
| `separation.malware_val_min_samples` | Floor for malware validation size | Prevents unstable thresholding from tiny malware_val | Raise if threshold/model selection is noisy |
| `separation.malware_val_ratio_seed` | RNG seed for subset selection | Reproducibility of malware_val subset | Keep fixed for comparable experiments |
| `separation.max_cross_similarity` | Prune very-high-similarity malware_val rows vs malware_test | Reduces near-duplicate leakage risk | Tighten (smaller threshold) for stricter independence; loosen if val size too small |
| `independence_audit.projected_similarity_threshold` | Max allowed projected cosine similarity across all configured top-k views | Enforces strict anti-pattern leakage in Stage-3 feature regime | Keep ≤ `separation.max_cross_similarity`; lower for stricter hygiene |
| `independence_audit.projected_max_iterations` | Max iterative passes to remove projected violators | Controls cleanup completeness vs runtime | Increase if residual violators persist after first passes |
| `independence_audit.require_min_malware_val_after_audit` | Minimum malware_val count after all mandatory pruning | Prevents unreliable tuning due to over-pruning | Increase for statistical stability; decrease only when data is limited |

### Outputs
| Parameter | Purpose | Impact on process | How to adjust |
|---|---|---|---|
| `outputs.cleaned_data_dir` | Cleaned dataset output dir | Feeds Stage-3 training/validation/testing | Keep stable path for pipeline reproducibility |
| `outputs.schema_dir` | Selected schema output dir | Stores selected feature contract | Keep under version control if used by deployment |
| `outputs.report_dir` | Selection/separation report output dir | Transparency and audit trail | Keep persistent for experiment tracking |

## 3) Model Optimization (`model_config.json`)

### Input datasets
| Parameter | Purpose | Impact on process | How to adjust |
|---|---|---|---|
| `data.train_benign_path` | Benign training matrix | Used for model fitting and benign baseline | Keep clean and representative |
| `data.val_benign_path` | Benign validation matrix | Threshold FPR calibration | Keep independent and representative |
| `data.val_malware_path` | Malware validation matrix | Feature ranking + threshold/model validation | Keep independent from malware test |
| `data.test_benign_path` | Benign test matrix | Final holdout FPR | Keep sealed |
| `data.test_malware_path` | Malware test matrix | Final holdout TPR | Keep sealed |

### Feature search
| Parameter | Purpose | Impact on process | How to adjust |
|---|---|---|---|
| `feature_selection.top_k_list` | Candidate feature counts in grid search | Controls model complexity vs generalization | Include small/medium/large candidates; remove extremes if unstable |

### Isolation Forest hyperparameters
| Parameter | Purpose | Impact on process | How to adjust |
|---|---|---|---|
| `model.n_estimators` | Number of trees | More trees improve stability, increase runtime/memory | 100–500 typical; raise if results are noisy |
| `model.max_samples` | Sample fraction/count per tree | Affects bias/variance and speed | Lower for speed/diversity, higher for stronger fit |
| `model.contamination` | Expected anomaly fraction in model internals | Shifts decision function offset; interacts with threshold strategy | Keep aligned with expected prevalence |
| `model.max_features` | Feature fraction per tree | Controls tree diversity and robustness | Lower can reduce overfitting; higher uses more signal per tree |
| `model.bootstrap` | Sampling with replacement toggle | Affects variance and ensemble diversity | Keep both in search if uncertain |
| `model.random_state` | RNG seed | Reproducibility | Keep fixed for comparisons |
| `model.n_jobs` | Parallel jobs | Runtime only | Tune for host capacity |

### Thresholding / model selection
| Parameter | Purpose | Impact on process | How to adjust |
|---|---|---|---|
| `thresholding.fpr_threshold` | Max allowed validation FPR | Primary precision/alert-budget control | Lower for fewer false positives; higher for more recall |
| `thresholding.val_fpr_delta` | Delta to derive `val_fpr_target` | Fine-tunes threshold strictness | Keep near zero for direct alignment |
| `thresholding.strategy` | Threshold selection objective (`fpr`, `f1`, `tpr`, `youden`, `model`) | Controls precision/recall tradeoff | Use `tpr` for recall-heavy, `f1` for balance |
| `thresholding.f_beta` | Beta for `f1`-style strategy weighting | Recall-vs-precision emphasis | >1 favors recall, <1 favors precision |

### Independence audit (mandatory)
Hardcoded policy (not configurable on/off):
1. Stage-2 manifest must exist
2. Stage-2 hard gates must pass
3. Stage-3 fingerprints must match Stage-2 manifest
4. Stage-3 cosine hard threshold must pass

Only threshold/sample/analysis levels are configurable below.

| Parameter | Purpose | Impact on process | How to adjust |
|---|---|---|---|
| `independence_audit.max_cross_similarity` | Stage-3 hard cap for val→test cosine similarity | Primary anti-pattern leakage gate in optimized feature space | Lower for stricter separation; may reduce available validation malware |
| `independence_audit.require_min_malware_val_samples` | Minimum malware_val count accepted by Stage-3 hard gate | Stabilizes validation reliability after strict pruning | Raise for stronger statistical confidence |
| `independence_audit.similarity_profile_thresholds` | Additional reported similarity cutoffs | Improves audit transparency | Keep a ladder near the hard cap (e.g., 0.99/0.995/0.999) |
| `independence_audit.knn_k_list` | k values for cross-split neighborhood diagnostics | Gives local mixing context | Include small and medium k values |
| `independence_audit.knn_sample_size_per_split` | Sample cap for kNN audit | Runtime vs precision tradeoff | Increase for tighter estimates |
| `independence_audit.split_auc_cv_folds` | CV folds for split predictability metric | Stability of separability estimate | Use 5–10 for robust reporting |
| `independence_audit.mmd_permutations` | Permutation count for MMD p-value | Statistical precision vs runtime | Increase for narrower p-value uncertainty |

### Output artifacts
| Parameter | Purpose | Impact on process | How to adjust |
|---|---|---|---|
| `outputs.report_dir` | Report files output | Experiment traceability | Keep persistent for audit |
| `outputs.results_dir` | Exported model/scaler/threshold artifacts | Deployment handoff | Keep stable path consumed by downstream tooling |
| `outputs.optimized_data_dir` | Optimized dataset exports | Reuse in analysis and split diagnostics | Keep consistent across pipeline |
| `outputs.results_csv` | Grid search table output | Model comparison and reproducibility | Keep versioned per run if needed |
| `outputs.optimized_params_json` | Best parameter summary | Deployment/config traceability | Keep under run artifacts |
| `outputs.optimized_features_csv` | Best feature list | Explainability and reproducibility | Track with model results |
| `outputs.roc_curve_svg` | ROC plot artifact | Visual quality check | Optional for automated CI |
| `outputs.pr_curve_svg` | PR curve artifact | Better under class imbalance | Keep for anomaly workloads |
| `outputs.score_distribution_svg` | Test score distribution | Threshold sanity checks | Keep for troubleshooting |
| `outputs.score_distribution_val_svg` | Validation score distribution | Threshold tuning diagnostics | Enable when debugging threshold behavior |
