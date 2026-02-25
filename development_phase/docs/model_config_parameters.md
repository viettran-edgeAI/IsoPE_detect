# Model Config Parameters (Non-Path)

This document explains the non-path parameters in development_phase/src/model_config.json and how they affect results.

Note: low-level filtering parameters such as variance/correlation are now configured in development_phase/src/feature_selection_config.json and applied by development_phase/src/feature_selection.py.

## feature_selection
- top_k_list: List of candidate feature counts kept after Cohen's d ranking. Smaller k reduces dimensionality and can improve generalization, but may underfit. Larger k can improve recall but may raise false positives.

## model (Isolation Forest)
- n_estimators: Number of trees. More trees usually improve stability and accuracy at the cost of longer training and more memory.
- max_samples: Subsample size per tree. Smaller values increase randomness and speed, but may reduce accuracy. Larger values use more data per tree and can improve accuracy but increase cost.
- contamination: Expected anomaly rate. This sets the internal offset used by predict and the zero-threshold in decision_function. In this pipeline, thresholds are selected from validation scores, so contamination mostly shifts scores but does not change their ordering. It matters only if threshold_strategy is set to model.
- max_features: Fraction of features per tree. Lower values increase randomness and can reduce overfitting. Higher values use more signals per tree and can improve accuracy when signal is spread across features.
- bootstrap: If true, trees sample with replacement, increasing diversity. This can help generalization on noisy data but may add variance on small datasets.
- random_state: RNG seed for reproducibility. Changing this changes sampling and can slightly change results.
- n_jobs: Parallelism for training. Affects runtime only, not model quality.

## thresholding
- fpr_threshold: Maximum allowed false positive rate on validation. Lower values make the model more conservative (lower FPR, lower TPR). Higher values allow more alerts for higher recall.
- val_fpr_delta: Required when val_fpr_target is not set. The target is computed as fpr_threshold minus val_fpr_delta.
- strategy: How the decision threshold is selected. Options are fpr, f1, tpr, youden, model. The choice shifts the precision/recall trade-off.
- f_beta: Beta used when strategy is f1. Higher beta emphasizes recall, lower beta emphasizes precision.

## independence_audit
- stage2_manifest_path: Path to the Stage-2 independence manifest used for consistency checks.
- max_cross_similarity: Hard gate threshold for max cross-split cosine similarity.
- mmd_permutations: Number of permutations used by the MMD RBF diagnostic.
- require_min_malware_val_samples: Minimum malware validation sample count required by hard checks.
- max_malware_val_percent_of_benign_train: Maximum allowed malware validation size relative to benign train. Supports percent-style values such as 5 or 8 (interpreted as 5% or 8%).
- independence_metrics_json: Output path for Stage-3 independence metrics JSON.
- independence_assessment_md: Output path for Stage-3 independence assessment markdown.

## model_optimization.py CLI overrides (independence diagnostics)
- --audit-similarity-profile-threshold (default: 0.995)
- --audit-distribution-diagnostic-k (default: 5)
- --audit-split-auc-classifier-folds (default: 3)

Diagnostic sampling behavior in Stage 3:
- kNN sample size per split is automatic: min(len(malware_val), 2000).
- kNN seed is fixed at 42.
- MMD sample size per split is automatic: len(malware_val).

## outputs (embedding-focused)
- results_dir: Directory for Stage-3 embedding handoff artifacts.
- output_prefix: Prefix used to derive model-name artifact outputs (default: model name).
- optimized_config_json: Output optimized model config JSON (`<model_name>_optimized_config.json`).
- scaler_params_json: Output scaler parameters JSON (`<model_name>_scaler_params.json`).
- feature_schema_json: Output feature-transform schema JSON (`<model_name>_feature_schema.json`).
- optimized_feature_set_json: Output optimized feature-set JSON (`<model_name>_optimized_features.json`).
