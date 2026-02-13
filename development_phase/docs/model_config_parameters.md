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
- val_fpr_delta: If val_fpr_target is not set, the target is computed as fpr_threshold minus val_fpr_delta. Positive values loosen the threshold, negative values tighten it.
- strategy: How the decision threshold is selected. Options are fpr, f1, tpr, youden, model. The choice shifts the precision/recall trade-off.
- f_beta: Beta used when strategy is f1. Higher beta emphasizes recall, lower beta emphasizes precision.
