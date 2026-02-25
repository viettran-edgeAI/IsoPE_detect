# Embedded Phase Design (Current Authoritative State)

Date: 2026-02-25  
Scope: `embedded_phase/`

## 1) Purpose

This document is the source of truth for the embedded Isolation Forest runtime.

Deployment target:
- **FPR < 0.05**
- **TPR > 0.95**

The embedded runtime is now **quantized-only**.

---

## 2) Runtime Architecture (Quantized Only)

Primary files:
- `embedded_phase/core/models/isolation_forest/if_model.h`
- `embedded_phase/core/models/isolation_forest/if_components.h`
- `embedded_phase/core/models/isolation_forest/if_base.h`
- `embedded_phase/core/models/isolation_forest/if_config.h`

Key properties:
- Packed-node quantized Isolation Forest runtime.
- Training and calibration operate on quantized NML datasets.
- Runtime inference path accepts quantized feature vectors (`uint8_t*`).
- Float/plain runtime code path has been removed from embedded_phase.

---

## 3) Resource Contract

Canonical resource root:
- `embedded_phase/core/models/isolation_forest/resources`

Core files used by embedded runtime:
- `<model>_dp.txt`
- `<model>_optimized_config.json`
- `<model>_iforest.bin` (for load/inference)
- `<model>_ben_train_nml.bin`, `<model>_ben_val_nml.bin`, `<model>_mal_val_nml.bin` (for training/calibration)

Produced by development phase (split artifact contract):
- `<model>_optimized_config.json` (model + thresholding config)
- `<model>_scaler_params.json` (scaler parameters; exported for parity/audit)
- `<model>_feature_schema.json` (feature-transform schema; exported for parity/audit)
- `<model>_optimized_features.json` (ordered optimized feature set)

Note:
- Embedded quantized runtime does not require float transform/scaler execution.
- Split scaler/schema artifacts are retained as canonical Stage-3 outputs.

---

## 4) Validation Workflow

Recommended validation sequence:
1. Regenerate Stage-3 artifacts from `development_phase/src/model_optimization.py`.
2. Rebuild quantized model artifacts as needed.
3. Run quantized evaluation on test splits and record FPR/TPR/ROC-AUC.

Primary metric gate:
- FPR on benign test must remain below target.
- TPR on malware test must remain above target.

---

## 5) Recent Issues and Resolution

1. **Preprocessing responsibilities were mixed**
   - Resolution: explicit Stage-3 split artifacts (config, scaler, schema, feature list) were standardized.

2. **Legacy compatibility alias remained in runtime**
   - Resolution: alias-based compatibility path was removed (`if_scaler_transform.h` deleted).

3. **Embedded code still contained float/plain IF path**
   - Resolution: float runtime and related embedded headers were removed; embedded runtime is quantized-only.

---

## 6) Current Decision

The embedded phase ships and validates a single runtime path:
- **Quantized Isolation Forest only**.

Any future experimentation with non-quantized baselines should remain outside embedded runtime scope.
