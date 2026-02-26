# Embedded Phase (Current Authoritative State)

Date: 2026-02-26  
Scope: `embedded_phase/`

This document is the source of truth for the embedded Isolation Forest runtime, deployment packaging, and validation workflow.

## 1) Purpose

Deployment target:
- **FPR < 0.05**
- **TPR > 0.95**

The embedded runtime is **quantized-only**.

---

## 2) Runtime Architecture (Quantized Only)

Primary IF model files:
- `embedded_phase/core/models/isolation_forest/if_model.h`
- `embedded_phase/core/models/isolation_forest/if_components.h`
- `embedded_phase/core/models/isolation_forest/if_base.h`
- `embedded_phase/core/models/isolation_forest/if_config.h`

Key properties:
- Packed-node quantized Isolation Forest runtime.
- Training and calibration operate on quantized NML datasets.
- Runtime inference path accepts quantized feature vectors (`uint8_t*`).
- Float/plain runtime code path has been removed from embedded phase.

### Status/err-code contract

- IF model layer is log-free and returns machine-readable status codes.
- Canonical status enum is defined in:
  - `embedded_phase/core/models/isolation_forest/if_status.h`
- Public inference result carries status via:
  - `embedded_phase/core/ml/eml_predict_result.h` (`eml_isolation_result_t::status_code`)
- Component-level status introspection is available through `last_status()` / `clear_status()` in IF model components.

---

## 2.1) Deployment Packaging Runtime (`src/model_engine`)

Production deployment wrappers are implemented in:
- `embedded_phase/src/model_engine/include/model_engine.hpp` (C++ API)
- `embedded_phase/src/model_engine/include/model_engine_c.h` (C ABI)
- `embedded_phase/src/model_engine/src/model_engine.cpp`
- `embedded_phase/src/model_engine/src/model_engine_c.cpp`

Deployment behavior:
- Loads only inference resources (resources 1..7 from `If_base.h`) from a model resource directory.
- Performs inference on quantized vectors and raw vectors.
- Provides optional PE-path/content inference hooks through extractor callbacks.
- Ships standalone binaries:
  - `pe_model_engine_cli`
  - `pe_model_engine_benchmark_cli`

---

## 3) Resource Contract

Canonical resource root:
- `embedded_phase/core/models/isolation_forest/resources`

Core files used by embedded runtime inference:
1. `<model>_dp.txt`
2. `<model>_qtz.bin`
3. `<model>_optimized_config.json`
4. `<model>_optimized_features.json`
5. `<model>_scaler_params.json`
6. `<model>_feature_schema.json`
7. `<model>_iforest.bin`

Additional files used for training/calibration/evaluation workflows:
- `<model>_ben_train_nml.bin`
- `<model>_ben_val_nml.bin`
- `<model>_mal_val_nml.bin`

Notes:
- Embedded runtime does not require float-transform/scaler execution to score quantized vectors.
- Split scaler/schema artifacts are retained as canonical Stage-3 outputs for parity/audit.

---

## 4) Validation Workflow

Recommended validation sequence:
1. Regenerate Stage-3 artifacts from `development_phase/src/model_optimization.py`.
2. Run `tools/resource_prepairer/prepare_datasets.py` to produce quantized resources for train/validation splits.
3. Place resources into canonical root: `embedded_phase/core/models/isolation_forest/resources`.
4. Validate model-engine runtime by running:
   - `pe_model_engine_tests`
   - `pe_model_engine_benchmark_cli`
5. Optionally run raw-PE evaluator:
   - `tools/model_tester/if_quantized_cpp_raw_pe_eval`
6. Record FPR/TPR/ROC-AUC and compare to deployment gates.

Primary metric gate:
- FPR on benign test must remain below target.
- TPR on malware test must remain above target.

---

## 5) Current Decision

The embedded phase ships and validates a single runtime path:
- **Quantized Isolation Forest only**.

Any future experimentation with non-quantized baselines remains outside embedded runtime scope.

---

## 6) Latest Validation Snapshot (2026-02-26)

Validation workflow rerun completed successfully (build, tests, and runtime evaluation).

Raw-PE quantized C++ evaluation (`development_phase/reports/if_quantized_cpp_raw_pe_eval.json`):
- Threshold: `-0.546877`
- FPR: `0.048058` (**pass** vs `< 0.05`)
- TPR: `0.903714` (**not met** vs `> 0.95`)
- ROC-AUC: `0.983336`

`model_engine` validation-split evaluation (`embedded_phase/src/model_engine/results/if_model_engine_eval_ubuntu.json`):
- Threshold: `-0.546877`
- FPR: `0.038454` (**pass** vs `< 0.05`)
- TPR: `0.915842` (**not met** vs `> 0.95`)
- ROC-AUC: `0.983938`
- AP: `0.860797`
- Confusion counts: `TP=370`, `FP=191`, `TN=4776`, `FN=34`

Build/runtime verification for `model_engine` on Ubuntu:
- CMake build: pass
- CTest (`pe_model_engine_tests`): pass
- Demo inference (`pe_model_engine_cli`): pass (`status=ok`)

Cross-platform build/run instructions:
- `embedded_phase/src/model_engine/MODEL_ENGINE_BUILD_RUN.md`
