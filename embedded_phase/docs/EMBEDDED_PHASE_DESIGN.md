# Embedded Phase Design (Current Authoritative State)

Date: 2026-02-24  
Scope: `embedded_phase/`

## 1) Purpose

This document is the current source of truth for the embedded Isolation Forest implementation and evaluation status.

Project target remains:
- **FPR < 0.05**
- **TPR > 0.95**

Given current findings, the project is running in **dual mode**:
1. **Quantized mode** (resource-efficient embedded path)
2. **Normal/Plain mode** (non-quantized baseline path)

---

## 2) Current Dual-Mode Architecture

### 2.1 Quantized mode (production-oriented embedded path)

Primary files:
- `embedded_phase/core/models/isolation_forest/if_model.h`
- `embedded_phase/core/models/isolation_forest/if_components.h`
- `embedded_phase/core/models/isolation_forest/if_base.h`

Key properties:
- Uses scaler + quantizer + packed/compact tree representation.
- Uses model resources in `embedded_phase/core/models/isolation_forest/resources`.
- Uses C++ threshold calibration (development-phase threshold/offset reuse has been removed).
- Supports two calibration entry points:
  - quantized validation BIN calibration
  - raw-PE validation directory calibration (`calibrate_threshold_from_pe_validation`)

Current status:
- Functional and integrated.
- Still shows a significant accuracy gap on raw PE end-to-end inference (details in benchmark section).

### 2.2 Normal mode (plain float baseline)

Primary files:
- `embedded_phase/core/models/isolation_forest/if_plain_float.h`
- `development_phase/src/if_plain_float_eval.cpp`

Key properties:
- No quantization.
- No compressed node format.
- Uses simple node structure (feature index, float threshold, child indices, leaf metadata).
- Trains on optimized CSV datasets from development_phase.
- Uses optimized configuration and optimized feature set from development artifacts.

Current status:
- Implemented and runnable.
- Produces strong TPR, with FPR slightly above target on test in current run.

---

## 3) Resource and Data Contracts

### 3.1 Quantized mode resources

Canonical resource root:
- `embedded_phase/core/models/isolation_forest/resources`

Important files:
- `iforest_qtz.bin`
- `iforest_dp.txt`
- `iforest_optimized_config.json`
- `iforest_optimized_features.json`
- `iforest_ben_train_nml.bin`
- `iforest_ben_val_nml.bin`
- `iforest_mal_val_nml.bin`

### 3.2 Plain mode datasets (optimized CSV)

- `development_phase/data/optimized/iforest_ben_train.csv`
- `development_phase/data/optimized/iforest_ben_val.csv`
- `development_phase/data/optimized/iforest_ben_test.csv`
- `development_phase/data/optimized/iforest_mal_val.csv`
- `development_phase/data/optimized/iforest_mal_test.csv`

### 3.3 Shared config/artifact sources

- `development_phase/results/iforest_optimized_config.json`
- `development_phase/results/iforest_optimized_features.json`

---

## 4) Benchmark Report (Latest)

### 4.1 Quantized mode

#### A) Quantized-domain validation (BIN-domain calibration check)
- Threshold: `-0.542177`
- Validation FPR: `0.034629`
- Validation TPR: `0.915842`
- Interpretation: quantized-domain separation is decent.

#### B) Raw PE domain calibration + evaluation (current operational view)
Calibration on:
- `datasets/BENIGN_VALIDATION_DATASET`
- `datasets/MALWARE_VALIDATION_DATASET`

Selected threshold:
- `-0.655372`

Validation:
- Benign success/fail: `4986 / 14`
- Malware success/fail: `2178 / 2`
- FPR: `0.049138`
- TPR: `0.750230`

Test:
- Benign success/fail: `4994 / 6`
- Malware success/fail: `4227 / 0`
- FPR: `0.051262`
- TPR: `0.718476`

#### C) Quantized mode conclusion
- FPR is near target after C++ raw-domain calibration.
- TPR is far below target in raw end-to-end mode.
- Root issue is not only thresholding; representation/domain mismatch remains significant.

### 4.2 Plain mode (non-quantized baseline)

Run configuration (from optimized config):
- `n_estimators=200`
- `max_samples=1.0`
- `max_features=1.0`
- `bootstrap=false`
- `random_state=42`
- target FPR cap for threshold selection: `0.05`

Selected threshold:
- `-0.506791`

Validation:
- Benign rows: `4967`
- Malware rows: `404`
- FPR: `0.048118`
- TPR: `0.925743`

Test:
- Benign rows: `4948`
- Malware rows: `4194`
- FPR: `0.057599`
- TPR: `0.959943`

#### Plain mode conclusion
- TPR meets target on test (>0.95).
- FPR is close but slightly above target on test (~0.058).
- This mode is currently the strongest accuracy baseline in C++.

---

## 5) Problems Encountered (Quantized Path)

1. **Threshold transfer mismatch**
   - Development-phase threshold/offset did not transfer reliably to embedded quantized runtime.
   - Fixed by moving threshold tuning to C++ calibration paths.

2. **Raw-domain degradation**
   - Even with raw-domain threshold calibration, quantized mode cannot recover required TPR at low FPR.
   - Indicates quantization + runtime extraction domain shift impact.

3. **Calibration trade-off ceiling**
   - Under FPR cap near 5%, quantized mode currently plateaus around ~0.72–0.75 TPR in raw PE evaluation.

---

## 6) Comparative Evaluation

### Quantized mode
Pros:
- Embedded-friendly memory/layout.
- Existing integration path in core/model_engine.

Cons:
- Fails current detection target in raw PE operation.
- Sensitive to domain mismatch and information loss.

### Plain mode
Pros:
- Much stronger detection performance profile.
- Easier interpretability and debugging.

Cons:
- Larger model/runtime footprint.
- Not yet integrated as production engine backend switch.

---

## 7) Current Decision

For now, maintain **both modes**:

1. **Quantized mode** stays as embedded deployment path under active improvement.
2. **Plain mode** stays as non-quantized benchmark/reference path and fallback baseline for detection quality.

Operationally:
- Accuracy reference should be tracked against plain mode.
- Quantized mode should be treated as optimization path until it meets target constraints.

---

## 8) Next Engineering Priorities

1. Add backend selection in model_engine (`quantized` vs `plain`).
2. Add unified benchmark harness to report both modes in one run.
3. Add feasibility reports at multiple FPR caps (`1%, 2%, 5%, 10%`) for transparent trade-off tracking.
4. Investigate quantized-path recovery options:
   - feature-space alignment checks,
   - split policy and score calibration updates,
   - selective de-quantization for high-impact features.

---

## 9) Acceptance Gate (Current)

A mode is considered deployment-ready only when both conditions hold on test:
- FPR < 0.05
- TPR > 0.95

Current snapshot:
- Quantized mode: **not ready** (TPR gap).
- Plain mode: **near-ready baseline** (TPR pass, slight FPR miss).
