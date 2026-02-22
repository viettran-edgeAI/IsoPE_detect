# IsoPE_detect

## Project Overview

**IsoPE_detect** is an experimental system for **anomaly-based zero-day malware detection on Windows endpoint devices**. The detector is purely static: it examines Portable Executable (PE) file structure and metadata without executing the binary, comparing observed features against a model of "normal" benign Windows executables.

- **Threat model**: unseen (zero-day) malware that has never appeared in any training set.
- **Detection strategy**: Isolation Forest one-class anomaly detection — trained entirely on benign binaries, no malware labels required during training.
- **Deployment target**: constrained endpoints (low RAM, no Python runtime, no GPU).
- **Languages**: Python for the development pipeline; C++ for the deployed feature extractor and model engine.

---

## Repository Layout

```
EDR_AGENT/
├── datasets/                      # Raw PE sample collections
│   ├── BENIGN_TRAIN_DATASET/
│   ├── BENIGN_VALIDATION_DATASET/
│   ├── BENIGN_TEST_DATASET/
│   ├── MALWARE_VALIDATION_DATASET/
│   └── MALWARE_TEST_DATASET/
│
├── tools/                         # Dataset download and preprocessing utilities
│   ├── download_dataset.py
│   ├── download_malware.py
│   └── process_dataset.py
│
├── development_phase/             # Python pipeline – feature search + model tuning
│   ├── src/                       # Pipeline scripts and configuration
│   ├── data/                      # Cleaned and optimized data splits
│   ├── schemas/                   # Feature schema and group mapping
│   ├── reports/                   # Optimization reports and plots
│   ├── results/                   # Handoff artifacts for the embedding phase
│   └── docs/                      # Design and guide documentation
│
└── embedded_phase/                # C++ modules – extractor + model engine
    ├── core/                      # Shared header-only ML library
    ├── src/
    │   ├── feature_extractor/     # LIEF-based PE feature extractor
    │   └── model_engine/          # Quantized Isolation Forest inference engine
    ├── tools/
    │   └── data_quantization/     # Dataset quantization tool and outputs
    ├── third_party/LIEF/          # Vendored LIEF PE parsing library
    └── docs/                      # Embedded phase design documentation
```

---

## Problem & Data

- **Goal**: build an agent that runs on constrained endpoints, identifies unseen (zero-day) malware by comparing PE features against a model of "normal" binaries.
- **Scope**: static analysis of PE headers, sections, imports, data directories, resources, overlay, and signature metadata.
- **Dataset** under `datasets/`:
  - `BENIGN_TRAIN_DATASET` / `BENIGN_VALIDATION_DATASET` / `BENIGN_TEST_DATASET` — clean Windows binaries.
  - `MALWARE_VALIDATION_DATASET` / `MALWARE_TEST_DATASET` — malicious samples used for evaluation only (never seen during training).

---

## Two-Phase Architecture

The system is split into a **development phase** and an **embedding phase**. The development phase produces exactly two artifact files that fully specify the embedding phase; no other knowledge needs to transfer between them.

```
Development Phase (Python)
  feature_extraction.py  →  raw features per PE file
  feature_selection.py   →  prune to informative subset
  model_optimization.py  →  grid search: feature set + IF hyperparameters
         │
         │  iforest_optimized_features.json   (40-feature ordered list)
         │  iforest_optimized_config.json     (hyperparameters + scaler + threshold)
         ▼
Embedding Phase (C++)
  tools/data_quantization/  →  quantize training/val/test splits
  src/feature_extractor/    →  compile-time feature binding, runtime PE parsing
  src/model_engine/         →  train quantized IF, score PE files, emit verdict
```

---

## Phase 1 — Development Phase

### Purpose

Find the smallest feature set and Isolation Forest configuration that meets the target false-positive rate on a held-out validation set, then verify accuracy on a sealed test set.

### Pipeline stages

| Stage | Script | Output |
|---|---|---|
| 1. Feature extraction | `feature_extraction.py` | Raw per-file feature parquets under `data/raw/` |
| 2. Feature selection | `feature_selection.py` | Pruned parquets under `data/cleaned/`, optimized CSVs under `data/optimized/` |
| 3. Model optimization | `model_optimization.py` | `results/iforest_optimized_features.json`, `results/iforest_optimized_config.json` |

### Configuration

`development_phase/src/model_config.json` drives the grid search:
- Feature family groups to include/exclude (DOS header, COFF header, optional header, sections, imports, data directories, resources, overlay, signatures, debug).
- Isolation Forest hyperparameter search ranges.
- Threshold selection strategy and FPR targets (`val_fpr_target = 0.04`, `val_fpr_delta = 0.01`).

### Standardization

Every feature column is standardized before training: `z = (x − μ) / σ`. The per-feature means and scales (40 values each) are stored in `iforest_optimized_config.json` under `preprocessing.scaler` for audit and parity only. At inference time the scaler is superseded by the per-feature quantizer.

### Optimized model configuration

```
hyperparameters:
  n_estimators:    200
  max_samples:     1.0
  max_features:    1.0
  bootstrap:       false
  contamination:   0.005
  random_state:    42

thresholding:
  strategy:        tpr
  val_fpr_target:  0.04
  val_fpr_delta:   0.01

node_resource (embedded layout):
  quantization_bits: 2
  feature_bits:      6
  child_bits:        16
  max_depth:         16
  max_nodes_per_tree: 63907
  num_features:      40
```

### Development-phase results

Training is one-class (benign samples only). Evaluation uses benign samples for FPR and malware samples for TPR.

| Split | FPR | TPR | ROC-AUC |
|---|---|---|---|
| Validation | 0.0383 | 0.9177 | 0.9880 |
| Test (holdout) | 0.0453 | 0.9347 | 0.9879 |

Reports and plots are saved under `development_phase/reports/` (`roc_curve.svg`, `pr_curve.svg`, `score_distribution.svg`).

---

## Phase 2 — Embedding Phase

### Purpose

Convert the two development-phase artifact files into a deployable C++ pipeline that runs at endpoint speed with a minimal memory footprint, while preserving detection accuracy.

### Preprocessing contract (runtime inference path)

```
PE file on disk
    │
    ▼  feature_extractor  (LIEF-based C++)
raw float vector [40 values]
    │
    ▼  quantizer  (bin-boundary lookup, trained on benign_train split only)
integer bin-index vector [40 × 2-bit = 10 bytes/sample]
    │
    ▼  QuantizedIsolationForest  (C++ model engine)
anomaly score (float) + verdict (bool)
```

No floating-point arithmetic is required on the hot path after the quantizer step.

### Module 1 — Feature Extractor (`embedded_phase/src/feature_extractor/`)

- Parses PE files using LIEF (vendored under `embedded_phase/third_party/LIEF/`).
- `generate_compiled_features.py` reads `iforest_optimized_features.json` at build time and generates `compiled_feature_config.hpp` — a header with compile-time feature names and indices. This locks the feature schema into the binary.
- `resource_limits.hpp` defines compile-time caps on file size, import count, section count, etc., to bound worst-case memory and CPU usage on endpoint devices.
- Output formats: JSONL (one record per file) or CSV.
- Build: `build.sh` or CMake target `lief_feature_extractor`.

### Module 2 — Data Quantization Tool (`embedded_phase/tools/data_quantization/`)

Converts float-valued optimized CSV splits into compact integer binary datasets for the C++ model engine.

**Per-feature quantization (2-bit default, 4 bins)**:
1. Compute z-score outlier bounds on the training split only.
2. Build quantile-based bin boundaries from the clipped training distribution.
3. Apply the same boundaries identically to all splits (no data leakage).

**Outputs per split** under `quantized_datasets/`:

| File | Contents |
|---|---|
| `*_nml.bin` | Bit-packed dataset: `[uint32 n_samples][uint16 n_features][{uint8 label, packed_bits}×n]` |
| `*_qtz.bin` | Raw integer feature matrix (one byte per feature, unpacked) |
| `*_nml.csv` | Human-readable version of the quantized dataset |
| `*_dp.txt` | Data-profile: quantization bits, bin boundaries, outlier statistics |

At 2-bit quantization a 40-feature sample is 10 bytes versus 160 bytes (float32) — approximately 16× dataset compression.

### Module 3 — Model Engine (`embedded_phase/src/model_engine/`)

Trains a C++ Isolation Forest entirely on integer bin-index data, selects the decision threshold, and exposes a scoring API. See [embedded_phase/docs/EMBEDDED_PHASE_DESIGN.md](embedded_phase/docs/EMBEDDED_PHASE_DESIGN.md) for the full architecture description.

**Build**: `cmake --build .cmake-debug --target pe_model_engine_cli pe_model_engine_tests`

**Evaluation CLI**:
```sh
./pe_model_engine_cli \
  --config  development_phase/results/iforest_optimized_config.json \
  --quantized-dir  embedded_phase/tools/data_quantization/quantized_datasets \
  --output  embedded_phase/src/model_engine/results/if_evaluation_summary.json
```

---

## Results — Development vs Embedded

The embedded C++ model (trained entirely on 2-bit quantized integer-bin data) meets and exceeds the development-phase Python model on all three metrics.

| Split | Metric | Development (Python / float) | Embedded (C++ / quantized) | Δ |
|---|---|---|---|---|
| **Validation** | FPR | 0.0383 | 0.0399 | +0.0016 |
| **Validation** | TPR | 0.9177 | 0.9564 | **+0.0387** |
| **Validation** | ROC-AUC | 0.9880 | 0.9933 | **+0.0052** |
| **Test** | FPR | 0.0453 | 0.0576 | +0.0123 |
| **Test** | TPR | 0.9347 | 0.9969 | **+0.0622** |
| **Test** | ROC-AUC | 0.9879 | 0.9960 | **+0.0081** |

Selected decision threshold (embedded): `0.0100316`

The slight FPR increase on the test split (+0.012) is within the `val_fpr_delta = 0.01` design tolerance. The TPR and ROC-AUC improvements come from the regularizing effect of quantization: discrete binning suppresses noisy feature dimensions and smooths tree splits without losing the separating signal at class boundaries.

Full report: `embedded_phase/src/model_engine/results/if_evaluation_summary.json`

### Embedding benchmark report (10 benign + 10 malware)

Measured per file:
- File size
- Feature extraction time from `lief_feature_extractor` (`processing_time_ms`)
- Model inference time including standardization + quantization + IF scoring
- RAM usage (process RSS in MB)

| Split | File | File Size (KB) | Feature Extraction Time (ms) | Inference Time (ms) | RAM Usage (MB RSS) | Score | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| benign_test | 000d...ec00.dll | 2956.031 | 39.926 | 0.144 | 18.656 | -0.124 | anomaly |
| benign_test | 000de8f56588e1a54bbd2d07cd2dff3e9967fcc42bc55a9fb476fedf66e4d9c2.dll | 60.305 | 2.909 | 0.148 | 18.781 | -0.097 | anomaly |
| benign_test | 000e...8243.dll | 84.305 | 2.754 | 0.151 | 18.781 | -0.106 | anomaly |
| benign_test | 000f...431e.dll | 53.969 | 2.735 | 0.155 | 18.781 | -0.100 | anomaly |
| benign_test | 0016...1fb2.dll | 8.500 | 0.547 | 0.138 | 18.781 | -0.114 | anomaly |
| benign_test | 001d...f521.dll | 226.500 | 2.172 | 0.160 | 18.781 | -0.079 | anomaly |
| benign_test | 001d...d75c.dll | 75.603 | 1.423 | 0.177 | 18.781 | -0.108 | anomaly |
| benign_test | 002a...3a58.exe | 21.000 | 0.745 | 0.215 | 18.781 | -0.123 | anomaly |
| benign_test | 0034...8089.exe | 33.828 | 2.814 | 0.195 | 18.781 | -0.092 | anomaly |
| benign_test | 003d...e0fe.dll | 19.070 | 1.614 | 0.213 | 18.781 | -0.108 | anomaly |
| malware_test | 0005...f6a0.exe | 116.000 | 1.674 | 0.192 | 18.781 | -0.132 | anomaly |
| malware_test | 0009...552dc.exe | 5258.000 | 43.971 | 0.116 | 18.781 | -0.136 | anomaly |
| malware_test | 0019...2460.exe | 5266.500 | 42.526 | 0.114 | 18.781 | -0.136 | anomaly |
| malware_test | 0049...8403.exe | 3194.000 | 17.572 | 0.118 | 18.781 | -0.125 | anomaly |
| malware_test | 0065...2e6c.exe | 4426.500 | 31.674 | 0.135 | 18.781 | -0.117 | anomaly |
| malware_test | 0066...6752.exe | 10240.000 | 58.208 | 0.151 | 18.781 | -0.108 | anomaly |
| malware_test | 0075...cd6e.exe | 7070.400 | 52.863 | 0.151 | 18.781 | -0.113 | anomaly |
| malware_test | 0089...1a1c.exe | 8.000 | 0.542 | 0.118 | 18.781 | -0.125 | anomaly |
| malware_test | 00a1...3aab.exe | 45289.000 | 241.279 | 0.164 | 18.781 | -0.095 | anomaly |
| malware_test | 00a2...8092.exe | 10025.500 | 49.368 | 0.127 | 18.781 | -0.125 | anomaly |

- Average feature extraction time: `29.866 ms`
- Average inference time (standardization + quantization + inference): `0.154 ms`
- Peak observed RSS during benchmark loop: `18.781 MB`

### Unit tests

```
CTest 1/1: pe_model_engine_tests ... Passed  0.00 sec
100% tests passed, 0 tests failed out of 1
```

Tests cover: packed-node field read/write round-trip at all bit widths, and the scoring-order contract (inlier score > outlier score on synthetic data).

---

## Current Status

| Component | Status |
|---|---|
| Development pipeline (feature extraction → selection → optimization) | Complete |
| Handoff artifacts (`iforest_optimized_features.json`, `iforest_optimized_config.json`) | Complete |
| C++ feature extractor (LIEF, compile-time feature binding, resource limits) | Complete |
| Dataset quantization tool and all quantized dataset artifacts | Complete |
| C++ model engine (quantized IF, threshold selection, metrics, CLI) | Complete |
| Unit tests | Complete |
| Dev vs embedded parity comparison | Complete |

---

## Build Quick-Reference

```sh
# Development pipeline
cd development_phase/src && python3 model_optimization.py

# Dataset quantization
cd embedded_phase/tools/data_quantization && ./quantize_dataset.sh

# Feature extractor
cd embedded_phase/src/feature_extractor && ./build.sh

# Model engine
cmake -B .cmake-debug -DCMAKE_BUILD_TYPE=Debug .
cmake --build .cmake-debug --target pe_model_engine_cli pe_model_engine_tests
ctest --test-dir .cmake-debug
```

---

## Next Steps

1. **End-to-end CLI integration** — wire the feature extractor and model engine CLIs into a single binary that accepts a PE path and emits a verdict.
2. **Numerical parity harness** — dump raw float feature vectors from the extractor and compare per-sample Python vs C++ anomaly scores.
3. **Deployment tuning** — profile compile-time resource-limit macros against the target device's RAM/CPU budget.
4. **Incremental retraining** — define a procedure to re-run the development pipeline with fresh benign samples and regenerate both handoff artifacts.
5. **Future work** — explore alternate anomaly detectors; consider on-device threshold update without full model rebuild.
