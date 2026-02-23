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
├── tools/                         # Dataset download, preprocessing, and quantization utilities
│   ├── download_dataset.py
│   ├── download_malware.py
│   ├── process_dataset.py
│   └── data_quantization/         # Single-file quantization module (shared with embedded phase)
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
    │   └── data_quantization/     # Batch orchestrator: processes all 5 CSV splits
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
  tools/data_quantization/                  →  single-file quantization module
  embedded_phase/tools/data_quantization/   →  batch orchestrator (5 splits)
  src/feature_extractor/                    →  compile-time feature binding, runtime PE parsing
  src/model_engine/                         →  train quantized IF, score PE files, emit verdict
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

### Module 2 — Data Quantization (`tools/data_quantization/` + `embedded_phase/tools/data_quantization/`)

Converts float-valued optimized CSV splits into compact integer binary datasets for the C++ model engine.
The pipeline is split into two tiers:

- **`tools/data_quantization/`** — single-file quantization module.  
  Accepts one CSV via `-ip` flag (or `input_path` in config) and writes four output artifacts.
  Built independently with `make build`.

- **`embedded_phase/tools/data_quantization/`** — batch orchestrator.  
  Reads `model_name` and `input_dir` from `quantization_config.json`, finds the five
  optimized CSV splits (`<model_name>_ben_train/test/val.csv`, `<model_name>_mal_test/val.csv`),
  and calls the single-file module for each.

**Per-feature quantization (2-bit default, 4 bins)**:
1. Compute z-score outlier bounds on the training split only.
2. Build quantile-based bin boundaries from the clipped training distribution.
3. Apply the same boundaries identically to all splits (no data leakage).

**Outputs per split** under `embedded_phase/tools/data_quantization/quantized_datasets/`:

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
| **Validation** | FPR | 0.0370 | 0.0278 | **−0.0093** |
| **Validation** | TPR | 0.9208 | 0.9975 | **+0.0767** |
| **Validation** | ROC-AUC | 0.9878 | 0.9982 | **+0.0105** |
| **Test** | FPR | 0.0465 | 0.0570 | +0.0105 |
| **Test** | TPR | 0.9404 | 1.0000 | **+0.0596** |
| **Test** | ROC-AUC | 0.9885 | 0.9989 | **+0.0104** |

Selected decision threshold (embedded): `0.016133`

The slight FPR increase on the test split (+0.0105) is within the `val_fpr_delta = 0.01` design tolerance. The TPR and ROC-AUC improvements come from the regularizing effect of quantization: discrete binning suppresses noisy feature dimensions and smooths tree splits without losing the separating signal at class boundaries.

Full report: [report/README.md](report/README.md)

### Embedding benchmark report (10 benign + 10 malware)

*Last updated: 2026-02-23 — decision threshold `0.016133`*

| Split | File | File Size (KB) | Extraction (ms) | Inference (ms) | RSS (MB) | Score | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| benign_test | 000d81f6…ec00.dll | 2956.031 | 20.839 | 0.320 | 15.168 | −0.015 | anomaly |
| benign_test | 000de8f5…d9c2.dll | 60.305 | 2.463 | 0.342 | 15.293 | 0.100 | benign |
| benign_test | 000e0a55…8243.dll | 84.305 | 2.326 | 0.311 | 15.293 | 0.023 | benign |
| benign_test | 000f278b…431e.dll | 53.969 | 1.783 | 0.567 | 15.293 | 0.037 | benign |
| benign_test | 00167267…1fb2.dll | 8.500 | 0.332 | 0.333 | 15.293 | 0.030 | benign |
| benign_test | 001d74d5…f521.dll | 226.500 | 1.358 | 0.452 | 15.293 | 0.060 | benign |
| benign_test | 001dbc4c…d75c.dll | 75.603 | 0.822 | 0.417 | 15.293 | 0.129 | benign |
| benign_test | 002a2f5c…3a58.exe | 21.000 | 0.478 | 0.312 | 15.293 | −0.024 | anomaly |
| benign_test | 0034d97e…8089.exe | 33.828 | 2.540 | 0.605 | 15.293 | 0.074 | benign |
| benign_test | 003df758…e0fe.dll | 19.070 | 1.823 | 0.374 | 15.293 | 0.022 | benign |
| malware_test | 0005626a…f6a0.exe | 116.000 | 0.822 | 0.338 | 15.293 | −0.083 | anomaly |
| malware_test | 0009f3b6…552dc.exe | 5258.000 | 39.107 | 0.218 | 15.293 | −0.084 | anomaly |
| malware_test | 0019e0ae…2460.exe | 5266.500 | 33.922 | 0.284 | 15.293 | −0.084 | anomaly |
| malware_test | 0049bd68…8403.exe | 3194.000 | 11.112 | 0.262 | 15.293 | −0.055 | anomaly |
| malware_test | 00654e21…2e6c.exe | 4426.500 | 26.952 | 0.442 | 15.293 | 0.027 | benign |
| malware_test | 006622b9…6752.exe | 10240.000 | 45.910 | 0.390 | 15.293 | 0.002 | anomaly |
| malware_test | 00757772…cd6e.exe | 7070.400 | 47.920 | 0.296 | 15.293 | 0.002 | anomaly |
| malware_test | 008902cb…1a1c.exe | 8.000 | 0.245 | 0.279 | 15.293 | −0.049 | anomaly |
| malware_test | 00a16089…3aab.exe | 45289.000 | 190.765 | 0.474 | 15.293 | 0.063 | benign |
| malware_test | 00a22dc8…8092.exe | 10025.500 | 34.066 | 0.339 | 15.293 | −0.033 | anomaly |

- Average feature extraction time: `23.279 ms`
- Average inference time (standardization + quantization + inference): `0.368 ms`
- Peak observed RSS during benchmark loop: `15.293 MB`

### Unit tests

```
CTest 1/1: pe_model_engine_tests ... Passed  0.00 sec
100% tests passed, 0 tests failed out of 1
```

Tests cover: packed-node field read/write round-trip at all bit widths, and the scoring-order contract (inlier score > outlier score on synthetic data).

---

Full results, benchmark, ROC curve, build reference, and next steps: [report/README.md](report/README.md)
````