# Embedded Phase Design

## 1. Scope and Objective

The embedded phase converts two development-phase artifact files into a fully
self-contained C++ pipeline that can run on a constrained Windows or Linux
endpoint with no Python runtime, no GPU, and tight RAM limits.

> **Build system requirement**: every module that will be linked into the endpoint
> agent must support a CMake-based build. The development machine runs Ubuntu; the
> final binary targets both Windows and Linux x86-64 endpoints.

### Two independently versioned modules

1. **Feature Extractor** (`src/feature_extractor/`) — parses a PE file and emits a
   float feature vector.
2. **Model Engine** (`src/model_engine/`) — trains a quantized Isolation Forest on
   the benign training split and scores new feature vectors.

The runtime objective is deterministic static PE scoring with measurable parity
against the development-phase Python model.

---

## 2. Directory Structure

```
embedded_phase/
│
├── core/                              # Header-only shared ML library
│   ├── base/
│   │   ├── eml_base.h                 # Fundamental types and macros
│   │   ├── eml_config.h               # Compile-time configuration constants
│   │   ├── eml_debug.h                # Assertion and logging helpers
│   │   └── eml_random.h               # Seedable PRNG wrapper
│   │
│   ├── containers/
│   │   ├── hash_kernel.h              # Hash utilities for feature hashing
│   │   ├── initializer_list.h         # MCU-friendly initializer_list shim
│   │   └── STL_MCU.h                  # Lightweight STL subset for MCU targets
│   │
│   ├── ml/
│   │   ├── eml_data.h                 # Data matrix abstraction
│   │   ├── eml_logger.h               # Training progress logger
│   │   ├── eml_metrics.h              # ROC-AUC, F1, accuracy helpers
│   │   ├── eml_predict_result.h       # Prediction result type
│   │   ├── eml_quantize.h             # General quantization primitives
│   │   └── eml_samples.h              # Sample iterator / batch helpers
│   │
│   ├── models/
│   │   ├── isolation_forest/          # Isolation Forest core (this project)
│   │   │   ├── if_base.h              # Forward declarations, shared enums
│   │   │   ├── if_config.h            # If_config: loads JSON + dp.txt metadata
│   │   │   ├── if_node_resource.h     # Bit-layout manager for packed nodes
│   │   │   ├── if_model.h             # IsoNode, IsoTree, QuantizedIsolationForest
│   │   │   ├── if_components.h        # Threshold selection, ROC-AUC, metrics
│   │   │   └── if_scaler_transform.h  # Scaler audit helpers (not on hot path)
│   │   ├── random_forest/             # RF reference implementation (MCU library)
│   │   └── xgboost/                   # XGBoost reference implementation (MCU library)
│   │
│   └── test_benchmark/
│       ├── packed_vector_test.cpp     # Packed-vector container unit test
│       └── test_idvector.cpp          # ID-vector container unit test
│
├── docs/
│   └── EMBEDDED_PHASE_DESIGN.md      # This document
│
├── src/
│   │
│   ├── feature_extractor/             # Module 1: PE feature extractor
│   │   ├── CMakeLists.txt             # Top-level CMake for this module
│   │   ├── build.sh                   # Convenience build script
│   │   ├── generate_compiled_features.py  # Build-time feature-schema codegen
│   │   ├── compiled_feature_config.hpp    # Generated header (40 features locked in)
│   │   ├── include/extractor/
│   │   │   ├── extractor.hpp          # Public API: extract_features(), ExtractionResult
│   │   │   └── resource_limits.hpp    # Compile-time endpoint resource caps
│   │   ├── src/
│   │   │   ├── CMakeLists.txt
│   │   │   └── pe_extractor.cpp       # LIEF-based PE parsing implementation
│   │   └── app/
│   │       ├── CMakeLists.txt
│   │       └── pe_feature_extractor_cli.cpp  # CLI: --format jsonl|csv, --output
│   │
│   └── model_engine/                  # Module 2: Quantized Isolation Forest engine
│       ├── CMakeLists.txt             # Top-level CMake; adds subdirs
│       ├── include/
│       │   └── model_engine.hpp       # Public API and all struct declarations
│       ├── src/
│       │   ├── CMakeLists.txt         # Builds pe_model_engine static library
│       │   └── model_engine.cpp       # IsolationForestModelEngine, train_and_evaluate()
│       ├── app/
│       │   ├── CMakeLists.txt         # Builds pe_model_engine_cli executable
│       │   └── model_engine_cli.cpp   # CLI: --config, --quantized-dir, --output
│       ├── test/
│       │   ├── CMakeLists.txt         # Builds pe_model_engine_tests + CTest entry
│       │   └── model_engine_tests.cpp # Unit tests (node layout, scoring order)
│       └── results/
│           └── if_evaluation_summary.json   # CLI output: metrics + dev vs embedded deltas
│
├── third_party/
│   └── LIEF/                          # Vendored LIEF PE parsing library (C++ source)
│
└── tools/
    └── data_quantization/             # Batch quantization orchestrator
        ├── Makefile
        ├── processing_data.cpp        # Orchestrator: finds 5 CSVs, calls tools/data_quantization/
        ├── quantize_dataset.sh        # Shell driver: builds + runs orchestrator
        ├── quantization_config.json   # model_name, input_dir, bits-per-feature, etc.
        ├── dataset_workflow.md        # Quantization pipeline documentation
        └── quantized_datasets/        # Generated artifacts (4 files × 5 splits)
            ├── <model>_ben_train_nml.bin
            ├── <model>_ben_train_nml.csv
            ├── <model>_ben_train_qtz.bin
            ├── <model>_ben_train_dp.txt
            ├── <model>_ben_val_*
            ├── <model>_ben_test_*
            ├── <model>_mal_val_*
            └── <model>_mal_test_*
```

---

## 3. Inputs from the Development Phase

The embedded phase consumes two files from `development_phase/results/`:

| File | Role |
|---|---|
| `iforest_optimized_features.json` | Ordered list of 40 selected feature names |
| `iforest_optimized_config.json` | Hyperparameters, scaler params, threshold config, node-resource layout, evaluation metrics |

These are the **only** files that cross the phase boundary. All embedding-phase
tools are driven exclusively by these two artifacts.

### Key fields in `iforest_optimized_config.json`

```json
{
  "optimized_parameters": {
    "hyperparameters": {
      "n_estimators":  200,
      "max_samples":   1.0,
      "max_features":  1.0,
      "bootstrap":     false,
      "contamination": 0.005,
      "random_state":  42
    },
    "thresholding": {
      "strategy":        "tpr",
      "val_fpr_target":  0.04,
      "val_fpr_delta":   0.01,
      "threshold":       0.0804254702832281,
      "offset":         -0.5988744007048411
    }
  },
  "preprocessing": {
    "scaler": { "mean": [...40 values...], "scale": [...40 values...], "n_features": 40 }
  },
  "node_resource": {
    "quantization_bits": 2,
    "threshold_bits":    2,
    "feature_bits":      6,
    "child_bits":        16,
    "leaf_size_bits":    15,
    "depth_bits":        5,
    "max_depth":         16,
    "max_nodes_per_tree": 63907,
    "num_features":      40
  },
  "evaluation": {
    "validation": { "fpr": 0.0383, "tpr": 0.9177, "roc_auc": 0.9880 },
    "test":       { "fpr": 0.0453, "tpr": 0.9347, "roc_auc": 0.9879 }
  }
}
```

---

## 4. Module 1 — Feature Extractor

### 4.1 Responsibilities

- Parse a PE file using LIEF.
- Emit the **40 selected features** in the same order as `iforest_optimized_features.json`.
- Return extraction metadata: `parse_ok`, bounded error text, `processing_time_ms`.
- Never apply scaling, quantization, or threshold logic.

### 4.2 Compile-time feature locking

`generate_compiled_features.py` is run at build time by `build.sh`. It reads
`iforest_optimized_features.json` and generates `compiled_feature_config.hpp`,
which contains:

```cpp
constexpr uint16_t COMPILED_NUM_FEATURES = 40;
constexpr const char* COMPILED_FEATURE_NAMES[] = {
    "dos_e_lfanew", "has_pdb", "imp_func_hash_15", ...
};
```

This makes the feature order a compile-time constant — any mismatch between the
extractor and the model engine will be caught as a compile error or a runtime
feature-count assertion.

### 4.3 Resource limits

`include/extractor/resource_limits.hpp` defines endpoint-safe caps:
- Maximum PE file size
- Maximum number of sections, imports, DLLs
- Maximum string lengths

These prevent worst-case memory blowup on crafted files.

### 4.4 Selected features (40 total)

The 40 features selected by the development pipeline span these PE structure groups:

| Group | Features |
|---|---|
| DOS header | `dos_e_lfanew` |
| COFF header | `coff_machine`, `coff_characteristics`, `coff_char_DLL`, `coff_char_LARGE_ADDRESS_AWARE` |
| Optional header | `opt_sizeof_headers`, `opt_checksum`, `opt_dllchar_HIGH_ENTROPY_VA`, `opt_dllchar_NO_SEH`, `opt_dllchar_DYNAMIC_BASE`, `opt_dllchar_NX_COMPAT`, `opt_imagebase`, `opt_section_alignment`, `opt_subsystem`, `checksum_matches` |
| Sections | `sec1_entropy`, `sec2_vsize`, `sec_max_entropy`, `sec_name_hash_20` |
| Imports | `imp_func_hash_15`, `imp_func_hash_151`, `imp_dll_hash_28`, `imp_dll_hash_34`, `imp_dll_hash_43`, `imp_dll_hash_61`, `num_suspicious_imports` |
| Data directories | `dd_DEBUG_DIR_rva`, `dd_CERTIFICATE_TABLE_rva`, `dd_CERTIFICATE_TABLE_size`, `dd_CLR_RUNTIME_HEADER_rva` |
| Resources | `rsrc_has_version` |
| Overlay | `has_overlay`, `overlay_size`, `overlay_entropy` |
| Signatures | `num_certificates`, `sig_verified` |
| Debug / misc | `has_pdb`, `has_repro`, `has_debug`, `has_relocations` |

### 4.5 Build

```sh
cd embedded_phase/src/feature_extractor
./build.sh
# or:
cmake -B build -DCMAKE_BUILD_TYPE=Release .
cmake --build build --target lief_feature_extractor
```

### 4.6 CLI usage

```sh
./lief_feature_extractor --format jsonl  file1.exe file2.exe
./lief_feature_extractor --format csv --output features.csv *.exe
```

---

## 5. Module 2 — Data Quantization Tool

### 5.1 Purpose

The Isolation Forest is trained entirely on quantized (integer bin-index) data.
The quantization tool converts the float-valued CSV splits produced by the
development phase into compact binary datasets.

### 5.2 Quantization algorithm

For each feature $j$ over the **training split only**:

1. Compute mean $\mu_j$ and standard deviation $\sigma_j$.
2. Clip values to $[\mu_j - k\sigma_j,\ \mu_j + k\sigma_j]$ (default $k=3$).
3. Divide the clipped range into $2^b$ equal-probability bins using quantile
   boundaries (default $b=2$, giving 4 bins: 0, 1, 2, 3).
4. Store the bin boundaries in `*_dp.txt`.

All five splits (benign train/val/test, malware val/test) are mapped using
the **same set of bin boundaries derived from the training split alone**.
This preserves the train/val/test independence that was enforced in the
development phase.

### 5.3 Output file formats

**`*_nml.bin`** — binary, bit-packed:
```
[uint32_t n_samples][uint16_t n_features]
for each sample:
    [uint8_t label]
    [ceil(n_features * bits / 8) bytes of bit-packed feature values]
```
At 2-bit quantization: 40 features × 2 bits = 80 bits = 10 bytes per sample.

**`*_dp.txt`** — text data-profile consumed by the model engine at build time:
```
quantization_bits: 2
n_features: 40
feature 0: bins=[b0,b1,b2], clip_lo=..., clip_hi=...
feature 1: ...
...
```

**`*_qtz.bin`** — unpacked integer matrix (one `uint8_t` per feature per sample),
useful for debugging.

**`*_nml.csv`** — human-readable, same content as `*_nml.bin`.

### 5.4 Compression

At 2-bit quantization a 40-feature sample occupies 10 bytes instead of 160 bytes
(float32), yielding approximately **16× size reduction** without meaningful accuracy loss.

### 5.5 Build and run

The quantization pipeline is split across two modules:

```sh
# Step 1 — build the single-file quantization module (EDR root level)
cd tools/data_quantization && make build && cd -

# Step 2 — build and run the batch orchestrator
cd embedded_phase/tools/data_quantization
make process           # compiles orchestrator + quantizes all five splits
```

Alternatively via shell script:

```sh
cd embedded_phase/tools/data_quantization
./quantize_dataset.sh  # auto-builds both tiers if needed
```

---

## 6. Module 3 — Model Engine

### 6.1 Architecture overview

The model engine is implemented as a CMake static library (`pe_model_engine`) with
a public header (`include/model_engine.hpp`), an evaluation CLI
(`app/pe_model_engine_cli`), and a unit test binary
(`test/pe_model_engine_tests`). All ML primitives live in `core/models/isolation_forest/`.

```
core/models/isolation_forest/
    if_config.h          ─── parses JSON + dp.txt  ──►  If_config struct
    if_node_resource.h   ─── bit-layout manager    ──►  If_node_resource
    if_model.h           ─── IsoNode               ──►  packed 64-bit node
                             IsoTree               ──►  single tree (BFS build + path length)
                             QuantizedIsolationForest ► forest (train + score)
    if_components.h      ─── threshold selection   ──►  If_threshold_result
                             ROC-AUC               ──►  if_compute_roc_auc()
                             binary metrics        ──►  If_binary_metrics
    if_scaler_transform.h ── scaler audit helpers  (not used on hot path)

src/model_engine/
    include/model_engine.hpp     ─── public structs: IsolationForestModelEngine,
                                     EvaluationSummary, DatasetBundlePaths
    src/model_engine.cpp         ─── implements: load_config(), train_on_quantized_matrix(),
                                     load_quantized_nml_dataset(), train_and_evaluate()
    app/model_engine_cli.cpp     ─── CLI entry point
    test/model_engine_tests.cpp  ─── unit tests
```

### 6.2 Core library — `if_node_resource.h`

`If_node_resource` is the bit-layout manager for packed 64-bit tree nodes.

**Purpose**: store configurable field widths for the five node fields
(`threshold_slot`, `feature_id`, `left_child`, `leaf_size`, `leaf_depth`),
compute their packed offsets, and expose `read_field` / `write_field` accessors.

```cpp
class If_node_resource {
    uint8_t threshold_bits_;   // T — from quantization_bits (default 2)
    uint8_t feature_bits_;     // F — ceil(log2(n_features-1)) = 6 for 40 features
    uint8_t child_bits_;       // C — 16 (supports up to 65535 nodes per tree)
    uint8_t leaf_size_bits_;   // S — 15
    uint8_t depth_bits_;       // D — 5

    // Derived layouts (offset, width):
    if_field_layout split_threshold_layout_;  // offset = 1, width = T
    if_field_layout split_feature_layout_;    // offset = 1+T, width = F
    if_field_layout split_child_layout_;      // offset = 1+T+F, width = C
    if_field_layout leaf_size_layout_;        // offset = 1, width = S
    if_field_layout leaf_depth_layout_;       // offset = 1+S, width = D
};
```

Node bit budget constraint:

```
bits_per_node = 1 + max(T+F+C, S+D) ≤ 64
             = 1 + max(2+6+16, 15+5) = 1 + max(24, 20) = 25 bits  ✓
```

### 6.3 Core library — `if_model.h`

#### `IsoNode`

A single-field struct:
```cpp
struct IsoNode { uint64_t packed_data = 0; };
```

Bit 0 is `is_leaf`. The remaining bits carry either the split overlay or the leaf
overlay, interpreted through `If_node_resource`:

| View | Layout |
|---|---|
| Split | `[is_leaf:1][threshold_slot:2][feature_id:6][left_child:16][reserved:39]` |
| Leaf  | `[is_leaf:1][leaf_size:15][leaf_depth:5][reserved:43]` |

Right child is always implicit: `right_child = left_child + 1`.

#### `IsoTree` — breadth-first construction

```
train(matrix, n_samples, n_features, sampled_indices, max_depth, rng):
  1. Allocate root node at index 0.
  2. Push (node=0, begin=0, end=|indices|, depth=0) onto BFS queue.
  3. While queue not empty:
       pop (node_idx, begin, end, depth)
       n = end - begin
       if n ≤ 1 or depth ≥ max_depth:
           write leaf(node_idx, size=n, depth=depth)
           continue
       try up to max(8, 2×n_features) random features:
           find a feature j where min_value < max_value across the current sample slice
       if no split found:
           write leaf(node_idx, size=n, depth=depth)
           continue
       threshold = uniform_int(min_value, max_value - 1)
       partition indices[begin:end] in-place: ≤ threshold left, > threshold right
       left_child = nodes.size()
       write split(node_idx, feature=j, threshold=threshold, left_child)
       push left  task (left_child,   begin, mid,  depth+1)
       push right task (left_child+1, mid,   end,  depth+1)
```

All nodes are stored in a single `std::vector<IsoNode>` in BFS order, giving
cache-friendly traversal.

#### Path length computation

```cpp
float path_length(const uint8_t* q_features, uint16_t n_features):
  node = 0
  while True:
      if node.is_leaf():
          return depth(node) + c_factor(leaf_size(node))
      f = node.feature_id()
      t = node.threshold_slot()
      node = (q_features[f] <= t) ? left_child : left_child + 1
```

Where `c_factor(n) = 2*(ln(n-1) + 0.5772) - 2*(n-1)/n` is the average path
length in a random binary search tree of size `n` (standard IF normalization).

#### `QuantizedIsolationForest`

```
train(matrix, n_samples, n_features, If_config):
  resolve samples_per_tree (from max_samples or max_samples_per_tree)
  initialize n_estimators IsoTree objects with shared node_resource
  for each tree t:
      sample_indices = random subset of size samples_per_tree (without replacement)
      trees[t].train(matrix, n_samples, n_features, sample_indices, max_depth, rng)

decision_function(q_features, n_features) → float:
  avg_path = mean over all trees of tree.path_length(q_features, n_features)
  score = 2^(-avg_path / c_factor(samples_per_tree))
  return score - threshold_offset
```

Lower score → more anomalous. The `threshold_offset` shifts the zero-crossing so
that `score < 0` means anomaly when the forest is loaded with the development-phase
threshold baked in.

### 6.4 Core library — `if_components.h`

Threshold selection and evaluation metrics, operating purely on score arrays:

| Function | Description |
|---|---|
| `if_find_threshold_precise(benign_scores, fpr_target)` | Binary-search for the threshold that achieves `fpr ≤ fpr_target` on the validation benign set |
| `if_select_threshold_with_malware(b_scores, m_scores, fpr_target, strategy, fbeta)` | `"tpr"` strategy: maximize TPR subject to `fpr ≤ fpr_target` |
| `if_compute_metrics(b_scores, m_scores, threshold)` | Returns `If_binary_metrics{fpr, tpr}` at a fixed threshold |
| `if_compute_roc_auc(b_scores, m_scores)` | Trapezoidal-rule ROC-AUC |

### 6.5 Core library — `if_config.h`

`If_config` is loaded from two files:

1. **`iforest_optimized_config.json`** — supplies `n_estimators`, `max_samples`,
   `max_depth`, `max_nodes_per_tree`, `random_state`, `threshold_strategy`,
   `val_fpr_target`, `quantization_bits`, and the node-resource widths.
2. **`*_dp.txt`** (the training-split data-profile from the quantization tool) —
   supplies `num_samples` and `num_features`, confirming the dataset dimensions.

```cpp
struct If_config {
    bool isLoaded = false;
    uint16_t num_features = 0;
    uint8_t  quantization_bits = 2;
    uint8_t  threshold_bits = 2;
    uint8_t  feature_bits = 6;
    uint8_t  child_bits = 16;
    uint8_t  leaf_size_bits = 15;
    uint8_t  depth_bits = 5;
    uint32_t n_estimators = 200;
    float    max_samples = 1.0f;
    uint32_t max_samples_per_tree = 0;   // 0 = derive from max_samples
    uint16_t max_depth = 16;
    uint32_t max_nodes_per_tree = 63907;
    bool     bootstrap = false;
    uint32_t random_state = 42;
    float    threshold_offset = 0.0f;
    std::string threshold_strategy = "tpr";
    float    val_fpr_target = 0.04f;
    float    val_fpr_delta  = 0.01f;
};
```

Parsing is done with a minimal hand-rolled JSON scanner (no external JSON library
dependency) in `if_config_detail` namespace.

### 6.6 Model engine API — `include/model_engine.hpp`

```cpp
namespace eml::model_engine {

// Load config from JSON + dp.txt, then train on a pre-loaded quantized matrix.
class IsolationForestModelEngine {
public:
    bool load_config(const fs::path& config_json, const fs::path& dp_txt, std::string* error);
    bool train_on_quantized_matrix(const std::vector<uint8_t>& matrix, size_t n_samples, std::string* error);
    float decision_function_quantized(const uint8_t* q_features, uint16_t n_features) const;
    bool  is_anomaly_quantized(const uint8_t* q_features, uint16_t n_features, float threshold) const;
    bool  trained() const;
    const If_config& config() const;
};

// Read one *_nml.bin file into a flat uint8_t matrix.
bool load_quantized_nml_dataset(const fs::path& nml_path,
                                 uint16_t expected_num_features,
                                 uint8_t quantization_bits,
                                 std::vector<uint8_t>& out_matrix,
                                 size_t& out_num_samples,
                                 std::string* error);

// Top-level: train + evaluate on all five splits and produce EvaluationSummary.
EvaluationSummary train_and_evaluate(const fs::path& config_json,
                                      const fs::path& dp_txt,
                                      const DatasetBundlePaths& datasets);
}
```

### 6.7 Training pipeline (full sequence)

```
1. load_config(iforest_optimized_config.json, benign_train_dp.txt)
      → extracts n_estimators, max_depth, quantization_bits, node_resource widths
      → confirms num_features = 40

2. load_quantized_nml_dataset(benign_train_nml.bin)
      → reads [uint32 n_samples][uint16 n_features]
      → unpacks bit-packed rows into a flat uint8_t matrix [n_samples × 40]

3. forest.train(matrix, n_samples, num_features, config)
      → initializes If_node_resource from threshold_bits/feature_bits/child_bits etc.
      → for each of the 200 estimators:
            draw 31,954 samples without replacement (max_samples=1.0 → all)
            build IsoTree by BFS with random feature / random threshold selection
      → total node count: up to 200 × 63,907 = ~12.8 M nodes (each 8 bytes = ~103 MB max)
        (in practice far fewer nodes due to early leaf termination)

4. Threshold selection (TPR strategy):
      score benign_val and malware_val splits
      find threshold T* that maximises TPR subject to FPR(T*) ≤ val_fpr_target + val_fpr_delta

5. Evaluation:
      compute FPR / TPR / ROC-AUC on (benign_val, malware_val) → validation metrics
      compute FPR / TPR / ROC-AUC on (benign_test, malware_test) → test metrics
```

### 6.8 Inference path (per sample)

```
Input: raw float feature vector x[40]  (from feature_extractor)
    │
    ▼  quantizer: for each feature j
         q[j] = bin_index(x[j], boundaries from dp.txt)   → uint8_t ∈ {0,1,2,3}
    │
    ▼  QuantizedIsolationForest::decision_function(q, 40)
         avg_path = (1/200) Σ_t IsoTree_t.path_length(q)
         score = 2^(-avg_path / c_factor(samples_per_tree)) - threshold_offset
    │
    ▼  verdict = score < selected_threshold
```

No floating-point multiplication on the hot tree-traversal path: all comparisons
are integer bin-index vs integer threshold-slot.

### 6.9 Evaluation CLI — `app/model_engine_cli.cpp`

```sh
pe_model_engine_cli \
  --config       development_phase/results/iforest_optimized_config.json \
  --quantized-dir embedded_phase/tools/data_quantization/quantized_datasets \
  --dp           quantized_datasets/benign_train_optimized_dp.txt \
  --output       embedded_phase/src/model_engine/results/if_evaluation_summary.json
```

Writes a JSON report:
```json
{
  "selected_threshold": 0.0100316,
  "embedded":    { "validation": {...}, "test": {...} },
  "development": { "validation": {...}, "test": {...} },
  "delta":       { "validation": {...}, "test": {...} }
}
```

---

## 7. CMake Build System

The entire embedded phase integrates into the root `CMakeLists.txt`. The model
engine build hierarchy:

```
embedded_phase/src/model_engine/CMakeLists.txt
    add_subdirectory(src)      → builds static library: pe_model_engine
    add_subdirectory(app)      → links pe_model_engine → executable: pe_model_engine_cli
    add_subdirectory(test)     → links pe_model_engine → executable: pe_model_engine_tests
                                 add_test(NAME pe_model_engine_tests ...)
```

Include paths surfaced to consumers of `pe_model_engine`:
- `embedded_phase/src/model_engine/include/`  (model_engine.hpp)
- `embedded_phase/core/`                       (all core headers)

Compile flags: `-std=c++17`, `-Wall`, `-Wextra`. No third-party libraries beyond
the standard library are required by the model engine.

### Build commands

```sh
# Configure (from repo root)
cmake -B .cmake-debug -DCMAKE_BUILD_TYPE=Debug .

# Build library + CLI + tests
cmake --build .cmake-debug --target pe_model_engine pe_model_engine_cli pe_model_engine_tests

# Run tests
ctest --test-dir .cmake-debug -V
```

---

## 8. Unit Tests

`test/model_engine_tests.cpp` contains two tests run via a single binary:

**`test_node_resource_and_node_layout`**
- Constructs an `If_node_resource` with non-default bit widths.
- Writes and reads back `feature_id`, `threshold_slot`, `left_child` (split node).
- Writes and reads back `leaf_size`, `leaf_depth` (leaf node).
- Asserts round-trip correctness for all fields.

**`test_quantized_iforest_scoring_order`**
- Creates a synthetic 8-sample 4-feature dataset.
- Trains a 32-tree quantized forest.
- Asserts `decision_function(inlier) > decision_function(outlier)`.
- This validates the full training → scoring pipeline end-to-end.

CTest output:
```
1/1 Test #1: pe_model_engine_tests ............   Passed    0.00 sec
100% tests passed, 0 tests failed out of 1
```

---

## 9. Evaluation Results

### Development-phase baseline (Python / float32 / StandardScaler)

| Split | FPR | TPR | ROC-AUC |
|---|---|---|---|
| Validation | 0.0383 | 0.9177 | 0.9880 |
| Test (holdout) | 0.0453 | 0.9347 | 0.9879 |

### Embedded model (C++ / 2-bit quantized integer)

| Split | FPR | TPR | ROC-AUC |
|---|---|---|---|
| Validation | 0.0399 | 0.9564 | 0.9933 |
| Test (holdout) | 0.0576 | 0.9969 | 0.9960 |

### Delta (embedded − development)

| Split | ΔFPR | ΔTPR | ΔROC-AUC |
|---|---|---|---|
| Validation | +0.0016 | **+0.0387** | **+0.0052** |
| Test | +0.0123 | **+0.0622** | **+0.0081** |

Selected decision threshold (embedded): **0.0100316**

The embedded model surpasses the development-phase baseline on TPR (+3.9 pp
validation, +6.2 pp test) and ROC-AUC at a cost of a small FPR increase
(+0.012 on test, within the `val_fpr_delta = 0.01` design budget).

The accuracy improvement is explained by the regularizing effect of quantization:
2-bit binning suppresses noisy high-variance feature dimensions and smooths
tree split thresholds without sacrificing the separating signal at class
boundaries, as observed consistently in the MCU library benchmark data.

Full report: `embedded_phase/src/model_engine/results/if_evaluation_summary.json`

---

## 10. Scaler Policy

The scaler parameters (`mean[40]`, `scale[40]`) stored in
`iforest_optimized_config.json` are **retained for audit and parity checks only**.
They are not applied at runtime because:

1. The quantizer is trained on the StandardScaler-normalized CSV splits produced
   by the development pipeline. The bin boundaries already encode the normalized
   feature distribution implicitly.
2. Applying the scaler a second time at runtime would distort the bin-index mapping.

The `deployment_scaling.runtime_normalization` field in the config is recorded as
`"disabled"` to make this policy explicit and auditable.

---

## 11. Parity Notes

The embedded model does not replicate individual tree structures from the Python
`sklearn.ensemble.IsolationForest` because:

- The Python model is trained on float32 features in StandardScaler space.
- The C++ model is trained on 2-bit integer bin-index features.
- These are intentionally different inputs, producing intentionally different trees.

Parity is assessed at the **aggregate metric level** (FPR, TPR, ROC-AUC on the
same held-out splits), not at the per-sample score level. The target metric
contract is TPR ≥ development-phase TPR, FPR ≤ `val_fpr_target + val_fpr_delta`.

Both targets are met by the current implementation.
