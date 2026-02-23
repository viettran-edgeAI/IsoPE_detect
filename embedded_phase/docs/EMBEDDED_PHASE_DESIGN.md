# Embedded Phase Design (Authoritative)

Date: 2026-02-24  
Scope: `embedded_phase/`

## 1) Purpose

This document is the single source of truth for the embedded Isolation Forest architecture and refactor direction.

The embedded phase is organized into two implementation tasks:

1. **Task 1 — Core model library** under `core/models/isolation_forest/`.
2. **Task 2 — Model engine packaging/runtime layer** under `src/model_engine/`.

Legacy architecture descriptions, historical benchmark/test/report workflows, and stale design history are intentionally excluded.

---

## 2) Current Target Architecture

### Task 1 — Core model library (`core/models/isolation_forest/`)

**Scope**: C++ developers, header-focused, no cross-platform CMake.

Owns domain model logic:
- model resource discovery and validation
- tree/container representation
- training and inference domain behavior
- model serialization/deserialization

Alternate usage patterns (e.g., direct header inclusion, library linkage, testing) are developer-responsibility. No CMake configuration required.

**Refactor direction**: `IsoForest` is the single integrated model owner that aggregates all components (if_base, if_config, quantizer, scaler, feature_extractor, if_tree_container).

### Task 2 — Model engine (`src/model_engine/`)

**Scope**: Production deployment, CMake required.

Owns runtime packaging and integration:
- endpoint-facing API boundary
- runtime model loading and lifecycle
- **CMake packaging and deployable integration surface** (required for production)
- diagnostics/error plumbing for runtime usage
- cross-platform build reproducibility and distribution

Task 2 wraps Task 1; Task 2 does not re-implement core model logic. Task 2 brings production rigor (CMake, deployment, versioning, artifact management).

---

## 3) Core Component Responsibilities

### `If_base`

Responsibility:
- owns model identity and resource root (`model_name`, `dir_path`)
- constructs canonical model file paths
- validates presence/readiness of required resources

Required constructor shape:
- `If_base(model_name, dir_path = ".")`

### `If_tree`

Responsibility:
- represents one Isolation Tree
- encapsulates node-level traversal/path-length behavior
- provides per-tree scoring primitives

### `If_tree_container`

Responsibility:
- owns and manages all `If_tree` instances of one model
- owns node-resource internals used for packed-tree layout
- serializes/deserializes forest artifacts
- provides ensemble-level tree iteration/scoring support

Design rule:
- node-resource internals remain encapsulated in `If_tree_container`.

### `IsoForest`

Responsibility:
- single high-level orchestration class for model lifecycle
- aggregates `If_base`, `If_config`, `quantizer`, `scaler`, `feature_extractor`, and `if_tree_container`
- exposes load/train/save/infer workflow as the primary API surface
- owns tree-level training orchestration and inference pipeline encapsulation

Expected constructor:
- `IsoForest(model_name, dir_path = ".")`

---

## 4) IsoForest Detailed Responsibilities

`IsoForest` owns and manages these integrated components:
- `if_base`: model identity and resource root
- `if_config`: configuration and dataset parameters
- `quantizer`: feature quantization (`eml_quantizer<problem_type::ISOLATION>`)
- `scaler`: feature standardization
- `feature_extractor`: PE feature extraction (resource: `${model_name}_optimized_features.json`)
- `if_tree_container`: ensemble of trained trees

---

## 5) Model Resource Naming and Location

Model resources are addressed by `model_name + dir_path`.

Canonical file set:
- `${dir_path}/${model_name}_iforest.bin`
- `${dir_path}/${model_name}_dp.txt`
- `${dir_path}/${model_name}_optimized_config.json`
- `${dir_path}/${model_name}_qtz.bin`
- `${dir_path}/${model_name}_optimized_features.json`

All Task 1 and Task 2 loading logic must derive paths from this convention only.

---

## 6) Initialization Sequence (Required)

Initialization must fail fast if any required resource is missing.

Ordered flow:
1. Initialize `if_base` with `model_name` and `dir_path`.
2. Scan/check all model resources derived from `if_base`.
3. If any required file is missing, return initialization failure.
4. Initialize `if_config`, `quantizer`, `scaler`, and `feature_extractor`.
5. Initialize `if_tree_container` and reserve tree slots.

Required model resources:
- `${dir_path}/${model_name}_iforest.bin` (when loading an existing model)
- `${dir_path}/${model_name}_dp.txt`
- `${dir_path}/${model_name}_optimized_config.json`
- `${dir_path}/${model_name}_qtz.bin`
- `${dir_path}/${model_name}_optimized_features.json`

---

## 7) Training Flow (Required)

Training is owned by `IsoForest`.

Required flow:
1. Load benign train file into RAM.
2. For each tree in loop:
	- create sub-dataset for that tree
	- initialize tree
	- train tree
	- add trained tree to `if_tree_container`
3. After training completes, release benign train data from RAM.

The implementation may use rebuilt internal classes, but this lifecycle behavior is mandatory.

---

## 8) Inference API and Pipeline (Required)

Inference API is defined at `IsoForest` level.

Method contract:
- input: PE file path/content
- output: `eml_isolation_result_t`

Encapsulated inference pipeline inside `IsoForest`:
1. PE file input
2. feature extraction (`feature_extractor`)
3. standardization (`scaler`)
4. quantization (`eml_quantizer`)
5. ensemble scoring via trees in `if_tree_container`
6. aggregate and evaluate anomaly result
7. return `eml_isolation_result_t`

No external caller should manually orchestrate these sub-steps.

---

## 9) Data Semantics (Isolation)

`if_data` uses `eml_data<problem_type::ISOLATION>` with **1-bit label semantics**:
- `0` = Negative / benign
- `1` = Positive / malware

Any train/eval/inference path must preserve this mapping exactly.

---

## 10) File Ownership Boundaries

- `core/models/isolation_forest/`: domain model internals (tree logic, config binding, serialization, inference behavior)
- `core/ml/`: shared ML result/status/metrics types used by core models
- `src/model_engine/`: runtime wrapper, packaging, integration API, deployment-oriented loading lifecycle

Ownership intent: keep model logic centralized in core; keep runtime integration concerns in model_engine.

---

## 11) Implementation Phases

### Phase 1 — Core abstractions (Task 1: C++ developer headers)
- finalize `If_tree` and `If_tree_container` boundaries
- keep tree-container IO format stable and documented
- validation: header compilation, no external dependencies beyond STL + eml

### Phase 2 — `IsoForest` consolidation (Task 1: C++ developer headers)
- route lifecycle through `IsoForest(model_name, dir_path)`
- unify resource discovery through `If_base`
- validation: direct header usage, standalone C++ linking

### Phase 3 — Shared type normalization (Task 1: C++ developer headers)
- centralize result/status/metrics contracts in `core/ml/`
- remove ad-hoc duplicates across modules
- validation: type consistency across Task 1 modules

### Phase 4 — Model engine packaging (Task 2: Production CMake)
- keep `src/model_engine/` as thin runtime/integration layer over Task 1
- enforce model loading by `model_name + dir_path`
- introduce CMake targets, versioning, artifact management for production deployment
- validation: CMake builds cleanly, produces deployable binaries

### Phase 5 — Completion validation (Task 1 + Task 2)
- Task 1: load/save/reload functional validation (C++ unit tests)
- Task 2: deterministic inference consistency validation across runtime paths (CMake-driven e2e tests)
- cross-artifact validation: serialized models exchangeable between Task 1 (dev) and Task 2 (prod)

---

## 12) Acceptance Criteria

### Task 1 complete when (C++ developer-focused headers)
- `IsoForest(model_name, dir_path)` performs strict resource scanning and fails on missing required files
- `IsoForest` initializes `if_base`, `if_config`, `quantizer`, `scaler`, `feature_extractor`, and `if_tree_container`
- `IsoForest` training follows the required RAM load → per-tree sub-dataset loop → RAM release lifecycle
- `IsoForest` inference takes PE input and returns `eml_isolation_result_t` via a fully encapsulated pipeline
- old fragmented structure can be removed without blocking the final design
- headers compile standalone without CMake (developer can link directly)
- zero dependency on header-external build tools

### Task 2 complete when (Production CMake deployment)
- model_engine loads model artifacts via `model_name + dir_path`
- model_engine remains packaging/runtime only (no duplicated core model logic)
- CMake packaging/runtime integration is clean and deployable
- e2e runtime validation: save model (Task 1) → load via CMake binary (Task 2) → infer identically
- cross-platform CMake build works on target deployment OS/architecture

---

## 13) Out of Scope

The following are not part of this design document:
- legacy architecture variants
- historical benchmark/test utility programs
- old reporting pipelines and stale benchmark artifacts

This document is implementation-oriented and forward-only.

---

## 14) Supplementary Guidance from `random_forest/` Reference

This section is **supplementary** to all instructions above (sections 1-13). It does not replace or delete any prior requirements.

Use `core/models/random_forest/` as a structural reference for how components/resources are:
- declared
- arranged
- initialized
- connected through one top-level model owner

### 14.1 Reference adoption rule

Adopt the `RandomForest`-style component wiring pattern for `IsoForest`, with one required change:
- in Isolation Forest, tree-building/training logic is pushed down to tree-level behavior (`If_tree`) and orchestrated through `if_tree_container`
- avoid keeping training ownership as a monolithic model-layer block copied from `RandomForest`

### 14.2 Component/resource parity target

When rebuilding internals, keep this parity in mind:
- `Rf_base` → `if_base` (model identity + resource scanning)
- `Rf_config` → `if_config` (config + data parameter binding)
- `eml_quantizer` usage pattern → same for Isolation (`problem_type::ISOLATION`)
- model preprocessor component pattern (e.g., HOG-related in RF) → `feature_extractor` in Isolation
- `Rf_tree_container` ownership pattern → `if_tree_container`
- top-level model owner (`RandomForest`) → single `IsoForest`

### 14.3 Initialization arrangement (supplement to Section 6)

Follow RF-like initialization ordering discipline:
1. initialize `if_base` first
2. validate/scan all model resources
3. initialize dependent components (`if_config`, quantizer, scaler, feature extractor)
4. initialize `if_tree_container` and reserve capacity

This ordering is mandatory unless a stricter ordering is required by correctness.

### 14.4 Resource arrangement notes

Resource naming remains the Isolation Forest canonical set already defined in this document.

In addition, feature extraction resource wiring must follow model-scoped naming discipline:
- `${dir_path}/${model_name}_optimized_features.json`

### 14.5 Explicit exclusions from RF reference

Ignore RF components that serve higher-complexity MCU workflows not required by current Isolation Forest scope, including but not limited to:
- dynamic modeling hooks
- adaptation feedback loops
- online retraining/pending-data workflows
- auxiliary runtime logging/report subsystems that are not part of the core IF contract

The RF reference is used for architecture style and initialization structure, not for importing unrelated complexity.

---

## 11) Supplementary Guidance from `random_forest/` Reference

This section is **supplementary** to all instructions above. It does not replace or delete any prior requirements.

Use `core/models/random_forest/` as a structural reference for how components/resources are:
- declared
- arranged
- initialized
- connected through one top-level model owner

### 11.1 Reference adoption rule

Adopt the `RandomForest`-style component wiring pattern for `IsoForest`, with one required change:
- in Isolation Forest, tree-building/training logic is pushed down to tree-level behavior (`If_tree`) and orchestrated through `if_tree_container`
- avoid keeping training ownership as a monolithic model-layer block copied from `RandomForest`

### 11.2 Component/resource parity target

When rebuilding internals, keep this parity in mind:
- `Rf_base` → `if_base` (model identity + resource scanning)
- `Rf_config` → `if_config` (config + data parameter binding)
- `eml_quantizer` usage pattern → same for Isolation (`problem_type::ISOLATION`)
- model preprocessor component pattern (e.g., HOG-related in RF) → `feature_extractor` in Isolation
- `Rf_tree_container` ownership pattern → `if_tree_container`
- top-level model owner (`RandomForest`) → single `IsoForest`

### 11.3 Initialization arrangement (supplement to Section 4)

Follow RF-like initialization ordering discipline:
1. initialize `if_base` first
2. validate/scan all model resources
3. initialize dependent components (`if_config`, quantizer, scaler, feature extractor)
4. initialize `if_tree_container` and reserve capacity

This ordering is mandatory unless a stricter ordering is required by correctness.

### 11.4 Resource arrangement notes

Resource naming remains the Isolation Forest canonical set already defined in this document.

In addition, feature extraction resource wiring must follow model-scoped naming discipline:
- `${dir_path}/${model_name}_optimized_features.json`

### 11.5 Explicit exclusions from RF reference

Ignore RF components that serve higher-complexity MCU workflows not required by current Isolation Forest scope, including but not limited to:
- dynamic modeling hooks
- adaptation feedback loops
- online retraining/pending-data workflows
- auxiliary runtime logging/report subsystems that are not part of the core IF contract

The RF reference is used for architecture style and initialization structure, not for importing unrelated complexity.
