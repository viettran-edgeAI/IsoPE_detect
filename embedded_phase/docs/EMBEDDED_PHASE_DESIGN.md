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

Owns domain model logic:
- model resource discovery and validation
- tree/container representation
- training and inference domain behavior
- model serialization/deserialization

### Task 2 — Model engine (`src/model_engine/`)

Owns runtime packaging and integration:
- endpoint-facing API boundary
- runtime model loading and lifecycle
- CMake packaging and deployable integration surface
- diagnostics/error plumbing for runtime usage

Task 2 wraps Task 1; Task 2 does not re-implement core model logic.

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
- aggregates `If_base`, configuration, data adapter, and tree container
- exposes load/train/save/infer workflow as the primary API surface

Expected constructor:
- `IsoForest(model_name, dir_path = ".")`

---

## 4) Model Resource Naming and Location

Model resources are addressed by `model_name + dir_path`.

Canonical file set:
- `${dir_path}/${model_name}_iforest.bin`
- `${dir_path}/${model_name}_dp.txt`
- `${dir_path}/${model_name}_optimized_config.json`
- `${dir_path}/${model_name}_qtz.bin`

All Task 1 and Task 2 loading logic must derive paths from this convention only.

---

## 5) Data Semantics (Isolation)

`if_data` uses `eml_data<problem_type::ISOLATION>` with **1-bit label semantics**:
- `0` = Negative / benign
- `1` = Positive / malware

Any train/eval/inference path must preserve this mapping exactly.

---

## 6) File Ownership Boundaries

- `core/models/isolation_forest/`: domain model internals (tree logic, config binding, serialization, inference behavior)
- `core/ml/`: shared ML result/status/metrics types used by core models
- `src/model_engine/`: runtime wrapper, packaging, integration API, deployment-oriented loading lifecycle

Ownership intent: keep model logic centralized in core; keep runtime integration concerns in model_engine.

---

## 7) Implementation Phases

### Phase 1 — Core abstractions
- finalize `If_tree` and `If_tree_container` boundaries
- keep tree-container IO format stable and documented

### Phase 2 — `IsoForest` consolidation
- route lifecycle through `IsoForest(model_name, dir_path)`
- unify resource discovery through `If_base`

### Phase 3 — Shared type normalization
- centralize result/status/metrics contracts in `core/ml/`
- remove ad-hoc duplicates across modules

### Phase 4 — Model engine packaging
- keep `src/model_engine/` as thin runtime/integration layer over Task 1
- enforce model loading by `model_name + dir_path`

### Phase 5 — Completion validation
- load/save/reload functional validation
- deterministic inference consistency validation across runtime paths

---

## 8) Acceptance Criteria

### Task 1 complete when
- `IsoForest(model_name, dir_path)` loads required resources from one directory
- model can be trained/retrained, saved, reloaded, and infer deterministically
- `If_tree_container` is the owner of tree memory + forest serialization boundary

### Task 2 complete when
- model_engine loads model artifacts via `model_name + dir_path`
- model_engine remains packaging/runtime only (no duplicated core model logic)
- CMake packaging/runtime integration is clean and deployable

---

## 9) Out of Scope

The following are not part of this design document:
- legacy architecture variants
- historical benchmark/test utility programs
- old reporting pipelines and stale benchmark artifacts

This document is implementation-oriented and forward-only.
