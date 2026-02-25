# IsoPE_detect

## Overview

IsoPE_detect is a static PE-based malware detection project for endpoint environments. It uses a two-phase workflow: Python-based model development and C++-based embedded runtime deployment. The development phase produces model artifacts, and the embedded phase consumes those artifacts for extraction, scoring, and runtime integration.

## Project Phases

### development_phase

The `development_phase` contains the data science pipeline:

- Data preparation and dataset normalization.
- Feature engineering and feature selection.
- Isolation Forest model optimization and threshold selection.
- Artifact generation for deployment handoff.

Primary output artifacts are generated in `development_phase/results/` and are intended to be consumed by the embedded runtime; datasets trimmed to those features appear under `development_phase/data/optimized/`.


### embedded_phase

The `embedded_phase` contains the C++ implementation for endpoint use:

- **Task 1** — Core model library (`core/models/isolation_forest/`): C++ developer-focused headers and APIs. No cross-platform CMake. Exposes `IsoForest` lifecycle (load/train/save/infer) and `If_tree`/`If_tree_container` primitives.
- **Task 2** — Model engine runtime (`src/model_engine/`): Production packaging with CMake. Thin integration layer over Task 1. Handles deployment-oriented loading, versioning, and cross-platform artifact management.
- Supporting components: PE feature extraction, embedded scoring, diagnostics.

This phase consumes the artifacts produced by `development_phase`.

## Repository Layout

- `development_phase/` — Python pipeline, schemas, configs, reports, and deployment artifacts.
- `embedded_phase/` — C++ core/runtime code, model engine, and embedded tooling.
- `datasets/` — benign and malware datasets used by the pipeline.
- `tools/` — shared utilities for data preparation workflows.

## Getting Started

1. Prepare raw binaries by placing benign/malware samples under the corresponding `datasets/` subdirectories.
2. Run earlier stages if needed (`feature_extraction.py`, `feature_selection.py`), or start at Stage 3 via:
   ```bash
   cd development_phase/src
   python model_optimization.py --config model_config.json
   ```
   which produces optimized CSV datasets and handoff artifacts in `development_phase/results/`.
3. Prepare embedded resources:
   ```bash
   python tools/resource_prepairer/prepare_datasets.py \
       --benign-train development_phase/data/optimized/iforest_ben_train.csv \
       --benign-val development_phase/data/optimized/iforest_ben_val.csv \
       --malware-val development_phase/data/optimized/iforest_mal_val.csv \
       --output-dir embedded_phase/core/models/isolation_forest/resources \
       --quantization-bits 3 --model-name iforest
   ```
   This step fits a quantizer on the train split and transforms the validation splits, emitting the quantized nml/bin artifacts used by the runtime.
4. Build and test the embedded runtime. For example:
   ```bash
   cd embedded_phase/src/feature_extractor && ./build.sh
   cd embedded_phase && cmake -S src/model_engine -B build && cmake --build build
   ```
5. Optionally run the raw-PE evaluation utility:
   ```bash
   /tmp/if_quantized_cpp_raw_pe_eval --repo-root . --model-name iforest
   ```

This covers the full development‑to‑embedded pipeline with automated quantization and deployment artifacts.

## Current Direction

The current focus is to keep a strict development‑to‑embedded contract: artifact generation in `development_phase`, artifact consumption in `embedded_phase`, and progressive refactoring of the embedded stack into a clean core model library plus production-ready `model_engine` packaging/runtime.  Additional tooling such as `tools/resource_prepairer` and `tools/model_tester` now allow full pipeline validation from optimized datasets to raw‑PE inference in one flow.

````