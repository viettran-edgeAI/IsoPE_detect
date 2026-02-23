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

Primary output artifacts are generated in `development_phase/results/` and are intended to be consumed by the embedded runtime.

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

1. Prepare datasets under `datasets/`.
2. Run the development pipeline in `development_phase/src/` to generate model artifacts.
3. Use those artifacts in `embedded_phase/` to build and run the C++ runtime.

## Current Direction

The current focus is to keep a strict development-to-embedded contract: artifact generation in `development_phase`, artifact consumption in `embedded_phase`, and progressive refactoring of the embedded stack into a clean core model library plus production-ready `model_engine` packaging/runtime.
````