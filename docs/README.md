# EDR_AGENT Documentation Overview

This project has two-phase workflow:
- **Development phase** (Python): build/optimize the model and export deployment artifacts.
- **Embedded phase** (C/C++): load artifacts and run endpoint-ready feature extraction + inference.
## Phase 1: Development (`development_phase`)

Purpose: build and validate the malware-detection model pipeline.

Main functionality:
- Prepare/normalize benign and malware datasets.
- Engineer and select PE features.
- Optimize Isolation Forest parameters and thresholding.
- Export deployment-ready artifacts (config, schema, scaler/quantization parameters).

Key outputs:
- `development_phase/results/` (artifact JSONs used by embedded runtime)
- `development_phase/data/optimized/` (optimized train/validation CSVs)
- `development_phase/reports/` (evaluation summaries)

## Phase 2: Embedded (`embedded_phase`)

Purpose: provide deployable C/C++ inference runtime for endpoint integration.

Main functionality:
- Load artifacts produced in `development_phase`.
- Extract PE features and run quantized/scaled inference.
- Expose runtime APIs and model-engine integration layers.

Major components:
- `embedded_phase/core/models/isolation_forest/` — core model library (`IsoForest`, tree container, preprocessing layers)
- `embedded_phase/src/model_engine/` — deployment-oriented runtime packaging and CLI utilities
- `embedded_phase/src/feature_extractor/` — feature extraction implementation

## Project Structure (high level)

- `development_phase/` — model building, optimization, artifact export
- `embedded_phase/` — embedded inference runtime and integration code
- `datasets/` — benign/malware train/validation/test data
- `tools/` — preprocessing, resource preparation, testing, and reporting tools
- `docs/` — architecture, workflows, and supporting documentation

## End-to-End Workflow

1. Prepare datasets.
2. Run development pipeline to produce optimized artifacts.
3. Run `tools/resource_prepairer/prepare_datasets.py` to package embedded resources.
4. Build embedded components and model engine.
5. Validate with raw-PE evaluation and reporting tools.

````