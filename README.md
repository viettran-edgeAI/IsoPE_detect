# IsoPE_detect

# Project overview

This repository contains an experimental system for **anomaly‑based
zero‑day malware detection on Windows endpoint devices**.  The detector
is purely static: it looks only at Portable Executable (PE) files
present on the host and tries to decide whether a given sample is
suspicious without executing it.

## 1. Problem & data

- **Goal**: build an agent that runs on constrained endpoints,
  identifies unseen (zero‑day) malware by comparing PE features against
  a model of “normal” binaries.
- **Scope**: static analysis of PE headers/sections/imports/resources.
- **Languages**: development in Python for rapid experimentation;
  production/extraction/engine in C++.
- **Dataset**: laid out under `datasets/`:
  - `BENIGN_TRAIN`/`VALIDATION_DATASET`/`TEST` – clean binaries.
  - `MALWARE_VALIDATION_DATASET`/`TEST` – malicious samples for
    evaluation only.  
  (malware is never used during training).

## 2. Two‑phase architecture

The work is split into a **development phase** and an **embedding
phase**.  each has its own directory tree:

- `development_phase/` – large‑scale python pipeline to find the best
  feature set and model configuration.
- `embedded_phase/` – lightweight C++ modules that will eventually be
  compiled into the endpoint agent.

The development phase produces a small set of artifacts that drive the
embedding phase.

### 2.1 Development phase

1. **Configuration**  
   `development_phase/src/model_config.json` defines the grid search
   range (feature families, hyper‑parameter ranges, validation targets,
   etc.).  The file now also records the names of the two output files
   that will be generated.

2. **Feature & model optimisation**  
   `model_optimization.py` (in `development_phase/src/`) reads the
   config, loads the benign training data, iterates over feature
   subsets and model parameters, standardises each candidate set, fits an
   IsolationForest, selects a decision threshold on a
   validation set, and scores on test data.  The pipeline
   enforces reproducibility (fixed random seeds, deterministic ordering).

3. **Results**  
   After a run you find in `development_phase/results/` only two files:

   * `optimized_feature_list.json` – ordered list of the 35‑40 features
     that survived optimisation.  This list is the “schema” for the
     extractor.
   * `model_engine_config.json` – all hyper‑parameters for the chosen
     isolation forest, the scaler statistics (means and scales for each
     feature), thresholding information, and a summary of validation /
     test metrics.  The previous multiple‑file layout was collapsed into
     this single JSON.

   These files are consumed by both the C++ feature extractor build
   script and, later, the model engine training utility.

4. **Scaler & standardisation**  
   During training every column is standardised:
   `z = (x – μ) / σ`.  Standardisation normalises heterogeneous feature
   ranges, stabilises the grid search and threshold calibration, and
   makes the validation/test metrics interpretable.  The scaler
   parameters are recorded in the config so that the embedded engine can
   either apply the same transform at runtime or fold the transform into
   the model.

### 2.2 Embedding phase

This is where C++ code lives:

- `embedded_phase/src/feature_extractor/` – a hand‑rolled, LIEF‑like
  extractor that produces the selected features.  A build‑time script
  (`generate_compiled_features.py`) reads
  `optimized_feature_list.json` and emits a header with the hard‑coded
  feature names and indices.

- `embedded_phase/src/model_engine/` – not yet fully implemented, but
  will contain code to train a fully-quantized C++ Isolation Forest on the
  optimized training split and to perform inference.  Before training, the
  dataset is fed through a **full quantization pipeline** adapted from the
  [viettran-edgeAI/MCU](https://github.com/viettran-edgeAI/MCU) library:
  each feature value is mapped to an integer bin index (default 2 bits = 4
  bins per feature), producing a bit-packed dataset and a binary quantizer
  artifact (`pe_quantizer.bin`).  The Isolation Forest is then trained on
  the integer-bin dataset, so all split thresholds are small integers rather
  than floats, shrinking the model by 10–150× and eliminating per-sample
  floating-point normalisation at runtime.  The engine respects the same
  feature order and will accept raw feature vectors from the extractor
  (quantization is applied as the first inference step).

- **Resource limits** – planner agent output was used to define
  compile‑time macros (`resource_limits.hpp`) that cap file size,
  memory, thread count, number of imports/sections/etc.  These limits
  are enforced in the extractor and are documented in
  `embedded_phase/src/feature_extractor/README.md`.

- **Scaler folding** – superseded by full quantization.  The quantizer's
  per-feature bin boundaries encode the same feature-range information as
  the StandardScaler, but in discrete form.  The original scaler statistics
  remain in `model_engine_config.json` for auditing and parity comparisons
  between the development-phase Python model (float features) and the
  embedded C++ model (integer bin features).

## 3. Steps taken so far

- Completed development‑phase pipeline; ran grid search, produced
  results files.
- Consolidated result artifacts into two files, added validation of
  required parameters (`val_fpr_delta`), renamed feature list output.
- Generated documentation for results, development phase, and embedding
  design.
- Built and smoke‑tested C++ feature extractor, integrated resource
  limits.
- Added scaler‑folding concept and updated config accordingly.

## 4. Next steps

1. **Model engine implementation**
   - Integrate dataset quantization tool (MCU library pipeline) to convert
     `benign_train_optimized.csv` → bit-packed dataset + `pe_quantizer.bin`.
   - Create C++ Isolation Forest that trains on the integer-bin dataset.
   - Implement runtime quantizer: raw float feature vector → bin-index
     integer vector before tree traversal.
   - Export a serialized, constant-time inference structure and
     production-ready decision rule.

2. **Parity & validation**
   - Develop a test harness that scores a fixed set of samples in both
     Python and C++ and reports per‑sample score differences and
     decision agreement.
   - Ensure edge‑case handling (`parse_ok=false`, missing features,
     overflow) is identical.

3. **Integration**
   - Wire feature extractor CLI and model engine CLI together.
   - Add unit tests and a small benchmark suite for resource‑limit
     behaviour.

4. **Deployment tuning**
   - Tune compile‑time limit macros for target devices.
   - Add a top‑level summary to the repository README.

5. **Future work**
   - Periodically re‑run development pipeline with fresh data.
   - Explore alternate anomaly detectors or incremental training.
   - Potentially implement on‑device threshold update without model
     rebuild.

---

This introduction should give you a clear view of where the project started,
what has been achieved, and the direction for completing the embedding
phase and turning the code into a deployable agent.  
Let me know when you want help scaffolding any of the next‑step components.
