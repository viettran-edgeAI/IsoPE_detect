# LIEF C++ Feature Extractor (Embedded Phase)

This directory contains a module-style PE feature extractor built with LIEF.

## Module layout

- `include/extractor/extractor.hpp`: public API (`FeatureVector`, `IExtractor`, `PEExtractor`)
- `src/pe_extractor.cpp`: library implementation
- `app/pe_feature_extractor_cli.cpp`: standalone CLI wrapper
- `compiled_feature_config.hpp`: generated compile-time feature selection

Each module has its own `CMakeLists.txt`:

- `feature_extractor/CMakeLists.txt`
- `feature_extractor/src/CMakeLists.txt`
- `feature_extractor/app/CMakeLists.txt`

## Public API

```cpp
#include "extractor/extractor.hpp"

extractor::PEExtractor ext;
extractor::FeatureVector features = ext.extract("sample.exe");
```

The model integration path only needs this API surface.

## Compile-time feature locking

The selected feature list is generated into C++ at build time and fixed after compilation.

- Source list: `development_phase/results/feature_names.json` (or another JSON list)
- Generated header: `embedded_phase/src/feature_extractor/compiled_feature_config.hpp`
- Runtime: no `--feature-names` option, no runtime schema parsing

## Build

### Prerequisite: build LIEF once

```bash
cd embedded_phase/third_party/LIEF
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLIEF_PYTHON_API=OFF \
  -DLIEF_EXAMPLES=OFF \
  -DLIEF_TESTS=OFF \
  -DLIEF_DOC=OFF \
  -DLIEF_INSTALL=OFF
cmake --build build -j$(nproc)
```

### Build extractor with default features

```bash
cd /home/viettran/Documents/visual_code/EDR_AGENT
embedded_phase/src/feature_extractor/build.sh
```

### Build extractor with a model-specific feature list

```bash
cd /home/viettran/Documents/visual_code/EDR_AGENT
FEATURE_NAMES_PATH=/absolute/path/to/feature_names.json \
embedded_phase/src/feature_extractor/build.sh
```

`build.sh` runs `generate_compiled_features.py` and then builds the `lief_feature_extractor` target with CMake.

## CLI usage

```bash
embedded_phase/src/feature_extractor/lief_feature_extractor \
  --format jsonl \
  datasets/BENIGN_TEST_DATASET/<file>.dll
```

```bash
embedded_phase/src/feature_extractor/lief_feature_extractor \
  --format csv \
  --output embedded_phase/src/feature_extractor/sample_output.csv \
  datasets/BENIGN_TEST_DATASET/<file1>.dll \
  datasets/MALWARE_TEST_DATASET/<file2>.exe
```

CLI options:

- `--format csv|jsonl`
- `--output <path>`

## Python ↔ C++ parity validation

Validation harness:

- `embedded_phase/src/feature_extractor/compare_cpp_python.py`

Run:

```bash
/home/viettran/Documents/visual_code/EDR_AGENT/.venv/bin/python \
  embedded_phase/src/feature_extractor/compare_cpp_python.py \
  --samples-per-class 3
```

Important: pass the same feature list via `--feature-names` that was used at compile time for `build.sh`, otherwise comparison will be invalid.

Outputs:

- `embedded_phase/src/feature_extractor/validation/parity_summary.json`
- `embedded_phase/src/feature_extractor/validation/cpp_processing_time.csv`
- `embedded_phase/src/feature_extractor/validation/feature_parity_details.csv`
