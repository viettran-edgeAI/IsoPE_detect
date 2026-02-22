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

- Source list: `development_phase/results/optimized_feature_list.json` (or another JSON list)
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
OPTIMIZED_FEATURE_LIST_PATH=/absolute/path/to/optimized_feature_list.json \
embedded_phase/src/feature_extractor/build.sh
```

`build.sh` runs `generate_compiled_features.py` and then builds the `lief_feature_extractor` target with CMake.

## Resource limits (endpoint hardening)

The extractor enforces compile-time limits to bound CPU/memory usage on endpoint devices.

Default limits are defined in:

- `embedded_phase/src/feature_extractor/include/extractor/resource_limits.hpp`

Current baseline:

- `EDR_PE_MAX_INPUT_FILE_BYTES`: `128 MiB`
- `EDR_PE_MAX_WORKING_SET_BYTES`: `256 MiB`
- `EDR_PE_MIN_INPUT_FILE_BYTES`: `512 bytes`
- `EDR_PE_MAX_PATH_BYTES`: `4096`
- `EDR_PE_MAX_CLI_FILES_PER_RUN`: `2048`
- `EDR_PE_MAX_CLI_TOTAL_INPUT_BYTES`: `2 GiB`
- `EDR_PE_MAX_THREADS`: `1` (single-thread policy)
- `EDR_PE_MAX_SECTIONS`: `10`
- `EDR_PE_MAX_IMPORT_DLLS`: `512`
- `EDR_PE_MAX_IMPORT_FUNCS_TOTAL`: `8192`
- `EDR_PE_MAX_HASHABLE_NAME_BYTES`: `256`
- `EDR_PE_MAX_HASH_UPDATES_PER_FILE`: `10000`
- `EDR_PE_MAX_OVERLAY_ENTROPY_BYTES`: `8192`
- `EDR_PE_MAX_SIGNATURES`: `4`
- `EDR_PE_MAX_CERTIFICATES_TOTAL`: `64`
- `EDR_PE_MAX_DEBUG_ENTRIES`: `128`
- `EDR_PE_MAX_RICH_ENTRIES`: `256`
- `EDR_PE_MAX_DATA_DIRECTORIES`: `16`
- `EDR_PE_MAX_ERROR_TEXT_BYTES`: `256`

Behavior notes:

- Hard input validation failures (path/file/size limits) produce `parse_ok=false` with a bounded error code.
- Feature-complexity truncation (for example import/signature/hash caps) keeps `parse_ok=true` and sets `error="resource_limit"`.

### Overriding limits at build time

You can override any limit with `-D` compile definitions.

Example:

```bash
cd /home/viettran/Documents/visual_code/EDR_AGENT/embedded_phase/src/feature_extractor
cmake -S . -B build \
  -DFEATURE_EXTRACTOR_FETCH_LIEF=OFF \
  -DCMAKE_CXX_FLAGS="-DEDR_PE_MAX_INPUT_FILE_BYTES=67108864 -DEDR_PE_MAX_IMPORT_FUNCS_TOTAL=4096"
cmake --build build -j$(nproc)
```

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

