# Model Engine Build & Run Guide (Windows + Ubuntu)

This guide covers how to build and run the embedded deployment runtime in `embedded_phase/src/model_engine`.

## 1) What this runtime does

`model_engine` is deployment-only:
- Loads the Isolation Forest resource set from `If_base` (resources 1..7).
- Performs inference (`infer_quantized`, `infer_raw`, optional PE callback path).
- Exposes both C++ and C library interfaces:
  - `pe_model_engine` (C++)
  - `pe_model_engine_c` (C ABI)
- Includes demo/evaluation CLIs:
  - `pe_model_engine_cli`
  - `pe_model_engine_benchmark_cli`

## 2) Required resources

Place these files under one directory (recommended canonical path: `embedded_phase/core/models/isolation_forest/resources`):

1. `<model>_dp.txt`
2. `<model>_qtz.bin`
3. `<model>_optimized_config.json`
4. `<model>_optimized_features.json`
5. `<model>_scaler_params.json`
6. `<model>_feature_schema.json`
7. `<model>_iforest.bin`

For evaluation CLI (validation scoring), also provide:
- `<model>_ben_val_nml.bin`
- `<model>_mal_val_nml.bin`

## 3) Ubuntu build and run

### 3.1 Install dependencies

```bash
sudo apt update
sudo apt install -y build-essential cmake
```

### 3.2 Clone and enter repo

```bash
git clone <your-repo-url> EDR_AGENT
cd EDR_AGENT
```

### 3.3 Configure and build

```bash
cmake -S . -B build_cmake_tools
cmake --build build_cmake_tools -j
```

If new app targets are added/renamed in CMake (for example, endpoint integration samples), rerun configure first:

```bash
cmake -S . -B build_cmake_tools
cmake --build build_cmake_tools -j
```

### 3.4 Run tests

```bash
ctest --test-dir build_cmake_tools --output-on-failure
```

### 3.5 Run one-sample inference demo

```bash
./build_cmake_tools/embedded_phase/src/model_engine/app/pe_model_engine_cli \
  --resource-dir embedded_phase/core/models/isolation_forest/resources \
  --model-name iforest
```

### 3.6 Run validation / test evaluation

The benchmark app reads quantized NML files. By default it expects **test** datasets named
`<model>_ben_test_nml.bin` and `<model>_mal_test_nml.bin` in the resource directory. Legacy
`--benign-val`/`--malware-val` options still work for validation splits.

```bash
./build_cmake_tools/embedded_phase/src/model_engine/app/pe_model_engine_benchmark_cli \
  --resource-dir embedded_phase/core/models/isolation_forest/resources \
  --model-name iforest \
  --json-output embedded_phase/src/model_engine/results/if_model_engine_eval_ubuntu.json
```

To override default paths you may supply explicit files:

```bash
./build_cmake_tools/embedded_phase/src/model_engine/app/pe_model_engine_benchmark_cli \
  --resource-dir embedded_phase/core/models/isolation_forest/resources \
  --model-name iforest \
  --benign-test path/to/ben_test_nml.bin \
  --malware-test path/to/mal_test_nml.bin \
  --json-output results.json
```
### 3.7 Run minimal C API sample

```bash
./build_cmake_tools/embedded_phase/src/model_engine/app/endpoint_agent_capi_sample \
  --resource-dir embedded_phase/core/models/isolation_forest/resources \
  --model-name iforest
```

## 4) Windows build and run (PowerShell)

### 4.1 Install dependencies

- Visual Studio 2022 Build Tools (Desktop development with C++)
- CMake (3.18+)
- Git

### 4.2 Clone and enter repo

```powershell
git clone <your-repo-url> EDR_AGENT
cd EDR_AGENT
```

### 4.3 Configure and build (Visual Studio generator)

```powershell
cmake -S . -B build_cmake_tools -G "Visual Studio 17 2022" -A x64
cmake --build build_cmake_tools --config Release
```

### 4.4 Run tests

```powershell
ctest --test-dir build_cmake_tools -C Release --output-on-failure
```

### 4.5 Run one-sample inference demo

```powershell
.\build_cmake_tools\embedded_phase\src\model_engine\app\Release\pe_model_engine_cli.exe `
  --resource-dir embedded_phase/core/models/isolation_forest/resources `
  --model-name iforest
```

### 4.6 Run validation-split evaluation

```powershell
.\build_cmake_tools\embedded_phase\src\model_engine\app\Release\pe_model_engine_benchmark_cli.exe `
  --resource-dir embedded_phase/core/models/isolation_forest/resources `
  --model-name iforest `
  --json-output embedded_phase/src/model_engine/results/if_model_engine_eval_windows.json
```

### 4.7 Run minimal C API sample

```powershell
.\build_cmake_tools\embedded_phase\src\model_engine\app\Release\endpoint_agent_capi_sample.exe `
  --resource-dir embedded_phase/core/models/isolation_forest/resources `
  --model-name iforest
```

## 5) Notes for endpoint integration

- Use the C++ API via `include/model_engine.hpp` when integrating into C++ agents.
- Use `include/model_engine_c.h` when integrating from C or FFI layers.
- The runtime itself does not require LIEF/external extractor binaries to run quantized/raw vector inference.
- PE-path/content inference requires providing extractor callbacks when default stub behavior is not sufficient.