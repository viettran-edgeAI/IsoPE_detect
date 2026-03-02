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

For evaluation CLI (test scoring), provide:
- `<model>_ben_test_nml.bin`
- `<model>_mal_test_nml.bin`

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
`<model>_ben_test_nml.bin` and `<model>_mal_test_nml.bin` in the resource directory.

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

## 4) Windows build and run (VS Code + PowerShell)

### 4.1 Prerequisites

- Install **CMake Tools** extension in VS Code.
- Install **MSYS2 MinGW64** (or equivalent GNU toolchain) with:
  - `gcc` / `x86_64-w64-mingw32-gcc`
  - `g++` / `x86_64-w64-mingw32-g++`
  - `ninja`
- Install **CMake** on Windows.

This workspace already has a preset for MinGW64:
- `msys2-mingw64-release` in `CMakePresets.json`.

The preset injects required MSYS2 runtime/toolchain paths automatically:
- `C:/msys64/usr/bin`
- `C:/msys64/mingw64/bin`

### 4.2 Configure and build (top-level CMake)

From repository root in PowerShell:

```powershell
cmake --preset msys2-mingw64-release
cmake --build --preset msys2-mingw64-release-build -j
```

Run tests from the same preset build directory:

```powershell
ctest --preset msys2-mingw64-release-test
```

If you switch presets/toolchains and hit stale-cache configure errors, clear the preset build folder and reconfigure:

```powershell
Remove-Item -Recurse -Force out/build/msys2-mingw64-release
cmake --preset msys2-mingw64-release
```

Before running benchmark/model tester, regenerate resource artifacts so quantized NML files match the selected model features:

```powershell
.\.venv\Scripts\python.exe tools\resource_prepairer\prepare_datasets.py `
  -c tools\resource_prepairer\resource_prepairer_config.json
```

Expected quantized split outputs in `embedded_phase/core/models/isolation_forest/resources`:

- `<model>_ben_val_nml.bin`
- `<model>_mal_val_nml.bin`
- `<model>_ben_test_nml.bin`
- `<model>_mal_test_nml.bin`

Expected model engine app outputs:

- `out/build/msys2-mingw64-release/embedded_phase/src/model_engine/app/pe_model_engine_benchmark_cli.exe`
- `out/build/msys2-mingw64-release/embedded_phase/src/model_engine/app/pe_model_engine_cli.exe`

### 4.3 Run `pe_model_engine_benchmark_cli.exe` (test sets only)

`pe_model_engine_benchmark_cli.exe` evaluates quantized **test** NML splits only:

```powershell
.\out\build\msys2-mingw64-release\embedded_phase\src\model_engine\app\pe_model_engine_benchmark_cli.exe `
  --resource-dir embedded_phase/core/models/isolation_forest/resources `
  --model-name iforest `
  --json-output embedded_phase/src/model_engine/results/if_model_engine_eval_windows.json
```

### 4.4 Build and run `tools/model_tester`

`tools/model_tester` is a standalone source and not part of top-level CMake targets.

Build:

```powershell
Push-Location tools/model_tester
$env:Path = "C:\msys64\usr\bin;C:\msys64\mingw64\bin;$env:Path"
& "C:\msys64\mingw64\bin\x86_64-w64-mingw32-g++.exe" `
  -std=c++17 if_quantized_cpp_raw_pe_eval.cpp `
  -I../../embedded_phase/src/.. `
  -I../../embedded_phase/third_party/LIEF/include `
  -I../../embedded_phase/third_party/LIEF/src `
  -I../../embedded_phase/third_party/LIEF/build_windows_x64/include `
  -I../../embedded_phase/third_party/LIEF/build_windows_x64 `
  -L../../embedded_phase/third_party/LIEF/build_windows_x64 `
  -lLIEF -lws2_32 -lbcrypt -O2 -o if_quantized_cpp_raw_pe_eval.exe
Pop-Location
```

Run:

```powershell
$env:Path = "C:\msys64\usr\bin;C:\msys64\mingw64\bin;$env:Path"
.\tools\model_tester\if_quantized_cpp_raw_pe_eval.exe --repo-root . --model-name iforest
```

This run consumes PE files from `datasets/BENIGN_TEST_DATASET` and `datasets/MALWARE_TEST_DATASET`, while loading model resources from `embedded_phase/core/models/isolation_forest/resources`.

Expected report output:

- `development_phase/reports/if_quantized_cpp_raw_pe_eval.txt`

### 4.5 VS Code CMake Tools flow (optional)

In VS Code:
1. `CMake: Select Configure Preset` → `MSYS2 MinGW64 + Ninja (GNU Release)`
2. `CMake: Configure`
3. `CMake: Build`
4. `CMake: Run Tests` (or run `ctest --preset msys2-mingw64-release-test`)
5. Run the executables above in the integrated PowerShell terminal.

## 5) Notes for endpoint integration

- Use the C++ API via `include/model_engine.hpp` when integrating into C++ agents.
- Use `include/model_engine_c.h` when integrating from C or FFI layers.
- The runtime itself does not require LIEF/external extractor binaries to run quantized/raw vector inference.
- PE-path/content inference requires providing extractor callbacks when default stub behavior is not sufficient.