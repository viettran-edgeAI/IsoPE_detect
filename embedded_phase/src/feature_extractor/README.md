# LIEF C++ Feature Extractor (Embedded Phase)

This directory contains a **standalone C++ PE feature extractor** built with LIEF, aligned with the optimized feature set from the completed development phase.

It is designed for the embedding pipeline and currently covers the exact optimized feature list from:

- `development_phase/results/feature_names.json`
- `development_phase/results/feature_group_mapping.json`

---

## 1) What this extractor does

The binary `lief_feature_extractor`:

1. Parses PE files with LIEF C++ (`LIEF::PE::Parser::parse`)
2. Computes the selected optimized features (40 features)
3. Outputs per-file features as:
   - `csv` (default), or
   - `jsonl`
4. Records **processing time per file** (`processing_time_ms`)

Core implementation file:

- `embedded_phase/src/feature_extractor/lief_feature_extractor.cpp`

---

## 2) Implemented feature parity (Python ↔ C++)

The implementation mirrors development-phase logic in `development_phase/src/feature_extraction.py` for the selected features:

- COFF / optional header fields and flags
- Section slot and aggregate entropy features
- Section name hash features
- Import DLL and API hash features
- Resource/version presence
- Debug/repro flags
- Rich header max build ID
- Signature-derived fields (`num_certificates`, `sig_verified`)
- Overlay size/entropy
- Data directory RVAs/sizes
- `checksum_matches`

### Hashing trick parity

For hashed string features, the C++ version reproduces the Python algorithm:

- `idx = little_endian_u64(md5(string)[:8]) % n_features`
- `sign = +1 if little_endian_u64(sha1(string)[:8]) % 2 == 0 else -1`
- `vector[idx] += sign`

This is implemented via LIEF's internal `hashstream` (`MD5`, `SHA1`) to avoid extra external build dependencies.

---

## 3) Build

## Prerequisite

LIEF source is cloned to:

- `embedded_phase/third_party/LIEF`

and built once as static library:

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

## Build extractor (no project CMake)

```bash
cd /home/viettran/Documents/visual_code/EDR_AGENT
embedded_phase/src/feature_extractor/build.sh
```

Output binary:

- `embedded_phase/src/feature_extractor/lief_feature_extractor`

---

## 4) Usage

## Basic

```bash
embedded_phase/src/feature_extractor/lief_feature_extractor \
  --feature-names development_phase/results/feature_names.json \
  datasets/BENIGN_TEST_DATASET/<file>.dll
```

## JSONL output

```bash
embedded_phase/src/feature_extractor/lief_feature_extractor \
  --feature-names development_phase/results/feature_names.json \
  --format jsonl \
  datasets/MALWARE_TEST_DATASET/<file>.exe
```

## Write output file

```bash
embedded_phase/src/feature_extractor/lief_feature_extractor \
  --feature-names development_phase/results/feature_names.json \
  --format csv \
  --output embedded_phase/src/feature_extractor/sample_output.csv \
  datasets/BENIGN_TEST_DATASET/<file1>.dll \
  datasets/MALWARE_TEST_DATASET/<file2>.exe
```

CLI options:

- `--feature-names <path>`: feature order source (defaults to `development_phase/results/feature_names.json`)
- `--format csv|jsonl`: output format
- `--output <path>`: write to file (otherwise stdout)

---

## 5) Python-vs-C++ parity validation

Validation harness:

- `embedded_phase/src/feature_extractor/compare_cpp_python.py`

It does the following:

1. Imports Python extractor function `extract_features` from development phase
2. Runs C++ extractor on the same files
3. Compares all optimized features in order
4. Uses absolute tolerance (`1e-6` default)
5. Writes:
   - per-file timing report
   - per-feature parity details
   - summary JSON

Run command:

```bash
/home/viettran/Documents/visual_code/EDR_AGENT/.venv/bin/python \
  embedded_phase/src/feature_extractor/compare_cpp_python.py \
  --samples-per-class 3
```

Generated outputs:

- `embedded_phase/src/feature_extractor/validation/parity_summary.json`
- `embedded_phase/src/feature_extractor/validation/cpp_processing_time.csv`
- `embedded_phase/src/feature_extractor/validation/feature_parity_details.csv`

---

## 6) Current verification result

From `embedded_phase/src/feature_extractor/validation/parity_summary.json`:

- `samples_tested`: 6
- `features_per_sample`: 40
- `parse_failures_python`: 0
- `files_with_mismatch_or_parse_error`: 0
- `total_feature_mismatches`: 0
- `tolerance`: `1e-6`

This confirms parity for sampled benign/malware PE files.

---

## 7) Processing-time measurements (current run)

From `embedded_phase/src/feature_extractor/validation/cpp_processing_time.csv`:

| label   | file (short) | processing_time_ms |
|---------|---------------|-------------------:|
| benign  | `000d81f6...ec00.dll` | 20.842012 |
| benign  | `000de8f5...d9c2.dll` | 2.349149 |
| benign  | `000e0a55...8243.dll` | 2.269791 |
| malware | `0005626a...f6a0.exe` | 0.840911 |
| malware | `0009f3b6...52dc.exe` | 33.609066 |
| malware | `0019e0ae...2460.exe` | 34.049850 |

Note: processing time depends on file structure/size and host load.

---

## 8) Notes for next embedding step

- This extractor currently focuses on feature parity and standalone execution.
- The next step is integrating this feature logic into your lightweight embedded pipeline API (buffer-based or stream-based interface), then wiring model scoring in C++ with:
  - `development_phase/results/scaler_params.json`
  - `development_phase/results/trees.json`
  - `development_phase/results/threshold.json`
