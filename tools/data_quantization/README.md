# tools/data_quantization — Single-File Quantization Module

Standalone C++ tool that quantizes **one CSV file** per invocation.
It is consumed both directly (for ad-hoc use) and as a sub-process by the
batch orchestrator located at
`embedded_phase/tools/data_quantization/processing_data`.

## Relationship to the embedded-phase orchestrator

```
embedded_phase/tools/data_quantization/processing_data  (orchestrator)
        │
        │  invokes 5 times (one per CSV split)
        ▼
tools/data_quantization/processing_data                 (this module)
```

The orchestrator automatically discovers the five CSV splits for a given
model name and calls this binary for each one.

## Input

A single CSV file whose path is supplied either via config or via the
`-ip` CLI flag.

## Output (always written to `quantized_datasets/` next to the binary)

| File | Contents |
|---|---|
| `<name>_nml.csv` | Quantized CSV (human-readable) |
| `<name>_nml.bin` | Bit-packed binary dataset |
| `<name>_qtz.bin` | Quantizer metadata (bin boundaries) |
| `<name>_dp.txt`  | Data profile (statistics, bit-width) |

## Configuration (`quantization_config.json`)

| Field | Default | Description |
|---|---|---|
| `input_path` | *(required)* | Path to the input CSV file |
| `model_name` | `""` (auto) | Output basename override |
| `quantization_bits` | `2` | Bits per feature value (1–8) |
| `header` | `auto` | `auto \| yes \| no` |
| `problem_type` | `isolation` | `isolation \| classification \| regression` |
| `remove_outliers` | `false` | Z-score outlier clipping |

Optional CLI-only field:

| Flag | Description |
|---|---|
| `-qp`, `--quantizer_path` | Reuse an existing quantizer file and run transform-only quantization (no re-fit, no new quantizer export). |

## Build

```bash
make build
# or manually:
g++ -std=c++17 -I../../embedded_phase/src -o processing_data processing_data.cpp
```

## Run

```bash
# Using config file:
./processing_data -c quantization_config.json

# Inline override:
./processing_data -ip ../../development_phase/data/optimized/iforest_ben_train.csv -qb 2 -pt isolation

# Transform-only mode using an existing quantizer:
./processing_data -ip ../../development_phase/data/optimized/iforest_ben_val.csv \
        -mn iforest_ben_val \
        -qp ../../embedded_phase/tools/data_quantization/quantized_datasets/iforest_qtz.bin
```
