# Dataset Quantization Tool — Batch Orchestrator

Quantize all five optimized CSV splits for a given model in one command.
Focused on `problem_type = isolation` (anomaly-detection workflow).

## Architecture

Two-tier design:

```
embedded_phase/tools/data_quantization/processing_data  ← orchestrator (this module)
        │  invokes 5 ×
        ▼
tools/data_quantization/processing_data                 ← single-file quantization module
```

The orchestrator reads `model_name` and `input_dir` from `quantization_config.json`,
fits a quantizer once on `<model_name>_ben_train.csv`, then transforms the other
four splits with the same quantizer.

## Inputs

Five optimized CSV files under `input_dir` (default `development_phase/data/optimized/`):

| File | Role |
|---|---|
| `<model_name>_ben_train.csv` | Benign training split |
| `<model_name>_ben_test.csv`  | Benign test split |
| `<model_name>_ben_val.csv`   | Benign validation split |
| `<model_name>_mal_test.csv`  | Malware test split |
| `<model_name>_mal_val.csv`   | Malware validation split |

## Output files (in `quantized_datasets/`)

Per split:

| File | Contents |
|---|---|
| `<name>_nml.csv` | Quantized CSV (human-readable) |
| `<name>_nml.bin` | Bit-packed binary dataset |
| `<name>_dp.txt`  | Data profile (statistics, bit-width) |

Model-level (single shared artifacts):

| File | Contents |
|---|---|
| `<model_name>_qtz.bin` | Shared quantizer metadata (fit on benign-train only) |
| `<model_name>_nml.bin` | Alias to benign-train quantized dataset |
| `<model_name>_nml.csv` | Alias to benign-train quantized CSV |
| `<model_name>_dp.txt` | Alias to benign-train data profile |

## Configuration (`quantization_config.json`)

| Field | Default | Description |
|---|---|---|
| `model_name` | *(required)* | e.g. `"iforest"` |
| `input_dir`  | *(required)* | Path to the five CSV splits |
| `quantization_bits` | `2` | Bits per feature (1–8) |
| `header` | `auto` | `auto \| yes \| no` |
| `problem_type` | `isolation` | `isolation \| classification \| regression` |
| `remove_outliers` | `false` | Z-score outlier clipping |

## Prerequisites

Build the single-file quantization module first:

```bash
cd ../../../tools/data_quantization && make build
```

## Build & run

```bash
# Via Makefile
make process

# Via shell script
./quantize_dataset.sh

# Explicit config
./quantize_dataset.sh -c quantization_config.json
```

## Makefile targets

```bash
make build       # Compile the orchestrator
make process     # Build + run batch quantization
make status      # Show binary + output status
make clean       # Remove orchestrator binary
make clean-all   # Remove binary and quantized_datasets/
```
