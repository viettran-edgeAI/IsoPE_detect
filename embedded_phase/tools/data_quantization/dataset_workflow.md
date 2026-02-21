# Dataset Quantization Workflow (Isolation)

## Overview

This toolchain is now a single isolation-focused workflow:

1. Load static PE-feature CSV from `datasets/`
2. Quantize features with `processing_data.cpp`
3. Export artifacts to `quantized_datasets/`

No clustering/PCA plotting stage is included.

## Inputs

- CSV file path from `quantization_config.json` (`input_path`)
- Recommended `problem_type`: `isolation`

## Outputs (always in `quantized_datasets/`)

- `<name>_nml.csv`
- `<name>_nml.bin`
- `<name>_qtz.bin`
- `<name>_dp.txt`

## Run

```bash
./quantize_dataset.sh
```

or:

```bash
./quantize_dataset.sh -c quantization_config.json
```

## Quick validation

```bash
ls -la quantized_datasets/
```

You should see the four output files for your dataset basename.
