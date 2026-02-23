# Dataset Quantization Workflow

## Overview

Batch pipeline for the embedding phase:

1. Configure `quantization_config.json` with `model_name` and `input_dir`.
2. Build the single-file quantization module (`tools/data_quantization/`).
3. Run the orchestrator — it processes all five CSV splits sequentially.
4. Check artifacts in `quantized_datasets/`.

## Inputs

Five CSVs under `input_dir` (set in `quantization_config.json`):

```
<model_name>_ben_train.csv
<model_name>_ben_test.csv
<model_name>_ben_val.csv
<model_name>_mal_test.csv
<model_name>_mal_val.csv
```

Default `input_dir`: `../../../development_phase/data/optimized`

## Outputs (always in `quantized_datasets/`)

For every split four files are produced:

- `<name>_nml.csv`   quantized CSV
- `<name>_nml.bin`   bit-packed binary
- `<name>_qtz.bin`   quantizer metadata
- `<name>_dp.txt`    data profile

## Step-by-step run

```bash
# 1. Build single-file processor (once)
cd tools/data_quantization && make build && cd -

# 2. Run batch quantization
cd embedded_phase/tools/data_quantization
./quantize_dataset.sh
```

or via Makefile:

```bash
cd embedded_phase/tools/data_quantization && make process
```

## Quick validation

```bash
ls -la embedded_phase/tools/data_quantization/quantized_datasets/
```

Expect 20 files (4 artifacts × 5 splits).
