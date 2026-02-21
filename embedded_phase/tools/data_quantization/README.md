# Dataset Quantization Tool (Isolation-Focused)

Quantize PE static-feature CSV datasets for endpoint anomaly detection workflows.
This version is focused on `problem_type = isolation` and does not include clustering/PCA plotting.

## What it does

- Reads CSV datasets from `datasets/`
- Quantizes features with configurable bit width (`1-8`)
- Exports quantized artifacts for embedding/retraining
- Writes all outputs to `quantized_datasets/`

## Output files

For input `<name>.csv`, generated files are:

- `quantized_datasets/<name>_nml.csv` (quantized CSV)
- `quantized_datasets/<name>_nml.bin` (packed binary dataset)
- `quantized_datasets/<name>_qtz.bin` (quantizer metadata)
- `quantized_datasets/<name>_dp.txt` (dataset parameters)

## Configuration

Edit `quantization_config.json`.

Supported fields:

- `input_path` (required): CSV path
- `model_name`: output basename override (`auto` uses input filename)
- `header`: `auto | yes | no`
- `problem_type`: `classification | regression | isolation`
- `quantization_bits`: `1..8`
- `remove_outliers`: `true | false`

## Recommended isolation config

```json
{
  "input_path": { "value": "datasets/benign_test_optimized.csv" },
  "problem_type": { "value": "isolation" },
  "quantization_bits": { "value": 2 },
  "header": { "value": "auto" },
  "model_name": { "value": "auto" },
  "remove_outliers": { "value": false }
}
```

## Run

```bash
./quantize_dataset.sh
```

or with explicit config:

```bash
./quantize_dataset.sh -c quantization_config.json
```

## Build and test with Makefile

```bash
make build
make process
make status
```

## Notes

- Clustering/visualization plotting scripts were removed because they target classification analysis and are not relevant to isolation workflows.
- Output location is fixed to `quantized_datasets/` (next to the config file).
