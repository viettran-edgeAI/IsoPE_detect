# tools/resource_prepairer

Config-driven dataset preparation tool for the embedded Isolation Forest runtime.

## Purpose

This tool prepares the quantized resources required by `if_model.h` for:

- Training artifacts from benign train split:
  - `<model_name>_qtz.bin`
  - `<model_name>_dp.txt`
  - `<model_name>_nml.bin`
- Validation datasets for threshold calibration:
  - `<model_name>_ben_val_nml.bin`
  - `<model_name>_mal_val_nml.bin`

It invokes `tools/data_quantization/processing_data` internally.

## Config file

Default config path:

- `tools/resource_prepairer/resource_prepairer_config.json`

The config uses the same `{"value": ...}` style as `quantization_config.json`.

## Run

```bash
python3 tools/resource_prepairer/prepare_datasets.py \
  -c tools/resource_prepairer/resource_prepairer_config.json
```

Optional CLI overrides:

- `--benign-train`
- `--benign-val`
- `--malware-val`
- `--output-dir`
- `--model-name`
- `--quantization-bits`
- `--quantization-tool`
- `--header`
- `--problem-type`
- `--remove-outliers`
- `--dev-optimized-config-dir`  _(new)_

## Behavior

1. Fits quantizer on benign train CSV.
2. Produces `<model_name>_qtz.bin` and `<model_name>_dp.txt`.
3. Reuses that quantizer to transform benign/malware validation CSVs.
4. Removes split-specific `_dp.txt`/`_qtz.bin` for validation outputs.
5. Writes all outputs to `output_dir`.
