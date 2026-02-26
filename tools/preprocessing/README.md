# Preprocessing Utilities

Preprocessing scripts help normalize and prepare raw binary datasets for use in
model training and evaluation.

Currently this directory includes:

* `process_dataset.py` – deduplicates files by SHA256, filters by
  OS-specific extensions, and writes a normalized dataset listing. Configuration
  can be supplied via `process_dataset_config.json` or CLI overrides.

## Usage

Activate the virtual environment and install any required packages:

```bash
source .venv/bin/activate
pip install pandas tqdm
```

Run with the default config:

```bash
python process_dataset.py
```

Or specify inputs and outputs directly:

```bash
python process_dataset.py --input_path "../raw/*" --os Windows --output_path "../BENIGN_TRAIN_DATASET"
```

Configuration options include input glob, target OS, output directory, and file
extension overrides. See `--help` for further details.