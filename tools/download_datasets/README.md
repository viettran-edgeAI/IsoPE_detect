# Download Datasets Tool

This folder contains scripts used to fetch and prepare malware datasets for
use by the development pipeline.

## Purpose

- `download_malware.py` pulls samples from MalwareBazaar (or other sources) based
  on configurable profiles, extracts and deduplicates them, and writes them to
  target directories.
- `download_dataset.py` provides a generic framework for downloading arbitrary
  collections defined in a JSON config.

## Setup & Usage

Activate the repository virtual environment and install dependencies if
necessary:

```bash
source .venv/bin/activate
pip install requests tqdm
```

Use the provided config files (`download_dataset_config.json`,
`download_malware_config.json`) or specify an alternate config via
`--config`.

Example malware download:

```bash
cd tools/download_datasets
python download_malware.py --config download_malware_config.json --profile validation
```

Adjust profiles in the JSON to control URLs, target directories, and required
sample counts. See the scripts' `--help` output for full CLI options.