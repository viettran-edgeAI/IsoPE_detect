# Tools

This folder contains two helper tools for preparing datasets.

## 1) process_dataset.py
Deduplicates files by SHA256, filters by OS-specific extensions, and writes a normalized dataset.

Config file:
- `process_dataset_config.json` or `process_dataset.config.json` (automatically loaded from the script directory if `--config` is not provided)

Run with config:
```
python process_dataset.py --config process_dataset.config.json
```

Or run without `--config` to use the default config file found next to the script:
```
python process_dataset.py
```

Override via CLI:
```
python process_dataset.py --input_path "../raw_datasets/*" --os Windows --output_path "../BENIGN_TRAIN_DATASET"
```

Config fields:
- input_glob: glob pattern for input directories or files.
- target_os: Windows | Linux | Android | iOS.
- output_dir: destination directory for hashed samples.
- extensions_override: optional list of extensions to use instead of the OS defaults.

## 2) download_malware.py
Downloads malware zips from MalwareBazaar, extracts samples, and builds a deduped dataset.

Config file:
- `download_malware_config.json` (automatically loaded from the script directory if `--config` is not provided)

Run with config:
```
python download_malware.py --config download_malware_config.json
```

Or run without `--config` to use the default config file found next to the script:
```
python download_malware.py
```

Select a profile:
```
python download_malware.py --config download_malware_config.json --profile validation
```

Config fields:
- default_profile: profile name to use when --profile is not provided.
- zip_password: default ZIP password (MalwareBazaar uses infected).
- keep_zips: keep downloaded zips if true.
- keep_temp: keep the temp extraction folder if true.
- profiles: per-profile settings:
  - zip_urls: list of daily zip links.
  - target_dir: output directory.
  - required_samples: stop once this count is reached.
  - temp_dir: optional temp extraction path.
