import os
import json
import hashlib
import shutil
import argparse
from glob import glob

# Extensions categorized by operating system
# Note: Some formats may not have file extensions (e.g., Linux executables)
OS_EXTENSIONS = {
    "Windows": {'.exe', '.dll', '.pyd', '.sys', '.drv', '.ocx', '.scr', '.cpl', '.ax', '.msi', '.com'},
    "Linux": {'', '.so', '.bin', '.elf', '.deb', '.rpm'},
    "Android": {'.apk', '.dex', '.so', '.jar', '.aar'},
    "iOS": {'.ipa', '.app', '.framework', '.dylib'}
}

def get_file_hash(filepath):
    """Calculates SHA256 hash of a file to use as its new filename."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        # Read in chunks to handle large files efficiently
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def process_files(input_path, output_path, target_os, extensions_override=None):
    # Create output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")
    
    source_dirs = glob(input_path)
    if not source_dirs:
        print(f"No source directories found matching pattern: {input_path}")
        return

    if extensions_override:
        valid_extensions = {ext.lower() for ext in extensions_override}
    else:
        valid_extensions = OS_EXTENSIONS.get(target_os, set())

    stats = {
        "processed": 0,
        "copied": 0,
        "duplicates": 0,
        "errors": 0,
        "skipped_ext": 0
    }

    print(f"Target OS: {target_os}")
    print(f"Filtering for extensions: {valid_extensions}")
    print(f"Found source directories/files: {source_dirs}")
    
    # helper set to track uniqueness across all folders
    seen_hashes = set()
    
    # 1. Pre-scan output dir to avoid re-doing work if script runs twice
    if os.path.exists(output_path):
        existing_files = os.listdir(output_path)
        for f in existing_files:
            name, _ = os.path.splitext(f)
            # Verify it looks like a hash (64 chars)
            if len(name) == 64: 
                seen_hashes.add(name)
    
    print(f"Starting with {len(seen_hashes)} files already in output directory.")

    for source_dir in source_dirs:
        if os.path.isfile(source_dir):
            # If the input_path matched a single file
            items_to_process = [(os.path.dirname(source_dir), None, [os.path.basename(source_dir)])]
        else:
            items_to_process = os.walk(source_dir)

        for root, _, files in items_to_process:
            for file in files:
                stats["processed"] += 1
                file_path = os.path.join(root, file)
                
                # --- Step 1: Filter by Extension ---
                _, ext = os.path.splitext(file)
                ext = ext.lower()
                
                if ext not in valid_extensions:
                    stats["skipped_ext"] += 1
                    continue
                
                try:
                    # --- Step 2: Calculate Hash (The "Feature Extraction" prep) ---
                    file_hash = get_file_hash(file_path)
                    
                    # --- Step 3: Deduplication ---
                    if file_hash in seen_hashes:
                        stats["duplicates"] += 1
                        continue
                    
                    # --- Step 4: Rename and Save ---
                    # New format: <SHA256_HASH>.<EXTENSION>
                    new_filename = f"{file_hash}{ext}"
                    dest_path = os.path.join(output_path, new_filename)
                    
                    shutil.copy2(file_path, dest_path)
                    seen_hashes.add(file_hash)
                    stats["copied"] += 1
                    
                    if stats["copied"] % 100 == 0:
                        print(f"  > Unique files copied so far: {stats['copied']}")
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    stats["errors"] += 1

    print("\n" + "="*40)
    print("PROCESSING COMPLETE")
    print("="*40)
    print(f"Total files scanned:       {stats['processed']}")
    print(f"New unique files copied:   {stats['copied']}")
    print(f"Duplicates skipped:        {stats['duplicates']}")
    print(f"Incompatible OS skipped:   {stats['skipped_ext']}")
    print(f"Final dataset size:        {len(seen_hashes)} files")
    print(f"Location:                  {os.path.abspath(output_path)}")

def _load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw datasets containing executable files, remove duplicates, and hash names.")
    parser.add_argument("--config", help="Path to JSON config file (defaults to 'process_dataset_config.json' or 'process_dataset.config.json' in the script directory)")
    parser.add_argument("-p", "--input_path", help="Path to the source directories (glob pattern supported)")
    parser.add_argument("-s", "--os", choices=["Windows", "Linux", "Android", "iOS"], help="Target operating system to filter files")
    parser.add_argument("-o", "--output_path", help="Path to the output directory")

    args = parser.parse_args()

    # Load config: prefer --config; otherwise try to load a default config
    # named 'process_dataset.config.json' located in the same directory as this script.
    if args.config:
        try:
            cfg = _load_config(args.config)
        except Exception as e:
            raise SystemExit(f"[!] Failed to load config '{args.config}': {e}")
    else:
        # Check for common default config filenames next to the script
        candidate_names = ["process_dataset.config.json", "process_dataset_config.json"]
        default_config_path = None
        for name in candidate_names:
            p = os.path.join(os.path.dirname(os.path.realpath(__file__)), name)
            if os.path.exists(p):
                default_config_path = p
                break

        if default_config_path:
            try:
                cfg = _load_config(default_config_path)
                print(f"[*] Loaded default config from {default_config_path}")
            except Exception as e:
                print(f"[!] Failed to load default config '{default_config_path}': {e}")
                cfg = {}
        else:
            cfg = {}

    input_path = args.input_path or cfg.get("input_glob")
    target_os = args.os or cfg.get("target_os")
    output_path = args.output_path or cfg.get("output_dir")
    extensions_override = cfg.get("extensions_override")

    if not input_path or not target_os or not output_path:
        raise SystemExit("Missing required inputs: input_path, target_os, output_path")

    process_files(input_path, output_path, target_os, extensions_override=extensions_override)

