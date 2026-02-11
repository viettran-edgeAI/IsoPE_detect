#!/usr/bin/env python3
import argparse
import json
import hashlib
import os
import shutil
import subprocess


TRAIN_ZIP_URLS = [
    "https://datalake.abuse.ch/malware-bazaar/daily/2026-02-06.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2026-02-05.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2026-02-04.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2026-02-03.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2026-02-02.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2026-02-01.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2026-01-31.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2026-01-30.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2026-01-29.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2026-01-28.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2026-01-27.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2026-01-26.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2026-01-25.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2026-01-24.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2026-01-23.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2026-01-22.zip",
]

VALIDATION_ZIP_URLS = [
    "https://datalake.abuse.ch/malware-bazaar/daily/2020-02-24.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2020-02-25.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2020-02-26.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2020-02-27.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2020-02-28.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2020-02-29.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2020-02-01.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2020-03-02.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2020-03-03.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2020-03-04.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2020-03-05.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2020-03-06.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2020-03-07.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2020-03-08.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2020-03-09.zip",
    "https://datalake.abuse.ch/malware-bazaar/daily/2020-03-10.zip",
]

PROFILE_DEFAULTS = {
    "train": {
        "zip_urls": TRAIN_ZIP_URLS,
        "target_dir": "MALWARE_DATASET",
        "required_samples": 5000,
    },
    "validation": {
        "zip_urls": VALIDATION_ZIP_URLS,
        "target_dir": "MALWARE_VALIDATION_DATASET",
        "required_samples": 1500,
    },
}


def get_file_sha256(filepath):
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except OSError:
        return None


def is_pe_file(filepath):
    try:
        with open(filepath, "rb") as f:
            return f.read(2) == b"MZ"
    except OSError:
        return False


def count_existing_samples(target_dir):
    if not os.path.exists(target_dir):
        return 0
    return len([f for f in os.listdir(target_dir) if f.endswith((".exe", ".dll"))])


def download_zip(url, zip_filename):
    print(f"[*] Downloading {url}...")
    result = subprocess.run(["wget", "-q", "--show-progress", "-O", zip_filename, url])
    return result.returncode == 0


def extract_zip(zip_filename, temp_dir, password):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    result = subprocess.run(["unzip", "-P", password, "-qq", "-o", zip_filename, "-d", temp_dir])
    return result.returncode == 0


def _load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Download and prepare malware datasets")
    parser.add_argument("--config", help="Path to JSON config file (defaults to 'download_malware_config.json' in the script directory)")
    parser.add_argument("--profile", help="Dataset profile to download")
    parser.add_argument("--zip-urls", nargs="*",
                        help="Override ZIP URLs list (space-separated)")
    parser.add_argument("--target-dir", help="Output directory for samples")
    parser.add_argument("--temp-dir", help="Temporary extraction directory")
    parser.add_argument("--required-samples", type=int, help="Target number of samples")
    parser.add_argument("--zip-password", default="infected", help="ZIP password")
    parser.add_argument("--keep-zips", action="store_true", help="Keep downloaded ZIPs")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temp extraction folder")
    args = parser.parse_args()

    # Load config: prefer --config; otherwise try to load a default config
    # named 'download_malware_config.json' located in the same directory as this script.
    if args.config:
        try:
            cfg = _load_config(args.config)
        except Exception as e:
            raise SystemExit(f"[!] Failed to load config '{args.config}': {e}")
    else:
        default_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "download_malware_config.json")
        if os.path.exists(default_config_path):
            try:
                cfg = _load_config(default_config_path)
                print(f"[*] Loaded default config from {default_config_path}")
            except Exception as e:
                print(f"[!] Failed to load default config '{default_config_path}': {e}")
                cfg = {}
        else:
            cfg = {}

    profiles = cfg.get("profiles") or PROFILE_DEFAULTS
    profile_name = args.profile or cfg.get("default_profile") or "train"
    if profile_name not in profiles:
        raise ValueError(f"Unknown profile '{profile_name}'. Available: {', '.join(sorted(profiles.keys()))}")
    profile_cfg = profiles[profile_name]

    zip_urls = args.zip_urls if args.zip_urls else profile_cfg.get("zip_urls", [])
    target_dir = args.target_dir or profile_cfg.get("target_dir")
    required_samples = args.required_samples or profile_cfg.get("required_samples")
    
    if not zip_urls or not target_dir or not required_samples:
        raise SystemExit("Missing required inputs: zip_urls, target_dir, required_samples")
    
    temp_dir = args.temp_dir or profile_cfg.get("temp_dir") or os.path.join(target_dir, f"_tmp_extract_{profile_name}")
    zip_password = args.zip_password or cfg.get("zip_password", "infected")
    keep_zips = args.keep_zips or cfg.get("keep_zips", False)
    keep_temp = args.keep_temp or cfg.get("keep_temp", False)

    os.makedirs(target_dir, exist_ok=True)

    count = count_existing_samples(target_dir)
    print(f"[*] Starting with {count} samples in {target_dir}.")

    for zip_url in zip_urls:
        if count >= required_samples:
            break

        zip_filename = zip_url.split("/")[-1]

        if not os.path.exists(zip_filename):
            if not download_zip(zip_url, zip_filename):
                print(f"[!] Download failed: {zip_url}")
                continue

        print(f"[*] Extracting {zip_filename}...")
        if not extract_zip(zip_filename, temp_dir, zip_password):
            print(f"[!] Extraction failed: {zip_filename}")
            continue

        print("[*] Processing files...")
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if count >= required_samples:
                    break
                src_path = os.path.join(root, file)
                if is_pe_file(src_path):
                    f_hash = get_file_sha256(src_path)
                    if not f_hash:
                        continue
                    ext = ".exe" if not file.lower().endswith(".dll") else ".dll"
                    dest = os.path.join(target_dir, f"{f_hash}{ext}")
                    if not os.path.exists(dest):
                        shutil.copy2(src_path, dest)
                        count += 1
                        if count % 100 == 0:
                            print(f"    - Total: {count}")

        if not keep_temp and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if not keep_zips and os.path.exists(zip_filename):
            os.remove(zip_filename)
        print(f"[*] Progress: {count}/{required_samples}")

    print(f"[+] Done. Final count: {count}")


if __name__ == "__main__":
    main()
