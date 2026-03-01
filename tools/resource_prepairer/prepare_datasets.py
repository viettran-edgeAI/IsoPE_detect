#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict


def _unwrap_config_value(raw: Any) -> Any:
    if isinstance(raw, dict) and "value" in raw:
        return raw["value"]
    return raw


def _parse_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        raise ValueError("Boolean value is missing")
    lowered = str(raw).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw}")


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        parsed = json.load(f)

    normalized: Dict[str, Any] = {}
    for key, value in parsed.items():
        normalized[key] = _unwrap_config_value(value)
    return normalized


def _resolve_path(repo_root: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def _run_quantization(
    *,
    quant_tool: Path,
    input_csv: Path,
    output_dir: Path,
    model_name: str,
    quantization_bits: int,
    header: str,
    problem_type: str,
    remove_outliers: bool,
    quantizer_path: Path | None,
    repo_root: Path,
) -> None:
    cmd = [
        str(quant_tool),
        "-ip",
        str(input_csv),
        "-od",
        str(output_dir),
        "-mn",
        model_name,
        "-qb",
        str(quantization_bits),
        "-hd",
        header,
        "-pt",
        problem_type,
        "-ro",
        "true" if remove_outliers else "false",
    ]
    if quantizer_path is not None:
        cmd.extend(["-qp", str(quantizer_path)])

    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def _load_feature_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        parsed = json.load(f)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"Invalid optimized feature list JSON: {path}")
    names: list[str] = []
    for entry in parsed:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(f"Invalid feature entry in {path}: {entry}")
        names.append(entry)
    return names


def _filter_csv_columns(input_csv: Path, output_csv: Path, selected_features: list[str]) -> None:
    with input_csv.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.reader(infile)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Input CSV is empty: {input_csv}")

        index_by_name = {name: idx for idx, name in enumerate(header)}
        missing = [name for name in selected_features if name not in index_by_name]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"Input CSV missing {len(missing)} required feature columns: {preview}"
            )

        selected_indices = [index_by_name[name] for name in selected_features]

        with output_csv.open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(selected_features)
            for row in reader:
                writer.writerow([row[idx] for idx in selected_indices])


def _delete_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected {label} file not found: {path}")


def _resolve_split_csv(input_dir: Path, model_name: str, split_suffix: str) -> Path:
    preferred = input_dir / f"{model_name}_{split_suffix}.csv"
    if preferred.exists() and preferred.is_file():
        return preferred

    matches = sorted(input_dir.glob(f"*_{split_suffix}.csv"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        joined = ", ".join(m.name for m in matches[:5])
        raise ValueError(
            f"Multiple candidates found for split '{split_suffix}' in {input_dir}: {joined}. "
            f"Provide a unique file named {model_name}_{split_suffix}.csv"
        )

    raise FileNotFoundError(
        f"Could not find CSV for split '{split_suffix}' in {input_dir}. "
        f"Expected {model_name}_{split_suffix}.csv"
    )


def _resolve_input_csvs(input_dir: Path, model_name: str) -> dict[str, Path]:
    split_map = {
        "benign_train": "ben_train",
        "benign_val": "ben_val",
        "malware_val": "mal_val",
        "benign_test": "ben_test",
        "malware_test": "mal_test",
    }
    resolved: dict[str, Path] = {}
    for label, suffix in split_map.items():
        resolved[label] = _resolve_split_csv(input_dir, model_name, suffix)
    return resolved


def main() -> int:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "-c",
        "--config",
        default="tools/resource_prepairer/resource_prepairer_config.json",
        help="Path to config JSON",
    )
    pre_args, _ = pre_parser.parse_known_args()

    config_path = _resolve_path(repo_root, pre_args.config)
    config = _load_config(config_path)

    parser = argparse.ArgumentParser(
        description="Prepare IF quantized resources for train/validation/test datasets"
    )
    parser.add_argument("-c", "--config", default=str(config_path))
    parser.add_argument(
        "--input-dir",
        default=config.get("input_dir", ""),
        help="Directory containing split CSV files for train/validation/test",
    )
    parser.add_argument(
        "--optimized-features-json",
        default=config.get("optimized_features_json", ""),
        help="Path to JSON array of selected feature names. If omitted, tries <output-dir>/<model-name>_optimized_features.json",
    )
    parser.add_argument("--output-dir", default=config.get("output_dir", ""))
    parser.add_argument("--model-name", default=config.get("model_name", "iforest"))
    parser.add_argument(
        "--quantization-bits",
        type=int,
        default=int(config.get("quantization_bits", 3)),
    )
    parser.add_argument(
        "--quantization-tool",
        default=config.get("quantization_tool", "tools/data_quantization/processing_data"),
    )
    parser.add_argument("--header", default=str(config.get("header", "auto")))
    parser.add_argument(
        "--problem-type",
        default=str(config.get("problem_type", "isolation")),
    )
    parser.add_argument(
        "--remove-outliers",
        default=str(config.get("remove_outliers", "false")),
        help="true/false",
    )
    parser.add_argument(
        "--dev-optimized-config-dir",
        default=config.get("dev_optimized_config_dir", ""),
        help="Directory of additional JSON configs to copy into the output directory."
    )
    args = parser.parse_args()

    input_dir = _resolve_path(repo_root, args.input_dir)
    output_dir = _resolve_path(repo_root, args.output_dir)
    quant_tool = _resolve_path(repo_root, args.quantization_tool)
    dev_cfg_dir: Path | None = None
    if args.dev_optimized_config_dir:
        dev_cfg_dir = _resolve_path(repo_root, args.dev_optimized_config_dir)

    features_json_path: Path | None = None
    if args.optimized_features_json:
        features_json_path = _resolve_path(repo_root, args.optimized_features_json)

    model_name = args.model_name.strip()
    if not model_name:
        raise ValueError("model_name cannot be empty")

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found or not a directory: {input_dir}")

    split_paths = _resolve_input_csvs(input_dir, model_name)
    benign_train = split_paths["benign_train"]
    benign_val = split_paths["benign_val"]
    malware_val = split_paths["malware_val"]
    benign_test = split_paths["benign_test"]
    malware_test = split_paths["malware_test"]

    if args.quantization_bits < 1 or args.quantization_bits > 8:
        raise ValueError("quantization_bits must be in range [1, 8]")

    header = str(args.header).strip().lower()
    if header not in {"auto", "yes", "no"}:
        raise ValueError("header must be one of: auto, yes, no")

    problem_type = str(args.problem_type).strip().lower()
    if problem_type not in {"isolation", "classification", "regression"}:
        raise ValueError(
            "problem_type must be one of: isolation, classification, regression"
        )

    remove_outliers = _parse_bool(args.remove_outliers)

    for dataset_path, label in [
        (benign_train, "benign_train"),
        (benign_val, "benign_val"),
        (malware_val, "malware_val"),
        (benign_test, "benign_test"),
        (malware_test, "malware_test"),
    ]:
        _ensure_file(dataset_path, label)

    if not quant_tool.exists():
        raise FileNotFoundError(
            f"Quantization tool not found: {quant_tool}. Build it first via tools/data_quantization/Makefile"
        )
    if not quant_tool.is_file():
        raise RuntimeError(f"Quantization tool path is not a file: {quant_tool}")

    output_dir.mkdir(parents=True, exist_ok=True)
    # optionally copy development-optimized JSON configs into output_dir
    if dev_cfg_dir:
        if not dev_cfg_dir.is_dir():
            raise FileNotFoundError(
                f"dev_optimized_config_dir not a directory: {dev_cfg_dir}"
            )
        json_files = list(dev_cfg_dir.glob("*.json"))
        for src in json_files:
            shutil.copy2(src, output_dir / src.name)
        print(f"Copied {len(json_files)} JSON file(s) from {dev_cfg_dir} to {output_dir}")

    if features_json_path is None:
        candidate = output_dir / f"{model_name}_optimized_features.json"
        if candidate.exists():
            features_json_path = candidate

    selected_features: list[str] | None = None
    if features_json_path is not None:
        _ensure_file(features_json_path, "optimized feature list JSON")
        selected_features = _load_feature_list(features_json_path)

    print("=== Resource Preparation Started ===")
    print(f"Repo root          : {repo_root}")
    print(f"Model name         : {model_name}")
    print(f"Input directory    : {input_dir}")
    print(f"Quantization bits  : {args.quantization_bits}")
    print(f"Output directory   : {output_dir}")
    if dev_cfg_dir:
        print(f"Dev config dir     : {dev_cfg_dir}")
    if selected_features is not None:
        print(f"Selected features  : {len(selected_features)} ({features_json_path})")

    temp_dir = Path(tempfile.mkdtemp(prefix="resource_prepairer_"))
    try:
        benign_train_quant_input = benign_train
        benign_val_quant_input = benign_val
        malware_val_quant_input = malware_val
        benign_test_quant_input = benign_test
        malware_test_quant_input = malware_test

        if selected_features is not None:
            benign_train_quant_input = temp_dir / "benign_train_selected.csv"
            benign_val_quant_input = temp_dir / "benign_val_selected.csv"
            malware_val_quant_input = temp_dir / "malware_val_selected.csv"
            benign_test_quant_input = temp_dir / "benign_test_selected.csv"
            malware_test_quant_input = temp_dir / "malware_test_selected.csv"

            _filter_csv_columns(benign_train, benign_train_quant_input, selected_features)
            _filter_csv_columns(benign_val, benign_val_quant_input, selected_features)
            _filter_csv_columns(malware_val, malware_val_quant_input, selected_features)
            _filter_csv_columns(benign_test, benign_test_quant_input, selected_features)
            _filter_csv_columns(malware_test, malware_test_quant_input, selected_features)

        # Step 1: fit quantizer on benign_train (produces <model>_qtz.bin + <model>_dp.txt + <model>_nml.bin)
        _run_quantization(
            quant_tool=quant_tool,
            input_csv=benign_train_quant_input,
            output_dir=output_dir,
            model_name=model_name,
            quantization_bits=args.quantization_bits,
            header=header,
            problem_type=problem_type,
            remove_outliers=remove_outliers,
            quantizer_path=None,
            repo_root=repo_root,
        )

        train_quantizer = output_dir / f"{model_name}_qtz.bin"
        train_dp = output_dir / f"{model_name}_dp.txt"
        train_nml_bin = output_dir / f"{model_name}_nml.bin"

        _ensure_file(train_quantizer, "train quantizer")
        _ensure_file(train_dp, "train data-params")
        _ensure_file(train_nml_bin, "train quantized dataset")

        # Optional canonical alias for training split used by runtime fallback logic.
        train_alias = output_dir / f"{model_name}_ben_train_nml.bin"
        shutil.copy2(train_nml_bin, train_alias)

        # Step 2: quantize benign_val using the train quantizer (transform-only)
        benign_val_prefix = f"{model_name}_ben_val"
        _run_quantization(
            quant_tool=quant_tool,
            input_csv=benign_val_quant_input,
            output_dir=output_dir,
            model_name=benign_val_prefix,
            quantization_bits=args.quantization_bits,
            header=header,
            problem_type=problem_type,
            remove_outliers=remove_outliers,
            quantizer_path=train_quantizer,
            repo_root=repo_root,
        )

        benign_val_nml_bin = output_dir / f"{benign_val_prefix}_nml.bin"
        _ensure_file(benign_val_nml_bin, "benign validation quantized dataset")

        # Step 3: quantize malware_val using the same train quantizer (transform-only)
        malware_val_prefix = f"{model_name}_mal_val"
        _run_quantization(
            quant_tool=quant_tool,
            input_csv=malware_val_quant_input,
            output_dir=output_dir,
            model_name=malware_val_prefix,
            quantization_bits=args.quantization_bits,
            header=header,
            problem_type=problem_type,
            remove_outliers=remove_outliers,
            quantizer_path=train_quantizer,
            repo_root=repo_root,
        )

        malware_val_nml_bin = output_dir / f"{malware_val_prefix}_nml.bin"
        _ensure_file(malware_val_nml_bin, "malware validation quantized dataset")

        # Step 4: quantize benign_test using the same train quantizer (transform-only)
        benign_test_prefix = f"{model_name}_ben_test"
        _run_quantization(
            quant_tool=quant_tool,
            input_csv=benign_test_quant_input,
            output_dir=output_dir,
            model_name=benign_test_prefix,
            quantization_bits=args.quantization_bits,
            header=header,
            problem_type=problem_type,
            remove_outliers=remove_outliers,
            quantizer_path=train_quantizer,
            repo_root=repo_root,
        )

        benign_test_nml_bin = output_dir / f"{benign_test_prefix}_nml.bin"
        _ensure_file(benign_test_nml_bin, "benign test quantized dataset")

        # Step 5: quantize malware_test using the same train quantizer (transform-only)
        malware_test_prefix = f"{model_name}_mal_test"
        _run_quantization(
            quant_tool=quant_tool,
            input_csv=malware_test_quant_input,
            output_dir=output_dir,
            model_name=malware_test_prefix,
            quantization_bits=args.quantization_bits,
            header=header,
            problem_type=problem_type,
            remove_outliers=remove_outliers,
            quantizer_path=train_quantizer,
            repo_root=repo_root,
        )

        malware_test_nml_bin = output_dir / f"{malware_test_prefix}_nml.bin"
        _ensure_file(malware_test_nml_bin, "malware test quantized dataset")

        # Keep train qtz/dp only; split-specific dp/qtz artifacts should be removed.
        _delete_if_exists(output_dir / f"{benign_val_prefix}_qtz.bin")
        _delete_if_exists(output_dir / f"{benign_val_prefix}_dp.txt")
        _delete_if_exists(output_dir / f"{malware_val_prefix}_qtz.bin")
        _delete_if_exists(output_dir / f"{malware_val_prefix}_dp.txt")
        _delete_if_exists(output_dir / f"{benign_test_prefix}_qtz.bin")
        _delete_if_exists(output_dir / f"{benign_test_prefix}_dp.txt")
        _delete_if_exists(output_dir / f"{malware_test_prefix}_qtz.bin")
        _delete_if_exists(output_dir / f"{malware_test_prefix}_dp.txt")

        print("\n=== Resource Preparation Complete ===")
        print("Generated required files:")
        print(f"  - {train_quantizer.name}")
        print(f"  - {train_dp.name}")
        print(f"  - {train_nml_bin.name}")
        print(f"  - {train_alias.name}")
        print(f"  - {benign_val_nml_bin.name}")
        print(f"  - {malware_val_nml_bin.name}")
        print(f"  - {benign_test_nml_bin.name}")
        print(f"  - {malware_test_nml_bin.name}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
