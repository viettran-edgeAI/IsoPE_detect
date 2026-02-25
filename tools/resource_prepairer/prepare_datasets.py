#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
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


def _delete_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected {label} file not found: {path}")


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
        description="Prepare IF quantized resources for train/validation datasets"
    )
    parser.add_argument("-c", "--config", default=str(config_path))
    parser.add_argument("--benign-train", default=config.get("benign_train", ""))
    parser.add_argument("--benign-val", default=config.get("benign_val", ""))
    parser.add_argument("--malware-val", default=config.get("malware_val", ""))
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

    benign_train = _resolve_path(repo_root, args.benign_train)
    benign_val = _resolve_path(repo_root, args.benign_val)
    malware_val = _resolve_path(repo_root, args.malware_val)
    output_dir = _resolve_path(repo_root, args.output_dir)
    quant_tool = _resolve_path(repo_root, args.quantization_tool)
    dev_cfg_dir: Path | None = None
    if args.dev_optimized_config_dir:
        dev_cfg_dir = _resolve_path(repo_root, args.dev_optimized_config_dir)

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

    model_name = args.model_name.strip()
    if not model_name:
        raise ValueError("model_name cannot be empty")

    print("=== Resource Preparation Started ===")
    print(f"Repo root          : {repo_root}")
    print(f"Model name         : {model_name}")
    print(f"Quantization bits  : {args.quantization_bits}")
    print(f"Output directory   : {output_dir}")
    if dev_cfg_dir:
        print(f"Dev config dir     : {dev_cfg_dir}")

    # Step 1: fit quantizer on benign_train (produces <model>_qtz.bin + <model>_dp.txt + <model>_nml.bin)
    _run_quantization(
        quant_tool=quant_tool,
        input_csv=benign_train,
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
        input_csv=benign_val,
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
        input_csv=malware_val,
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

    # Keep train qtz/dp only; validation splits should not keep split-specific dp/qtz artifacts.
    _delete_if_exists(output_dir / f"{benign_val_prefix}_qtz.bin")
    _delete_if_exists(output_dir / f"{benign_val_prefix}_dp.txt")
    _delete_if_exists(output_dir / f"{malware_val_prefix}_qtz.bin")
    _delete_if_exists(output_dir / f"{malware_val_prefix}_dp.txt")

    print("\n=== Resource Preparation Complete ===")
    print("Generated required files:")
    print(f"  - {train_quantizer.name}")
    print(f"  - {train_dp.name}")
    print(f"  - {train_nml_bin.name}")
    print(f"  - {train_alias.name}")
    print(f"  - {benign_val_nml_bin.name}")
    print(f"  - {malware_val_nml_bin.name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
