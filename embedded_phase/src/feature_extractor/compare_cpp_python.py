#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path


def load_python_extractor(root: Path):
    extractor_path = root / "development_phase" / "src" / "feature_extraction.py"
    spec = importlib.util.spec_from_file_location("dev_feature_extraction", extractor_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load python extractor module: {extractor_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "extract_features"):
        raise RuntimeError("development_phase/src/feature_extraction.py has no extract_features")
    return module.extract_features


def run_cpp_extractor(extractor_bin: Path, feature_names_path: Path, target_file: Path):
    command = [
        str(extractor_bin),
        "--feature-names",
        str(feature_names_path),
        "--format",
        "jsonl",
        str(target_file),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=True)
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"No JSON output from C++ extractor for {target_file}")
    return json.loads(lines[0])


def numeric(value):
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return float(value)
    try:
        return float(value)
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Compare C++ LIEF feature extractor with Python extractor")
    parser.add_argument("--extractor", type=Path, default=Path("embedded_phase/src/feature_extractor/lief_feature_extractor"))
    parser.add_argument("--feature-names", type=Path, default=Path("development_phase/results/feature_names.json"))
    parser.add_argument("--benign-dir", type=Path, default=Path("datasets/BENIGN_TEST_DATASET"))
    parser.add_argument("--malware-dir", type=Path, default=Path("datasets/MALWARE_TEST_DATASET"))
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output-dir", type=Path, default=Path("embedded_phase/src/feature_extractor/validation"))
    args = parser.parse_args()

    root = Path.cwd().resolve()
    feature_names = json.loads((root / args.feature_names).resolve().read_text(encoding="utf-8"))
    if not isinstance(feature_names, list) or not feature_names:
        raise RuntimeError("feature_names.json must be a non-empty JSON array")

    extract_python = load_python_extractor(root)
    extractor_bin = (root / args.extractor).resolve()
    if not extractor_bin.exists():
        raise RuntimeError(f"Missing C++ extractor binary: {extractor_bin}")

    benign_files = sorted([p.resolve() for p in (root / args.benign_dir).iterdir() if p.is_file()])[: args.samples_per_class]
    malware_files = sorted([p.resolve() for p in (root / args.malware_dir).iterdir() if p.is_file()])[: args.samples_per_class]
    samples = [("benign", p) for p in benign_files] + [("malware", p) for p in malware_files]
    if not samples:
        raise RuntimeError("No samples found")

    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_csv = output_dir / "cpp_processing_time.csv"
    details_csv = output_dir / "feature_parity_details.csv"
    summary_json = output_dir / "parity_summary.json"

    timing_rows = []
    detail_rows = []

    total_mismatches = 0
    files_with_mismatch = 0
    parse_failures = 0

    for label, sample_path in samples:
        py_raw = extract_python(str(sample_path))
        if py_raw is None:
            parse_failures += 1
            continue
        py_features = {name: numeric(py_raw.get(name, 0.0)) for name in feature_names}

        cpp_result = run_cpp_extractor(extractor_bin, (root / args.feature_names).resolve(), sample_path)
        parse_ok = bool(cpp_result.get("parse_ok", False))
        cpp_features_raw = cpp_result.get("features", {})
        cpp_features = {name: numeric(cpp_features_raw.get(name, 0.0)) for name in feature_names}
        processing_time_ms = numeric(cpp_result.get("processing_time_ms", 0.0))

        timing_rows.append(
            {
                "label": label,
                "file": str(sample_path.relative_to(root)),
                "parse_ok": int(parse_ok),
                "processing_time_ms": f"{processing_time_ms:.6f}",
            }
        )

        file_mismatches = 0
        for feature in feature_names:
            py_val = py_features[feature]
            cpp_val = cpp_features[feature]
            abs_diff = abs(py_val - cpp_val)
            is_match = abs_diff <= args.tolerance
            if not is_match:
                file_mismatches += 1

            detail_rows.append(
                {
                    "label": label,
                    "file": str(sample_path.relative_to(root)),
                    "feature": feature,
                    "python_value": f"{py_val:.9f}",
                    "cpp_value": f"{cpp_val:.9f}",
                    "abs_diff": f"{abs_diff:.9f}",
                    "match": int(is_match),
                }
            )

        total_mismatches += file_mismatches
        if file_mismatches > 0 or not parse_ok:
            files_with_mismatch += 1

    with timing_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "file", "parse_ok", "processing_time_ms"])
        writer.writeheader()
        writer.writerows(timing_rows)

    with details_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "file", "feature", "python_value", "cpp_value", "abs_diff", "match"],
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    summary = {
        "samples_tested": len(samples),
        "features_per_sample": len(feature_names),
        "parse_failures_python": parse_failures,
        "files_with_mismatch_or_parse_error": files_with_mismatch,
        "total_feature_mismatches": total_mismatches,
        "tolerance": args.tolerance,
        "timing_report": str(timing_csv.relative_to(root)),
        "detail_report": str(details_csv.relative_to(root)),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
