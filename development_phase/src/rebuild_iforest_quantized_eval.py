#!/usr/bin/env python3

import argparse
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest


def load_quantized_nml_dataset(nml_path: Path, expected_num_features: int, quantization_bits: int):
    with nml_path.open("rb") as f:
        header = f.read(6)
        if len(header) != 6:
            raise RuntimeError(f"Invalid nml header: {nml_path}")

        num_samples, num_features = struct.unpack("<IH", header)
        if int(num_features) != int(expected_num_features):
            raise RuntimeError(
                f"Feature mismatch in {nml_path}: {num_features} vs expected {expected_num_features}"
            )

        if quantization_bits < 1 or quantization_bits > 8:
            raise RuntimeError(f"Invalid quantization_bits={quantization_bits}")

        packed_feature_bytes = (int(num_features) * int(quantization_bits) + 7) // 8
        feature_mask = 0xFF if quantization_bits == 8 else (1 << quantization_bits) - 1

        matrix = np.zeros((int(num_samples), int(num_features)), dtype=np.uint8)
        labels = np.zeros(int(num_samples), dtype=np.uint8)

        for row in range(int(num_samples)):
            label_bytes = f.read(1)
            if len(label_bytes) != 1:
                raise RuntimeError(f"Unexpected EOF (label) in {nml_path} at row {row}")
            labels[row] = label_bytes[0]

            packed = f.read(packed_feature_bytes)
            if len(packed) != packed_feature_bytes:
                raise RuntimeError(f"Unexpected EOF (features) in {nml_path} at row {row}")

            for col in range(int(num_features)):
                bit_pos = col * quantization_bits
                byte_idx = bit_pos // 8
                bit_offset = bit_pos % 8

                if bit_offset + quantization_bits <= 8:
                    value = (packed[byte_idx] >> bit_offset) & feature_mask
                else:
                    bits_first = 8 - bit_offset
                    bits_second = quantization_bits - bits_first
                    low = (packed[byte_idx] >> bit_offset) & ((1 << bits_first) - 1)
                    high = packed[byte_idx + 1] & ((1 << bits_second) - 1)
                    value = low | (high << bits_first)

                matrix[row, col] = value

    return matrix.astype(np.float32), labels


def run_processing_data(processing_data_exe: Path,
                        input_csv: Path,
                        model_name: str,
                        quantizer_path: Path,
                        quant_bits: int):
    cmd = [
        str(processing_data_exe),
        "-ip", str(input_csv),
        "-mn", model_name,
        "-pt", "isolation",
        "-qb", str(int(quant_bits)),
        "-ro", "false",
        "-qp", str(quantizer_path),
    ]
    subprocess.run(cmd, check=True, cwd=str(processing_data_exe.parent))


def evaluate_predictions(decision_scores: np.ndarray, threshold: float, positive_label: int):
    pred_is_anomaly = decision_scores < threshold
    total = int(decision_scores.shape[0])
    anomaly_count = int(pred_is_anomaly.sum())

    if positive_label == 0:
        fpr = anomaly_count / total if total else 0.0
        tpr = 0.0
    else:
        tpr = anomaly_count / total if total else 0.0
        fpr = 0.0

    return {
        "total": total,
        "anomaly_count": anomaly_count,
        "anomaly_rate": float(anomaly_count / total) if total else 0.0,
        "fpr": float(fpr),
        "tpr": float(tpr),
    }


def main():
    parser = argparse.ArgumentParser(description="Rebuild IF from quantized benign-train and evaluate on benign/malware test")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--train-nml", type=Path, default=None)
    parser.add_argument("--quantizer", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    config_path = (args.config or (repo_root / "development_phase/results/iforest_optimized_config.json")).resolve()
    train_nml_path = (args.train_nml or (repo_root / "embedded_phase/core/models/isolation_forest/resources/iforest_ben_train_nml.bin")).resolve()
    quantizer_path = (args.quantizer or (repo_root / "embedded_phase/core/models/isolation_forest/resources/iforest_qtz.bin")).resolve()
    output_path = (args.output or (repo_root / "development_phase/reports/python_quantized_rebuild_eval.json")).resolve()

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    num_features = int(cfg["optimized_feature_set"]["n_features"])
    hyper = cfg["optimized_parameters"]["hyperparameters"]
    thr_cfg = cfg["optimized_parameters"]["thresholding"]
    threshold = float(thr_cfg["threshold"])
    saved_offset = float(thr_cfg["offset"])
    quant_bits = int(cfg.get("node_resource", {}).get("quantization_bits", 0))

    dp_path = repo_root / "embedded_phase/core/models/isolation_forest/resources/iforest_dp.txt"
    if dp_path.exists():
        for line in dp_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("quantization_coefficient,"):
                try:
                    quant_bits = int(line.split(",", 1)[1].strip())
                except ValueError:
                    pass
                break
    if quant_bits <= 0:
        quant_bits = 2

    tools_dir = repo_root / "tools/data_quantization"
    processing_data_exe = tools_dir / "processing_data"
    quantized_dir = tools_dir / "quantized_datasets"
    quantized_dir.mkdir(parents=True, exist_ok=True)

    ben_test_csv = repo_root / "development_phase/data/optimized/iforest_ben_test.csv"
    mal_test_csv = repo_root / "development_phase/data/optimized/iforest_mal_test.csv"
    ben_test_model = "iforest_ben_test"
    mal_test_model = "iforest_mal_test"

    run_processing_data(processing_data_exe, ben_test_csv, ben_test_model, quantizer_path, quant_bits)
    run_processing_data(processing_data_exe, mal_test_csv, mal_test_model, quantizer_path, quant_bits)

    ben_test_nml = quantized_dir / f"{ben_test_model}_nml.bin"
    mal_test_nml = quantized_dir / f"{mal_test_model}_nml.bin"

    x_train, _ = load_quantized_nml_dataset(train_nml_path, num_features, quant_bits)
    x_ben_test, _ = load_quantized_nml_dataset(ben_test_nml, num_features, quant_bits)
    x_mal_test, _ = load_quantized_nml_dataset(mal_test_nml, num_features, quant_bits)

    model = IsolationForest(
        n_estimators=int(hyper["n_estimators"]),
        max_samples=hyper["max_samples"],
        max_features=hyper["max_features"],
        bootstrap=bool(hyper["bootstrap"]),
        contamination=float(hyper["contamination"]),
        random_state=int(hyper["random_state"]),
        n_jobs=-1,
    )

    model.fit(x_train)

    ben_score_samples = model.score_samples(x_ben_test)
    mal_score_samples = model.score_samples(x_mal_test)

    ben_decision = ben_score_samples - saved_offset
    mal_decision = mal_score_samples - saved_offset

    ben_metrics = evaluate_predictions(ben_decision, threshold, positive_label=0)
    mal_metrics = evaluate_predictions(mal_decision, threshold, positive_label=1)

    report = {
        "config_path": str(config_path),
        "train_nml": str(train_nml_path),
        "ben_test_nml": str(ben_test_nml),
        "mal_test_nml": str(mal_test_nml),
        "num_features": num_features,
        "quantization_bits": quant_bits,
        "saved_threshold": threshold,
        "saved_offset": saved_offset,
        "model_offset_after_fit": float(model.offset_),
        "benign_test": ben_metrics,
        "malware_test": mal_metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
