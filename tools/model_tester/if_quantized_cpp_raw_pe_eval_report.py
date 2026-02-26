#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_report(report_path: Path) -> Tuple[Dict[str, Dict[str, str]], List[Tuple[float, float]], List[Tuple[float, float]]]:
    sections: Dict[str, Dict[str, str]] = {}
    pr_curve: List[Tuple[float, float]] = []
    roc_curve: List[Tuple[float, float]] = []

    current_section = ""
    with report_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].strip().lower()
                sections.setdefault(current_section, {})
                continue

            if current_section == "pr_curve":
                if line.lower() == "recall,precision":
                    continue
                parts = line.split(",")
                if len(parts) == 2:
                    pr_curve.append((float(parts[0]), float(parts[1])))
                continue

            if current_section == "roc_curve":
                if line.lower() == "fpr,tpr":
                    continue
                parts = line.split(",")
                if len(parts) == 2:
                    roc_curve.append((float(parts[0]), float(parts[1])))
                continue

            if "=" in line and current_section:
                key, value = line.split("=", 1)
                sections[current_section][key.strip()] = value.strip()

    return sections, pr_curve, roc_curve


def value(section: Dict[str, str], key: str, default: float = 0.0) -> float:
    text = section.get(key)
    if text is None:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def save_sample_count_chart(sample_counts: Dict[str, str], output_dir: Path) -> None:
    labels = ["Train", "Validation", "Test"]
    values = [
        value(sample_counts, "train_samples"),
        value(sample_counts, "validation_samples"),
        value(sample_counts, "test_samples"),
    ]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, values)
    plt.title("Dataset Sample Counts")
    plt.ylabel("Samples")
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{int(val)}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_dir / "if_quantized_sample_counts.png", dpi=160)
    plt.close()


def save_model_size_chart(model_section: Dict[str, str], output_dir: Path) -> None:
    ram_mb = value(model_section, "model_ram_size_bytes") / (1024.0 * 1024.0)
    file_mb = value(model_section, "model_file_size_bytes") / (1024.0 * 1024.0)

    plt.figure(figsize=(8, 4.5))
    labels = ["RAM Size", "Model File Size"]
    values = [ram_mb, file_mb]
    bars = plt.bar(labels, values)
    plt.title("Model Footprint")
    plt.ylabel("Size (MB)")
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{val:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_dir / "if_quantized_model_sizes.png", dpi=160)
    plt.close()


def save_speed_chart(speed_section: Dict[str, str], output_dir: Path) -> None:
    ms_per_file = value(speed_section, "avg_inference_ms_per_file")
    ms_per_mb = value(speed_section, "avg_inference_ms_per_mb")

    plt.figure(figsize=(8, 4.5))
    labels = ["Avg ms / file", "Avg ms / MB"]
    values = [ms_per_file, ms_per_mb]
    bars = plt.bar(labels, values)
    plt.title("Inference Speed")
    plt.ylabel("Milliseconds")
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{val:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_dir / "if_quantized_inference_speed.png", dpi=160)
    plt.close()


def save_pr_curve(pr_curve: List[Tuple[float, float]], metrics_section: Dict[str, str], output_dir: Path) -> None:
    recalls = [point[0] for point in pr_curve]
    precisions = [point[1] for point in pr_curve]
    ap = value(metrics_section, "average_precision")

    plt.figure(figsize=(6.5, 6.0))
    plt.plot(recalls, precisions, label=f"PR Curve (AP={ap:.4f})")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_dir / "if_quantized_pr_curve.png", dpi=180)
    plt.close()


def save_roc_curve(roc_curve: List[Tuple[float, float]], metrics_section: Dict[str, str], output_dir: Path) -> None:
    fprs = [point[0] for point in roc_curve]
    tprs = [point[1] for point in roc_curve]
    auc = value(metrics_section, "roc_auc")

    plt.figure(figsize=(6.5, 6.0))
    plt.plot(fprs, tprs, label=f"ROC Curve (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", alpha=0.6, label="Random")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "if_quantized_roc_curve.png", dpi=180)
    plt.close()


def save_metrics_panel(metrics_section: Dict[str, str], speed_section: Dict[str, str], output_dir: Path) -> None:
    fpr = value(metrics_section, "fpr")
    tpr = value(metrics_section, "tpr")
    auc = value(metrics_section, "roc_auc")
    ap = value(metrics_section, "average_precision")
    ms_per_file = value(speed_section, "avg_inference_ms_per_file")
    ms_per_mb = value(speed_section, "avg_inference_ms_per_mb")

    text = (
        "Raw-PE Quantized C++ Evaluation\n\n"
        f"FPR: {fpr:.6f}\n"
        f"TPR: {tpr:.6f}\n"
        f"ROC AUC: {auc:.6f}\n"
        f"AP: {ap:.6f}\n"
        f"Avg inference speed per file: {ms_per_file:.6f} ms\n"
        f"Avg inference speed per MB: {ms_per_mb:.6f} ms"
    )

    plt.figure(figsize=(8.5, 4.0))
    plt.axis("off")
    plt.text(0.02, 0.98, text, va="top", ha="left", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "if_quantized_metrics_summary.png", dpi=180)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate charts from if_quantized_cpp_raw_pe_eval.txt")
    parser.add_argument(
        "--report-txt",
        type=Path,
        default=Path("development_phase/reports/if_quantized_cpp_raw_pe_eval.txt"),
        help="Path to the benchmark txt report",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for generated charts/images",
    )
    args = parser.parse_args()

    if not args.report_txt.exists():
        raise FileNotFoundError(f"Report file not found: {args.report_txt}")

    sections, pr_curve, roc_curve = parse_report(args.report_txt)

    sample_counts = sections.get("sample_counts", {})
    model_section = sections.get("model", {})
    metrics_section = sections.get("metrics", {})
    speed_section = sections.get("speed", {})

    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_sample_count_chart(sample_counts, args.output_dir)
    save_model_size_chart(model_section, args.output_dir)
    save_speed_chart(speed_section, args.output_dir)
    save_pr_curve(pr_curve, metrics_section, args.output_dir)
    save_roc_curve(roc_curve, metrics_section, args.output_dir)
    save_metrics_panel(metrics_section, speed_section, args.output_dir)

    print(f"Report charts generated in: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
