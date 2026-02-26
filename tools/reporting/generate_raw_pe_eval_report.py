#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


SectionMap = Dict[str, Dict[str, str]]
PointList = List[Tuple[float, float]]


def parse_report(report_path: Path) -> Tuple[SectionMap, PointList, PointList]:
	sections: SectionMap = {}
	pr_curve: PointList = []
	roc_curve: PointList = []

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


def int_value(section: Dict[str, str], key: str, default: int = 0) -> int:
	return int(round(value(section, key, float(default))))


def save_sample_count_chart(sample_counts: Dict[str, str], output_dir: Path) -> Path:
	train_benign = int_value(sample_counts, "train_samples")
	train_malware = int_value(sample_counts, "train_malware_samples", 0)
	val_benign = int_value(sample_counts, "validation_benign_samples")
	val_malware = int_value(sample_counts, "validation_malware_samples")

	groups = ["Train", "Validation"]
	benign_values = [train_benign, val_benign]
	malware_values = [train_malware, val_malware]

	x = list(range(len(groups)))
	width = 0.35

	plt.figure(figsize=(8, 4.8))
	benign_bars = plt.bar([pos - width / 2 for pos in x], benign_values, width=width, color="#2ca02c", label="Benign")
	malware_bars = plt.bar([pos + width / 2 for pos in x], malware_values, width=width, color="#ff7f0e", label="Malware")

	plt.xticks(x, groups)
	plt.title("Training and Validation Sample Counts")
	plt.ylabel("Samples")
	plt.legend()

	for bars in (benign_bars, malware_bars):
		for bar in bars:
			y = bar.get_height()
			plt.text(bar.get_x() + bar.get_width() / 2.0, y, f"{int(y)}", ha="center", va="bottom")

	plt.tight_layout()
	output_path = output_dir / "raw_pe_sample_counts.png"
	plt.savefig(output_path, dpi=170)
	plt.close()
	return output_path


def save_pr_curve(pr_curve: PointList, metrics: Dict[str, str], output_dir: Path) -> Path:
	recalls = [point[0] for point in pr_curve]
	precisions = [point[1] for point in pr_curve]
	ap = value(metrics, "average_precision")

	plt.figure(figsize=(6.6, 6.2))
	plt.plot(recalls, precisions, color="#1f77b4", linewidth=2.0, label=f"PR Curve (AP={ap:.4f})")
	plt.xlim(0.0, 1.0)
	plt.ylim(0.0, 1.05)
	plt.xlabel("Recall")
	plt.ylabel("Precision")
	plt.title("Precision-Recall Curve")
	plt.grid(True, alpha=0.25)
	plt.legend(loc="lower left")
	plt.tight_layout()
	output_path = output_dir / "raw_pe_pr_curve.png"
	plt.savefig(output_path, dpi=180)
	plt.close()
	return output_path


def save_roc_curve_logfpr(roc_curve: PointList, metrics: Dict[str, str], output_dir: Path) -> Path:
	auc = value(metrics, "roc_auc")

	# plot full ROC curve points first (converted to log-scale on FPR)
	x_curve: List[float] = []
	y_curve: List[float] = []
	for (f, t) in roc_curve:
		# avoid log10(0) by clamping to a small floor
		clamped = max(f, 1e-4)
		x_curve.append(math.log10(clamped))
		y_curve.append(t)

	# single test-set metrics for annotation
	raw_fpr = value(metrics, "fpr")
	fpr = max(raw_fpr, 1e-4)
	tpr = value(metrics, "tpr")

	plt.figure(figsize=(6.8, 6.2))
	# ROC curve line
	plt.plot(x_curve, y_curve, color="#1f77b4", linewidth=2.0, label=f"ROC Curve (AUC={auc:.4f})")
	# highlight the specific test-point
	x_point = math.log10(fpr)
	plt.scatter([x_point], [tpr], color="red", s=30, label="[FPR, TPR] point", zorder=3)

	# draw crosshair lines through the test-set point to axes
	plt.axhline(y=tpr, color="red", linestyle="--", linewidth=1, alpha=0.5)
	plt.axvline(x=x_point, color="red", linestyle="--", linewidth=1, alpha=0.5)

	annotation_text = f"Test-set [FPR, TPR]\n[{raw_fpr:.6f}, {tpr:.6f}]"
	plt.annotate(
		annotation_text,
		xy=(x_point, tpr),
		xytext=(12, -30),  # slightly below the point
		textcoords="offset points",
		ha="left",
		va="bottom",
		fontsize=9,
		bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.7", "alpha": 0.9},
		arrowprops={"arrowstyle": "->", "color": "0.4", "lw": 0.8},
	)

	# lower legend slightly so it doesn't obscure curve/marker
	plt.legend(loc="lower right", bbox_to_anchor=(1, -0.05))

	plt.xlim(-4.0, 0.0)
	plt.ylim(0.0, 1.05)
	plt.xticks([-4, -3, -2, -1, 0], ["10^-4", "10^-3", "10^-2", "10^-1", "10^0"])
	plt.xlabel("False Positive Rate (log scale ticks)")
	plt.ylabel("True Positive Rate")
	plt.title("ROC Curve")
	plt.grid(True, alpha=0.25)
	plt.legend(loc="lower right")
	plt.tight_layout()
	output_path = output_dir / "raw_pe_roc_curve_log_fpr.png"
	plt.savefig(output_path, dpi=180)
	plt.close()
	return output_path


def write_markdown_report(
	sections: SectionMap,
	pr_chart: Path,
	roc_chart: Path,
	markdown_path: Path,
	quant_bits: int,
) -> None:
	metadata = sections.get("metadata", {})
	sample_counts = sections.get("sample_counts", {})
	model = sections.get("model", {})
	metrics = sections.get("metrics", {})
	speed = sections.get("speed", {})

	now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

	text = f"""# Raw-PE Evaluation Report (Quantization {quant_bits}-bit)

Generated at: {now}

## Metadata

- Repo root: {metadata.get('repo_root', '')}
- Model name: {metadata.get('model_name', '')}
- Threshold: {metadata.get('threshold', '')}
- Quantization bits: {quant_bits}

## Sample Counts

- Train benign: {sample_counts.get('train_samples', '0')}
- Train malware: {sample_counts.get('train_malware_samples', '0')}
- Validation benign: {sample_counts.get('validation_benign_samples', '0')}
- Validation malware: {sample_counts.get('validation_malware_samples', '0')}
- Validation total: {sample_counts.get('validation_samples', '0')}
- Test benign: {sample_counts.get('test_benign_samples', '0')}
- Test malware: {sample_counts.get('test_malware_samples', '0')}
- Test total: {sample_counts.get('test_samples', '0')}

## Model Parameters and Footprint

- Model RAM size (bytes): {model.get('model_ram_size_bytes', '0')}
- Model file size (bytes): {model.get('model_file_size_bytes', '0')}
- Model file path: {model.get('model_file_path', '')}

## Metrics

- FPR: {metrics.get('fpr', '0')}
- TPR: {metrics.get('tpr', '0')}
- ROC-AUC: {metrics.get('roc_auc', '0')}
- AP: {metrics.get('average_precision', '0')}
- PRC-AUC: {metrics.get('prc_auc', '0')}

## Performance

- Benign total inference time (sec): {speed.get('benign_total_time_sec', '0')}
- Malware total inference time (sec): {speed.get('malware_total_time_sec', '0')}
- Total inference time (sec): {speed.get('total_inference_time_sec', '0')}
- Average inference speed per file (ms): {speed.get('avg_inference_ms_per_file', '0')}
- Average inference speed per MB (ms): {speed.get('avg_inference_ms_per_mb', '0')}

## Charts

### Precision-Recall Curve

![PR Curve]({pr_chart.name})

### ROC Curve (log-spaced FPR ticks)

![ROC Curve]({roc_chart.name})
"""

	markdown_path.write_text(text, encoding="utf-8")


def main() -> int:
	parser = argparse.ArgumentParser(description="Generate raw-PE evaluation charts and markdown report")
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
		help="Directory for generated charts and report",
	)
	parser.add_argument(
		"--output-md",
		type=Path,
		default=Path("reports/README.md"),
		help="Output markdown report path",
	)
	parser.add_argument(
		"--quantization-bits",
		type=int,
		default=4,
		help="Quantization bits used for the measured run",
	)
	args = parser.parse_args()

	if not args.report_txt.exists():
		raise FileNotFoundError(f"Report file not found: {args.report_txt}")

	sections, pr_curve, roc_curve = parse_report(args.report_txt)
	metrics = sections.get("metrics", {})

	args.output_dir.mkdir(parents=True, exist_ok=True)
	args.output_md.parent.mkdir(parents=True, exist_ok=True)

	pr_chart = save_pr_curve(pr_curve, metrics, args.output_dir)
	roc_chart = save_roc_curve_logfpr(roc_curve, metrics, args.output_dir)

	write_markdown_report(
		sections=sections,
		pr_chart=pr_chart,
		roc_chart=roc_chart,
		markdown_path=args.output_md,
		quant_bits=args.quantization_bits,
	)

	print(f"Generated: {pr_chart}")
	print(f"Generated: {roc_chart}")
	print(f"Generated: {args.output_md}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
