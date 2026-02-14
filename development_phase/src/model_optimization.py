#!/usr/bin/env python3
"""
Stage 3: Model Training, Optimization, and Malware Split Analysis
=================================================================
Train Isolation Forest with grid search over features + hyperparameters.
Uses Cohen's d feature selection from cleaned data, then evaluates all configurations.
Appends a malware validation vs test clustering analysis for split quality.

Reads settings from a JSON config and writes reports/artifacts.

Usage:
    python model_optimization.py [--config PATH]
    # defaults to 'model_config.json' in the same directory as this script

Outputs:
    ../data/optimized/  (optimized datasets with best feature set)
    ../reports/optimization_results.csv
    ../reports/optimized_params.json
    ../reports/optimized_features.csv
    ../reports/roc_curve.svg
    ../reports/malware_val_test_embedding.png
    ../reports/malware_val_test_embedding.csv
    ../reports/malware_val_test_cluster_report.md
    ../reports/score_distribution.svg
    ../results  (model parameters & benchmarks for C++ embedding)
"""

import argparse
import hashlib
import json
import time
from itertools import product
from pathlib import Path
from typing import Literal, Union, Any, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler


def _compute_dataset_fingerprint(df: pd.DataFrame, name: str) -> str:
    """Compute SHA256 fingerprint of a dataset for audit trail."""
    df_hash = pd.util.hash_pandas_object(df, index=False)
    combined = hashlib.sha256()
    combined.update(np.asarray(df_hash.values).tobytes())
    fingerprint = combined.hexdigest()[:16]
    return fingerprint


def _count_cross_split_overlap(source_df: pd.DataFrame, target_df: pd.DataFrame) -> int:
    """Count exact duplicate rows in target_df that also exist in source_df."""
    if len(source_df) == 0 or len(target_df) == 0:
        return 0
    source_hashes = set(pd.util.hash_pandas_object(source_df, index=False).tolist())
    target_hashes = pd.util.hash_pandas_object(target_df, index=False)
    return int(target_hashes.isin(source_hashes).sum())


def _as_list(value):
    if isinstance(value, list):
        return value
    return [value]


def _load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(config_path, raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return config_path.parent / path


def _validate_config(cfg):
    required_top = ["data", "feature_selection", "model", "thresholding", "outputs"]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")
    if "fpr_threshold" not in cfg["thresholding"]:
        raise ValueError("thresholding.fpr_threshold is required")
    if "top_k_list" not in cfg["feature_selection"]:
        raise ValueError("feature_selection.top_k_list is required")
    threshold_cfg = cfg.get("thresholding", {})
    strategy = threshold_cfg.get("strategy", "fpr")
    if strategy not in {"fpr", "f1", "tpr", "youden", "model"}:
        raise ValueError("thresholding.strategy must be one of: fpr, f1, tpr, youden, model")


def _find_threshold_precise(scores_benign, max_fpr):
    sorted_scores = np.sort(scores_benign)
    n = len(sorted_scores)
    max_fp = int(np.floor(max_fpr * n))

    if max_fp == 0:
        threshold = sorted_scores[0] - 1e-10
        actual_fpr = 0.0
    else:
        threshold = sorted_scores[max_fp - 1]
        actual_fpr = (scores_benign < threshold).mean()

    return threshold, actual_fpr


def _select_threshold_with_malware(scores_benign, scores_malware, max_fpr, strategy, beta=1.0):
    thresholds = np.unique(np.concatenate([scores_benign, scores_malware]))
    thresholds.sort()

    fpr = (scores_benign[:, None] < thresholds[None, :]).mean(axis=0)
    tpr = (scores_malware[:, None] < thresholds[None, :]).mean(axis=0)

    if max_fpr is None:
        valid = np.ones_like(fpr, dtype=bool)
    else:
        valid = fpr <= max_fpr

    if not np.any(valid):
        threshold, fpr_val = _find_threshold_precise(scores_benign, max_fpr=max_fpr)
        tpr_val = (scores_malware < threshold).mean()
        return threshold, fpr_val, tpr_val, None

    if strategy == "tpr":
        metric = tpr
    elif strategy == "youden":
        metric = tpr - fpr
    else:
        n_benign = len(scores_benign)
        n_malware = len(scores_malware)
        tp = tpr * n_malware
        fp = fpr * n_benign
        precision = tp / (tp + fp + 1e-12)
        recall = tpr
        beta2 = beta * beta
        metric = (1.0 + beta2) * precision * recall / (beta2 * precision + recall + 1e-12)

    metric_masked = np.where(valid, metric, -np.inf)
    best_idx = int(np.argmax(metric_masked))
    return thresholds[best_idx], fpr[best_idx], tpr[best_idx], metric[best_idx]


def _select_top_features(X_train, X_malware, feature_names, top_k):
    """Rank features with Cohen's d between benign training and malware validation."""
    mean_benign = X_train.mean(axis=0)
    mean_malware = X_malware.mean(axis=0)
    std_benign = X_train.std(axis=0)
    std_malware = X_malware.std(axis=0)
    pooled_std = np.sqrt(0.5 * (std_benign * std_benign + std_malware * std_malware)) + 1e-10

    cohens_d = np.abs(mean_benign - mean_malware) / pooled_std
    top_k = min(int(top_k), int(len(feature_names)))
    top_indices = np.argsort(cohens_d)[-top_k:]

    return top_indices, feature_names[top_indices], cohens_d[top_indices]


def _infer_raw_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    raw_path = Path(str(path).replace("/cleaned/", "/raw/").replace("_clean.parquet", "_raw.parquet"))
    return raw_path if raw_path.exists() else None


def _plot_embedding(embedding: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"val": "#1f77b4", "test": "#ff7f0e"}
    plt.figure(figsize=(10, 8))
    for split in ["val", "test"]:
        mask = labels == split
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=18,
            alpha=0.7,
            c=colors[split],
            label=f"{split} (n={mask.sum()})",
            linewidths=0,
        )
    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _compute_imphash_overlap(df_val: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    if "imphash" not in df_val.columns or "imphash" not in df_test.columns:
        return {
            "proxy": "imphash",
            "available": False,
            "reason": "imphash column not present",
        }

    val_imphash = df_val["imphash"]
    test_imphash = df_test["imphash"]
    val_nonzero = val_imphash[val_imphash != 0]
    test_nonzero = test_imphash[test_imphash != 0]

    val_set = set(val_nonzero.astype(np.int64))
    test_set = set(test_nonzero.astype(np.int64))

    shared = val_set & test_set
    union = val_set | test_set

    jaccard = len(shared) / len(union) if union else 0.0
    val_shared_rate = (val_nonzero.isin(shared)).mean() if len(val_nonzero) else 0.0
    test_shared_rate = (test_nonzero.isin(shared)).mean() if len(test_nonzero) else 0.0

    return {
        "proxy": "imphash",
        "available": True,
        "val_samples": int(len(val_imphash)),
        "test_samples": int(len(test_imphash)),
        "val_nonzero": int(len(val_nonzero)),
        "test_nonzero": int(len(test_nonzero)),
        "val_unique": int(len(val_set)),
        "test_unique": int(len(test_set)),
        "shared_unique": int(len(shared)),
        "jaccard_unique": float(jaccard),
        "val_shared_rate": float(val_shared_rate),
        "test_shared_rate": float(test_shared_rate),
    }


def _run_malware_split_analysis(
    optimized_data_dir: Path,
    report_dir: Path,
    val_data_path: Path | None,
    test_data_path: Path | None,
) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError:
        print("Skipping malware split analysis: matplotlib is not installed.")
        print("Install with: pip install matplotlib umap-learn")
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    import umap

    val_opt_path = optimized_data_dir / "malware_val_optimized.parquet"
    test_opt_path = optimized_data_dir / "malware_test_optimized.parquet"

    if not val_opt_path.exists() or not test_opt_path.exists():
        raise FileNotFoundError("Optimized malware datasets not found. Run optimization first.")

    df_val_opt = pd.read_parquet(val_opt_path)
    df_test_opt = pd.read_parquet(test_opt_path)

    labels = np.array(["val"] * len(df_val_opt) + ["test"] * len(df_test_opt))
    X = np.vstack([df_val_opt.values, df_test_opt.values])

    pca_components = min(20, X.shape[1])
    X_pca = PCA(n_components=pca_components, random_state=42).fit_transform(X)

    umap_model = umap.UMAP(
        n_neighbors=10,
        min_dist=0.1,
        metric="euclidean",
        n_epochs=200,
        low_memory=True,
        random_state=42,
        n_components=2,
    )
    umap_embedding = np.asarray(umap_model.fit_transform(X_pca))

    umap_15_model = umap.UMAP(
        n_neighbors=10,
        min_dist=0.1,
        metric="euclidean",
        n_epochs=200,
        low_memory=True,
        random_state=42,
        n_components=15,
    )
    umap_15_embedding = np.asarray(umap_15_model.fit_transform(X_pca))

    sil_umap_2d = float(silhouette_score(umap_embedding, labels))
    sil_umap_15d = float(silhouette_score(umap_15_embedding, labels))

    combined_png = report_dir / "malware_val_test_embedding.png"
    colors = {"val": "#1f77b4", "test": "#ff7f0e"}
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    axes_flat = axes.ravel()
    for idx, ax in enumerate(axes_flat):
        x_idx = idx
        y_idx = (idx + 1) % 15
        for split in ["val", "test"]:
            mask = labels == split
            ax.scatter(
                umap_15_embedding[mask, x_idx],
                umap_15_embedding[mask, y_idx],
                s=18,
                alpha=0.7,
                c=colors[split],
                label=f"{split} (n={mask.sum()})",
                linewidths=0,
            )
        ax.set_title(f"UMAP d{x_idx + 1} vs d{y_idx + 1}")
        ax.set_xlabel(f"d{x_idx + 1}")
        ax.set_ylabel(f"d{y_idx + 1}")
    axes_flat[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(combined_png, dpi=200)
    plt.close(fig)

    embedding_csv = report_dir / "malware_val_test_embedding.csv"
    embed_df = pd.DataFrame(
        {
            "split": labels,
            "umap2_1": umap_embedding[:, 0],
            "umap2_2": umap_embedding[:, 1],
            **{f"umap15_{i + 1}": umap_15_embedding[:, i] for i in range(15)},
        }
    )
    embed_df.to_csv(embedding_csv, index=False)

    proxy_metrics = None
    val_raw_path = _infer_raw_path(val_data_path)
    test_raw_path = _infer_raw_path(test_data_path)
    if val_raw_path and test_raw_path:
        df_val_raw = pd.read_parquet(val_raw_path)
        df_test_raw = pd.read_parquet(test_raw_path)
        proxy_metrics = _compute_imphash_overlap(df_val_raw, df_test_raw)

    report_path = report_dir / "malware_val_test_cluster_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Malware Validation vs Test: Cluster Analysis\n\n")
        f.write(f"Samples (val/test): {len(df_val_opt)} / {len(df_test_opt)}\n\n")
        f.write("## Embedding Separation\n\n")
        f.write(f"- PCA pre-reduction: {pca_components} dims\n")
        f.write(f"- Silhouette (UMAP 2D): {sil_umap_2d:.4f}\n")
        f.write(f"- Silhouette (UMAP 15D): {sil_umap_15d:.4f}\n")
        f.write("- Visualization: UMAP 15D projected as a 5x3 subplot frame\n\n")
        f.write("## Family Proxy (Imphash)\n\n")
        if proxy_metrics and proxy_metrics.get("available"):
            f.write(f"- Nonzero imphash (val/test): {proxy_metrics['val_nonzero']} / {proxy_metrics['test_nonzero']}\n")
            f.write(f"- Unique imphash (val/test): {proxy_metrics['val_unique']} / {proxy_metrics['test_unique']}\n")
            f.write(f"- Shared unique imphash: {proxy_metrics['shared_unique']}\n")
            f.write(f"- Jaccard (unique): {proxy_metrics['jaccard_unique']:.4f}\n")
            f.write(f"- Shared-rate (val/test): {proxy_metrics['val_shared_rate']:.4f} / {proxy_metrics['test_shared_rate']:.4f}\n")
        else:
            reason = proxy_metrics.get("reason") if proxy_metrics else "raw datasets not available"
            f.write(f"- Imphash not available ({reason})\n")

    print("Cluster analysis complete")
    print(f"  Combined plot: {combined_png}")
    print(f"  Embeddings: {embedding_csv}")
    print(f"  Report: {report_path}")


# ═══════════════════════════════════════════════════════════════
# SVG VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def _write_roc_svg(fpr, tpr, auc_value, out_path, fpr_threshold):
    width, height = 720, 420
    margin = 60
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    x_min = 0.0001
    x_max = 1.0
    log_min = np.log10(x_min)
    log_max = np.log10(x_max)

    def get_px(x):
        if x <= x_min:
            return margin
        norm_x = (np.log10(x) - log_min) / (log_max - log_min)
        return margin + min(norm_x, 1.0) * plot_w

    points = []
    for x, y in zip(fpr, tpr):
        px = get_px(x)
        py = height - margin - y * plot_h
        points.append(f"{px:.1f},{py:.1f}")

    tpr_at_threshold = np.interp(fpr_threshold, fpr, tpr)
    px_thresh = get_px(fpr_threshold)
    py_thresh = height - margin - tpr_at_threshold * plot_h

    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#f7f7f2'/>",
        f"<text x='{margin}' y='30' font-family='Arial' font-size='16' fill='#333'>ROC Curve (Best Model)</text>",
        f"<text x='{margin}' y='50' font-family='Arial' font-size='12' fill='#333'>AUC = {auc_value:.4f}</text>",
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='#333' stroke-width='1'/>",
        f"<line x1='{margin}' y1='{height - margin}' x2='{margin}' y2='{margin}' stroke='#333' stroke-width='1'/>",
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{margin}' stroke='#999' stroke-dasharray='4,4'/>",
        f"<line x1='{px_thresh:.1f}' y1='{height - margin}' x2='{px_thresh:.1f}' y2='{margin}' stroke='#e53e3e' stroke-width='1.5' stroke-dasharray='5,5'/>",
        f"<circle cx='{px_thresh:.1f}' cy='{py_thresh:.1f}' r='4' fill='#e53e3e'/>",
        f"<text x='{px_thresh + 5:.1f}' y='{py_thresh - 10:.1f}' font-family='Arial' font-size='11' fill='#e53e3e' font-weight='bold'>[{tpr_at_threshold:.4f}, {fpr_threshold:.4f}]</text>",
    ]

    tick_values = [0.0001, 0.001, 0.01, 0.1, 1.0]
    tick_labels = ["10^{-4}", "10^{-3}", "10^{-2}", "10^{-1}", "10^{0}"]
    for tick, label in zip(tick_values, tick_labels):
        px = get_px(tick)
        svg.append(f"<line x1='{px:.1f}' y1='{height - margin}' x2='{px:.1f}' y2='{height - margin + 5}' stroke='#333' stroke-width='1' />")
        svg.append(f"<text x='{px:.1f}' y='{height - margin + 18}' font-family='Arial' font-size='10' text-anchor='middle' fill='#333'>{label}</text>")

    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        py = height - margin - tick * plot_h
        svg.append(f"<line x1='{margin}' y1='{py:.1f}' x2='{margin - 5}' y2='{py:.1f}' stroke='#333' stroke-width='1' />")
        svg.append(f"<text x='{margin - 8}' y='{py + 4:.1f}' font-family='Arial' font-size='10' text-anchor='end' fill='#333'>{tick:.1f}</text>")

    svg.extend([
        f"<polyline fill='none' stroke='#2b6cb0' stroke-width='2' points='{' '.join(points)}'/>",
        f"<text x='{width - margin}' y='{height - margin + 30}' font-family='Arial' font-size='12' text-anchor='end' fill='#333'>False Positive Rate</text>",
        f"<text x='{margin - 40}' y='{height // 2}' font-family='Arial' font-size='12' fill='#333' transform='rotate(-90 {margin - 40},{height // 2})' text-anchor='middle'>True Positive Rate</text>",
        "</svg>",
    ])
    out_path.write_text("\n".join(svg), encoding="utf-8")


def _write_pr_svg(precision, recall, ap_value, out_path):
    width, height = 720, 420
    margin = 60
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    points = []
    for r, p in zip(recall, precision):
        px = margin + r * plot_w
        py = height - margin - p * plot_h
        points.append(f"{px:.1f},{py:.1f}")

    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#f7f7f2'/>",
        f"<text x='{margin}' y='30' font-family='Arial' font-size='16' fill='#333'>Precision-Recall Curve (Best Model)</text>",
        f"<text x='{margin}' y='50' font-family='Arial' font-size='12' fill='#333'>AP = {ap_value:.4f}</text>",
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='#333' stroke-width='1'/>",
        f"<line x1='{margin}' y1='{height - margin}' x2='{margin}' y2='{margin}' stroke='#333' stroke-width='1'/>",
    ]

    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        px = margin + tick * plot_w
        py = height - margin - tick * plot_h
        svg.append(f"<line x1='{px:.1f}' y1='{height - margin}' x2='{px:.1f}' y2='{height - margin + 5}' stroke='#333' stroke-width='1' />")
        svg.append(f"<text x='{px:.1f}' y='{height - margin + 18}' font-family='Arial' font-size='10' text-anchor='middle' fill='#333'>{tick:.1f}</text>")
        svg.append(f"<line x1='{margin}' y1='{py:.1f}' x2='{margin - 5}' y2='{py:.1f}' stroke='#333' stroke-width='1' />")
        svg.append(f"<text x='{margin - 8}' y='{py + 4:.1f}' font-family='Arial' font-size='10' text-anchor='end' fill='#333'>{tick:.1f}</text>")

    svg.extend([
        f"<polyline fill='none' stroke='#2f855a' stroke-width='2' points='{' '.join(points)}'/>",
        f"<text x='{width - margin}' y='{height - margin + 30}' font-family='Arial' font-size='12' text-anchor='end' fill='#333'>Recall</text>",
        f"<text x='{margin - 40}' y='{height // 2}' font-family='Arial' font-size='12' fill='#333' transform='rotate(-90 {margin - 40},{height // 2})' text-anchor='middle'>Precision</text>",
        "</svg>",
    ])

    out_path.write_text("\n".join(svg), encoding="utf-8")


def _write_score_hist_svg(scores_benign, scores_malware, out_path):
    width, height = 720, 420
    margin = 60
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    all_scores = np.concatenate([scores_benign, scores_malware])
    bins = np.linspace(all_scores.min(), all_scores.max(), 30)
    ben_counts, _ = np.histogram(scores_benign, bins=bins)
    mal_counts, _ = np.histogram(scores_malware, bins=bins)

    max_count = max(ben_counts.max(), mal_counts.max())
    bin_w = plot_w / (len(bins) - 1)

    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#f7f7f2'/>",
        f"<text x='{margin}' y='30' font-family='Arial' font-size='16' fill='#333'>Score Distribution (Decision Function)</text>",
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='#333' stroke-width='1'/>",
        f"<line x1='{margin}' y1='{height - margin}' x2='{margin}' y2='{margin}' stroke='#333' stroke-width='1'/>",
    ]

    for i in [0, 7, 14, 21, 28]:
        px = margin + i * bin_w
        svg.append(f"<line x1='{px:.1f}' y1='{height - margin}' x2='{px:.1f}' y2='{height - margin + 5}' stroke='#333' stroke-width='1' />")
        svg.append(f"<text x='{px:.1f}' y='{height - margin + 18}' font-family='Arial' font-size='10' text-anchor='middle' fill='#333'>{bins[i]:.2f}</text>")

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        count_tick = frac * max_count
        py = height - margin - (count_tick / max_count) * plot_h if max_count > 0 else height - margin
        svg.append(f"<line x1='{margin}' y1='{py:.1f}' x2='{margin - 5}' y2='{py:.1f}' stroke='#333' stroke-width='1' />")
        svg.append(f"<text x='{margin - 8}' y='{py + 4:.1f}' font-family='Arial' font-size='10' text-anchor='end' fill='#333'>{int(count_tick)}</text>")

    for i in range(len(bins) - 1):
        x = margin + i * bin_w
        ben_h = (ben_counts[i] / max_count) * plot_h if max_count > 0 else 0
        mal_h = (mal_counts[i] / max_count) * plot_h if max_count > 0 else 0
        svg.append(
            f"<rect x='{x:.1f}' y='{height - margin - ben_h:.1f}' width='{bin_w - 2:.1f}' height='{ben_h:.1f}' fill='#2b6cb0' fill-opacity='0.6'/>"
        )
        svg.append(
            f"<rect x='{x:.1f}' y='{height - margin - mal_h:.1f}' width='{bin_w - 2:.1f}' height='{mal_h:.1f}' fill='#d69e2e' fill-opacity='0.6'/>"
        )

    svg.extend([
        f"<text x='{width - margin}' y='{height - margin + 30}' font-family='Arial' font-size='12' text-anchor='end' fill='#333'>Decision Score</text>",
        f"<text x='{margin - 35}' y='{margin}' font-family='Arial' font-size='12' fill='#333' transform='rotate(-90 {margin - 35},{margin})'>Count</text>",
        f"<rect x='{width - margin - 150}' y='{margin}' width='12' height='12' fill='#2b6cb0' fill-opacity='0.6'/>",
        f"<text x='{width - margin - 130}' y='{margin + 10}' font-family='Arial' font-size='12' fill='#333'>Benign</text>",
        f"<rect x='{width - margin - 150}' y='{margin + 20}' width='12' height='12' fill='#d69e2e' fill-opacity='0.6'/>",
        f"<text x='{width - margin - 130}' y='{margin + 30}' font-family='Arial' font-size='12' fill='#333'>Malware</text>",
        "</svg>",
    ])
    out_path.write_text("\n".join(svg), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════
# RESULT ARTIFACT EXPORT
# ═══════════════════════════════════════════════════════════════

def export_results(model, scaler, feature_names, threshold, fpr_threshold,
                   val_fpr, val_tpr, test_fpr, test_tpr, auc_value, best_params,
                   group_mapping, results_dir, threshold_strategy="fpr"):
    """Export model parameters and benchmarks for C++ embedding."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # Scaler parameters (JSON for C++)
    scaler_export = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "n_features": int(scaler.n_features_in_),
    }
    with open(results_dir / "scaler_params.json", "w") as f:
        json.dump(scaler_export, f, indent=2)

    # Threshold
    threshold_export = {
        "threshold": float(threshold),
        "fpr_target": float(fpr_threshold),
        "val_fpr_achieved": float(val_fpr),
        "val_tpr": float(val_tpr),
        "test_fpr": float(test_fpr),
        "test_tpr": float(test_tpr),
        "auc": float(auc_value),
        "threshold_strategy": str(threshold_strategy),
    }
    with open(results_dir / "threshold.json", "w") as f:
        json.dump(threshold_export, f, indent=2)

    # Feature names
    with open(results_dir / "feature_names.json", "w") as f:
        json.dump(list(feature_names), f, indent=2)

    # Tree structures
    trees_export = []
    for i, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        trees_export.append({
            "tree_index": i,
            "n_nodes": int(tree.node_count),
            "max_depth": int(tree.max_depth),
            "feature": tree.feature.tolist(),
            "threshold": tree.threshold.tolist(),
            "children_left": tree.children_left.tolist(),
            "children_right": tree.children_right.tolist(),
            "n_node_samples": tree.n_node_samples.tolist(),
            "features_used": sorted(set(tree.feature[tree.feature >= 0].tolist())),
        })

    max_samples_val = best_params.get("max_samples", "auto")
    with open(results_dir / "trees.json", "w") as f:
        json.dump({
            "n_trees": len(trees_export),
            "n_features": len(feature_names),
            "max_samples": max_samples_val,
            "offset": float(model.offset_),
            "trees": trees_export,
        }, f)

    # Feature group mapping
    with open(results_dir / "feature_group_mapping.json", "w") as f:
        json.dump(group_mapping, f, indent=2)

    # Benchmarks summary
    benchmarks = {
        "model_type": "IsolationForest",
        "hyperparameters": {k: (v if not isinstance(v, np.integer) else int(v)) for k, v in best_params.items()},
        "n_features": len(feature_names),
        "n_trees": len(trees_export),
        "threshold": float(threshold),
        "offset": float(model.offset_),
        "metrics": {
            "roc_auc": float(auc_value),
            "fpr_benign_test": float(test_fpr),
            "tpr_malware_test": float(test_tpr),
            "fpr_benign_val": float(val_fpr),
            "tpr_malware_val": float(val_tpr),
        },
        "note": "Model not saved - C++ version will be built from these parameters"
    }
    with open(results_dir / "benchmarks.json", "w") as f:
        json.dump(benchmarks, f, indent=2)

    print(f"\n  Results exported to: {results_dir}")
    for p in sorted(results_dir.glob("*")):
        sz = p.stat().st_size
        unit = "KB" if sz > 1024 else "B"
        val = sz / 1024 if sz > 1024 else sz
        print(f"    {p.name:<35} {val:>8.1f} {unit}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    script_dir = Path(__file__).resolve().parent
    default_config_path = script_dir / "model_config.json"

    parser = argparse.ArgumentParser(description="Stage 3: Model Training & Optimization")
    parser.add_argument("--config", type=str, default=None,
                        help=f"Path to JSON config file (defaults to {default_config_path})")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. Provide --config to specify another config file.")

    cfg = _load_config(config_path)
    _validate_config(cfg)

    config_path = config_path.resolve()

    data_cfg = cfg["data"]
    train_path = _resolve_path(config_path, data_cfg["train_benign_path"])
    val_path = _resolve_path(config_path, data_cfg["val_benign_path"])
    test_b_path = _resolve_path(config_path, data_cfg["test_benign_path"])
    test_m_path = _resolve_path(config_path, data_cfg["test_malware_path"])
    val_m_raw = data_cfg.get("val_malware_path")
    val_m_path = _resolve_path(config_path, val_m_raw) if val_m_raw else None

    if val_path.resolve() == test_b_path.resolve():
        raise ValueError("val_benign_path and test_benign_path must be different (sealed test requirement)")
    if val_m_path and val_m_path.resolve() == test_m_path.resolve():
        raise ValueError("val_malware_path and test_malware_path must be different (sealed test requirement)")

    feature_cfg = cfg["feature_selection"]
    top_k_list = [int(x) for x in _as_list(feature_cfg["top_k_list"])]

    model_cfg = cfg["model"]
    n_estimators_list = [int(x) for x in _as_list(model_cfg.get("n_estimators", 300))]

    def _parse_max_samples(x) -> Union[int, float, Literal["auto"]]:
        if isinstance(x, str) and x.lower() == "auto":
            return "auto"
        try:
            f = float(x)
            if isinstance(x, float) or (isinstance(x, str) and "." in x):
                return f
            return int(f)
        except (ValueError, TypeError):
            raise ValueError(f"max_samples must be float, int, or 'auto', got {type(x).__name__}: {x}")

    max_samples_list = [_parse_max_samples(x) for x in _as_list(model_cfg.get("max_samples", 0.5))]
    contamination_list = [float(x) for x in _as_list(model_cfg.get("contamination", 0.2))]
    max_features_list = [float(x) for x in _as_list(model_cfg.get("max_features", 1.0))]
    bootstrap_list = [bool(x) for x in _as_list(model_cfg.get("bootstrap", False))]
    random_state = int(model_cfg.get("random_state", 42))
    n_jobs = int(model_cfg.get("n_jobs", -1))

    threshold_cfg = cfg["thresholding"]
    fpr_threshold = float(threshold_cfg["fpr_threshold"])
    threshold_strategy = threshold_cfg.get("strategy", "fpr")
    f_beta = float(threshold_cfg.get("f_beta", 1.0))
    expose_test_during_search = bool(threshold_cfg.get("expose_test_during_search", False))
    
    # CRITICAL: Guard against test data leakage during grid search
    if expose_test_during_search:
        raise ValueError(
            "CONFIGURATION ERROR: 'expose_test_during_search' is set to true, which violates "
            "the sealed test set principle. Test data must NOT be used during hyperparameter "
            "optimization as this compromises model evaluation integrity. Set to false for "
            "valid model training. This parameter exists for debugging only and must never "
            "be enabled in production or for any results reporting."
        )
    
    val_fpr_target = threshold_cfg.get("val_fpr_target")
    if val_fpr_target is None:
        val_fpr_delta = float(threshold_cfg.get("val_fpr_delta", 0.005))
        val_fpr_target = max(fpr_threshold - val_fpr_delta, 0.0)
    else:
        val_fpr_target = float(val_fpr_target)

    outputs_cfg = cfg["outputs"]
    report_dir = _resolve_path(config_path, outputs_cfg.get("report_dir", "../reports"))
    results_dir = _resolve_path(config_path, outputs_cfg.get("results_dir", "../results/final"))
    optimized_data_dir = _resolve_path(config_path, outputs_cfg.get("optimized_data_dir", "../data/optimized"))
    results_csv = _resolve_path(config_path, outputs_cfg.get("results_csv", "../reports/optimization_results.csv"))
    params_json = _resolve_path(config_path, outputs_cfg.get("optimized_params_json", "../reports/optimized_params.json"))
    features_csv = _resolve_path(config_path, outputs_cfg.get("optimized_features_csv", "../reports/optimized_features.csv"))
    roc_svg = _resolve_path(config_path, outputs_cfg.get("roc_curve_svg", "../reports/roc_curve.svg"))
    pr_svg = _resolve_path(config_path, outputs_cfg.get("pr_curve_svg", "../reports/pr_curve.svg"))
    score_svg = _resolve_path(config_path, outputs_cfg.get("score_distribution_svg", "../reports/score_distribution.svg"))
    val_score_svg_raw = outputs_cfg.get("score_distribution_val_svg")
    val_score_svg = _resolve_path(config_path, val_score_svg_raw) if val_score_svg_raw else None

    report_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    optimized_data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 3: MODEL TRAINING & OPTIMIZATION")
    print("=" * 70)

    print("\nLoading training and validation data...")
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)

    if val_m_path:
        if not val_m_path.exists():
            raise FileNotFoundError(f"val_malware_path not found: {val_m_path}")
        df_val_m = pd.read_parquet(val_m_path)
        print("  Using malware validation split for feature ranking and selection.")
    else:
        raise ValueError(
            "data.val_malware_path is required to avoid test leakage during feature selection."
        )

    print(f"  Training samples: {len(df_train)}")
    print(f"  Validation benign samples: {len(df_val)}")
    print(f"  Validation malware samples: {len(df_val_m)}")
    print("  Test data will be loaded after model selection (sealed until final evaluation)")
    
    # Dataset fingerprinting for audit trail
    print("\n--- Dataset Fingerprinting (Audit Trail) ---")
    train_fp = _compute_dataset_fingerprint(df_train, "train")
    val_fp = _compute_dataset_fingerprint(df_val, "val_benign")
    val_m_fp = _compute_dataset_fingerprint(df_val_m, "val_malware")
    print(f"  Training (benign):      {train_fp}")
    print(f"  Validation (benign):    {val_fp}")
    print(f"  Validation (malware):   {val_m_fp}")
    
    print("\n--- Split Independence Focus ---")
    print("  Primary independence check in Stage 3: malware_val vs malware_test")

    print("\nPreparing features...")
    selected_cols = np.array(df_train.columns.tolist())

    X_train_var = df_train[selected_cols].values.astype(np.float32)
    X_val_var = df_val[selected_cols].values.astype(np.float32)
    X_val_m_var = df_val_m[selected_cols].values.astype(np.float32)
    
    print(f"  Features: {len(selected_cols)}")
    print("  Feature ranking source: benign_train vs malware_val (Cohen's d)")
    print(f"  Validation malware used for ranking/thresholding: {len(X_val_m_var)} samples")

    # Load group mapping for artifact export
    schema_dir = _resolve_path(config_path, "../schemas")
    group_mapping_path = schema_dir / "feature_group_mapping.json"
    group_mapping = {}
    if group_mapping_path.exists():
        with open(group_mapping_path) as fh:
            group_mapping = json.load(fh)

    all_results = []
    total_configs = (
        len(top_k_list)
        * len(n_estimators_list)
        * len(max_samples_list)
        * len(contamination_list)
        * len(max_features_list)
        * len(bootstrap_list)
    )

    print(f"\nRunning {total_configs} experiments...")
    print("=" * 70)

    config_num = 0
    for n_feat in top_k_list:
        top_indices, top_names, top_d = _select_top_features(
            X_train_var, X_val_m_var, selected_cols, top_k=n_feat
        )

        X_train_sel = X_train_var[:, top_indices]
        X_val_sel = X_val_var[:, top_indices]
        X_val_m_thr_sel = X_val_m_var[:, top_indices]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_val_scaled = scaler.transform(X_val_sel)
        X_val_m_thr_scaled = scaler.transform(X_val_m_thr_sel)

        for n_trees, max_samp, contam, max_feat, boot in product(
            n_estimators_list, max_samples_list, contamination_list, max_features_list, bootstrap_list
        ):
            config_num += 1
            t0 = time.time()

            model = IsolationForest(
                n_estimators=n_trees,
                max_samples=cast(Any, max_samp),
                max_features=max_feat,
                bootstrap=boot,
                contamination=contam,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            model.fit(X_train_scaled)

            scores_val = model.decision_function(X_val_scaled)
            scores_vm = model.decision_function(X_val_m_thr_scaled)

            if threshold_strategy == "fpr":
                threshold, fpr_val = _find_threshold_precise(scores_val, max_fpr=val_fpr_target)
                tpr_val = (scores_vm < threshold).mean()
                metric_val = tpr_val
            elif threshold_strategy == "model":
                threshold = 0.0
                fpr_val = (scores_val < threshold).mean()
                tpr_val = (scores_vm < threshold).mean()
                metric_val = tpr_val
            else:
                threshold, fpr_val, tpr_val, metric_val = _select_threshold_with_malware(
                    scores_val,
                    scores_vm,
                    max_fpr=val_fpr_target,
                    strategy=threshold_strategy,
                    beta=f_beta,
                )
                if metric_val is None:
                    metric_val = tpr_val

            y_true_val = np.concatenate([np.zeros(len(scores_val)), np.ones(len(scores_vm))])
            y_scores_val = np.concatenate([-scores_val, -scores_vm])
            auc_val = roc_auc_score(y_true_val, y_scores_val)

            # Test data is sealed - not loaded or evaluated during grid search
            fpr_test = np.nan
            tpr_test = np.nan
            auc = np.nan

            elapsed = time.time() - t0

            all_results.append({
                "n_features": n_feat,
                "n_estimators": n_trees,
                "max_samples": max_samp,
                "contamination": contam,
                "max_features": max_feat,
                "bootstrap": boot,
                "threshold": threshold,
                "fpr_val": fpr_val,
                "fpr_test": fpr_test,
                "tpr_val": tpr_val,
                "tpr_test": tpr_test,
                "metric_val": metric_val,
                "auc_val": auc_val,
                "auc": auc,
                "time_s": elapsed,
            })

            status = "OK" if fpr_val <= fpr_threshold else "HIGH"
            if config_num % 10 == 0 or status == "OK":
                samp_str = f"{max_samp}" if isinstance(max_samp, str) else f"{float(max_samp):.1f}"
                cont_str = f"{contam:.3f}"
                print(
                    f"[{config_num:3d}/{total_configs}] feat={n_feat:3d}, trees={n_trees:3d}, "
                    f"samp={samp_str}, cont={cont_str}, maxf={max_feat:.1f}, boot={int(boot)}: "
                    f"TPRv={tpr_val * 100:5.2f}%, FPRv={fpr_val * 100:5.2f}% "
                    f"[{elapsed:.1f}s] {status}"
                )

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(results_csv, index=False)

    valid_results = df_results[df_results["fpr_val"] <= fpr_threshold]
    if len(valid_results) == 0:
        raise RuntimeError("No configuration met the configured FPR threshold")

    valid_results = valid_results.copy()
    valid_results["fpr_gap"] = (valid_results["fpr_val"] - val_fpr_target).abs()

    best = valid_results.sort_values(
        by=["tpr_val", "auc_val", "fpr_gap", "time_s"],
        ascending=[False, False, True, True],
    ).iloc[0]

    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    print(f"  Features:        {int(best['n_features'].item())}")
    print(f"  n_estimators:    {int(best['n_estimators'].item())}")
    print(f"  max_samples:     {best['max_samples'].item()}")
    print(f"  max_features:    {best['max_features'].item()}")
    print(f"  bootstrap:       {bool(best['bootstrap'].item())}")
    print(f"  contamination:   {best['contamination'].item()}")
    print(f"  threshold_strat: {threshold_strategy}")
    print(f"  fpr_threshold:   {fpr_threshold}")
    print(f"  val_fpr_target:  {val_fpr_target}")
    print(f"  f_beta:          {f_beta}")
    print(f"  random_state:    {random_state}")

    print("\n  Generating final holdout evaluation report...")

    best_n_features = int(best["n_features"].item())
    best_n_estimators = int(best["n_estimators"].item())
    
    # Cast/verify max_samples for Pylance and handle 'auto' comparison
    best_max_raw = best["max_samples"].item()
    if isinstance(best_max_raw, str) and best_max_raw == "auto":
        best_max_samples = "auto"
    else:
        best_max_samples = float(best_max_raw)
        if best_max_samples > 1.0:
            best_max_samples = int(best_max_samples)

    top_indices, top_names, top_d = _select_top_features(
        X_train_var, X_val_m_var, selected_cols, top_k=best_n_features
    )

    best_features = pd.DataFrame(
        {"feature": top_names, "cohens_d": top_d}
    ).sort_values("cohens_d", ascending=False)
    best_features.to_csv(features_csv, index=False)

    params_payload = {
        "n_features": best_n_features,
        "n_estimators": best_n_estimators,
        "max_samples": best_max_samples,
        "max_features": float(best["max_features"].item()),
        "bootstrap": bool(best["bootstrap"].item()),
        "contamination": float(best["contamination"].item()),
        "threshold_strategy": threshold_strategy,
        "f_beta": float(f_beta),
        "fpr_threshold": float(fpr_threshold),
        "val_fpr_target": float(val_fpr_target),
    }
    params_json.write_text(json.dumps(params_payload, indent=2), encoding="utf-8")

    # ── Load test data NOW (after model selection) ──
    print(f"\n{'=' * 70}")
    print("LOADING TEST DATA (Sealed Until Now)")
    print(f"{'=' * 70}")
    df_test_b = pd.read_parquet(test_b_path)
    df_test_m = pd.read_parquet(test_m_path)
    overlap_malware_val_test = _count_cross_split_overlap(df_val_m, df_test_m)

    print(f"  Test benign samples:  {len(df_test_b)}")
    print(f"  Test malware samples: {len(df_test_m)}")
    
    # Test data fingerprinting
    test_b_fp = _compute_dataset_fingerprint(df_test_b, "test_benign")
    test_m_fp = _compute_dataset_fingerprint(df_test_m, "test_malware")
    print(f"  Test (benign) fingerprint:  {test_b_fp}")
    print(f"  Test (malware) fingerprint: {test_m_fp}")

    print("\n--- Malware Val/Test Independence Verification ---")
    if overlap_malware_val_test == 0:
        print("  ✓ malware_val and malware_test are independent (exact overlap = 0)")
    else:
        print(
            "  ⚠ malware_val overlaps malware_test by "
            f"{overlap_malware_val_test} samples"
        )

    # ── Retrain best model and generate visualizations ──
    X_test_b_var = df_test_b[selected_cols].values.astype(np.float32)
    X_test_m_var = df_test_m[selected_cols].values.astype(np.float32)
    
    X_train = X_train_var[:, top_indices]
    X_val = X_val_var[:, top_indices]
    X_test_b = X_test_b_var[:, top_indices]
    X_test_m = X_test_m_var[:, top_indices]
    X_val_m_thr = X_val_m_var[:, top_indices]
    X_val_m_full = X_val_m_var[:, top_indices]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_b_scaled = scaler.transform(X_test_b)
    X_test_m_scaled = scaler.transform(X_test_m)
    X_val_m_thr_scaled = scaler.transform(X_val_m_thr)
    X_val_m_full_scaled = scaler.transform(X_val_m_full)

    model = IsolationForest(
        n_estimators=best_n_estimators,
        max_samples=cast(Any, best_max_samples),
        max_features=float(best["max_features"].item()),
        bootstrap=bool(best["bootstrap"].item()),
        contamination=float(best["contamination"].item()),
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train_scaled)

    # Threshold on validation
    scores_val_final = model.decision_function(X_val_scaled)
    scores_vm_final = model.decision_function(X_val_m_thr_scaled)
    if threshold_strategy == "fpr":
        final_threshold, final_val_fpr = _find_threshold_precise(scores_val_final, max_fpr=val_fpr_target)
        final_val_tpr = (scores_vm_final < final_threshold).mean()
    elif threshold_strategy == "model":
        final_threshold = 0.0
        final_val_fpr = (scores_val_final < final_threshold).mean()
        final_val_tpr = (scores_vm_final < final_threshold).mean()
    else:
        final_threshold, final_val_fpr, final_val_tpr, _ = _select_threshold_with_malware(
            scores_val_final,
            scores_vm_final,
            max_fpr=val_fpr_target,
            strategy=threshold_strategy,
            beta=f_beta,
        )

    scores_b = model.decision_function(X_test_b_scaled)
    scores_m = model.decision_function(X_test_m_scaled)

    final_fpr = (scores_b < final_threshold).mean()
    final_tpr = (scores_m < final_threshold).mean()

    y_true = np.concatenate([np.zeros(len(scores_b)), np.ones(len(scores_m))])
    y_scores = np.concatenate([-scores_b, -scores_m])

    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_scores)
    auc_value = roc_auc_score(y_true, y_scores)
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_scores)
    ap_value = average_precision_score(y_true, y_scores)

    print("\n" + "=" * 70)
    print("FINAL HOLDOUT TEST RESULTS")
    print("=" * 70)
    print(f"  Threshold: {float(final_threshold):.6f}")
    print(f"  Test TPR:  {final_tpr * 100:.2f}%")
    print(f"  Test FPR:  {final_fpr * 100:.2f}%")
    print(f"  Test ROC-AUC: {auc_value:.4f}")
    print(f"  Test PR-AUC:  {ap_value:.4f}")

    _write_roc_svg(fpr_arr, tpr_arr, auc_value, roc_svg, fpr_threshold)
    _write_pr_svg(precision_arr, recall_arr, ap_value, pr_svg)
    _write_score_hist_svg(scores_b, scores_m, score_svg)
    if val_score_svg:
        scores_vm_full = model.decision_function(X_val_m_full_scaled)
        _write_score_hist_svg(scores_val_final, scores_vm_full, val_score_svg)

    # ── Export model artifacts ──
    best_params = {
        "n_estimators": best_n_estimators,
        "max_samples": best_max_samples,
        "max_features": float(best["max_features"].item()),
        "bootstrap": bool(best["bootstrap"].item()),
        "contamination": float(best["contamination"].item()),
        "random_state": random_state,
    }

    # Build group mapping for selected features
    selected_group_mapping = {name: group_mapping.get(name, "unknown") for name in top_names}

    export_results(
        model=model,
        scaler=scaler,
        feature_names=top_names,
        threshold=final_threshold,
        fpr_threshold=fpr_threshold,
        val_fpr=final_val_fpr,
        val_tpr=final_val_tpr,
        test_fpr=final_fpr,
        test_tpr=final_tpr,
        auc_value=auc_value,
        best_params=best_params,
        group_mapping=selected_group_mapping,
        results_dir=results_dir,
        threshold_strategy=threshold_strategy,
    )

    # ── Save optimized datasets with best feature set ──
    print(f"\n{'=' * 70}")
    print(f"SAVING OPTIMIZED DATASETS ({len(top_names)} features)")
    print(f"{'=' * 70}")

    # Reconstruct scaled datasets with the best features from the already-processed data
    # We already have X_train, X_val, X_test_b, X_test_m with best features and scaling
    dataset_map = {
        "benign_train": X_train_scaled,
        "benign_val": X_val_scaled,
        "benign_test": X_test_b_scaled,
        "malware_test": X_test_m_scaled,
    }

    if val_m_path:
        dataset_map["malware_val"] = X_val_m_full_scaled
    
    for name, data in dataset_map.items():
        df_optimized = pd.DataFrame(data, columns=top_names)
        out_path = optimized_data_dir / f"{name}_optimized.parquet"
        df_optimized.to_parquet(out_path, engine="pyarrow", compression="snappy")
        csv_path = optimized_data_dir / f"{name}_optimized.csv"
        df_optimized.to_csv(csv_path, index=False)
        # print(f"  Saved {name}: {df_optimized.shape} -> {out_path} | {csv_path}")


if __name__ == "__main__":
    main()
