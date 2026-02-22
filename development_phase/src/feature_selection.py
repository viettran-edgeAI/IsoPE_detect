#!/usr/bin/env python3
"""
Stage 2: Feature Filtering and Selection
=======================================
Load raw PE feature datasets, apply variance/correlation/stability filtering,
and save cleaned datasets for model optimization.

Usage:
    python feature_selection.py [--config PATH] [--corr-threshold X] [--variance-threshold X]

Outputs:
    ../data/cleaned/<model_name>_benign_train_clean.parquet
    ../data/cleaned/<model_name>_benign_val_clean.parquet
    ../data/cleaned/<model_name>_benign_test_clean.parquet
    ../data/cleaned/<model_name>_malware_val_clean.parquet
    ../data/cleaned/<model_name>_malware_test_clean.parquet
    ../schemas/feature_schema_selected.json
    ../reports/feature_selection_report.md
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as sp_kurtosis
from sklearn.preprocessing import StandardScaler


def _stable_json_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _drop_internal_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop duplicated rows inside a single dataset."""
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(cleaned)
    return cleaned, int(removed)


def _drop_cross_split_duplicates(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Drop rows in target_df that exactly match any row in source_df."""
    if len(source_df) == 0 or len(target_df) == 0:
        return target_df, 0

    source_hashes = set(pd.util.hash_pandas_object(source_df, index=False).tolist())
    target_hashes = pd.util.hash_pandas_object(target_df, index=False)
    overlap_mask = target_hashes.isin(source_hashes)
    removed = int(overlap_mask.sum())
    if removed == 0:
        return target_df, 0
    cleaned = target_df.loc[~overlap_mask].reset_index(drop=True)
    return cleaned, removed


def _extract_numeric_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.empty((len(df), 0), dtype=np.float32)
    return (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32, copy=False)
    )


def _normalize_rows_l2(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def _max_cosine_to_reference(query: np.ndarray, ref: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    if query.size == 0 or ref.size == 0:
        return np.zeros(query.shape[0], dtype=np.float32)
    ref_t = ref.T
    out = np.empty(query.shape[0], dtype=np.float32)
    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        sims = query[start:end] @ ref_t
        out[start:end] = sims.max(axis=1)
    return out


def _compute_cross_similarity_profile(
    val_norm: np.ndarray,
    test_norm: np.ndarray,
    thresholds: list[float],
) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "val_to_test": {},
        "test_to_val": {},
        "threshold_counts": {},
    }
    if val_norm.size == 0 or test_norm.size == 0:
        return profile

    val_to_test = _max_cosine_to_reference(val_norm, test_norm)
    test_to_val = _max_cosine_to_reference(test_norm, val_norm)

    profile["val_to_test"] = {
        "q50": float(np.quantile(val_to_test, 0.50)),
        "q90": float(np.quantile(val_to_test, 0.90)),
        "q99": float(np.quantile(val_to_test, 0.99)),
        "max": float(np.max(val_to_test)),
    }
    profile["test_to_val"] = {
        "q50": float(np.quantile(test_to_val, 0.50)),
        "q90": float(np.quantile(test_to_val, 0.90)),
        "q99": float(np.quantile(test_to_val, 0.99)),
        "max": float(np.max(test_to_val)),
    }

    for thr in thresholds:
        key = f">={thr:.4f}"
        profile["threshold_counts"][key] = {
            "val_to_test": int((val_to_test >= thr).sum()),
            "test_to_val": int((test_to_val >= thr).sum()),
        }
    return profile


def _compute_knn_cross_split_metrics(
    val_norm: np.ndarray,
    test_norm: np.ndarray,
    k_list: list[int],
    sample_size_per_split: int,
    seed: int,
) -> dict[str, Any]:
    from sklearn.neighbors import NearestNeighbors

    if val_norm.size == 0 or test_norm.size == 0:
        return {
            "available": False,
            "reason": "empty split",
        }

    rng = np.random.default_rng(seed)
    val = val_norm
    test = test_norm
    if sample_size_per_split > 0 and len(val) > sample_size_per_split:
        idx = rng.choice(len(val), size=sample_size_per_split, replace=False)
        val = val[idx]
    if sample_size_per_split > 0 and len(test) > sample_size_per_split:
        idx = rng.choice(len(test), size=sample_size_per_split, replace=False)
        test = test[idx]

    X = np.vstack([val, test])
    y = np.concatenate([np.zeros(len(val), dtype=np.int8), np.ones(len(test), dtype=np.int8)])

    k_list = sorted({int(k) for k in k_list if int(k) > 0})
    if not k_list:
        k_list = [1, 5]
    max_k = min(max(k_list), len(X) - 1)
    if max_k <= 0:
        return {
            "available": False,
            "reason": "insufficient samples",
        }

    nn = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine")
    nn.fit(X)
    _, idx = nn.kneighbors(X)
    neighbors = idx[:, 1 : max_k + 1]
    neighbor_labels = y[neighbors]

    out: dict[str, Any] = {
        "available": True,
        "sampled_val": int(len(val)),
        "sampled_test": int(len(test)),
        "k": {},
    }
    for k in k_list:
        if k > max_k:
            continue
        frac_other = (neighbor_labels[:, :k] != y[:, None]).mean(axis=1)
        val_mask = y == 0
        test_mask = y == 1
        out["k"][str(k)] = {
            "val_to_test_cross_rate": float(frac_other[val_mask].mean()) if val_mask.any() else 0.0,
            "test_to_val_cross_rate": float(frac_other[test_mask].mean()) if test_mask.any() else 0.0,
            "global_cross_rate": float(frac_other.mean()),
        }
    return out


def _select_top_features_cohens_d(
    X_train: np.ndarray,
    X_malware: np.ndarray,
    feature_names: np.ndarray,
    top_k: int,
) -> np.ndarray:
    mean_benign = X_train.mean(axis=0)
    mean_malware = X_malware.mean(axis=0)
    std_benign = X_train.std(axis=0)
    std_malware = X_malware.std(axis=0)
    pooled_std = np.sqrt(0.5 * (std_benign * std_benign + std_malware * std_malware)) + 1e-10
    cohens_d = np.abs(mean_benign - mean_malware) / pooled_std

    top_k = min(int(top_k), int(len(feature_names)))
    return np.argsort(cohens_d)[-top_k:]


def _iterative_projected_similarity_pruning(
    benign_train_clean: pd.DataFrame,
    malware_val_clean: pd.DataFrame,
    malware_test_clean: pd.DataFrame,
    top_k_list: list[int],
    similarity_threshold: float,
    max_iterations: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stats: dict[str, Any] = {
        "enabled": True,
        "similarity_threshold": float(similarity_threshold),
        "top_k_list": [int(x) for x in top_k_list],
        "before": int(len(malware_val_clean)),
        "after": int(len(malware_val_clean)),
        "removed": 0,
        "iterations": 0,
        "per_iteration": [],
        "final_topk_max_similarity": {},
        "all_topk_pass": False,
    }

    if len(malware_val_clean) == 0 or len(malware_test_clean) == 0:
        stats["all_topk_pass"] = False
        return malware_val_clean, stats

    top_k_list = sorted({int(x) for x in top_k_list if int(x) > 0})
    if not top_k_list:
        stats["all_topk_pass"] = True
        return malware_val_clean, stats

    train_values = benign_train_clean.to_numpy(dtype=np.float32, copy=False)
    test_values = malware_test_clean.to_numpy(dtype=np.float32, copy=False)
    feature_names = np.array(malware_val_clean.columns.tolist())

    work = malware_val_clean.copy().reset_index(drop=True)
    total_removed = 0

    for iteration in range(1, max(1, int(max_iterations)) + 1):
        if len(work) == 0:
            break

        val_values = work.to_numpy(dtype=np.float32, copy=False)
        offending = np.zeros(len(work), dtype=bool)
        iter_topk: dict[str, Any] = {}

        for top_k in top_k_list:
            top_idx = _select_top_features_cohens_d(
                train_values,
                val_values,
                feature_names,
                top_k=top_k,
            )
            X_train_sel = train_values[:, top_idx]
            X_val_sel = val_values[:, top_idx]
            X_test_sel = test_values[:, top_idx]

            scaler = StandardScaler()
            scaler.fit(X_train_sel)

            X_val_scaled = scaler.transform(X_val_sel)
            X_test_scaled = scaler.transform(X_test_sel)

            val_norm = _normalize_rows_l2(X_val_scaled.astype(np.float32, copy=False))
            test_norm = _normalize_rows_l2(X_test_scaled.astype(np.float32, copy=False))
            max_sim = _max_cosine_to_reference(val_norm, test_norm)

            viol = max_sim >= float(similarity_threshold)
            offending |= viol

            iter_topk[str(top_k)] = {
                "max_similarity": float(max_sim.max()) if len(max_sim) else 0.0,
                "violating_rows": int(viol.sum()),
            }

        removed_now = int(offending.sum())
        stats["per_iteration"].append(
            {
                "iteration": int(iteration),
                "val_count_before": int(len(work)),
                "removed_rows": removed_now,
                "top_k": iter_topk,
            }
        )

        if removed_now == 0:
            break

        work = work.loc[~offending].reset_index(drop=True)
        total_removed += removed_now

    stats["iterations"] = int(len(stats["per_iteration"]))
    stats["removed"] = int(total_removed)
    stats["after"] = int(len(work))

    final_pass = True
    final_topk_max: dict[str, float] = {}
    if len(work) == 0:
        final_pass = False
    else:
        val_values = work.to_numpy(dtype=np.float32, copy=False)
        for top_k in top_k_list:
            top_idx = _select_top_features_cohens_d(
                train_values,
                val_values,
                feature_names,
                top_k=top_k,
            )
            X_train_sel = train_values[:, top_idx]
            X_val_sel = val_values[:, top_idx]
            X_test_sel = test_values[:, top_idx]

            scaler = StandardScaler()
            scaler.fit(X_train_sel)

            X_val_scaled = scaler.transform(X_val_sel)
            X_test_scaled = scaler.transform(X_test_sel)

            val_norm = _normalize_rows_l2(X_val_scaled.astype(np.float32, copy=False))
            test_norm = _normalize_rows_l2(X_test_scaled.astype(np.float32, copy=False))
            max_sim = _max_cosine_to_reference(val_norm, test_norm)
            max_value = float(max_sim.max()) if len(max_sim) else 0.0
            final_topk_max[str(top_k)] = max_value
            if max_value >= float(similarity_threshold):
                final_pass = False

    stats["final_topk_max_similarity"] = final_topk_max
    stats["all_topk_pass"] = bool(final_pass)
    return work, stats


def _build_stage2_independence_manifest(
    config_path: Path,
    audit_cfg: dict[str, Any],
    split_separation_stats: dict[str, Any],
    malware_val_ratio_stats: dict[str, Any],
    malware_val_clean: pd.DataFrame,
    malware_test_clean: pd.DataFrame,
    knn_metrics: dict[str, Any],
    similarity_profile: dict[str, Any],
    projected_similarity_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_hash = _stable_json_hash(_load_config(config_path))

    run_basis = {
        "config_hash": config_hash,
        "malware_val_count": int(len(malware_val_clean)),
        "malware_test_count": int(len(malware_test_clean)),
        "audit_mode": "enforce",
    }
    run_id = _stable_json_hash(run_basis)

    min_required = int(audit_cfg.get("require_min_malware_val_after_audit", 300))
    min_samples_pass = int(len(malware_val_clean)) >= min_required
    projected_pass = True
    if projected_similarity_stats is not None:
        projected_pass = bool(projected_similarity_stats.get("all_topk_pass", False))

    all_pass = bool(split_separation_stats.get("sealed_pass", False) and min_samples_pass and projected_pass)

    return {
        "schema_version": "2.0",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "config_path": str(config_path),
        "config_hash": config_hash,
        "audit_mode": "enforce",
        "counts": {
            "malware_val_clean": int(len(malware_val_clean)),
            "malware_test_clean": int(len(malware_test_clean)),
        },
        "separation": split_separation_stats,
        "malware_val_size_control": malware_val_ratio_stats,
        "quantitative_checks": {
            "knn_cross_split": knn_metrics,
            "cross_similarity_profile": similarity_profile,
            "projected_topk_similarity": projected_similarity_stats,
        },
        "gates": {
            "require_min_malware_val_after_audit": min_required,
            "min_samples_pass": bool(min_samples_pass),
            "sealed_pass": bool(split_separation_stats.get("sealed_pass", False)),
            "projected_topk_similarity_pass": bool(projected_pass),
            "all_pass": bool(all_pass),
        },
    }


def _pick_malware_validation_subset(df: pd.DataFrame, target_size: int, seed: int) -> pd.DataFrame:
    if len(df) <= target_size:
        return df
    if target_size <= 0:
        return df.iloc[0:0].copy()

    rng = np.random.default_rng(seed)

    if "imphash" not in df.columns:
        idx = rng.choice(df.index.to_numpy(), size=target_size, replace=False)
        return df.loc[idx].copy()

    imphash = pd.to_numeric(df["imphash"], errors="coerce").fillna(0).astype(np.int64)
    nonzero_mask = imphash != 0

    selected_indices = []
    nonzero_df = df[nonzero_mask]
    nonzero_hash = imphash[nonzero_mask]
    if len(nonzero_df) > 0:
        grouped = nonzero_df.groupby(nonzero_hash, sort=False)
        reps = grouped.head(1)
        rep_indices = np.array(reps.index.to_numpy(), copy=True)
        rng.shuffle(rep_indices)
        selected_indices.extend(rep_indices.tolist())

    if len(selected_indices) > target_size:
        selected_indices = selected_indices[:target_size]
        return df.loc[selected_indices].copy()

    remaining_indices = [idx for idx in df.index.to_numpy() if idx not in set(selected_indices)]
    need = target_size - len(selected_indices)
    if need > 0 and remaining_indices:
        fill = rng.choice(np.array(remaining_indices), size=min(need, len(remaining_indices)), replace=False)
        selected_indices.extend(fill.tolist())

    return df.loc[selected_indices].copy()


def enforce_malware_split_separation(
    malware_val_df: pd.DataFrame,
    malware_test_df: pd.DataFrame,
    benign_test_count: int,
    separation_cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    val_work = malware_val_df.copy()
    test_work = malware_test_df.copy()

    enforce_imphash_disjoint = True
    max_cross_similarity_raw = separation_cfg.get("max_cross_similarity", None)
    max_cross_similarity = (
        float(max_cross_similarity_raw) if max_cross_similarity_raw is not None else None
    )

    stats = {
        "val_before": int(len(val_work)),
        "val_after": int(len(val_work)),
        "test_count": int(len(test_work)),
        "exact_overlap": 0,
        "exact_overlap_removed": 0,
        "exact_overlap_after": 0,
        "shared_imphash_unique": 0,
        "shared_imphash_removed_rows": 0,
        "imphash_jaccard": 0.0,
        "val_shared_rate": 0.0,
        "test_shared_rate": 0.0,
        "enforce_imphash_disjoint": enforce_imphash_disjoint,
        "similarity_threshold": max_cross_similarity,
        "similarity_removed_rows": 0,
        "max_similarity_before": 0.0,
        "max_similarity_after": 0.0,
        "sealed_pass": True,
    }

    if len(val_work) > 0 and len(test_work) > 0:
        val_sig = pd.util.hash_pandas_object(val_work, index=False).astype(np.uint64)
        test_sig = pd.util.hash_pandas_object(test_work, index=False).astype(np.uint64)
        test_sig_set = set(test_sig.tolist())
        overlap_mask = val_sig.isin(test_sig_set)
        stats["exact_overlap"] = int(overlap_mask.sum())
        if stats["exact_overlap"] > 0:
            val_work = val_work.loc[~overlap_mask].reset_index(drop=True)
            stats["exact_overlap_removed"] = int(stats["exact_overlap"])
            stats["val_after"] = int(len(val_work))

    if "imphash" in val_work.columns and "imphash" in test_work.columns:
        val_imphash = pd.to_numeric(val_work["imphash"], errors="coerce").fillna(0).astype(np.int64)
        test_imphash = pd.to_numeric(test_work["imphash"], errors="coerce").fillna(0).astype(np.int64)

        test_nonzero_set = set(test_imphash[test_imphash != 0].tolist())

        if enforce_imphash_disjoint and len(test_nonzero_set) > 0:
            family_overlap_mask = (val_imphash != 0) & val_imphash.isin(test_nonzero_set)
            removed_family_rows = int(family_overlap_mask.sum())
            if removed_family_rows > 0:
                val_work = val_work.loc[~family_overlap_mask].reset_index(drop=True)
                stats["shared_imphash_removed_rows"] = removed_family_rows
                stats["val_after"] = int(len(val_work))

    if max_cross_similarity is not None and len(val_work) > 0 and len(test_work) > 0:
        common_cols = [c for c in val_work.columns if c in test_work.columns]
        val_mat = _extract_numeric_matrix(val_work, common_cols)
        test_mat = _extract_numeric_matrix(test_work, common_cols)
        val_norm = _normalize_rows_l2(val_mat)
        test_norm = _normalize_rows_l2(test_mat)

        max_sim_before = _max_cosine_to_reference(val_norm, test_norm)
        stats["max_similarity_before"] = float(max_sim_before.max()) if len(max_sim_before) else 0.0

        similarity_mask = max_sim_before >= float(max_cross_similarity)
        sim_removed = int(similarity_mask.sum())
        if sim_removed > 0:
            val_work = val_work.loc[~similarity_mask].reset_index(drop=True)
            stats["similarity_removed_rows"] = sim_removed
            stats["val_after"] = int(len(val_work))

        if len(val_work) > 0:
            val_mat_after = _extract_numeric_matrix(val_work, common_cols)
            val_norm_after = _normalize_rows_l2(val_mat_after)
            max_sim_after = _max_cosine_to_reference(val_norm_after, test_norm)
            stats["max_similarity_after"] = float(max_sim_after.max()) if len(max_sim_after) else 0.0
        else:
            stats["max_similarity_after"] = 0.0

    if len(val_work) > 0 and len(test_work) > 0:
        val_sig_after = pd.util.hash_pandas_object(val_work, index=False).astype(np.uint64)
        test_sig_after = pd.util.hash_pandas_object(test_work, index=False).astype(np.uint64)
        stats["exact_overlap_after"] = int(val_sig_after.isin(set(test_sig_after.tolist())).sum())

    if "imphash" in val_work.columns and "imphash" in test_work.columns:
        val_imphash_final = pd.to_numeric(val_work["imphash"], errors="coerce").fillna(0).astype(np.int64)
        test_imphash_final = pd.to_numeric(test_work["imphash"], errors="coerce").fillna(0).astype(np.int64)

        val_nonzero = val_imphash_final[val_imphash_final != 0]
        test_nonzero = test_imphash_final[test_imphash_final != 0]
        val_set = set(val_nonzero.tolist())
        test_set = set(test_nonzero.tolist())
        shared = val_set & test_set
        union = val_set | test_set

        stats["shared_imphash_unique"] = int(len(shared))
        stats["imphash_jaccard"] = float(len(shared) / len(union)) if union else 0.0
        stats["val_shared_rate"] = float(val_nonzero.isin(shared).mean()) if len(val_nonzero) else 0.0
        stats["test_shared_rate"] = float(test_nonzero.isin(shared).mean()) if len(test_nonzero) else 0.0

    exact_ok = stats["exact_overlap_after"] == 0
    family_ok = stats["shared_imphash_unique"] == 0
    similarity_ok = (
        max_cross_similarity is None
        or stats["max_similarity_after"] < float(max_cross_similarity)
    )
    stats["sealed_pass"] = bool(exact_ok and family_ok and similarity_ok)
    return val_work, stats


def _downsample_malware_val_relative_to_benign_val(
    malware_val_df: pd.DataFrame,
    benign_val_count: int,
    ratio: float,
    seed: int,
    min_samples: int,
) -> tuple[pd.DataFrame, dict]:
    """Downsample malware validation set to a ratio of benign validation size."""
    ratio = max(0.0, min(float(ratio), 1.0))
    before = int(len(malware_val_df))

    if before == 0 or benign_val_count <= 0 or ratio <= 0.0:
        target_size = 0
    else:
        target_size = max(1, int(round(benign_val_count * ratio)))

    min_samples = max(0, int(min_samples))
    if before > 0 and min_samples > 0:
        target_size = max(target_size, min_samples)

    target_size = min(target_size, before)
    sampled_df = _pick_malware_validation_subset(malware_val_df, target_size=target_size, seed=seed)

    stats = {
        "ratio": float(ratio),
        "min_samples": int(min_samples),
        "benign_val_count": int(benign_val_count),
        "before": before,
        "target": int(target_size),
        "after": int(len(sampled_df)),
        "removed": int(before - len(sampled_df)),
    }
    return sampled_df, stats

def _to_float(value) -> float:
    try:
        return float(np.real(value))
    except (TypeError, ValueError):
        return 0.0


def _load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(config_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return config_path.parent / path


def _sanitize_model_name(raw_name: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in str(raw_name).strip())
    return cleaned or "iforest"


def _render_model_name(path_template: str, model_name: str) -> str:
    return str(path_template).replace("{model_name}", model_name)


def _validate_config(cfg: dict) -> None:
    required_top = ["filtering"]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")


def variance_filter(
    df: pd.DataFrame,
    group_mapping: dict,
    variance_threshold: float = 0.0,
    norm_var_threshold: float = 1e-7,
) -> tuple[list[str], list[dict]]:
    """Remove low-information features.

    Rules:
    - Any feature with raw variance <= variance_threshold is removed.
    - Binary flags with minority class ratio < 0.5% are removed.
    - Hash buckets with nonzero fraction < 1% are removed.
    - Continuous features with normalized variance < norm_var_threshold are removed.
    """
    removed = []
    kept = []

    for col in df.columns:
        vals = df[col]
        unique = vals.nunique()
        raw_var = _to_float(vals.var())

        if raw_var <= variance_threshold or unique <= 1:
            removed.append(
                {
                    "name": col,
                    "group": group_mapping.get(col, "?"),
                    "reason": "raw_variance_threshold",
                    "value": round(raw_var, 10),
                }
            )
            continue

        is_binary = set(vals.unique()).issubset({0, 1, 0.0, 1.0})
        is_hash = "hash_" in col

        if is_binary:
            minority = min(vals.mean(), 1 - vals.mean())
            if minority < 0.005:
                removed.append(
                    {
                        "name": col,
                        "group": group_mapping.get(col, "?"),
                        "reason": "near_constant_binary",
                        "value": round(float(minority), 6),
                    }
                )
                continue
        elif is_hash:
            nonzero_ratio = (vals != 0).mean()
            if nonzero_ratio < 0.01:
                removed.append(
                    {
                        "name": col,
                        "group": group_mapping.get(col, "?"),
                        "reason": "inactive_hash_bucket",
                        "value": round(float(nonzero_ratio), 6),
                    }
                )
                continue
        else:
            rng = _to_float(vals.max() - vals.min())
            norm_var = raw_var / (rng**2 + 1e-15) if rng > 0 else 0.0
            if norm_var < norm_var_threshold:
                removed.append(
                    {
                        "name": col,
                        "group": group_mapping.get(col, "?"),
                        "reason": "near_zero_normalized_variance",
                        "value": round(float(norm_var), 10),
                    }
                )
                continue

        kept.append(col)

    return kept, removed


def correlation_pruning(df: pd.DataFrame, threshold: float = 0.95, group_mapping: dict | None = None) -> tuple[list[str], list[dict]]:
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    pairs = []
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            num = pd.to_numeric(val, errors="coerce")
            if not pd.isna(num) and num > threshold:
                pairs.append((idx, col, float(num)))

    pairs.sort(key=lambda x: -x[2])

    to_remove = set()
    removed_log = []
    for f1, f2, corr_val in pairs:
        if f1 in to_remove or f2 in to_remove:
            continue
        deg1 = sum(1 for a, b, _ in pairs if (b == f1 or a == f1) and b not in to_remove and a not in to_remove)
        deg2 = sum(1 for a, b, _ in pairs if (b == f2 or a == f2) and b not in to_remove and a not in to_remove)
        var1, var2 = _to_float(df[f1].var()), _to_float(df[f2].var())

        if deg1 > deg2 or (deg1 == deg2 and var1 < var2):
            drop, keep = f1, f2
        else:
            drop, keep = f2, f1
        to_remove.add(drop)
        removed_log.append(
            {
                "dropped": drop,
                "kept": keep,
                "correlation": round(corr_val, 4),
                "dropped_group": group_mapping.get(drop, "?") if group_mapping else "?",
            }
        )

    kept = [c for c in df.columns if c not in to_remove]
    return kept, removed_log


def stability_filter(df: pd.DataFrame, group_mapping: dict) -> tuple[list[str], list[str], list[dict]]:
    removed = []
    kept = []
    log_transform = []

    for col in df.columns:
        vals = df[col]
        is_binary = set(vals.unique()).issubset({0, 1, 0.0, 1.0})

        if is_binary:
            fill = float(vals.mean())
            if fill < 0.001 and (1 - fill) < 0.001:
                removed.append(
                    {
                        "name": col,
                        "group": group_mapping.get(col, "?"),
                        "reason": "binary_empty",
                        "value": 0,
                    }
                )
                continue
            kept.append(col)
            continue

        mu = float(vals.mean())
        sigma = float(vals.std())
        cv = sigma / (abs(mu) + 1e-10)

        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        outlier_frac = float(((vals > q3 + 5 * iqr) | (vals < q1 - 5 * iqr)).mean())

        kurt = float(sp_kurtosis(vals, fisher=True, nan_policy="omit"))
        fill_rate = float((vals != 0).mean())

        reasons = []
        if cv > 50:
            reasons.append("high_cv")
        if outlier_frac > 0.05:
            reasons.append("outlier_dominated")
        if kurt > 100:
            reasons.append("extreme_kurtosis")
        if fill_rate < 0.10:
            reasons.append("sparse_fill")

        if reasons:
            log_vals = np.log1p(np.abs(vals))
            log_cv = float(log_vals.std() / (abs(log_vals.mean()) + 1e-10))
            log_kurt = float(sp_kurtosis(log_vals, fisher=True, nan_policy="omit"))

            if log_cv < 10 and log_kurt < 20:
                kept.append(col)
                log_transform.append(col)
                continue
            removed.append(
                {
                    "name": col,
                    "group": group_mapping.get(col, "?"),
                    "reason": "|".join(reasons),
                    "cv": round(cv, 2),
                    "outlier_frac": round(outlier_frac, 4),
                    "kurtosis": round(kurt, 1),
                    "fill_rate": round(fill_rate, 4),
                }
            )
            continue

        kept.append(col)

    return kept, log_transform, removed


def apply_transforms(df: pd.DataFrame, selected_cols: list[str], log_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in selected_cols if c not in df.columns]
    for c in missing:
        df[c] = 0.0

    out = df[selected_cols].copy()
    for c in log_cols:
        if c in out.columns:
            out[c] = np.log1p(np.abs(out[c].astype(np.float64))).astype(np.float32)
    return out


def generate_report(
    all_raw_cols,
    kept_after_var,
    var_removed,
    kept_after_corr,
    corr_removed,
    kept_after_stab,
    stab_removed,
    log_transform_cols,
    group_mapping,
    split_separation_stats=None,
    malware_val_ratio_stats=None,
    independence_manifest=None,
    projected_similarity_stats=None,
) -> str:
    selected_features = kept_after_stab
    grp_before = Counter(group_mapping.get(c, "?") for c in all_raw_cols if c in group_mapping)
    selected_groups = Counter(group_mapping.get(c, "?") for c in selected_features)

    lines = [
        "# Feature Selection Report\n",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}\n",
        "## Summary\n",
        f"- Raw features loaded: **{len(all_raw_cols)}**",
        f"- After variance filtering: **{len(kept_after_var)}** (removed {len(var_removed)})",
        f"- After correlation pruning: **{len(kept_after_corr)}** (removed {len(corr_removed)})",
        f"- After stability filtering: **{len(kept_after_stab)}** (removed {len(stab_removed)})",
        f"- Log-transformed features: **{len(log_transform_cols)}**",
        f"- **Final selected: {len(selected_features)} features in {len(selected_groups)} groups**\n",
        "## Per-Group Summary\n",
        "| Group | Raw | After Var | After Corr | After Stab |",
        "|-------|-----|-----------|------------|------------|",
    ]

    for grp in sorted(grp_before.keys()):
        raw_n = grp_before[grp]
        var_n = Counter(group_mapping.get(c, "?") for c in kept_after_var).get(grp, 0)
        cor_n = Counter(group_mapping.get(c, "?") for c in kept_after_corr).get(grp, 0)
        stb_n = selected_groups.get(grp, 0)
        lines.append(f"| {grp} | {raw_n} | {var_n} | {cor_n} | {stb_n} |")

    lines.append(f"\n## Selected Features ({len(selected_features)} total)\n")
    for grp in sorted(selected_groups.keys()):
        cols = [c for c in selected_features if group_mapping.get(c, "?") == grp]
        lines.append(f"### {grp} ({len(cols)} features)")
        for c in cols:
            tag = " [log1p]" if c in log_transform_cols else ""
            lines.append(f"- `{c}`{tag}")
        lines.append("")

    if split_separation_stats is not None:
        lines.extend(
            [
                "## Malware Validation/Test Independence Audit\n",
                f"- Sealed-test check: **{split_separation_stats.get('sealed_pass', False)}**",
                f"- Exact overlap after cleanup: **{split_separation_stats.get('exact_overlap_after', 0)}**",
                f"- Shared unique imphash after cleanup: **{split_separation_stats.get('shared_imphash_unique', 0)}**",
                f"- Max cosine similarity after cleanup: **{split_separation_stats.get('max_similarity_after', 0.0):.4f}**",
            ]
        )

    if independence_manifest is not None:
        lines.extend(
            [
                "\n## Independence Manifest\n",
                f"- Run ID: **{independence_manifest.get('run_id', 'n/a')}**",
                f"- Manifest path: **{independence_manifest.get('manifest_path', 'n/a')}**",
                f"- Hard gates passed: **{independence_manifest.get('all_pass', False)}**",
            ]
        )

    if projected_similarity_stats is not None:
        lines.extend(
            [
                "\n## Projected Similarity Pruning\n",
                f"- Enabled: **{projected_similarity_stats.get('enabled', False)}**",
                f"- Threshold: **{projected_similarity_stats.get('similarity_threshold', 0.0):.4f}**",
                f"- malware_val before/after: **{projected_similarity_stats.get('before', 0)} / {projected_similarity_stats.get('after', 0)}**",
                f"- Removed by projected pruning: **{projected_similarity_stats.get('removed', 0)}**",
                f"- All top-k constraints pass: **{projected_similarity_stats.get('all_topk_pass', False)}**",
            ]
        )

    if malware_val_ratio_stats is not None:
        lines.extend(
            [
                "\n## Malware Validation Size Control\n",
                f"- Ratio to benign_val: **{malware_val_ratio_stats.get('ratio', 0.0):.4f}**",
                f"- Min malware_val floor: **{malware_val_ratio_stats.get('min_samples', 0)}**",
                f"- benign_val samples: **{malware_val_ratio_stats.get('benign_val_count', 0)}**",
                f"- malware_val before: **{malware_val_ratio_stats.get('before', 0)}**",
                f"- malware_val target: **{malware_val_ratio_stats.get('target', 0)}**",
                f"- malware_val after: **{malware_val_ratio_stats.get('after', 0)}**",
                f"- removed by ratio cap: **{malware_val_ratio_stats.get('removed', 0)}**",
            ]
        )

    return "\n".join(lines)


def main():
    script_dir = Path(__file__).resolve().parent
    default_config_path = script_dir / "feature_selection_config.json"

    parser = argparse.ArgumentParser(description="Stage 2: Feature Filtering and Selection")
    parser.add_argument("--config", type=str, default=None, help=f"Path to JSON config file (defaults to {default_config_path})")
    parser.add_argument("--corr-threshold", type=float, default=None, help="Correlation pruning threshold (overrides config)")
    parser.add_argument("--variance-threshold", type=float, default=None, help="Raw variance threshold (overrides config)")
    parser.add_argument("--norm-var-threshold", type=float, default=None, help="Normalized variance threshold for continuous features (overrides config)")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path
    cfg = _load_config(config_path)
    _validate_config(cfg)

    # Load pipeline config (sibling to stage config)
    pipeline_cfg_path = (config_path.parent / "pipeline_config.json").resolve()
    pipeline_cfg = _load_config(pipeline_cfg_path) if pipeline_cfg_path.is_file() else {}
    paths_cfg = pipeline_cfg.get("paths", {})

    model_name_raw = str(pipeline_cfg.get("model_name", "iforest"))
    model_name = _sanitize_model_name(model_name_raw)
    if model_name != model_name_raw:
        print(f"Warning: model_name sanitized from '{model_name_raw}' to '{model_name}'")

    raw_data_dir = _resolve_path(pipeline_cfg_path, paths_cfg.get("raw_data_dir", "../data/raw"))
    train_path  = raw_data_dir / f"{model_name}_benign_train_raw.parquet"
    val_path    = raw_data_dir / f"{model_name}_benign_val_raw.parquet"
    test_b_path = raw_data_dir / f"{model_name}_benign_test_raw.parquet"
    test_m_path = raw_data_dir / f"{model_name}_malware_test_raw.parquet"
    val_m_path  = raw_data_dir / f"{model_name}_malware_val_raw.parquet"

    filt_cfg = cfg["filtering"]
    variance_threshold = args.variance_threshold if args.variance_threshold is not None else float(filt_cfg.get("variance_threshold", 0.0))
    corr_threshold = args.corr_threshold if args.corr_threshold is not None else float(filt_cfg.get("corr_threshold", 0.95))
    norm_var_threshold = args.norm_var_threshold if args.norm_var_threshold is not None else float(filt_cfg.get("norm_var_threshold", 1e-7))

    cleaned_dir = _resolve_path(pipeline_cfg_path, paths_cfg.get("cleaned_data_dir", "../data/cleaned"))
    schema_dir  = _resolve_path(pipeline_cfg_path, paths_cfg.get("schema_dir", "../schemas"))
    report_dir  = _resolve_path(pipeline_cfg_path, paths_cfg.get("report_dir", "../reports"))

    separation_cfg = cfg.get("separation", {})
    malware_val_ratio = float(separation_cfg.get("malware_val_ratio_to_benign_val", 0.03))
    malware_val_ratio_seed = int(separation_cfg.get("malware_val_ratio_seed", 42))
    malware_val_min_samples = int(separation_cfg.get("malware_val_min_samples", 500))
    audit_cfg = cfg.get("independence_audit", {})
    audit_mode = "enforce"
    manifest_out_path = report_dir / "malware_val_test_independence_manifest_stage2.json"
    knn_k_list = [int(x) for x in audit_cfg.get("knn_k_list", [1, 5, 10])]
    knn_sample_size = int(audit_cfg.get("knn_sample_size_per_split", 2000))
    knn_seed = int(audit_cfg.get("knn_seed", 42))
    similarity_thresholds = [float(x) for x in audit_cfg.get("similarity_profile_thresholds", [0.99, 0.995, 0.999])]
    min_malware_val_after_audit = int(audit_cfg.get("require_min_malware_val_after_audit", 300))
    projected_pruning_enabled = True
    projected_similarity_threshold = float(audit_cfg.get("projected_similarity_threshold", separation_cfg.get("max_cross_similarity", 0.995)))
    projected_max_iterations = int(audit_cfg.get("projected_max_iterations", 8))

    projected_top_k_raw = audit_cfg.get("projected_top_k_list")
    if projected_top_k_raw is None:
        model_cfg_path = config_path.parent / "model_config.json"
        projected_top_k_raw = [20, 25, 40, 100]
        if model_cfg_path.exists():
            try:
                model_cfg = _load_config(model_cfg_path)
                projected_top_k_raw = model_cfg.get("feature_selection", {}).get("top_k_list", projected_top_k_raw)
            except Exception:
                projected_top_k_raw = [20, 25, 40, 100]
    projected_top_k_list = [int(x) for x in projected_top_k_raw]

    cleaned_dir.mkdir(parents=True, exist_ok=True)
    schema_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    group_mapping_path = schema_dir / "feature_group_mapping.json"
    if not group_mapping_path.exists():
        raise FileNotFoundError(
            f"Missing group mapping at {group_mapping_path}. Run feature_extraction.py first."
        )

    with open(group_mapping_path, "r", encoding="utf-8") as f:
        group_mapping = json.load(f)

    print("=" * 70)
    print("STAGE 2: FEATURE FILTERING & SELECTION")
    print("=" * 70)
    print(f"Model name: {model_name}")
    print(
        f"Configuration: variance_threshold={variance_threshold}, "
        f"corr_threshold={corr_threshold}, norm_var_threshold={norm_var_threshold}"
    )

    print("\nLoading training and validation datasets...")
    # Load only train/val data initially - test data loaded after feature selection
    dfs_train_val = {
        "benign_train": pd.read_parquet(train_path),
        "benign_val": pd.read_parquet(val_path),
        "malware_val": pd.read_parquet(val_m_path),
    }

    print("\n--- Automatic Duplicate Cleanup (Train/Validation) ---")
    for split_name in ["benign_train", "benign_val", "malware_val"]:
        cleaned_df, removed = _drop_internal_duplicates(dfs_train_val[split_name])
        dfs_train_val[split_name] = cleaned_df
        if removed > 0:
            print(f"  Removed {removed} in-split duplicates from {split_name}")

    removed_val_vs_train = 0
    dfs_train_val["benign_val"], removed_val_vs_train = _drop_cross_split_duplicates(
        dfs_train_val["benign_train"],
        dfs_train_val["benign_val"],
    )
    if removed_val_vs_train > 0:
        print(
            f"  Removed {removed_val_vs_train} cross-split duplicates from benign_val "
            f"that overlapped with benign_train"
        )
    if len(dfs_train_val["benign_val"]) == 0:
        raise ValueError("benign_val became empty after duplicate cleanup against benign_train")
    
    print(f"  Training (benign):      {len(dfs_train_val['benign_train'])} samples")
    print(f"  Validation (benign):    {len(dfs_train_val['benign_val'])} samples")
    print(f"  Validation (malware):   {len(dfs_train_val['malware_val'])} samples")
    print("  Test data will be loaded after feature selection (sealed until needed)")

    print("\n--- Benign Split Independence Verification ---")
    print(
        "  ✓ Applied automatic deduplication between benign_train and benign_val "
        f"(removed {removed_val_vs_train} rows)"
    )

    # Use benign_train columns as canonical raw order
    all_raw_cols = list(dfs_train_val["benign_train"].columns)
    print(f"\n  Raw features: {len(all_raw_cols)}")

    print("\n--- Variance Filtering ---")
    # Feature selection uses ONLY benign training data because:
    # 1. Anomaly detection learns "normal" behavior from benign samples
    # 2. Test data must remain sealed to prevent leakage
    # 3. Validation data is excluded to avoid look-ahead bias
    kept_after_var, var_removed = variance_filter(
        dfs_train_val["benign_train"],
        group_mapping,
        variance_threshold=variance_threshold,
        norm_var_threshold=norm_var_threshold,
    )
    print(f"  {len(all_raw_cols)} -> {len(kept_after_var)} features (removed {len(var_removed)})")

    print("\n--- Correlation Pruning ---")
    df_var_filtered = dfs_train_val["benign_train"][kept_after_var]
    kept_after_corr, corr_removed = correlation_pruning(
        df_var_filtered,
        threshold=corr_threshold,
        group_mapping=group_mapping,
    )
    print(f"  {len(kept_after_var)} -> {len(kept_after_corr)} features (removed {len(corr_removed)})")

    print("\n--- Stability Filtering ---")
    df_corr_filtered = dfs_train_val["benign_train"][kept_after_corr]
    kept_after_stab, log_transform_cols, stab_removed = stability_filter(df_corr_filtered, group_mapping)
    print(f"  {len(kept_after_corr)} -> {len(kept_after_stab)} features (removed {len(stab_removed)})")
    print(f"  Log-transform rescued: {len(log_transform_cols)}")

    selected_features = kept_after_stab

    # ── Malware Validation/Test Independence Verification ──
    print("\n--- Loading Test Data (After Feature Selection) ---")
    dfs_test = {
        "benign_test": pd.read_parquet(test_b_path),
        "malware_test": pd.read_parquet(test_m_path),
    }

    print("\n--- Automatic Duplicate Cleanup (Test Splits) ---")
    for split_name in ["benign_test", "malware_test"]:
        cleaned_df, removed = _drop_internal_duplicates(dfs_test[split_name])
        dfs_test[split_name] = cleaned_df
        if removed > 0:
            print(f"  Removed {removed} in-split duplicates from {split_name}")

    removed_test_vs_train = 0
    removed_test_vs_val = 0
    dfs_test["benign_test"], removed_test_vs_train = _drop_cross_split_duplicates(
        dfs_train_val["benign_train"],
        dfs_test["benign_test"],
    )
    dfs_test["benign_test"], removed_test_vs_val = _drop_cross_split_duplicates(
        dfs_train_val["benign_val"],
        dfs_test["benign_test"],
    )
    if removed_test_vs_train > 0 or removed_test_vs_val > 0:
        print(
            f"  Removed {removed_test_vs_train} benign_test rows overlapping benign_train and "
            f"{removed_test_vs_val} rows overlapping benign_val"
        )
    if len(dfs_test["benign_test"]) == 0:
        raise ValueError("benign_test became empty after duplicate cleanup")
    print(f"  Test (benign):  {len(dfs_test['benign_test'])} samples")
    print(f"  Test (malware): {len(dfs_test['malware_test'])} samples")

    # Verify benign test/validation independence
    print("\n--- Benign Test/Validation Independence Verification ---")
    print(
        "  ✓ Applied automatic deduplication for benign_test vs benign_val "
        f"(removed {removed_test_vs_val} rows)"
    )
    
    # Verify malware validation/test independence
    print("\n--- Malware Validation/Test Independence Verification ---")
    malware_val_checked, split_separation_stats = enforce_malware_split_separation(
        dfs_train_val["malware_val"],
        dfs_test["malware_test"],
        len(dfs_test["benign_test"]),
        separation_cfg
    )
    
    print(f"  Exact sample overlap:        {split_separation_stats['exact_overlap']}")
    print(f"  Exact overlap removed:       {split_separation_stats['exact_overlap_removed']}")
    print(f"  Exact overlap after cleanup: {split_separation_stats['exact_overlap_after']}")
    print(f"  Enforce imphash disjoint:    {split_separation_stats['enforce_imphash_disjoint']}")
    print(f"  Family-overlap rows removed: {split_separation_stats['shared_imphash_removed_rows']}")
    print(f"  Shared imphash families:     {split_separation_stats['shared_imphash_unique']}")
    print(f"  Imphash Jaccard similarity:  {split_separation_stats['imphash_jaccard']:.4f}")
    print(f"  Val shared rate:             {split_separation_stats['val_shared_rate']:.4f}")
    print(f"  Test shared rate:            {split_separation_stats['test_shared_rate']:.4f}")
    if split_separation_stats['similarity_threshold'] is not None:
        print(f"  Similarity threshold:        {split_separation_stats['similarity_threshold']:.4f}")
        print(f"  Similarity rows removed:     {split_separation_stats['similarity_removed_rows']}")
        print(
            f"  Max similarity (before/after): "
            f"{split_separation_stats['max_similarity_before']:.4f} / "
            f"{split_separation_stats['max_similarity_after']:.4f}"
        )
    
    if not split_separation_stats.get("sealed_pass", False):
        raise RuntimeError(
            "TEST SET CONTAMINATION DETECTED! "
            f"Exact overlap after cleanup={split_separation_stats['exact_overlap_after']}, "
            f"shared imphash families={split_separation_stats['shared_imphash_unique']}, "
            f"max similarity after={split_separation_stats['max_similarity_after']:.4f}."
        )
    print("  ✓ Malware validation/test sets are independent (sealed test set verified)")
    
    # Update malware_val if cleaned by verification
    dfs_train_val["malware_val"] = malware_val_checked

    # Final step: cap malware_val to a small percentage of benign_val
    print("\n--- Final Step: Malware Validation Size Control ---")
    malware_val_ratio_stats = None
    dfs_train_val["malware_val"], malware_val_ratio_stats = _downsample_malware_val_relative_to_benign_val(
        dfs_train_val["malware_val"],
        benign_val_count=len(dfs_train_val["benign_val"]),
        ratio=malware_val_ratio,
        seed=malware_val_ratio_seed,
        min_samples=malware_val_min_samples,
    )
    print(
        f"  Ratio target (malware_val / benign_val): {malware_val_ratio_stats['ratio']:.4f} "
        f"(seed={malware_val_ratio_seed}, min_samples={malware_val_min_samples})"
    )
    print(
        f"  malware_val: {malware_val_ratio_stats['before']} -> {malware_val_ratio_stats['after']} "
        f"(target={malware_val_ratio_stats['target']}, removed={malware_val_ratio_stats['removed']})"
    )
    if len(dfs_train_val["malware_val"]) == 0:
        raise ValueError("malware_val became empty after ratio downsampling; increase malware_val_ratio_to_benign_val")
    if len(dfs_train_val["malware_val"]) < min_malware_val_after_audit:
        raise RuntimeError(
            "malware_val is too small after Stage 2 audit "
            f"({len(dfs_train_val['malware_val'])} < {min_malware_val_after_audit})."
        )
    
    # Combine all datasets for transformation
    dfs_raw = {**dfs_train_val, **dfs_test}

    print(f"\n--- Saving cleaned datasets ({len(selected_features)} features) ---")
    cleaned_outputs: dict[str, pd.DataFrame] = {}
    for name, df in dfs_raw.items():
        df_clean = apply_transforms(df, selected_features, log_transform_cols)
        cleaned_outputs[name] = df_clean
        out_path = cleaned_dir / f"{model_name}_{name}_clean.parquet"
        df_clean.to_parquet(out_path, engine="pyarrow", compression="snappy")
        print(f"  Saved {name}: {df_clean.shape} -> {out_path}")

    projected_similarity_stats = {
        "enabled": False,
        "similarity_threshold": float(projected_similarity_threshold),
        "before": int(len(cleaned_outputs["malware_val"])),
        "after": int(len(cleaned_outputs["malware_val"])),
        "removed": 0,
        "all_topk_pass": True,
    }

    if projected_pruning_enabled:
        print("\n--- Projected Similarity Pruning (Stage-3 Feature Regime) ---")
        print(
            "  Applying split-hygiene pruning against malware_test before model training; "
            "this does not modify benign training data."
        )

        malware_val_pruned, projected_similarity_stats = _iterative_projected_similarity_pruning(
            benign_train_clean=cleaned_outputs["benign_train"],
            malware_val_clean=cleaned_outputs["malware_val"],
            malware_test_clean=cleaned_outputs["malware_test"],
            top_k_list=projected_top_k_list,
            similarity_threshold=projected_similarity_threshold,
            max_iterations=projected_max_iterations,
        )
        cleaned_outputs["malware_val"] = malware_val_pruned

        malware_val_out_path = cleaned_dir / f"{model_name}_malware_val_clean.parquet"
        malware_val_pruned.to_parquet(malware_val_out_path, engine="pyarrow", compression="snappy")

        print(f"  top-k list: {projected_top_k_list}")
        print(
            f"  malware_val projected pruning: {projected_similarity_stats.get('before', 0)} -> "
            f"{projected_similarity_stats.get('after', 0)} "
            f"(removed={projected_similarity_stats.get('removed', 0)})"
        )
        print(f"  all top-k constraints pass: {projected_similarity_stats.get('all_topk_pass', False)}")

    malware_val_clean = cleaned_outputs["malware_val"]
    if len(malware_val_clean) < min_malware_val_after_audit:
        raise RuntimeError(
            "malware_val is too small after projected similarity pruning "
            f"({len(malware_val_clean)} < {min_malware_val_after_audit})."
        )

    malware_test_clean = cleaned_outputs["malware_test"]
    val_norm = _normalize_rows_l2(malware_val_clean.to_numpy(dtype=np.float32, copy=False))
    test_norm = _normalize_rows_l2(malware_test_clean.to_numpy(dtype=np.float32, copy=False))
    similarity_profile = _compute_cross_similarity_profile(
        val_norm,
        test_norm,
        thresholds=similarity_thresholds,
    )
    knn_metrics = _compute_knn_cross_split_metrics(
        val_norm,
        test_norm,
        k_list=knn_k_list,
        sample_size_per_split=knn_sample_size,
        seed=knn_seed,
    )

    stage2_manifest = _build_stage2_independence_manifest(
        config_path=config_path,
        audit_cfg=audit_cfg,
        split_separation_stats=split_separation_stats,
        malware_val_ratio_stats=malware_val_ratio_stats,
        malware_val_clean=malware_val_clean,
        malware_test_clean=malware_test_clean,
        knn_metrics=knn_metrics,
        similarity_profile=similarity_profile,
        projected_similarity_stats=projected_similarity_stats,
    )
    manifest_out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_out_path.write_text(json.dumps(stage2_manifest, indent=2), encoding="utf-8")
    print(f"\n  Stage-2 independence manifest: {manifest_out_path}")

    if audit_mode == "enforce" and not stage2_manifest["gates"]["all_pass"]:
        raise RuntimeError(
            "Stage-2 independence manifest failed hard gates. "
            f"Run ID={stage2_manifest['run_id']}"
        )

    feature_status = {col: {"status": "removed", "stage": "initial"} for col in all_raw_cols}

    var_removed_names = {r["name"] for r in var_removed}
    for col in all_raw_cols:
        if col in var_removed_names:
            feature_status[col] = {"status": "removed", "stage": "variance_filter"}

    corr_removed_names = {r["dropped"] for r in corr_removed}
    for col in kept_after_var:
        if col in corr_removed_names:
            feature_status[col] = {"status": "removed", "stage": "correlation_prune"}

    stab_removed_names = {r["name"] for r in stab_removed}
    for col in kept_after_corr:
        if col in stab_removed_names:
            feature_status[col] = {"status": "removed", "stage": "stability_filter"}

    for col in selected_features:
        feature_status[col] = {
            "status": "selected_log_transform" if col in log_transform_cols else "selected",
            "stage": "passed_all",
        }

    schema = {
        "schema_version": "1.0",
        "created_date": time.strftime("%Y-%m-%d"),
        "total_raw_features": len(all_raw_cols),
        "selected_features": len(selected_features),
        "feature_order": selected_features,
        "log_transform_features": log_transform_cols,
        "feature_status": feature_status,
        "group_mapping": {c: group_mapping.get(c, "?") for c in selected_features},
        "filtering_parameters": {
            "variance_threshold": variance_threshold,
            "norm_var_threshold": norm_var_threshold,
            "corr_threshold": corr_threshold,
        },
    }
    schema_path = schema_dir / "feature_schema_selected.json"
    with open(schema_path, "w", encoding="utf-8") as fh:
        json.dump(schema, fh, indent=2)
    print(f"\n  Feature schema saved to {schema_path}")

    report = generate_report(
        all_raw_cols,
        kept_after_var,
        var_removed,
        kept_after_corr,
        corr_removed,
        kept_after_stab,
        stab_removed,
        log_transform_cols,
        group_mapping,
        split_separation_stats,
        malware_val_ratio_stats,
        {
            "run_id": stage2_manifest.get("run_id"),
            "manifest_path": str(manifest_out_path),
            "all_pass": stage2_manifest.get("gates", {}).get("all_pass", False),
        },
        projected_similarity_stats,
    )
    report_path = report_dir / "feature_selection_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report saved to {report_path}")

    selected_groups = Counter(group_mapping.get(c, "?") for c in selected_features)
    print(f"\n{'=' * 70}")
    print("FEATURE SELECTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Raw:      {len(all_raw_cols)} features")
    print(f"  Selected: {len(selected_features)} features in {len(selected_groups)} groups")
    print(f"  Log-transformed: {len(log_transform_cols)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
