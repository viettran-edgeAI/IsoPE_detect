#!/usr/bin/env python3
"""
Stage 2: Feature Filtering and Selection
=======================================
Load raw PE feature datasets, apply variance/correlation/stability filtering,
and save cleaned datasets for model optimization.

Usage:
    python feature_selection.py [--config PATH] [--corr-threshold X] [--variance-threshold X]

Outputs:
    ../data/cleaned/benign_train_clean.parquet
    ../data/cleaned/benign_val_clean.parquet
    ../data/cleaned/benign_test_clean.parquet
    ../data/cleaned/malware_val_clean.parquet
    ../data/cleaned/malware_test_clean.parquet
    ../schemas/feature_schema_selected.json
    ../reports/feature_selection_report.md
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as sp_kurtosis


def _compute_dataset_fingerprint(df: pd.DataFrame, name: str) -> str:
    """Compute SHA256 fingerprint of a dataset for audit trail."""
    df_hash = pd.util.hash_pandas_object(df, index=False)
    combined = hashlib.sha256()
    combined.update(np.asarray(df_hash.values).tobytes())
    fingerprint = combined.hexdigest()[:16]
    return fingerprint


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

    enforce_imphash_disjoint = bool(separation_cfg.get("enforce_imphash_disjoint", True))
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
    family_ok = (not enforce_imphash_disjoint) or (stats["shared_imphash_unique"] == 0)
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


def _validate_config(cfg: dict) -> None:
    required_top = ["data", "filtering", "outputs"]
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
                f"- Validation size before control: **{split_separation_stats.get('val_before', 0)}**",
                f"- Validation size after audit: **{split_separation_stats.get('val_after', 0)}**",
                f"- Exact duplicate overlap: **{split_separation_stats.get('exact_overlap', 0)}**",
                f"- Exact duplicate removed: **{split_separation_stats.get('exact_overlap_removed', 0)}**",
                f"- Exact duplicate after cleanup: **{split_separation_stats.get('exact_overlap_after', 0)}**",
                f"- Enforce imphash disjoint: **{split_separation_stats.get('enforce_imphash_disjoint', False)}**",
                f"- Rows removed by family disjoint: **{split_separation_stats.get('shared_imphash_removed_rows', 0)}**",
                f"- Shared unique imphash: **{split_separation_stats.get('shared_imphash_unique', 0)}**",
                f"- Imphash Jaccard (unique): **{split_separation_stats.get('imphash_jaccard', 0.0):.4f}**",
                f"- Shared-rate (val/test): **{split_separation_stats.get('val_shared_rate', 0.0):.4f} / {split_separation_stats.get('test_shared_rate', 0.0):.4f}**",
                f"- Similarity threshold: **{split_separation_stats.get('similarity_threshold', None)}**",
                f"- Rows removed by similarity: **{split_separation_stats.get('similarity_removed_rows', 0)}**",
                f"- Max cosine similarity (before/after): **{split_separation_stats.get('max_similarity_before', 0.0):.4f} / {split_separation_stats.get('max_similarity_after', 0.0):.4f}**",
                f"- Sealed-test check (exact overlap == 0): **{split_separation_stats.get('sealed_pass', False)}**",
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

    data_cfg = cfg["data"]
    train_path = _resolve_path(config_path, data_cfg["train_benign_path"])
    val_path = _resolve_path(config_path, data_cfg["val_benign_path"])
    test_b_path = _resolve_path(config_path, data_cfg["test_benign_path"])
    test_m_path = _resolve_path(config_path, data_cfg["test_malware_path"])
    val_m_path = _resolve_path(config_path, data_cfg["val_malware_path"])

    filt_cfg = cfg["filtering"]
    variance_threshold = args.variance_threshold if args.variance_threshold is not None else float(filt_cfg.get("variance_threshold", 0.0))
    corr_threshold = args.corr_threshold if args.corr_threshold is not None else float(filt_cfg.get("corr_threshold", 0.95))
    norm_var_threshold = args.norm_var_threshold if args.norm_var_threshold is not None else float(filt_cfg.get("norm_var_threshold", 1e-7))

    outputs_cfg = cfg["outputs"]
    cleaned_dir = _resolve_path(config_path, outputs_cfg.get("cleaned_data_dir", "../data/cleaned"))
    schema_dir = _resolve_path(config_path, outputs_cfg.get("schema_dir", "../schemas"))
    report_dir = _resolve_path(config_path, outputs_cfg.get("report_dir", "../reports"))

    separation_cfg = cfg.get("separation", {})
    malware_val_ratio = float(separation_cfg.get("malware_val_ratio_to_benign_val", 0.03))
    malware_val_ratio_seed = int(separation_cfg.get("malware_val_ratio_seed", 42))
    malware_val_min_samples = int(separation_cfg.get("malware_val_min_samples", 500))

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
    
    # Dataset fingerprinting for audit trail
    print("\n--- Dataset Fingerprinting (Audit Trail) ---")
    train_fp = _compute_dataset_fingerprint(dfs_train_val["benign_train"], "train")
    val_fp = _compute_dataset_fingerprint(dfs_train_val["benign_val"], "val_benign")
    val_m_fp = _compute_dataset_fingerprint(dfs_train_val["malware_val"], "val_malware")
    print(f"  Training (benign):      {train_fp}")
    print(f"  Validation (benign):    {val_fp}")
    print(f"  Validation (malware):   {val_m_fp}")
    
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
    
    # Test data fingerprinting
    test_b_fp = _compute_dataset_fingerprint(dfs_test["benign_test"], "test_benign")
    test_m_fp = _compute_dataset_fingerprint(dfs_test["malware_test"], "test_malware")
    print(f"  Test (benign) fingerprint:  {test_b_fp}")
    print(f"  Test (malware) fingerprint: {test_m_fp}")
    
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
    
    # Combine all datasets for transformation
    dfs_raw = {**dfs_train_val, **dfs_test}

    print(f"\n--- Saving cleaned datasets ({len(selected_features)} features) ---")
    for name, df in dfs_raw.items():
        df_clean = apply_transforms(df, selected_features, log_transform_cols)
        out_path = cleaned_dir / f"{name}_clean.parquet"
        df_clean.to_parquet(out_path, engine="pyarrow", compression="snappy")
        print(f"  Saved {name}: {df_clean.shape} -> {out_path}")

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
