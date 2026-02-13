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
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as sp_kurtosis


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
        rep_indices = reps.index.to_numpy()
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
    cfg = separation_cfg or {}
    if not cfg.get("enabled", True):
        return malware_val_df, {"enabled": False, "reason": "disabled_by_config", "val_before": int(len(malware_val_df)), "val_after": int(len(malware_val_df))}

    min_ratio = float(cfg.get("min_val_ratio_to_benign_test", 0.05))
    max_ratio = float(cfg.get("max_val_ratio_to_benign_test", 0.08))
    near_dup_cosine_threshold = float(cfg.get("near_dup_cosine_threshold", 0.995))
    random_seed = int(cfg.get("random_seed", 42))

    val_work = malware_val_df.copy()
    test_work = malware_test_df.copy()

    stats = {
        "enabled": True,
        "val_before": int(len(val_work)),
        "test_count": int(len(test_work)),
        "exact_removed": 0,
        "imphash_removed": 0,
        "near_removed": 0,
        "downsample_removed": 0,
        "near_method": "none",
        "near_threshold": near_dup_cosine_threshold,
    }

    # 1) Exact cross-split duplicate removal (target: 0 overlap)
    val_sig = pd.util.hash_pandas_object(val_work, index=False)
    test_sig_set = set(pd.util.hash_pandas_object(test_work, index=False).astype(np.uint64).tolist())
    exact_mask = ~val_sig.astype(np.uint64).isin(test_sig_set)
    stats["exact_removed"] = int((~exact_mask).sum())
    val_work = val_work.loc[exact_mask].copy()

    # 2) Near-duplicate control signal (prefer ssdeep/tlsh if available; fallback to hash-vector cosine)
    near_score = pd.Series(np.zeros(len(val_work), dtype=np.float32), index=val_work.index)
    near_flag = pd.Series(np.zeros(len(val_work), dtype=bool), index=val_work.index)
    ssdeep_col = "ssdeep" if "ssdeep" in val_work.columns and "ssdeep" in test_work.columns else None
    tlsh_col = "tlsh" if "tlsh" in val_work.columns and "tlsh" in test_work.columns else None

    if ssdeep_col is not None:
        try:
            import ssdeep  # type: ignore

            test_sigs = [str(x) for x in test_work[ssdeep_col].dropna().astype(str).tolist() if x]
            if test_sigs:
                for idx, sig in val_work[ssdeep_col].dropna().astype(str).items():
                    max_score = max(ssdeep.compare(sig, ts) for ts in test_sigs)
                    near_score.at[idx] = max_score / 100.0
                    if max_score >= 90:
                        near_flag.at[idx] = True
                stats["near_method"] = "ssdeep"
        except Exception:
            pass

    if tlsh_col is not None and stats["near_method"] == "none":
        try:
            import tlsh  # type: ignore

            test_hashes = [str(x) for x in test_work[tlsh_col].dropna().astype(str).tolist() if x and x != "TNULL"]
            if test_hashes:
                for idx, sig in val_work[tlsh_col].dropna().astype(str).items():
                    if not sig or sig == "TNULL":
                        continue
                    min_dist = min(tlsh.diff(sig, t) for t in test_hashes)
                    near_score.at[idx] = 1.0 / (1.0 + float(min_dist))
                    if min_dist <= 20:
                        near_flag.at[idx] = True
                stats["near_method"] = "tlsh"
        except Exception:
            pass

    if stats["near_method"] == "none":
        near_cols = [
            c
            for c in val_work.columns
            if c == "imphash"
            or c.startswith("imp_dll_hash_")
            or c.startswith("imp_func_hash_")
            or c.startswith("exp_func_hash_")
            or c.startswith("sec_name_hash_")
        ]
        near_cols = [c for c in near_cols if c in test_work.columns]
        if near_cols:
            val_mat = _normalize_rows_l2(_extract_numeric_matrix(val_work, near_cols))
            test_mat = _normalize_rows_l2(_extract_numeric_matrix(test_work, near_cols))
            max_sim = _max_cosine_to_reference(val_mat, test_mat)
            near_score = pd.Series(max_sim, index=val_work.index)
            near_flag = near_score >= near_dup_cosine_threshold
            stats["near_method"] = "hash_vector_cosine"

    # 3) Family overlap signal (imphash proxy)
    imphash_shared_flag = pd.Series(np.zeros(len(val_work), dtype=bool), index=val_work.index)
    if "imphash" in val_work.columns and "imphash" in test_work.columns:
        val_imphash = pd.to_numeric(val_work["imphash"], errors="coerce").fillna(0).astype(np.int64)
        test_imphash = pd.to_numeric(test_work["imphash"], errors="coerce").fillna(0).astype(np.int64)
        shared = set(val_imphash[val_imphash != 0].tolist()) & set(test_imphash[test_imphash != 0].tolist())
        imphash_shared_flag = val_imphash.isin(shared)

    # 4) Risk-aware selection to keep val size in configured budget while minimizing overlap
    min_target = max(1, int(round(min_ratio * benign_test_count)))
    max_target = max(min_target, int(round(max_ratio * benign_test_count)))

    before_select = len(val_work)
    target_size = min(before_select, max_target)
    if target_size > 0 and before_select > target_size:
        risk = near_score.astype(np.float64) + imphash_shared_flag.astype(np.float64) * 1.5
        tie_breaker = np.random.default_rng(random_seed).uniform(0.0, 1e-6, size=before_select)
        rank_df = pd.DataFrame(
            {
                "idx": val_work.index.to_numpy(),
                "risk": risk.to_numpy(),
                "near_score": near_score.to_numpy(),
                "shared_imphash": imphash_shared_flag.astype(np.int8).to_numpy(),
                "tie": tie_breaker,
            }
        ).sort_values(["risk", "near_score", "shared_imphash", "tie"], ascending=[True, True, True, True])

        keep_idx = rank_df.head(target_size)["idx"].to_numpy()
        removed_idx = set(val_work.index.to_numpy()) - set(keep_idx.tolist())
        val_work = val_work.loc[keep_idx].copy()

        removed_mask = rank_df["idx"].isin(list(removed_idx))
        stats["near_removed"] = int(((rank_df.loc[removed_mask, "near_score"] >= near_dup_cosine_threshold)).sum())
        stats["imphash_removed"] = int((rank_df.loc[removed_mask, "shared_imphash"] > 0).sum())
        stats["downsample_removed"] = int(before_select - len(val_work))
    else:
        stats["near_removed"] = int(near_flag.sum())
        stats["imphash_removed"] = int(imphash_shared_flag.sum())
        stats["downsample_removed"] = 0

    stats["val_after"] = int(len(val_work))
    stats["target_min"] = int(min_target)
    stats["target_max"] = int(max_target)
    stats["actual_ratio_to_benign_test"] = float(len(val_work) / max(1, benign_test_count))

    if "imphash" in val_work.columns and "imphash" in test_work.columns:
        val_imphash = pd.to_numeric(val_work["imphash"], errors="coerce").fillna(0).astype(np.int64)
        test_imphash = pd.to_numeric(test_work["imphash"], errors="coerce").fillna(0).astype(np.int64)
        val_nonzero = set(val_imphash[val_imphash != 0].tolist())
        test_nonzero = set(test_imphash[test_imphash != 0].tolist())
        shared = val_nonzero & test_nonzero
        stats["imphash_shared_unique_after"] = int(len(shared))
        stats["imphash_jaccard_after"] = float(len(shared) / len(val_nonzero | test_nonzero)) if (val_nonzero or test_nonzero) else 0.0

    return val_work, stats

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
                "## Malware Validation/Test Separation\n",
                f"- Validation size before control: **{split_separation_stats.get('val_before', 0)}**",
                f"- Validation size after control: **{split_separation_stats.get('val_after', 0)}**",
                f"- Exact duplicates removed: **{split_separation_stats.get('exact_removed', 0)}**",
                f"- Near-duplicates removed ({split_separation_stats.get('near_method', 'none')}): **{split_separation_stats.get('near_removed', 0)}**",
                f"- Shared imphash removed: **{split_separation_stats.get('imphash_removed', 0)}**",
                f"- Downsample removed: **{split_separation_stats.get('downsample_removed', 0)}**",
                f"- Target range (5-8% benign test): **[{split_separation_stats.get('target_min', 0)}, {split_separation_stats.get('target_max', 0)}]**",
                f"- Actual validation ratio to benign test: **{split_separation_stats.get('actual_ratio_to_benign_test', 0.0):.4f}**",
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
    separation_cfg = cfg.get("separation_control", {})

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

    print("\nLoading raw datasets...")
    dfs_raw = {
        "benign_train": pd.read_parquet(train_path),
        "benign_val": pd.read_parquet(val_path),
        "benign_test": pd.read_parquet(test_b_path),
        "malware_val": pd.read_parquet(val_m_path),
        "malware_test": pd.read_parquet(test_m_path),
    }

    # Use benign_train columns as canonical raw order
    all_raw_cols = list(dfs_raw["benign_train"].columns)
    print(f"  Raw features: {len(all_raw_cols)}")

    print("\n--- Variance Filtering ---")
    kept_after_var, var_removed = variance_filter(
        dfs_raw["benign_train"],
        group_mapping,
        variance_threshold=variance_threshold,
        norm_var_threshold=norm_var_threshold,
    )
    print(f"  {len(all_raw_cols)} -> {len(kept_after_var)} features (removed {len(var_removed)})")

    print("\n--- Correlation Pruning ---")
    df_var_filtered = dfs_raw["benign_train"][kept_after_var]
    kept_after_corr, corr_removed = correlation_pruning(
        df_var_filtered,
        threshold=corr_threshold,
        group_mapping=group_mapping,
    )
    print(f"  {len(kept_after_var)} -> {len(kept_after_corr)} features (removed {len(corr_removed)})")

    print("\n--- Stability Filtering ---")
    df_corr_filtered = dfs_raw["benign_train"][kept_after_corr]
    kept_after_stab, log_transform_cols, stab_removed = stability_filter(df_corr_filtered, group_mapping)
    print(f"  {len(kept_after_corr)} -> {len(kept_after_stab)} features (removed {len(stab_removed)})")
    print(f"  Log-transform rescued: {len(log_transform_cols)}")

    selected_features = kept_after_stab

    malware_val_cleaned_raw, split_separation_stats = enforce_malware_split_separation(
        malware_val_df=dfs_raw["malware_val"],
        malware_test_df=dfs_raw["malware_test"],
        benign_test_count=len(dfs_raw["benign_test"]),
        separation_cfg=separation_cfg,
    )
    dfs_raw["malware_val"] = malware_val_cleaned_raw

    print("\n--- Malware Split Separation Control ---")
    print(
        "  malware_val: "
        f"{split_separation_stats.get('val_before', len(malware_val_cleaned_raw))} -> "
        f"{split_separation_stats.get('val_after', len(malware_val_cleaned_raw))} "
        f"(exact={split_separation_stats.get('exact_removed', 0)}, "
        f"near={split_separation_stats.get('near_removed', 0)}, "
        f"imphash={split_separation_stats.get('imphash_removed', 0)}, "
        f"downsample={split_separation_stats.get('downsample_removed', 0)})"
    )
    print(
        f"  ratio_to_benign_test={split_separation_stats.get('actual_ratio_to_benign_test', 0.0):.4f} "
        f"(target {split_separation_stats.get('target_min', 0)}-{split_separation_stats.get('target_max', 0)} samples)"
    )

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
