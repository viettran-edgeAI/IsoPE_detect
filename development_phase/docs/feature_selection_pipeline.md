# Anomaly Detection System for Zero-Day Malware Detection
## Static PE Analysis with Isolation Forest — Feature Selection & Training Pipeline

---

## 1. Feature Selection Pipeline Architecture

### Overview

The pipeline follows a **multi-stage funnel** architecture, progressively narrowing ~625 raw features down to a compact, stable, and discriminative set suitable for Isolation Forest anomaly detection. Each stage has a clear contract: input features → filtering criterion → output features + metadata log.

```
Stage 1: Extraction        → ~625 raw features (all LIEF-derived)
Stage 2: Cleaning          → ~600 features (NaN/inf/type fixes, drop unparseable)
Stage 3: Variance Filter   → ~400 features (remove near-zero-variance)
Stage 4: Correlation Prune → ~250 features (remove redundant pairs)
Stage 5: Stability Filter  → ~200 features (remove unstable distributions)
Stage 6: Model-Based       → ~40-70 features (optional importance ranking)
         ─────────────────
         Final Feature Set  → stored in feature_schema.json + Parquet
```

### Design Rationale

| Stage | Purpose | Why Before Next Stage |
|-------|---------|----------------------|
| **Extraction** | Materialize all possible signals | Maximizes coverage; no information loss |
| **Cleaning** | Ensure numeric validity | Downstream stats (variance, correlation) require clean numerics |
| **Variance** | Remove uninformative constants | Reduces correlation matrix size; constants add noise to Isolation Forest splits |
| **Correlation** | Remove redundant signals | Isolation Forest randomly selects features; redundant features dilute anomaly signal by wasting split budget |
| **Stability** | Remove noisy/outlier-dominated features | Features with extreme CV or heavy-tail outliers create spurious isolation paths that inflate FPR |
| **Model-Based** | Final discriminative ranking | Validates remaining features actually contribute to anomaly separation on validation set |

### Key Principle: Benign-Only Training

All filtering stages (3–6) operate exclusively on the **benign training set** (15,400 samples). The validation set (1,500 benign) is held out for threshold tuning and optional Stage 6 importance estimation. Malware data is never seen during feature selection or training — only during final evaluation.

---

## 2. Stage 1: Extraction — All Raw Features from LIEF

### 2.1 Feature Groups Summary

| Group | Prefix | Count | Description |
|-------|--------|-------|-------------|
| DOS Header | `dos_` | 4 | magic, e_lfanew, checksum, num_relocs |
| COFF Header | `coff_` | 12 | machine, num_sections, timestamp, sizeof_opt_header, characteristics (raw) + 6 boolean flags |
| Optional Header | `opt_` | 36 | magic, linker versions, sizes, entrypoint, imagebase, alignment, versions, subsystem, dll_characteristics + 11 boolean DLL char flags, checksum_matches |
| Sections (per-section) | `sec{i}_` | 80 | 8 features × 10 section slots |
| Sections (aggregate) | `sec_` | 7 | mean/max/min/std entropy, total sizes, mean ratio |
| Sections (name hash) | `sec_name_hash_` | 32 | Feature-hashed section names |
| Sections (counts) | `num_*_sections` | 3 | exec, write, rwx counts |
| Imports (scalar) | `imp_` | 5+2 | counts + imphash + suspicious count |
| Imports (DLL hash) | `imp_dll_hash_` | 64 | Feature-hashed DLL names |
| Imports (func hash) | `imp_func_hash_` | 256 | Feature-hashed function names |
| Exports | `exp_` | 4+32 | scalar + name hash |
| Resources | `rsrc_` | 15 | Counts, sizes, entropy, type flags |
| TLS | `tls_` | 5 | Callback count, sizes, flags |
| Debug | `dbg_/has_` | 7 | Types present, PDB presence |
| Signatures | `sig_` | 4 | Presence, count, verification |
| Relocations | `reloc_` | 3 | Block/entry counts |
| Rich Header | `rich_` | 6 | Key, entries, build stats |
| Load Config | `lc_` | 6 | SEH, CFG, security cookie |
| Overlay | `ovl_` | 4 | Size, ratio, entropy |
| Data Directories | `dd_` | 32 | RVA + size per type |
| Delay Imports | `dimp_` | 2 | Presence, count |
| File Metadata | `file_size` | 1 | File size |
| **TOTAL** | | **~625** | |

### 2.2 Extraction Rules

- Parse with `lief.PE.parse(filepath)`; if None → skip, log to failures
- Each feature group wrapped in try/except (one group failure ≠ total failure)
- Sections: pad/truncate to exactly 10 slots
- Import/Export names: feature hashing to fixed-size vectors
- Output: Parquet format per split

---

## 3. Stage 2: Cleaning

| Step | Operation | Detail |
|------|-----------|--------|
| 3.1 | Replace inf/-inf with NaN | `df.replace([np.inf, -np.inf], np.nan)` |
| 3.2 | Drop columns >50% NaN | Unreliable parsing |
| 3.3 | Impute NaN | Binary→0, Continuous→median(train), Hash→0.0 |
| 3.4 | Type coercion | All → float32 |
| 3.5 | Drop constant columns | `nunique() == 1` |
| 3.6 | Clip extremes | [Q0.1, Q99.9] percentiles from train |

Output: `cleaning_config.json` with imputation values, clip bounds, dropped columns.

---

## 4. Stage 3: Variance Filtering

| Feature Type | Metric | Threshold |
|-------------|--------|-----------|
| Binary flags | Minority class ratio | < 0.5% |
| Hash vector | Nonzero fraction | < 1% |
| Continuous | Normalized variance | < 1e-6 |

**Expected: ~350-450 features remain**

---

## 5. Stage 4: Correlation Pruning

- Threshold: |Pearson correlation| > 0.95
- Strategy: drop the feature with more correlated partners (highest graph degree)
- Tie-breaking: keep higher-priority group, higher variance, simpler name
- Special: drop `_rva` columns when correlated with `_size` counterparts

**Expected: ~250-300 features remain**

---

## 6. Stage 5: Stability Filtering

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Coefficient of Variation | CV > 50 | Mostly noise |
| Outlier Fraction | > 5% beyond Q1/Q3 ± 5·IQR | Heavy-tailed |
| Excess Kurtosis | > 100 | Extreme tails |
| Fill Rate (non-binary) | < 10% non-default | Sparse feature |

**Log-transform rescue**: Before removing for high CV/kurtosis, check if `log1p(abs(x))` normalizes. Flag for transform instead of removal.

**Expected: ~200-250 features remain**

---

## 7. Stage 6: Model-Based Selection (Optional)

### Permutation Importance
1. Train IF on all remaining features
2. For each feature: shuffle → re-score → measure AUC drop
3. Drop features with importance < 0

### Feature Group Ablation
- Remove each group → measure AUC drop
- High impact (>0.5%): KEEP
- Low impact (<0.5%) + high C++ complexity: REMOVE

**Expected final: ~120-200 features**

---

## 8. Model Training & Evaluation

### Hyperparameters

| Parameter | Initial | Search Range |
|-----------|---------|--------------|
| n_estimators | 200 | {100, 200, 300, 500, 1000} |
| max_samples | 0.7 | {256, 0.4, 0.7, 0.9} |
| max_features | 1.0 | {0.5, 0.7, 1.0} |

### Threshold Tuning
- Use benign validation set (1,500 samples)
- Find threshold t* such that FPR ≤ 2%
- t* = percentile(scores_val, 2) as starting point

### Evaluation Metrics
| Metric | Computed On | Target |
|--------|------------|--------|
| FPR | benign_val (tune), benign_test (verify) | ≤ 2% |
| TPR | malware_test | >85% |
| AUC-ROC | benign_test + malware_test | >0.90 |

---

## 9. Iterative Refinement

```
OUTER LOOP (max 10 iterations):
  INNER LOOP: Grid search hyperparameters
  ABLATION: Test each feature group removal
  CONVERGENCE: |ΔAUC| < 0.001 for 2 iterations → STOP
```

### Decision Matrix for Feature Groups
```
                   AUC Impact
                   High (>0.5%)    Low (<0.5%)
C++ Complexity  ┌──────────────┬──────────────┐
  Low           │  KEEP        │  KEEP        │
                ├──────────────┼──────────────┤
  High          │  KEEP        │  REMOVE      │
                └──────────────┴──────────────┘
```

---

## 10. Data Splits Protocol

| Set | Size | Usage |
|-----|------|-------|
| Benign Train | 32000 | Feature selection + IF training |
| Benign Val | 5000 | Threshold tuning, hyperparameter selection |
| Malware Val | 4200 | Threshold tuning, hyperparametere selection |
| Benign Test | 2,800 | Final FPR verification (never tuned on) |
| Malware Test | 2,500 | TPR/AUC evaluation (never trained on) |

**Golden Rules:**
1. Feature filtering thresholds → Benign Train ONLY
2. Imputation values → Benign Train ONLY
3. Threshold t* → Benign Val ONLY
4. AUC/TPR → Benign Test + Malware Test
5. If Benign Test FPR > 2.5% → threshold overfit, retune

---

## 11. Directory Structure

```
data/
├── raw/                          # Stage 1 output
├── cleaned/                      # Stage 2 output
├── filtered/                     # Stages 3-5 output
└── final/                        # Final feature set
schemas/
├── feature_schema.json           # Feature group registry
└── feature_schema_final.json     # Locked after convergence
artifacts/
├── model_v{X}.joblib
├── threshold_v{X}.json
├── cleaning_config.json
└── evaluation_report_v{X}.json
```
