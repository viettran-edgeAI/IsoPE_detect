# Iterative Optimization Plan for Malware Detection System

**Date:** February 9, 2026  
**Goal:** Increase TPR from 7.88% to >50% while maintaining FPR ≤ 2%  
**Approach:** Fast iteration on feature selection + hyperparameters after ONE-TIME raw feature extraction

---

## 🎯 Problem Analysis

**Current Baseline Issues:**
- **TPR 7.88%** = catching only ~197 out of 2,500 malware samples (CRITICAL FAILURE)
- **FPR 1.93%** = ~97 false alarms out of 5,000 benign (ACCEPTABLE)
- **Model is TOO CONSERVATIVE** - threshold too tight, missing most malware

**Root Causes:**
1. **Contamination too low** - Isolation Forest trained on benign data expects very few anomalies
2. **Feature selection possibly too aggressive** (623 → 487 features)
3. **Hyperparameters may favor precision over recall**
4. **Threshold might be suboptimal for our imbalanced scenario**

---

## 📊 Phase 0: ONE-TIME Raw Feature Extraction

**DO THIS ONCE** (slow, ~30-60 minutes):
```python
# Extract 623 raw features from ALL samples
extract_features(benign_train=32000, benign_val=5000, 
                 benign_test=5000, malware_test=2500)
# Save: raw_features_train.csv, raw_features_val.csv, 
#       raw_features_test_benign.csv, raw_features_test_malware.csv
```

**Output:** ~44,500 samples × 623 features (save to `data/raw/`)

---

## 🔄 Iterative Optimization Loop Structure

Each iteration takes ~1-2 minutes:
1. Select features using strategy X
2. Train Isolation Forest with params Y
3. Evaluate on validation set
4. If promising, evaluate on test set
5. Log results, visualize, decide next step

```python
# Main loop pseudocode
results = []
for experiment in experiments:
    # Feature selection (< 5 seconds)
    X_train_selected = select_features(X_train_raw, strategy=experiment.feature_strategy)
    X_val_selected = select_features(X_val_raw, strategy=experiment.feature_strategy)
    
    # Model training (< 30 seconds)
    model = IsolationForest(**experiment.hyperparameters)
    model.fit(X_train_selected)
    
    # Evaluation (< 10 seconds)
    scores_val_benign = model.score_samples(X_val_selected)
    scores_val_malware = model.score_samples(X_test_malware_selected)
    
    # Compute metrics
    metrics = compute_metrics(scores_val_benign, scores_val_malware, 
                             fpr_target=0.02)
    results.append({**experiment, **metrics})
    
    # Visualize & decide
    plot_roc_curve(scores_val_benign, scores_val_malware)
    if metrics['fpr'] <= 0.02 and metrics['tpr'] > best_tpr:
        best_model = model
        best_tpr = metrics['tpr']
```

---

## 🎨 Part 1: Feature Selection Strategies (7 approaches)

### Strategy 1: **Baseline Reproduction** (sanity check)
- **Purpose:** Verify we can reproduce 7.88% TPR with current setup
- **Features:** 487 (variance > 0.05, correlation < 0.95, stability pass)
- **Expected TPR:** ~7-8%
- **Time:** 1 min

### Strategy 2: **No Filtering (All Features)**
- **Purpose:** See if aggressive filtering hurt performance
- **Features:** All 623 raw features
- **Rationale:** Maybe we threw away discriminative features
- **Expected TPR:** 10-15% (more features = more signal?)
- **Time:** 2 min

### Strategy 3: **Relaxed Filtering**
- **Purpose:** Less aggressive thresholds
- **Features:** ~550-600 features
- **Thresholds:**
  - Variance > 0.01 (was 0.05)
  - Correlation < 0.98 (was 0.95)
  - Stability: relax pass criteria
- **Expected TPR:** 12-18%
- **Time:** 1 min

### Strategy 4: **Feature Group Ablation (Top Performers)**
- **Purpose:** Use only feature groups that showed highest importance
- **Approach:** Based on your ablation results, select top 3-5 groups
- **Example groups to try:**
  - Import table features (imports are malware indicators)
  - Section characteristics (packed/obfuscated sections)
  - Header anomalies (malformed PE headers)
- **Features:** ~200-300 (focused subset)
- **Expected TPR:** 15-25% (domain-targeted features)
- **Time:** 1 min

### Strategy 5: **Model-Based Feature Importance**
- **Purpose:** Let Isolation Forest tell us which features matter
- **Method:**
  1. Train IF on all 623 features
  2. Compute contamination-based feature importance (variance of scores)
  3. Keep top K features (K = 200, 300, 400, 500)
- **Expected TPR:** 15-30%
- **Time:** 2 min per K value

### Strategy 6: **Statistical Relevance Filtering**
- **Purpose:** Features that differ most between benign training and known anomaly patterns
- **Method:**
  1. Use a small labeled malware subset (if available) or synthetic anomalies
  2. Compute mutual information or statistical divergence
  3. Keep top 300-400 features with highest divergence
- **Expected TPR:** 20-35%
- **Time:** 2 min

### Strategy 7: **Hybrid: Variance + Group Priority**
- **Purpose:** Balance statistical filtering with domain knowledge
- **Method:**
  1. Apply minimal variance filter (> 0.02)
  2. Prioritize high-value groups (imports, sections, headers)
  3. Add correlated features if from different groups
- **Features:** ~450 features
- **Expected TPR:** 18-30%
- **Time:** 1 min

---

## ⚙️ Part 2: Hyperparameter Search Strategy

### Critical Parameters for TPR Improvement:

#### **A. Contamination** (MOST IMPORTANT for TPR)
Current: likely 0.01-0.05 (too conservative)

**Experiments:**
- contamination = 0.10 (expect more anomalies)
- contamination = 0.15
- contamination = 0.20
- contamination = 0.25

**Rationale:** Higher contamination = lower anomaly threshold = more malware caught BUT higher FPR. Need to find sweet spot where FPR ≤ 2%.

**Expected Impact:** TPR could jump from 7% → 25-40% with optimal contamination

---

#### **B. Number of Trees (n_estimators)**
Current: 100

**Experiments:**
- n_estimators = 200 (moderate increase)
- n_estimators = 500 (strong ensemble)
- n_estimators = 1000 (maximum stability)

**Rationale:** More trees = more stable anomaly scores, better generalization

**Expected Impact:** TPR +2-5%, FPR -0.2-0.5%

---

#### **C. Sampling Strategy (max_samples)**
Current: 0.5

**Experiments:**
- max_samples = 256 (fixed small sample)
- max_samples = 512
- max_samples = 0.3 (30% of training data)
- max_samples = 0.7 (70% of training data)
- max_samples = 'auto' (min(256, n_samples))

**Rationale:** Smaller samples = more diverse trees = better anomaly detection

**Expected Impact:** TPR +3-8%, depends on dataset structure

---

#### **D. Feature Subsampling (max_features)**
Current: 0.7

**Experiments:**
- max_features = 0.5 (more diversity)
- max_features = 0.9 (more features per tree)
- max_features = 1.0 (all features)
- max_features = sqrt(n_features) (RF-style)

**Rationale:** Balance between diversity and information

**Expected Impact:** TPR +1-4%

---

#### **E. Bootstrap**
Current: False

**Experiments:**
- bootstrap = True (with replacement)

**Rationale:** Bootstrap can increase diversity in trees

**Expected Impact:** TPR +2-5%

---

### Recommended Hyperparameter Combinations:

```python
# Grid search configurations (sorted by priority)
param_grid = [
    # Experiment 1: Aggressive contamination search
    {
        'contamination': [0.10, 0.15, 0.20, 0.25],
        'n_estimators': [100],
        'max_samples': [0.5],
        'max_features': [0.7],
        'bootstrap': [False]
    },
    
    # Experiment 2: Optimize sampling with best contamination
    {
        'contamination': [best_from_exp1],
        'n_estimators': [200, 500],
        'max_samples': [256, 512, 0.3, 0.7],
        'max_features': [0.7],
        'bootstrap': [False, True]
    },
    
    # Experiment 3: Feature diversity
    {
        'contamination': [best_from_exp1],
        'n_estimators': [best_from_exp2],
        'max_samples': [best_from_exp2],
        'max_features': [0.5, 0.7, 0.9, 1.0],
        'bootstrap': [best_from_exp2]
    },
    
    # Experiment 4: Strong ensemble
    {
        'contamination': [best_from_exp1],
        'n_estimators': [1000],
        'max_samples': [best_from_exp2],
        'max_features': [best_from_exp3],
        'bootstrap': [True]
    }
]
```

---

## 📈 Part 3: Evaluation Metrics & Decision Criteria

### Primary Metrics:
1. **TPR (True Positive Rate)** - % of malware detected
   - Current: 7.88%
   - Target: **≥ 50%** (catch at least half of malware)
   - Stretch goal: **≥ 70%**

2. **FPR (False Positive Rate)** - % of benign flagged as malware
   - Current: 1.93%
   - **Hard constraint: ≤ 2%**

3. **ROC AUC** - Overall discrimination ability
   - Current: 0.727
   - Target: **≥ 0.80** (but secondary to TPR/FPR)

### Secondary Metrics:
- **Precision** = TP / (TP + FP)
- **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)
- **Detection Rate at FPR=1%** (stricter threshold)
- **Detection Rate at FPR=5%** (looser threshold for comparison)

### Success Criteria:

**Tier 1 (Minimum Viable):**
- TPR ≥ 40%
- FPR ≤ 2%
- AUC ≥ 0.75

**Tier 2 (Target):**
- TPR ≥ 50%
- FPR ≤ 2%
- AUC ≥ 0.80

**Tier 3 (Excellent):**
- TPR ≥ 70%
- FPR ≤ 2%
- AUC ≥ 0.85

### When to Stop Iterating:

1. **Success:** Achieve Tier 2 or better
2. **Plateau:** Last 3 experiments show < 2% TPR improvement
3. **Overfitting:** Validation TPR >> Test TPR (> 10% gap)
4. **Budget:** After 15 experiments or 4 hours of compute time
5. **Diminishing Returns:** Cost of further tuning > benefit

---

## 🔢 Part 4: Iteration Order & Experiment Plan

### **Phase 1: Quick Wins (Experiments 1-4, ~10 min total)**

**Goal:** Identify if contamination or feature set is the bottleneck

#### Experiment 1: Baseline Reproduction
- Features: 487 (original filtering)
- Params: n_estimators=100, max_samples=0.5, max_features=0.7, contamination=0.10
- **Decision:** Is contamination the main issue?

#### Experiment 2: Contamination Sweep (Critical)
- Features: 487
- Params: Same as baseline, but contamination in [0.15, 0.20, 0.25]
- **Decision:** Find optimal contamination that gives FPR ≈ 1.8-2.0%

#### Experiment 3: All Features + Best Contamination
- Features: 623 (no filtering)
- Params: Use best contamination from Exp 2
- **Decision:** Did feature filtering hurt us?

#### Experiment 4: Relaxed Filtering + Best Contamination
- Features: ~550 (relaxed thresholds)
- Params: Use best contamination from Exp 2
- **Decision:** Is there a feature sweet spot?

**Phase 1 Decision Point:**
- If TPR > 40%: Continue to Phase 2 (hyperparameter tuning)
- If TPR still < 20%: Focus on feature engineering (Phase 2B)

---

### **Phase 2A: Hyperparameter Optimization (Experiments 5-9, ~15 min)**

**Prerequisites:** Best contamination + feature set from Phase 1

#### Experiment 5: More Trees
- Features: Best from Phase 1
- Params: n_estimators=200, then 500, best contamination
- **Decision:** Is stability improved?

#### Experiment 6: Sampling Strategy
- Features: Best from Phase 1
- Params: Try max_samples in [256, 512, 0.3, 0.7], n_estimators=200
- **Decision:** Optimal subsample size?

#### Experiment 7: Bootstrap Enabled
- Features: Best from Phase 1  
- Params: bootstrap=True, best params from Exp 5-6
- **Decision:** Does bootstrap help?

#### Experiment 8: Feature Diversity
- Features: Best from Phase 1
- Params: max_features in [0.5, 0.9], best params from Exp 5-7
- **Decision:** Optimal feature sampling?

#### Experiment 9: Strong Ensemble
- Features: Best from Phase 1
- Params: n_estimators=1000, all optimal params from Exp 5-8
- **Decision:** Final performance check

**Phase 2A Decision Point:**
- If TPR ≥ 50% & FPR ≤ 2%: SUCCESS, go to validation
- If TPR 35-50%: Try Phase 2B (feature strategies)
- If TPR < 35%: Serious reconsideration needed

---

### **Phase 2B: Advanced Feature Strategies (Experiments 10-14, ~20 min)**

**Run this if Phase 2A didn't hit target**

#### Experiment 10: Feature Group Ablation
- Features: Top 3 groups only (~200-300 features)
- Params: Best from Phase 2A
- **Decision:** Are we diluting signal with noise?

#### Experiment 11: Model-Based Selection (Top 300)
- Features: Top 300 by IF importance
- Params: Best from Phase 2A
- **Decision:** Let the model choose features

#### Experiment 12: Model-Based Selection (Top 400)
- Features: Top 400 by IF importance
- Params: Best from Phase 2A

#### Experiment 13: Statistical Relevance Filter
- Features: Top 350 by mutual information
- Params: Best from Phase 2A

#### Experiment 14: Hybrid Strategy
- Features: Variance + Group priority (~450)
- Params: Best from Phase 2A

**Phase 2B Decision Point:**
- Select best performing feature strategy
- If improvement > 5% TPR, proceed to Phase 3
- Otherwise, analyze failure modes

---

### **Phase 3: Threshold Optimization & Ensemble (Experiments 15-17, ~15 min)**

#### Experiment 15: Optimal Threshold Search
- Model: Best model from Phase 2
- Method: Grid search thresholds for FPR = 1%, 1.5%, 2%
- **Goal:** Maximize TPR at each FPR level

#### Experiment 16: Multi-Model Soft Voting
- Models: Top 3 models from all experiments
- Method: Average anomaly scores, find optimal threshold
- **Goal:** Ensemble for stability

#### Experiment 17: Calibrated Threshold
- Model: Best single model
- Method: Isotonic or Platt scaling on validation set
- **Goal:** Better probability estimates

---

### **Phase 4: Final Validation (Experiment 18, ~5 min)**

#### Experiment 18: Test Set Evaluation
- Model: Best model from Phase 3
- Evaluation: Full test set (5K benign + 2.5K malware)
- Metrics: All metrics, confusion matrix, per-family analysis
- **Goal:** Confirm generalization, no overfitting

---

## 💾 Part 5: Code Structure & Experiment Tracking

### Recommended Notebook Structure:

```markdown
## Iterative Optimization Notebook

### 1. Setup & Configuration
- Imports
- Paths
- Utility functions (load_data, select_features, train_model, evaluate)

### 2. Load Raw Features (ONE TIME)
- Load all 4 datasets
- Basic EDA
- Save to memory

### 3. Experiment Loop
- Define experiment configurations
- For each experiment:
  - Feature selection
  - Model training
  - Evaluation
  - Logging
  - Visualization

### 4. Results Analysis
- Comparison table
- TPR vs FPR scatter plot
- Feature importance analysis
- Best model selection

### 5. Final Validation
- Test set evaluation
- Detailed metrics
- Confusion matrix
- Save final model
```

---

### Results Tracking DataFrame:

```python
import pandas as pd

results_df = pd.DataFrame(columns=[
    'experiment_id',
    'experiment_name',
    'phase',
    'n_features',
    'feature_strategy',
    'n_estimators',
    'max_samples',
    'max_features',
    'contamination',
    'bootstrap',
    'val_tpr',
    'val_fpr',
    'val_auc',
    'val_precision',
    'val_f1',
    'threshold',
    'training_time_sec',
    'notes'
])

# After each experiment
results_df = results_df.append({
    'experiment_id': exp_id,
    'experiment_name': 'Contamination Sweep - 0.15',
    'phase': 1,
    'n_features': 487,
    'feature_strategy': 'baseline_filtering',
    'n_estimators': 100,
    'max_samples': 0.5,
    'max_features': 0.7,
    'contamination': 0.15,
    'bootstrap': False,
    'val_tpr': tpr,
    'val_fpr': fpr,
    'val_auc': auc,
    'val_precision': precision,
    'val_f1': f1,
    'threshold': threshold,
    'training_time_sec': time_elapsed,
    'notes': 'Significant TPR improvement over baseline'
}, ignore_index=True)

# Save after each experiment
results_df.to_csv('iteration_results.csv', index=False)
```

---

### Key Visualizations to Generate:

#### 1. **TPR vs FPR Trade-off Plot**
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(results_df['val_fpr'], results_df['val_tpr'], 
           c=results_df['experiment_id'], s=100, alpha=0.7, cmap='viridis')
plt.axvline(x=0.02, color='red', linestyle='--', label='FPR Target (2%)')
plt.axhline(y=0.50, color='green', linestyle='--', label='TPR Target (50%)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Experiment Results: TPR vs FPR')
plt.colorbar(label='Experiment ID')
plt.legend()
plt.grid(True, alpha=0.3)
```

#### 2. **ROC Curves Comparison**
```python
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(10, 8))
for exp_id in top_5_experiments:
    fpr, tpr, _ = roc_curve(y_true, scores[exp_id])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Exp {exp_id} (AUC={roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlim([0.0, 0.1])  # Zoom to low FPR region
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Top 5 Models')
plt.legend()
plt.grid(True, alpha=0.3)
```

#### 3. **Feature Importance Heatmap**
```python
import seaborn as sns

# For top 3 models, extract feature importances
feature_importances = pd.DataFrame({
    'feature': feature_names,
    'exp_5': importances_exp5,
    'exp_9': importances_exp9,
    'exp_12': importances_exp12
})

top_features = feature_importances.nlargest(30, 'exp_5')  # Top 30

plt.figure(figsize=(12, 10))
sns.heatmap(top_features[['exp_5', 'exp_9', 'exp_12']].T, 
           xticklabels=top_features['feature'],
           yticklabels=['Exp 5', 'Exp 9', 'Exp 12'],
           cmap='YlOrRd', annot=False)
plt.title('Feature Importance Comparison - Top 30 Features')
plt.tight_layout()
```

#### 4. **Hyperparameter Impact Analysis**
```python
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Contamination impact
axes[0, 0].scatter(results_df['contamination'], results_df['val_tpr'])
axes[0, 0].set_xlabel('Contamination')
axes[0, 0].set_ylabel('TPR')

# n_estimators impact
axes[0, 1].scatter(results_df['n_estimators'], results_df['val_tpr'])
axes[0, 1].set_xlabel('n_estimators')
axes[0, 1].set_ylabel('TPR')

# max_samples impact
axes[0, 2].scatter(results_df['max_samples'], results_df['val_tpr'])
axes[0, 2].set_xlabel('max_samples')
axes[0, 2].set_ylabel('TPR')

# Similar for FPR
axes[1, 0].scatter(results_df['contamination'], results_df['val_fpr'])
axes[1, 0].set_xlabel('Contamination')
axes[1, 0].set_ylabel('FPR')
axes[1, 0].axhline(y=0.02, color='red', linestyle='--')

# ... more plots
plt.tight_layout()
```

#### 5. **Progress Timeline**
```python
plt.figure(figsize=(12, 6))
plt.plot(results_df['experiment_id'], results_df['val_tpr'], 
        marker='o', label='TPR', linewidth=2)
plt.plot(results_df['experiment_id'], results_df['val_fpr'] * 25,  # Scale for viz
        marker='s', label='FPR (×25)', linewidth=2)
plt.axhline(y=0.50, color='green', linestyle='--', alpha=0.5, label='TPR Target')
plt.xlabel('Experiment ID')
plt.ylabel('Rate')
plt.title('Optimization Progress Over Experiments')
plt.legend()
plt.grid(True, alpha=0.3)
```

---

## 🧪 Sample Experiment Code Template

```python
def run_experiment(experiment_config, X_train_raw, X_val_raw, 
                   X_test_benign_raw, X_test_malware_raw, results_df):
    """
    Run a single experiment with given configuration.
    
    Args:
        experiment_config: Dict with feature_strategy and hyperparameters
        X_train_raw: Raw training features (NxM)
        X_val_raw: Raw validation features (NxM)
        X_test_benign_raw: Raw test benign features (NxM)
        X_test_malware_raw: Raw test malware features (NxM)
        results_df: DataFrame to append results
    
    Returns:
        Updated results_df, trained model, selected features
    """
    import time
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import roc_auc_score, roc_curve
    
    exp_id = len(results_df) + 1
    print(f"\n{'='*60}")
    print(f"Experiment {exp_id}: {experiment_config['name']}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # 1. Feature Selection
    print(f"Feature selection: {experiment_config['feature_strategy']}...")
    feature_selector = get_feature_selector(experiment_config['feature_strategy'])
    feature_mask = feature_selector.fit(X_train_raw)
    
    X_train = X_train_raw[:, feature_mask]
    X_val = X_val_raw[:, feature_mask]
    X_test_benign = X_test_benign_raw[:, feature_mask]
    X_test_malware = X_test_malware_raw[:, feature_mask]
    
    n_features = X_train.shape[1]
    print(f"  -> Selected {n_features} features")
    
    # 2. Model Training
    print(f"Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=experiment_config['n_estimators'],
        max_samples=experiment_config['max_samples'],
        max_features=experiment_config['max_features'],
        contamination=experiment_config['contamination'],
        bootstrap=experiment_config['bootstrap'],
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)
    print(f"  -> Training complete")
    
    # 3. Validation Evaluation
    print(f"Evaluating on validation set...")
    scores_val_benign = model.score_samples(X_val)
    scores_val_malware = model.score_samples(X_test_malware)
    
    # Combine scores and labels
    scores_val = np.concatenate([scores_val_benign, scores_val_malware])
    y_val = np.concatenate([
        np.zeros(len(scores_val_benign)),  # benign = 0
        np.ones(len(scores_val_malware))   # malware = 1
    ])
    
    # ROC AUC
    val_auc = roc_auc_score(y_val, -scores_val)  # Negative because lower score = more anomalous
    
    # Find threshold for FPR ≈ 2%
    fpr_vals, tpr_vals, thresholds = roc_curve(y_val, -scores_val)
    target_fpr = 0.02
    idx = np.argmin(np.abs(fpr_vals - target_fpr))
    threshold = thresholds[idx]
    val_fpr = fpr_vals[idx]
    val_tpr = tpr_vals[idx]
    
    # Predictions at this threshold
    y_pred = (-scores_val >= threshold).astype(int)
    
    # Precision and F1
    tp = np.sum((y_pred == 1) & (y_val == 1))
    fp = np.sum((y_pred == 1) & (y_val == 0))
    fn = np.sum((y_pred == 0) & (y_val == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = val_tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    training_time = time.time() - start_time
    
    # 4. Log Results
    print(f"\nResults:")
    print(f"  TPR: {val_tpr*100:.2f}%")
    print(f"  FPR: {val_fpr*100:.2f}%")
    print(f"  AUC: {val_auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Time: {training_time:.1f}s")
    
    # 5. Append to Results DataFrame
    new_row = {
        'experiment_id': exp_id,
        'experiment_name': experiment_config['name'],
        'phase': experiment_config['phase'],
        'n_features': n_features,
        'feature_strategy': experiment_config['feature_strategy'],
        'n_estimators': experiment_config['n_estimators'],
        'max_samples': experiment_config['max_samples'],
        'max_features': experiment_config['max_features'],
        'contamination': experiment_config['contamination'],
        'bootstrap': experiment_config['bootstrap'],
        'val_tpr': val_tpr,
        'val_fpr': val_fpr,
        'val_auc': val_auc,
        'val_precision': precision,
        'val_f1': f1,
        'threshold': threshold,
        'training_time_sec': training_time,
        'notes': experiment_config.get('notes', '')
    }
    
    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save checkpoint
    results_df.to_csv('iteration_results.csv', index=False)
    
    return results_df, model, feature_mask, threshold
```

---

## 📋 Complete Experiment Configuration List

```python
experiments = [
    # Phase 1: Quick Wins
    {
        'name': 'Baseline Reproduction',
        'phase': 1,
        'feature_strategy': 'baseline_filtering',  # 487 features
        'n_estimators': 100,
        'max_samples': 0.5,
        'max_features': 0.7,
        'contamination': 0.10,
        'bootstrap': False,
        'notes': 'Verify baseline, expect ~8% TPR'
    },
    {
        'name': 'Contamination 0.15',
        'phase': 1,
        'feature_strategy': 'baseline_filtering',
        'n_estimators': 100,
        'max_samples': 0.5,
        'max_features': 0.7,
        'contamination': 0.15,
        'bootstrap': False,
        'notes': 'First contamination increase'
    },
    {
        'name': 'Contamination 0.20',
        'phase': 1,
        'feature_strategy': 'baseline_filtering',
        'n_estimators': 100,
        'max_samples': 0.5,
        'max_features': 0.7,
        'contamination': 0.20,
        'bootstrap': False,
        'notes': 'Aggressive contamination'
    },
    {
        'name': 'Contamination 0.25',
        'phase': 1,
        'feature_strategy': 'baseline_filtering',
        'n_estimators': 100,
        'max_samples': 0.5,
        'max_features': 0.7,
        'contamination': 0.25,
        'bootstrap': False,
        'notes': 'Very aggressive, likely too high FPR'
    },
    {
        'name': 'All Features + Best Contamination',
        'phase': 1,
        'feature_strategy': 'no_filtering',  # 623 features
        'n_estimators': 100,
        'max_samples': 0.5,
        'max_features': 0.7,
        'contamination': '<BEST_FROM_EXP_2_3_4>',
        'bootstrap': False,
        'notes': 'Test if filtering hurt performance'
    },
    
    # Phase 2A: Hyperparameter Optimization
    {
        'name': '200 Trees',
        'phase': 2,
        'feature_strategy': '<BEST_FROM_PHASE_1>',
        'n_estimators': 200,
        'max_samples': 0.5,
        'max_features': 0.7,
        'contamination': '<BEST_FROM_PHASE_1>',
        'bootstrap': False,
        'notes': 'Moderate ensemble size'
    },
    {
        'name': '500 Trees',
        'phase': 2,
        'feature_strategy': '<BEST_FROM_PHASE_1>',
        'n_estimators': 500,
        'max_samples': 0.5,
        'max_features': 0.7,
        'contamination': '<BEST_FROM_PHASE_1>',
        'bootstrap': False,
        'notes': 'Strong ensemble'
    },
    {
        'name': 'Small Subsample (256)',
        'phase': 2,
        'feature_strategy': '<BEST_FROM_PHASE_1>',
        'n_estimators': 200,
        'max_samples': 256,
        'max_features': 0.7,
        'contamination': '<BEST_FROM_PHASE_1>',
        'bootstrap': False,
        'notes': 'Fixed small subsample for diversity'
    },
    {
        'name': 'Large Subsample (0.7)',
        'phase': 2,
        'feature_strategy': '<BEST_FROM_PHASE_1>',
        'n_estimators': 200,
        'max_samples': 0.7,
        'max_features': 0.7,
        'contamination': '<BEST_FROM_PHASE_1>',
        'bootstrap': False,
        'notes': 'More data per tree'
    },
    {
        'name': 'Bootstrap Enabled',
        'phase': 2,
        'feature_strategy': '<BEST_FROM_PHASE_1>',
        'n_estimators': 200,
        'max_samples': '<BEST_FROM_EXP_8_9>',
        'max_features': 0.7,
        'contamination': '<BEST_FROM_PHASE_1>',
        'bootstrap': True,
        'notes': 'Sampling with replacement'
    },
    {
        'name': 'Feature Diversity 0.5',
        'phase': 2,
        'feature_strategy': '<BEST_FROM_PHASE_1>',
        'n_estimators': 200,
        'max_samples': '<BEST_SO_FAR>',
        'max_features': 0.5,
        'contamination': '<BEST_FROM_PHASE_1>',
        'bootstrap': '<BEST_SO_FAR>',
        'notes': 'More feature diversity'
    },
    {
        'name': 'Strong Ensemble (1000 trees)',
        'phase': 2,
        'feature_strategy': '<BEST_FROM_PHASE_1>',
        'n_estimators': 1000,
        'max_samples': '<BEST_SO_FAR>',
        'max_features': '<BEST_SO_FAR>',
        'contamination': '<BEST_FROM_PHASE_1>',
        'bootstrap': '<BEST_SO_FAR>',
        'notes': 'Maximum stability ensemble'
    },
    
    # Phase 2B: Feature Strategies (if needed)
    {
        'name': 'Top Feature Groups',
        'phase': 2.5,
        'feature_strategy': 'top_groups',  # 200-300 features
        'n_estimators': '<BEST_FROM_PHASE_2A>',
        'max_samples': '<BEST_FROM_PHASE_2A>',
        'max_features': '<BEST_FROM_PHASE_2A>',
        'contamination': '<BEST_FROM_PHASE_1>',
        'bootstrap': '<BEST_FROM_PHASE_2A>',
        'notes': 'Domain knowledge selection'
    },
    {
        'name': 'IF Feature Importance Top 300',
        'phase': 2.5,
        'feature_strategy': 'if_importance_300',
        'n_estimators': '<BEST_FROM_PHASE_2A>',
        'max_samples': '<BEST_FROM_PHASE_2A>',
        'max_features': '<BEST_FROM_PHASE_2A>',
        'contamination': '<BEST_FROM_PHASE_1>',
        'bootstrap': '<BEST_FROM_PHASE_2A>',
        'notes': 'Model-driven selection'
    },
    {
        'name': 'Statistical Relevance Top 350',
        'phase': 2.5,
        'feature_strategy': 'statistical_top_350',
        'n_estimators': '<BEST_FROM_PHASE_2A>',
        'max_samples': '<BEST_FROM_PHASE_2A>',
        'max_features': '<BEST_FROM_PHASE_2A>',
        'contamination': '<BEST_FROM_PHASE_1>',
        'bootstrap': '<BEST_FROM_PHASE_2A>',
        'notes': 'Statistical divergence selection'
    }
]
```

---

## 🎯 Expected Outcomes & Troubleshooting

### Scenario 1: Contamination Solves It (Most Likely)
**If** Experiment 2-4 show TPR jumps to 30-50% range:
- **Success!** Main issue was contamination parameter
- Proceed to hyperparameter tuning for stability
- Likely final TPR: 50-70%

### Scenario 2: Features Are the Problem
**If** Phase 1 all show TPR < 20%:
- Need better feature selection
- Try Phase 2B experiments
- Consider feature engineering
- Review ablation study results more carefully

### Scenario 3: Model Limitations
**If** Phase 2A & 2B both show TPR < 30%:
- Isolation Forest might not be suitable
- Consider supervised learning (if you can label training data)
- Try One-Class SVM, Autoencoders, or Ensemble of multiple unsupervised methods

### Scenario 4: Trade-off Wall
**If** Can't get both TPR > 50% AND FPR ≤ 2%:
- Re-evaluate business requirements
- Consider two-stage detection (high sensitivity filter + precise classifier)
- Analyze false positives - are they edge cases or legitimate benign variants?

---

## 📌 Key Takeaways

1. **Contamination is critical** - Start here, expect biggest impact
2. **Feature quality > quantity** - More features ≠ better (but test it!)
3. **Iterate fast** - Each experiment should be < 2 minutes
4. **Log everything** - Track all configs and results systematically
5. **Visualize progressively** - See patterns emerging across experiments
6. **Don't over-optimize** - Stop when you hit target or plateau
7. **Validate at the end** - Test set is ground truth, avoid peeking

---

## 🚀 Next Steps After This Plan

1. **Implement experiment harness** in notebook
2. **Run Phase 1** (4 experiments, ~10 min)
3. **Analyze results**, decide on Phase 2A vs 2B
4. **Continue iterating** until success or budget exhausted
5. **Final validation** on test set
6. **Save best model** with full configuration
7. **Document findings** in model card

**Estimated Total Time:** 1-3 hours depending on number of experiments needed

Good luck! 🎯
