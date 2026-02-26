# Raw-PE Evaluation Report (Quantization 4-bit)

Generated at: 2026-02-27 02:57:27

## Metadata

- Repo root: /home/viettran/Documents/visual_code/EDR_AGENT/.
- Model name: iforest
- Threshold: -0.541805
- Quantization bits: 4

## Sample Counts

- Train benign: 31954
- Train malware: 0
- Validation benign: 4967
- Validation malware: 404
- Validation total: 5371
- Test benign: 5000
- Test malware: 4227
- Test total: 9227

## Model Parameters and Footprint

- Model RAM size (bytes): 4663482
- Model file size (bytes): 8256197
- Model file path: /home/viettran/Documents/visual_code/EDR_AGENT/./embedded_phase/core/models/isolation_forest/resources/iforest_iforest.bin

## Metrics

- FPR: 0.047457
- TPR: 0.937781
- ROC-AUC: 0.987157
- AP: 0.983043
- PRC-AUC: 0.983043

## Performance

- Benign total inference time (sec): 24.525598
- Malware total inference time (sec): 129.589752
- Total inference time (sec): 154.115350
- Average inference speed per file (ms): 16.702650
- Average inference speed per MB (ms): 5.442300

## Charts

### Precision-Recall Curve

![PR Curve](raw_pe_pr_curve.png)

### ROC Curve (log-spaced FPR ticks)

![ROC Curve](raw_pe_roc_curve_log_fpr.png)
