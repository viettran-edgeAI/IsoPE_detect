# Model Tester Tools

This directory contains utilities for exercising and evaluating trained models
against file-based datasets and raw binaries.

* `if_quantized_cpp_raw_pe_eval.cpp` – C++ program that loads an embedded
  Isolation Forest model, scans a benign/malware directory of PE samples, and
  computes FPR/TPR/AUC metrics. It now also emits a detailed `.txt` report used
  by the reporting tools.
* `if_quantized_cpp_raw_pe_eval_report.py` – lightweight Python helper used by
  developers to visualize and summarize the text report.

### Building and Running

To compile the C++ evaluator:

```bash
g++ -std=c++17 if_quantized_cpp_raw_pe_eval.cpp -I../../embedded_phase/src/.. -O2 -o if_quantized_cpp_raw_pe_eval
```

Run the binary with repository root and model name:

```bash
/tmp/if_quantized_cpp_raw_pe_eval --repo-root . --model-name iforest
```

The script `if_quantized_cpp_raw_pe_eval_report.py` is documented in the main
`tools/reporting` README. Activate the virtualenv and run with Python:

```bash
python if_quantized_cpp_raw_pe_eval_report.py --report-txt development_phase/reports/if_quantized_cpp_raw_pe_eval.txt --output-dir reports
```