# Isolation Forest Err-Code Refactor Plan

Date: 2026-02-26  
Scope root: `embedded_phase/core`  
Primary focus: `embedded_phase/core/models/isolation_forest`

---

## 1) Problem Statement and Scope Boundaries

### Problem Statement
Current isolation-forest model-layer code relies heavily on:
- `bool` return values (low diagnostic precision)
- `eml_debug(...)` logging from inside model-layer classes
- success/failure collapse at API boundaries (`success=false` without machine-readable failure reason)

This makes failure triage difficult and violates a clean embedded layering policy where model-layer code should report status, not emit logs.

### Scope Boundaries (Authoritative)
- **In scope**:
  - Isolation-forest model layer and directly coupled preprocessing/model components under `embedded_phase/core/models/isolation_forest`
  - Isolation result/status surface under `embedded_phase/core/ml/eml_predict_result.h`
- **Out of scope**:
  - Global logger implementation (`embedded_phase/core/ml/eml_logger.h`)
  - Generic debug sink (`embedded_phase/core/base/eml_debug.h`) except consumption removal in IF model layer
  - Non-IF datasets/metrics infrastructure (`eml_data.h`, non-IF paths) unless required for compatibility wrappers
- **Policy target**:
  - **No logging in model layer** (`if_*` model components return status only)
  - Logging decisions move to caller/application/service layer

---

## 2) Proposed Err-Code / Status Architecture

## 2.1 New status type
Introduce a dedicated status header in IF model layer:
- **New file**: `embedded_phase/core/models/isolation_forest/if_status.h`
- Add:
  - `enum class eml_status_code : uint16_t`
  - `struct if_status { eml_status_code code; const char* context; }` (context optional, static string only)
  - `constexpr bool if_ok(eml_status_code)`
  - `const char* eml_status_to_string(eml_status_code)`

## 2.2 Canonical enum values (phase-1 set)
Recommended initial set:
- `ok = 0`
- `invalid_argument`
- `not_initialized`
- `not_loaded`
- `resource_not_ready`
- `resource_missing`
- `resource_scan_failed`
- `config_open_failed`
- `config_parse_failed`
- `config_invalid`
- `feature_list_missing`
- `feature_count_mismatch`
- `extract_callback_unset`
- `extract_content_callback_unset`
- `extract_failed`
- `extract_size_mismatch`
- `transform_not_ready`
- `transform_failed`
- `scaler_not_ready`
- `scaler_failed`
- `quantizer_not_loaded`
- `quantizer_failed`
- `dataset_open_failed`
- `dataset_read_failed`
- `dataset_format_invalid`
- `tree_build_failed`
- `model_layout_invalid`
- `model_save_failed`
- `model_load_failed`
- `threshold_calibration_failed`
- `threshold_persist_failed`
- `internal_error`

## 2.3 Mapping rules
- Every current `return false;` in IF model-layer code maps to one explicit `eml_status_code`.
- Use a one-to-one failure mapping where feasible; avoid umbrella `internal_error` unless truly unknown.
- Keep mapping tables local to components for maintainability (e.g., extractor/config/model/container).

## 2.4 Propagation rules
- Each IF component gets `last_status_` with:
  - `eml_status_code last_status() const`
  - `void clear_status()`
- Caller propagation:
  - callee returns status code
  - caller stores/forwards same code unless it intentionally remaps with more context
- Public orchestration (`IsoForest`) owns final status returned to external callers.

## 2.5 Default handling
- Entry to public operation clears status to `ok`.
- On first failure, set deterministic non-`ok` code and return.
- Unknown/unmapped failures default to `internal_error`.

## 2.6 Compatibility strategy
- Preserve existing bool APIs as wrappers:
  - new canonical API: `*_status(...) -> eml_status_code`
  - existing API: call canonical, return `code == ok`
- Preserve existing `eml_isolation_result_t` usage:
  - add status field(s) (see file plan below)
  - retain `success` for backward compatibility
- Preserve extractor callback ABI for phase 1:
  - existing callback remains `bool`
  - `false` maps to `extract_failed`
  - optional phase 2: status-rich callback overload

---

## 3) Per-file Modification Plan (`embedded_phase/core`) with Exact Intent

## 3.1 New file
1. `embedded_phase/core/models/isolation_forest/if_status.h`
   - Define IF status enum, status helpers, string mapping.
   - Keep header-only and allocation-free.

## 3.2 Existing IF model-layer files
2. `embedded_phase/core/models/isolation_forest/if_base.h`
   - Remove `eml_debug(...)` from resource scanning/init paths.
   - Convert key operations to status-returning variants (`scan_current_resource_status`, `init_status`, `update_resource_status`).
   - Keep current bool/flag queries intact.

3. `embedded_phase/core/models/isolation_forest/if_config.h`
   - Replace logging-based failures with explicit status codes for:
     - dp/config open/read/parse failures
     - invalid/missing required fields
     - threshold persist failures
   - Add `last_status_` and status-returning versions of load/persist functions.
   - Keep bool wrappers for compatibility.

4. `embedded_phase/core/models/isolation_forest/if_feature_extractor.h`
   - Replace `eml_debug(...)` with status mapping in all init/extract paths.
   - Add explicit boundary codes around extractor callbacks:
     - callback unset
     - callback failed
     - feature count mismatch
   - Add `last_status_` accessors and status-returning APIs.

5. `embedded_phase/core/models/isolation_forest/if_scaler_layer.h`
   - Remove `eml_debug(...)` and add status codes for file/read/shape failures.
   - Add `last_status_` with status-returning init/transform helpers.

6. `embedded_phase/core/models/isolation_forest/if_feature_transform_layer.h`
   - Add status-returning initialization/transform path (currently bool-only, no logs).
   - Distinguish schema read failure vs schema mismatch vs invalid input.

7. `embedded_phase/core/models/isolation_forest/if_components.h`
   - Add status codes for model binary save/load/format/checksum/layout failures.
   - Add container-level `last_status_` so `IsoForest` can propagate concrete I/O/model errors.
   - Do not add logging.

8. `embedded_phase/core/models/isolation_forest/if_model.h`
   - Central orchestration migration:
     - replace inline `eml_debug(...)` messages with status propagation
     - add `last_status_` and public `last_status()`
     - add canonical `*_status` methods for init/load/build/infer/quantize/calibrate/save
   - Keep existing bool and existing infer signatures as wrappers for compatibility.
   - At `If_feature_extractor` boundaries (`init`, `extract_from_pe`, `extract_from_pe_content`), preserve exact originating status to caller/result.

9. `embedded_phase/core/models/isolation_forest/if_feature_extractor_default.cpp`
   - Keep callback behavior; optionally set a consistent failure mapping contract in comments.
   - No logging additions.

## 3.3 IF public result surface
10. `embedded_phase/core/ml/eml_predict_result.h`
    - Extend `eml_isolation_result_t` with machine-readable status:
      - `eml_status_code status_code` (or `uint16_t status_code` if include constraints require decoupling)
      - optional `const char* status_text` helper at call site (not stored if memory-sensitive)
    - Ensure `clear()` resets status to `ok`/`internal_error` per chosen convention.

## 3.4 Files intentionally not modified (but scanned)
- `embedded_phase/core/base/eml_debug.h`: remains generic debug sink.
- `embedded_phase/core/ml/eml_quantize.h`, `embedded_phase/core/ml/eml_data.h`, `embedded_phase/core/ml/eml_logger.h`: contain logging but outside IF model-layer authority for this refactor.

---

## 4) Logging Hotspots Found (Scan Summary)

## 4.1 Isolation-forest hotspots
- `if_base.h`: resource scan and init emits many info/warn/error messages.
- `if_config.h`: dp/config parse and threshold persistence emit warnings/errors.
- `if_feature_extractor.h`: init/parsing mismatch and config read failures emit errors.
- `if_scaler_layer.h`: scaler init failures emit errors.
- `if_model.h`: initialization orchestration emits error logs around extractor/transform/scaler/quantizer setup.

## 4.2 Boundary-specific hotspot (`If_feature_extractor`)
- `IsoForest::initialize_components()` hard-fails after extractor init failure and logs at model layer.
- `IsoForest::infer_pe_path()` and `IsoForest::infer_pe_content()` drop failures to `success=false` without a propagated machine-readable reason.

## 4.3 Non-IF but relevant output primitives
- `core/base/eml_debug.h` uses `std::cerr`.
- No direct `printf/fprintf/cout/cerr` usage in IF model files except `std::snprintf` numeric formatting helper in `if_config.h` (not output).

---

## 5) Risks and Mitigations

1. **Risk: API churn for existing callers**
   - Mitigation: maintain bool wrappers and old method names; introduce additive `*_status` APIs.

2. **Risk: Missed error-code mapping (fallback to generic failures)**
   - Mitigation: mandatory mapping pass over every `return false` in IF files; add review checklist item per file.

3. **Risk: Include dependency cycles (new status header)**
   - Mitigation: keep `if_status.h` minimal, standalone, no heavy includes.

4. **Risk: Binary size increase from status strings**
   - Mitigation: use compile-time optional string table (`#if IF_STATUS_ENABLE_TEXT`), default off in embedded builds.

5. **Risk: Behavior drift in training/inference flow**
   - Mitigation: no algorithmic changes; status-only refactor; validate with existing quantized workflow metrics.

6. **Risk: Ambiguous callback failures at extractor boundary**
   - Mitigation: explicit mapping (`extract_callback_unset`, `extract_failed`, `extract_size_mismatch`) and preserve first failure code.

---

## 6) Validation Checklist (Tied to EMBEDDED_PHASE.md §4 Workflow)

## Step 1 — Regenerate Stage-3 artifacts
- Run `development_phase/src/model_optimization.py`.
- Verify expected files exist and remain parseable:
  - `<model>_optimized_config.json`
  - `<model>_optimized_features.json`
  - `<model>_scaler_params.json`
  - `<model>_feature_schema.json`
- Negative test: corrupt/missing file should produce deterministic IF status codes, no model-layer logs.

## Step 2 — Prepare quantized resources
- Run `tools/resource_prepairer/prepare_datasets.py`.
- Validate status mapping for missing/invalid NML resources (`dataset_open_failed`, `dataset_format_invalid`).

## Step 3 — Place resources into canonical root
- Ensure resources under `embedded_phase/core/models/isolation_forest/resources`.
- Validate `If_base` status flow for resource readiness without emitting logs.

## Step 4 — End-to-end evaluation
- Run `tools/model_tester/if_quantized_cpp_raw_pe_eval`.
- Confirm:
  - normal path returns `ok`
  - forced failure scenarios return specific non-`ok` codes surfaced via result/status APIs
  - no model-layer `eml_debug(...)` output in IF files

## Step 5 — Metric gate + regression check
- Record FPR/TPR/ROC-AUC and compare to deployment gates in EMBEDDED_PHASE.md.
- Confirm no metric regression (refactor should be behavior-preserving).
- Run static grep gate:
  - `embedded_phase/core/models/isolation_forest/**` contains no `eml_debug(`
  - no new `std::cout/std::cerr/printf` in IF model files

---

## 7) Rollout Steps (Order of Edits)

1. Add `if_status.h` (enum + helpers).
2. Extend `eml_isolation_result_t` with status field(s) in `eml_predict_result.h`.
3. Refactor leaf components first (status + wrappers, remove logs):
   - `if_feature_extractor.h`
   - `if_scaler_layer.h`
   - `if_feature_transform_layer.h`
   - `if_config.h`
   - `if_base.h`
4. Refactor storage/model container status propagation:
   - `if_components.h`
5. Refactor orchestrator last:
   - `if_model.h` (propagate component status end-to-end; remove logs)
6. Apply optional callback-boundary clarifications in `if_feature_extractor_default.cpp`.
7. Run section-4 validation workflow and grep-based no-logging gate.
8. Freeze compatibility pass (ensure all legacy bool APIs still compile and behave identically on success path).

---

## 8) Acceptance Criteria

- IF model layer has **zero direct log emission**.
- Every failure path in IF model layer maps to a deterministic status code.
- Existing callers using bool APIs continue to work.
- `eml_isolation_result_t` carries machine-readable failure reason.
- Section-4 workflow passes and model metrics remain within current deployment gates.
