#include "model_engine.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <vector>

namespace {

std::filesystem::path resolve_resource_dir() {
    std::filesystem::path cursor = std::filesystem::current_path();
    for (size_t depth = 0; depth < 10u; ++depth) {
        const auto candidate = cursor / "embedded_phase/core/models/isolation_forest/resources";
        if (std::filesystem::exists(candidate)) {
            return std::filesystem::weakly_canonical(candidate);
        }

        if (!cursor.has_parent_path()) {
            break;
        }
        cursor = cursor.parent_path();
    }

    return {};
}

}  // namespace

int main() {
    const std::filesystem::path resource_dir = resolve_resource_dir();
    if (resource_dir.empty()) {
        std::cerr << "pe_model_engine_tests: unable to resolve resource directory\n";
        return 1;
    }

    eml::model_engine::IsolationForestModelEngine engine;
    std::string error;

    const bool loaded = engine.load_model("iforest", resource_dir, &error);
    assert(loaded);
    if (!loaded) {
        std::cerr << "pe_model_engine_tests: load failed: " << error << "\n";
        return 1;
    }

    const auto metadata = engine.metadata();
    assert(metadata.loaded);
    assert(metadata.num_features > 0u);
    assert(metadata.quantization_bits > 0u);

    const std::filesystem::path benign_val_path = resource_dir / "iforest_ben_val_nml.bin";
    std::vector<uint8_t> benign_matrix;
    size_t benign_samples = 0u;

    const bool loaded_nml = eml::model_engine::load_quantized_nml_dataset(
        benign_val_path,
        metadata.num_features,
        metadata.quantization_bits,
        benign_matrix,
        benign_samples,
        &error);

    assert(loaded_nml);
    assert(benign_samples > 0u);
    if (!loaded_nml || benign_samples == 0u) {
        std::cerr << "pe_model_engine_tests: failed to load benign nml: " << error << "\n";
        return 1;
    }

    eml::eml_isolation_result_t inference;
    const bool infer_ok = engine.infer_quantized(benign_matrix.data(), metadata.num_features, inference, &error);
    assert(infer_ok);
    assert(inference.success);
    assert(inference.status_code == eml::eml_status_code::ok);

    eml::model_engine::EvaluationSummary summary;
    const bool eval_ok = eml::model_engine::evaluate_test_splits(
        engine,
        resource_dir / "iforest_ben_test_nml.bin",
        resource_dir / "iforest_mal_test_nml.bin",
        summary,
        &error);

    assert(eval_ok);
    assert(summary.success);
    assert(summary.benign_samples > 0u);
    assert(summary.malware_samples > 0u);
    assert(summary.fpr >= 0.0f && summary.fpr <= 1.0f);
    assert(summary.tpr >= 0.0f && summary.tpr <= 1.0f);

    std::cout << "pe_model_engine_tests: PASS"
              << " threshold=" << summary.threshold
              << " fpr=" << summary.fpr
              << " tpr=" << summary.tpr
              << " roc_auc=" << summary.roc_auc
              << "\n";

    return 0;
}
