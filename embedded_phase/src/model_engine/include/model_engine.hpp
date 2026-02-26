#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "models/isolation_forest/if_components.h"
#include "models/isolation_forest/if_model.h"

namespace eml::model_engine {

struct EngineMetadata {
    std::string model_name;
    std::filesystem::path resource_dir;
    uint16_t num_features = 0u;
    uint8_t quantization_bits = 0u;
    float decision_threshold = 0.0f;
    eml_status_code status_code = eml_status_code::ok;
    bool loaded = false;
};

struct EvaluationSummary {
    size_t benign_samples = 0u;
    size_t malware_samples = 0u;
    size_t true_positive = 0u;
    size_t false_positive = 0u;
    size_t true_negative = 0u;
    size_t false_negative = 0u;
    float threshold = 0.0f;
    float fpr = 0.0f;
    float tpr = 0.0f;
    float roc_auc = 0.5f;
    float average_precision = 0.0f;
    eml_status_code status_code = eml_status_code::ok;
    bool success = false;
};

class IsolationForestModelEngine {
public:
    IsolationForestModelEngine() = default;

    bool load_model(const std::string& model_name = "iforest",
                    const std::filesystem::path& resource_dir = eml::IsoForest::default_resource_dir(),
                    std::string* error = nullptr);

    void unload();

    bool infer_quantized(const uint8_t* quantized_features,
                         uint16_t feature_count,
                         eml_isolation_result_t& out_result,
                         std::string* error = nullptr) const;

    bool infer_quantized(const std::vector<uint8_t>& quantized_features,
                         eml_isolation_result_t& out_result,
                         std::string* error = nullptr) const;

    bool infer_raw(const float* raw_features,
                   uint16_t feature_count,
                   eml_isolation_result_t& out_result,
                   std::string* error = nullptr) const;

    bool infer_raw(const std::vector<float>& raw_features,
                   eml_isolation_result_t& out_result,
                   std::string* error = nullptr) const;

    void set_extract_callback(IsoForest::extract_callback_t callback);
    void set_extract_content_callback(IsoForest::extract_content_callback_t callback);

    bool infer_pe_path(const std::filesystem::path& pe_path,
                       eml_isolation_result_t& out_result,
                       std::string* error = nullptr) const;

    bool infer_pe_content(const uint8_t* pe_content,
                          size_t pe_size,
                          eml_isolation_result_t& out_result,
                          std::string* error = nullptr) const;

    bool loaded() const;
    eml_status_code last_status() const;
    const char* last_status_string() const;
    const std::string& last_error() const;
    EngineMetadata metadata() const;

    const eml::IsoForest& model() const;

private:
    eml::IsoForest model_{};
    std::string model_name_{};
    std::filesystem::path resource_dir_{};
    mutable eml_status_code last_status_code_ = eml_status_code::ok;
    mutable std::string last_error_;

    void set_error(eml_status_code status, const std::string& message) const;
    void clear_error() const;
};

bool load_quantized_nml_dataset(const std::filesystem::path& nml_path,
                                uint16_t expected_num_features,
                                uint8_t quantization_bits,
                                std::vector<uint8_t>& out_matrix,
                                size_t& out_num_samples,
                                std::string* error = nullptr);

bool evaluate_validation_splits(const IsolationForestModelEngine& engine,
                                const std::filesystem::path& benign_val_nml_path,
                                const std::filesystem::path& malware_val_nml_path,
                                EvaluationSummary& out_summary,
                                std::string* error = nullptr);

}  // namespace eml::model_engine
