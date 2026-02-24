#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "models/isolation_forest/if_model.h"

namespace eml {
namespace model_engine {

    struct DatasetBundlePaths {
        std::filesystem::path benign_train;
        std::filesystem::path benign_val;
        std::filesystem::path benign_test;
        std::filesystem::path malware_val;
        std::filesystem::path malware_test;
    };

    struct DevelopmentMetrics {
        float val_fpr = 0.0f;
        float val_tpr = 0.0f;
        float val_roc_auc = 0.0f;
        float test_fpr = 0.0f;
        float test_tpr = 0.0f;
        float test_roc_auc = 0.0f;
    };

    struct EvaluationSummary {
        bool ok = false;
        std::string message;

        float selected_threshold = 0.0f;
        If_binary_metrics validation;
        If_binary_metrics test;

        DevelopmentMetrics development;

        // Per-sample scores for the test splits (populated when save_scores=true)
        std::vector<float> test_benign_scores;
        std::vector<float> test_malware_scores;
    };

    class IsolationForestModelEngine {
    private:
        If_config config_{};
        IsoForest model_{};
        uint16_t num_features_ = 0;

    public:
        bool load_config(const std::filesystem::path& model_engine_config,
                         const std::filesystem::path& dp_txt,
                         std::string* error = nullptr);

        bool train_on_quantized_matrix(const std::vector<uint8_t>& matrix,
                                       size_t num_samples,
                                       std::string* error = nullptr);

        float decision_function_quantized(const uint8_t* quantized_features,
                                          uint16_t feature_count) const;

        bool is_anomaly_quantized(const uint8_t* quantized_features,
                                  uint16_t feature_count,
                                  float threshold) const;

        const If_tree_container& forest() const;

        const If_config& config() const;
        uint16_t num_features() const;
        bool trained() const;
    };

    bool load_quantized_nml_dataset(const std::filesystem::path& nml_path,
                                    uint16_t expected_num_features,
                                    uint8_t quantization_bits,
                                    std::vector<uint8_t>& out_matrix,
                                    size_t& out_num_samples,
                                    std::string* error = nullptr);

    bool load_development_metrics(const std::filesystem::path& model_engine_config,
                                  DevelopmentMetrics& out,
                                  std::string* error = nullptr);

    EvaluationSummary train_and_evaluate(const std::filesystem::path& model_engine_config,
                                         const std::filesystem::path& dp_txt,
                                         const DatasetBundlePaths& dataset_paths,
                                         bool save_scores = false);

} // namespace model_engine
} // namespace eml
