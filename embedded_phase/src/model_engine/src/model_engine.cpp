#include "model_engine.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>

namespace eml {
namespace model_engine {

    namespace {

        inline bool read_exact(std::ifstream& fin, void* dst, size_t bytes) {
            fin.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
            return static_cast<size_t>(fin.gcount()) == bytes;
        }

        inline void set_error(std::string* error, const std::string& msg) {
            if (error) {
                *error = msg;
            }
        }

        std::vector<float> score_dataset(const If_tree_container& forest,
                                         const std::vector<uint8_t>& matrix,
                                         size_t num_samples,
                                         uint16_t num_features) {
            std::vector<float> scores;
            scores.reserve(num_samples);
            for (size_t i = 0; i < num_samples; ++i) {
                const uint8_t* sample = &matrix[i * static_cast<size_t>(num_features)];
                scores.push_back(forest.decision_function(sample, num_features));
            }
            return scores;
        }

    } // namespace

    bool IsolationForestModelEngine::load_config(const std::filesystem::path& model_engine_config,
                                                 const std::filesystem::path& dp_txt,
                                                 std::string* error) {
        config_ = If_config{};
        if (!config_.load_from_files(dp_txt, model_engine_config)) {
            set_error(error, "Failed to parse config or dp metadata");
            return false;
        }
        if (!model_.init_from_config(config_)) {
            set_error(error, "Failed to initialize IsoForest from loaded config");
            return false;
        }
        num_features_ = config_.num_features;
        return true;
    }

    bool IsolationForestModelEngine::train_on_quantized_matrix(const std::vector<uint8_t>& matrix,
                                                                size_t num_samples,
                                                                std::string* error) {
        (void)matrix;
        (void)num_samples;

        if (!config_.isLoaded) {
            set_error(error, "Config must be loaded before training");
            return false;
        }

        if (!model_.build_model(true)) {
            set_error(error, "Isolation forest build failed");
            return false;
        }
        return true;
    }

    float IsolationForestModelEngine::decision_function_quantized(const uint8_t* quantized_features,
                                                                   uint16_t feature_count) const {
        return model_.decision_function(quantized_features, feature_count);
    }

    bool IsolationForestModelEngine::is_anomaly_quantized(const uint8_t* quantized_features,
                                                           uint16_t feature_count,
                                                           float threshold) const {
        return model_.is_anomaly(quantized_features, feature_count, threshold);
    }

    const If_tree_container& IsolationForestModelEngine::forest() const {
        return model_.tree_container();
    }

    const If_config& IsolationForestModelEngine::config() const {
        return config_;
    }

    uint16_t IsolationForestModelEngine::num_features() const {
        return num_features_;
    }

    bool IsolationForestModelEngine::trained() const {
        return model_.loaded();
    }

    bool load_quantized_nml_dataset(const std::filesystem::path& nml_path,
                                    uint16_t expected_num_features,
                                    uint8_t quantization_bits,
                                    std::vector<uint8_t>& out_matrix,
                                    size_t& out_num_samples,
                                    std::string* error) {
        out_matrix.clear();
        out_num_samples = 0u;

        std::ifstream fin(nml_path, std::ios::binary);
        if (!fin.is_open()) {
            set_error(error, "Cannot open dataset: " + nml_path.string());
            return false;
        }

        uint32_t num_samples_u32 = 0u;
        uint16_t num_features_u16 = 0u;
        if (!read_exact(fin, &num_samples_u32, sizeof(num_samples_u32)) ||
            !read_exact(fin, &num_features_u16, sizeof(num_features_u16))) {
            set_error(error, "Failed to read dataset header: " + nml_path.string());
            return false;
        }

        if (num_features_u16 != expected_num_features) {
            set_error(error, "Feature-count mismatch in dataset: " + nml_path.string());
            return false;
        }

        if (quantization_bits < 1u || quantization_bits > 8u) {
            set_error(error, "Invalid quantization bits");
            return false;
        }

        const uint16_t packed_feature_bytes = static_cast<uint16_t>(
            (static_cast<uint32_t>(num_features_u16) * quantization_bits + 7u) / 8u
        );
        const uint8_t feature_mask = quantization_bits == 8u
            ? 0xFFu
            : static_cast<uint8_t>((1u << quantization_bits) - 1u);

        out_num_samples = static_cast<size_t>(num_samples_u32);
        out_matrix.resize(out_num_samples * static_cast<size_t>(num_features_u16), 0u);

        std::vector<uint8_t> packed(packed_feature_bytes, 0u);
        for (size_t row = 0; row < out_num_samples; ++row) {
            uint8_t label = 0u;
            if (!read_exact(fin, &label, sizeof(label))) {
                set_error(error, "Failed to read dataset label payload");
                return false;
            }
            (void)label;

            if (!read_exact(fin, packed.data(), packed.size())) {
                set_error(error, "Failed to read packed feature payload");
                return false;
            }

            uint8_t* out_row = &out_matrix[row * static_cast<size_t>(num_features_u16)];
            for (uint16_t col = 0; col < num_features_u16; ++col) {
                const uint32_t bit_pos = static_cast<uint32_t>(col) * quantization_bits;
                const uint32_t byte_index = bit_pos / 8u;
                const uint8_t bit_offset = static_cast<uint8_t>(bit_pos % 8u);

                uint8_t value = 0u;
                if (bit_offset + quantization_bits <= 8u) {
                    value = static_cast<uint8_t>((packed[byte_index] >> bit_offset) & feature_mask);
                } else {
                    const uint8_t bits_first = static_cast<uint8_t>(8u - bit_offset);
                    const uint8_t bits_second = static_cast<uint8_t>(quantization_bits - bits_first);
                    const uint8_t low = static_cast<uint8_t>((packed[byte_index] >> bit_offset) & ((1u << bits_first) - 1u));
                    const uint8_t high = static_cast<uint8_t>(packed[byte_index + 1u] & ((1u << bits_second) - 1u));
                    value = static_cast<uint8_t>(low | (high << bits_first));
                }
                out_row[col] = value;
            }
        }

        return true;
    }

    bool load_development_metrics(const std::filesystem::path& model_engine_config,
                                  DevelopmentMetrics& out,
                                  std::string* error) {
        std::string json;
        if (!if_config_detail::read_text_file(model_engine_config, json)) {
            set_error(error, "Cannot read config for development metrics");
            return false;
        }

        size_t validation_pos = json.find("\"validation\"");
        size_t test_pos = json.find("\"test\"");
        if (validation_pos == std::string::npos || test_pos == std::string::npos) {
            set_error(error, "Missing evaluation blocks in development config");
            return false;
        }

        double value = 0.0;
        if (!if_config_detail::extract_number(json, "fpr", value, validation_pos)) {
            return false;
        }
        out.val_fpr = static_cast<float>(value);
        if (!if_config_detail::extract_number(json, "tpr", value, validation_pos)) {
            return false;
        }
        out.val_tpr = static_cast<float>(value);
        if (!if_config_detail::extract_number(json, "roc_auc", value, validation_pos)) {
            return false;
        }
        out.val_roc_auc = static_cast<float>(value);

        if (!if_config_detail::extract_number(json, "fpr", value, test_pos)) {
            return false;
        }
        out.test_fpr = static_cast<float>(value);
        if (!if_config_detail::extract_number(json, "tpr", value, test_pos)) {
            return false;
        }
        out.test_tpr = static_cast<float>(value);
        if (!if_config_detail::extract_number(json, "roc_auc", value, test_pos)) {
            return false;
        }
        out.test_roc_auc = static_cast<float>(value);

        return true;
    }

    EvaluationSummary train_and_evaluate(const std::filesystem::path& model_engine_config,
                                         const std::filesystem::path& dp_txt,
                                         const DatasetBundlePaths& dataset_paths,
                                         bool save_scores) {
        EvaluationSummary summary;

        IsolationForestModelEngine engine;
        std::string err;
        if (!engine.load_config(model_engine_config, dp_txt, &err)) {
            summary.message = err;
            return summary;
        }

        const If_config& cfg = engine.config();

        std::vector<uint8_t> train_matrix;
        std::vector<uint8_t> benign_val_matrix;
        std::vector<uint8_t> benign_test_matrix;
        std::vector<uint8_t> malware_val_matrix;
        std::vector<uint8_t> malware_test_matrix;
        size_t n_train = 0u;
        size_t n_bval = 0u;
        size_t n_btest = 0u;
        size_t n_mval = 0u;
        size_t n_mtest = 0u;

        if (!load_quantized_nml_dataset(dataset_paths.benign_train,
                                        cfg.num_features,
                                        cfg.quantization_bits,
                                        train_matrix,
                                        n_train,
                                        &err)) {
            summary.message = err;
            return summary;
        }

        if (!load_quantized_nml_dataset(dataset_paths.benign_val,
                                        cfg.num_features,
                                        cfg.quantization_bits,
                                        benign_val_matrix,
                                        n_bval,
                                        &err)) {
            summary.message = err;
            return summary;
        }

        if (!load_quantized_nml_dataset(dataset_paths.benign_test,
                                        cfg.num_features,
                                        cfg.quantization_bits,
                                        benign_test_matrix,
                                        n_btest,
                                        &err)) {
            summary.message = err;
            return summary;
        }

        if (!load_quantized_nml_dataset(dataset_paths.malware_val,
                                        cfg.num_features,
                                        cfg.quantization_bits,
                                        malware_val_matrix,
                                        n_mval,
                                        &err)) {
            summary.message = err;
            return summary;
        }

        if (!load_quantized_nml_dataset(dataset_paths.malware_test,
                                        cfg.num_features,
                                        cfg.quantization_bits,
                                        malware_test_matrix,
                                        n_mtest,
                                        &err)) {
            summary.message = err;
            return summary;
        }

        if (!engine.train_on_quantized_matrix(train_matrix, n_train, &err)) {
            summary.message = err;
            return summary;
        }

        const std::vector<float> val_b_scores = score_dataset(
            engine.forest(),
            benign_val_matrix,
            n_bval,
            cfg.num_features
        );
        const std::vector<float> val_m_scores = score_dataset(
            engine.forest(),
            malware_val_matrix,
            n_mval,
            cfg.num_features
        );

        If_threshold_result threshold;
        if (cfg.threshold_strategy == "fpr") {
            threshold = if_find_threshold_precise(val_b_scores, cfg.val_fpr_target);
            size_t tp = 0u;
            for (float s : val_m_scores) {
                if (s < threshold.threshold) {
                    ++tp;
                }
            }
            threshold.tpr = val_m_scores.empty()
                ? 0.0f
                : static_cast<float>(tp) / static_cast<float>(val_m_scores.size());
        } else if (cfg.threshold_strategy == "model") {
            threshold.threshold = 0.0f;
            const If_binary_metrics metrics = if_compute_metrics(val_b_scores, val_m_scores, threshold.threshold);
            threshold.fpr = metrics.fpr();
            threshold.tpr = metrics.tpr();
            threshold.metric = metrics.tpr();
            threshold.has_metric = true;
        } else {
            threshold = if_select_threshold_with_malware(
                val_b_scores,
                val_m_scores,
                cfg.val_fpr_target,
                cfg.threshold_strategy,
                1.0f
            );
        }

        summary.selected_threshold = threshold.threshold;
        summary.validation = if_compute_metrics(val_b_scores, val_m_scores, summary.selected_threshold);

        const std::vector<float> test_b_scores = score_dataset(
            engine.forest(),
            benign_test_matrix,
            n_btest,
            cfg.num_features
        );
        const std::vector<float> test_m_scores = score_dataset(
            engine.forest(),
            malware_test_matrix,
            n_mtest,
            cfg.num_features
        );

        summary.test = if_compute_metrics(test_b_scores, test_m_scores, summary.selected_threshold);

        if (save_scores) {
            summary.test_benign_scores = test_b_scores;
            summary.test_malware_scores = test_m_scores;
        }

        if (!load_development_metrics(model_engine_config, summary.development, &err)) {
            summary.message = err;
            return summary;
        }

        summary.ok = true;
        summary.message = "ok";
        return summary;
    }

} // namespace model_engine
} // namespace eml
