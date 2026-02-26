#include "model_engine.hpp"

#include <fstream>
#include <sstream>
#include <utility>

namespace eml::model_engine {

namespace {

void assign_error(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
}

bool read_exact(std::ifstream& stream, void* dst, size_t bytes) {
    stream.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    return static_cast<size_t>(stream.gcount()) == bytes;
}

}  // namespace

void IsolationForestModelEngine::set_error(eml_status_code status, const std::string& message) const {
    last_status_code_ = status;
    last_error_ = message;
}

void IsolationForestModelEngine::clear_error() const {
    last_status_code_ = eml_status_code::ok;
    last_error_.clear();
}

bool IsolationForestModelEngine::load_model(const std::string& model_name,
                                            const std::filesystem::path& resource_dir,
                                            std::string* error) {
    clear_error();

    if (model_name.empty()) {
        set_error(eml_status_code::empty_model_name, "model_name is empty");
        assign_error(error, last_error_);
        return false;
    }

    if (!model_.init(model_name, resource_dir)) {
        set_error(model_.last_status(), std::string("failed to initialize model: ") + eml_status_to_string(model_.last_status()));
        assign_error(error, last_error_);
        return false;
    }

    if (!model_.load()) {
        set_error(model_.last_status(), std::string("failed to load model resources: ") + eml_status_to_string(model_.last_status()));
        assign_error(error, last_error_);
        return false;
    }

    model_name_ = model_name;
    resource_dir_ = resource_dir;
    clear_error();
    assign_error(error, std::string{});
    return true;
}

void IsolationForestModelEngine::unload() {
    model_ = eml::IsoForest();
    model_name_.clear();
    resource_dir_.clear();
    clear_error();
}

bool IsolationForestModelEngine::infer_quantized(const uint8_t* quantized_features,
                                                 uint16_t feature_count,
                                                 eml_isolation_result_t& out_result,
                                                 std::string* error) const {
    clear_error();
    out_result.clear();

    if (!model_.loaded()) {
        set_error(eml_status_code::not_loaded, "model is not loaded");
        assign_error(error, last_error_);
        return false;
    }

    out_result = model_.infer_quantized(quantized_features, feature_count);
    if (!out_result.success) {
        set_error(out_result.status_code, std::string("quantized inference failed: ") + eml_status_to_string(out_result.status_code));
        assign_error(error, last_error_);
        return false;
    }

    clear_error();
    assign_error(error, std::string{});
    return true;
}

bool IsolationForestModelEngine::infer_quantized(const std::vector<uint8_t>& quantized_features,
                                                 eml_isolation_result_t& out_result,
                                                 std::string* error) const {
    return infer_quantized(
        quantized_features.empty() ? nullptr : quantized_features.data(),
        static_cast<uint16_t>(quantized_features.size()),
        out_result,
        error);
}

bool IsolationForestModelEngine::infer_raw(const float* raw_features,
                                           uint16_t feature_count,
                                           eml_isolation_result_t& out_result,
                                           std::string* error) const {
    clear_error();
    out_result.clear();

    if (!model_.loaded()) {
        set_error(eml_status_code::not_loaded, "model is not loaded");
        assign_error(error, last_error_);
        return false;
    }

    out_result = model_.infer_raw(raw_features, feature_count);
    if (!out_result.success) {
        set_error(out_result.status_code, std::string("raw inference failed: ") + eml_status_to_string(out_result.status_code));
        assign_error(error, last_error_);
        return false;
    }

    clear_error();
    assign_error(error, std::string{});
    return true;
}

bool IsolationForestModelEngine::infer_raw(const std::vector<float>& raw_features,
                                           eml_isolation_result_t& out_result,
                                           std::string* error) const {
    return infer_raw(
        raw_features.empty() ? nullptr : raw_features.data(),
        static_cast<uint16_t>(raw_features.size()),
        out_result,
        error);
}

void IsolationForestModelEngine::set_extract_callback(IsoForest::extract_callback_t callback) {
    model_.set_extract_callback(std::move(callback));
}

void IsolationForestModelEngine::set_extract_content_callback(IsoForest::extract_content_callback_t callback) {
    model_.set_extract_content_callback(std::move(callback));
}

bool IsolationForestModelEngine::infer_pe_path(const std::filesystem::path& pe_path,
                                               eml_isolation_result_t& out_result,
                                               std::string* error) const {
    clear_error();
    out_result.clear();

    if (!model_.loaded()) {
        set_error(eml_status_code::not_loaded, "model is not loaded");
        assign_error(error, last_error_);
        return false;
    }

    out_result = model_.infer_pe_path(pe_path);
    if (!out_result.success) {
        set_error(out_result.status_code, std::string("PE-path inference failed: ") + eml_status_to_string(out_result.status_code));
        assign_error(error, last_error_);
        return false;
    }

    clear_error();
    assign_error(error, std::string{});
    return true;
}

bool IsolationForestModelEngine::infer_pe_content(const uint8_t* pe_content,
                                                  size_t pe_size,
                                                  eml_isolation_result_t& out_result,
                                                  std::string* error) const {
    clear_error();
    out_result.clear();

    if (!model_.loaded()) {
        set_error(eml_status_code::not_loaded, "model is not loaded");
        assign_error(error, last_error_);
        return false;
    }

    out_result = model_.infer_pe_content(pe_content, pe_size);
    if (!out_result.success) {
        set_error(out_result.status_code, std::string("PE-content inference failed: ") + eml_status_to_string(out_result.status_code));
        assign_error(error, last_error_);
        return false;
    }

    clear_error();
    assign_error(error, std::string{});
    return true;
}

bool IsolationForestModelEngine::loaded() const {
    return model_.loaded();
}

eml_status_code IsolationForestModelEngine::last_status() const {
    return last_status_code_;
}

const char* IsolationForestModelEngine::last_status_string() const {
    return eml_status_to_string(last_status_code_);
}

const std::string& IsolationForestModelEngine::last_error() const {
    return last_error_;
}

EngineMetadata IsolationForestModelEngine::metadata() const {
    EngineMetadata metadata;
    metadata.model_name = model_name_;
    metadata.resource_dir = resource_dir_;
    metadata.loaded = model_.loaded();
    metadata.status_code = model_.last_status();
    metadata.num_features = model_.num_features();
    if (metadata.loaded) {
        metadata.quantization_bits = model_.config().quantization_bits;
        metadata.decision_threshold = model_.config().decision_threshold;
    }
    return metadata;
}

const eml::IsoForest& IsolationForestModelEngine::model() const {
    return model_;
}

bool load_quantized_nml_dataset(const std::filesystem::path& nml_path,
                                uint16_t expected_num_features,
                                uint8_t quantization_bits,
                                std::vector<uint8_t>& out_matrix,
                                size_t& out_num_samples,
                                std::string* error) {
    out_matrix.clear();
    out_num_samples = 0u;

    if (nml_path.empty()) {
        assign_error(error, "nml path is empty");
        return false;
    }

    if (expected_num_features == 0u) {
        assign_error(error, "expected_num_features must be > 0");
        return false;
    }

    if (quantization_bits < 1u || quantization_bits > 8u) {
        assign_error(error, "quantization_bits must be in [1, 8]");
        return false;
    }

    std::ifstream input(nml_path, std::ios::binary);
    if (!input.is_open()) {
        assign_error(error, std::string("failed to open nml file: ") + nml_path.string());
        return false;
    }

    uint32_t num_samples_u32 = 0u;
    uint16_t num_features_u16 = 0u;
    if (!read_exact(input, &num_samples_u32, sizeof(num_samples_u32)) ||
        !read_exact(input, &num_features_u16, sizeof(num_features_u16))) {
        assign_error(error, std::string("failed to read nml header: ") + nml_path.string());
        return false;
    }

    if (num_features_u16 != expected_num_features) {
        std::ostringstream oss;
        oss << "feature count mismatch in nml file: expected=" << expected_num_features
            << " got=" << num_features_u16;
        assign_error(error, oss.str());
        return false;
    }

    const uint16_t packed_feature_bytes = static_cast<uint16_t>(
        (static_cast<uint32_t>(num_features_u16) * quantization_bits + 7u) / 8u);
    const uint8_t feature_mask = quantization_bits == 8u
        ? 0xFFu
        : static_cast<uint8_t>((1u << quantization_bits) - 1u);

    out_num_samples = static_cast<size_t>(num_samples_u32);
    out_matrix.assign(out_num_samples * static_cast<size_t>(num_features_u16), 0u);

    std::vector<uint8_t> packed(packed_feature_bytes, 0u);
    for (size_t row = 0u; row < out_num_samples; ++row) {
        uint8_t label = 0u;
        if (!read_exact(input, &label, sizeof(label))) {
            assign_error(error, "failed to read nml label");
            return false;
        }

        if (!read_exact(input, packed.data(), packed.size())) {
            assign_error(error, "failed to read nml packed feature row");
            return false;
        }

        uint8_t* out_row = &out_matrix[row * static_cast<size_t>(num_features_u16)];
        for (uint16_t col = 0u; col < num_features_u16; ++col) {
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

    assign_error(error, std::string{});
    return true;
}

bool evaluate_validation_splits(const IsolationForestModelEngine& engine,
                                const std::filesystem::path& benign_val_nml_path,
                                const std::filesystem::path& malware_val_nml_path,
                                EvaluationSummary& out_summary,
                                std::string* error) {
    out_summary = EvaluationSummary{};

    if (!engine.loaded()) {
        assign_error(error, "model engine is not loaded");
        out_summary.status_code = eml_status_code::not_loaded;
        return false;
    }

    const EngineMetadata metadata = engine.metadata();
    if (metadata.num_features == 0u || metadata.quantization_bits == 0u) {
        assign_error(error, "invalid model metadata");
        out_summary.status_code = eml_status_code::invalid_configuration;
        return false;
    }

    std::vector<uint8_t> benign_matrix;
    std::vector<uint8_t> malware_matrix;
    size_t benign_samples = 0u;
    size_t malware_samples = 0u;
    std::string loader_error;

    if (!load_quantized_nml_dataset(
            benign_val_nml_path,
            metadata.num_features,
            metadata.quantization_bits,
            benign_matrix,
            benign_samples,
            &loader_error)) {
        assign_error(error, loader_error);
        out_summary.status_code = eml_status_code::file_read_failed;
        return false;
    }

    if (!load_quantized_nml_dataset(
            malware_val_nml_path,
            metadata.num_features,
            metadata.quantization_bits,
            malware_matrix,
            malware_samples,
            &loader_error)) {
        assign_error(error, loader_error);
        out_summary.status_code = eml_status_code::file_read_failed;
        return false;
    }

    if (benign_samples == 0u || malware_samples == 0u) {
        assign_error(error, "validation datasets are empty");
        out_summary.status_code = eml_status_code::size_mismatch;
        return false;
    }

    eml_isolation_metrics metrics;
    metrics.init(eval_metric::ROC_AUC);

    eml_isolation_result_t result;
    std::string infer_error;

    out_summary.benign_samples = benign_samples;
    out_summary.malware_samples = malware_samples;

    for (size_t row = 0u; row < benign_samples; ++row) {
        const uint8_t* sample = &benign_matrix[row * static_cast<size_t>(metadata.num_features)];
        if (!engine.infer_quantized(sample, metadata.num_features, result, &infer_error)) {
            assign_error(error, infer_error);
            out_summary.status_code = result.status_code;
            return false;
        }
        out_summary.threshold = result.threshold;
        metrics.update(false, result.is_anomaly, -result.anomaly_score);
    }

    for (size_t row = 0u; row < malware_samples; ++row) {
        const uint8_t* sample = &malware_matrix[row * static_cast<size_t>(metadata.num_features)];
        if (!engine.infer_quantized(sample, metadata.num_features, result, &infer_error)) {
            assign_error(error, infer_error);
            out_summary.status_code = result.status_code;
            return false;
        }
        out_summary.threshold = result.threshold;
        metrics.update(true, result.is_anomaly, -result.anomaly_score);
    }

    out_summary.true_positive = static_cast<size_t>(metrics.true_positives());
    out_summary.false_positive = static_cast<size_t>(metrics.false_positives());
    out_summary.true_negative = static_cast<size_t>(metrics.true_negatives());
    out_summary.false_negative = static_cast<size_t>(metrics.false_negatives());
    out_summary.fpr = metrics.fpr();
    out_summary.tpr = metrics.tpr();
    out_summary.roc_auc = metrics.roc_auc();
    out_summary.average_precision = metrics.average_precision();
    out_summary.status_code = eml_status_code::ok;
    out_summary.success = true;

    assign_error(error, std::string{});
    return true;
}

}  // namespace eml::model_engine
