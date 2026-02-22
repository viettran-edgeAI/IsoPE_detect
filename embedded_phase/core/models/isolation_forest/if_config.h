#pragma once

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

#include "if_base.h"

namespace eml {

    namespace if_config_detail {

        inline bool read_text_file(const std::filesystem::path& file_path, std::string& out) {
            std::ifstream fin(file_path, std::ios::in);
            if (!fin.is_open()) {
                return false;
            }
            out.assign((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
            return true;
        }

        inline bool read_exact(std::ifstream& fin, void* dst, size_t bytes) {
            fin.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
            return static_cast<size_t>(fin.gcount()) == bytes;
        }

        inline bool extract_key_pos(const std::string& json, const std::string& key, size_t& key_pos, size_t from = 0) {
            key_pos = json.find(std::string("\"") + key + "\"", from);
            return key_pos != std::string::npos;
        }

        inline bool extract_string(const std::string& json, const std::string& key, std::string& out, size_t from = 0) {
            size_t key_pos = 0;
            if (!extract_key_pos(json, key, key_pos, from)) {
                return false;
            }
            size_t colon = json.find(':', key_pos);
            if (colon == std::string::npos) {
                return false;
            }
            size_t q1 = json.find('"', colon + 1);
            if (q1 == std::string::npos) {
                return false;
            }
            size_t q2 = json.find('"', q1 + 1);
            if (q2 == std::string::npos || q2 <= q1) {
                return false;
            }
            out = json.substr(q1 + 1, q2 - q1 - 1);
            return true;
        }

        inline bool extract_number(const std::string& json, const std::string& key, double& out, size_t from = 0) {
            size_t key_pos = 0;
            if (!extract_key_pos(json, key, key_pos, from)) {
                return false;
            }

            size_t colon = json.find(':', key_pos);
            if (colon == std::string::npos) {
                return false;
            }

            size_t pos = colon + 1;
            while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\r' || json[pos] == '\n')) {
                ++pos;
            }
            if (pos >= json.size()) {
                return false;
            }

            const char* begin = json.c_str() + pos;
            char* end = nullptr;
            errno = 0;
            const double v = std::strtod(begin, &end);
            if (begin == end || errno != 0) {
                return false;
            }

            out = v;
            return true;
        }

        inline bool extract_bool(const std::string& json, const std::string& key, bool& out, size_t from = 0) {
            size_t key_pos = 0;
            if (!extract_key_pos(json, key, key_pos, from)) {
                return false;
            }

            size_t colon = json.find(':', key_pos);
            if (colon == std::string::npos) {
                return false;
            }

            size_t pos = colon + 1;
            while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\r' || json[pos] == '\n')) {
                ++pos;
            }

            if (json.compare(pos, 4, "true") == 0) {
                out = true;
                return true;
            }
            if (json.compare(pos, 5, "false") == 0) {
                out = false;
                return true;
            }
            return false;
        }

        inline bool extract_float_array(const std::string& json, const std::string& key, vector<float>& out, size_t from = 0) {
            size_t key_pos = 0;
            if (!extract_key_pos(json, key, key_pos, from)) {
                return false;
            }

            size_t open = json.find('[', key_pos);
            if (open == std::string::npos) {
                return false;
            }

            size_t close = open;
            int depth = 0;
            for (; close < json.size(); ++close) {
                if (json[close] == '[') {
                    ++depth;
                } else if (json[close] == ']') {
                    --depth;
                    if (depth == 0) {
                        break;
                    }
                }
            }
            if (close == std::string::npos || close <= open) {
                return false;
            }

            out.clear();
            out.reserve(64);

            const char* p = json.c_str() + open + 1;
            const char* end = json.c_str() + close;
            while (p < end) {
                while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n' || *p == ',')) {
                    ++p;
                }
                if (p >= end) {
                    break;
                }

                char* next = nullptr;
                errno = 0;
                float v = std::strtof(p, &next);
                if (next == p || errno != 0) {
                    while (p < end && *p != ',') {
                        ++p;
                    }
                    continue;
                }

                out.push_back(v);
                p = next;
            }

            return !out.empty();
        }

        inline std::string trim(const std::string& in) {
            if (in.empty()) {
                return in;
            }
            size_t begin = 0;
            while (begin < in.size() && (in[begin] == ' ' || in[begin] == '\t' || in[begin] == '\r' || in[begin] == '\n')) {
                ++begin;
            }
            if (begin == in.size()) {
                return {};
            }
            size_t end = in.size() - 1;
            while (end > begin && (in[end] == ' ' || in[end] == '\t' || in[end] == '\r' || in[end] == '\n')) {
                --end;
            }
            return in.substr(begin, end - begin + 1);
        }

    } // namespace if_config_detail


    class If_config {
    private:
        const If_base* base_ptr = nullptr;

        bool load_dp_txt(const std::filesystem::path& dp_path) {
            std::ifstream fin(dp_path, std::ios::in);
            if (!fin.is_open()) {
                eml_debug(0, "❌ IF config: failed to open dp txt: ", dp_path.string().c_str());
                return false;
            }

            num_samples = 0;
            num_features = 0;
            num_labels = 0;
            quantization_bits = 2;
            problem = problem_type::ISOLATION;
            samples_per_label.clear();

            std::string line;
            bool is_header = true;
            while (std::getline(fin, line)) {
                line = if_config_detail::trim(line);
                if (line.empty()) {
                    continue;
                }

                if (is_header) {
                    is_header = false;
                    if (line == "parameter,value") {
                        continue;
                    }
                }

                const size_t comma = line.find(',');
                if (comma == std::string::npos) {
                    continue;
                }

                const std::string key = if_config_detail::trim(line.substr(0, comma));
                const std::string value = if_config_detail::trim(line.substr(comma + 1));

                if (key == "quantization_coefficient") {
                    quantization_bits = static_cast<uint8_t>(std::max(1, std::min(8, std::atoi(value.c_str()))));
                } else if (key == "num_features") {
                    num_features = static_cast<uint16_t>(std::max(0, std::atoi(value.c_str())));
                } else if (key == "num_samples") {
                    num_samples = static_cast<sample_idx_type>(std::max(0, std::atoi(value.c_str())));
                } else if (key == "num_labels") {
                    num_labels = static_cast<uint16_t>(std::max(0, std::atoi(value.c_str())));
                    samples_per_label.clear();
                    samples_per_label.resize(num_labels, 0);
                } else if (key == "problem_type") {
                    problem = problemTypeFromString(value);
                } else if (key.rfind("samples_label_", 0) == 0) {
                    int idx = std::atoi(key.substr(std::strlen("samples_label_")).c_str());
                    if (idx >= 0) {
                        const size_t label_idx = static_cast<size_t>(idx);
                        if (label_idx >= samples_per_label.size()) {
                            samples_per_label.resize(label_idx + 1, 0);
                        }
                        samples_per_label[label_idx] = static_cast<sample_idx_type>(std::max(0, std::atoi(value.c_str())));
                    }
                }
            }

            if (problem == problem_type::UNKNOWN) {
                problem = problem_type::ISOLATION;
            }

            if (problem != problem_type::ISOLATION) {
                eml_debug(1, "⚠️ IF config: dp file problem_type is not isolation: ", problemTypeToString(problem).c_str());
            }

            if (num_features == 0) {
                eml_debug(0, "❌ IF config: invalid dp txt (num_features=0)");
                return false;
            }

            return true;
        }

        bool load_dp_bin(const std::filesystem::path& dp_path) {
            // IFDP v1 binary format used by IF tools/build pipeline:
            // [magic:'IFDP'(4)][version:u8]
            // [quant_bits:u8][num_features:u16][num_samples:u32][num_labels:u16]
            // [problem_type:u8][max_depth:u16][label_counts:u32 * num_labels]
            std::ifstream fin(dp_path, std::ios::binary);
            if (!fin.is_open()) {
                eml_debug(0, "❌ IF config: failed to open dp bin: ", dp_path.string().c_str());
                return false;
            }

            char magic[4] = {0, 0, 0, 0};
            if (!if_config_detail::read_exact(fin, magic, sizeof(magic))) {
                return false;
            }
            if (!(magic[0] == 'I' && magic[1] == 'F' && magic[2] == 'D' && magic[3] == 'P')) {
                eml_debug(1, "⚠️ IF config: dp bin magic mismatch (expected IFDP)");
                return false;
            }

            uint8_t version = 0;
            uint8_t problem_u8 = 0;
            uint16_t max_depth_local = 0;

            if (!if_config_detail::read_exact(fin, &version, sizeof(version)) ||
                !if_config_detail::read_exact(fin, &quantization_bits, sizeof(quantization_bits)) ||
                !if_config_detail::read_exact(fin, &num_features, sizeof(num_features)) ||
                !if_config_detail::read_exact(fin, &num_samples, sizeof(num_samples)) ||
                !if_config_detail::read_exact(fin, &num_labels, sizeof(num_labels)) ||
                !if_config_detail::read_exact(fin, &problem_u8, sizeof(problem_u8)) ||
                !if_config_detail::read_exact(fin, &max_depth_local, sizeof(max_depth_local))) {
                eml_debug(0, "❌ IF config: failed to read dp bin header");
                return false;
            }

            if (version != 1) {
                eml_debug(1, "⚠️ IF config: unknown dp bin version: ", static_cast<int>(version));
            }

            problem = static_cast<problem_type>(problem_u8);
            if (problem == problem_type::UNKNOWN) {
                problem = problem_type::ISOLATION;
            }

            if (max_depth_local > 0) {
                max_depth = max_depth_local;
            }

            samples_per_label.clear();
            samples_per_label.resize(num_labels, 0);
            for (uint16_t i = 0; i < num_labels; ++i) {
                uint32_t count = 0;
                if (!if_config_detail::read_exact(fin, &count, sizeof(count))) {
                    eml_debug(0, "❌ IF config: failed to read dp bin label counts");
                    return false;
                }
                samples_per_label[i] = static_cast<sample_idx_type>(count);
            }

            if (quantization_bits < 1) quantization_bits = 1;
            if (quantization_bits > 8) quantization_bits = 8;

            if (num_features == 0) {
                eml_debug(0, "❌ IF config: invalid dp bin (num_features=0)");
                return false;
            }

            return true;
        }

        bool load_model_engine_config(const std::filesystem::path& config_file) {
            std::string json;
            if (!if_config_detail::read_text_file(config_file, json)) {
                eml_debug(0, "❌ IF config: failed to read model_engine_config.json: ", config_file.string().c_str());
                return false;
            }

            std::string model_type;
            if (if_config_detail::extract_string(json, "model_type", model_type)) {
                if (!model_type.empty() && model_type != "IsolationForest") {
                    eml_debug(1, "⚠️ IF config: model_type is ", model_type.c_str(), ", expected IsolationForest");
                }
            }

            double value = 0.0;
            bool bool_value = false;

            if (if_config_detail::extract_number(json, "n_features", value) && value > 0.0) {
                if (num_features == 0) {
                    num_features = static_cast<uint16_t>(value);
                }
            }

            if (if_config_detail::extract_number(json, "n_estimators", value) && value > 0.0) {
                n_estimators = static_cast<uint16_t>(value);
            }
            if (if_config_detail::extract_number(json, "max_samples", value) && value > 0.0) {
                max_samples = static_cast<float>(value);
            }
            if (if_config_detail::extract_number(json, "max_features", value) && value > 0.0) {
                max_features = static_cast<float>(value);
            }
            if (if_config_detail::extract_bool(json, "bootstrap", bool_value)) {
                bootstrap = bool_value;
            }
            if (if_config_detail::extract_number(json, "contamination", value) && value >= 0.0) {
                contamination = static_cast<float>(value);
            }
            if (if_config_detail::extract_number(json, "random_state", value) && value >= 0.0) {
                random_state = static_cast<uint32_t>(value);
            }

            std::string strategy;
            if (if_config_detail::extract_string(json, "strategy", strategy)) {
                threshold_strategy = strategy;
            }
            if (if_config_detail::extract_number(json, "threshold", value)) {
                decision_threshold = static_cast<float>(value);
            }
            if (if_config_detail::extract_number(json, "fpr_threshold", value)) {
                fpr_threshold = static_cast<float>(value);
            }
            if (if_config_detail::extract_number(json, "val_fpr_target", value)) {
                val_fpr_target = static_cast<float>(value);
            }
            if (if_config_detail::extract_number(json, "val_fpr_delta", value)) {
                val_fpr_delta = static_cast<float>(value);
            }
            if (if_config_detail::extract_number(json, "offset", value)) {
                threshold_offset = static_cast<float>(value);
            }

            scaler_mean.clear();
            scaler_scale.clear();
            if_config_detail::extract_float_array(json, "mean", scaler_mean);
            if_config_detail::extract_float_array(json, "scale", scaler_scale);

            if (scaler_mean.size() == 0 || scaler_scale.size() == 0 || scaler_mean.size() != scaler_scale.size()) {
                eml_debug(0, "❌ IF config: invalid scaler arrays in model_engine_config.json");
                return false;
            }

            if (num_features == 0) {
                num_features = static_cast<uint16_t>(scaler_mean.size());
            }

            if (num_features != scaler_mean.size()) {
                eml_debug_2(0, "❌ IF config: num_features mismatch scaler_mean size ",
                            static_cast<uint32_t>(num_features),
                            " vs ",
                            static_cast<uint32_t>(scaler_mean.size()));
                return false;
            }

            return recompute_node_layout_bits();
        }

    public:
        bool isLoaded = false;

        problem_type problem = problem_type::ISOLATION;

        // Dataset / quantization
        uint8_t quantization_bits = 2;
        uint16_t num_features = 0;
        sample_idx_type num_samples = 0;
        uint16_t num_labels = 0;
        vector<sample_idx_type> samples_per_label;

        // IF hyperparameters
        uint16_t n_estimators = 200;
        float max_samples = 1.0f;
        float max_features = 1.0f;
        bool bootstrap = false;
        float contamination = 0.005f;
        uint32_t random_state = 42;

        // Threshold policy
        std::string threshold_strategy = "tpr";
        float decision_threshold = 0.0f;
        float fpr_threshold = 0.0f;
        float val_fpr_target = 0.0f;
        float val_fpr_delta = 0.0f;
        float threshold_offset = 0.0f;

        // Scaler
        vector<float> scaler_mean;
        vector<float> scaler_scale;

        // Node packing layout controls
        uint8_t threshold_bits = 2;
        uint8_t feature_bits = 0;
        uint8_t child_bits = 1;
        uint8_t leaf_size_bits = 1;
        uint8_t depth_bits = 1;

        uint16_t max_depth = 16;
        uint32_t max_nodes_per_tree = 0;
        uint32_t max_samples_per_tree = 1;

        std::filesystem::path loaded_dp_path;
        std::filesystem::path loaded_config_path;

        If_config() = default;

        explicit If_config(const If_base* base) {
            init(base);
        }

        void init(const If_base* base) {
            base_ptr = base;
            isLoaded = false;
        }

        void set_base(const If_base* base) {
            base_ptr = base;
        }

        bool recompute_node_layout_bits() {
            if (num_features == 0) {
                return false;
            }

            if (quantization_bits < 1) quantization_bits = 1;
            if (quantization_bits > 8) quantization_bits = 8;

            threshold_bits = quantization_bits;
            feature_bits = desired_bits(static_cast<uint32_t>(num_features - 1));
            if (feature_bits == 0) feature_bits = 1;

            uint32_t resolved_samples = static_cast<uint32_t>(num_samples);
            if (resolved_samples == 0) {
                resolved_samples = 1;
            }

            if (max_samples <= 1.0f) {
                float ratio = max_samples;
                if (ratio <= 0.0f) ratio = 1.0f;
                max_samples_per_tree = static_cast<uint32_t>(std::ceil(ratio * static_cast<float>(resolved_samples)));
            } else {
                max_samples_per_tree = static_cast<uint32_t>(std::ceil(max_samples));
            }
            if (max_samples_per_tree == 0) {
                max_samples_per_tree = 1;
            }

            leaf_size_bits = desired_bits(max_samples_per_tree);
            if (leaf_size_bits == 0) leaf_size_bits = 1;

            if (max_depth == 0) {
                max_depth = 16;
            }
            depth_bits = desired_bits(static_cast<uint32_t>(max_depth));
            if (depth_bits == 0) depth_bits = 1;

            uint32_t nodes_by_depth = 0;
            if (max_depth >= 31) {
                nodes_by_depth = std::numeric_limits<uint32_t>::max();
            } else {
                nodes_by_depth = (1u << (static_cast<uint32_t>(max_depth) + 1u)) - 1u;
            }

            uint32_t nodes_by_samples = (resolved_samples <= 1u)
                ? 1u
                : std::min(std::numeric_limits<uint32_t>::max(), resolved_samples * 2u - 1u);

            max_nodes_per_tree = std::min(nodes_by_depth, nodes_by_samples);
            if (max_nodes_per_tree == 0) {
                max_nodes_per_tree = 1;
            }

            child_bits = desired_bits(max_nodes_per_tree - 1u);
            if (child_bits == 0) child_bits = 1;

            return true;
        }

        bool load_scaler_only(const std::filesystem::path& config_path) {
            std::string json;
            if (!if_config_detail::read_text_file(config_path, json)) {
                return false;
            }

            scaler_mean.clear();
            scaler_scale.clear();
            if (!if_config_detail::extract_float_array(json, "mean", scaler_mean)) {
                return false;
            }
            if (!if_config_detail::extract_float_array(json, "scale", scaler_scale)) {
                return false;
            }

            if (scaler_mean.size() == 0 || scaler_mean.size() != scaler_scale.size()) {
                return false;
            }

            num_features = static_cast<uint16_t>(scaler_mean.size());
            loaded_config_path = config_path;
            return true;
        }

        bool load_from_files(const std::filesystem::path& dp_file,
                             const std::filesystem::path& model_engine_config_file) {
            isLoaded = false;

            bool dp_ok = false;
            const std::string ext = dp_file.extension().string();
            if (ext == ".bin") {
                dp_ok = load_dp_bin(dp_file);
            } else {
                dp_ok = load_dp_txt(dp_file);
            }

            if (!dp_ok) {
                return false;
            }

            if (!load_model_engine_config(model_engine_config_file)) {
                return false;
            }

            loaded_dp_path = dp_file;
            loaded_config_path = model_engine_config_file;
            isLoaded = true;
            return true;
        }

        bool load_from_base() {
            if (!base_ptr || !base_ptr->ready_to_use()) {
                eml_debug(0, "❌ IF config: base resource is not ready");
                return false;
            }

            std::filesystem::path dp_path;
            bool dp_ok = false;

            if (base_ptr->dp_bin_exists()) {
                dp_path = base_ptr->get_dp_bin_path();
                dp_ok = load_dp_bin(dp_path);
                if (!dp_ok) {
                    eml_debug(1, "⚠️ IF config: failed to parse _dp.bin, trying _dp.txt fallback");
                }
            }

            if (!dp_ok && base_ptr->dp_txt_exists()) {
                dp_path = base_ptr->get_dp_txt_path();
                dp_ok = load_dp_txt(dp_path);
            }

            if (!dp_ok) {
                eml_debug(0, "❌ IF config: no valid data-params file found");
                return false;
            }

            const std::filesystem::path cfg_path = base_ptr->get_config_path();
            if (!load_model_engine_config(cfg_path)) {
                return false;
            }

            loaded_dp_path = dp_path;
            loaded_config_path = cfg_path;
            isLoaded = true;
            return true;
        }

        size_t memory_usage() const {
            size_t usage = sizeof(*this);
            usage += scaler_mean.size() * sizeof(float);
            usage += scaler_scale.size() * sizeof(float);
            usage += samples_per_label.size() * sizeof(sample_idx_type);
            usage += threshold_strategy.size();
            return usage;
        }
    };

} // namespace eml
