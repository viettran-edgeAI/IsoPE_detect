#pragma once

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include "../../base/eml_base.h"
namespace eml {

    class If_scaler_layer {
    private:
        vector<float> means_;
        vector<float> scales_;
        uint16_t num_features_ = 0;
        bool loaded_ = false;
        float min_scale_epsilon_ = 1e-12f;
        mutable eml_status_code last_status_code_ = eml_status_code::ok;

        inline void set_status(eml_status_code status) const {
            last_status_code_ = status;
        }

        static bool read_text_file(const std::filesystem::path& file_path, std::string& out) {
            std::ifstream fin(file_path, std::ios::in);
            if (!fin.is_open()) {
                return false;
            }
            out.assign((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
            return true;
        }

        static bool extract_array_payload(const std::string& json,
                                          const std::string& key,
                                          std::string& out_payload) {
            const size_t key_pos = json.find(std::string("\"") + key + "\"");
            if (key_pos == std::string::npos) {
                return false;
            }

            const size_t open = json.find('[', key_pos);
            if (open == std::string::npos) {
                return false;
            }

            int depth = 0;
            size_t close = open;
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

            if (close <= open || close == std::string::npos) {
                return false;
            }

            out_payload = json.substr(open + 1, close - open - 1);
            return true;
        }

        static bool parse_float_array(const std::string& json,
                                      const std::string& key,
                                      vector<float>& out_values) {
            out_values.clear();

            std::string payload;
            if (!extract_array_payload(json, key, payload)) {
                return false;
            }

            size_t position = 0;
            while (position < payload.size()) {
                while (position < payload.size() &&
                       (payload[position] == ' ' || payload[position] == '\t' || payload[position] == '\n' || payload[position] == '\r' || payload[position] == ',')) {
                    ++position;
                }
                if (position >= payload.size()) {
                    break;
                }

                size_t end = position;
                while (end < payload.size() && payload[end] != ',') {
                    ++end;
                }

                const std::string token = payload.substr(position, end - position);
                try {
                    out_values.push_back(std::stof(token));
                } catch (...) {
                    return false;
                }
                position = end + 1;
            }

            return !out_values.empty();
        }

    public:
        If_scaler_layer() = default;

        bool init_from_file(const std::filesystem::path& scaler_params_path,
                            uint16_t expected_num_features = 0u) {
            set_status(eml_status_code::ok);
            std::string json;
            if (!read_text_file(scaler_params_path, json)) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }

            vector<float> means;
            vector<float> scales;
            if (!parse_float_array(json, "mean", means) || !parse_float_array(json, "scale", scales)) {
                set_status(eml_status_code::json_parse_failed);
                return false;
            }

            if (expected_num_features > 0u && means.size() != expected_num_features) {
                set_status(eml_status_code::feature_count_mismatch);
                return false;
            }

            return init(means, scales);
        }

        bool init(const vector<float>& means, const vector<float>& scales) {
            loaded_ = false;
            set_status(eml_status_code::ok);

            if (means.size() == 0 || means.size() != scales.size()) {
                set_status(eml_status_code::size_mismatch);
                return false;
            }

            means_ = means;
            scales_ = scales;
            num_features_ = static_cast<uint16_t>(means_.size());
            loaded_ = true;
            set_status(eml_status_code::ok);
            return true;
        }

        void release() {
            means_.clear();
            scales_.clear();
            means_.shrink_to_fit();
            scales_.shrink_to_fit();
            num_features_ = 0;
            loaded_ = false;
            set_status(eml_status_code::ok);
        }

        bool loaded() const { return loaded_; }
        uint16_t num_features() const { return num_features_; }

        void set_min_scale_epsilon(float epsilon) {
            min_scale_epsilon_ = (epsilon > 0.0f) ? epsilon : 1e-12f;
        }

        bool transform(const float* in_features, uint16_t feature_count, float* out_features) const {
            if (!loaded_) {
                set_status(eml_status_code::not_loaded);
                return false;
            }
            if (!in_features || !out_features) {
                set_status(eml_status_code::invalid_argument);
                return false;
            }
            if (feature_count != num_features_) {
                set_status(eml_status_code::size_mismatch);
                return false;
            }

            for (uint16_t feature_index = 0; feature_index < num_features_; ++feature_index) {
                const float sigma = scales_[feature_index];
                const float denom = (std::fabs(sigma) < min_scale_epsilon_)
                    ? (sigma < 0.0f ? -min_scale_epsilon_ : min_scale_epsilon_)
                    : sigma;
                out_features[feature_index] = (in_features[feature_index] - means_[feature_index]) / denom;
            }
            set_status(eml_status_code::ok);
            return true;
        }

        bool transform_inplace(float* features, uint16_t feature_count) const {
            if (!loaded_) {
                set_status(eml_status_code::not_loaded);
                return false;
            }
            if (!features) {
                set_status(eml_status_code::invalid_argument);
                return false;
            }
            if (feature_count != num_features_) {
                set_status(eml_status_code::size_mismatch);
                return false;
            }

            for (uint16_t feature_index = 0; feature_index < num_features_; ++feature_index) {
                const float sigma = scales_[feature_index];
                const float denom = (std::fabs(sigma) < min_scale_epsilon_)
                    ? (sigma < 0.0f ? -min_scale_epsilon_ : min_scale_epsilon_)
                    : sigma;
                features[feature_index] = (features[feature_index] - means_[feature_index]) / denom;
            }
            set_status(eml_status_code::ok);
            return true;
        }

        const vector<float>& means() const { return means_; }
        const vector<float>& scales() const { return scales_; }
        eml_status_code last_status() const { return last_status_code_; }
        void clear_status() { set_status(eml_status_code::ok); }
    };

} // namespace eml
