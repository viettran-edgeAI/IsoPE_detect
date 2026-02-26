#pragma once

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include "../../base/eml_base.h"
namespace eml {

    class If_feature_transform_layer {
    private:
        vector<uint8_t> log1p_abs_mask_;
        uint16_t num_features_ = 0;
        bool loaded_ = false;
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

        static bool parse_string_array(const std::string& json,
                                       const std::string& key,
                                       vector<std::string>& out,
                                       size_t from = 0) {
            out.clear();

            const size_t key_pos = json.find(std::string("\"") + key + "\"", from);
            if (key_pos == std::string::npos) {
                return false;
            }

            const size_t open = json.find('[', key_pos);
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

            const std::string_view payload(json.c_str() + open + 1, close - open - 1);
            size_t position = 0;
            while (position < payload.size()) {
                const size_t q1 = payload.find('"', position);
                if (q1 == std::string_view::npos) {
                    break;
                }
                const size_t q2 = payload.find('"', q1 + 1);
                if (q2 == std::string_view::npos || q2 <= q1 + 1) {
                    break;
                }
                out.push_back(std::string(payload.substr(q1 + 1, q2 - q1 - 1)));
                position = q2 + 1;
            }

            return true;
        }

        static int find_feature_index(const vector<std::string>& feature_order, const std::string& name) {
            for (size_t index = 0; index < feature_order.size(); ++index) {
                if (feature_order[index] == name) {
                    return static_cast<int>(index);
                }
            }
            return -1;
        }

    public:
        If_feature_transform_layer() = default;

        bool init_passthrough(uint16_t feature_count) {
            if (feature_count == 0u) {
                set_status(eml_status_code::invalid_argument);
                return false;
            }

            num_features_ = feature_count;
            log1p_abs_mask_.assign(feature_count, 0u);
            loaded_ = true;
            set_status(eml_status_code::ok);
            return true;
        }

        bool init_from_feature_schema(const std::filesystem::path& schema_path,
                                      const vector<std::string>& feature_order,
                                      uint16_t expected_num_features = 0u) {
            loaded_ = false;
            log1p_abs_mask_.clear();
            num_features_ = 0u;
            set_status(eml_status_code::ok);

            if (schema_path.empty()) {
                set_status(eml_status_code::empty_path);
                return false;
            }

            std::string schema_json;
            if (!read_text_file(schema_path, schema_json)) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }

            vector<std::string> schema_order;
            (void)parse_string_array(schema_json, "feature_order", schema_order);

            const vector<std::string>* effective_order = nullptr;
            if (!feature_order.empty()) {
                effective_order = &feature_order;
            } else if (!schema_order.empty()) {
                effective_order = &schema_order;
            } else {
                set_status(eml_status_code::invalid_configuration);
                return false;
            }

            num_features_ = static_cast<uint16_t>(effective_order->size());
            if (expected_num_features > 0u && num_features_ != expected_num_features) {
                set_status(eml_status_code::feature_count_mismatch);
                return false;
            }

            log1p_abs_mask_.assign(num_features_, 0u);

            vector<std::string> log_features;
            if (parse_string_array(schema_json, "log_transform_features", log_features)) {
                for (const auto& feature_name : log_features) {
                    const int feature_index = find_feature_index(*effective_order, feature_name);
                    if (feature_index >= 0) {
                        log1p_abs_mask_[static_cast<size_t>(feature_index)] = 1u;
                    }
                }
            }

            loaded_ = true;
            set_status(eml_status_code::ok);
            return true;
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

            for (uint16_t feature_index = 0; feature_index < feature_count; ++feature_index) {
                float value = in_features[feature_index];
                if (log1p_abs_mask_[feature_index] != 0u) {
                    value = std::log1p(std::fabs(value));
                }
                out_features[feature_index] = value;
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

            for (uint16_t feature_index = 0; feature_index < feature_count; ++feature_index) {
                if (log1p_abs_mask_[feature_index] != 0u) {
                    features[feature_index] = std::log1p(std::fabs(features[feature_index]));
                }
            }

            set_status(eml_status_code::ok);
            return true;
        }

        void release() {
            log1p_abs_mask_.clear();
            log1p_abs_mask_.shrink_to_fit();
            num_features_ = 0u;
            loaded_ = false;
            set_status(eml_status_code::ok);
        }

        bool loaded() const { return loaded_; }
        uint16_t num_features() const { return num_features_; }
        eml_status_code last_status() const { return last_status_code_; }
        void clear_status() { set_status(eml_status_code::ok); }
    };

} // namespace eml
