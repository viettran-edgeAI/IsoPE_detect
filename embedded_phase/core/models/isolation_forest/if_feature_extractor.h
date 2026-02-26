#pragma once

#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <string_view>

#include "../../base/eml_base.h"

namespace eml {

    class If_feature_extractor {
    public:
        using extract_callback_t = std::function<bool(const std::filesystem::path&,
                                                      const vector<std::string>&,
                                                      vector<float>&)>;
        using extract_content_callback_t = std::function<bool(const uint8_t*,
                                                              size_t,
                                                              const vector<std::string>&,
                                                              vector<float>&)>;

    private:
        std::filesystem::path feature_config_path_;
        vector<std::string> feature_names_;
        extract_callback_t extractor_callback_;
        extract_content_callback_t extractor_content_callback_;
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

        static bool parse_feature_list_json(const std::string& json, vector<std::string>& out) {
            out.clear();

            const size_t open = json.find('[');
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

            if (close <= open || close == std::string::npos) {
                return false;
            }

            const std::string_view payload(json.c_str() + open + 1, close - open - 1);
            size_t pos = 0;
            while (pos < payload.size()) {
                const size_t q1 = payload.find('"', pos);
                if (q1 == std::string_view::npos) {
                    break;
                }
                const size_t q2 = payload.find('"', q1 + 1);
                if (q2 == std::string_view::npos || q2 <= q1 + 1) {
                    break;
                }

                out.push_back(std::string(payload.substr(q1 + 1, q2 - q1 - 1)));
                pos = q2 + 1;
            }

            return !out.empty();
        }

        static bool parse_string_array_by_key(const std::string& json,
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
            size_t pos = 0;
            while (pos < payload.size()) {
                const size_t q1 = payload.find('"', pos);
                if (q1 == std::string_view::npos) {
                    break;
                }
                const size_t q2 = payload.find('"', q1 + 1);
                if (q2 == std::string_view::npos || q2 <= q1 + 1) {
                    break;
                }

                out.push_back(std::string(payload.substr(q1 + 1, q2 - q1 - 1)));
                pos = q2 + 1;
            }

            return !out.empty();
        }

        static bool parse_string_by_key(const std::string& json,
                                        const std::string& key,
                                        std::string& out,
                                        size_t from = 0) {
            const size_t key_pos = json.find(std::string("\"") + key + "\"", from);
            if (key_pos == std::string::npos) {
                return false;
            }

            const size_t colon = json.find(':', key_pos);
            if (colon == std::string::npos) {
                return false;
            }

            const size_t q1 = json.find('"', colon + 1);
            if (q1 == std::string::npos) {
                return false;
            }

            const size_t q2 = json.find('"', q1 + 1);
            if (q2 == std::string::npos || q2 <= q1 + 1) {
                return false;
            }

            out = json.substr(q1 + 1, q2 - q1 - 1);
            return true;
        }

    public:
        If_feature_extractor() = default;

        bool init(const std::filesystem::path& feature_config_path,
                  uint16_t expected_num_features = 0) {
            loaded_ = false;
            feature_names_.clear();
            feature_config_path_.clear();
            set_status(eml_status_code::ok);

            if (feature_config_path.empty()) {
                set_status(eml_status_code::empty_path);
                return false;
            }

            std::string json;
            if (!read_text_file(feature_config_path, json)) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }

            if (!parse_feature_list_json(json, feature_names_)) {
                set_status(eml_status_code::json_parse_failed);
                return false;
            }

            if (expected_num_features > 0 && feature_names_.size() != expected_num_features) {
                set_status(eml_status_code::feature_count_mismatch);
                return false;
            }

            feature_config_path_ = feature_config_path;
            loaded_ = true;
            set_status(eml_status_code::ok);
            return true;
        }

        bool init_from_optimized_config(const std::filesystem::path& optimized_config_path,
                                        uint16_t expected_num_features = 0) {
            loaded_ = false;
            feature_names_.clear();
            feature_config_path_.clear();
            set_status(eml_status_code::ok);

            if (optimized_config_path.empty()) {
                set_status(eml_status_code::empty_path);
                return false;
            }

            std::string json;
            if (!read_text_file(optimized_config_path, json)) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }

            const size_t optimized_feature_set_pos = json.find("\"optimized_feature_set\"");
            if (!parse_string_array_by_key(json, "features", feature_names_, optimized_feature_set_pos)) {
                std::string source_file;
                if (parse_string_by_key(json, "source_file", source_file, optimized_feature_set_pos)) {
                    const std::filesystem::path source_path = std::filesystem::path(source_file);
                    std::filesystem::path resolved = source_path;
                    if (resolved.is_relative()) {
                        resolved = optimized_config_path.parent_path() / resolved;
                    }

                    std::string list_json;
                    if (!read_text_file(resolved, list_json) || !parse_feature_list_json(list_json, feature_names_)) {
                        set_status(eml_status_code::json_parse_failed);
                        return false;
                    }
                } else {
                    set_status(eml_status_code::invalid_configuration);
                    return false;
                }
            }

            if (expected_num_features > 0 && feature_names_.size() != expected_num_features) {
                set_status(eml_status_code::feature_count_mismatch);
                return false;
            }

            feature_config_path_ = optimized_config_path;
            loaded_ = true;
            set_status(eml_status_code::ok);
            return true;
        }

        void set_extract_callback(extract_callback_t callback) {
            extractor_callback_ = std::move(callback);
        }

        void set_extract_content_callback(extract_content_callback_t callback) {
            extractor_content_callback_ = std::move(callback);
        }

        /// Initialize using a runtime-provided feature list (no file I/O).
        bool init_from_feature_list(const vector<std::string>& features,
                                    uint16_t expected_num_features = 0) {
            loaded_ = false;
            feature_names_.clear();
            feature_config_path_.clear();
            set_status(eml_status_code::ok);

            if (features.empty()) {
                set_status(eml_status_code::invalid_argument);
                return false;
            }

            feature_names_ = features;
            if (expected_num_features > 0 && feature_names_.size() != expected_num_features) {
                set_status(eml_status_code::feature_count_mismatch);
                feature_names_.clear();
                return false;
            }

            loaded_ = true;
            set_status(eml_status_code::ok);
            return true;
        }

        bool extract_from_pe(const std::filesystem::path& pe_path, vector<float>& out_features) const {
            out_features.clear();

            if (!loaded_) {
                set_status(eml_status_code::not_loaded);
                return false;
            }

            if (!extractor_callback_) {
                set_status(eml_status_code::callback_not_set);
                return false;
            }

            if (!extractor_callback_(pe_path, feature_names_, out_features)) {
                out_features.clear();
                set_status(eml_status_code::callback_failed);
                return false;
            }

            if (out_features.size() != feature_names_.size()) {
                out_features.clear();
                set_status(eml_status_code::size_mismatch);
                return false;
            }

            set_status(eml_status_code::ok);
            return true;
        }

        bool extract_from_pe_content(const uint8_t* pe_content,
                                     size_t pe_size,
                                     vector<float>& out_features) const {
            out_features.clear();

            if (!loaded_) {
                set_status(eml_status_code::not_loaded);
                return false;
            }

            if (!pe_content || pe_size == 0u) {
                set_status(eml_status_code::invalid_argument);
                return false;
            }

            if (!extractor_content_callback_) {
                set_status(eml_status_code::callback_not_set);
                return false;
            }

            if (!extractor_content_callback_(pe_content, pe_size, feature_names_, out_features)) {
                out_features.clear();
                set_status(eml_status_code::callback_failed);
                return false;
            }

            if (out_features.size() != feature_names_.size()) {
                out_features.clear();
                set_status(eml_status_code::size_mismatch);
                return false;
            }

            set_status(eml_status_code::ok);
            return true;
        }

        bool validate_pre_extracted(const vector<float>& features) const {
            return loaded_ && !feature_names_.empty() && features.size() == feature_names_.size();
        }

        void release() {
            feature_names_.clear();
            feature_names_.shrink_to_fit();
            feature_config_path_.clear();
            extractor_callback_ = nullptr;
            extractor_content_callback_ = nullptr;
            loaded_ = false;
            set_status(eml_status_code::ok);
        }

        bool loaded() const { return loaded_; }
        size_t feature_count() const { return feature_names_.size(); }
        const vector<std::string>& feature_names() const { return feature_names_; }
        const std::filesystem::path& feature_config_path() const { return feature_config_path_; }
        eml_status_code last_status() const { return last_status_code_; }
        void clear_status() { set_status(eml_status_code::ok); }
    };

} // namespace eml
