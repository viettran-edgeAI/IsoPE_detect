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

            if (feature_config_path.empty()) {
                eml_debug(0, "❌ IF feature_extractor init failed: empty feature config path");
                return false;
            }

            std::string json;
            if (!read_text_file(feature_config_path, json)) {
                eml_debug(0, "❌ IF feature_extractor init failed: cannot read feature config: ", feature_config_path.string().c_str());
                return false;
            }

            if (!parse_feature_list_json(json, feature_names_)) {
                eml_debug(0, "❌ IF feature_extractor init failed: invalid feature list JSON");
                return false;
            }

            if (expected_num_features > 0 && feature_names_.size() != expected_num_features) {
                eml_debug_2(0,
                            "❌ IF feature_extractor init failed: feature count mismatch ",
                            static_cast<uint32_t>(feature_names_.size()),
                            " vs expected ",
                            static_cast<uint32_t>(expected_num_features));
                return false;
            }

            feature_config_path_ = feature_config_path;
            loaded_ = true;
            return true;
        }

        bool init_from_optimized_config(const std::filesystem::path& optimized_config_path,
                                        uint16_t expected_num_features = 0) {
            loaded_ = false;
            feature_names_.clear();
            feature_config_path_.clear();

            if (optimized_config_path.empty()) {
                eml_debug(0, "❌ IF feature_extractor init failed: empty optimized config path");
                return false;
            }

            std::string json;
            if (!read_text_file(optimized_config_path, json)) {
                eml_debug(0, "❌ IF feature_extractor init failed: cannot read optimized config: ", optimized_config_path.string().c_str());
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
                        eml_debug(0, "❌ IF feature_extractor init failed: cannot parse source feature list from optimized config");
                        return false;
                    }
                } else {
                    eml_debug(0, "❌ IF feature_extractor init failed: missing optimized_feature_set.features");
                    return false;
                }
            }

            if (expected_num_features > 0 && feature_names_.size() != expected_num_features) {
                eml_debug_2(0,
                            "❌ IF feature_extractor init failed: feature count mismatch ",
                            static_cast<uint32_t>(feature_names_.size()),
                            " vs expected ",
                            static_cast<uint32_t>(expected_num_features));
                return false;
            }

            feature_config_path_ = optimized_config_path;
            loaded_ = true;
            return true;
        }

        void set_extract_callback(extract_callback_t callback) {
            extractor_callback_ = std::move(callback);
        }

        void set_extract_content_callback(extract_content_callback_t callback) {
            extractor_content_callback_ = std::move(callback);
        }

        bool extract_from_pe(const std::filesystem::path& pe_path, vector<float>& out_features) const {
            out_features.clear();

            if (!loaded_) {
                return false;
            }

            if (!extractor_callback_) {
                return false;
            }

            if (!extractor_callback_(pe_path, feature_names_, out_features)) {
                out_features.clear();
                return false;
            }

            if (out_features.size() != feature_names_.size()) {
                out_features.clear();
                return false;
            }

            return true;
        }

        bool extract_from_pe_content(const uint8_t* pe_content,
                                     size_t pe_size,
                                     vector<float>& out_features) const {
            out_features.clear();

            if (!loaded_ || !pe_content || pe_size == 0u) {
                return false;
            }

            if (!extractor_content_callback_) {
                return false;
            }

            if (!extractor_content_callback_(pe_content, pe_size, feature_names_, out_features)) {
                out_features.clear();
                return false;
            }

            if (out_features.size() != feature_names_.size()) {
                out_features.clear();
                return false;
            }

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
        }

        bool loaded() const { return loaded_; }
        size_t feature_count() const { return feature_names_.size(); }
        const vector<std::string>& feature_names() const { return feature_names_; }
        const std::filesystem::path& feature_config_path() const { return feature_config_path_; }
    };

} // namespace eml
