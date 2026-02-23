#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>

#include "../../base/eml_base.h"

namespace eml {

    typedef enum If_base_flags : uint16_t {
        IF_BASE_DATA_EXIST      = 1u << 0,
        IF_DP_TXT_EXIST         = 1u << 1,
        IF_QTZ_FILE_EXIST       = 1u << 2,
        IF_CONFIG_FILE_EXIST    = 1u << 3,
        IF_MODEL_FILE_EXIST     = 1u << 4,
        IF_ABLE_TO_INFERENCE    = 1u << 5,
        IF_ABLE_TO_TRAINING     = 1u << 6,
        IF_SCANNED              = 1u << 7
    } If_base_flags;

    class If_base {
    private:
        uint16_t flags = 0;
        char model_name[EML_PATH_BUFFER] = {0};

        std::filesystem::path resource_dir =
            std::filesystem::path("embedded_phase/tools/data_quantization/quantized_datasets");
        // config_path is derived from model_name at init() when left empty (default).
        // Format: development_phase/results/{model_name}_optimized_config.json
        std::filesystem::path config_path;

        inline void set_flag(If_base_flags f) {
            flags |= static_cast<uint16_t>(f);
        }

        inline bool has_flag(If_base_flags f) const {
            return (flags & static_cast<uint16_t>(f)) != 0;
        }

        std::filesystem::path build_model_artifact_path(const char* suffix) const {
            if (model_name[0] == '\0') {
                return {};
            }
            return resource_dir / (std::string(model_name) + suffix);
        }

        static void write_path_to_buffer(const std::filesystem::path& p,
                                         char* buffer,
                                         size_t buffer_size) {
            if (!buffer || buffer_size == 0) {
                return;
            }
            const std::string path_str = p.string();
            std::strncpy(buffer, path_str.c_str(), buffer_size - 1);
            buffer[buffer_size - 1] = '\0';
        }

        void scan_current_resource() {
            flags = 0;

            if (model_name[0] == '\0') {
                eml_debug(0, "❌ IF resource scan failed: empty model name");
                return;
            }

            const auto nml_path = get_nml_path();
            if (std::filesystem::exists(nml_path)) {
                set_flag(IF_BASE_DATA_EXIST);
                eml_debug(2, "✅ Found IF base dataset: ", nml_path.string().c_str());
            } else {
                eml_debug(1, "⚠️ IF base dataset not found: ", nml_path.string().c_str());
            }

            const auto dp_txt_path = get_dp_txt_path();
            if (std::filesystem::exists(dp_txt_path)) {
                set_flag(IF_DP_TXT_EXIST);
                eml_debug(2, "✅ Found IF data params (txt): ", dp_txt_path.string().c_str());
            }

            if (!has_flag(IF_DP_TXT_EXIST)) {
                eml_debug(1, "⚠️ IF data params file not found (_dp.bin or _dp.txt)");
            }

            const auto qtz_path = get_qtz_path();
            if (std::filesystem::exists(qtz_path)) {
                set_flag(IF_QTZ_FILE_EXIST);
                eml_debug(2, "✅ Found IF quantizer: ", qtz_path.string().c_str());
            } else {
                eml_debug(1, "⚠️ IF quantizer not found: ", qtz_path.string().c_str());
            }

            if (!config_path.empty() && std::filesystem::exists(config_path)) {
                set_flag(IF_CONFIG_FILE_EXIST);
                eml_debug(2, "✅ Found model engine config: ", config_path.string().c_str());
            } else {
                eml_debug(1, "⚠️ optimized_config.json not found: ", config_path.string().c_str());
            }

            const auto model_path = get_model_path();
            if (std::filesystem::exists(model_path)) {
                set_flag(IF_MODEL_FILE_EXIST);
                eml_debug(2, "✅ Found IF model binary: ", model_path.string().c_str());
            }

            if (has_flag(IF_QTZ_FILE_EXIST) && has_flag(IF_CONFIG_FILE_EXIST) && has_flag(IF_MODEL_FILE_EXIST)) {
                set_flag(IF_ABLE_TO_INFERENCE);
            }

            if (has_flag(IF_BASE_DATA_EXIST) && has_flag(IF_QTZ_FILE_EXIST) &&
                 has_flag(IF_DP_TXT_EXIST) && has_flag(IF_CONFIG_FILE_EXIST)) {
                set_flag(IF_ABLE_TO_TRAINING);
            }

            set_flag(IF_SCANNED);
        }

    public:
        If_base() = default;

        explicit If_base(const char* model_name_input,
                         const std::filesystem::path& resource_dir_input =
                             std::filesystem::path("embedded_phase/tools/data_quantization/quantized_datasets"),
                         const std::filesystem::path& config_path_input = {}) {
            init(model_name_input, resource_dir_input, config_path_input);
        }

        void init(const char* model_name_input,
                  const std::filesystem::path& resource_dir_input =
                      std::filesystem::path("embedded_phase/tools/data_quantization/quantized_datasets"),
                  const std::filesystem::path& config_path_input = {}) {
            if (!model_name_input || std::strlen(model_name_input) == 0) {
                model_name[0] = '\0';
                flags = 0;
                eml_debug(0, "❌ IF base init failed: empty model name");
                return;
            }

            std::strncpy(model_name, model_name_input, EML_PATH_BUFFER - 1);
            model_name[EML_PATH_BUFFER - 1] = '\0';

            resource_dir = resource_dir_input;
            // Derive config path from model_name when caller uses default (empty path)
            if (config_path_input.empty()) {
                config_path = std::filesystem::path("development_phase/results/")
                              / (std::string(model_name) + "_optimized_config.json");
            } else {
                config_path = config_path_input;
            }

            scan_current_resource();
        }

        void set_resource_dir(const std::filesystem::path& dir) {
            resource_dir = dir;
            if (model_name[0] != '\0') {
                scan_current_resource();
            }
        }

        void set_config_path(const std::filesystem::path& path) {
            config_path = path;
            if (model_name[0] != '\0') {
                scan_current_resource();
            }
        }

        void update_resource_status() {
            if (model_name[0] == '\0') {
                return;
            }
            scan_current_resource();
        }

        std::filesystem::path get_nml_path() const { return build_model_artifact_path("_nml.bin"); }
        std::filesystem::path get_qtz_path() const { return build_model_artifact_path("_qtz.bin"); }
        std::filesystem::path get_dp_bin_path() const { return build_model_artifact_path("_dp.bin"); }
        std::filesystem::path get_dp_txt_path() const { return build_model_artifact_path("_dp.txt"); }
        std::filesystem::path get_model_path() const { return build_model_artifact_path("_if.bin"); }
        const std::filesystem::path& get_config_path() const { return config_path; }
        const std::filesystem::path& get_resource_dir() const { return resource_dir; }

        void get_nml_path(char* buffer, size_t buffer_size) const {
            write_path_to_buffer(get_nml_path(), buffer, buffer_size);
        }
        void get_qtz_path(char* buffer, size_t buffer_size) const {
            write_path_to_buffer(get_qtz_path(), buffer, buffer_size);
        }
        void get_dp_bin_path(char* buffer, size_t buffer_size) const {
            write_path_to_buffer(get_dp_bin_path(), buffer, buffer_size);
        }
        void get_dp_txt_path(char* buffer, size_t buffer_size) const {
            write_path_to_buffer(get_dp_txt_path(), buffer, buffer_size);
        }
        void get_model_path(char* buffer, size_t buffer_size) const {
            write_path_to_buffer(get_model_path(), buffer, buffer_size);
        }
        void get_config_path(char* buffer, size_t buffer_size) const {
            write_path_to_buffer(get_config_path(), buffer, buffer_size);
        }

        void get_model_name(char* buffer, size_t buffer_size) const {
            if (!buffer || buffer_size == 0) {
                return;
            }
            std::strncpy(buffer, model_name, buffer_size - 1);
            buffer[buffer_size - 1] = '\0';
        }

        const char* model_name_cstr() const { return model_name; }

        bool ready_to_use() const { return model_name[0] != '\0' && has_flag(IF_SCANNED); }
        bool nml_exists() const { return has_flag(IF_BASE_DATA_EXIST); }
        bool qtz_exists() const { return has_flag(IF_QTZ_FILE_EXIST); }
        bool dp_txt_exists() const { return has_flag(IF_DP_TXT_EXIST); }
        bool config_exists() const { return has_flag(IF_CONFIG_FILE_EXIST); }
        bool model_exists() const { return has_flag(IF_MODEL_FILE_EXIST); }
        bool ready_for_training() const { return has_flag(IF_ABLE_TO_TRAINING); }
        bool ready_for_inference() const { return has_flag(IF_ABLE_TO_INFERENCE); }

        uint16_t status_flags() const { return flags; }
    };

} // namespace eml
