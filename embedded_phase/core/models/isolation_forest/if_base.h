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
        IF_FEATURE_FILE_EXIST   = 1u << 5,
        IF_ABLE_TO_INFERENCE    = 1u << 6,
        IF_ABLE_TO_TRAINING     = 1u << 7,
        IF_SCANNED              = 1u << 8,
        IF_SCALER_FILE_EXIST    = 1u << 9,
        IF_SCHEMA_FILE_EXIST    = 1u << 10
    } If_base_flags;

    /*
    resouces managed by If_base:
    - <model_name>_dp.txt (dataset parameters)                  - If_config
    - <model_name>_qtz.bin (quantizer)                          - If_quantizer
    - <model_name>_optimized_config.json (model engine config)  - If_config
    - <model_name>_optimized_features.json (optimized feature list)         - If_feature_extractor
    - <model_name>_scaler_params.json (feature scaler parameters)           - If_scaler_layer
    - <model_name>_feature_schema.json (feature schema for transform layer) - If_feature_transform_layer
    - <model_name>_iforest.bin (model binary for inference)     - If_tree_container
    - <model_name>_ben_train_nml.bin (benign train dataset for IF) - IsoForest
    */

    class If_base {
    private:
        uint16_t flags = 0;
        char model_name[EML_PATH_BUFFER] = {0};
        std::filesystem::path dir_path = std::filesystem::path(".");

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
            return dir_path / (std::string(model_name) + suffix);
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

        bool has_required_core_resources_internal() const {
            return has_flag(IF_DP_TXT_EXIST)
                && has_flag(IF_CONFIG_FILE_EXIST)
                && has_flag(IF_QTZ_FILE_EXIST)
                && has_flag(IF_SCALER_FILE_EXIST)
                && has_flag(IF_SCHEMA_FILE_EXIST);
        }

        bool has_required_training_resources() const {
            return has_required_core_resources_internal();
        }

        void scan_current_resource() {
            flags = 0;

            if (model_name[0] == '\0') {
                eml_debug(0, "❌ IF resource scan failed: empty model name");
                return;
            }

            const auto nml_path = get_nml_path();
            if (!nml_path.empty() && std::filesystem::exists(nml_path)) {
                set_flag(IF_BASE_DATA_EXIST);
                eml_debug(2, "✅ Found IF benign-train dataset: ", nml_path.string().c_str());
            }

            const auto dp_txt_path = get_dp_txt_path();
            if (std::filesystem::exists(dp_txt_path)) {
                set_flag(IF_DP_TXT_EXIST);
                eml_debug(2, "✅ Found IF data params (txt): ", dp_txt_path.string().c_str());
            } else {
                eml_debug(1, "⚠️ IF data params file not found: ", dp_txt_path.string().c_str());
            }

            const auto qtz_path = get_qtz_path();
            if (std::filesystem::exists(qtz_path)) {
                set_flag(IF_QTZ_FILE_EXIST);
                eml_debug(2, "✅ Found IF quantizer: ", qtz_path.string().c_str());
            } else {
                eml_debug(1, "⚠️ IF quantizer not found: ", qtz_path.string().c_str());
            }

            const auto cfg_path = get_config_path();
            if (std::filesystem::exists(cfg_path)) {
                set_flag(IF_CONFIG_FILE_EXIST);
                eml_debug(2, "✅ Found IF optimized config: ", cfg_path.string().c_str());
            } else {
                eml_debug(1, "⚠️ IF optimized config not found: ", cfg_path.string().c_str());
            }

            const auto features_path = get_feature_config_path();
            if (std::filesystem::exists(features_path)) {
                set_flag(IF_FEATURE_FILE_EXIST);
                eml_debug(2, "✅ Found IF optimized features: ", features_path.string().c_str());
            } else {
                eml_debug(1, "⚠️ IF optimized features not found: ", features_path.string().c_str());
            }

            const auto scaler_path = get_scaler_params_path();
            if (std::filesystem::exists(scaler_path)) {
                set_flag(IF_SCALER_FILE_EXIST);
                eml_debug(2, "✅ Found IF scaler params: ", scaler_path.string().c_str());
            } else {
                eml_debug(1, "⚠️ IF scaler params not found: ", scaler_path.string().c_str());
            }

            const auto schema_path = get_feature_schema_path();
            if (std::filesystem::exists(schema_path)) {
                set_flag(IF_SCHEMA_FILE_EXIST);
                eml_debug(2, "✅ Found IF feature schema: ", schema_path.string().c_str());
            } else {
                eml_debug(1, "⚠️ IF feature schema not found: ", schema_path.string().c_str());
            }

            const auto model_path = get_model_path();
            if (std::filesystem::exists(model_path)) {
                set_flag(IF_MODEL_FILE_EXIST);
                eml_debug(2, "✅ Found IF model binary: ", model_path.string().c_str());
            } else {
                eml_debug(1, "⚠️ IF model binary not found: ", model_path.string().c_str());
            }

            if (has_required_training_resources()) {
                set_flag(IF_ABLE_TO_TRAINING);
            }

            if (has_required_training_resources() && has_flag(IF_MODEL_FILE_EXIST)) {
                set_flag(IF_ABLE_TO_INFERENCE);
            }

            set_flag(IF_SCANNED);
        }

    public:
        If_base() = default;

        explicit If_base(const char* model_name_input,
                         const std::filesystem::path& dir_path_input = std::filesystem::path("."),
                         const std::filesystem::path& config_path_input = {}) {
            (void)config_path_input;
            init(model_name_input, dir_path_input);
        }

        void init(const char* model_name_input,
                  const std::filesystem::path& dir_path_input = std::filesystem::path("."),
                  const std::filesystem::path& config_path_input = {}) {
            (void)config_path_input;

            if (!model_name_input || std::strlen(model_name_input) == 0) {
                model_name[0] = '\0';
                flags = 0;
                eml_debug(0, "❌ IF base init failed: empty model name");
                return;
            }

            std::strncpy(model_name, model_name_input, EML_PATH_BUFFER - 1);
            model_name[EML_PATH_BUFFER - 1] = '\0';
            dir_path = dir_path_input;

            scan_current_resource();
        }

        void set_dir_path(const std::filesystem::path& dir) {
            dir_path = dir;
            if (model_name[0] != '\0') {
                scan_current_resource();
            }
        }

        void set_resource_dir(const std::filesystem::path& dir) {
            set_dir_path(dir);
        }

        void set_config_path(const std::filesystem::path& path) {
            (void)path;
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

        std::filesystem::path get_nml_path() const {
            return build_model_artifact_path("_ben_train_nml.bin");
        }

        std::filesystem::path get_benign_train_nml_path() const {
            return build_model_artifact_path("_ben_train_nml.bin");
        }

        std::filesystem::path get_benign_val_nml_path() const {
            return build_model_artifact_path("_ben_val_nml.bin");
        }

        std::filesystem::path get_malware_val_nml_path() const {
            return build_model_artifact_path("_mal_val_nml.bin");
        }

        std::filesystem::path get_qtz_path() const { return build_model_artifact_path("_qtz.bin"); }
        std::filesystem::path get_dp_bin_path() const { return build_model_artifact_path("_dp.bin"); }
        std::filesystem::path get_dp_txt_path() const { return build_model_artifact_path("_dp.txt"); }
        std::filesystem::path get_model_path() const { return build_model_artifact_path("_iforest.bin"); }

        std::filesystem::path get_iforest_bin_path() const {
            return build_model_artifact_path("_iforest.bin");
        }

        std::filesystem::path get_config_path() const {
            return build_model_artifact_path("_optimized_config.json");
        }

        std::filesystem::path get_scaler_params_path() const {
            return build_model_artifact_path("_scaler_params.json");
        }

        std::filesystem::path get_feature_config_path() const {
            return build_model_artifact_path("_optimized_features.json");
        }

        std::filesystem::path get_feature_schema_path() const {
            return build_model_artifact_path("_feature_schema.json");
        }

        const std::filesystem::path& get_resource_dir() const { return dir_path; }
        const std::filesystem::path& get_dir_path() const { return dir_path; }

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
        void get_feature_config_path(char* buffer, size_t buffer_size) const {
            write_path_to_buffer(get_feature_config_path(), buffer, buffer_size);
        }
        void get_scaler_params_path(char* buffer, size_t buffer_size) const {
            write_path_to_buffer(get_scaler_params_path(), buffer, buffer_size);
        }
        void get_feature_schema_path(char* buffer, size_t buffer_size) const {
            write_path_to_buffer(get_feature_schema_path(), buffer, buffer_size);
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
        bool feature_config_exists() const { return has_flag(IF_FEATURE_FILE_EXIST); }
        bool scaler_params_exists() const { return has_flag(IF_SCALER_FILE_EXIST); }
        bool feature_schema_exists() const { return has_flag(IF_SCHEMA_FILE_EXIST); }
        bool model_exists() const { return has_flag(IF_MODEL_FILE_EXIST); }
        bool has_required_core_resources() const { return has_required_core_resources_internal(); }
        bool ready_for_training() const { return has_flag(IF_ABLE_TO_TRAINING); }
        bool ready_for_inference() const { return has_flag(IF_ABLE_TO_INFERENCE); }

        uint16_t status_flags() const { return flags; }
    };

} // namespace eml
