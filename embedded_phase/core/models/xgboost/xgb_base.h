#pragma once

/**
 * @file xgb_base.h
 * @brief XGBoost base resource manager for MCU
 * 
 * This header provides the XG_base class which manages model files and validates
 * resource availability for XGBoost models on microcontrollers.
 * 
 * Similar to Rf_base for Random Forest, XG_base:
 * - Scans and validates required model files
 * - Provides file path builders for all XGBoost components
 * - Tracks resource availability flags for inference/training readiness
 */

#include "../../containers/STL_MCU.h"
#include "../../base/eml_base.h"
#include "../../../Rf_file_manager.h"
#include "../../../Rf_board_config.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace eml {

    // XGBoost-specific constants
    static constexpr uint8_t  XG_PATH_BUFFER       = 64;
    static constexpr uint8_t  XG_MAX_LABEL_LENGTH  = 32;
    static constexpr uint16_t XG_MAX_LABELS        = 1024;
    static constexpr uint16_t XG_MAX_FEATURES      = 4096;
    static constexpr uint16_t XG_MAX_BOOST_ROUNDS  = 500;
    static constexpr uint8_t  XG_MAX_DEPTH         = 32;

    // XGBoost model file magic number: "XGMC" (XGBoost MCU Compact)
    static constexpr uint32_t XG_MODEL_MAGIC   = 0x58474D43;
    static constexpr uint32_t XG_MODEL_VERSION = 2;

    /*
    NOTE : XGBoost file components (with each model)
        1.  model_name_nml.bin       : base data (dataset)
        2.  model_name_xgb_config.json: model configuration file 
        3.  model_name_qtz.bin       : quantizer binary (feature quantizer, label mapping, outlier filtering flag)
        4.  model_name_xgb.bin       : model file (all trees) in unified format
        5.  model_name_hogcfg.json   : HOG & Camera configuration file (for hog_transform - model preprocessing)
        6.  model_name_ifl.bin       : inference log file (predictions, actual labels, metrics over time)
        7.  model_name_tlog.csv      : time log file (detailed timing of forest events)
        8.  model_name_mlog.csv      : memory log file (detailed memory usage of forest events)
    */

    // Flags indicating the status of XGBoost model files
    typedef enum XG_base_flags : uint16_t {
        XG_BASE_DATA_EXIST      = 1 << 0,   // _nml.bin exists
        XG_CONFIG_FILE_EXIST    = 1 << 1,   // _xgb_config.json exists
        XG_QTZ_FILE_EXIST       = 1 << 2,   // _qtz.bin exists
        XG_MODEL_FILE_EXIST     = 1 << 3,   // _xgb.bin exists
        XG_HOG_CONFIG_EXIST     = 1 << 4,   // _hogcfg.json exists
        XG_INFER_LOG_EXIST      = 1 << 5,   // _ifl.bin exists
        XG_ABLE_TO_INFERENCE    = 1 << 6,   // Ready for inference
        XG_ABLE_TO_TRAINING     = 1 << 7,   // Ready for training
        XG_BASE_DATA_IS_CSV     = 1 << 8,   // Base data is CSV (needs conversion)
        XG_SCANNED              = 1 << 9    // Resource scan completed
    } XG_base_flags;

    /**
     * @brief Base resource manager for XGBoost models
     * 
     * Manages model file paths and validates resource availability.
     * Similar pattern to Rf_base for Random Forest models.
     */
    class XG_base {
    private:
        mutable uint16_t flags = 0;
        char model_name[XG_PATH_BUFFER] = {0};

    public:
        XG_base() : flags(0) {}

        explicit XG_base(const char* name) : flags(0) {
            init(name);
        }

        // ======================== Initialization ========================

        /**
         * @brief Initialize the base manager with model name and scan resources
         * @param name Model name (folder name in filesystem)
         */
        void init(const char* name) {
            eml_debug(1, "🔧 Initializing XGBoost resource manager");
            
            if (!name || strlen(name) == 0) {
                eml_debug(0, "❌ Model name is empty. The process is aborted.");
                model_name[0] = '\0';
                flags = 0;
                return;
            }

            strncpy(model_name, name, XG_PATH_BUFFER - 1);
            model_name[XG_PATH_BUFFER - 1] = '\0';
            
            scan_current_resource();
        }

        /**
         * @brief Re-scan resources to update status flags
         */
        void update_resource_status() {
            eml_debug(1, "🔄 Updating XGBoost resource status");
            if (model_name[0] == '\0') {
                eml_debug(0, "❌ Model name is empty. Cannot update resource status.");
                return;
            }
            flags = 0;
            scan_current_resource();
        }

    private:
        /**
         * @brief Scan filesystem for model files and update status flags
         */
        void scan_current_resource() {
            char filepath[XG_PATH_BUFFER];

            // Check: base data exists (binary or csv)
            build_file_path(filepath, "_nml.bin");
            if (!RF_FS_EXISTS(filepath)) {
                build_file_path(filepath, "_nml.csv");
                if (RF_FS_EXISTS(filepath)) {
                    eml_debug(1, "🔄 Found CSV dataset, needs conversion to binary format.");
                    flags |= XG_BASE_DATA_IS_CSV;
                } else {
                    eml_debug(1, "⚠️ No base data file found (training data not available)");
                }
            } else {
                eml_debug(1, "✅ Found base data file: ", filepath);
                flags |= XG_BASE_DATA_EXIST;
            }

            // Check: quantizer file exists
            build_file_path(filepath, "_qtz.bin");
            if (RF_FS_EXISTS(filepath)) {
                eml_debug(1, "✅ Found quantizer file: ", filepath);
                flags |= XG_QTZ_FILE_EXIST;
            } else {
                eml_debug(1, "⚠️ No quantizer file found: ", filepath);
            }

            // Check: config file exists
            build_file_path(filepath, "_xgb_config.json");
            if (RF_FS_EXISTS(filepath)) {
                eml_debug(1, "✅ Found config file: ", filepath);
                flags |= XG_CONFIG_FILE_EXIST;
            } else {
                eml_debug(0, "❌ No config file found: ", filepath);
                eml_debug(1, "🔂 Switching to manual configuration");
            }

            // Check: model file exists
            build_file_path(filepath, "_xgb.bin");
            if (RF_FS_EXISTS(filepath)) {
                eml_debug(1, "✅ Found XGBoost model file: ", filepath);
                flags |= XG_MODEL_FILE_EXIST;
            } else {
                eml_debug(1, "⚠️ No XGBoost model file found: ", filepath);
            }

            // Check: HOG config file exists (optional)
            build_file_path(filepath, "_hogcfg.json");
            if (RF_FS_EXISTS(filepath)) {
                eml_debug(1, "✅ Found HOG config file: ", filepath);
                flags |= XG_HOG_CONFIG_EXIST;
            }

            // Check: inference log file exists (optional)
            build_file_path(filepath, "_ifl.bin");
            if (RF_FS_EXISTS(filepath)) {
                eml_debug(2, "✅ Found inference log file: ", filepath);
                flags |= XG_INFER_LOG_EXIST;
            }

            // Determine inference readiness: model file + config + (quantizer for raw input)
            if ((flags & XG_MODEL_FILE_EXIST) && (flags & XG_CONFIG_FILE_EXIST)) {
                flags |= XG_ABLE_TO_INFERENCE;
                eml_debug(1, "✅ Model is ready for inference.");
            } else {
                eml_debug(0, "⚠️ Model is NOT ready for inference.");
                if (!(flags & XG_MODEL_FILE_EXIST)) {
                    eml_debug(0, "   - Missing model file (_xgb.bin)");
                }
                if (!(flags & XG_CONFIG_FILE_EXIST)) {
                    eml_debug(0, "   - Missing config file (_xgb_config.json)");
                }
            }

            // Determine training readiness: base data + quantizer + config
            if ((flags & XG_BASE_DATA_EXIST) && (flags & XG_QTZ_FILE_EXIST) && (flags & XG_CONFIG_FILE_EXIST)) {
                flags |= XG_ABLE_TO_TRAINING;
                eml_debug(1, "✅ Model is ready for training/re-training.");
            } else {
                eml_debug(1, "⚠️ Model is NOT ready for training.");
                if (!(flags & XG_BASE_DATA_EXIST)) {
                    eml_debug(1, "   - Missing base data file (_nml.bin)");
                }
                if (!(flags & XG_QTZ_FILE_EXIST)) {
                    eml_debug(1, "   - Missing quantizer file (_qtz.bin)");
                }
            }

            flags |= XG_SCANNED;
        }

    public:
        // ======================== File Path Builders ========================

        /**
         * @brief Build a file path with the given suffix
         * @param out_path Output buffer for the path
         * @param suffix File suffix (e.g., "_xgb_config.json")
         * @param buffer_size Size of output buffer (default: XG_PATH_BUFFER)
         */
        void build_file_path(char* out_path, const char* suffix, int buffer_size = XG_PATH_BUFFER) const {
            if (!out_path || buffer_size <= 0) return;
            if (buffer_size > XG_PATH_BUFFER) buffer_size = XG_PATH_BUFFER;
            if (!suffix) suffix = "";

            // Primary layout: folder per model, files prefixed by model name
            // Example: /digit_data/digit_data_xgb_config.json
            snprintf(out_path, buffer_size, "/%s/%s%s", model_name, model_name, suffix);
            if (RF_FS_EXISTS(out_path)) {
                return;
            }

            // Fallback layout: flat file at root (no folder)
            // Example: /digit_data_xgb_config.json
            snprintf(out_path, buffer_size, "/%s%s", model_name, suffix);
            if (RF_FS_EXISTS(out_path)) {
                return;
            }

            // Default: use primary layout even if file doesn't exist yet
            snprintf(out_path, buffer_size, "/%s/%s%s", model_name, model_name, suffix);
        }

        // Specific file path getters
        inline void get_qtz_path(char* buffer, int buffer_size = XG_PATH_BUFFER) const { 
            build_file_path(buffer, "_qtz.bin", buffer_size); 
        }
        
        inline void get_base_data_path(char* buffer, int buffer_size = XG_PATH_BUFFER) const { 
            build_file_path(buffer, "_nml.bin", buffer_size); 
        }
        
        inline void get_config_path(char* buffer, int buffer_size = XG_PATH_BUFFER) const { 
            build_file_path(buffer, "_xgb_config.json", buffer_size); 
        }
        
        inline void get_model_path(char* buffer, int buffer_size = XG_PATH_BUFFER) const { 
            build_file_path(buffer, "_xgb.bin", buffer_size); 
        }
        
        inline void get_infer_log_path(char* buffer, int buffer_size = XG_PATH_BUFFER) const { 
            build_file_path(buffer, "_ifl.bin", buffer_size); 
        }
        
        inline void get_memory_log_path(char* buffer, int buffer_size = XG_PATH_BUFFER) const { 
            build_file_path(buffer, "_mlog.csv", buffer_size); 
        }
        
        inline void get_time_log_path(char* buffer, int buffer_size = XG_PATH_BUFFER) const { 
            build_file_path(buffer, "_tlog.csv", buffer_size); 
        }

        inline void get_hog_config_path(char* buffer, int buffer_size = XG_PATH_BUFFER) const {
            build_file_path(buffer, "_hogcfg.json", buffer_size);
        }

        // ======================== Accessors ========================

        const char* getModelName() const { return model_name; }

        void get_model_name(char* buffer, size_t buffer_size) const {
            if (buffer && buffer_size > 0) {
                strncpy(buffer, model_name, buffer_size - 1);
                buffer[buffer_size - 1] = '\0';
            }
        }

        // ======================== Status Checkers ========================

        inline bool ready_to_use() const { 
            return (model_name[0] != '\0') && (flags & XG_SCANNED); 
        }

        inline bool base_data_exists() const { 
            return (flags & XG_BASE_DATA_EXIST) != 0; 
        }

        inline bool base_data_is_csv() const { 
            return (flags & XG_BASE_DATA_IS_CSV) != 0; 
        }

        inline bool config_file_exists() const { 
            return (flags & XG_CONFIG_FILE_EXIST) != 0; 
        }

        inline bool qtz_file_exists() const { 
            return (flags & XG_QTZ_FILE_EXIST) != 0; 
        }

        inline bool model_file_exists() const { 
            return (flags & XG_MODEL_FILE_EXIST) != 0; 
        }

        inline bool hog_config_exists() const { 
            return (flags & XG_HOG_CONFIG_EXIST) != 0; 
        }

        inline bool able_to_inference() const { 
            return (flags & XG_ABLE_TO_INFERENCE) != 0; 
        }

        inline bool able_to_training() const { 
            return (flags & XG_ABLE_TO_TRAINING) != 0; 
        }

        // ======================== Status Setters (for dynamic updates) ========================

        void set_model_file_status(bool exists) {
            if (exists) {
                flags |= XG_MODEL_FILE_EXIST;
            } else {
                flags &= ~XG_MODEL_FILE_EXIST;
            }
            update_inference_training_flags();
        }

        void set_config_status(bool exists) {
            if (exists) {
                flags |= XG_CONFIG_FILE_EXIST;
            } else {
                flags &= ~XG_CONFIG_FILE_EXIST;
            }
            update_inference_training_flags();
        }

        void set_base_data_status(bool exists) {
            if (exists) {
                flags |= XG_BASE_DATA_EXIST;
                flags &= ~XG_BASE_DATA_IS_CSV;
            } else {
                flags &= ~XG_BASE_DATA_EXIST;
            }
            update_inference_training_flags();
        }

        void set_qtz_status(bool exists) {
            if (exists) {
                flags |= XG_QTZ_FILE_EXIST;
            } else {
                flags &= ~XG_QTZ_FILE_EXIST;
            }
            update_inference_training_flags();
        }

    private:
        void update_inference_training_flags() {
            // Update inference readiness
            if ((flags & XG_MODEL_FILE_EXIST) && (flags & XG_CONFIG_FILE_EXIST)) {
                flags |= XG_ABLE_TO_INFERENCE;
            } else {
                flags &= ~XG_ABLE_TO_INFERENCE;
            }

            // Update training readiness
            if ((flags & XG_BASE_DATA_EXIST) && (flags & XG_QTZ_FILE_EXIST) && (flags & XG_CONFIG_FILE_EXIST)) {
                flags |= XG_ABLE_TO_TRAINING;
            } else {
                flags &= ~XG_ABLE_TO_TRAINING;
            }
        }

    public:
        // ======================== Copy/Assignment ========================

        XG_base& operator=(const XG_base& other) {
            if (this != &other) {
                flags = other.flags;
                strncpy(model_name, other.model_name, XG_PATH_BUFFER - 1);
                model_name[XG_PATH_BUFFER - 1] = '\0';
            }
            return *this;
        }

        XG_base(const XG_base& other) {
            flags = other.flags;
            strncpy(model_name, other.model_name, XG_PATH_BUFFER - 1);
            model_name[XG_PATH_BUFFER - 1] = '\0';
        }

        // ======================== Memory Usage ========================

        size_t memory_usage() const {
            return sizeof(XG_base);
        }
    };

} // namespace eml
