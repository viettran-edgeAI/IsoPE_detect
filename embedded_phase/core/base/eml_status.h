#pragma once

#include <cstdint>

namespace eml {

    enum class eml_status_code : uint16_t {
        ok = 0,
        empty_model_name,
        empty_path,
        base_not_ready,
        resource_missing,
        file_open_failed,
        file_read_failed,
        file_write_failed,
        json_parse_failed,
        invalid_configuration,
        invalid_argument,
        feature_count_mismatch,
        size_mismatch,
        callback_not_set,
        callback_failed,
        not_loaded
    };

    inline const char* eml_status_to_string(eml_status_code status) {
        switch (status) {
            case eml_status_code::ok:
                return "ok";
            case eml_status_code::empty_model_name:
                return "empty_model_name";
            case eml_status_code::empty_path:
                return "empty_path";
            case eml_status_code::base_not_ready:
                return "base_not_ready";
            case eml_status_code::resource_missing:
                return "resource_missing";
            case eml_status_code::file_open_failed:
                return "file_open_failed";
            case eml_status_code::file_read_failed:
                return "file_read_failed";
            case eml_status_code::file_write_failed:
                return "file_write_failed";
            case eml_status_code::json_parse_failed:
                return "json_parse_failed";
            case eml_status_code::invalid_configuration:
                return "invalid_configuration";
            case eml_status_code::invalid_argument:
                return "invalid_argument";
            case eml_status_code::feature_count_mismatch:
                return "feature_count_mismatch";
            case eml_status_code::size_mismatch:
                return "size_mismatch";
            case eml_status_code::callback_not_set:
                return "callback_not_set";
            case eml_status_code::callback_failed:
                return "callback_failed";
            case eml_status_code::not_loaded:
                return "not_loaded";
            default:
                return "unknown";
        }
    }

} // namespace eml
