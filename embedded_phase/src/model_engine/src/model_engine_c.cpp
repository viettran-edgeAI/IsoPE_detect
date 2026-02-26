#include "model_engine_c.h"

#include "model_engine.hpp"

#include <new>
#include <string>

struct pe_model_engine_handle {
    eml::model_engine::IsolationForestModelEngine engine;
    std::string error_cache;
};

namespace {

void write_result(const eml::eml_isolation_result_t& src, pe_model_engine_result* dst) {
    if (!dst) {
        return;
    }

    dst->prediction_time_us = src.prediction_time;
    dst->anomaly_score = src.anomaly_score;
    dst->threshold = src.threshold;
    dst->status_code = static_cast<uint16_t>(src.status_code);
    dst->is_anomaly = src.is_anomaly ? 1u : 0u;
    dst->success = src.success ? 1u : 0u;
}

}  // namespace

extern "C" {

pe_model_engine_handle* pe_model_engine_create(void) {
    try {
        return new pe_model_engine_handle();
    } catch (...) {
        return nullptr;
    }
}

void pe_model_engine_destroy(pe_model_engine_handle* handle) {
    delete handle;
}

int pe_model_engine_load(pe_model_engine_handle* handle,
                         const char* model_name,
                         const char* resource_dir) {
    if (!handle) {
        return 0;
    }

    const std::string model_name_value = (model_name && model_name[0] != '\0') ? model_name : "iforest";
    const std::filesystem::path resource_dir_value =
        (resource_dir && resource_dir[0] != '\0')
            ? std::filesystem::path(resource_dir)
            : eml::IsoForest::default_resource_dir();

    if (!handle->engine.load_model(model_name_value, resource_dir_value, &handle->error_cache)) {
        return 0;
    }

    handle->error_cache.clear();
    return 1;
}

int pe_model_engine_infer_quantized(pe_model_engine_handle* handle,
                                    const uint8_t* quantized_features,
                                    uint16_t feature_count,
                                    pe_model_engine_result* out_result) {
    if (!handle || !out_result) {
        return 0;
    }

    eml::eml_isolation_result_t result;
    if (!handle->engine.infer_quantized(quantized_features, feature_count, result, &handle->error_cache)) {
        write_result(result, out_result);
        return 0;
    }

    write_result(result, out_result);
    return 1;
}

int pe_model_engine_infer_raw(pe_model_engine_handle* handle,
                              const float* raw_features,
                              uint16_t feature_count,
                              pe_model_engine_result* out_result) {
    if (!handle || !out_result) {
        return 0;
    }

    eml::eml_isolation_result_t result;
    if (!handle->engine.infer_raw(raw_features, feature_count, result, &handle->error_cache)) {
        write_result(result, out_result);
        return 0;
    }

    write_result(result, out_result);
    return 1;
}

const char* pe_model_engine_last_error(pe_model_engine_handle* handle) {
    if (!handle) {
        return "invalid_handle";
    }

    if (!handle->error_cache.empty()) {
        return handle->error_cache.c_str();
    }

    return handle->engine.last_error().c_str();
}

uint16_t pe_model_engine_last_status(pe_model_engine_handle* handle) {
    if (!handle) {
        return static_cast<uint16_t>(eml::eml_status_code::invalid_argument);
    }
    return static_cast<uint16_t>(handle->engine.last_status());
}

uint16_t pe_model_engine_num_features(pe_model_engine_handle* handle) {
    if (!handle) {
        return 0u;
    }
    return handle->engine.metadata().num_features;
}

}  // extern "C"
