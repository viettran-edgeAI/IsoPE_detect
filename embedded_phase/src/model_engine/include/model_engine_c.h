#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pe_model_engine_handle pe_model_engine_handle;

typedef struct pe_model_engine_result {
    size_t prediction_time_us;
    float anomaly_score;
    float threshold;
    uint16_t status_code;
    uint8_t is_anomaly;
    uint8_t success;
} pe_model_engine_result;

pe_model_engine_handle* pe_model_engine_create(void);
void pe_model_engine_destroy(pe_model_engine_handle* handle);

int pe_model_engine_load(pe_model_engine_handle* handle,
                         const char* model_name,
                         const char* resource_dir);

int pe_model_engine_infer_quantized(pe_model_engine_handle* handle,
                                    const uint8_t* quantized_features,
                                    uint16_t feature_count,
                                    pe_model_engine_result* out_result);

int pe_model_engine_infer_raw(pe_model_engine_handle* handle,
                              const float* raw_features,
                              uint16_t feature_count,
                              pe_model_engine_result* out_result);

const char* pe_model_engine_last_error(pe_model_engine_handle* handle);
uint16_t pe_model_engine_last_status(pe_model_engine_handle* handle);
uint16_t pe_model_engine_num_features(pe_model_engine_handle* handle);

#ifdef __cplusplus
}
#endif
