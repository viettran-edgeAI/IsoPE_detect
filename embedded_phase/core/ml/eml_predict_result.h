#pragma once

#include <cstddef>
#include "../base/eml_base.h"
#include "../base/eml_status.h"
#include "eml_samples.h"

namespace eml {

    template<problem_type ProblemType = problem_type::CLASSIFICATION, size_t LabelMax = EML_MAX_LABEL_LENGTH>
    struct eml_predict_result_t {
        using label_type = eml_label_t<ProblemType>;

        size_t prediction_time = 0;
        char label[LabelMax] = {'\0'};
        label_type predicted_class = EML_ERROR_LABEL<ProblemType>;
        float confidence = 0.0f;
        float* probabilities = nullptr;
        bool success = false;

        void clear() {
            prediction_time = 0;
            label[0] = '\0';
            predicted_class = EML_ERROR_LABEL<ProblemType>;
            confidence = 0.0f;
            probabilities = nullptr;
            success = false;
        }
    };

    using eml_predict_result = eml_predict_result_t<problem_type::CLASSIFICATION>;

    struct eml_regression_result_t {
        size_t prediction_time = 0;
        float value = 0.0f;
        bool success = false;

        void clear() {
            prediction_time = 0;
            value = 0.0f;
            success = false;
        }
    };

    // Isolation forest anomaly score result
    struct eml_isolation_result_t {
        size_t prediction_time = 0;  // inference time in microseconds
        float anomaly_score = 0.0f;  // raw anomaly score from isolation forest
        bool is_anomaly = false;     // true if score exceeds threshold
        float threshold = 0.0f;      // decision threshold used
        eml_status_code status_code = eml_status_code::ok;
        bool success = false;

        void clear() {
            prediction_time = 0;
            anomaly_score = 0.0f;
            is_anomaly = false;
            threshold = 0.0f;
            status_code = eml_status_code::ok;
            success = false;
        }
    };

} // namespace eml
