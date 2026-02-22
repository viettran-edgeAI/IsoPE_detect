#pragma once

#include <cstddef>
#include "../base/eml_base.h"
#include "eml_samples.h"

namespace mcu {

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

} // namespace mcu
