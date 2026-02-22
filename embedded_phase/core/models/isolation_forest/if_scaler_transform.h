#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>

#include "if_config.h"
#include "../../ml/eml_quantize.h"

namespace eml {

    class If_scaler_transform {
    private:
        vector<float> means_;
        vector<float> scales_;
        uint16_t num_features_ = 0;
        bool is_loaded_ = false;
        float min_scale_epsilon_ = 1e-12f;

    public:
        If_scaler_transform() = default;

        explicit If_scaler_transform(const If_config& cfg) {
            (void)init(cfg);
        }

        bool init(const If_config& cfg) {
            return init(cfg.scaler_mean, cfg.scaler_scale);
        }

        bool init(const vector<float>& means, const vector<float>& scales) {
            is_loaded_ = false;

            if (means.size() == 0 || means.size() != scales.size()) {
                eml_debug(0, "❌ If_scaler_transform init failed: mean/scale size mismatch");
                return false;
            }

            means_ = means;
            scales_ = scales;
            num_features_ = static_cast<uint16_t>(means_.size());
            is_loaded_ = true;
            return true;
        }

        bool init_from_model_engine_config(const std::filesystem::path& model_engine_config_path) {
            If_config temp;
            if (!temp.load_scaler_only(model_engine_config_path)) {
                eml_debug(0, "❌ If_scaler_transform: failed to load scaler from model_engine_config.json");
                return false;
            }
            return init(temp.scaler_mean, temp.scaler_scale);
        }

        void release() {
            means_.clear();
            scales_.clear();
            means_.shrink_to_fit();
            scales_.shrink_to_fit();
            num_features_ = 0;
            is_loaded_ = false;
        }

        bool loaded() const {
            return is_loaded_;
        }

        uint16_t num_features() const {
            return num_features_;
        }

        void set_min_scale_epsilon(float eps) {
            min_scale_epsilon_ = (eps > 0.0f) ? eps : 1e-12f;
        }

        float min_scale_epsilon() const {
            return min_scale_epsilon_;
        }

        bool transform(const float* raw_features, uint16_t feature_count, float* scaled_out) const {
            if (!is_loaded_ || !raw_features || !scaled_out) {
                return false;
            }
            if (feature_count != num_features_) {
                return false;
            }

            for (uint16_t i = 0; i < num_features_; ++i) {
                const float sigma = scales_[i];
                const float denom = (std::fabs(sigma) < min_scale_epsilon_)
                    ? (sigma < 0.0f ? -min_scale_epsilon_ : min_scale_epsilon_)
                    : sigma;
                scaled_out[i] = (raw_features[i] - means_[i]) / denom;
            }
            return true;
        }

        bool transform(const float* raw_features, vector<float>& scaled_out) const {
            if (!is_loaded_ || !raw_features) {
                return false;
            }
            scaled_out.resize(num_features_, 0.0f);
            return transform(raw_features, num_features_, scaled_out.data());
        }

        bool transform_inplace(float* features, uint16_t feature_count) const {
            if (!is_loaded_ || !features) {
                return false;
            }
            if (feature_count != num_features_) {
                return false;
            }

            for (uint16_t i = 0; i < num_features_; ++i) {
                const float sigma = scales_[i];
                const float denom = (std::fabs(sigma) < min_scale_epsilon_)
                    ? (sigma < 0.0f ? -min_scale_epsilon_ : min_scale_epsilon_)
                    : sigma;
                features[i] = (features[i] - means_[i]) / denom;
            }
            return true;
        }

        bool transform_and_quantize(const float* raw_features,
                                    eml_quantizer<problem_type::ISOLATION>& quantizer,
                                    packed_vector<8>& quantized_out,
                                    uint16_t* drift_feature = nullptr,
                                    float* drift_value = nullptr) const {
            if (!is_loaded_ || !raw_features) {
                return false;
            }

            vector<float> scaled;
            scaled.resize(num_features_, 0.0f);
            if (!transform(raw_features, num_features_, scaled.data())) {
                return false;
            }

            return quantizer.quantizeFeatures(scaled.data(), quantized_out, drift_feature, drift_value);
        }

        const vector<float>& means() const { return means_; }
        const vector<float>& scales() const { return scales_; }

        size_t memory_usage() const {
            return sizeof(*this)
                + means_.size() * sizeof(float)
                + scales_.size() * sizeof(float);
        }
    };

} // namespace eml
