#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "../../ml/eml_quantize.h"
#include "../../ml/eml_predict_result.h"
#include "../../ml/eml_metrics.h"

#include "if_components.h"

namespace eml {

    class IsoForest {
    public:
        using If_quantizer = eml_quantizer<problem_type::ISOLATION>;
        using extract_callback_t = If_feature_extractor::extract_callback_t;
        using extract_content_callback_t = If_feature_extractor::extract_content_callback_t;

    private:
        If_base if_base_{};
        If_config if_config_{};
        If_feature_extractor if_feature_extractor_{};
        If_feature_transform_layer if_feature_transform_layer_{};
        If_scaler_layer if_scaler_layer_{};
        If_quantizer if_quantizer_{};
        If_tree_container if_tree_container_{};

        uint16_t num_features_ = 0u;
        bool initialized_ = false;
        bool preprocessing_initialized_ = false;
        bool loaded_ = false;
        mutable eml_status_code last_status_code_ = eml_status_code::ok;

        inline void set_status(eml_status_code status) const {
            last_status_code_ = status;
        }

        void release_preprocessing_components() {
            if_feature_extractor_.release();
            if_feature_transform_layer_.release();
            if_scaler_layer_.release();
            if_quantizer_.releaseQuantizer();
            preprocessing_initialized_ = false;
        }

        static bool read_exact(std::ifstream& fin, void* dst, size_t bytes) {
            fin.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
            return static_cast<size_t>(fin.gcount()) == bytes;
        }

        static uint32_t resolve_samples_per_tree(const If_config& cfg, size_t num_samples) {
            if (num_samples == 0u) {
                return 1u;
            }

            uint32_t resolved = cfg.max_samples_per_tree;
            if (resolved == 0u) {
                if (cfg.max_samples <= 1.0f) {
                    const float ratio = cfg.max_samples <= 0.0f ? 1.0f : cfg.max_samples;
                    resolved = static_cast<uint32_t>(std::ceil(ratio * static_cast<float>(num_samples)));
                } else {
                    resolved = static_cast<uint32_t>(std::ceil(cfg.max_samples));
                }
            }

            if (resolved == 0u) {
                resolved = 1u;
            }
            if (resolved > num_samples) {
                resolved = static_cast<uint32_t>(num_samples);
            }
            return resolved;
        }

        static std::vector<uint32_t> sample_indices(size_t num_samples,
                                                    uint32_t sample_size,
                                                    bool bootstrap,
                                                    std::mt19937& rng) {
            std::vector<uint32_t> out;
            out.reserve(sample_size);

            if (num_samples == 0u || sample_size == 0u) {
                return out;
            }

            if (bootstrap) {
                std::uniform_int_distribution<uint32_t> dist(0u, static_cast<uint32_t>(num_samples - 1u));
                for (uint32_t i = 0; i < sample_size; ++i) {
                    out.push_back(dist(rng));
                }
                return out;
            }

            out.resize(num_samples);
            std::iota(out.begin(), out.end(), 0u);
            std::shuffle(out.begin(), out.end(), rng);
            out.resize(sample_size);
            return out;
        }

        struct If_build_task {
            uint32_t node_index = 0u;
            size_t begin = 0u;
            size_t end = 0u;
            uint16_t depth = 0u;
        };

        static bool build_tree_from_quantized_matrix(If_tree& tree,
                                                     const If_node_resource* resource,
                                                     const uint8_t* matrix,
                                                     uint16_t num_features,
                                                     const std::vector<uint32_t>& sampled_indices,
                                                     uint16_t max_depth,
                                                     uint32_t max_nodes_per_tree,
                                                     std::mt19937& rng) {
            if (!resource || !resource->valid() || !matrix || num_features == 0u || sampled_indices.empty()) {
                return false;
            }

            std::vector<uint32_t> indices = sampled_indices;

            tree.set_node_resources(resource, true);
            tree.reserve_nodes(std::max<uint32_t>(8u, std::min<uint32_t>(max_nodes_per_tree, 2048u)));
            (void)tree.append_node(IsoNode{}); // root placeholder
            if (tree.node_count() != 1u) {
                return false;
            }

            std::vector<If_build_task> queue;
            queue.reserve(256u);
            queue.push_back(If_build_task{0u, 0u, indices.size(), 0u});

            uint16_t max_tree_depth = 0u;
            size_t queue_head = 0u;

            while (queue_head < queue.size()) {
                const If_build_task task = queue[queue_head++];
                const size_t sample_count = task.end - task.begin;

                if (sample_count <= 1u || task.depth >= max_depth) {
                    if (!tree.set_leaf_node(task.node_index, static_cast<uint32_t>(sample_count), task.depth)) {
                        return false;
                    }
                    max_tree_depth = static_cast<uint16_t>(std::max<uint16_t>(max_tree_depth, task.depth));
                    continue;
                }

                bool found_split = false;
                uint16_t split_feature = 0u;
                uint8_t split_threshold = 0u;
                const uint32_t attempts = std::max<uint32_t>(8u, static_cast<uint32_t>(num_features) * 2u);
                std::uniform_int_distribution<uint16_t> feature_dist(0u, static_cast<uint16_t>(num_features - 1u));

                for (uint32_t attempt = 0u; attempt < attempts; ++attempt) {
                    const uint16_t feature = feature_dist(rng);
                    std::array<uint8_t, 256> value_seen{};
                    uint16_t unique_count = 0u;

                    for (size_t index = task.begin; index < task.end; ++index) {
                        const uint8_t value = matrix[static_cast<size_t>(indices[index]) * num_features + feature];
                        if (value_seen[value] == 0u) {
                            value_seen[value] = 1u;
                            ++unique_count;
                        }
                    }

                    if (unique_count < 2u) {
                        continue;
                    }

                    std::vector<uint8_t> unique_values;
                    unique_values.reserve(unique_count);
                    for (uint16_t value = 0u; value <= 255u; ++value) {
                        if (value_seen[value] != 0u) {
                            unique_values.push_back(static_cast<uint8_t>(value));
                        }
                    }

                    if (unique_values.size() < 2u) {
                        continue;
                    }

                    std::uniform_int_distribution<size_t> threshold_idx_dist(0u, unique_values.size() - 2u);
                    found_split = true;
                    split_feature = feature;
                    split_threshold = unique_values[threshold_idx_dist(rng)];
                    break;
                }

                if (!found_split) {
                    if (!tree.set_leaf_node(task.node_index, static_cast<uint32_t>(sample_count), task.depth)) {
                        return false;
                    }
                    max_tree_depth = static_cast<uint16_t>(std::max<uint16_t>(max_tree_depth, task.depth));
                    continue;
                }

                size_t left = task.begin;
                size_t right = task.end;
                while (left < right) {
                    const uint8_t value = matrix[static_cast<size_t>(indices[left]) * num_features + split_feature];
                    if (value <= split_threshold) {
                        ++left;
                    } else {
                        --right;
                        std::swap(indices[left], indices[right]);
                    }
                }

                const size_t mid = left;
                if (mid == task.begin || mid == task.end || tree.node_count() + 2u > max_nodes_per_tree) {
                    if (!tree.set_leaf_node(task.node_index, static_cast<uint32_t>(sample_count), task.depth)) {
                        return false;
                    }
                    max_tree_depth = static_cast<uint16_t>(std::max<uint16_t>(max_tree_depth, task.depth));
                    continue;
                }

                const uint32_t left_child = tree.append_node(IsoNode{});
                const uint32_t right_child = tree.append_node(IsoNode{});
                if (right_child != left_child + 1u) {
                    return false;
                }

                if (!tree.set_split_node(task.node_index, split_feature, split_threshold, left_child)) {
                    return false;
                }

                const uint16_t child_depth = static_cast<uint16_t>(task.depth + 1u);
                queue.push_back(If_build_task{left_child, task.begin, mid, child_depth});
                queue.push_back(If_build_task{right_child, mid, task.end, child_depth});
            }

            return tree.finalize(max_tree_depth);
        }

        bool initialize_components() {
            set_status(eml_status_code::ok);
            initialized_ = false;
            loaded_ = false;
            num_features_ = 0u;
            release_preprocessing_components();

            if_base_.update_resource_status();
            if (!if_base_.ready_to_use()) {
                const eml_status_code base_status = if_base_.last_status();
                set_status(base_status == eml_status_code::ok ? eml_status_code::base_not_ready : base_status);
                return false;
            }

            if (!if_base_.has_required_core_resources()) {
                const eml_status_code base_status = if_base_.last_status();
                set_status(base_status == eml_status_code::ok ? eml_status_code::resource_missing : base_status);
                return false;
            }

            if_config_.init(&if_base_);
            if_config_.set_base(&if_base_);
            if (!if_config_.load_from_base()) {
                set_status(if_config_.last_status());
                return false;
            }

            num_features_ = if_config_.num_features;

            if (!if_feature_extractor_.init(if_base_.get_feature_config_path(), num_features_)) {
                set_status(if_feature_extractor_.last_status());
                return false;
            }

            if (!if_feature_transform_layer_.init_from_feature_schema(
                    if_base_.get_feature_schema_path(),
                    if_feature_extractor_.feature_names(),
                    num_features_)) {
                set_status(if_feature_transform_layer_.last_status());
                return false;
            }

            if (!if_scaler_layer_.init_from_file(if_base_.get_scaler_params_path(), num_features_)) {
                set_status(if_scaler_layer_.last_status());
                return false;
            }

            if (!if_quantizer_.loadQuantizer(if_base_.get_qtz_path().string().c_str())) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }

            preprocessing_initialized_ = true;

            if_tree_container_.unload_model();
            initialized_ = true;
            set_status(eml_status_code::ok);
            return true;
        }

        bool ensure_initialized() {
            if (initialized_) {
                set_status(eml_status_code::ok);
                return true;
            }
            return initialize_components();
        }

        bool apply_feature_transforms(const float* raw_features,
                                      uint16_t feature_count,
                                      vector<float>& out_scaled_features) const {
            out_scaled_features.clear();

            if (!raw_features || feature_count == 0u) {
                set_status(eml_status_code::invalid_argument);
                return false;
            }

            if (!preprocessing_initialized_) {
                set_status(eml_status_code::not_loaded);
                return false;
            }

            if (!if_feature_transform_layer_.loaded()) {
                const eml_status_code layer_status = if_feature_transform_layer_.last_status();
                set_status(layer_status == eml_status_code::ok ? eml_status_code::not_loaded : layer_status);
                return false;
            }

            if (!if_scaler_layer_.loaded()) {
                const eml_status_code scaler_status = if_scaler_layer_.last_status();
                set_status(scaler_status == eml_status_code::ok ? eml_status_code::not_loaded : scaler_status);
                return false;
            }

            if (feature_count != num_features_) {
                set_status(eml_status_code::feature_count_mismatch);
                return false;
            }

            vector<float> transformed(feature_count, 0.0f);
            if (!if_feature_transform_layer_.transform(raw_features, feature_count, transformed.data())) {
                set_status(if_feature_transform_layer_.last_status());
                return false;
            }

            out_scaled_features.resize(feature_count, 0.0f);
            if (!if_scaler_layer_.transform(transformed.data(), feature_count, out_scaled_features.data())) {
                out_scaled_features.clear();
                set_status(if_scaler_layer_.last_status());
                return false;
            }

            set_status(eml_status_code::ok);
            return true;
        }

        bool quantize_raw_feature_vector(const float* raw_features,
                                         uint16_t feature_count,
                                         vector<uint8_t>& out_quantized) const {
            out_quantized.clear();

            if (!raw_features || feature_count == 0u) {
                set_status(eml_status_code::invalid_argument);
                return false;
            }

            if (!if_quantizer_.loaded()) {
                set_status(eml_status_code::not_loaded);
                return false;
            }

            vector<float> scaled_features;
            if (!apply_feature_transforms(raw_features, feature_count, scaled_features)) {
                return false;
            }

            packed_vector<8> quantized_buffer;
            quantized_buffer.resize(feature_count, 0u);
            (void)if_quantizer_.quantizeFeatures(scaled_features.data(), quantized_buffer, nullptr, nullptr);

            out_quantized.resize(feature_count, 0u);
            for (uint16_t feature_index = 0; feature_index < feature_count; ++feature_index) {
                out_quantized[feature_index] = quantized_buffer[feature_index];
            }

            set_status(eml_status_code::ok);
            return true;
        }

        static bool load_quantized_nml_dataset(const std::filesystem::path& nml_path,
                                               uint16_t expected_num_features,
                                               uint8_t quantization_bits,
                                               std::vector<uint8_t>& out_matrix,
                                               size_t& out_num_samples) {
            out_matrix.clear();
            out_num_samples = 0u;

            std::ifstream fin(nml_path, std::ios::binary);
            if (!fin.is_open()) {
                return false;
            }

            uint32_t num_samples_u32 = 0u;
            uint16_t num_features_u16 = 0u;
            if (!read_exact(fin, &num_samples_u32, sizeof(num_samples_u32)) ||
                !read_exact(fin, &num_features_u16, sizeof(num_features_u16))) {
                return false;
            }

            if (num_features_u16 != expected_num_features) {
                return false;
            }

            if (quantization_bits < 1u || quantization_bits > 8u) {
                return false;
            }

            const uint16_t packed_feature_bytes = static_cast<uint16_t>(
                (static_cast<uint32_t>(num_features_u16) * quantization_bits + 7u) / 8u
            );
            const uint8_t feature_mask = quantization_bits == 8u
                ? 0xFFu
                : static_cast<uint8_t>((1u << quantization_bits) - 1u);

            out_num_samples = static_cast<size_t>(num_samples_u32);
            out_matrix.resize(out_num_samples * static_cast<size_t>(num_features_u16), 0u);

            std::vector<uint8_t> packed(packed_feature_bytes, 0u);
            for (size_t row = 0; row < out_num_samples; ++row) {
                uint8_t label = 0u;
                if (!read_exact(fin, &label, sizeof(label))) {
                    return false;
                }
                (void)label;

                if (!read_exact(fin, packed.data(), packed.size())) {
                    return false;
                }

                uint8_t* out_row = &out_matrix[row * static_cast<size_t>(num_features_u16)];
                for (uint16_t col = 0; col < num_features_u16; ++col) {
                    const uint32_t bit_pos = static_cast<uint32_t>(col) * quantization_bits;
                    const uint32_t byte_index = bit_pos / 8u;
                    const uint8_t bit_offset = static_cast<uint8_t>(bit_pos % 8u);

                    uint8_t value = 0u;
                    if (bit_offset + quantization_bits <= 8u) {
                        value = static_cast<uint8_t>((packed[byte_index] >> bit_offset) & feature_mask);
                    } else {
                        const uint8_t bits_first = static_cast<uint8_t>(8u - bit_offset);
                        const uint8_t bits_second = static_cast<uint8_t>(quantization_bits - bits_first);
                        const uint8_t low = static_cast<uint8_t>((packed[byte_index] >> bit_offset) & ((1u << bits_first) - 1u));
                        const uint8_t high = static_cast<uint8_t>(packed[byte_index + 1u] & ((1u << bits_second) - 1u));
                        value = static_cast<uint8_t>(low | (high << bits_first));
                    }
                    out_row[col] = value;
                }
            }

            return true;
        }

        static std::vector<float> score_quantized_matrix(const If_tree_container& tree_container,
                                                         const std::vector<uint8_t>& matrix,
                                                         size_t num_samples,
                                                         uint16_t num_features) {
            std::vector<float> scores;
            if (num_samples == 0u || num_features == 0u) {
                return scores;
            }

            const size_t expected = num_samples * static_cast<size_t>(num_features);
            if (matrix.size() != expected) {
                return scores;
            }

            scores.reserve(num_samples);
            for (size_t row = 0; row < num_samples; ++row) {
                const uint8_t* sample = &matrix[row * static_cast<size_t>(num_features)];
                scores.push_back(tree_container.decision_function(sample, num_features));
            }
            return scores;
        }

        static float resolve_validation_fpr_target(const If_config& cfg) {
            float target = cfg.val_fpr_target;
            if (!(target > 0.0f && target < 1.0f)) {
                target = cfg.fpr_threshold;
            }
            if (!(target > 0.0f && target < 1.0f)) {
                target = cfg.contamination;
            }
            if (!(target > 0.0f && target < 1.0f)) {
                target = 0.05f;
            }
            return target;
        }

        static float select_threshold_from_validation(const std::vector<float>& benign_scores,
                                                      const std::vector<float>& malware_scores,
                                                      float target_fpr,
                                                      float& out_fpr,
                                                      float& out_tpr) {
            out_fpr = 0.0f;
            out_tpr = 0.0f;

            if (benign_scores.empty() || malware_scores.empty()) {
                return 0.0f;
            }

            std::vector<float> candidates;
            candidates.reserve(benign_scores.size() + malware_scores.size());
            candidates.insert(candidates.end(), benign_scores.begin(), benign_scores.end());
            candidates.insert(candidates.end(), malware_scores.begin(), malware_scores.end());
            std::sort(candidates.begin(), candidates.end());
            candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

            const float max_fpr = std::max(0.0f, std::min(1.0f, target_fpr));
            const size_t benign_count = benign_scores.size();
            const size_t malware_count = malware_scores.size();

            bool found = false;
            float best_threshold = candidates.front();
            float best_fpr = 1.0f;
            float best_tpr = 0.0f;

            for (float threshold : candidates) {
                size_t fp = 0u;
                for (float score : benign_scores) {
                    if (score < threshold) {
                        ++fp;
                    }
                }

                size_t tp = 0u;
                for (float score : malware_scores) {
                    if (score < threshold) {
                        ++tp;
                    }
                }

                const float fpr = static_cast<float>(fp) / static_cast<float>(benign_count);
                const float tpr = static_cast<float>(tp) / static_cast<float>(malware_count);
                if (fpr > max_fpr) {
                    continue;
                }

                if (!found ||
                    tpr > best_tpr ||
                    (tpr == best_tpr && fpr < best_fpr)) {
                    found = true;
                    best_threshold = threshold;
                    best_fpr = fpr;
                    best_tpr = tpr;
                }
            }

            if (!found) {
                std::vector<float> sorted_benign = benign_scores;
                std::sort(sorted_benign.begin(), sorted_benign.end());

                const size_t max_fp = static_cast<size_t>(
                    std::floor(max_fpr * static_cast<float>(sorted_benign.size()))
                );

                if (max_fp == 0u) {
                    best_threshold = sorted_benign.front() - 1e-10f;
                } else {
                    best_threshold = sorted_benign[max_fp - 1u];
                }

                size_t fp = 0u;
                size_t tp = 0u;
                for (float score : benign_scores) {
                    if (score < best_threshold) {
                        ++fp;
                    }
                }
                for (float score : malware_scores) {
                    if (score < best_threshold) {
                        ++tp;
                    }
                }

                best_fpr = static_cast<float>(fp) / static_cast<float>(benign_count);
                best_tpr = static_cast<float>(tp) / static_cast<float>(malware_count);
            }

            out_fpr = best_fpr;
            out_tpr = best_tpr;
            return best_threshold;
        }

        // 
        bool calibrate_threshold_from_validation_datasets(const std::filesystem::path& benign_val_nml_path = {},
                                                          const std::filesystem::path& malware_val_nml_path = {}) {
            if (!loaded_) {
                set_status(eml_status_code::not_loaded);
                return false;
            }

            if (!if_tree_container_.trained()) {
                const eml_status_code tree_status = if_tree_container_.last_status();
                set_status(tree_status == eml_status_code::ok ? eml_status_code::not_loaded : tree_status);
                return false;
            }

            const std::filesystem::path benign_path = benign_val_nml_path.empty()
                ? if_base_.get_benign_val_nml_path()
                : benign_val_nml_path;
            const std::filesystem::path malware_path = malware_val_nml_path.empty()
                ? if_base_.get_malware_val_nml_path()
                : malware_val_nml_path;

            if (benign_path.empty() || malware_path.empty()) {
                set_status(eml_status_code::empty_path);
                return false;
            }

            if (!std::filesystem::exists(benign_path) || !std::filesystem::exists(malware_path)) {
                set_status(eml_status_code::resource_missing);
                return false;
            }

            std::vector<uint8_t> benign_matrix;
            std::vector<uint8_t> malware_matrix;
            size_t benign_samples = 0u;
            size_t malware_samples = 0u;

            if (!load_quantized_nml_dataset(benign_path,
                                            if_config_.num_features,
                                            if_config_.quantization_bits,
                                            benign_matrix,
                                            benign_samples)) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }

            if (!load_quantized_nml_dataset(malware_path,
                                            if_config_.num_features,
                                            if_config_.quantization_bits,
                                            malware_matrix,
                                            malware_samples)) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }

            const std::vector<float> benign_scores = score_quantized_matrix(
                if_tree_container_, benign_matrix, benign_samples, if_config_.num_features);
            const std::vector<float> malware_scores = score_quantized_matrix(
                if_tree_container_, malware_matrix, malware_samples, if_config_.num_features);

            if (benign_scores.empty() || malware_scores.empty()) {
                set_status(eml_status_code::size_mismatch);
                return false;
            }

            const float target_fpr = resolve_validation_fpr_target(if_config_);
            float calibrated_fpr = 0.0f;
            float calibrated_tpr = 0.0f;
            const float calibrated_threshold = select_threshold_from_validation(
                benign_scores,
                malware_scores,
                target_fpr,
                calibrated_fpr,
                calibrated_tpr
            );

            if_config_.decision_threshold = calibrated_threshold;
            if_config_.fpr_threshold = calibrated_fpr;
            if_config_.threshold_offset = 0.0f;
            if_tree_container_.set_threshold_offset(0.0f);
            set_status(eml_status_code::ok);
            return true;
        }

    public:
        static std::filesystem::path default_resource_dir() {
            return std::filesystem::path("embedded_phase/core/models/isolation_forest/resources");
        }

        IsoForest() = default;

        explicit IsoForest(const std::string& model_name,
                           const std::filesystem::path& dir_path = std::filesystem::path(".")) {
            (void)init(model_name, dir_path);
        }

        bool init(const std::string& model_name = "iforest",
                  const std::filesystem::path& dir_path = std::filesystem::path(".")) {
            if_base_.init(model_name.c_str(), dir_path);
            if_config_.init(&if_base_);
            return initialize_components();
        }

        bool load() {
            if (!ensure_initialized()) {
                return false;
            }

            if (!if_base_.model_exists()) {
                const eml_status_code base_status = if_base_.last_status();
                set_status(base_status == eml_status_code::ok ? eml_status_code::resource_missing : base_status);
                return false;
            }

            if_tree_container_.unload_model();
            if (!if_tree_container_.load_model_binary(if_base_.get_model_path())) {
                set_status(if_tree_container_.last_status());
                return false;
            }

            if_config_.threshold_offset = 0.0f;
            if_tree_container_.set_threshold_offset(0.0f);

            loaded_ = true;
            set_status(eml_status_code::ok);
            return true;
        }

        bool init_from_config(const If_config& config) {
            if (!config.isLoaded || config.num_features == 0) {
                set_status(eml_status_code::invalid_configuration);
                return false;
            }

            release_preprocessing_components();
            if_config_ = config;
            num_features_ = if_config_.num_features;
            initialized_ = true;
            loaded_ = false;
            if_tree_container_.unload_model();
            set_status(eml_status_code::ok);
            return true;
        }

        bool build_model(bool enable_calibration = true,
                         const std::filesystem::path& benign_train_nml_path = {}) {
            if (!ensure_initialized()) {
                return false;
            }

            const std::filesystem::path dataset_path = benign_train_nml_path.empty()
                ? if_base_.get_nml_path()
                : benign_train_nml_path;

            if (dataset_path.empty()) {
                set_status(eml_status_code::empty_path);
                return false;
            }

            std::vector<uint8_t> matrix;
            size_t num_samples = 0u;
            if (!load_quantized_nml_dataset(
                    dataset_path,
                    if_config_.num_features,
                    if_config_.quantization_bits,
                    matrix,
                    num_samples)) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }

            const uint16_t feature_count = if_config_.num_features;

            if_tree_container_.unload_model();
            if (!if_tree_container_.set_node_resource_layout(if_config_.threshold_bits,
                                                             if_config_.feature_bits,
                                                             if_config_.child_bits,
                                                             if_config_.leaf_size_bits,
                                                             if_config_.depth_bits)) {
                set_status(if_tree_container_.last_status());
                return false;
            }

            const uint32_t samples_per_tree = resolve_samples_per_tree(if_config_, num_samples);
            if_config_.threshold_offset = 0.0f;
            if_tree_container_.set_threshold_offset(0.0f);

            const uint16_t n_estimators = std::max<uint16_t>(1u, if_config_.n_estimators);
            const uint16_t max_depth = std::max<uint16_t>(1u, if_config_.max_depth);
            const uint32_t max_nodes_per_tree = std::max<uint32_t>(1u, if_config_.max_nodes_per_tree);

            std::vector<If_tree> trained_trees;
            trained_trees.reserve(n_estimators);

            std::mt19937 rng(if_config_.random_state);
            for (uint16_t tree_index = 0; tree_index < n_estimators; ++tree_index) {
                If_tree tree;

                const std::vector<uint32_t> sampled = sample_indices(
                    num_samples,
                    samples_per_tree,
                    if_config_.bootstrap,
                    rng
                );

                if (!build_tree_from_quantized_matrix(tree,
                                                      if_tree_container_.node_resource_ptr(),
                                                      matrix.data(),
                                                      feature_count,
                                                      sampled,
                                                      max_depth,
                                                      max_nodes_per_tree,
                                                      rng)) {
                    if_tree_container_.unload_model();
                    loaded_ = false;
                    set_status(eml_status_code::invalid_configuration);
                    return false;
                }

                trained_trees.push_back(std::move(tree));
            }

            if_tree_container_.load_trained_forest(std::move(trained_trees), samples_per_tree, 0.0f);
            loaded_ = if_tree_container_.trained();

            matrix.clear();
            matrix.shrink_to_fit();

            if (!loaded_) {
                set_status(eml_status_code::not_loaded);
                return false;
            }

            if (enable_calibration && !calibrate_threshold_from_validation_datasets()) {
                return false;
            }

            if (!if_config_.persist_threshold_to_config()) {
                set_status(if_config_.last_status());
                return false;
            }

            if (!save_model()) {
                return false;
            }

            if_base_.update_resource_status();

            set_status(eml_status_code::ok);
            return true;
        }

        bool quantize_raw_feature_buffer(const float* raw_features,
                                         uint16_t feature_count,
                                         vector<uint8_t>& out_quantized) const {
            return quantize_raw_feature_vector(raw_features, feature_count, out_quantized);
        }

        void set_extract_callback(extract_callback_t callback) {
            if_feature_extractor_.set_extract_callback(std::move(callback));
        }

        void set_extract_content_callback(extract_content_callback_t callback) {
            if_feature_extractor_.set_extract_content_callback(std::move(callback));
        }

        bool save_model() const {
            if (!if_tree_container_.save_model_binary(if_base_.get_iforest_bin_path())) {
                set_status(if_tree_container_.last_status());
                return false;
            }
            set_status(eml_status_code::ok);
            return true;
        }

        float decision_function(const uint8_t* quantized_features,
                                uint16_t feature_count) const {
            return if_tree_container_.decision_function(quantized_features, feature_count);
        }

        bool is_anomaly(const uint8_t* quantized_features,
                        uint16_t feature_count,
                        float threshold) const {
            return if_tree_container_.is_anomaly(quantized_features, feature_count, threshold);
        }

        eml_isolation_result_t infer_quantized(const uint8_t* quantized_features,
                               uint16_t feature_count) const {
            eml_isolation_result_t result;
            const auto start = std::chrono::steady_clock::now();

            if (!loaded_) {
                result.status_code = eml_status_code::not_loaded;
                set_status(result.status_code);
                result.success = false;
                return result;
            }

            if (!quantized_features || feature_count == 0u) {
                result.status_code = eml_status_code::invalid_argument;
                set_status(result.status_code);
                result.success = false;
                return result;
            }

            const float active_threshold = if_config_.decision_threshold;

            result.anomaly_score = decision_function(quantized_features, feature_count);
            result.threshold = active_threshold;
            result.is_anomaly = result.anomaly_score < active_threshold;
            result.status_code = eml_status_code::ok;
            result.success = true;
            set_status(eml_status_code::ok);

            const auto end = std::chrono::steady_clock::now();
            result.prediction_time = static_cast<size_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
            return result;
        }

        eml_isolation_result_t infer_raw(const float* raw_features,
                         uint16_t feature_count) const {
            eml_isolation_result_t result;
            const auto start = std::chrono::steady_clock::now();

            if (!loaded_) {
                result.status_code = eml_status_code::not_loaded;
                set_status(result.status_code);
                result.success = false;
                return result;
            }

            if (!raw_features || feature_count == 0u) {
                result.status_code = eml_status_code::invalid_argument;
                set_status(result.status_code);
                result.success = false;
                return result;
            }

            vector<uint8_t> quantized_features;
            if (!quantize_raw_feature_vector(raw_features, feature_count, quantized_features)) {
                result.status_code = last_status_code_;
                if (result.status_code == eml_status_code::ok) {
                    result.status_code = eml_status_code::invalid_configuration;
                }
                set_status(result.status_code);
                result.success = false;
                return result;
            }

            const float active_threshold = if_config_.decision_threshold;
            result.anomaly_score = decision_function(quantized_features.data(), feature_count);
            result.threshold = active_threshold;
            result.is_anomaly = result.anomaly_score < active_threshold;
            result.status_code = eml_status_code::ok;
            result.success = true;
            set_status(eml_status_code::ok);

            const auto end = std::chrono::steady_clock::now();
            result.prediction_time = static_cast<size_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
            return result;
        }

        eml_isolation_result_t infer_pe_path(const std::filesystem::path& pe_path) const {
            eml_isolation_result_t result;
            if (!loaded_) {
                result.status_code = eml_status_code::not_loaded;
                set_status(result.status_code);
                result.success = false;
                return result;
            }

            vector<float> raw_features;
            if (!if_feature_extractor_.extract_from_pe(pe_path, raw_features)) {
                result.status_code = if_feature_extractor_.last_status();
                if (result.status_code == eml_status_code::ok) {
                    result.status_code = eml_status_code::callback_failed;
                }
                set_status(result.status_code);
                result.success = false;
                return result;
            }

            return infer_raw(raw_features.data(), static_cast<uint16_t>(raw_features.size()));
        }

        eml_isolation_result_t infer_pe_content(const uint8_t* pe_content,
                                                size_t pe_size) const {
            eml_isolation_result_t result;
            if (!loaded_) {
                result.status_code = eml_status_code::not_loaded;
                set_status(result.status_code);
                result.success = false;
                return result;
            }

            if (!pe_content || pe_size == 0u) {
                result.status_code = eml_status_code::invalid_argument;
                set_status(result.status_code);
                result.success = false;
                return result;
            }

            vector<float> raw_features;
            if (!if_feature_extractor_.extract_from_pe_content(pe_content, pe_size, raw_features)) {
                result.status_code = if_feature_extractor_.last_status();
                if (result.status_code == eml_status_code::ok) {
                    result.status_code = eml_status_code::callback_failed;
                }
                set_status(result.status_code);
                result.success = false;
                return result;
            }

            return infer_raw(raw_features.data(), static_cast<uint16_t>(raw_features.size()));
        }

        bool initialized() const { return initialized_; }
        bool loaded() const { return loaded_; }
        uint16_t num_features() const { return num_features_; }
        eml_status_code last_status() const { return last_status_code_; }
        void clear_status() { set_status(eml_status_code::ok); }

        const If_base& base() const { return if_base_; }
        const If_config& config() const { return if_config_; }
        const If_feature_extractor& feature_extractor() const { return if_feature_extractor_; }
        const If_feature_transform_layer& feature_transform_layer() const { return if_feature_transform_layer_; }
        const If_scaler_layer& scaler_layer() const { return if_scaler_layer_; }
        const If_quantizer& quantizer() const { return if_quantizer_; }
        const If_tree_container& tree_container() const { return if_tree_container_; }
    };

    struct If_threshold_result {
        float threshold = 0.0f;
        float fpr = 0.0f;
        float tpr = 0.0f;
        float metric = 0.0f;
        bool has_metric = false;
    };

    inline If_threshold_result if_find_threshold_precise(const std::vector<float>& benign_scores, float max_fpr) {
        If_threshold_result out;
        if (benign_scores.empty()) {
            return out;
        }

        std::vector<float> sorted = benign_scores;
        std::sort(sorted.begin(), sorted.end());
        const size_t n = sorted.size();
        size_t max_fp = 0u;
        if (max_fpr > 0.0f) {
            max_fp = static_cast<size_t>(std::floor(max_fpr * static_cast<float>(n)));
        }

        if (max_fp == 0u) {
            out.threshold = sorted.front() - 1e-10f;
            out.fpr = 0.0f;
            return out;
        }

        out.threshold = sorted[max_fp - 1u];
        size_t fp = 0u;
        for (float s : benign_scores) {
            if (s < out.threshold) {
                ++fp;
            }
        }
        out.fpr = static_cast<float>(fp) / static_cast<float>(n);
        return out;
    }

    inline If_threshold_result if_select_threshold_with_malware(const std::vector<float>& benign_scores,
                                                                const std::vector<float>& malware_scores,
                                                                float max_fpr,
                                                                const std::string& strategy,
                                                                float beta = 1.0f) {
        If_threshold_result out;
        if (benign_scores.empty() || malware_scores.empty()) {
            return out;
        }

        std::vector<float> thresholds;
        thresholds.reserve(benign_scores.size() + malware_scores.size());
        thresholds.insert(thresholds.end(), benign_scores.begin(), benign_scores.end());
        thresholds.insert(thresholds.end(), malware_scores.begin(), malware_scores.end());
        std::sort(thresholds.begin(), thresholds.end());
        thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());

        const size_t n_b = benign_scores.size();
        const size_t n_m = malware_scores.size();

        float best_metric = -std::numeric_limits<float>::infinity();
        bool found = false;

        for (float thr : thresholds) {
            size_t fp = 0u;
            size_t tp = 0u;
            for (float s : benign_scores) {
                if (s < thr) ++fp;
            }
            for (float s : malware_scores) {
                if (s < thr) ++tp;
            }

            const float fpr = static_cast<float>(fp) / static_cast<float>(n_b);
            const float tpr = static_cast<float>(tp) / static_cast<float>(n_m);
            if (fpr > max_fpr) {
                continue;
            }

            float metric = tpr;
            if (strategy == "youden") {
                metric = tpr - fpr;
            } else if (strategy == "f1" || strategy == "fpr") {
                const float precision = static_cast<float>(tp) / (static_cast<float>(tp + fp) + 1e-12f);
                const float recall = tpr;
                const float b2 = beta * beta;
                metric = (1.0f + b2) * precision * recall / (b2 * precision + recall + 1e-12f);
            }

            if (!found || metric > best_metric) {
                found = true;
                best_metric = metric;
                out.threshold = thr;
                out.fpr = fpr;
                out.tpr = tpr;
                out.metric = metric;
                out.has_metric = true;
            }
        }

        if (!found) {
            const If_threshold_result fallback = if_find_threshold_precise(benign_scores, max_fpr);
            out.threshold = fallback.threshold;
            out.fpr = fallback.fpr;
            size_t tp = 0u;
            for (float s : malware_scores) {
                if (s < out.threshold) {
                    ++tp;
                }
            }
            out.tpr = static_cast<float>(tp) / static_cast<float>(malware_scores.size());
            out.metric = out.tpr;
            out.has_metric = false;
        }

        return out;
    }

    using If_binary_metrics = eml_isolation_metrics;

    inline If_binary_metrics if_compute_metrics(const std::vector<float>& benign_scores,
                                                const std::vector<float>& malware_scores,
                                                float threshold) {
        If_binary_metrics out;
        out.reset();
        out.set_metric(eval_metric::ROC_AUC);

        for (float s : benign_scores) {
            // Higher -s means more anomalous
            out.update(false, s < threshold, -s);
        }

        for (float s : malware_scores) {
            out.update(true, s < threshold, -s);
        }

        return out;
    }

} // namespace eml
