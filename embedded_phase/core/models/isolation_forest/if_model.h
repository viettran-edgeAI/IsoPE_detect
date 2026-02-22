#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "if_config.h"
#include "if_node_resource.h"

namespace eml {

    struct IsoNode {
        uint64_t packed_data = 0ull;

        bool is_leaf() const {
            return (packed_data & 1ull) != 0ull;
        }

        void set_is_leaf(bool leaf) {
            if (leaf) {
                packed_data |= 1ull;
            } else {
                packed_data &= ~1ull;
            }
        }

        void set_split(const If_node_resource& resource,
                       uint16_t feature_id,
                       uint8_t threshold_slot,
                       uint32_t left_child_index) {
            packed_data = 0ull;
            set_is_leaf(false);
            resource.write_field(packed_data, resource.get_split_threshold_layout(), threshold_slot);
            resource.write_field(packed_data, resource.get_split_feature_layout(), feature_id);
            resource.write_field(packed_data, resource.get_split_child_layout(), left_child_index);
        }

        void set_leaf(const If_node_resource& resource,
                      uint32_t leaf_size,
                      uint16_t leaf_depth) {
            packed_data = 0ull;
            set_is_leaf(true);
            resource.write_field(packed_data, resource.get_leaf_size_layout(), leaf_size);
            resource.write_field(packed_data, resource.get_leaf_depth_layout(), leaf_depth);
        }

        uint8_t threshold_slot(const If_node_resource& resource) const {
            return static_cast<uint8_t>(resource.read_field(packed_data, resource.get_split_threshold_layout()));
        }

        uint16_t feature_id(const If_node_resource& resource) const {
            return static_cast<uint16_t>(resource.read_field(packed_data, resource.get_split_feature_layout()));
        }

        uint32_t left_child(const If_node_resource& resource) const {
            return static_cast<uint32_t>(resource.read_field(packed_data, resource.get_split_child_layout()));
        }

        uint32_t leaf_size(const If_node_resource& resource) const {
            return static_cast<uint32_t>(resource.read_field(packed_data, resource.get_leaf_size_layout()));
        }

        uint16_t leaf_depth(const If_node_resource& resource) const {
            return static_cast<uint16_t>(resource.read_field(packed_data, resource.get_leaf_depth_layout()));
        }
    };

    inline double if_c_factor(uint32_t n) {
        if (n <= 1u) {
            return 0.0;
        }
        if (n == 2u) {
            return 1.0;
        }
        const double nd = static_cast<double>(n);
        return 2.0 * (std::log(nd - 1.0) + 0.5772156649015329) - (2.0 * (nd - 1.0) / nd);
    }

    class IsoTree {
    private:
        struct BuildTask {
            uint32_t node_index = 0;
            size_t begin = 0;
            size_t end = 0;
            uint16_t depth = 0;
        };

        std::vector<IsoNode> nodes_;
        If_node_resource resource_{};
        uint16_t depth_ = 0;
        bool is_loaded_ = false;

    public:
        void set_resource(const If_node_resource& resource) {
            resource_ = resource;
        }

        const If_node_resource& resource() const {
            return resource_;
        }

        bool train(const uint8_t* matrix,
                   size_t num_samples,
                   uint16_t num_features,
                   const std::vector<uint32_t>& sampled_indices,
                   uint16_t max_depth,
                   uint32_t max_nodes_per_tree,
                   std::mt19937& rng) {
            (void)num_samples;
            if (!matrix || num_features == 0 || sampled_indices.empty() || !resource_.valid()) {
                return false;
            }

            std::vector<uint32_t> indices = sampled_indices;
            nodes_.clear();
            nodes_.reserve(std::max<uint32_t>(8u, std::min<uint32_t>(max_nodes_per_tree, 2048u)));
            nodes_.push_back(IsoNode{});

            std::vector<BuildTask> queue;
            queue.reserve(256);
            queue.push_back(BuildTask{0u, 0u, indices.size(), 0u});

            depth_ = 0;
            size_t queue_head = 0;

            while (queue_head < queue.size()) {
                const BuildTask task = queue[queue_head++];
                const size_t sample_count = task.end - task.begin;

                if (sample_count <= 1u || task.depth >= max_depth) {
                    nodes_[task.node_index].set_leaf(resource_, static_cast<uint32_t>(sample_count), task.depth);
                    depth_ = static_cast<uint16_t>(std::max<uint16_t>(depth_, task.depth));
                    continue;
                }

                bool found_split = false;
                uint16_t split_feature = 0;
                uint8_t min_value = 0;
                uint8_t max_value = 0;
                const uint32_t attempts = std::max<uint32_t>(8u, static_cast<uint32_t>(num_features) * 2u);
                std::uniform_int_distribution<uint16_t> feature_dist(0u, static_cast<uint16_t>(num_features - 1u));

                for (uint32_t a = 0; a < attempts; ++a) {
                    const uint16_t feature = feature_dist(rng);
                    uint8_t local_min = 255u;
                    uint8_t local_max = 0u;
                    for (size_t i = task.begin; i < task.end; ++i) {
                        const uint8_t value = matrix[static_cast<size_t>(indices[i]) * num_features + feature];
                        if (value < local_min) local_min = value;
                        if (value > local_max) local_max = value;
                    }
                    if (local_min < local_max) {
                        found_split = true;
                        split_feature = feature;
                        min_value = local_min;
                        max_value = local_max;
                        break;
                    }
                }

                if (!found_split) {
                    nodes_[task.node_index].set_leaf(resource_, static_cast<uint32_t>(sample_count), task.depth);
                    depth_ = static_cast<uint16_t>(std::max<uint16_t>(depth_, task.depth));
                    continue;
                }

                std::uniform_int_distribution<int> threshold_dist(
                    static_cast<int>(min_value),
                    static_cast<int>(max_value) - 1
                );
                const uint8_t threshold = static_cast<uint8_t>(threshold_dist(rng));

                size_t left = task.begin;
                size_t right = task.end;
                while (left < right) {
                    const uint8_t value = matrix[static_cast<size_t>(indices[left]) * num_features + split_feature];
                    if (value <= threshold) {
                        ++left;
                    } else {
                        --right;
                        std::swap(indices[left], indices[right]);
                    }
                }

                const size_t mid = left;
                if (mid == task.begin || mid == task.end || nodes_.size() + 2u > max_nodes_per_tree) {
                    nodes_[task.node_index].set_leaf(resource_, static_cast<uint32_t>(sample_count), task.depth);
                    depth_ = static_cast<uint16_t>(std::max<uint16_t>(depth_, task.depth));
                    continue;
                }

                const uint32_t left_child = static_cast<uint32_t>(nodes_.size());
                nodes_[task.node_index].set_split(resource_, split_feature, threshold, left_child);

                nodes_.push_back(IsoNode{});
                nodes_.push_back(IsoNode{});

                const uint16_t child_depth = static_cast<uint16_t>(task.depth + 1u);
                queue.push_back(BuildTask{left_child, task.begin, mid, child_depth});
                queue.push_back(BuildTask{static_cast<uint32_t>(left_child + 1u), mid, task.end, child_depth});
            }

            is_loaded_ = !nodes_.empty();
            return is_loaded_;
        }

        float path_length(const uint8_t* quantized_features, uint16_t num_features) const {
            if (!is_loaded_ || !quantized_features || num_features == 0 || nodes_.empty()) {
                return 0.0f;
            }

            uint32_t node_index = 0;
            while (node_index < nodes_.size()) {
                const IsoNode& node = nodes_[node_index];
                if (node.is_leaf()) {
                    const uint32_t leaf_size = std::max<uint32_t>(1u, node.leaf_size(resource_));
                    const uint16_t leaf_depth = node.leaf_depth(resource_);
                    const double path = static_cast<double>(leaf_depth) + if_c_factor(leaf_size);
                    return static_cast<float>(path);
                }

                const uint16_t feature = node.feature_id(resource_);
                if (feature >= num_features) {
                    return 0.0f;
                }

                const uint8_t threshold = node.threshold_slot(resource_);
                const uint32_t left_child = node.left_child(resource_);
                const uint32_t next = (quantized_features[feature] <= threshold)
                    ? left_child
                    : static_cast<uint32_t>(left_child + 1u);

                if (next >= nodes_.size()) {
                    return 0.0f;
                }
                node_index = next;
            }

            return 0.0f;
        }

        size_t node_count() const { return nodes_.size(); }
        uint16_t depth() const { return depth_; }
        bool loaded() const { return is_loaded_; }
    };

    class QuantizedIsolationForest {
    private:
        std::vector<IsoTree> trees_;
        If_node_resource resource_{};
        If_config config_{};
        uint32_t samples_per_tree_ = 1u;
        float threshold_offset_ = 0.0f;
        bool trained_ = false;

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

    public:
        bool train(const uint8_t* matrix,
                   size_t num_samples,
                   uint16_t num_features,
                   const If_config& cfg) {
            if (!matrix || num_samples == 0u || num_features == 0 || !cfg.isLoaded) {
                return false;
            }

            config_ = cfg;
            threshold_offset_ = cfg.threshold_offset;
            if (!resource_.set_bits(cfg.threshold_bits,
                                    cfg.feature_bits,
                                    cfg.child_bits,
                                    cfg.leaf_size_bits,
                                    cfg.depth_bits)) {
                return false;
            }

            samples_per_tree_ = resolve_samples_per_tree(cfg, num_samples);
            const uint16_t n_estimators = std::max<uint16_t>(1u, cfg.n_estimators);
            const uint32_t max_nodes_per_tree = std::max<uint32_t>(1u, cfg.max_nodes_per_tree);
            const uint16_t max_depth = std::max<uint16_t>(1u, cfg.max_depth);

            std::mt19937 rng(cfg.random_state);
            trees_.clear();
            trees_.reserve(n_estimators);

            for (uint16_t i = 0; i < n_estimators; ++i) {
                IsoTree tree;
                tree.set_resource(resource_);
                const std::vector<uint32_t> picked = sample_indices(num_samples, samples_per_tree_, cfg.bootstrap, rng);
                if (!tree.train(matrix,
                                num_samples,
                                num_features,
                                picked,
                                max_depth,
                                max_nodes_per_tree,
                                rng)) {
                    return false;
                }
                trees_.push_back(tree);
            }

            trained_ = !trees_.empty();
            return trained_;
        }

        float score_samples(const uint8_t* quantized_features, uint16_t num_features) const {
            if (!trained_ || !quantized_features || trees_.empty()) {
                return 0.0f;
            }

            double acc = 0.0;
            for (const IsoTree& tree : trees_) {
                acc += static_cast<double>(tree.path_length(quantized_features, num_features));
            }
            const double avg_path = acc / static_cast<double>(trees_.size());
            const double c_n = if_c_factor(samples_per_tree_);
            if (c_n <= 0.0) {
                return 0.0f;
            }

            const double anomaly_score = std::pow(2.0, -avg_path / c_n);
            return static_cast<float>(-anomaly_score);
        }

        float decision_function(const uint8_t* quantized_features, uint16_t num_features) const {
            return score_samples(quantized_features, num_features) - threshold_offset_;
        }

        bool is_anomaly(const uint8_t* quantized_features,
                        uint16_t num_features,
                        float threshold) const {
            return decision_function(quantized_features, num_features) < threshold;
        }

        size_t num_trees() const { return trees_.size(); }
        uint32_t samples_per_tree() const { return samples_per_tree_; }
        bool trained() const { return trained_; }
        float threshold_offset() const { return threshold_offset_; }
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

    inline double if_roc_auc(const std::vector<float>& y_scores, const std::vector<uint8_t>& y_true) {
        const size_t n = y_scores.size();
        if (n == 0u || y_true.size() != n) {
            return 0.0;
        }

        size_t n_pos = 0u;
        for (uint8_t y : y_true) {
            if (y != 0u) {
                ++n_pos;
            }
        }
        const size_t n_neg = n - n_pos;
        if (n_pos == 0u || n_neg == 0u) {
            return 0.5;
        }

        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(), [&y_scores](size_t a, size_t b) {
            return y_scores[a] < y_scores[b];
        });

        double rank_sum_pos = 0.0;
        size_t i = 0u;
        while (i < n) {
            size_t j = i + 1u;
            while (j < n && y_scores[order[j]] == y_scores[order[i]]) {
                ++j;
            }

            const double avg_rank = 0.5 * (static_cast<double>(i + 1u) + static_cast<double>(j));
            for (size_t k = i; k < j; ++k) {
                if (y_true[order[k]] != 0u) {
                    rank_sum_pos += avg_rank;
                }
            }
            i = j;
        }

        const double u = rank_sum_pos - (static_cast<double>(n_pos) * static_cast<double>(n_pos + 1u) * 0.5);
        return u / (static_cast<double>(n_pos) * static_cast<double>(n_neg));
    }

    struct If_binary_metrics {
        float fpr = 0.0f;
        float tpr = 0.0f;
        float roc_auc = 0.0f;
    };

    inline If_binary_metrics if_compute_metrics(const std::vector<float>& benign_scores,
                                                const std::vector<float>& malware_scores,
                                                float threshold) {
        If_binary_metrics out;
        if (benign_scores.empty() || malware_scores.empty()) {
            return out;
        }

        size_t fp = 0u;
        for (float s : benign_scores) {
            if (s < threshold) {
                ++fp;
            }
        }

        size_t tp = 0u;
        for (float s : malware_scores) {
            if (s < threshold) {
                ++tp;
            }
        }

        out.fpr = static_cast<float>(fp) / static_cast<float>(benign_scores.size());
        out.tpr = static_cast<float>(tp) / static_cast<float>(malware_scores.size());

        std::vector<float> y_scores;
        std::vector<uint8_t> y_true;
        y_scores.reserve(benign_scores.size() + malware_scores.size());
        y_true.reserve(benign_scores.size() + malware_scores.size());

        for (float s : benign_scores) {
            y_scores.push_back(-s);
            y_true.push_back(0u);
        }
        for (float s : malware_scores) {
            y_scores.push_back(-s);
            y_true.push_back(1u);
        }

        out.roc_auc = static_cast<float>(if_roc_auc(y_scores, y_true));
        return out;
    }

} // namespace eml
