#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <vector>

namespace eml {

    inline double plain_if_c_factor(uint32_t n) {
        if (n <= 1u) {
            return 0.0;
        }
        if (n == 2u) {
            return 1.0;
        }
        const double nd = static_cast<double>(n);
        return 2.0 * (std::log(nd - 1.0) + 0.5772156649015329) - (2.0 * (nd - 1.0) / nd);
    }

    struct PlainIsoNode {
        bool is_leaf = true;
        uint16_t feature_index = 0u;
        float threshold = 0.0f;
        uint32_t left_child = 0u;
        uint32_t right_child = 0u;
        uint32_t leaf_size = 1u;
        uint16_t depth = 0u;
    };

    class PlainIsoTree {
    private:
        struct BuildTask {
            uint32_t node_index = 0u;
            size_t begin = 0u;
            size_t end = 0u;
            uint16_t depth = 0u;
        };

        std::vector<PlainIsoNode> nodes_;
        uint16_t max_depth_ = 1u;

    public:
        bool train(const std::vector<float>& matrix,
                   uint16_t num_features,
                   std::vector<uint32_t> sampled_indices,
                   uint16_t max_depth,
                   std::mt19937& rng,
                   float max_features_fraction = 1.0f) {
            if (matrix.empty() || num_features == 0u || sampled_indices.empty() || max_depth == 0u) {
                return false;
            }

            nodes_.clear();
            nodes_.reserve(std::max<size_t>(32u, sampled_indices.size()));
            nodes_.push_back(PlainIsoNode{});

            std::queue<BuildTask> tasks;
            tasks.push(BuildTask{0u, 0u, sampled_indices.size(), 0u});

            max_depth_ = 0u;

            const uint16_t feature_trials = static_cast<uint16_t>(std::max<int>(8, static_cast<int>(num_features) * 2));
            std::uniform_int_distribution<uint16_t> feature_dist(0u, static_cast<uint16_t>(num_features - 1u));

            uint16_t num_candidate_features = static_cast<uint16_t>(
                std::max<double>(1.0, std::round(max_features_fraction * static_cast<double>(num_features)))
            );
            if (num_candidate_features > num_features) {
                num_candidate_features = num_features;
            }

            while (!tasks.empty()) {
                const BuildTask task = tasks.front();
                tasks.pop();

                const size_t sample_count = task.end - task.begin;
                if (sample_count <= 1u || task.depth >= max_depth) {
                    PlainIsoNode& leaf = nodes_[task.node_index];
                    leaf.is_leaf = true;
                    leaf.leaf_size = static_cast<uint32_t>(sample_count);
                    leaf.depth = task.depth;
                    max_depth_ = std::max<uint16_t>(max_depth_, task.depth);
                    continue;
                }

                bool found_split = false;
                uint16_t split_feature = 0u;
                float split_min = 0.0f;
                float split_max = 0.0f;

                for (uint16_t trial = 0u; trial < feature_trials; ++trial) {
                    const uint16_t feature = feature_dist(rng);

                    float local_min = std::numeric_limits<float>::infinity();
                    float local_max = -std::numeric_limits<float>::infinity();

                    for (size_t index = task.begin; index < task.end; ++index) {
                        const size_t row = sampled_indices[index];
                        const float value = matrix[row * static_cast<size_t>(num_features) + feature];
                        local_min = std::min(local_min, value);
                        local_max = std::max(local_max, value);
                    }

                    if (std::isfinite(local_min) && std::isfinite(local_max) && local_min < local_max) {
                        found_split = true;
                        split_feature = feature;
                        split_min = local_min;
                        split_max = local_max;
                        if (trial + 1u >= num_candidate_features) {
                            break;
                        }
                    }
                }

                if (!found_split) {
                    PlainIsoNode& leaf = nodes_[task.node_index];
                    leaf.is_leaf = true;
                    leaf.leaf_size = static_cast<uint32_t>(sample_count);
                    leaf.depth = task.depth;
                    max_depth_ = std::max<uint16_t>(max_depth_, task.depth);
                    continue;
                }

                std::uniform_real_distribution<float> threshold_dist(split_min, split_max);
                const float threshold = threshold_dist(rng);

                size_t left = task.begin;
                size_t right = task.end;
                while (left < right) {
                    const size_t row = sampled_indices[left];
                    const float value = matrix[row * static_cast<size_t>(num_features) + split_feature];
                    if (value < threshold) {
                        ++left;
                    } else {
                        --right;
                        std::swap(sampled_indices[left], sampled_indices[right]);
                    }
                }

                const size_t mid = left;
                if (mid == task.begin || mid == task.end) {
                    PlainIsoNode& leaf = nodes_[task.node_index];
                    leaf.is_leaf = true;
                    leaf.leaf_size = static_cast<uint32_t>(sample_count);
                    leaf.depth = task.depth;
                    max_depth_ = std::max<uint16_t>(max_depth_, task.depth);
                    continue;
                }

                const uint32_t left_child_index = static_cast<uint32_t>(nodes_.size());
                const uint32_t right_child_index = static_cast<uint32_t>(left_child_index + 1u);

                PlainIsoNode& split = nodes_[task.node_index];
                split.is_leaf = false;
                split.feature_index = split_feature;
                split.threshold = threshold;
                split.left_child = left_child_index;
                split.right_child = right_child_index;
                split.depth = task.depth;

                nodes_.push_back(PlainIsoNode{});
                nodes_.push_back(PlainIsoNode{});

                const uint16_t child_depth = static_cast<uint16_t>(task.depth + 1u);
                tasks.push(BuildTask{left_child_index, task.begin, mid, child_depth});
                tasks.push(BuildTask{right_child_index, mid, task.end, child_depth});
            }

            return !nodes_.empty();
        }

        float path_length(const float* sample, uint16_t num_features) const {
            if (!sample || num_features == 0u || nodes_.empty()) {
                return 0.0f;
            }

            uint32_t node_index = 0u;
            while (node_index < nodes_.size()) {
                const PlainIsoNode& node = nodes_[node_index];
                if (node.is_leaf) {
                    const uint32_t leaf_size = std::max<uint32_t>(1u, node.leaf_size);
                    const double adjusted_path = static_cast<double>(node.depth) + plain_if_c_factor(leaf_size);
                    return static_cast<float>(adjusted_path);
                }

                if (node.feature_index >= num_features) {
                    return 0.0f;
                }

                const float value = sample[node.feature_index];
                node_index = (value < node.threshold) ? node.left_child : node.right_child;
            }

            return 0.0f;
        }

        size_t node_count() const { return nodes_.size(); }
        uint16_t max_depth() const { return max_depth_; }
    };

    class PlainIsoForest {
    private:
        std::vector<PlainIsoTree> trees_;
        uint16_t num_features_ = 0u;
        uint16_t n_estimators_ = 200u;
        uint32_t max_samples_per_tree_ = 1u;
        float max_features_fraction_ = 1.0f;
        bool bootstrap_ = false;
        uint32_t random_state_ = 42u;
        uint16_t max_depth_ = 1u;
        float threshold_ = 0.0f;

        static uint32_t resolve_max_samples(size_t num_samples, float max_samples_param) {
            if (num_samples == 0u) {
                return 1u;
            }

            uint32_t resolved = 1u;
            if (max_samples_param <= 1.0f) {
                const float ratio = (max_samples_param <= 0.0f) ? 1.0f : max_samples_param;
                resolved = static_cast<uint32_t>(std::ceil(ratio * static_cast<float>(num_samples)));
            } else {
                resolved = static_cast<uint32_t>(std::ceil(max_samples_param));
            }

            resolved = std::max<uint32_t>(1u, resolved);
            resolved = std::min<uint32_t>(resolved, static_cast<uint32_t>(num_samples));
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
                for (uint32_t i = 0u; i < sample_size; ++i) {
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
        bool train(const std::vector<float>& matrix,
                   size_t num_samples,
                   uint16_t num_features,
                   uint16_t n_estimators,
                   float max_samples,
                   float max_features,
                   bool bootstrap,
                   uint32_t random_state) {
            if (matrix.empty() || num_samples == 0u || num_features == 0u) {
                return false;
            }

            if (matrix.size() != num_samples * static_cast<size_t>(num_features)) {
                return false;
            }

            num_features_ = num_features;
            n_estimators_ = std::max<uint16_t>(1u, n_estimators);
            bootstrap_ = bootstrap;
            random_state_ = random_state;

            max_samples_per_tree_ = resolve_max_samples(num_samples, max_samples);
            max_features_fraction_ = std::max(0.01f, std::min(1.0f, max_features));
            max_depth_ = static_cast<uint16_t>(
                std::max<int>(1, static_cast<int>(std::ceil(std::log2(static_cast<double>(max_samples_per_tree_)))))
            );

            trees_.clear();
            trees_.reserve(n_estimators_);

            std::mt19937 rng(random_state_);
            for (uint16_t tree_index = 0u; tree_index < n_estimators_; ++tree_index) {
                std::vector<uint32_t> sampled = sample_indices(
                    num_samples,
                    max_samples_per_tree_,
                    bootstrap_,
                    rng
                );

                PlainIsoTree tree;
                if (!tree.train(matrix,
                                num_features_,
                                std::move(sampled),
                                max_depth_,
                                rng,
                                max_features_fraction_)) {
                    trees_.clear();
                    return false;
                }
                trees_.push_back(std::move(tree));
            }

            return !trees_.empty();
        }

        float score_sample(const float* sample) const {
            if (!sample || trees_.empty() || num_features_ == 0u) {
                return 0.0f;
            }

            double path_sum = 0.0;
            for (const PlainIsoTree& tree : trees_) {
                path_sum += static_cast<double>(tree.path_length(sample, num_features_));
            }

            const double avg_path = path_sum / static_cast<double>(trees_.size());
            const double c_n = plain_if_c_factor(max_samples_per_tree_);
            if (c_n <= 0.0) {
                return 0.0f;
            }

            const double anomaly_score = std::pow(2.0, -avg_path / c_n);
            return static_cast<float>(-anomaly_score);
        }

        std::vector<float> score_matrix(const std::vector<float>& matrix, size_t num_samples) const {
            std::vector<float> out;
            if (trees_.empty() || num_features_ == 0u || matrix.size() != num_samples * static_cast<size_t>(num_features_)) {
                return out;
            }

            out.reserve(num_samples);
            for (size_t row = 0u; row < num_samples; ++row) {
                const float* sample = &matrix[row * static_cast<size_t>(num_features_)];
                out.push_back(score_sample(sample));
            }
            return out;
        }

        static float select_threshold_tpr_with_fpr_cap(const std::vector<float>& benign_scores,
                                                       const std::vector<float>& malware_scores,
                                                       float max_fpr,
                                                       float& out_fpr,
                                                       float& out_tpr) {
            out_fpr = 0.0f;
            out_tpr = 0.0f;

            if (benign_scores.empty() || malware_scores.empty()) {
                return 0.0f;
            }

            std::vector<float> thresholds;
            thresholds.reserve(benign_scores.size() + malware_scores.size());
            thresholds.insert(thresholds.end(), benign_scores.begin(), benign_scores.end());
            thresholds.insert(thresholds.end(), malware_scores.begin(), malware_scores.end());
            std::sort(thresholds.begin(), thresholds.end());
            thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());

            const float cap = std::max(0.0f, std::min(1.0f, max_fpr));
            const size_t benign_count = benign_scores.size();
            const size_t malware_count = malware_scores.size();

            bool found = false;
            float best_threshold = thresholds.front();
            float best_fpr = 1.0f;
            float best_tpr = -1.0f;

            for (float threshold : thresholds) {
                size_t fp = 0u;
                size_t tp = 0u;

                for (float score : benign_scores) {
                    if (score < threshold) {
                        ++fp;
                    }
                }

                for (float score : malware_scores) {
                    if (score < threshold) {
                        ++tp;
                    }
                }

                const float fpr = static_cast<float>(fp) / static_cast<float>(benign_count);
                if (fpr > cap) {
                    continue;
                }

                const float tpr = static_cast<float>(tp) / static_cast<float>(malware_count);
                if (!found || tpr > best_tpr || (tpr == best_tpr && fpr < best_fpr)) {
                    found = true;
                    best_threshold = threshold;
                    best_fpr = fpr;
                    best_tpr = tpr;
                }
            }

            if (!found) {
                std::vector<float> sorted_benign = benign_scores;
                std::sort(sorted_benign.begin(), sorted_benign.end());
                const size_t max_fp = static_cast<size_t>(std::floor(cap * static_cast<float>(sorted_benign.size())));
                best_threshold = (max_fp == 0u) ? (sorted_benign.front() - 1e-10f) : sorted_benign[max_fp - 1u];

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

        static void compute_metrics(const std::vector<float>& benign_scores,
                                    const std::vector<float>& malware_scores,
                                    float threshold,
                                    float& out_fpr,
                                    float& out_tpr) {
            out_fpr = 0.0f;
            out_tpr = 0.0f;
            if (benign_scores.empty() || malware_scores.empty()) {
                return;
            }

            size_t fp = 0u;
            size_t tp = 0u;
            for (float score : benign_scores) {
                if (score < threshold) {
                    ++fp;
                }
            }
            for (float score : malware_scores) {
                if (score < threshold) {
                    ++tp;
                }
            }

            out_fpr = static_cast<float>(fp) / static_cast<float>(benign_scores.size());
            out_tpr = static_cast<float>(tp) / static_cast<float>(malware_scores.size());
        }

        uint16_t num_features() const { return num_features_; }
        uint16_t n_estimators() const { return n_estimators_; }
        uint32_t max_samples_per_tree() const { return max_samples_per_tree_; }
        uint16_t max_depth() const { return max_depth_; }

        void set_threshold(float threshold) { threshold_ = threshold; }
        float threshold() const { return threshold_; }
        bool predict_is_anomaly(const float* sample) const {
            return score_sample(sample) < threshold_;
        }
    };

} // namespace eml
