#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

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

    class If_tree {
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
                uint8_t split_threshold = 0;
                const uint32_t attempts = std::max<uint32_t>(8u, static_cast<uint32_t>(num_features) * 2u);
                std::uniform_int_distribution<uint16_t> feature_dist(0u, static_cast<uint16_t>(num_features - 1u));

                for (uint32_t a = 0; a < attempts; ++a) {
                    const uint16_t feature = feature_dist(rng);
                    std::array<uint8_t, 256> value_seen{};
                    uint16_t unique_count = 0u;

                    for (size_t i = task.begin; i < task.end; ++i) {
                        const uint8_t value = matrix[static_cast<size_t>(indices[i]) * num_features + feature];
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
                    nodes_[task.node_index].set_leaf(resource_, static_cast<uint32_t>(sample_count), task.depth);
                    depth_ = static_cast<uint16_t>(std::max<uint16_t>(depth_, task.depth));
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
                if (mid == task.begin || mid == task.end || nodes_.size() + 2u > max_nodes_per_tree) {
                    nodes_[task.node_index].set_leaf(resource_, static_cast<uint32_t>(sample_count), task.depth);
                    depth_ = static_cast<uint16_t>(std::max<uint16_t>(depth_, task.depth));
                    continue;
                }

                const uint32_t left_child = static_cast<uint32_t>(nodes_.size());
                nodes_[task.node_index].set_split(resource_, split_feature, split_threshold, left_child);

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

        bool load_serialized(const If_node_resource& resource,
                             const uint64_t* packed_nodes,
                             size_t node_count,
                             uint16_t depth) {
            if (!resource.valid() || !packed_nodes || node_count == 0u) {
                return false;
            }

            resource_ = resource;
            nodes_.clear();
            nodes_.reserve(node_count);
            for (size_t i = 0; i < node_count; ++i) {
                IsoNode node;
                node.packed_data = packed_nodes[i];
                nodes_.push_back(node);
            }

            depth_ = depth;
            is_loaded_ = true;
            return true;
        }

        const std::vector<IsoNode>& nodes() const {
            return nodes_;
        }

        size_t node_count() const { return nodes_.size(); }
        uint16_t depth() const { return depth_; }
        bool loaded() const { return is_loaded_; }
    };

    class If_tree_container {
    private:
        std::vector<If_tree> trees_;
        If_node_resource resource_{};
        uint32_t samples_per_tree_ = 1u;
        float threshold_offset_ = 0.0f;
        bool trained_ = false;

    public:
        void unload_model() {
            trees_.clear();
            trees_.shrink_to_fit();
            trained_ = false;
        }

        void reserve_tree_slots(uint16_t tree_count) {
            if (tree_count > 0u) {
                trees_.reserve(tree_count);
            }
        }

        bool set_node_resource_layout(uint8_t threshold_bits,
                                      uint8_t feature_bits,
                                      uint8_t child_bits,
                                      uint8_t leaf_size_bits,
                                      uint8_t depth_bits) {
            return resource_.set_bits(threshold_bits, feature_bits, child_bits, leaf_size_bits, depth_bits);
        }

        void set_samples_per_tree(uint32_t samples_per_tree) {
            samples_per_tree_ = std::max<uint32_t>(1u, samples_per_tree);
        }

        void set_threshold_offset(float threshold_offset) {
            threshold_offset_ = threshold_offset;
        }

        void add_trained_tree(const If_tree& tree) {
            trees_.push_back(tree);
            trained_ = !trees_.empty();
        }

        const If_node_resource& node_resource() const {
            return resource_;
        }

        float score_samples(const uint8_t* quantized_features, uint16_t num_features) const {
            if (!trained_ || !quantized_features || trees_.empty()) {
                return 0.0f;
            }

            double acc = 0.0;
            for (const If_tree& tree : trees_) {
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

        bool save_model_binary(const std::filesystem::path& file_path) const {
            if (!trained_ || trees_.empty() || !resource_.valid()) {
                return false;
            }

            const std::filesystem::path parent = file_path.parent_path();
            if (!parent.empty()) {
                std::filesystem::create_directories(parent);
            }
            std::ofstream fout(file_path, std::ios::binary | std::ios::trunc);
            if (!fout.is_open()) {
                return false;
            }

            const auto write_exact = [&fout](const void* src, size_t bytes) -> bool {
                fout.write(reinterpret_cast<const char*>(src), static_cast<std::streamsize>(bytes));
                return static_cast<bool>(fout);
            };

            const char magic[4] = {'I', 'F', 'R', '1'};
            const uint16_t version = 1u;
            const uint32_t tree_count = static_cast<uint32_t>(trees_.size());

            if (!write_exact(magic, sizeof(magic))) return false;
            if (!write_exact(&version, sizeof(version))) return false;

            const uint8_t threshold_bits = resource_.threshold_bits();
            const uint8_t feature_bits = resource_.feature_bits();
            const uint8_t child_bits = resource_.child_bits();
            const uint8_t leaf_size_bits = resource_.leaf_size_bits();
            const uint8_t depth_bits = resource_.depth_bits();

            if (!write_exact(&threshold_bits, sizeof(threshold_bits))) return false;
            if (!write_exact(&feature_bits, sizeof(feature_bits))) return false;
            if (!write_exact(&child_bits, sizeof(child_bits))) return false;
            if (!write_exact(&leaf_size_bits, sizeof(leaf_size_bits))) return false;
            if (!write_exact(&depth_bits, sizeof(depth_bits))) return false;

            if (!write_exact(&samples_per_tree_, sizeof(samples_per_tree_))) return false;
            if (!write_exact(&threshold_offset_, sizeof(threshold_offset_))) return false;
            if (!write_exact(&tree_count, sizeof(tree_count))) return false;

            for (const If_tree& tree : trees_) {
                const uint16_t tree_depth = tree.depth();
                const uint32_t node_count = static_cast<uint32_t>(tree.node_count());
                if (!write_exact(&tree_depth, sizeof(tree_depth))) return false;
                if (!write_exact(&node_count, sizeof(node_count))) return false;
                const std::vector<IsoNode>& nodes = tree.nodes();
                for (const IsoNode& node : nodes) {
                    if (!write_exact(&node.packed_data, sizeof(node.packed_data))) return false;
                }
            }

            return true;
        }

        bool load_model_binary(const std::filesystem::path& file_path) {
            std::ifstream fin(file_path, std::ios::binary);
            if (!fin.is_open()) {
                return false;
            }

            const auto read_exact = [&fin](void* dst, size_t bytes) -> bool {
                fin.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
                return static_cast<size_t>(fin.gcount()) == bytes;
            };

            char magic[4] = {0, 0, 0, 0};
            uint16_t version = 0u;
            if (!read_exact(magic, sizeof(magic)) ||
                !read_exact(&version, sizeof(version))) {
                return false;
            }
            if (magic[0] != 'I' || magic[1] != 'F' || magic[2] != 'R' || magic[3] != '1' || version != 1u) {
                return false;
            }

            uint8_t threshold_bits = 0u;
            uint8_t feature_bits = 0u;
            uint8_t child_bits = 0u;
            uint8_t leaf_size_bits = 0u;
            uint8_t depth_bits = 0u;
            if (!read_exact(&threshold_bits, sizeof(threshold_bits))) return false;
            if (!read_exact(&feature_bits, sizeof(feature_bits))) return false;
            if (!read_exact(&child_bits, sizeof(child_bits))) return false;
            if (!read_exact(&leaf_size_bits, sizeof(leaf_size_bits))) return false;
            if (!read_exact(&depth_bits, sizeof(depth_bits))) return false;

            if (!resource_.set_bits(threshold_bits, feature_bits, child_bits, leaf_size_bits, depth_bits)) {
                return false;
            }

            if (!read_exact(&samples_per_tree_, sizeof(samples_per_tree_))) return false;
            if (!read_exact(&threshold_offset_, sizeof(threshold_offset_))) return false;

            uint32_t tree_count = 0u;
            if (!read_exact(&tree_count, sizeof(tree_count))) return false;

            trees_.clear();
            trees_.reserve(tree_count);

            for (uint32_t i = 0; i < tree_count; ++i) {
                uint16_t tree_depth = 0u;
                uint32_t node_count = 0u;
                if (!read_exact(&tree_depth, sizeof(tree_depth))) return false;
                if (!read_exact(&node_count, sizeof(node_count))) return false;
                if (node_count == 0u) {
                    return false;
                }

                std::vector<uint64_t> packed_nodes(node_count, 0ull);
                for (uint32_t n = 0u; n < node_count; ++n) {
                    if (!read_exact(&packed_nodes[n], sizeof(uint64_t))) {
                        return false;
                    }
                }

                If_tree tree;
                if (!tree.load_serialized(resource_, packed_nodes.data(), packed_nodes.size(), tree_depth)) {
                    return false;
                }
                trees_.push_back(tree);
            }

            trained_ = !trees_.empty();
            return trained_;
        }

        size_t num_trees() const { return trees_.size(); }
        uint32_t samples_per_tree() const { return samples_per_tree_; }
        bool trained() const { return trained_; }
        float threshold_offset() const { return threshold_offset_; }
    };

} // namespace eml
