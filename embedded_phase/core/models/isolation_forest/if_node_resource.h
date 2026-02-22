#pragma once

#include <algorithm>
#include <cstdint>

namespace eml {

    struct if_field_layout {
        uint8_t offset = 0;
        uint8_t width = 0;
    };

    class If_node_resource {
    private:
        uint8_t threshold_bits_ = 2;
        uint8_t feature_bits_ = 1;
        uint8_t child_bits_ = 1;
        uint8_t leaf_size_bits_ = 1;
        uint8_t depth_bits_ = 1;

        if_field_layout split_threshold_layout_{};
        if_field_layout split_feature_layout_{};
        if_field_layout split_child_layout_{};

        if_field_layout leaf_size_layout_{};
        if_field_layout leaf_depth_layout_{};

        uint8_t bits_per_node_ = 0;

        static uint8_t normalize_bits(uint8_t bits) {
            return bits == 0 ? 1 : bits;
        }

        static uint64_t mask_of(uint8_t width) {
            if (width == 0) {
                return 0ull;
            }
            if (width >= 64) {
                return ~0ull;
            }
            return (1ull << width) - 1ull;
        }

        void recompute_layouts() {
            threshold_bits_ = normalize_bits(threshold_bits_);
            feature_bits_ = normalize_bits(feature_bits_);
            child_bits_ = normalize_bits(child_bits_);
            leaf_size_bits_ = normalize_bits(leaf_size_bits_);
            depth_bits_ = normalize_bits(depth_bits_);

            split_threshold_layout_ = {1u, threshold_bits_};
            split_feature_layout_ = {static_cast<uint8_t>(1u + threshold_bits_), feature_bits_};
            split_child_layout_ = {static_cast<uint8_t>(1u + threshold_bits_ + feature_bits_), child_bits_};

            leaf_size_layout_ = {1u, leaf_size_bits_};
            leaf_depth_layout_ = {static_cast<uint8_t>(1u + leaf_size_bits_), depth_bits_};

            const uint8_t split_bits = static_cast<uint8_t>(1u + threshold_bits_ + feature_bits_ + child_bits_);
            const uint8_t leaf_bits = static_cast<uint8_t>(1u + leaf_size_bits_ + depth_bits_);
            bits_per_node_ = static_cast<uint8_t>(std::max(split_bits, leaf_bits));
        }

    public:
        If_node_resource() {
            recompute_layouts();
        }

        bool set_bits(uint8_t threshold_bits,
                      uint8_t feature_bits,
                      uint8_t child_bits,
                      uint8_t leaf_size_bits,
                      uint8_t depth_bits) {
            threshold_bits_ = threshold_bits;
            feature_bits_ = feature_bits;
            child_bits_ = child_bits;
            leaf_size_bits_ = leaf_size_bits;
            depth_bits_ = depth_bits;
            recompute_layouts();
            return bits_per_node_ <= 64;
        }

        bool valid() const {
            return bits_per_node_ <= 64;
        }

        uint8_t bits_per_node() const {
            return bits_per_node_;
        }

        uint8_t threshold_bits() const { return threshold_bits_; }
        uint8_t feature_bits() const { return feature_bits_; }
        uint8_t child_bits() const { return child_bits_; }
        uint8_t leaf_size_bits() const { return leaf_size_bits_; }
        uint8_t depth_bits() const { return depth_bits_; }

        if_field_layout get_split_threshold_layout() const { return split_threshold_layout_; }
        if_field_layout get_split_feature_layout() const { return split_feature_layout_; }
        if_field_layout get_split_child_layout() const { return split_child_layout_; }
        if_field_layout get_leaf_size_layout() const { return leaf_size_layout_; }
        if_field_layout get_leaf_depth_layout() const { return leaf_depth_layout_; }

        uint64_t read_field(uint64_t packed, const if_field_layout& layout) const {
            if (layout.width == 0) {
                return 0ull;
            }
            return (packed >> layout.offset) & mask_of(layout.width);
        }

        void write_field(uint64_t& packed, const if_field_layout& layout, uint64_t value) const {
            if (layout.width == 0) {
                return;
            }
            const uint64_t field_mask = mask_of(layout.width);
            const uint64_t shifted_mask = field_mask << layout.offset;
            packed = (packed & ~shifted_mask) | ((value & field_mask) << layout.offset);
        }
    };

} // namespace eml
