#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../../base/eml_base.h"
#include "if_base.h"
#include "if_config.h"
#include "if_feature_extractor.h"
#include "if_scaler_layer.h"
#include "if_feature_transform_layer.h"
#include "../../base/eml_status.h"

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

        bool set_node_layouts(uint8_t threshold_bits,
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
        packed_vector<64, IsoNode> nodes_;
        const If_node_resource* resource_ = nullptr;
        uint16_t depth_ = 0;
        bool is_loaded_ = false;

        bool ensure_layout() {
            if (!resource_ || !resource_->valid()) {
                return false;
            }
            if (nodes_.get_bits_per_value() != resource_->bits_per_node()) {
                nodes_.set_bits_per_value(resource_->bits_per_node());
            }
            return true;
        }

    public:
        void set_node_resources(const If_node_resource* resource, bool reset_storage = false) {
            resource_ = resource;
            if (reset_storage) {
                reset_node_storage();
            }
        }

        const If_node_resource* resource() const {
            return resource_;
        }

        void reset_node_storage(size_t reserve_count = 0u) {
            depth_ = 0u;
            is_loaded_ = false;
            if (!ensure_layout()) {
                nodes_.clear();
                return;
            }
            nodes_.clear();
            if (reserve_count > 0u) {
                nodes_.reserve(reserve_count);
            }
        }

        void reserve_nodes(size_t reserve_count) {
            if (reserve_count > 0u) {
                (void)ensure_layout();
                nodes_.reserve(reserve_count);
            }
        }

        uint32_t append_node(const IsoNode& node = IsoNode{}) {
            if (!ensure_layout()) {
                return 0u;
            }
            nodes_.push_back(node);
            return static_cast<uint32_t>(nodes_.size() - 1u);
        }

        bool set_split_node(uint32_t node_index,
                            uint16_t feature_id,
                            uint8_t threshold_slot,
                            uint32_t left_child_index) {
            if (!resource_ || node_index >= nodes_.size()) {
                return false;
            }
            IsoNode node;
            node.set_split(*resource_, feature_id, threshold_slot, left_child_index);
            nodes_.set(node_index, node);
            return true;
        }

        bool set_leaf_node(uint32_t node_index,
                           uint32_t leaf_size,
                           uint16_t leaf_depth) {
            if (!resource_ || node_index >= nodes_.size()) {
                return false;
            }
            IsoNode node;
            node.set_leaf(*resource_, leaf_size, leaf_depth);
            nodes_.set(node_index, node);
            return true;
        }

        bool finalize(uint16_t depth) {
            depth_ = depth;
            is_loaded_ = !nodes_.empty();
            return is_loaded_;
        }

        float path_length(const uint8_t* quantized_features, uint16_t num_features) const {
            if (!is_loaded_ || !resource_ || !quantized_features || num_features == 0u || nodes_.empty()) {
                return 0.0f;
            }

            uint32_t node_index = 0;
            while (node_index < nodes_.size()) {
                const IsoNode node = nodes_.get(node_index);
                if (node.is_leaf()) {
                    const uint32_t leaf_size = std::max<uint32_t>(1u, node.leaf_size(*resource_));
                    const uint16_t leaf_depth = node.leaf_depth(*resource_);
                    const double path = static_cast<double>(leaf_depth) + if_c_factor(leaf_size);
                    return static_cast<float>(path);
                }

                const uint16_t feature = node.feature_id(*resource_);
                if (feature >= num_features) {
                    return 0.0f;
                }

                const uint8_t threshold = node.threshold_slot(*resource_);
                const uint32_t left_child = node.left_child(*resource_);
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

        bool load_serialized(const If_node_resource* resource,
                             const uint64_t* packed_nodes,
                             size_t node_count,
                             uint16_t depth) {
            if (!resource || !resource->valid() || !packed_nodes || node_count == 0u) {
                return false;
            }

            resource_ = resource;
            nodes_.set_bits_per_value(resource_->bits_per_node());
            nodes_.resize(node_count, IsoNode{});
            for (size_t i = 0; i < node_count; ++i) {
                IsoNode node;
                node.packed_data = packed_nodes[i];
                nodes_.set(i, node);
            }

            depth_ = depth;
            is_loaded_ = true;
            return true;
        }

        uint64_t packed_node(size_t index) const {
            if (index >= nodes_.size()) {
                return 0ull;
            }
            return nodes_.get(index).packed_data;
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
        mutable eml_status_code last_status_code_ = eml_status_code::ok;

        inline void set_status(eml_status_code status) const {
            last_status_code_ = status;
        }

        static constexpr char k_model_magic_[4] = {'I', 'F', 'M', 'Q'};
        static constexpr uint16_t k_model_version_ = 1u;
        static constexpr uint8_t k_endian_little_flag_ = 1u;
        static constexpr uint16_t k_model_header_size_ = 32u;
        static constexpr uint16_t k_model_header_min_size_ = 17u;
        static constexpr uint32_t k_checksum_seed_ = 2166136261u;
        static constexpr uint32_t k_checksum_prime_ = 16777619u;

        static bool write_exact(std::ostream& out, const void* src, size_t bytes) {
            out.write(reinterpret_cast<const char*>(src), static_cast<std::streamsize>(bytes));
            return static_cast<bool>(out);
        }

        static bool read_exact(std::istream& in, void* dst, size_t bytes) {
            in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
            return static_cast<size_t>(in.gcount()) == bytes;
        }

        static uint32_t checksum_update(uint32_t checksum, const uint8_t* data, size_t size) {
            uint32_t current = checksum;
            for (size_t i = 0; i < size; ++i) {
                current ^= static_cast<uint32_t>(data[i]);
                current *= k_checksum_prime_;
            }
            return current;
        }

        static bool write_u8(std::ostream& out, uint8_t value) {
            return write_exact(out, &value, sizeof(value));
        }

        static bool write_u16_le(std::ostream& out, uint16_t value) {
            const uint8_t bytes[2] = {
                static_cast<uint8_t>(value & 0xFFu),
                static_cast<uint8_t>((value >> 8u) & 0xFFu)
            };
            return write_exact(out, bytes, sizeof(bytes));
        }

        static bool write_u32_le(std::ostream& out, uint32_t value) {
            const uint8_t bytes[4] = {
                static_cast<uint8_t>(value & 0xFFu),
                static_cast<uint8_t>((value >> 8u) & 0xFFu),
                static_cast<uint8_t>((value >> 16u) & 0xFFu),
                static_cast<uint8_t>((value >> 24u) & 0xFFu)
            };
            return write_exact(out, bytes, sizeof(bytes));
        }

        static bool write_u64_le(std::ostream& out, uint64_t value) {
            const uint8_t bytes[8] = {
                static_cast<uint8_t>(value & 0xFFull),
                static_cast<uint8_t>((value >> 8u) & 0xFFull),
                static_cast<uint8_t>((value >> 16u) & 0xFFull),
                static_cast<uint8_t>((value >> 24u) & 0xFFull),
                static_cast<uint8_t>((value >> 32u) & 0xFFull),
                static_cast<uint8_t>((value >> 40u) & 0xFFull),
                static_cast<uint8_t>((value >> 48u) & 0xFFull),
                static_cast<uint8_t>((value >> 56u) & 0xFFull)
            };
            return write_exact(out, bytes, sizeof(bytes));
        }

        static bool write_f32_le(std::ostream& out, float value) {
            uint32_t bits = 0u;
            std::memcpy(&bits, &value, sizeof(bits));
            return write_u32_le(out, bits);
        }

        static bool read_u8(std::istream& in, uint8_t& value) {
            return read_exact(in, &value, sizeof(value));
        }

        static bool read_u16_le(std::istream& in, uint16_t& value) {
            uint8_t bytes[2] = {0u, 0u};
            if (!read_exact(in, bytes, sizeof(bytes))) {
                return false;
            }
            value = static_cast<uint16_t>(
                static_cast<uint16_t>(bytes[0]) |
                (static_cast<uint16_t>(bytes[1]) << 8u)
            );
            return true;
        }

        static bool read_u32_le(std::istream& in, uint32_t& value) {
            uint8_t bytes[4] = {0u, 0u, 0u, 0u};
            if (!read_exact(in, bytes, sizeof(bytes))) {
                return false;
            }
            value = static_cast<uint32_t>(bytes[0]) |
                    (static_cast<uint32_t>(bytes[1]) << 8u) |
                    (static_cast<uint32_t>(bytes[2]) << 16u) |
                    (static_cast<uint32_t>(bytes[3]) << 24u);
            return true;
        }

        static bool read_u64_le(std::istream& in, uint64_t& value) {
            uint8_t bytes[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
            if (!read_exact(in, bytes, sizeof(bytes))) {
                return false;
            }
            value = static_cast<uint64_t>(bytes[0]) |
                    (static_cast<uint64_t>(bytes[1]) << 8u) |
                    (static_cast<uint64_t>(bytes[2]) << 16u) |
                    (static_cast<uint64_t>(bytes[3]) << 24u) |
                    (static_cast<uint64_t>(bytes[4]) << 32u) |
                    (static_cast<uint64_t>(bytes[5]) << 40u) |
                    (static_cast<uint64_t>(bytes[6]) << 48u) |
                    (static_cast<uint64_t>(bytes[7]) << 56u);
            return true;
        }

        static bool read_f32_le(std::istream& in, float& value) {
            uint32_t bits = 0u;
            if (!read_u32_le(in, bits)) {
                return false;
            }
            std::memcpy(&value, &bits, sizeof(value));
            return true;
        }

        static bool write_u8_with_checksum(std::ostream& out, uint8_t value, uint32_t& checksum) {
            if (!write_u8(out, value)) {
                return false;
            }
            checksum = checksum_update(checksum, &value, sizeof(value));
            return true;
        }

        static bool write_u16_le_with_checksum(std::ostream& out, uint16_t value, uint32_t& checksum) {
            const uint8_t bytes[2] = {
                static_cast<uint8_t>(value & 0xFFu),
                static_cast<uint8_t>((value >> 8u) & 0xFFu)
            };
            if (!write_exact(out, bytes, sizeof(bytes))) {
                return false;
            }
            checksum = checksum_update(checksum, bytes, sizeof(bytes));
            return true;
        }

        static bool write_u32_le_with_checksum(std::ostream& out, uint32_t value, uint32_t& checksum) {
            const uint8_t bytes[4] = {
                static_cast<uint8_t>(value & 0xFFu),
                static_cast<uint8_t>((value >> 8u) & 0xFFu),
                static_cast<uint8_t>((value >> 16u) & 0xFFu),
                static_cast<uint8_t>((value >> 24u) & 0xFFu)
            };
            if (!write_exact(out, bytes, sizeof(bytes))) {
                return false;
            }
            checksum = checksum_update(checksum, bytes, sizeof(bytes));
            return true;
        }

        static bool write_u64_le_with_checksum(std::ostream& out, uint64_t value, uint32_t& checksum) {
            const uint8_t bytes[8] = {
                static_cast<uint8_t>(value & 0xFFull),
                static_cast<uint8_t>((value >> 8u) & 0xFFull),
                static_cast<uint8_t>((value >> 16u) & 0xFFull),
                static_cast<uint8_t>((value >> 24u) & 0xFFull),
                static_cast<uint8_t>((value >> 32u) & 0xFFull),
                static_cast<uint8_t>((value >> 40u) & 0xFFull),
                static_cast<uint8_t>((value >> 48u) & 0xFFull),
                static_cast<uint8_t>((value >> 56u) & 0xFFull)
            };
            if (!write_exact(out, bytes, sizeof(bytes))) {
                return false;
            }
            checksum = checksum_update(checksum, bytes, sizeof(bytes));
            return true;
        }

        static bool write_f32_le_with_checksum(std::ostream& out, float value, uint32_t& checksum) {
            uint32_t bits = 0u;
            std::memcpy(&bits, &value, sizeof(bits));
            return write_u32_le_with_checksum(out, bits, checksum);
        }

        static bool read_u8_with_checksum(std::istream& in, uint8_t& value, uint32_t& checksum) {
            if (!read_u8(in, value)) {
                return false;
            }
            checksum = checksum_update(checksum, &value, sizeof(value));
            return true;
        }

        static bool read_u16_le_with_checksum(std::istream& in, uint16_t& value, uint32_t& checksum) {
            uint8_t bytes[2] = {0u, 0u};
            if (!read_exact(in, bytes, sizeof(bytes))) {
                return false;
            }
            checksum = checksum_update(checksum, bytes, sizeof(bytes));
            value = static_cast<uint16_t>(
                static_cast<uint16_t>(bytes[0]) |
                (static_cast<uint16_t>(bytes[1]) << 8u)
            );
            return true;
        }

        static bool read_u32_le_with_checksum(std::istream& in, uint32_t& value, uint32_t& checksum) {
            uint8_t bytes[4] = {0u, 0u, 0u, 0u};
            if (!read_exact(in, bytes, sizeof(bytes))) {
                return false;
            }
            checksum = checksum_update(checksum, bytes, sizeof(bytes));
            value = static_cast<uint32_t>(bytes[0]) |
                    (static_cast<uint32_t>(bytes[1]) << 8u) |
                    (static_cast<uint32_t>(bytes[2]) << 16u) |
                    (static_cast<uint32_t>(bytes[3]) << 24u);
            return true;
        }

        static bool read_u64_le_with_checksum(std::istream& in, uint64_t& value, uint32_t& checksum) {
            uint8_t bytes[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
            if (!read_exact(in, bytes, sizeof(bytes))) {
                return false;
            }
            checksum = checksum_update(checksum, bytes, sizeof(bytes));
            value = static_cast<uint64_t>(bytes[0]) |
                    (static_cast<uint64_t>(bytes[1]) << 8u) |
                    (static_cast<uint64_t>(bytes[2]) << 16u) |
                    (static_cast<uint64_t>(bytes[3]) << 24u) |
                    (static_cast<uint64_t>(bytes[4]) << 32u) |
                    (static_cast<uint64_t>(bytes[5]) << 40u) |
                    (static_cast<uint64_t>(bytes[6]) << 48u) |
                    (static_cast<uint64_t>(bytes[7]) << 56u);
            return true;
        }

        static bool read_f32_le_with_checksum(std::istream& in, float& value, uint32_t& checksum) {
            uint32_t bits = 0u;
            if (!read_u32_le_with_checksum(in, bits, checksum)) {
                return false;
            }
            std::memcpy(&value, &bits, sizeof(value));
            return true;
        }

        bool compute_payload_size(uint64_t& payload_size) const {
            constexpr uint64_t k_payload_fixed_size =
                static_cast<uint64_t>(5u + sizeof(uint32_t) + sizeof(float) + sizeof(uint32_t));
            uint64_t total = k_payload_fixed_size;

            for (const If_tree& tree : trees_) {
                const uint64_t node_count = static_cast<uint64_t>(tree.node_count());
                constexpr uint64_t k_tree_meta_size = static_cast<uint64_t>(sizeof(uint16_t) + sizeof(uint32_t));
                const uint64_t tree_bytes = k_tree_meta_size + (node_count * sizeof(uint64_t));
                if (tree_bytes > (std::numeric_limits<uint64_t>::max() - total)) {
                    return false;
                }
                total += tree_bytes;
            }

            payload_size = total;
            return true;
        }

        bool load_model_binary_modern(std::ifstream& fin) {
            uint16_t version = 0u;
            uint8_t endian_flag = 0u;
            uint16_t header_size = 0u;
            uint64_t payload_size = 0ull;

            if (!read_u16_le(fin, version)) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }
            if (!read_u8(fin, endian_flag)) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }
            if (!read_u16_le(fin, header_size)) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }
            if (!read_u64_le(fin, payload_size)) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }

            if (version != k_model_version_) {
                set_status(eml_status_code::invalid_configuration);
                return false;
            }
            if (endian_flag != k_endian_little_flag_) {
                set_status(eml_status_code::invalid_configuration);
                return false;
            }
            if (header_size < k_model_header_min_size_) {
                set_status(eml_status_code::invalid_configuration);
                return false;
            }

            fin.seekg(0, std::ios::end);
            const std::streamoff file_size = fin.tellg();
            if (file_size <= 0) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }

            const uint64_t required_size = static_cast<uint64_t>(header_size) + payload_size + sizeof(uint32_t);
            if (required_size > static_cast<uint64_t>(file_size)) {
                set_status(eml_status_code::size_mismatch);
                return false;
            }

            fin.clear();
            fin.seekg(static_cast<std::streamoff>(header_size), std::ios::beg);
            if (!fin.good()) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }

            uint8_t threshold_bits = 0u;
            uint8_t feature_bits = 0u;
            uint8_t child_bits = 0u;
            uint8_t leaf_size_bits = 0u;
            uint8_t depth_bits = 0u;
            uint32_t checksum = k_checksum_seed_;

            if (!read_u8_with_checksum(fin, threshold_bits, checksum)) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }
            if (!read_u8_with_checksum(fin, feature_bits, checksum)) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }
            if (!read_u8_with_checksum(fin, child_bits, checksum)) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }
            if (!read_u8_with_checksum(fin, leaf_size_bits, checksum)) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }
            if (!read_u8_with_checksum(fin, depth_bits, checksum)) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }

            if (!resource_.set_node_layouts(threshold_bits, feature_bits, child_bits, leaf_size_bits, depth_bits)) {
                set_status(eml_status_code::invalid_configuration);
                return false;
            }

            if (!read_u32_le_with_checksum(fin, samples_per_tree_, checksum)) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }
            if (!read_f32_le_with_checksum(fin, threshold_offset_, checksum)) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }

            uint32_t tree_count = 0u;
            if (!read_u32_le_with_checksum(fin, tree_count, checksum)) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }
            if (tree_count == 0u) {
                set_status(eml_status_code::invalid_configuration);
                return false;
            }

            trees_.clear();
            trees_.reserve(tree_count);

            for (uint32_t i = 0u; i < tree_count; ++i) {
                uint16_t tree_depth = 0u;
                uint32_t node_count = 0u;
                if (!read_u16_le_with_checksum(fin, tree_depth, checksum)) {
                    set_status(eml_status_code::file_read_failed);
                    return false;
                }
                if (!read_u32_le_with_checksum(fin, node_count, checksum)) {
                    set_status(eml_status_code::file_read_failed);
                    return false;
                }
                if (node_count == 0u) {
                    set_status(eml_status_code::invalid_configuration);
                    return false;
                }

                std::vector<uint64_t> packed_nodes(node_count, 0ull);
                for (uint32_t node_index = 0u; node_index < node_count; ++node_index) {
                    if (!read_u64_le_with_checksum(fin, packed_nodes[node_index], checksum)) {
                        set_status(eml_status_code::file_read_failed);
                        return false;
                    }
                }

                If_tree tree;
                if (!tree.load_serialized(&resource_, packed_nodes.data(), packed_nodes.size(), tree_depth)) {
                    set_status(eml_status_code::invalid_configuration);
                    return false;
                }
                trees_.push_back(std::move(tree));
            }

            uint32_t expected_checksum = 0u;
            fin.clear();
            fin.seekg(static_cast<std::streamoff>(static_cast<uint64_t>(header_size) + payload_size), std::ios::beg);
            if (!read_u32_le(fin, expected_checksum)) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }
            if (expected_checksum != checksum) {
                set_status(eml_status_code::invalid_configuration);
                return false;
            }

            trained_ = !trees_.empty();
            set_status(trained_ ? eml_status_code::ok : eml_status_code::not_loaded);
            return trained_;
        }

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
            if (!resource_.set_node_layouts(threshold_bits, feature_bits, child_bits, leaf_size_bits, depth_bits)) {
                set_status(eml_status_code::invalid_configuration);
                return false;
            }
            set_status(eml_status_code::ok);
            return true;
        }

        void set_samples_per_tree(uint32_t samples_per_tree) {
            samples_per_tree_ = std::max<uint32_t>(1u, samples_per_tree);
        }

        void set_threshold_offset(float threshold_offset) {
            threshold_offset_ = threshold_offset;
        }

        void add_trained_tree(const If_tree& tree) {
            trees_.push_back(tree);
            trees_.back().set_node_resources(&resource_);
            trained_ = !trees_.empty();
        }

        void load_trained_forest(std::vector<If_tree>&& trees,
                                 uint32_t samples_per_tree,
                                 float threshold_offset) {
            trees_ = std::move(trees);
            for (If_tree& tree : trees_) {
                tree.set_node_resources(&resource_);
            }
            samples_per_tree_ = std::max<uint32_t>(1u, samples_per_tree);
            threshold_offset_ = threshold_offset;
            trained_ = !trees_.empty();
        }

        const If_node_resource& node_resource() const {
            return resource_;
        }

        const If_node_resource* node_resource_ptr() const {
            return &resource_;
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
            if (!trained_ || trees_.empty()) {
                set_status(eml_status_code::not_loaded);
                return false;
            }
            if (!resource_.valid()) {
                set_status(eml_status_code::invalid_configuration);
                return false;
            }

            uint64_t payload_size = 0ull;
            if (!compute_payload_size(payload_size)) {
                set_status(eml_status_code::size_mismatch);
                return false;
            }

            const std::filesystem::path parent = file_path.parent_path();
            if (!parent.empty()) {
                std::filesystem::create_directories(parent);
            }

            const std::filesystem::path temp_path = file_path.string() + ".tmp";
            std::ofstream fout(temp_path, std::ios::binary | std::ios::trunc);
            if (!fout.is_open()) {
                set_status(eml_status_code::file_open_failed);
                return false;
            }

            if (!write_exact(fout, k_model_magic_, sizeof(k_model_magic_))) {
                set_status(eml_status_code::file_write_failed);
                return false;
            }
            if (!write_u16_le(fout, k_model_version_)) {
                set_status(eml_status_code::file_write_failed);
                return false;
            }
            if (!write_u8(fout, k_endian_little_flag_)) {
                set_status(eml_status_code::file_write_failed);
                return false;
            }
            if (!write_u16_le(fout, k_model_header_size_)) {
                set_status(eml_status_code::file_write_failed);
                return false;
            }
            if (!write_u64_le(fout, payload_size)) {
                set_status(eml_status_code::file_write_failed);
                return false;
            }

            constexpr std::array<uint8_t, k_model_header_size_ - k_model_header_min_size_> k_header_padding = {};
            if (!write_exact(fout, k_header_padding.data(), k_header_padding.size())) {
                set_status(eml_status_code::file_write_failed);
                return false;
            }

            const uint8_t threshold_bits = resource_.threshold_bits();
            const uint8_t feature_bits = resource_.feature_bits();
            const uint8_t child_bits = resource_.child_bits();
            const uint8_t leaf_size_bits = resource_.leaf_size_bits();
            const uint8_t depth_bits = resource_.depth_bits();
            const uint32_t tree_count = static_cast<uint32_t>(trees_.size());
            uint32_t checksum = k_checksum_seed_;

            if (!write_u8_with_checksum(fout, threshold_bits, checksum)) {
                set_status(eml_status_code::file_write_failed);
                return false;
            }
            if (!write_u8_with_checksum(fout, feature_bits, checksum)) {
                set_status(eml_status_code::file_write_failed);
                return false;
            }
            if (!write_u8_with_checksum(fout, child_bits, checksum)) {
                set_status(eml_status_code::file_write_failed);
                return false;
            }
            if (!write_u8_with_checksum(fout, leaf_size_bits, checksum)) {
                set_status(eml_status_code::file_write_failed);
                return false;
            }
            if (!write_u8_with_checksum(fout, depth_bits, checksum)) {
                set_status(eml_status_code::file_write_failed);
                return false;
            }

            if (!write_u32_le_with_checksum(fout, samples_per_tree_, checksum)) {
                set_status(eml_status_code::file_write_failed);
                return false;
            }
            if (!write_f32_le_with_checksum(fout, threshold_offset_, checksum)) {
                set_status(eml_status_code::file_write_failed);
                return false;
            }
            if (!write_u32_le_with_checksum(fout, tree_count, checksum)) {
                set_status(eml_status_code::file_write_failed);
                return false;
            }

            for (const If_tree& tree : trees_) {
                const uint16_t tree_depth = tree.depth();
                const uint32_t node_count = static_cast<uint32_t>(tree.node_count());

                if (!write_u16_le_with_checksum(fout, tree_depth, checksum)) {
                    set_status(eml_status_code::file_write_failed);
                    return false;
                }
                if (!write_u32_le_with_checksum(fout, node_count, checksum)) {
                    set_status(eml_status_code::file_write_failed);
                    return false;
                }
                for (uint32_t node_index = 0u; node_index < node_count; ++node_index) {
                    const uint64_t packed_data = tree.packed_node(node_index);
                    if (!write_u64_le_with_checksum(fout, packed_data, checksum)) {
                        set_status(eml_status_code::file_write_failed);
                        return false;
                    }
                }
            }

            if (!write_u32_le(fout, checksum)) {
                set_status(eml_status_code::file_write_failed);
                return false;
            }

            fout.flush();
            if (!fout.good()) {
                set_status(eml_status_code::file_write_failed);
                return false;
            }
            fout.close();

            std::error_code ec;
            std::filesystem::remove(file_path, ec);
            ec.clear();
            std::filesystem::rename(temp_path, file_path, ec);
            if (ec) {
                std::filesystem::remove(temp_path, ec);
                set_status(eml_status_code::file_write_failed);
                return false;
            }

            set_status(eml_status_code::ok);
            return true;
        }

        bool load_model_binary(const std::filesystem::path& file_path) {
            std::ifstream fin(file_path, std::ios::binary);
            if (!fin.is_open()) {
                set_status(eml_status_code::file_open_failed);
                return false;
            }

            char magic[4] = {0, 0, 0, 0};
            if (!read_exact(fin, magic, sizeof(magic))) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }

            fin.clear();
            fin.seekg(0, std::ios::beg);

            if (std::memcmp(magic, k_model_magic_, sizeof(k_model_magic_)) != 0) {
                set_status(eml_status_code::invalid_configuration);
                return false;
            }

            if (!read_exact(fin, magic, sizeof(magic))) {
                set_status(eml_status_code::file_read_failed);
                return false;
            }
            const bool loaded = load_model_binary_modern(fin);
            if (!loaded) {
                return false;
            }
            set_status(eml_status_code::ok);
            return true;
        }

        size_t num_trees() const { return trees_.size(); }
        uint32_t samples_per_tree() const { return samples_per_tree_; }
        bool trained() const { return trained_; }
        float threshold_offset() const { return threshold_offset_; }
        eml_status_code last_status() const { return last_status_code_; }
        void clear_status() { set_status(eml_status_code::ok); }
    };



} // namespace eml
