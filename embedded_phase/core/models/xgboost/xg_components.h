#pragma once

/**
 * @file xg_components.h
 * @brief XGBoost components for MCU inference/training
 *
 * This header defines the core XGBoost MCU-side types:
 * - Build-time node structure (float weights)
 * - Inference node structure (quantized weights in packed_vector)
 * - Node packing resource (dynamic bit layout)
 * - Packed bit stream storage with variable bits_per_node
 * - Tree, forest container, config loader
 * 
 * MCU Compact Format v2:
 * - Magic: 0x58474D43 ("XGMC")
 * - Packed bit stream with variable bits_per_node
 * - Tree offsets array for direct tree access
 * 
 * Two-Phase Training:
 * - Phase 1 (Calibration): Build trees to collect min/max leaf weights
 * - Phase 2 (Streaming): Build trees and quantize/save per-round
 * 
 * Note: XG_base class is defined in xgb_base.h
 */

#include "xgb_base.h"
#include "../../containers/STL_MCU.h"
#include "../../base/eml_base.h"
// Shared storage and board configuration (RF_* prefixed utilities are shared across models).
#include "../../../Rf_file_manager.h"
#include "../../../Rf_board_config.h"

#if defined(ESP_PLATFORM)
    #include "esp_system.h"
    #if RF_BOARD_SUPPORTS_PSRAM
    #include <esp_psram.h>
    #endif
#endif

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>

namespace mcu {

    using xg_label_type  = uint16_t;
    using xg_sample_type = uint32_t;
    using xg_node_type   = uint32_t;

    // Type alias for sample ID storage (uses ID_vector for memory efficiency)
    using XG_sample_id_set = ID_vector<xg_sample_type, 1>;

    // Error label constant
    static constexpr xg_label_type XG_ERROR_LABEL  = static_cast<xg_label_type>(~static_cast<xg_label_type>(0));

    // ============================================================================
    // RAM Tracking Helper for training
    // ============================================================================
    struct XG_RAM_Tracker {
        size_t current_usage = 0;
        size_t peak_usage = 0;
        
        void add(size_t bytes) {
            current_usage += bytes;
            if (current_usage > peak_usage) {
                peak_usage = current_usage;
            }
        }
        
        void release(size_t bytes) {
            if (bytes <= current_usage) {
                current_usage -= bytes;
            } else {
                current_usage = 0;
            }
        }
        
        void reset() {
            current_usage = 0;
            peak_usage = 0;
        }
        
        size_t getPeak() const { return peak_usage; }
        size_t getCurrent() const { return current_usage; }
    };

    // ============================================================================
    // XG_Building_node - Node structure for on-device training
    // Uses float weight during training; quantization happens when saving
    // This is similar to how the PC trainer works
    // ============================================================================

    struct XG_Building_node {
        bool is_leaf = true;
        uint16_t feature_id = 0;
        uint16_t threshold = 0;
        uint32_t left_child_idx = 0;
        float weight = 0.0f;  // Float weight during training (quantized when saving)

        XG_Building_node() = default;

        static XG_Building_node makeSplit(uint16_t fid, uint16_t thresh, uint32_t left_child) {
            XG_Building_node n;
            n.is_leaf = false;
            n.feature_id = fid;
            n.threshold = thresh;
            n.left_child_idx = left_child;
            n.weight = 0.0f;
            return n;
        }

        static XG_Building_node makeLeaf(float w) {
            XG_Building_node n;
            n.is_leaf = true;
            n.weight = w;
            n.feature_id = 0;
            n.threshold = 0;
            n.left_child_idx = 0;
            return n;
        }
    };

    // Alias for backward compatibility
    using XG_node = XG_Building_node;

    // ============================================================================
    // XG_node_resource - Bit layout configuration for inference packed stream
    // Defines how nodes are packed in the binary model file format
    // ============================================================================

    class XG_node_resource {
    public:
        uint8_t bits_per_node = 17;     // Total bits per node in packed stream
        uint8_t scale_factor_bits = 2;  // Fractional bits for fixed-point weight
        uint8_t label_bits = 2;         // Bits to store signed weight (two's complement)
        uint8_t feature_bits = 8;
        uint8_t threshold_bits = 8;
        uint8_t child_bits = 15;

        bool init(uint8_t bits_per_node_, uint8_t scale_factor_bits_, uint8_t label_bits_, 
                  uint8_t feature_bits_, uint8_t threshold_bits_, uint8_t child_bits_) {
            bits_per_node = bits_per_node_;
            scale_factor_bits = scale_factor_bits_;
            label_bits = label_bits_;
            feature_bits = feature_bits_;
            threshold_bits = threshold_bits_;
            child_bits = child_bits_;

            if (bits_per_node == 0 || bits_per_node > 32) return false;
            if (scale_factor_bits > 30) return false;
            if (label_bits == 0 || label_bits > 31) return false;
            if (feature_bits == 0 || threshold_bits == 0 || child_bits == 0) return false;
            if (feature_bits > 31 || threshold_bits > 31 || child_bits > 31) return false;
            
            // Verify bits_per_node is sufficient for both layouts
            uint8_t split_bits = 1 + feature_bits + threshold_bits + child_bits;
            uint8_t leaf_bits = 1 + label_bits;
            uint8_t min_bits = (split_bits > leaf_bits) ? split_bits : leaf_bits;
            if (bits_per_node < min_bits) return false;
            
            return true;
        }

        // Convenience init (auto-computes bits_per_node)
        bool init(uint8_t scale_factor_bits_, uint8_t label_bits_, uint8_t feature_bits_, 
                  uint8_t threshold_bits_, uint8_t child_bits_) {
            uint8_t split_bits = 1 + feature_bits_ + threshold_bits_ + child_bits_;
            uint8_t leaf_bits = 1 + label_bits_;
            uint8_t computed_bits_per_node = (split_bits > leaf_bits) ? split_bits : leaf_bits;
            return init(computed_bits_per_node, scale_factor_bits_, label_bits_, feature_bits_, 
                        threshold_bits_, child_bits_);
        }

        // Extract node from packed_vector at given node index
        template<uint8_t bits>
        [[gnu::always_inline]] inline uint32_t extractNode(const packed_vector<bits>& packed_data, size_t node_idx) const {
            return static_cast<uint32_t>(packed_data.get(node_idx));
        }

        [[gnu::always_inline]] inline bool isLeaf(uint32_t data) const {
            // MSB (bit bits_per_node-1) is leaf flag
            return ((data >> (bits_per_node - 1)) & 1) != 0;
        }

        [[gnu::always_inline]] inline uint16_t getFeatureID(uint32_t data) const {
            // Split layout: [is_leaf=0 | feature_id | threshold | left_child]
            // feature_id starts after is_leaf bit
            int32_t shift = bits_per_node - 1 - feature_bits;
            const uint32_t mask = (1UL << feature_bits) - 1UL;
            return static_cast<uint16_t>((data >> shift) & mask);
        }

        [[gnu::always_inline]] inline uint16_t getThreshold(uint32_t data) const {
            int32_t shift = bits_per_node - 1 - feature_bits - threshold_bits;
            const uint32_t mask = (1UL << threshold_bits) - 1UL;
            return static_cast<uint16_t>((data >> shift) & mask);
        }

        [[gnu::always_inline]] inline uint32_t getLeftChildIndex(uint32_t data) const {
            int32_t shift = bits_per_node - 1 - feature_bits - threshold_bits - child_bits;
            const uint32_t mask = (1UL << child_bits) - 1UL;
            return static_cast<uint32_t>((data >> shift) & mask);
        }

        [[gnu::always_inline]] inline uint32_t getRightChildIndex(uint32_t data) const {
            return getLeftChildIndex(data) + 1u;
        }

        [[gnu::always_inline]] inline int32_t getLeafWeightQuantized(uint32_t data) const {
            // Leaf layout: [is_leaf=1 | quantized_weight | padding]
            // Weight is stored right after is_leaf bit
            int32_t shift = bits_per_node - 1 - label_bits;
            const uint32_t mask = (1UL << label_bits) - 1UL;
            uint32_t raw = (data >> shift) & mask;
            
            // Sign extend from label_bits
            const uint32_t sign_bit = 1UL << (label_bits - 1);
            if (raw & sign_bit) {
                raw |= ~mask;
            }
            return static_cast<int32_t>(raw);
        }

        [[gnu::always_inline]] inline float getLeafWeight(uint32_t data) const {
            const int32_t signed_w = getLeafWeightQuantized(data);
            return quantized_to_float<int32_t>(signed_w, scale_factor_bits);
        }
        
        // Pack functions for on-device training (create packed node value)
        [[gnu::always_inline]] inline uint32_t packSplit(uint16_t feature_id, uint16_t threshold, uint32_t left_child_idx) const {
            // Split layout: [is_leaf=0 | feature_id | threshold | left_child]
            uint32_t result = 0;  // is_leaf = 0
            
            int32_t shift = bits_per_node - 1 - feature_bits;
            result |= (static_cast<uint32_t>(feature_id) & ((1UL << feature_bits) - 1)) << shift;
            
            shift -= threshold_bits;
            result |= (static_cast<uint32_t>(threshold) & ((1UL << threshold_bits) - 1)) << shift;
            
            shift -= child_bits;
            result |= (static_cast<uint32_t>(left_child_idx) & ((1UL << child_bits) - 1)) << shift;
            
            return result;
        }

        [[gnu::always_inline]] inline uint32_t packLeaf(float weight) const {
            // Leaf layout: [is_leaf=1 | quantized_weight | padding]
            uint32_t result = 1UL << (bits_per_node - 1);  // is_leaf = 1
            
            int32_t scaled = float_to_quantized<int32_t>(weight, scale_factor_bits);
            
            const int32_t min_w = -(1L << (label_bits - 1));
            const int32_t max_w = (1L << (label_bits - 1)) - 1L;
            if (scaled < min_w) scaled = min_w;
            if (scaled > max_w) scaled = max_w;
            
            int32_t shift = bits_per_node - 1 - label_bits;
            const uint32_t mask = (1UL << label_bits) - 1UL;
            result |= (static_cast<uint32_t>(scaled) & mask) << shift;
            
            return result;
        }

        // Pack leaf from already-quantized weight (used when converting from build_nodes)
        [[gnu::always_inline]] inline uint32_t packLeafQ(int32_t quantized_weight) const {
            // Leaf layout: [is_leaf=1 | quantized_weight | padding]
            uint32_t result = 1UL << (bits_per_node - 1);  // is_leaf = 1
            
            int32_t shift = bits_per_node - 1 - label_bits;
            const uint32_t mask = (1UL << label_bits) - 1UL;
            result |= (static_cast<uint32_t>(quantized_weight) & mask) << shift;
            
            return result;
        }
    };

    // ============================================================================
    // XG_tree - Lightweight tree representation for XGBoost
    // 
    // Since XGBoost has many small trees, nodes are stored centrally in 
    // XG_tree_container using packed_vector. XG_tree acts as an intermediary
    // holding only metadata and offsets into the central storage.
    // 
    // Three modes:
    // 1. Building mode: Uses local vector<XG_Building_node> with float weights
    //    for on-device training. After training, nodes are quantized when saving.
    // 2. Inference mode (packed local): Uses packed_vector<32> for quantized nodes
    //    when loaded from file or after quantization.
    // 3. Inference mode (container ref): References nodes in XG_tree_container's 
    //    packed storage. Uses node_offset + node_count to locate tree's nodes.
    // ============================================================================

    class XG_tree {
    public:
        // Build-time storage: float-based nodes for accurate training
        vector<XG_Building_node> build_nodes;
        
        // Inference-time storage: quantized nodes in packed format
        // Used when tree owns its packed data (loaded from file or after quantization)
        packed_vector<32> packed_nodes;

        // Inference-time metadata (when nodes are in container's central storage)
        uint32_t node_offset = 0;   // Offset into container's packed storage (in nodes)
        uint32_t node_count = 0;    // Number of nodes in this tree
        
        // Tree metadata
        uint8_t tree_index = 255;
        uint16_t depth = 0;
        bool is_loaded = false;
        bool in_build_mode = true;      // true = nodes in build_nodes
        bool owns_packed_data = false;  // true = packed_nodes owned locally; false = in container
        
        const XG_node_resource* resource = nullptr;

        XG_tree() = default;
        explicit XG_tree(uint8_t idx) : tree_index(idx) {}
        
        XG_tree(const XG_tree& other) = default;
        XG_tree& operator=(const XG_tree& other) = default;
        XG_tree(XG_tree&& other) noexcept = default;
        XG_tree& operator=(XG_tree&& other) noexcept = default;

        void setResource(const XG_node_resource* res) { 
            resource = res; 
        }

        // Add nodes directly during training (no quantization)
        void addNode(const XG_Building_node& node) {
            build_nodes.push_back(node);
            in_build_mode = true;
            owns_packed_data = false;
            is_loaded = true;
        }

        void addSplitNode(uint16_t feature_id, uint16_t threshold, uint32_t left_child_idx) {
            build_nodes.push_back(XG_Building_node::makeSplit(feature_id, threshold, left_child_idx));
            in_build_mode = true;
            owns_packed_data = false;
            is_loaded = true;
        }

        void addLeafNode(float weight) {
            build_nodes.push_back(XG_Building_node::makeLeaf(weight));
            in_build_mode = true;
            owns_packed_data = false;
            is_loaded = true;
        }

        uint32_t countNodes() const { 
            if (in_build_mode) {
                return static_cast<uint32_t>(build_nodes.size()); 
            }
            if (owns_packed_data) {
                return static_cast<uint32_t>(packed_nodes.size());
            }
            return node_count;
        }

        uint32_t countLeafNodes() const {
            if (in_build_mode) {
                uint32_t count = 0;
                for (const auto& node : build_nodes) {
                    if (node.is_leaf) count++;
                }
                return count;
            }
            if (owns_packed_data && resource) {
                uint32_t count = 0;
                for (size_t i = 0; i < packed_nodes.size(); ++i) {
                    uint32_t packed = static_cast<uint32_t>(packed_nodes.get(i));
                    if (resource->isLeaf(packed)) count++;
                }
                return count;
            }
            return 0;
        }

        uint16_t getTreeDepth() const {
            if (in_build_mode && build_nodes.empty()) return 0;
            return depth > 0 ? depth : getDepthRecursive(0);
        }

        size_t memoryUsage() const {
            size_t total = sizeof(*this);
            total += build_nodes.size() * sizeof(XG_Building_node);
            total += packed_nodes.memory_usage();
            return total;
        }

        /**
         * @brief Quantize float weights and convert to packed format
         * @param quant_bits Number of fractional bits for quantization
         * @param min_weight Optional: pre-computed min weight for scaling
         * @param max_weight Optional: pre-computed max weight for scaling
         * @return true if successful
         */
        bool quantizeAndPack(uint8_t quant_bits, float* out_min_weight = nullptr, float* out_max_weight = nullptr) {
            if (!in_build_mode || build_nodes.empty() || !resource) {
                return false;
            }

            // Find min/max weights for reporting
            float min_w = std::numeric_limits<float>::max();
            float max_w = std::numeric_limits<float>::lowest();
            for (const auto& node : build_nodes) {
                if (node.is_leaf) {
                    if (node.weight < min_w) min_w = node.weight;
                    if (node.weight > max_w) max_w = node.weight;
                }
            }
            if (out_min_weight) *out_min_weight = min_w;
            if (out_max_weight) *out_max_weight = max_w;

            // Pack nodes into local packed_vector
            packed_nodes.set_bits_per_value(resource->bits_per_node);
            packed_nodes.resize(build_nodes.size());

            for (size_t i = 0; i < build_nodes.size(); ++i) {
                const XG_Building_node& bn = build_nodes[i];
                uint32_t packed_val;
                
                if (bn.is_leaf) {
                    packed_val = resource->packLeaf(bn.weight);
                } else {
                    packed_val = resource->packSplit(bn.feature_id, bn.threshold, bn.left_child_idx);
                }
                packed_nodes.set(i, packed_val);
            }

            // Release build nodes and switch to packed mode
            build_nodes.clear();
            build_nodes.shrink_to_fit();
            
            in_build_mode = false;
            owns_packed_data = true;
            node_count = static_cast<uint32_t>(packed_nodes.size());
            is_loaded = true;
            
            return true;
        }

        /**
         * @brief Write packed tree to file stream
         * @param file Open file handle for writing
         * @return Number of bytes written, 0 on error
         */
        size_t writeToFile(File& file) const {
            if (!owns_packed_data || packed_nodes.empty() || !resource) {
                return 0;
            }
            
            // Write node count
            uint32_t count = static_cast<uint32_t>(packed_nodes.size());
            file.write(reinterpret_cast<const uint8_t*>(&count), sizeof(count));
            
            // Convert packed_vector to byte stream (MSB first) and write
            uint8_t bits_per_node = resource->bits_per_node;
            size_t total_bits = static_cast<size_t>(count) * bits_per_node;
            size_t packed_bytes = (total_bits + 7) / 8;
            
            vector<uint8_t> raw_data(packed_bytes, 0);
            for (uint32_t i = 0; i < count; ++i) {
                uint32_t node_val = static_cast<uint32_t>(packed_nodes.get(i));
                size_t bit_offset = static_cast<size_t>(i) * bits_per_node;
                for (int b = bits_per_node - 1; b >= 0; --b) {
                    uint8_t bit = (node_val >> b) & 1;
                    size_t bit_pos = bit_offset + (bits_per_node - 1 - b);
                    size_t byte_idx = bit_pos / 8;
                    size_t bit_in_byte = 7 - (bit_pos % 8);
                    raw_data[byte_idx] |= (bit << bit_in_byte);
                }
            }
            
            file.write(raw_data.data(), raw_data.size());
            return sizeof(count) + raw_data.size();
        }

        /**
         * @brief Read packed tree from file stream
         * @param file Open file handle for reading
         * @return true if successful
         */
        bool readFromFile(File& file) {
            if (!resource) return false;
            
            // Read node count
            uint32_t count = 0;
            if (file.read(reinterpret_cast<uint8_t*>(&count), sizeof(count)) != sizeof(count)) {
                return false;
            }
            
            uint8_t bits_per_node = resource->bits_per_node;
            size_t total_bits = static_cast<size_t>(count) * bits_per_node;
            size_t packed_bytes = (total_bits + 7) / 8;
            
            // Read raw byte stream
            vector<uint8_t> raw_data(packed_bytes);
            if (file.read(raw_data.data(), packed_bytes) != packed_bytes) {
                return false;
            }
            
            // Convert to packed_vector
            packed_nodes.set_bits_per_value(bits_per_node);
            packed_nodes.resize(count);
            
            for (uint32_t i = 0; i < count; ++i) {
                uint32_t node_val = 0;
                size_t bit_offset = static_cast<size_t>(i) * bits_per_node;
                for (int b = bits_per_node - 1; b >= 0; --b) {
                    size_t bit_pos = bit_offset + (bits_per_node - 1 - b);
                    size_t byte_idx = bit_pos / 8;
                    size_t bit_in_byte = 7 - (bit_pos % 8);
                    uint8_t bit = (raw_data[byte_idx] >> bit_in_byte) & 1;
                    node_val |= (static_cast<uint32_t>(bit) << b);
                }
                packed_nodes.set(i, node_val);
            }
            
            node_count = count;
            in_build_mode = false;
            owns_packed_data = true;
            is_loaded = true;
            
            return true;
        }

        // Prediction for build mode (nodes stored locally with float weights)
        [[gnu::hot]] float predictSample(const uint8_t* features, uint16_t num_features) const {
            if (in_build_mode) {
                return predictBuildMode(features, num_features);
            } else if (owns_packed_data) {
                return predictPackedMode(features, num_features);
            }
            return 0.0f;
        }

        template<uint8_t bits>
        [[gnu::hot]] float predictSample(const packed_vector<bits>& features) const {
            if (in_build_mode) {
                return predictBuildModePacked(features);
            } else if (owns_packed_data) {
                return predictPackedModePacked(features);
            }
            return 0.0f;
        }

        template<typename SampleType>
        [[gnu::hot]] float predictSample(const SampleType& sample) const {
            if (in_build_mode) {
                return predictBuildModeSample(sample);
            } else if (owns_packed_data) {
                return predictPackedModeSample(sample);
            }
            return 0.0f;
        }

        void clear() {
            build_nodes.clear();
            build_nodes.shrink_to_fit();
            packed_nodes.clear();
            node_offset = 0;
            node_count = 0;
            depth = 0;
            is_loaded = false;
            in_build_mode = true;
            owns_packed_data = false;
        }

        // Release build nodes and switch to inference mode (when nodes are moved to container)
        void releaseToInferenceMode(uint32_t offset, uint32_t count) {
            build_nodes.clear();
            build_nodes.shrink_to_fit();
            packed_nodes.clear();
            node_offset = offset;
            node_count = count;
            in_build_mode = false;
            owns_packed_data = false;
            is_loaded = true;
        }

    private:
        // ==================== Build Mode Predictions ====================
        [[gnu::hot]] float predictBuildMode(const uint8_t* features, uint16_t num_features) const {
            if (build_nodes.empty()) return 0.0f;

            uint32_t currentIndex = 0;
            while (currentIndex < build_nodes.size()) {
                const XG_Building_node& node = build_nodes[currentIndex];
                if (node.is_leaf) {
                    return node.weight;
                }

                if (node.feature_id >= num_features) return 0.0f;

                uint16_t feature_value = features[node.feature_id];
                if (feature_value <= node.threshold) {
                    currentIndex = node.left_child_idx;
                } else {
                    currentIndex = node.left_child_idx + 1;
                }

                if (currentIndex >= build_nodes.size()) return 0.0f;
            }
            return 0.0f;
        }

        template<uint8_t bits>
        [[gnu::hot]] float predictBuildModePacked(const packed_vector<bits>& features) const {
            if (build_nodes.empty()) return 0.0f;

            uint32_t currentIndex = 0;
            while (currentIndex < build_nodes.size()) {
                const XG_Building_node& node = build_nodes[currentIndex];
                if (node.is_leaf) {
                    return node.weight;
                }

                if (node.feature_id >= features.size()) return 0.0f;

                uint16_t feature_value = static_cast<uint16_t>(features.get(node.feature_id));
                if (feature_value <= node.threshold) {
                    currentIndex = node.left_child_idx;
                } else {
                    currentIndex = node.left_child_idx + 1;
                }

                if (currentIndex >= build_nodes.size()) return 0.0f;
            }
            return 0.0f;
        }

        template<typename SampleType>
        [[gnu::hot]] float predictBuildModeSample(const SampleType& sample) const {
            if (build_nodes.empty()) return 0.0f;

            uint32_t currentIndex = 0;
            while (currentIndex < build_nodes.size()) {
                const XG_Building_node& node = build_nodes[currentIndex];
                if (node.is_leaf) {
                    return node.weight;
                }

                uint16_t feature_value = static_cast<uint16_t>(sample.features.get(node.feature_id));
                if (feature_value <= node.threshold) {
                    currentIndex = node.left_child_idx;
                } else {
                    currentIndex = node.left_child_idx + 1;
                }

                if (currentIndex >= build_nodes.size()) break;
            }
            return 0.0f;
        }

        // ==================== Packed Mode Predictions ====================
        [[gnu::hot]] float predictPackedMode(const uint8_t* features, uint16_t num_features) const {
            if (packed_nodes.empty() || !resource) return 0.0f;

            uint32_t currentIndex = 0;
            while (currentIndex < packed_nodes.size()) {
                uint32_t packed = static_cast<uint32_t>(packed_nodes.get(currentIndex));
                
                if (resource->isLeaf(packed)) {
                    return resource->getLeafWeight(packed);
                }

                uint16_t feature_id = resource->getFeatureID(packed);
                if (feature_id >= num_features) return 0.0f;

                uint16_t threshold = resource->getThreshold(packed);
                uint16_t feature_value = features[feature_id];
                
                if (feature_value <= threshold) {
                    currentIndex = resource->getLeftChildIndex(packed);
                } else {
                    currentIndex = resource->getRightChildIndex(packed);
                }

                if (currentIndex >= packed_nodes.size()) return 0.0f;
            }
            return 0.0f;
        }

        template<uint8_t bits>
        [[gnu::hot]] float predictPackedModePacked(const packed_vector<bits>& features) const {
            if (packed_nodes.empty() || !resource) return 0.0f;

            uint32_t currentIndex = 0;
            while (currentIndex < packed_nodes.size()) {
                uint32_t packed = static_cast<uint32_t>(packed_nodes.get(currentIndex));
                
                if (resource->isLeaf(packed)) {
                    return resource->getLeafWeight(packed);
                }

                uint16_t feature_id = resource->getFeatureID(packed);
                if (feature_id >= features.size()) return 0.0f;

                uint16_t threshold = resource->getThreshold(packed);
                uint16_t feature_value = static_cast<uint16_t>(features.get(feature_id));
                
                if (feature_value <= threshold) {
                    currentIndex = resource->getLeftChildIndex(packed);
                } else {
                    currentIndex = resource->getRightChildIndex(packed);
                }

                if (currentIndex >= packed_nodes.size()) return 0.0f;
            }
            return 0.0f;
        }

        template<typename SampleType>
        [[gnu::hot]] float predictPackedModeSample(const SampleType& sample) const {
            if (packed_nodes.empty() || !resource) return 0.0f;

            uint32_t currentIndex = 0;
            while (currentIndex < packed_nodes.size()) {
                uint32_t packed = static_cast<uint32_t>(packed_nodes.get(currentIndex));
                
                if (resource->isLeaf(packed)) {
                    return resource->getLeafWeight(packed);
                }

                uint16_t feature_id = resource->getFeatureID(packed);
                uint16_t threshold = resource->getThreshold(packed);
                uint16_t feature_value = static_cast<uint16_t>(sample.features.get(feature_id));
                
                if (feature_value <= threshold) {
                    currentIndex = resource->getLeftChildIndex(packed);
                } else {
                    currentIndex = resource->getRightChildIndex(packed);
                }

                if (currentIndex >= packed_nodes.size()) break;
            }
            return 0.0f;
        }

        uint16_t getDepthRecursive(uint32_t nodeIndex) const {
            if (!in_build_mode || nodeIndex >= build_nodes.size()) return 0;
            
            const XG_Building_node& node = build_nodes[nodeIndex];
            if (node.is_leaf) return 1;

            uint32_t leftIndex = node.left_child_idx;
            uint32_t rightIndex = node.left_child_idx + 1;

            uint16_t leftDepth = getDepthRecursive(leftIndex);
            uint16_t rightDepth = getDepthRecursive(rightIndex);
            return 1 + ((leftDepth > rightDepth) ? leftDepth : rightDepth);
        }
    };

    // ============================================================================
    // Config
    // ============================================================================

    class XG_config {
    private:
        bool is_loaded = false;

    public:
        // Dataset metadata (from *_xgb_config.json)
        uint32_t random_seed = 42;
        // Dataset parameters
        uint16_t num_features = 0;
        xg_label_type num_labels = 2;
        xg_sample_type num_samples = 0;
        vector<uint32_t> samples_per_label;

        // Training summary (from *_xgb_config.json)
        float test_precision = -1.0f;

        // Feature quantization (bits per feature value)
        uint8_t quantization_coefficient = 2;

        // Node packing layout (must match *_xgb_config.json)
        uint8_t scale_factor_bits = 12;
        uint8_t label_bits = 12;
        uint8_t feature_bits = 8;
        uint8_t threshold_bits = 8;
        uint8_t child_bits = 15;

        // XGBoost parameters
        uint16_t num_boost_rounds = 100;
        float learning_rate = 0.3f;
        float lambda = 1.0f;
        float alpha = 0.0f;
        float gamma = 0.0f;
        float max_delta_step = 0.0f;

        // Tree parameters
        uint8_t max_depth = 6;
        uint16_t min_child_weight = 1;
        float subsample = 1.0f;
        float colsample_bytree = 1.0f;

        // Training metadata
        char objective[32] = "multi:softprob";
        char eval_metric[16] = "mlogloss";
        
        // Training ratios
        float train_ratio = 0.8f;
        float test_ratio = 0.2f;
        float valid_ratio = 0.0f;

        // Early stopping (training-time)
        bool early_stopping = false;
        uint16_t early_stopping_rounds = 10;
        float early_stopping_threshold = 0.001f;
        
        // MCU retraining parameters
        bool enable_retrain = true;
        bool enable_auto_config = false;
        xg_sample_type max_samples = 0; // 0 = unlimited

        // Inference parameters
        uint32_t total_trees = 0;
        float early_exit_score_limit = 0.0f; // 0 = disabled

        XG_config() { setDefaults(); }

        void setDefaults() {
            random_seed = 42;
            num_features = 0;
            num_labels = 2;
            num_samples = 0;
            samples_per_label.clear();
            test_precision = -1.0f;
            quantization_coefficient = 2;

            scale_factor_bits = 12;
            label_bits = 12;
            feature_bits = 8;
            threshold_bits = 8;
            child_bits = 15;

            num_boost_rounds = 100;
            learning_rate = 0.3f;
            lambda = 1.0f;
            alpha = 0.0f;
            gamma = 0.0f;
            max_delta_step = 0.0f;

            max_depth = 6;
            min_child_weight = 1;
            subsample = 1.0f;
            colsample_bytree = 1.0f;

            strncpy(objective, "multi:softprob", sizeof(objective));
            objective[sizeof(objective) - 1] = '\0';
            strncpy(eval_metric, "mlogloss", sizeof(eval_metric));
            eval_metric[sizeof(eval_metric) - 1] = '\0';

            train_ratio = 0.8f;
            test_ratio = 0.2f;
            valid_ratio = 0.0f;
            early_stopping = false;
            early_stopping_rounds = 10;
            early_stopping_threshold = 0.001f;
            enable_retrain = true;
            enable_auto_config = false;
            max_samples = 0;

            total_trees = 0;
            early_exit_score_limit = 0.0f;
            is_loaded = false;
        }

        bool loaded() const { return is_loaded; }

        bool loadConfig(const char* config_path) {
            if (!config_path) {
                eml_debug(0, "❌ Config path is null");
                return false;
            }

            File file = RF_FS_OPEN(config_path, RF_FILE_READ);
            if (!file) {
                eml_debug(0, "❌ Failed to open config: ", config_path);
                return false;
            }

            size_t file_size = file.size();
            if (file_size > 4096) {
                eml_debug(0, "❌ Config file too large");
                file.close();
                return false;
            }

            char* buffer = new char[file_size + 1];
            if (!buffer) {
                file.close();
                return false;
            }

            size_t bytes_read = file.read(reinterpret_cast<uint8_t*>(buffer), file_size);
            buffer[bytes_read] = '\0';
            file.close();

            auto extractInt = [&buffer](const char* key) -> int {
                const char* pos = strstr(buffer, key);
                if (!pos) return -1;
                pos = strchr(pos, ':');
                if (!pos) return -1;
                return atoi(pos + 1);
            };

            auto extractFloat = [&buffer](const char* key) -> float {
                const char* pos = strstr(buffer, key);
                if (!pos) return -1.0f;
                pos = strchr(pos, ':');
                if (!pos) return -1.0f;
                return static_cast<float>(atof(pos + 1));
            };

            auto extractString = [&buffer](const char* key, char* out, size_t out_len) -> bool {
                const char* pos = strstr(buffer, key);
                if (!pos) return false;
                pos = strchr(pos, ':');
                if (!pos) return false;
                pos = strchr(pos, '"');
                if (!pos) return false;
                pos++;
                const char* end = strchr(pos, '"');
                if (!end) return false;
                size_t len = static_cast<size_t>(end - pos);
                if (len >= out_len) len = out_len - 1;
                memcpy(out, pos, len);
                out[len] = '\0';
                return true;
            };

            int val;
            float fval;

            if ((val = extractInt("\"random_seed\"")) >= 0) random_seed = static_cast<uint32_t>(val);
            if ((val = extractInt("\"quantization_coefficient\"")) >= 0) quantization_coefficient = static_cast<uint8_t>(val);
            if ((val = extractInt("\"num_features\"")) >= 0) num_features = static_cast<uint16_t>(val);
            if ((val = extractInt("\"num_samples\"")) >= 0) num_samples = static_cast<xg_sample_type>(val);
            if ((val = extractInt("\"num_labels\"")) >= 0) num_labels = static_cast<xg_label_type>(val);
            if ((fval = extractFloat("\"test_precision\"")) >= 0.0f) test_precision = fval;

            if (num_labels > 0) {
                samples_per_label.clear();
                samples_per_label.resize(num_labels, 0);
                for (xg_label_type i = 0; i < num_labels; ++i) {
                    char key[32];
                    snprintf(key, sizeof(key), "\"samples_label_%u\"", static_cast<unsigned>(i));
                    int count = extractInt(key);
                    if (count >= 0) samples_per_label[i] = static_cast<uint32_t>(count);
                }
            }

            // dataset fields are optional (dp.csv may have them)
            if ((val = extractInt("\"scale_factor_bits\"")) >= 0) scale_factor_bits = static_cast<uint8_t>(val);
            if ((val = extractInt("\"label_bits\"")) >= 0) label_bits = static_cast<uint8_t>(val);
            if ((val = extractInt("\"feature_bits\"")) >= 0) feature_bits = static_cast<uint8_t>(val);
            if ((val = extractInt("\"threshold_bits\"")) >= 0) threshold_bits = static_cast<uint8_t>(val);
            if ((val = extractInt("\"child_bits\"")) >= 0) child_bits = static_cast<uint8_t>(val);

            if ((val = extractInt("\"num_boost_rounds\"")) >= 0) num_boost_rounds = static_cast<uint16_t>(val);
            if ((val = extractInt("\"max_depth\"")) >= 0) max_depth = static_cast<uint8_t>(val);
            if ((val = extractInt("\"min_child_weight\"")) >= 0) min_child_weight = static_cast<uint16_t>(val);

            if ((fval = extractFloat("\"learning_rate\"")) >= 0.0f) learning_rate = fval;
            if ((fval = extractFloat("\"lambda\"")) >= 0.0f) lambda = fval;
            if ((fval = extractFloat("\"alpha\"")) >= 0.0f) alpha = fval;
            if ((fval = extractFloat("\"gamma\"")) >= 0.0f) gamma = fval;
            if ((fval = extractFloat("\"max_delta_step\"")) >= 0.0f) max_delta_step = fval;
            if ((fval = extractFloat("\"subsample\"")) >= 0.0f) subsample = fval;
            if ((fval = extractFloat("\"colsample_bytree\"")) >= 0.0f) colsample_bytree = fval;

            (void)extractString("\"objective\"", objective, sizeof(objective));
            (void)extractString("\"eval_metric\"", eval_metric, sizeof(eval_metric));
            if ((fval = extractFloat("\"early_exit_score_limit\"")) >= 0.0f) early_exit_score_limit = fval;
            
            // Training ratios and MCU retraining parameters
            if ((fval = extractFloat("\"train_ratio\"")) >= 0.0f) train_ratio = fval;
            if ((fval = extractFloat("\"test_ratio\"")) >= 0.0f) test_ratio = fval;
            if ((fval = extractFloat("\"valid_ratio\"")) >= 0.0f) valid_ratio = fval;
            if ((fval = extractFloat("\"early_stopping_threshold\"")) >= 0.0f) early_stopping_threshold = fval;
            
            // Boolean parameters - extractBool helper
            auto extractBool = [&buffer](const char* key) -> int {
                const char* pos = strstr(buffer, key);
                if (!pos) return -1;
                pos = strchr(pos, ':');
                if (!pos) return -1;
                pos++;
                while (*pos == ' ' || *pos == '\t') pos++;
                if (strncmp(pos, "true", 4) == 0) return 1;
                if (strncmp(pos, "false", 5) == 0) return 0;
                return -1;
            };
            
            int bval;
            if ((bval = extractBool("\"enable_retrain\"")) >= 0) enable_retrain = (bval == 1);
            if ((bval = extractBool("\"enable_auto_config\"")) >= 0) enable_auto_config = (bval == 1);
            if ((bval = extractBool("\"early_stopping\"")) >= 0) early_stopping = (bval == 1);
            if ((val = extractInt("\"early_stopping_rounds\"")) >= 0) early_stopping_rounds = static_cast<uint16_t>(val);
            if ((val = extractInt("\"max_samples\"")) >= 0) max_samples = static_cast<xg_sample_type>(val);

            delete[] buffer;

            // sync quantization bits and threshold bits (common in quantized setups)
            if (threshold_bits > 0 && quantization_coefficient == 0) {
                quantization_coefficient = threshold_bits;
            }
            if (quantization_coefficient > 0 && threshold_bits == 0) {
                threshold_bits = quantization_coefficient;
            }

            // Ensure bit layout fits in 32-bit node (bit 31 reserved for leaf flag)
            uint16_t total_bits = (uint16_t)feature_bits + (uint16_t)threshold_bits + (uint16_t)child_bits;
            if (total_bits > 31) {
                uint16_t excess = total_bits - 31;
                if (child_bits > excess) {
                    child_bits -= excess;
                    eml_debug(0, "⚠️ Adjusted child_bits to ", (int)child_bits, " to fit 32-bit layout");
                } else {
                    eml_debug(0, "❌ Invalid bit layout: cannot adjust to fit 32-bit node");
                    return false;
                }
            }

            total_trees = static_cast<uint32_t>(num_boost_rounds) * static_cast<uint32_t>(num_labels);
            is_loaded = true;
            return true;
        }

        size_t memoryUsage() const {
            return sizeof(*this);
        }
    };

    // ============================================================================
    // Forest container - Central storage for XGBoost trees
    // 
    // Two storage modes:
    // 1. Inference mode (packed stream): Uses packed_vector<32> with bits_per_node
    //    set via set_bits_per_value() for memory-efficient storage.
    // 2. Training mode (build trees): Uses trees_by_class with XG_tree objects
    //    that have their own vector<XG_Building_node> for on-device training.
    // 
    // After training, build nodes can be converted to packed stream format.
    // ============================================================================

    class XG_tree_container {
    private:
        // Packed node storage (inference mode - loaded from file)
        // Uses packed_vector<32> with bits_per_node set via set_bits_per_value()
        packed_vector<32> packed_data;      // Packed nodes (variable bits per node)
        vector<uint32_t> tree_offsets;      // Node offset for each tree (in nodes)
        
        // On-device training support (training mode)
        vector<vector<XG_tree>> trees_by_class;  // XG_tree with packed_vector build_nodes
        bool using_packed_stream = false;        // True when loaded from file format
        
        XG_config* config_ptr = nullptr;
        XG_node_resource node_resource;

        uint8_t model_quant_bits = 0;
        uint32_t total_nodes = 0;
        uint32_t total_trees_count = 0;
        uint16_t num_classes = 0;
        uint16_t num_boost_rounds_count = 0;
        float model_learning_rate = 0.3f;
        uint32_t total_leaves = 0;
        uint16_t max_depth = 0;
        bool is_loaded = false;

    public:
        bool init(XG_config* config) {
            config_ptr = config;
            if (!config_ptr) return false;

            is_loaded = false;
            using_packed_stream = false;
            total_nodes = 0;
            total_leaves = 0;
            max_depth = 0;
            num_classes = config_ptr->num_labels;
            num_boost_rounds_count = config_ptr->num_boost_rounds;
            total_trees_count = static_cast<uint32_t>(num_classes) * num_boost_rounds_count;
            model_learning_rate = config_ptr->learning_rate;

            // Compute bits_per_node from config
            uint8_t split_bits = 1 + config_ptr->feature_bits + config_ptr->threshold_bits + config_ptr->child_bits;
            uint8_t leaf_bits = 1 + config_ptr->label_bits;
            uint8_t bits_per_node = (split_bits > leaf_bits) ? split_bits : leaf_bits;
            
            if (!node_resource.init(bits_per_node, config_ptr->scale_factor_bits, config_ptr->label_bits, 
                                    config_ptr->feature_bits, config_ptr->threshold_bits, config_ptr->child_bits)) {
                eml_debug(0, "❌ Invalid XG node layout bits");
                return false;
            }

            // Initialize legacy tree storage for on-device training
            trees_by_class.clear();
            trees_by_class.resize(config_ptr->num_labels);
            for (auto& v : trees_by_class) {
                v.clear();
                v.reserve(config_ptr->num_boost_rounds);
            }

            model_quant_bits = config_ptr->quantization_coefficient;
            return true;
        }

        void releaseForest() {
            packed_data.clear();
            tree_offsets.clear();
            for (auto& v : trees_by_class) {
                for (auto& t : v) t.clear();
                v.clear();
            }
            // Keep trees_by_class sized for training - just clear contents
            if (config_ptr && trees_by_class.size() != config_ptr->num_labels) {
                trees_by_class.resize(config_ptr->num_labels);
                for (auto& v : trees_by_class) {
                    v.reserve(config_ptr->num_boost_rounds);
                }
            }
            using_packed_stream = false;
            total_nodes = 0;
            total_leaves = 0;
            max_depth = 0;
            is_loaded = false;
        }

        bool loaded() const { return is_loaded; }
        bool isPackedStream() const { return using_packed_stream; }

        size_t numClasses() const { return num_classes; }
        size_t numTreesForClass(xg_label_type c) const { 
            if (using_packed_stream) {
                return num_boost_rounds_count;
            }
            return (c < trees_by_class.size()) ? trees_by_class[c].size() : 0; 
        }

        size_t numTreesTotal() const {
            if (using_packed_stream) {
                return total_trees_count;
            }
            size_t n = 0;
            for (const auto& v : trees_by_class) n += v.size();
            return n;
        }

        uint32_t getTotalNodes() const { return total_nodes; }
        uint32_t getTotalLeaves() const { return total_leaves; }
        uint16_t getMaxDepth() const { return max_depth; }
        uint8_t getQuantBits() const { return model_quant_bits; }
        float getLearningRate() const { return model_learning_rate; }
        const XG_node_resource& getNodeResource() const { return node_resource; }

        // Get tree for on-device training mode
        const XG_tree& getTree(xg_label_type class_idx, size_t tree_idx) const {
            return trees_by_class[class_idx][tree_idx];
        }

        XG_tree& getTree(xg_label_type class_idx, size_t tree_idx) {
            return trees_by_class[class_idx][tree_idx];
        }

        // Add tree for on-device training (tree stores nodes in its packed_vector build_nodes)
        void addTree(xg_label_type class_idx, XG_tree&& tree) {
            // Auto-resize if needed (safety check)
            if (class_idx >= trees_by_class.size()) {
                if (config_ptr && class_idx < config_ptr->num_labels) {
                    trees_by_class.resize(config_ptr->num_labels);
                } else {
                    eml_debug(0, "❌ addTree: invalid class_idx ", (int)class_idx);
                    return;
                }
            }
            
            tree.setResource(&node_resource);

            // Count nodes from the new XG_tree structure
            total_nodes += tree.countNodes();
            total_leaves += tree.countLeafNodes();
            uint16_t depth = tree.getTreeDepth();
            if (depth > max_depth) max_depth = depth;

            trees_by_class[class_idx].push_back(std::move(tree));
            using_packed_stream = false;  // Mark as training mode (not packed stream)
            is_loaded = true;  // Mark as having content
        }

        // ============================
        // Packed bit stream prediction
        // ============================
        
        // Predict raw score for a single tree using packed format
        // tree_idx: index into tree_offsets (round-major order)
        [[gnu::hot]] float predictTreePacked(uint32_t tree_idx, const uint8_t* features, uint16_t num_features) const {
            if (tree_idx >= total_trees_count) return 0.0f;
            
            uint32_t base_node = tree_offsets[tree_idx];
            
            // Determine tree size (nodes until next tree or end)
            uint32_t tree_end = (tree_idx + 1 < total_trees_count) ? tree_offsets[tree_idx + 1] : total_nodes;
            uint32_t tree_size = tree_end - base_node;
            if (tree_size == 0) return 0.0f;
            
            uint32_t node_idx = 0;  // Relative index within tree
            
            while (node_idx < tree_size) {
                uint32_t packed_node = node_resource.extractNode(packed_data, base_node + node_idx);
                
                if (node_resource.isLeaf(packed_node)) {
                    return node_resource.getLeafWeight(packed_node);
                }
                
                // Split node: extract feature, threshold, left_child
                uint16_t feature_id = node_resource.getFeatureID(packed_node);
                uint16_t threshold = node_resource.getThreshold(packed_node);
                uint32_t left_child = node_resource.getLeftChildIndex(packed_node);
                
                // Get feature value
                uint16_t feature_value = (feature_id < num_features) ? features[feature_id] : 0;
                
                // Navigate: if feature_value <= threshold, go left; else go right
                if (feature_value <= threshold) {
                    node_idx = left_child;
                } else {
                    node_idx = left_child + 1;
                }
            }
            
            return 0.0f;  // Shouldn't reach here if tree is valid
        }

        template<uint8_t bits>
        [[gnu::hot]] float predictTreePacked(uint32_t tree_idx, const packed_vector<bits>& features) const {
            if (tree_idx >= total_trees_count) return 0.0f;
            
            uint32_t base_node = tree_offsets[tree_idx];
            uint32_t tree_end = (tree_idx + 1 < total_trees_count) ? tree_offsets[tree_idx + 1] : total_nodes;
            uint32_t tree_size = tree_end - base_node;
            if (tree_size == 0) return 0.0f;
            
            uint32_t node_idx = 0;
            
            while (node_idx < tree_size) {
                uint32_t packed_node = node_resource.extractNode(packed_data, base_node + node_idx);
                
                if (node_resource.isLeaf(packed_node)) {
                    return node_resource.getLeafWeight(packed_node);
                }
                
                uint16_t feature_id = node_resource.getFeatureID(packed_node);
                uint16_t threshold = node_resource.getThreshold(packed_node);
                uint32_t left_child = node_resource.getLeftChildIndex(packed_node);
                
                uint16_t feature_value = (feature_id < features.size()) ? static_cast<uint16_t>(features.get(feature_id)) : 0;
                
                if (feature_value <= threshold) {
                    node_idx = left_child;
                } else {
                    node_idx = left_child + 1;
                }
            }
            
            return 0.0f;
        }

        [[gnu::hot]] int32_t predictTreePackedQ(uint32_t tree_idx, const uint8_t* features, uint16_t num_features) const {
            if (tree_idx >= total_trees_count) return 0;
            
            uint32_t base_node = tree_offsets[tree_idx];
            uint32_t tree_end = (tree_idx + 1 < total_trees_count) ? tree_offsets[tree_idx + 1] : total_nodes;
            uint32_t tree_size = tree_end - base_node;
            if (tree_size == 0) return 0;
            
            uint32_t node_idx = 0;
            
            while (node_idx < tree_size) {
                uint32_t packed_node = node_resource.extractNode(packed_data, base_node + node_idx);
                
                if (node_resource.isLeaf(packed_node)) {
                    return node_resource.getLeafWeightQuantized(packed_node);
                }
                
                uint16_t feature_id = node_resource.getFeatureID(packed_node);
                uint16_t threshold = node_resource.getThreshold(packed_node);
                uint32_t left_child = node_resource.getLeftChildIndex(packed_node);
                
                uint16_t feature_value = (feature_id < num_features) ? features[feature_id] : 0;
                
                if (feature_value <= threshold) {
                    node_idx = left_child;
                } else {
                    node_idx = left_child + 1;
                }
            }
            
            return 0;
        }

        // ============================
        // File I/O
        // ============================

        bool loadForest(const char* model_path) {
            if (!model_path || !config_ptr) return false;

            File file = RF_FS_OPEN(model_path, RF_FILE_READ);
            if (!file) {
                eml_debug(0, "❌ Failed to open model: ", model_path);
                return false;
            }

            // Read and verify header
            uint32_t magic = 0, version = 0;
            if (file.read(reinterpret_cast<uint8_t*>(&magic), sizeof(magic)) != sizeof(magic) ||
                file.read(reinterpret_cast<uint8_t*>(&version), sizeof(version)) != sizeof(version)) {
                file.close();
                return false;
            }

            eml_debug(1, "Model magic=0x", magic, " version=", version);

            if (magic != XG_MODEL_MAGIC) {
                eml_debug(0, "❌ Invalid magic number (expected 0x58474D43)");
                file.close();
                return false;
            }
            
            if (version != XG_MODEL_VERSION) {
                eml_debug(0, "❌ Unsupported model version (expected ", XG_MODEL_VERSION, ")");
                file.close();
                return false;
            }

            // Read MCU compact format v2 header
            // num_classes(2) + num_boost_rounds(2) + total_trees(4) + total_nodes(4) = 12 bytes
            if (file.read(reinterpret_cast<uint8_t*>(&num_classes), sizeof(num_classes)) != sizeof(num_classes) ||
                file.read(reinterpret_cast<uint8_t*>(&num_boost_rounds_count), sizeof(num_boost_rounds_count)) != sizeof(num_boost_rounds_count)) {
                file.close();
                return false;
            }
            
            if (file.read(reinterpret_cast<uint8_t*>(&total_trees_count), sizeof(total_trees_count)) != sizeof(total_trees_count) ||
                file.read(reinterpret_cast<uint8_t*>(&total_nodes), sizeof(total_nodes)) != sizeof(total_nodes)) {
                file.close();
                return false;
            }

            // Validate header consistency
            if (total_trees_count != static_cast<uint32_t>(num_classes) * num_boost_rounds_count) {
                eml_debug(0, "❌ Header inconsistency: total_trees != num_classes * num_boost_rounds");
                file.close();
                return false;
            }

            eml_debug(1, "Model: classes=", num_classes, " rounds=", num_boost_rounds_count, " trees=", total_trees_count, " nodes=", total_nodes);

            // Read layout parameters
            // bits_per_node(1) + label_bits(1) + feature_bits(1) + threshold_bits(1) + child_bits(1) + scale_factor_bits(1) + learning_rate(4) = 10 bytes
            uint8_t bits_per_node, label_bits_, feature_bits_, threshold_bits_, child_bits_, scale_factor_bits_;
            if (file.read(reinterpret_cast<uint8_t*>(&bits_per_node), sizeof(bits_per_node)) != sizeof(bits_per_node) ||
                file.read(reinterpret_cast<uint8_t*>(&label_bits_), sizeof(label_bits_)) != sizeof(label_bits_) ||
                file.read(reinterpret_cast<uint8_t*>(&feature_bits_), sizeof(feature_bits_)) != sizeof(feature_bits_) ||
                file.read(reinterpret_cast<uint8_t*>(&threshold_bits_), sizeof(threshold_bits_)) != sizeof(threshold_bits_) ||
                file.read(reinterpret_cast<uint8_t*>(&child_bits_), sizeof(child_bits_)) != sizeof(child_bits_) ||
                file.read(reinterpret_cast<uint8_t*>(&scale_factor_bits_), sizeof(scale_factor_bits_)) != sizeof(scale_factor_bits_)) {
                file.close();
                return false;
            }

            if (file.read(reinterpret_cast<uint8_t*>(&model_learning_rate), sizeof(model_learning_rate)) != sizeof(model_learning_rate)) {
                file.close();
                return false;
            }

            eml_debug(1, "Layout: bits_per_node=", (int)bits_per_node, " label=", (int)label_bits_, 
                      " feature=", (int)feature_bits_, " threshold=", (int)threshold_bits_, " child=", (int)child_bits_,
                      " scale_factor=", (int)scale_factor_bits_);
            eml_debug(1, "Learning rate from model: ", model_learning_rate);

            // Sanity check: label_bits should match scale_factor_bits for proper weight encoding
            if (label_bits_ != scale_factor_bits_) {
                eml_debug(0, "⚠️ Warning: label_bits (", (int)label_bits_, ") != scale_factor_bits (", (int)scale_factor_bits_, 
                          "). Model may have corrupted weights - consider retraining on PC.");
            }

            // Initialize node resource with loaded parameters
            if (!node_resource.init(bits_per_node, scale_factor_bits_, label_bits_, feature_bits_, threshold_bits_, child_bits_)) {
                eml_debug(0, "❌ Invalid node layout parameters");
                file.close();
                return false;
            }

            model_quant_bits = threshold_bits_;  // Quantization bits match threshold bits

            // Read tree offsets
            tree_offsets.resize(total_trees_count);
            size_t offsets_bytes = static_cast<size_t>(total_trees_count) * sizeof(uint32_t);
            if (file.read(reinterpret_cast<uint8_t*>(tree_offsets.data()), offsets_bytes) != offsets_bytes) {
                eml_debug(0, "❌ Failed to read tree offsets");
                file.close();
                tree_offsets.clear();
                return false;
            }

            // Read packed bit stream
            size_t total_bits = static_cast<size_t>(total_nodes) * bits_per_node;
            size_t packed_bytes = (total_bits + 7) / 8;
            
            // Read into temporary buffer first
            vector<uint8_t> raw_packed_data(packed_bytes);
            if (file.read(raw_packed_data.data(), packed_bytes) != packed_bytes) {
                eml_debug(0, "❌ Failed to read packed data");
                file.close();
                packed_data.clear();
                tree_offsets.clear();
                return false;
            }
            
            // Convert to packed_vector with bits_per_node as bits_per_value
            packed_data.set_bits_per_value(bits_per_node);
            packed_data.resize(total_nodes);
            
            // Unpack from byte stream (MSB first) to packed_vector
            for (uint32_t i = 0; i < total_nodes; ++i) {
                uint32_t node_val = 0;
                size_t bit_offset = static_cast<size_t>(i) * bits_per_node;
                for (int b = bits_per_node - 1; b >= 0; --b) {
                    size_t bit_pos = bit_offset + (bits_per_node - 1 - b);
                    size_t byte_idx = bit_pos / 8;
                    size_t bit_in_byte = 7 - (bit_pos % 8);
                    uint8_t bit = (raw_packed_data[byte_idx] >> bit_in_byte) & 1;
                    node_val |= (static_cast<uint32_t>(bit) << b);
                }
                packed_data.set(i, node_val);
            }

            file.close();

            // Update config with loaded values
            config_ptr->num_labels = num_classes;
            config_ptr->num_boost_rounds = num_boost_rounds_count;
            config_ptr->total_trees = total_trees_count;
            config_ptr->learning_rate = model_learning_rate;
            config_ptr->scale_factor_bits = scale_factor_bits_;
            config_ptr->label_bits = label_bits_;
            config_ptr->feature_bits = feature_bits_;
            config_ptr->threshold_bits = threshold_bits_;
            config_ptr->child_bits = child_bits_;

            // Resize trees_by_class to match loaded model (for on-device training compatibility)
            if (trees_by_class.size() != num_classes) {
                trees_by_class.clear();
                trees_by_class.resize(num_classes);
                for (auto& v : trees_by_class) {
                    v.clear();
                    v.reserve(num_boost_rounds_count);
                }
            }

            using_packed_stream = true;
            is_loaded = true;
            
            eml_debug(1, "✅ Loaded MCU compact model: ", packed_bytes, " bytes packed data + ", 
                      offsets_bytes, " bytes offsets = ", packed_bytes + offsets_bytes, " bytes total");
            
            return true;
        }

        bool saveForest(const char* model_path) {
            if (!model_path || !config_ptr) return false;

            // If using packed stream, save it directly
            if (using_packed_stream && !packed_data.empty()) {
                return savePackedFormat(model_path);
            }

            // Otherwise, convert from tree structure and save
            return saveFromTrees(model_path);
        }

    private:
        bool savePackedFormat(const char* model_path) {
            File file = RF_FS_OPEN(model_path, RF_FILE_WRITE);
            if (!file) {
                eml_debug(0, "❌ Failed to open for writing: ", model_path);
                return false;
            }

            // Write header
            uint32_t magic = XG_MODEL_MAGIC;
            uint32_t version = XG_MODEL_VERSION;
            
            file.write(reinterpret_cast<uint8_t*>(&magic), sizeof(magic));
            file.write(reinterpret_cast<uint8_t*>(&version), sizeof(version));
            file.write(reinterpret_cast<uint8_t*>(&num_classes), sizeof(num_classes));
            file.write(reinterpret_cast<uint8_t*>(&num_boost_rounds_count), sizeof(num_boost_rounds_count));
            file.write(reinterpret_cast<uint8_t*>(&total_trees_count), sizeof(total_trees_count));
            file.write(reinterpret_cast<uint8_t*>(&total_nodes), sizeof(total_nodes));

            // Write layout parameters
            uint8_t bits_per_node = node_resource.bits_per_node;
            uint8_t label_bits_ = node_resource.label_bits;
            uint8_t feature_bits_ = node_resource.feature_bits;
            uint8_t threshold_bits_ = node_resource.threshold_bits;
            uint8_t child_bits_ = node_resource.child_bits;
            uint8_t scale_factor_bits_ = node_resource.scale_factor_bits;

            file.write(reinterpret_cast<const uint8_t*>(&bits_per_node), sizeof(bits_per_node));
            file.write(reinterpret_cast<const uint8_t*>(&label_bits_), sizeof(label_bits_));
            file.write(reinterpret_cast<const uint8_t*>(&feature_bits_), sizeof(feature_bits_));
            file.write(reinterpret_cast<const uint8_t*>(&threshold_bits_), sizeof(threshold_bits_));
            file.write(reinterpret_cast<const uint8_t*>(&child_bits_), sizeof(child_bits_));
            file.write(reinterpret_cast<const uint8_t*>(&scale_factor_bits_), sizeof(scale_factor_bits_));
            file.write(reinterpret_cast<const uint8_t*>(&model_learning_rate), sizeof(model_learning_rate));

            // Write tree offsets
            file.write(reinterpret_cast<const uint8_t*>(tree_offsets.data()), tree_offsets.size() * sizeof(uint32_t));

            // Convert packed_vector to byte stream (MSB first) and write
            size_t total_bits = static_cast<size_t>(total_nodes) * bits_per_node;
            size_t packed_bytes = (total_bits + 7) / 8;
            vector<uint8_t> raw_packed_data(packed_bytes, 0);
            
            for (uint32_t i = 0; i < total_nodes; ++i) {
                uint32_t node_val = static_cast<uint32_t>(packed_data.get(i));
                size_t bit_offset = static_cast<size_t>(i) * bits_per_node;
                for (int b = bits_per_node - 1; b >= 0; --b) {
                    uint8_t bit = (node_val >> b) & 1;
                    size_t bit_pos = bit_offset + (bits_per_node - 1 - b);
                    size_t byte_idx = bit_pos / 8;
                    size_t bit_in_byte = 7 - (bit_pos % 8);
                    raw_packed_data[byte_idx] |= (bit << bit_in_byte);
                }
            }
            
            file.write(raw_packed_data.data(), raw_packed_data.size());

            file.close();
            return true;
        }

        bool saveFromTrees(const char* model_path) {
            // Convert tree structure (build_nodes with float weights) to packed format and save
            // Quantization happens HERE at save time (like PC version)
            if (trees_by_class.empty()) {
                eml_debug(0, "❌ No trees to save");
                return false;
            }

            // Check if any trees have content
            bool has_trees = false;
            for (const auto& v : trees_by_class) {
                for (const auto& t : v) {
                    if (t.in_build_mode && t.build_nodes.size() > 0) {
                        has_trees = true;
                        break;
                    }
                }
                if (has_trees) break;
            }
            if (!has_trees) {
                eml_debug(0, "❌ No trees to save (trees empty or not in build mode)");
                return false;
            }

            // Calculate totals
            uint16_t nc = static_cast<uint16_t>(trees_by_class.size());
            uint16_t nr = (nc > 0 && trees_by_class[0].size() > 0) 
                ? static_cast<uint16_t>(trees_by_class[0].size()) 
                : config_ptr->num_boost_rounds;
            uint32_t tt = static_cast<uint32_t>(nc) * nr;
            
            // Count total nodes and build tree offsets
            vector<uint32_t> new_offsets;
            new_offsets.reserve(tt);
            uint32_t node_offset = 0;
            uint32_t new_total_nodes = 0;
            
            // Round-major ordering: tree[t] = trees_by_class[t % nc][t / nc]
            for (uint32_t t = 0; t < tt; ++t) {
                uint16_t class_idx = static_cast<uint16_t>(t % nc);
                uint16_t round_idx = static_cast<uint16_t>(t / nc);
                
                if (class_idx < trees_by_class.size() && round_idx < trees_by_class[class_idx].size()) {
                    new_offsets.push_back(node_offset);
                    uint32_t tree_nodes = trees_by_class[class_idx][round_idx].countNodes();
                    node_offset += tree_nodes;
                    new_total_nodes += tree_nodes;
                }
            }

            // Pack nodes into bit stream using inference format (bits_per_node)
            // QUANTIZATION HAPPENS HERE when converting from float weights
            size_t total_bits = static_cast<size_t>(new_total_nodes) * node_resource.bits_per_node;
            size_t packed_bytes = (total_bits + 7) / 8;
            vector<uint8_t> new_packed_data(packed_bytes, 0);

            size_t bit_pos = 0;
            for (uint32_t t = 0; t < tt; ++t) {
                uint16_t class_idx = static_cast<uint16_t>(t % nc);
                uint16_t round_idx = static_cast<uint16_t>(t / nc);
                
                if (class_idx >= trees_by_class.size() || round_idx >= trees_by_class[class_idx].size()) continue;
                
                const XG_tree& tree = trees_by_class[class_idx][round_idx];
                if (!tree.in_build_mode) continue;
                
                // Convert each build_node (float) to inference format (quantized) and pack
                for (size_t i = 0; i < tree.build_nodes.size(); ++i) {
                    const XG_Building_node& bn = tree.build_nodes[i];
                    uint64_t node_val;
                    
                    if (bn.is_leaf) {
                        // Leaf: quantize float weight and pack
                        // [is_leaf=1 | quantized_weight | padding]
                        node_val = node_resource.packLeaf(bn.weight);  // This quantizes internally
                    } else {
                        // Split: [is_leaf=0 | feature_id | threshold | left_child]
                        node_val = node_resource.packSplit(
                            bn.feature_id,
                            bn.threshold,
                            bn.left_child_idx
                        );
                    }
                    
                    // Write bits_per_node bits, MSB first
                    for (int b = node_resource.bits_per_node - 1; b >= 0; --b) {
                        uint8_t bit = (node_val >> b) & 1;
                        size_t byte_idx = bit_pos / 8;
                        size_t bit_in_byte = 7 - (bit_pos % 8);  // MSB first in byte
                        new_packed_data[byte_idx] |= (bit << bit_in_byte);
                        bit_pos++;
                    }
                }
            }

            // Now save using the packed format
            File file = RF_FS_OPEN(model_path, RF_FILE_WRITE);
            if (!file) {
                eml_debug(0, "❌ Failed to open for writing: ", model_path);
                return false;
            }

            uint32_t magic = XG_MODEL_MAGIC;
            uint32_t version = XG_MODEL_VERSION;
            float lr = config_ptr->learning_rate;
            
            file.write(reinterpret_cast<uint8_t*>(&magic), sizeof(magic));
            file.write(reinterpret_cast<uint8_t*>(&version), sizeof(version));
            file.write(reinterpret_cast<uint8_t*>(&nc), sizeof(nc));
            file.write(reinterpret_cast<uint8_t*>(&nr), sizeof(nr));
            file.write(reinterpret_cast<uint8_t*>(&tt), sizeof(tt));
            file.write(reinterpret_cast<uint8_t*>(&new_total_nodes), sizeof(new_total_nodes));

            uint8_t bits_per_node = node_resource.bits_per_node;
            uint8_t label_bits_ = node_resource.label_bits;
            uint8_t feature_bits_ = node_resource.feature_bits;
            uint8_t threshold_bits_ = node_resource.threshold_bits;
            uint8_t child_bits_ = node_resource.child_bits;
            uint8_t scale_factor_bits_ = node_resource.scale_factor_bits;

            file.write(reinterpret_cast<const uint8_t*>(&bits_per_node), sizeof(bits_per_node));
            file.write(reinterpret_cast<const uint8_t*>(&label_bits_), sizeof(label_bits_));
            file.write(reinterpret_cast<const uint8_t*>(&feature_bits_), sizeof(feature_bits_));
            file.write(reinterpret_cast<const uint8_t*>(&threshold_bits_), sizeof(threshold_bits_));
            file.write(reinterpret_cast<const uint8_t*>(&child_bits_), sizeof(child_bits_));
            file.write(reinterpret_cast<const uint8_t*>(&scale_factor_bits_), sizeof(scale_factor_bits_));
            file.write(reinterpret_cast<const uint8_t*>(&lr), sizeof(lr));

            file.write(reinterpret_cast<const uint8_t*>(new_offsets.data()), new_offsets.size() * sizeof(uint32_t));
            file.write(new_packed_data.data(), new_packed_data.size());

            file.close();
            return true;
        }

    public:
        // ============================
        // Per-Round Streaming Support
        // ============================
        
        /**
         * @brief Begin streaming save - write header with placeholder values
         * @param model_path Path to model file
         * @return true if successful
         */
        bool beginStreamingSave(const char* model_path) {
            if (!model_path || !config_ptr) return false;
            
            // Store path for later
            streaming_model_path_[0] = '\0';
            strncat(streaming_model_path_, model_path, XG_PATH_BUFFER - 1);
            
            File file = RF_FS_OPEN(model_path, RF_FILE_WRITE);
            if (!file) {
                eml_debug(0, "❌ Failed to open for streaming write: ", model_path);
                return false;
            }
            
            // Write header with initial values (will be updated at end)
            uint32_t magic = XG_MODEL_MAGIC;
            uint32_t version = XG_MODEL_VERSION;
            
            file.write(reinterpret_cast<uint8_t*>(&magic), sizeof(magic));
            file.write(reinterpret_cast<uint8_t*>(&version), sizeof(version));
            
            // Placeholder values - will be updated at finalize
            uint16_t nc = config_ptr->num_labels;
            uint16_t nr = 0;  // Will be updated
            uint32_t tt = 0;  // Will be updated
            uint32_t tn = 0;  // Will be updated
            
            file.write(reinterpret_cast<uint8_t*>(&nc), sizeof(nc));
            file.write(reinterpret_cast<uint8_t*>(&nr), sizeof(nr));
            file.write(reinterpret_cast<uint8_t*>(&tt), sizeof(tt));
            file.write(reinterpret_cast<uint8_t*>(&tn), sizeof(tn));
            
            // Write layout parameters
            uint8_t bits_per_node = node_resource.bits_per_node;
            uint8_t label_bits_ = node_resource.label_bits;
            uint8_t feature_bits_ = node_resource.feature_bits;
            uint8_t threshold_bits_ = node_resource.threshold_bits;
            uint8_t child_bits_ = node_resource.child_bits;
            uint8_t scale_factor_bits_ = node_resource.scale_factor_bits;
            float lr = config_ptr->learning_rate;
            
            file.write(reinterpret_cast<const uint8_t*>(&bits_per_node), sizeof(bits_per_node));
            file.write(reinterpret_cast<const uint8_t*>(&label_bits_), sizeof(label_bits_));
            file.write(reinterpret_cast<const uint8_t*>(&feature_bits_), sizeof(feature_bits_));
            file.write(reinterpret_cast<const uint8_t*>(&threshold_bits_), sizeof(threshold_bits_));
            file.write(reinterpret_cast<const uint8_t*>(&child_bits_), sizeof(child_bits_));
            file.write(reinterpret_cast<const uint8_t*>(&scale_factor_bits_), sizeof(scale_factor_bits_));
            file.write(reinterpret_cast<const uint8_t*>(&lr), sizeof(lr));
            
            file.close();
            
            // Reset streaming state
            streaming_rounds_ = 0;
            streaming_trees_ = 0;
            streaming_nodes_ = 0;
            streaming_tree_offsets_.clear();
            streaming_packed_data_.clear();
            
            return true;
        }
        
        /**
         * @brief Append a round of trees (one per class) to streaming buffer
         * @param trees Array of K trees (one per class), already quantized
         * @return true if successful
         */
        bool appendRoundToStream(vector<XG_tree>& trees) {
            if (trees.size() != config_ptr->num_labels) {
                eml_debug(0, "❌ appendRoundToStream: expected ", config_ptr->num_labels, " trees, got ", trees.size());
                return false;
            }
            
            // Round-major ordering: for round r, trees are [class0, class1, ..., classK-1]
            for (xg_label_type k = 0; k < config_ptr->num_labels; ++k) {
                XG_tree& tree = trees[k];
                
                // Ensure tree is quantized and packed
                if (tree.in_build_mode) {
                    if (!tree.quantizeAndPack(node_resource.scale_factor_bits)) {
                        eml_debug(0, "❌ Failed to quantize tree for class ", k);
                        return false;
                    }
                }
                
                // Record offset
                streaming_tree_offsets_.push_back(streaming_nodes_);
                
                // Append packed nodes to streaming buffer
                uint8_t bits_per_node = node_resource.bits_per_node;
                for (size_t i = 0; i < tree.packed_nodes.size(); ++i) {
                    uint32_t node_val = static_cast<uint32_t>(tree.packed_nodes.get(i));
                    
                    // Expand streaming_packed_data_ as needed
                    size_t bit_offset = static_cast<size_t>(streaming_nodes_) * bits_per_node;
                    size_t required_bytes = (bit_offset + bits_per_node + 7) / 8;
                    if (streaming_packed_data_.size() < required_bytes) {
                        streaming_packed_data_.resize(required_bytes, 0);
                    }
                    
                    // Pack node bits (MSB first)
                    for (int b = bits_per_node - 1; b >= 0; --b) {
                        uint8_t bit = (node_val >> b) & 1;
                        size_t bit_pos = bit_offset + (bits_per_node - 1 - b);
                        size_t byte_idx = bit_pos / 8;
                        size_t bit_in_byte = 7 - (bit_pos % 8);
                        streaming_packed_data_[byte_idx] |= (bit << bit_in_byte);
                    }
                    
                    streaming_nodes_++;
                }
                
                streaming_trees_++;
                
                // Update statistics
                total_nodes += tree.countNodes();
                total_leaves += tree.countLeafNodes();
                uint16_t d = tree.getTreeDepth();
                if (d > max_depth) max_depth = d;
                
                // Clear tree to free RAM
                tree.clear();
            }
            
            streaming_rounds_++;
            return true;
        }
        
        /**
         * @brief Finalize streaming save - write tree offsets, packed data, and update header
         * @return true if successful
         */
        bool finalizeStreamingSave() {
            if (streaming_model_path_[0] == '\0') {
                eml_debug(0, "❌ No streaming save in progress");
                return false;
            }
            
            // Open file in read+write mode to update header
            File file = RF_FS_OPEN(streaming_model_path_, "r+b");
            if (!file) {
                eml_debug(0, "❌ Failed to reopen for finalization: ", streaming_model_path_);
                return false;
            }
            
            // Seek to position after magic+version (8 bytes)
            file.seek(8);
            
            // Update header values
            uint16_t nc = config_ptr->num_labels;
            uint16_t nr = streaming_rounds_;
            uint32_t tt = streaming_trees_;
            uint32_t tn = streaming_nodes_;
            
            file.write(reinterpret_cast<uint8_t*>(&nc), sizeof(nc));
            file.write(reinterpret_cast<uint8_t*>(&nr), sizeof(nr));
            file.write(reinterpret_cast<uint8_t*>(&tt), sizeof(tt));
            file.write(reinterpret_cast<uint8_t*>(&tn), sizeof(tn));
            
            // Seek to end of header (8 + 12 + 10 = 30 bytes)
            file.seek(30);
            
            // Write tree offsets
            file.write(reinterpret_cast<const uint8_t*>(streaming_tree_offsets_.data()), 
                       streaming_tree_offsets_.size() * sizeof(uint32_t));
            
            // Write packed data
            file.write(streaming_packed_data_.data(), streaming_packed_data_.size());
            
            file.close();
            
            // Update container state
            num_boost_rounds_count = streaming_rounds_;
            total_trees_count = streaming_trees_;
            total_nodes = streaming_nodes_;
            
            // Clear streaming buffers
            streaming_tree_offsets_.clear();
            streaming_packed_data_.clear();
            streaming_model_path_[0] = '\0';
            
            eml_debug(1, "✅ Streaming save complete: ", streaming_rounds_, " rounds, ", 
                      streaming_trees_, " trees, ", streaming_nodes_, " nodes");
            
            is_loaded = false;  // Model is on disk, not in RAM
            return true;
        }
        
        /**
         * @brief Clear all trees from container (for per-round streaming)
         */
        void clearTrees() {
            for (auto& v : trees_by_class) {
                for (auto& t : v) t.clear();
                v.clear();
            }
            total_nodes = 0;
            total_leaves = 0;
            max_depth = 0;
        }
        
        // Streaming state
    private:
        char streaming_model_path_[XG_PATH_BUFFER] = {'\0'};
        uint16_t streaming_rounds_ = 0;
        uint32_t streaming_trees_ = 0;
        uint32_t streaming_nodes_ = 0;
        vector<uint32_t> streaming_tree_offsets_;
        vector<uint8_t> streaming_packed_data_;

    public:
        size_t memoryUsage() const {
            size_t total = sizeof(*this);
            total += packed_data.memory_usage();
            total += tree_offsets.size() * sizeof(uint32_t);
            for (const auto& v : trees_by_class) {
                for (const auto& t : v) {
                    total += t.memoryUsage();
                }
            }
            return total;
        }
    };

} // namespace mcu
