#pragma once

#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <limits>
#include <string>
#include <sstream>
#include <vector>

#include "../base/eml_base.h"
#include "eml_samples.h"
#include "eml_quantize.h"

namespace eml {

    // eml data (core dataset container)
    template<problem_type ProblemType = problem_type::CLASSIFICATION>
    class eml_data {
    public:
        using sample_type = eml_sample_t<ProblemType>;
        using label_type  = eml_label_t<ProblemType>;
        using sampleID_set = ID_vector<sample_idx_type>;

    private:

        static constexpr size_t MAX_CHUNKS_SIZE = 1048576; // max bytes per chunk (1MB for Linux)

        // Maximum dataset file size (1GB)
        static constexpr size_t MAX_DATASET_BYTES = 1073741824ULL;

        // Chunked packed storage - eliminates both heap overhead per sample and large contiguous allocations
        vector<packed_vector<8>> sampleChunks;  // Multiple chunks of packed features (up to 8 bits per value)
        packed_vector<8> allLabels;             // Labels storage
        uint16_t bitsPerSample = 0;             // Number of bits per sample (numFeatures * quantization_coefficient)
        sample_idx_type samplesEachChunk = 0;   // Maximum samples per chunk
        size_t size_ = 0;
        uint8_t quantization_coefficient = 2;   // Bits per feature value (1-8)
        char file_path[EML_PATH_BUFFER] = {0};  // dataset file_path
        size_t chunk_size = MAX_CHUNKS_SIZE;    // chunk size in bytes (configurable)

        // Pending quantizer update mapping (concept drift): applied on next RAM load.
        eml_quantizer_update_filter quantizer_update_filter;

        static constexpr char EML_DATA_MAGIC_[4] = {'E', 'M', 'L', 'D'};
        static constexpr uint16_t EML_DATA_VERSION_ = 1u;
        static constexpr uint8_t EML_ENDIAN_LITTLE_ = 1u;
        static constexpr uint16_t EML_DATA_HEADER_SIZE_ = 32u;
        static constexpr uint32_t EML_CHECKSUM_SEED_ = 2166136261u;
        static constexpr uint32_t EML_CHECKSUM_PRIME_ = 16777619u;

        struct dataset_file_layout {
            bool modern_format = false;
            uint32_t num_samples = 0u;
            uint16_t num_features = 0u;
            uint8_t quant_bits = 0u;
            uint8_t label_size = 0u;
            uint8_t label_bits = 0u;
            uint16_t header_size = 0u;
            uint16_t packed_feature_bytes = 0u;
            uint16_t record_size = 0u;
            uint64_t data_offset = 0ull;
            uint64_t data_size = 0ull;
            uint64_t checksum_offset = 0ull;
            uint32_t checksum = 0u;
        };

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
                current *= EML_CHECKSUM_PRIME_;
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

        static bool compute_packed_feature_bytes(uint16_t num_features,
                                                 uint8_t quant_bits,
                                                 uint16_t& packed_feature_bytes) {
            if (num_features == 0u || quant_bits < 1u || quant_bits > 8u) {
                return false;
            }
            const uint32_t total_bits = static_cast<uint32_t>(num_features) * quant_bits;
            packed_feature_bytes = static_cast<uint16_t>((total_bits + 7u) / 8u);
            return packed_feature_bytes > 0u;
        }

        static bool verify_payload_checksum(std::ifstream& file,
                                            uint64_t data_offset,
                                            uint64_t data_size,
                                            uint32_t expected_checksum) {
            file.clear();
            file.seekg(static_cast<std::streamoff>(data_offset), std::ios::beg);
            if (!file.good()) {
                return false;
            }

            std::vector<uint8_t> buffer(65536u, 0u);
            uint32_t checksum = EML_CHECKSUM_SEED_;
            uint64_t remaining = data_size;

            while (remaining > 0u) {
                const size_t chunk_size = static_cast<size_t>(std::min<uint64_t>(remaining, buffer.size()));
                if (!read_exact(file, buffer.data(), chunk_size)) {
                    return false;
                }
                checksum = checksum_update(checksum, buffer.data(), chunk_size);
                remaining -= chunk_size;
            }

            return checksum == expected_checksum;
        }

        static bool write_dataset_header(std::ostream& out,
                                         uint32_t num_samples,
                                         uint16_t num_features,
                                         uint8_t quant_bits,
                                         uint8_t label_size,
                                         uint8_t label_bits,
                                         uint64_t data_size) {
            if (!write_exact(out, EML_DATA_MAGIC_, sizeof(EML_DATA_MAGIC_))) return false;
            if (!write_u16_le(out, EML_DATA_VERSION_)) return false;
            if (!write_u8(out, EML_ENDIAN_LITTLE_)) return false;
            if (!write_u16_le(out, EML_DATA_HEADER_SIZE_)) return false;
            if (!write_u64_le(out, data_size)) return false;
            if (!write_u32_le(out, num_samples)) return false;
            if (!write_u16_le(out, num_features)) return false;
            if (!write_u8(out, quant_bits)) return false;
            if (!write_u8(out, label_size)) return false;
            if (!write_u8(out, label_bits)) return false;
            if (!write_u8(out, 0u)) return false;

            constexpr std::array<uint8_t, EML_DATA_HEADER_SIZE_ - 28u> k_padding = {};
            if (!write_exact(out, k_padding.data(), k_padding.size())) return false;
            return true;
        }

        bool parse_dataset_layout(std::ifstream& file,
                                  dataset_file_layout& layout,
                                  uint8_t expected_quant_bits,
                                  uint8_t expected_label_size,
                                  bool validate_checksum = true) const {
            layout = dataset_file_layout{};

            file.clear();
            file.seekg(0, std::ios::end);
            const std::streamoff file_size_off = file.tellg();
            if (file_size_off <= 0) {
                return false;
            }
            const uint64_t file_size = static_cast<uint64_t>(file_size_off);

            file.clear();
            file.seekg(0, std::ios::beg);
            char magic[4] = {0, 0, 0, 0};
            if (!read_exact(file, magic, sizeof(magic))) {
                return false;
            }

            if (std::memcmp(magic, EML_DATA_MAGIC_, sizeof(EML_DATA_MAGIC_)) == 0) {
                uint16_t version = 0u;
                uint8_t endian_flag = 0u;
                uint16_t header_size = 0u;
                uint64_t data_size = 0ull;
                uint32_t num_samples = 0u;
                uint16_t num_features = 0u;
                uint8_t quant_bits = 0u;
                uint8_t label_size = 0u;
                uint8_t label_bits = 0u;
                uint8_t reserved = 0u;

                if (!read_u16_le(file, version) ||
                    !read_u8(file, endian_flag) ||
                    !read_u16_le(file, header_size) ||
                    !read_u64_le(file, data_size) ||
                    !read_u32_le(file, num_samples) ||
                    !read_u16_le(file, num_features) ||
                    !read_u8(file, quant_bits) ||
                    !read_u8(file, label_size) ||
                    !read_u8(file, label_bits) ||
                    !read_u8(file, reserved)) {
                    return false;
                }

                if (version != EML_DATA_VERSION_ || endian_flag != EML_ENDIAN_LITTLE_) {
                    return false;
                }
                if (header_size < EML_DATA_HEADER_SIZE_ || header_size > file_size) {
                    return false;
                }
                if (quant_bits < 1u || quant_bits > 8u || quant_bits != expected_quant_bits) {
                    return false;
                }
                if (label_size != expected_label_size) {
                    return false;
                }

                uint16_t packed_feature_bytes = 0u;
                if (!compute_packed_feature_bytes(num_features, quant_bits, packed_feature_bytes)) {
                    return false;
                }
                const uint16_t record_size = static_cast<uint16_t>(packed_feature_bytes + label_size);
                const uint64_t expected_data_size = static_cast<uint64_t>(num_samples) * static_cast<uint64_t>(record_size);
                if (data_size != expected_data_size) {
                    return false;
                }

                const uint64_t checksum_offset = static_cast<uint64_t>(header_size) + data_size;
                if (checksum_offset + sizeof(uint32_t) > file_size) {
                    return false;
                }

                file.clear();
                file.seekg(static_cast<std::streamoff>(checksum_offset), std::ios::beg);
                uint32_t checksum = 0u;
                if (!read_u32_le(file, checksum)) {
                    return false;
                }

                if (validate_checksum && !verify_payload_checksum(file, static_cast<uint64_t>(header_size), data_size, checksum)) {
                    return false;
                }

                layout.modern_format = true;
                layout.num_samples = num_samples;
                layout.num_features = num_features;
                layout.quant_bits = quant_bits;
                layout.label_size = label_size;
                layout.label_bits = label_bits;
                layout.header_size = header_size;
                layout.packed_feature_bytes = packed_feature_bytes;
                layout.record_size = record_size;
                layout.data_offset = static_cast<uint64_t>(header_size);
                layout.data_size = data_size;
                layout.checksum_offset = checksum_offset;
                layout.checksum = checksum;
                return true;
            }

            return false;
        }

        static bool rewrite_modern_header(std::fstream& file,
                                          uint32_t num_samples,
                                          uint16_t num_features,
                                          uint8_t quant_bits,
                                          uint8_t label_size,
                                          uint8_t label_bits,
                                          uint64_t data_size) {
            file.clear();
            file.seekp(0, std::ios::beg);
            if (!file.good()) {
                return false;
            }
            return write_dataset_header(file, num_samples, num_features, quant_bits, label_size, label_bits, data_size);
        }

        static bool write_modern_checksum(std::fstream& file,
                                          uint64_t data_offset,
                                          uint64_t data_size) {
            file.clear();
            file.seekg(static_cast<std::streamoff>(data_offset), std::ios::beg);
            if (!file.good()) {
                return false;
            }

            std::vector<uint8_t> buffer(65536u, 0u);
            uint32_t checksum = EML_CHECKSUM_SEED_;
            uint64_t remaining = data_size;

            while (remaining > 0u) {
                const size_t chunk_size = static_cast<size_t>(std::min<uint64_t>(remaining, buffer.size()));
                if (!read_exact(file, buffer.data(), chunk_size)) {
                    return false;
                }
                checksum = checksum_update(checksum, buffer.data(), chunk_size);
                remaining -= chunk_size;
            }

            file.clear();
            file.seekp(static_cast<std::streamoff>(data_offset + data_size), std::ios::beg);
            if (!file.good()) {
                return false;
            }
            if (!write_u32_le(file, checksum)) {
                return false;
            }

            file.flush();
            return file.good();
        }

    public:
        bool isLoaded = false;

        // Lightweight, zero-allocation view into a single sample's features.
        // Designed for hot training loops that only need random feature reads.
        struct features_view {
            const packed_vector<8>* chunk = nullptr;
            size_t start = 0;
            uint16_t num_features = 0;

            inline uint16_t size() const { return num_features; }

            inline uint8_t operator[](size_t featureIndex) const {
                if (!chunk || featureIndex >= static_cast<size_t>(num_features)) {
                    return 0;
                }
                return static_cast<uint8_t>((*chunk)[start + featureIndex]);
            }
        };

        // Lightweight, zero-allocation view into a sample (label + features_view).
        // Implicitly convertible to sample_type for compatibility.
        struct sample_view {
            label_type label = 0;
            features_view features;

            inline uint8_t operator[](size_t featureIndex) const {
                return features[featureIndex];
            }

            inline sample_type materialize() const {
                if (!features.chunk || features.num_features == 0) {
                    return sample_type();
                }
                return sample_type::template create_from_slice<8>(
                    label,
                    *features.chunk,
                    features.start,
                    features.start + static_cast<size_t>(features.num_features)
                );
            }

            inline operator sample_type() const {
                return materialize();
            }
        };

        eml_data() = default;

        bool init(const char* path, uint16_t num_features, label_type num_labels, uint8_t quant_bits, sample_idx_type num_samples) {
            if (!path || num_features == 0) {
                return false;
            }
            strncpy(this->file_path, path, EML_PATH_BUFFER);
            this->file_path[EML_PATH_BUFFER - 1] = '\0';

            if (quant_bits < 1) quant_bits = 1;
            if (quant_bits > 8) quant_bits = 8;
            quantization_coefficient = quant_bits;

            bitsPerSample = static_cast<uint16_t>(num_features * quantization_coefficient);

            uint8_t label_bits = desired_bits(static_cast<uint32_t>(num_labels));
            if (label_bits == 0) label_bits = 1;
            if (label_bits > 8) label_bits = 8;
            allLabels.set_bits_per_value(label_bits);

            updateSamplesEachChunk();
            eml_debug_2(1, "ℹ️ eml_data initialized (", samplesEachChunk, "samples/chunk): ", path);
            isLoaded = false;
            size_ = num_samples;
            sampleChunks.clear();
            allLabels.clear();
            quantizer_update_filter.clear();
            return isProperlyInitialized();
        }

        void set_chunk_size(size_t bytes) {
            if (bytes == 0) {
                return;
            }
            chunk_size = bytes;
            updateSamplesEachChunk();
        }

        size_t get_chunk_size() const {
            return chunk_size;
        }

        eml_quantizer_update_filter& get_update_filter() { return quantizer_update_filter; }
        const eml_quantizer_update_filter& get_update_filter() const { return quantizer_update_filter; }

        void clear_update_filter() { quantizer_update_filter.clear(); }

        // Helper method to reconstruct sample from chunked packed storage
        sample_type getSample(size_t sampleIndex) const {
            if (!isLoaded) {
                eml_debug(2, "❌ eml_data not loaded. Call loadData() first.");
                return sample_type();
            }
            if (sampleIndex >= size_) {
                eml_debug_2(2, "❌ Sample index out of bounds: ", sampleIndex, "size: ", size_);
                return sample_type();
            }
            pair<size_t, size_t> location = getChunkLocation(sampleIndex);
            size_t numFeatures = bitsPerSample / quantization_coefficient;
            return sample_type::template create_from_slice<8>(
                static_cast<label_type>(allLabels[sampleIndex]),
                sampleChunks[location.first],
                location.second * numFeatures,
                (location.second + 1) * numFeatures
            );
        }

        // Apply a mapping filter to currently loaded (RAM) quantized samples.
        // This is used for immediate remapping after a quantizer update/shrink.
        bool apply_update_filter_inplace(const eml_quantizer_update_filter& filter) {
            if (!isLoaded) {
                return false;
            }
            const uint16_t numFeatures = bitsPerSample / quantization_coefficient;
            if (!filter.active() || filter.numFeatures() != numFeatures || filter.groupsPerFeature() != (1u << quantization_coefficient)) {
                return false;
            }
            const uint16_t gpf = filter.groupsPerFeature();
            for (size_t ci = 0; ci < sampleChunks.size(); ++ci) {
                packed_vector<8>& chunk = sampleChunks[ci];
                const size_t chunkSize = chunk.size();
                for (size_t ei = 0; ei < chunkSize; ++ei) {
                    const uint16_t fidx = static_cast<uint16_t>(ei % numFeatures);
                    const uint8_t oldVal = static_cast<uint8_t>(chunk[ei]);
                    if (oldVal < gpf) {
                        const uint8_t newVal = filter.map(fidx, oldVal);
                        chunk.set_unsafe(ei, newVal);
                    }
                }
            }
            return true;
        }

        // Iterator class (returns sample_type by value for read-only querying)
        class iterator {
        private:
            eml_data* data_;
            size_t index_;

        public:
            iterator(eml_data* data, size_t index) : data_(data), index_(index) {}

            sample_type operator*() const {
                return data_->getSample(index_);
            }

            iterator& operator++() {
                ++index_;
                return *this;
            }

            iterator operator++(int) {
                iterator temp = *this;
                ++index_;
                return temp;
            }

            bool operator==(const iterator& other) const {
                return data_ == other.data_ && index_ == other.index_;
            }

            bool operator!=(const iterator& other) const {
                return !(*this == other);
            }
        };

        // Iterator support
        iterator begin() { return iterator(this, 0); }
        iterator end() { return iterator(this, size_); }

        // Fast array access operator: returns a zero-copy view for hot paths.
        // If a full owning sample is needed, it will be materialized via implicit conversion.
        sample_view operator[](size_t index) {
            return static_cast<const eml_data&>(*this)[index];
        }

        // Const version of array access operator
        sample_view operator[](size_t index) const {
            sample_view view;
            if (!isLoaded || index >= size_) {
                return view;
            }
            const uint16_t numFeatures = static_cast<uint16_t>(bitsPerSample / quantization_coefficient);
            const pair<size_t, size_t> location = getChunkLocation(index);
            const size_t chunkIndex = location.first;
            const size_t localIndex = location.second;
            if (chunkIndex >= sampleChunks.size() || numFeatures == 0) {
                return view;
            }

            view.label = static_cast<label_type>(allLabels[index]);
            view.features.chunk = &sampleChunks[chunkIndex];
            view.features.num_features = numFeatures;
            view.features.start = localIndex * static_cast<size_t>(numFeatures);
            return view;
        }

        // Validate that the eml_data has been properly initialized
        bool isProperlyInitialized() const {
            return bitsPerSample > 0 && samplesEachChunk > 0;
        }

    private:
        // Calculate maximum samples per chunk based on bitsPerSample
        void updateSamplesEachChunk() {
            if (bitsPerSample > 0) {
                // Each sample needs bitsPerSample bits, chunk_size is in bytes (8 bits each)
                samplesEachChunk = static_cast<sample_idx_type>((chunk_size * 8) / bitsPerSample);
                if (samplesEachChunk == 0) samplesEachChunk = 1; // At least 1 sample per chunk
            }
        }

        // Get chunk index and local index within chunk for a given sample index
        pair<size_t, size_t> getChunkLocation(size_t sampleIndex) const {
            size_t chunkIndex = sampleIndex / samplesEachChunk;
            size_t localIndex = sampleIndex % samplesEachChunk;
            return make_pair(chunkIndex, localIndex);
        }

        // Ensure we have enough chunks to store the given number of samples
        void ensureChunkCapacity(size_t totalSamples) {
            size_t requiredChunks = (totalSamples + samplesEachChunk - 1) / samplesEachChunk;
            while (sampleChunks.size() < requiredChunks) {
                packed_vector<8> newChunk;
                // Reserve space for elements (each element uses quantization_coefficient bits)
                size_t elementsPerSample = bitsPerSample / quantization_coefficient;  // numFeatures
                newChunk.set_bits_per_value(quantization_coefficient);
                newChunk.reserve(samplesEachChunk * elementsPerSample);
                sampleChunks.push_back(newChunk); // Add new empty chunk
            }
        }

        // Helper method to store sample in chunked packed storage
        bool storeSample(const sample_type& sample, size_t sampleIndex) {
            if (!isProperlyInitialized()) {
                eml_debug(2, "❌ Store sample failed: eml_data not properly initialized.");
                return false;
            }

            // Store label
            if (sampleIndex == allLabels.size()) {
                // Appending in order (fast path)
                allLabels.push_back(static_cast<uint32_t>(sample.label));
            } else if (sampleIndex < allLabels.size()) {
                // Overwrite existing position
                allLabels.set(sampleIndex, static_cast<uint32_t>(sample.label));
            } else {
                // Rare case: out-of-order insert; fill gaps with 0
                allLabels.resize(sampleIndex + 1, 0);
                allLabels.push_back(static_cast<uint32_t>(sample.label));
            }

            // Ensure we have enough chunks
            ensureChunkCapacity(sampleIndex + 1);

            auto location = getChunkLocation(sampleIndex);
            size_t chunkIndex = location.first;
            size_t localIndex = location.second;

            // Store features in packed format within the specific chunk
            size_t elementsPerSample = bitsPerSample / quantization_coefficient;  // numFeatures
            size_t startElementIndex = localIndex * elementsPerSample;
            size_t requiredSizeInChunk = startElementIndex + elementsPerSample;

            if (sampleChunks[chunkIndex].size() < requiredSizeInChunk) {
                sampleChunks[chunkIndex].resize(requiredSizeInChunk);
            }

            // Store each feature as one element in the packed_vector (with variable bpv)
            for (size_t featureIdx = 0; featureIdx < sample.features.size(); featureIdx++) {
                size_t elementIndex = startElementIndex + featureIdx;
                uint8_t featureValue = sample.features[featureIdx];

                // Store value directly as one element (bpv determined by quantization_coefficient)
                if (elementIndex < sampleChunks[chunkIndex].size()) {
                    sampleChunks[chunkIndex].set(elementIndex, featureValue);
                }
            }
            return true;
        }

    private:
        // Load data from CSV format (used only once for initial dataset conversion)
        bool loadCSVData(const char* csvfile_path, uint16_t numFeatures) {
            if (isLoaded) {
                // clear existing data
                sampleChunks.clear();
                allLabels.clear();
                size_ = 0;
                isLoaded = false;
            }

            std::ifstream file(csvfile_path, std::ios::in);
            if (!file.is_open()) {
                eml_debug(0, "❌ Failed to open CSV file for reading: ", csvfile_path);
                return false;
            }

            if (numFeatures == 0) {
                // Read header line to determine number of features
                std::string line;
                if (!std::getline(file, line) || line.empty()) {
                    eml_debug(0, "❌ CSV file is empty or missing header: ", csvfile_path);
                    file.close();
                    return false;
                }
                // Trim trailing whitespace
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ' || line.back() == '\t')) {
                    line.pop_back();
                }
                int commaCount = 0;
                for (char c : line) {
                    if (c == ',') commaCount++;
                }
                numFeatures = static_cast<uint16_t>(commaCount);
            }

            // Set bitsPerSample and calculate chunk parameters only if not already initialized
            if (bitsPerSample == 0) {
                bitsPerSample = static_cast<uint16_t>(numFeatures * quantization_coefficient);
                updateSamplesEachChunk();
            } else {
                // Validate that the provided numFeatures matches the initialized bitsPerSample
                uint16_t expectedFeatures = static_cast<uint16_t>(bitsPerSample / quantization_coefficient);
                if (numFeatures != expectedFeatures) {
                    eml_debug_2(0, "❌ Feature count mismatch: expected ", expectedFeatures, ", found ", numFeatures);
                    file.close();
                    return false;
                }
            }

            sample_idx_type linesProcessed = 0;
            sample_idx_type emptyLines = 0;
            sample_idx_type validSamples = 0;
            sample_idx_type invalidSamples = 0;

            // Pre-allocate for efficiency
            allLabels.reserve(1000); // Initial capacity

            std::string line;
            while (std::getline(file, line)) {
                // Trim trailing whitespace
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ' || line.back() == '\t')) {
                    line.pop_back();
                }
                linesProcessed++;

                if (line.empty()) {
                    emptyLines++;
                    continue;
                }

                sample_type s;
                s.features.clear();
                s.features.reserve(numFeatures);

                uint16_t fieldIdx = 0;
                std::istringstream iss(line);
                std::string tok;
                while (std::getline(iss, tok, ',')) {
                    label_type v = static_cast<label_type>(std::stoi(tok));

                    if (fieldIdx == 0) {
                        s.label = v;
                    } else {
                        s.features.push_back(v);
                    }

                    fieldIdx++;
                }

                // Validate the sample
                if (fieldIdx != static_cast<uint16_t>(numFeatures + 1)) {
                    eml_debug_2(2, "❌ Invalid field count in line ", linesProcessed, ": expected ", numFeatures + 1);
                    invalidSamples++;
                    continue;
                }

                if (s.features.size() != numFeatures) {
                    eml_debug_2(2, "❌ Feature count mismatch in line ", linesProcessed, ": expected ", numFeatures);
                    invalidSamples++;
                    continue;
                }

                s.features.shrink_to_fit();

                // Store in chunked packed format
                storeSample(s, validSamples);
                validSamples++;
            }
            size_ = validSamples;

            eml_debug(1, "📋 CSV Processing Results: ");
            eml_debug(1, "   Lines processed: ", linesProcessed);
            eml_debug(1, "   Empty lines: ", emptyLines);
            eml_debug(1, "   Valid samples: ", validSamples);
            eml_debug(1, "   Invalid samples: ", invalidSamples);
            eml_debug(1, "   Total samples in memory: ", size_);
            eml_debug(1, "   Chunks used: ", sampleChunks.size());

            allLabels.shrink_to_fit();
            for (auto& chunk : sampleChunks) {
                chunk.shrink_to_fit();
            }
            file.close();
            isLoaded = true;
            // NOTE: CSV file is NOT deleted after loading (Linux behavior - user manages files)
            eml_debug(1, "✅ CSV data loaded: ", csvfile_path);
            return true;
        }

    public:
        uint8_t get_bits_per_label() const {
            return allLabels.get_bits_per_value();
        }

        int total_chunks() const {
            return static_cast<int>(size_ / samplesEachChunk + (size_ % samplesEachChunk != 0 ? 1 : 0));
        }

        uint16_t total_features() const {
            return static_cast<uint16_t>(bitsPerSample / quantization_coefficient);
        }

        sample_idx_type samplesPerChunk() const {
            return samplesEachChunk;
        }

        size_t size() const {
            return size_;
        }

        /**
         * @brief Calculate approximate memory usage of this eml_data instance
         * @return Memory usage in bytes
         */
        size_t memoryUsage() const {
            size_t total = sizeof(*this);
            // Labels storage
            total += allLabels.memory_usage();
            // Sample chunks storage
            for (const auto& chunk : sampleChunks) {
                total += chunk.memory_usage();
            }
            // Quantizer update filter (if any)
            total += quantizer_update_filter.memoryUsage();
            return total;
        }

        void setFilePath(const char* path) {
            if (!path) return;
            strncpy(this->file_path, path, EML_PATH_BUFFER);
            this->file_path[EML_PATH_BUFFER - 1] = '\0';
        }

        void getFilePath(char* buffer) const {
            if (buffer) {
                strncpy(buffer, this->file_path, EML_PATH_BUFFER);
            }
        }

        // Fast accessors for training-time hot paths (avoid reconstructing sample_type)
        inline uint16_t num_features() const { return static_cast<uint16_t>(bitsPerSample / quantization_coefficient); }

        inline label_type getLabel(size_t sampleIndex) const {
            if (sampleIndex >= size_) return 0;
            return static_cast<label_type>(allLabels[sampleIndex]);
        }

        inline uint16_t getFeature(size_t sampleIndex, uint16_t featureIndex) const {
            if (!isProperlyInitialized()) return 0;
            uint16_t nf = static_cast<uint16_t>(bitsPerSample / quantization_coefficient);
            if (featureIndex >= nf || sampleIndex >= size_) return 0;
            auto loc = getChunkLocation(sampleIndex);
            size_t chunkIndex = loc.first;
            size_t localIndex = loc.second;
            if (chunkIndex >= sampleChunks.size()) return 0;
            size_t elementsPerSample = nf;
            size_t startElementIndex = localIndex * elementsPerSample;
            size_t elementIndex = startElementIndex + featureIndex;
            if (elementIndex >= sampleChunks[chunkIndex].size()) return 0;
            return static_cast<uint16_t>(sampleChunks[chunkIndex][elementIndex]);
        }

        // Reserve space for a specified number of samples
        void reserve(size_t numSamples) {
            if (!isProperlyInitialized()) {
                eml_debug(1, "❌ Cannot reserve space: eml_data not properly initialized", file_path);
                return;
            }
            allLabels.reserve(numSamples);
            ensureChunkCapacity(numSamples);
            eml_debug_2(2, "📦 Reserved space for", numSamples, "samples, used chunks: ", sampleChunks.size());
        }

        bool convertCSVtoBinary(const char* csvfile_path, uint16_t numFeatures = 0) {
            eml_debug(1, "🔄 Converting CSV to binary format from: ", csvfile_path);
            if (!loadCSVData(csvfile_path, numFeatures)) return false;
            if (!releaseData(false)) return false;
            eml_debug(1, "✅ CSV converted to binary and saved: ", file_path);
            return true;
        }

        /**
         * @brief Save data to file system in binary format and clear from RAM.
         * @param reuse If true, keeps data in RAM after saving; if false, clears data from RAM.
         * @note: after first time eml_data created, it must be releaseData(false) to save data
         */
        bool releaseData(bool reuse = true) {
            if (!isLoaded) return false;

            if (!reuse) {
                eml_debug(1, "💾 Saving data to file system and clearing from RAM...");
                const uint32_t numSamples = static_cast<uint32_t>(size_);
                const uint16_t numFeatures = static_cast<uint16_t>(bitsPerSample / quantization_coefficient);
                uint16_t packedFeatureBytes = 0u;
                if (!compute_packed_feature_bytes(numFeatures, quantization_coefficient, packedFeatureBytes)) {
                    eml_debug(0, "❌ Failed to derive packed-feature bytes");
                    return false;
                }

                const uint16_t recordSize = static_cast<uint16_t>(sizeof(label_type) + packedFeatureBytes);
                const uint64_t dataSize = static_cast<uint64_t>(numSamples) * static_cast<uint64_t>(recordSize);
                const uint8_t labelBits = static_cast<uint8_t>(allLabels.get_bits_per_value());

                const std::filesystem::path target_path(file_path);
                if (!target_path.parent_path().empty()) {
                    std::filesystem::create_directories(target_path.parent_path());
                }
                const std::filesystem::path temp_path = target_path.string() + ".tmp";
                std::ofstream file(temp_path, std::ios::binary | std::ios::trunc);
                if (!file.is_open()) {
                    eml_debug(0, "❌ Failed to open binary file for writing: ", file_path);
                    return false;
                }
                eml_debug(2, "📂 Saving data to: ", file_path);

                if (!write_dataset_header(file,
                                          numSamples,
                                          numFeatures,
                                          quantization_coefficient,
                                          static_cast<uint8_t>(sizeof(label_type)),
                                          labelBits,
                                          dataSize)) {
                    eml_debug(0, "❌ Failed to write dataset header: ", file_path);
                    file.close();
                    std::filesystem::remove(temp_path);
                    return false;
                }

                static constexpr size_t WRITE_BUFFER_SIZE = 4096;
                std::vector<uint8_t> writeBuffer(WRITE_BUFFER_SIZE, 0u);
                size_t bufferPos = 0;
                uint32_t checksum = EML_CHECKSUM_SEED_;

                for (sample_idx_type i = 0; i < size_; i++) {
                    // Reconstruct sample from chunked packed storage
                    sample_type s = getSample(i);

                    // Write label to buffer
                    std::memcpy(&writeBuffer[bufferPos], &s.label, sizeof(label_type));
                    bufferPos += sizeof(label_type);

                    // Initialize packed feature area to 0
                    std::memset(&writeBuffer[bufferPos], 0, packedFeatureBytes);

                    // Pack features into buffer according to quantization_coefficient
                    for (size_t j = 0; j < s.features.size(); ++j) {
                        uint32_t bitPosition = static_cast<uint32_t>(j) * quantization_coefficient;
                        uint16_t byteIndex = bitPosition / 8;
                        uint8_t bitOffset = bitPosition % 8;
                        uint8_t feature_value = s.features[j] & ((1 << quantization_coefficient) - 1);

                        if (bitOffset + quantization_coefficient <= 8) {
                            // Feature fits in single byte
                            writeBuffer[bufferPos + byteIndex] |= (feature_value << bitOffset);
                        } else {
                            // Feature spans two bytes
                            uint8_t bitsInFirstByte = 8 - bitOffset;
                            writeBuffer[bufferPos + byteIndex] |= (feature_value << bitOffset);
                            writeBuffer[bufferPos + byteIndex + 1] |= (feature_value >> bitsInFirstByte);
                        }
                    }
                    bufferPos += packedFeatureBytes;

                    // Flush buffer when full or last sample
                    if (bufferPos + recordSize > WRITE_BUFFER_SIZE || i == size_ - 1) {
                        if (!write_exact(file, writeBuffer.data(), bufferPos)) {
                            eml_debug(0, "❌ Failed to write dataset payload: ", file_path);
                            file.close();
                            std::filesystem::remove(temp_path);
                            return false;
                        }
                        checksum = checksum_update(checksum, writeBuffer.data(), bufferPos);
                        bufferPos = 0;
                    }
                }

                if (!write_u32_le(file, checksum)) {
                    eml_debug(0, "❌ Failed to write dataset checksum: ", file_path);
                    file.close();
                    std::filesystem::remove(temp_path);
                    return false;
                }

                file.flush();
                if (!file.good()) {
                    file.close();
                    std::filesystem::remove(temp_path);
                    return false;
                }
                file.close();

                std::error_code ec;
                std::filesystem::remove(target_path, ec);
                ec.clear();
                std::filesystem::rename(temp_path, target_path, ec);
                if (ec) {
                    eml_debug(0, "❌ Failed to finalize dataset file: ", file_path);
                    std::filesystem::remove(temp_path, ec);
                    return false;
                }

                // If we wrote the remapped data back to storage, any pending update filter is now obsolete.
                quantizer_update_filter.clear();
            }

            // Clear chunked memory
            sampleChunks.clear();
            sampleChunks.shrink_to_fit();
            allLabels.clear();
            allLabels.shrink_to_fit();
            isLoaded = false;
            eml_debug_2(1, "✅ Data saved(", size_, "samples) to: ", file_path);
            return true;
        }

        // Load data using sequential indices from file system in binary format
        bool loadData(bool re_use = true) {
            if (isLoaded || !isProperlyInitialized()) return false;
            eml_debug(1, "📂 Loading data from: ", file_path);

            std::ifstream file(file_path, std::ios::binary);
            if (!file.is_open()) {
                eml_debug(0, "❌ Failed to open data file: ", file_path);
                if (std::filesystem::exists(file_path)) {
                    std::filesystem::remove(file_path);
                }
                return false;
            }

            dataset_file_layout layout;
            if (!parse_dataset_layout(file,
                                      layout,
                                      quantization_coefficient,
                                      static_cast<uint8_t>(sizeof(label_type)),
                                      true)) {
                eml_debug(0, "❌ Failed to parse dataset header/checksum: ", file_path);
                file.close();
                return false;
            }

            const uint32_t numSamples = layout.num_samples;
            const uint16_t numFeatures = layout.num_features;
            const uint16_t packedFeatureBytes = layout.packed_feature_bytes;
            const size_t recordSize = layout.record_size;
            const size_t elementsPerSample = numFeatures;

            if (numFeatures * quantization_coefficient != bitsPerSample) {
                eml_debug_2(0, "❌ Feature count mismatch: expected ", bitsPerSample / quantization_coefficient, ",found ", numFeatures);
                file.close();
                return false;
            }

            if (layout.modern_format && layout.label_bits >= 1u && layout.label_bits <= 8u) {
                allLabels.set_bits_per_value(layout.label_bits);
            }

            size_ = numSamples;

            allLabels.clear();
            allLabels.reserve(numSamples);
            sampleChunks.clear();
            ensureChunkCapacity(numSamples);

            size_t remaining = numSamples;
            for (size_t ci = 0; ci < sampleChunks.size(); ++ci) {
                const size_t chunkSamples = remaining > samplesEachChunk ? samplesEachChunk : remaining;
                const size_t reqElems = chunkSamples * elementsPerSample;
                sampleChunks[ci].resize(reqElems, 0);
                remaining -= chunkSamples;
                if (remaining == 0u) {
                    break;
                }
            }

            file.clear();
            file.seekg(static_cast<std::streamoff>(layout.data_offset), std::ios::beg);
            if (!file.good()) {
                file.close();
                return false;
            }

            const size_t max_batch_bytes = std::max<size_t>(65536u, recordSize);
            std::vector<uint8_t> ioBuf(max_batch_bytes, 0u);

            size_t processed = 0u;
            while (processed < numSamples) {
                size_t maxSamplesByBuf = ioBuf.size() / recordSize;
                if (maxSamplesByBuf == 0u) {
                    maxSamplesByBuf = 1u;
                }
                const size_t batchSamples = std::min<size_t>(numSamples - processed, maxSamplesByBuf);
                const size_t bytesToRead = batchSamples * recordSize;

                if (!read_exact(file, ioBuf.data(), bytesToRead)) {
                    eml_debug(0, "❌ Read batch failed: ", file_path);
                    file.close();
                    return false;
                }

                for (size_t bi = 0u; bi < batchSamples; ++bi) {
                    const size_t off = bi * recordSize;
                    label_type lbl;
                    std::memcpy(&lbl, ioBuf.data() + off, sizeof(label_type));
                    allLabels.push_back(static_cast<uint32_t>(lbl));

                    const uint8_t* packed = ioBuf.data() + off + sizeof(label_type);
                    const size_t sampleIndex = processed + bi;

                    const auto loc = getChunkLocation(sampleIndex);
                    const size_t chunkIndex = loc.first;
                    const size_t localIndex = loc.second;
                    const size_t startElementIndex = localIndex * elementsPerSample;

                    for (uint16_t j = 0; j < numFeatures; ++j) {
                        const uint32_t bitPosition = static_cast<uint32_t>(j) * quantization_coefficient;
                        const uint16_t byteIndex = bitPosition / 8u;
                        const uint8_t bitOffset = static_cast<uint8_t>(bitPosition % 8u);

                        uint8_t fv = 0u;
                        if (bitOffset + quantization_coefficient <= 8u) {
                            const uint8_t mask = static_cast<uint8_t>(((1u << quantization_coefficient) - 1u) << bitOffset);
                            fv = static_cast<uint8_t>((packed[byteIndex] & mask) >> bitOffset);
                        } else {
                            const uint8_t bitsInFirstByte = static_cast<uint8_t>(8u - bitOffset);
                            const uint8_t bitsInSecondByte = static_cast<uint8_t>(quantization_coefficient - bitsInFirstByte);
                            const uint8_t mask1 = static_cast<uint8_t>(((1u << bitsInFirstByte) - 1u) << bitOffset);
                            const uint8_t mask2 = static_cast<uint8_t>((1u << bitsInSecondByte) - 1u);
                            fv = static_cast<uint8_t>(((packed[byteIndex] & mask1) >> bitOffset) |
                                 ((packed[byteIndex + 1u] & mask2) << bitsInFirstByte));
                        }

                        const size_t elemIndex = startElementIndex + j;
                        if (elemIndex < sampleChunks[chunkIndex].size()) {
                            sampleChunks[chunkIndex].set_unsafe(elemIndex, fv);
                        }
                    }
                }

                processed += batchSamples;
            }

            allLabels.shrink_to_fit();
            for (auto& chunk : sampleChunks) {
                chunk.shrink_to_fit();
            }

            // Apply mapping if a quantizer update was recorded.
            if (quantizer_update_filter.active() &&
                quantizer_update_filter.numFeatures() == numFeatures &&
                quantizer_update_filter.groupsPerFeature() == (1u << quantization_coefficient)) {
                eml_debug(1, "🔁 Applying quantizer update filter to loaded data");
                (void)apply_update_filter_inplace(quantizer_update_filter);
                // One-shot application completed.
                quantizer_update_filter.clear();
            }
            isLoaded = true;
            file.close();
            if (!re_use) {
                eml_debug(1, "♻️ Single-load mode: removing file after loading: ", file_path);
                std::filesystem::remove(file_path); // Remove file after loading in single mode
            }
            eml_debug_2(1, "✅ Data loaded(", sampleChunks.size(), "chunks): ", file_path);
            return true;
        }

        /**
         * @brief Load specific samples from another eml_data source by sample IDs.
         * @param source The source eml_data to load samples from.
         * @param sample_IDs A sorted set of sample IDs to load from the source.
         * @param save_ram If true, release source data(if loaded) during process to avoid both datasets in RAM.
         * @note: The state of the source data will be automatically restored, no need to reload.
         */
        bool loadData(eml_data& source, const sampleID_set& sample_IDs, bool save_ram = true) {
            // Only the source must exist on file system; destination can be an in-memory buffer
            if (!std::filesystem::exists(source.file_path)) {
                eml_debug(0, "❌ Source file does not exist: ", source.file_path);
                return false;
            }

            std::ifstream file(source.file_path, std::ios::binary);
            if (!file.is_open()) {
                eml_debug(0, "❌ Failed to open source file: ", source.file_path);
                return false;
            }
            bool pre_loaded = source.isLoaded;
            if (pre_loaded && save_ram) {
                source.releaseData();
            }
            // set all_labels bits_per_value according to source
            uint8_t bpl = source.get_bits_per_label();
            allLabels.set_bits_per_value(bpl);

            dataset_file_layout layout;
            if (!source.parse_dataset_layout(file,
                                             layout,
                                             source.quantization_coefficient,
                                             static_cast<uint8_t>(sizeof(label_type)),
                                             true)) {
                eml_debug(0, "❌ Failed to parse source header/checksum: ", source.file_path);
                file.close();
                return false;
            }

            const uint32_t numSamples = layout.num_samples;
            const uint16_t numFeatures = layout.num_features;
            const uint16_t packedFeatureBytes = layout.packed_feature_bytes;
            const size_t sampleDataSize = layout.record_size;
            const uint64_t dataOffset = layout.data_offset;

            if (layout.modern_format && layout.label_bits >= 1u && layout.label_bits <= 8u) {
                allLabels.set_bits_per_value(layout.label_bits);
            }

            // Clear current data and initialize parameters
            sampleChunks.clear();
            allLabels.clear();
            bitsPerSample = static_cast<uint16_t>(numFeatures * source.quantization_coefficient);
            quantization_coefficient = source.quantization_coefficient;
            updateSamplesEachChunk();

            // Reserve space for requested samples
            size_t numRequestedSamples = sample_IDs.size();
            allLabels.reserve(numRequestedSamples);

            eml_debug_2(2, "📦 Loading ", numRequestedSamples, "samples from source: ", source.file_path);

            size_t addedSamples = 0;
            // Since sample_IDs are sorted in ascending order, we can read efficiently
            for (sample_idx_type sampleIdx : sample_IDs) {
                if (sampleIdx >= numSamples) {
                    eml_debug_2(2, "⚠️ Sample ID ", sampleIdx, "exceeds source sample count ", numSamples);
                    continue;
                }

                // Calculate file position for this sample
                const uint64_t sampleFilePos = dataOffset +
                    (static_cast<uint64_t>(sampleIdx) * static_cast<uint64_t>(sampleDataSize));

                // Seek to the sample position
                file.seekg(static_cast<std::streamoff>(sampleFilePos));
                if (!file.good()) {
                    eml_debug_2(2, "⚠️ Failed to seek to sample ", sampleIdx, "position ", sampleFilePos);
                    continue;
                }

                sample_type s;

                // Read label
                if (!read_exact(file, &s.label, sizeof(s.label))) {
                    eml_debug(2, "⚠️ Failed to read label for sample ", sampleIdx);
                    continue;
                }

                // Read packed features
                s.features.clear();
                s.features.reserve(numFeatures);

                std::vector<uint8_t> packedBuffer(packedFeatureBytes, 0);
                if (!read_exact(file, packedBuffer.data(), packedFeatureBytes)) {
                    eml_debug(2, "⚠️ Failed to read features for sample ", sampleIdx);
                    continue;
                }

                // Unpack features from bytes according to quantization_coefficient
                for (uint16_t j = 0; j < numFeatures; j++) {
                    // Calculate bit position for this feature
                    uint32_t bitPosition = static_cast<uint32_t>(j) * quantization_coefficient;
                    uint16_t byteIndex = bitPosition / 8;
                    uint8_t bitOffset = bitPosition % 8;

                    // Extract the feature value (might span byte boundaries)
                    uint8_t feature = 0;
                    if (bitOffset + quantization_coefficient <= 8) {
                        // Feature fits in single byte
                        uint8_t mask = ((1 << quantization_coefficient) - 1) << bitOffset;
                        feature = (packedBuffer[byteIndex] & mask) >> bitOffset;
                    } else {
                        // Feature spans two bytes
                        uint8_t bitsInFirstByte = 8 - bitOffset;
                        uint8_t bitsInSecondByte = quantization_coefficient - bitsInFirstByte;
                        uint8_t mask1 = ((1 << bitsInFirstByte) - 1) << bitOffset;
                        uint8_t mask2 = (1 << bitsInSecondByte) - 1;
                        feature = ((packedBuffer[byteIndex] & mask1) >> bitOffset) |
                                  ((packedBuffer[byteIndex + 1] & mask2) << bitsInFirstByte);
                    }
                    s.features.push_back(feature);
                }
                s.features.shrink_to_fit();

                // Store in chunked packed format using addedSamples as the new index
                storeSample(s, addedSamples);
                addedSamples++;
            }

            size_ = addedSamples;
            allLabels.shrink_to_fit();
            for (auto& chunk : sampleChunks) {
                chunk.shrink_to_fit();
            }
            isLoaded = true;
            file.close();
            if (pre_loaded && save_ram) {
                source.loadData();
            }
            eml_debug_2(2, "✅ Loaded ", addedSamples, "samples from source: ", source.file_path);
            return true;
        }

        /**
         * @brief Load a specific chunk of samples from another eml_data source.
         * @param source The source eml_data to load samples from.
         * @param chunkIndex The index of the chunk to load (0-based).
         * @param save_ram If true, release source data(if loaded) during process to avoid both datasets in RAM.
         * @note: this function will call loadData(source, chunkIDs) internally.
         */
        bool loadChunk(eml_data& source, size_t chunkIndex, bool save_ram = true) {
            eml_debug_2(2, "📂 Loading chunk ", chunkIndex, "from source: ", source.file_path);
            if (chunkIndex >= static_cast<size_t>(source.total_chunks())) {
                eml_debug_2(2, "❌ Chunk index ", chunkIndex, "out of bounds : total chunks=", source.total_chunks());
                return false;
            }
            bool pre_loaded = source.isLoaded;

            sample_idx_type startSample = static_cast<sample_idx_type>(chunkIndex * source.samplesEachChunk);
            sample_idx_type endSample = startSample + source.samplesEachChunk;
            if (endSample > source.size()) {
                endSample = static_cast<sample_idx_type>(source.size());
            }
            if (startSample >= endSample) {
                eml_debug_2(2, "❌ Invalid chunk range: start ", startSample, ", end ", endSample);
                return false;
            }
            sampleID_set chunkIDs(startSample, endSample - 1);
            chunkIDs.fill();
            loadData(source, chunkIDs, save_ram);
            if (pre_loaded && save_ram) {
                source.loadData();
            }
            return true;
        }

        /**
         *@brief: copy assignment (but not copy file_path to avoid file system over-writing)
         *@note : eml_data will be put into release state. loadData() to reload into RAM if needed.
        */
        eml_data& operator=(const eml_data& other) {
            purgeData(); // Clear existing data safely
            if (this != &other) {
                if (std::filesystem::exists(other.file_path)) {
                    std::ifstream testFile(other.file_path, std::ios::binary);
                    if (testFile.is_open()) {
                        dataset_file_layout layout;
                        const bool headerValid = other.parse_dataset_layout(
                            testFile,
                            layout,
                            other.quantization_coefficient,
                            static_cast<uint8_t>(sizeof(label_type)),
                            true);
                        testFile.close();

                        if (headerValid) {
                            std::error_code ec;
                            std::filesystem::copy_file(other.file_path, file_path,
                                std::filesystem::copy_options::overwrite_existing, ec);
                            if (ec) {
                                eml_debug(0, "❌ Failed to copy source file: ", other.file_path);
                            }
                        } else {
                            eml_debug(0, "❌ Source file has invalid header: ", other.file_path);
                        }
                    } else {
                        eml_debug(0, "❌ Cannot open source file: ", other.file_path);
                    }
                } else {
                    eml_debug(0, "❌ Source file does not exist: ", other.file_path);
                }
                bitsPerSample = other.bitsPerSample;
                samplesEachChunk = other.samplesEachChunk;
                isLoaded = false; // Always start in unloaded state
                size_ = other.size_;
                quantization_coefficient = other.quantization_coefficient;
                // Deep copy of labels if loaded in memory
                allLabels = other.allLabels; // vector has its own copy semantics
            }
            return *this;
        }

        // Clear data at both memory and file system
        void purgeData() {
            // Clear in-memory structures first
            sampleChunks.clear();
            sampleChunks.shrink_to_fit();
            allLabels.clear();
            allLabels.shrink_to_fit();
            isLoaded = false;
            size_ = 0;
            bitsPerSample = 0;
            samplesEachChunk = 0;

            // Then remove the file system file if one was specified
            if (std::filesystem::exists(file_path)) {
                std::filesystem::remove(file_path);
                eml_debug(1, "🗑️ Deleted file: ", file_path);
            }
        }

        /**
         * @brief Add new data directly to file without loading into RAM
         * @param samples Vector of new samples to add
         * @param extend If false, keeps file size same (overwrites old data from start);
         *               if true, appends new data while respecting size limits
         * @return : deleted labels
         * @note Directly writes to file system file to save RAM. File must exist and be properly initialized.
         */
        vector<label_type> addNewData(const vector<sample_type>& samples, sample_idx_type max_samples = 0) {
            vector<label_type> deletedLabels;

            if (!isProperlyInitialized()) {
                eml_debug(0, "❌ eml_data not properly initialized. Cannot add new data.");
                return deletedLabels;
            }
            if (!std::filesystem::exists(file_path)) {
                eml_debug(0, "⚠️ File does not exist for adding new data: ", file_path);
                return deletedLabels;
            }
            if (samples.size() == 0) {
                eml_debug(1, "⚠️ No samples to add");
                return deletedLabels;
            }

            dataset_file_layout layout;
            {
                std::ifstream file(file_path, std::ios::binary);
                if (!file.is_open()) {
                    eml_debug(0, "❌ Failed to open file for adding new data: ", file_path);
                    return deletedLabels;
                }

                if (!parse_dataset_layout(file,
                                          layout,
                                          quantization_coefficient,
                                          static_cast<uint8_t>(sizeof(label_type)),
                                          true)) {
                    eml_debug(0, "❌ Failed to parse dataset header/checksum: ", file_path);
                    file.close();
                    return deletedLabels;
                }
                file.close();
            }

            uint32_t currentNumSamples = layout.num_samples;
            const uint16_t numFeatures = layout.num_features;
            const uint16_t packedFeatureBytes = layout.packed_feature_bytes;
            const size_t sampleDataSize = layout.record_size;
            const uint64_t headerSize = layout.data_offset;
            const bool modernFormat = layout.modern_format;
            const uint8_t labelBits = (layout.label_bits >= 1u && layout.label_bits <= 8u)
                ? layout.label_bits
                : static_cast<uint8_t>(allLabels.get_bits_per_value());

            // Validate feature count compatibility
            if (!samples.empty() && samples[0].features.size() != numFeatures) {
                eml_debug_2(0, "❌ Feature count mismatch: expected ", numFeatures, ", found ", samples[0].features.size());
                return deletedLabels;
            }

            uint32_t newNumSamples;
            uint64_t writePosition;

            // Append mode: add to existing samples
            newNumSamples = currentNumSamples + static_cast<uint32_t>(samples.size());

            // Apply max_samples limit if specified
            if (max_samples > 0 && newNumSamples > max_samples) {
                eml_debug_2(1, "📊 Applying max_samples limit: ", max_samples, " (current: ", currentNumSamples);
                // Calculate how many oldest samples to remove
                sample_idx_type samples_to_remove = static_cast<sample_idx_type>(newNumSamples - max_samples);
                newNumSamples = max_samples;

                // Read labels of samples that will be removed (oldest samples at the beginning)
                {
                    std::ifstream readFile(file_path, std::ios::binary);
                    if (readFile.is_open()) {
                        readFile.seekg(static_cast<std::streamoff>(headerSize)); // Skip to data section
                        for (sample_idx_type i = 0; i < samples_to_remove && i < currentNumSamples; i++) {
                            label_type label;
                            if (read_exact(readFile, &label, sizeof(label))) {
                                deletedLabels.push_back(label);
                            }
                            // Skip the packed features to get to next sample
                            readFile.seekg(static_cast<std::streamoff>(packedFeatureBytes), std::ios::cur);
                        }
                        readFile.close();
                    }
                }

                // Shift remaining samples to the beginning (remove oldest)
                // This is done by reading samples after the removed ones and writing them at the beginning
                if (samples_to_remove < currentNumSamples) {
                    sample_idx_type samples_to_keep = currentNumSamples - samples_to_remove;
                    vector<uint8_t> temp_buffer(sampleDataSize, 0u);

                    std::fstream shiftFile(file_path, std::ios::in | std::ios::out | std::ios::binary);
                    if (shiftFile.is_open()) {
                        // Read and shift each sample
                        for (sample_idx_type i = 0; i < samples_to_keep; i++) {
                            uint64_t read_pos = headerSize +
                                (static_cast<uint64_t>(samples_to_remove + i) * static_cast<uint64_t>(sampleDataSize));
                            uint64_t write_pos = headerSize +
                                (static_cast<uint64_t>(i) * static_cast<uint64_t>(sampleDataSize));

                            shiftFile.seekg(static_cast<std::streamoff>(read_pos));
                            if (read_exact(shiftFile, temp_buffer.data(), sampleDataSize)) {
                                shiftFile.seekp(static_cast<std::streamoff>(write_pos));
                                write_exact(shiftFile, temp_buffer.data(), temp_buffer.size());
                            }
                        }
                        shiftFile.close();
                    }

                    currentNumSamples = samples_to_keep;
                    eml_debug_2(1, "♻️  Removed ", samples_to_remove, " oldest samples, kept ", samples_to_keep);
                }
            }

            const size_t trailerSize = modernFormat ? sizeof(uint32_t) : 0u;
            size_t newFileSize = static_cast<size_t>(headerSize) + (static_cast<size_t>(newNumSamples) * sampleDataSize) + trailerSize;
            if (newFileSize > MAX_DATASET_BYTES) {
                size_t maxSamplesBySize = (MAX_DATASET_BYTES - static_cast<size_t>(headerSize) - trailerSize) / sampleDataSize;
                eml_debug(2, "⚠️ Limiting samples by file size to ", maxSamplesBySize);
                newNumSamples = static_cast<uint32_t>(maxSamplesBySize);
                newFileSize = static_cast<size_t>(headerSize) + (static_cast<size_t>(newNumSamples) * sampleDataSize) + trailerSize;
            }

            writePosition = headerSize + (static_cast<uint64_t>(currentNumSamples) * static_cast<uint64_t>(sampleDataSize));

            // Calculate actual number of samples to write
            uint32_t samplesToWrite = (newNumSamples - currentNumSamples);

            eml_debug_2(1, "📝 Adding ", samplesToWrite, "samples to ", file_path);
            eml_debug_2(2, "📊 Dataset info: current=", currentNumSamples, ", new_total=", newNumSamples);

            // Open file for writing (r+ mode to update existing file)
            std::fstream file(file_path, std::ios::in | std::ios::out | std::ios::binary);
            if (!file.is_open()) {
                eml_debug(0, "❌ Failed to open file for writing: ", file_path);
                return deletedLabels;
            }

            // Update header with new sample count
            if (modernFormat) {
                const uint64_t dataSize = static_cast<uint64_t>(newNumSamples) * static_cast<uint64_t>(sampleDataSize);
                if (!rewrite_modern_header(file,
                                           newNumSamples,
                                           numFeatures,
                                           quantization_coefficient,
                                           static_cast<uint8_t>(sizeof(label_type)),
                                           labelBits,
                                           dataSize)) {
                    eml_debug(0, "❌ Failed to rewrite modern header: ", file_path);
                    file.close();
                    return deletedLabels;
                }
            } else {
                eml_debug(0, "❌ Unsupported dataset format (legacy headers are no longer supported): ", file_path);
                file.close();
                return deletedLabels;
            }

            // Seek to write position
            file.seekp(static_cast<std::streamoff>(writePosition));
            if (!file.good()) {
                eml_debug_2(0, "❌ Failed seek to write position ", writePosition, ": ", file_path);
                file.close();
                return deletedLabels;
            }

            // Write samples directly to file
            uint32_t written = 0;
            for (uint32_t i = 0; i < samplesToWrite && i < samples.size(); ++i) {
                const sample_type& sample = samples[i];

                // Validate sample feature count
                if (sample.features.size() != numFeatures) {
                    eml_debug_2(2, "⚠️ Skipping sample ", i, " due to feature count mismatch: ", file_path);
                    continue;
                }

                // Write label
                if (!write_exact(file, &sample.label, sizeof(sample.label))) {
                    eml_debug_2(0, "❌ Write label failed at sample ", i, ": ", file_path);
                    break;
                }

                // Pack and write features
                std::vector<uint8_t> packedBuffer(packedFeatureBytes, 0);

                // Pack features according to quantization_coefficient
                for (size_t j = 0; j < sample.features.size(); ++j) {
                    uint32_t bitPosition = static_cast<uint32_t>(j) * quantization_coefficient;
                    uint16_t byteIndex = bitPosition / 8;
                    uint8_t bitOffset = bitPosition % 8;
                    uint8_t feature_value = sample.features[j] & ((1 << quantization_coefficient) - 1);

                    if (bitOffset + quantization_coefficient <= 8) {
                        // Feature fits in single byte
                        packedBuffer[byteIndex] |= (feature_value << bitOffset);
                    } else {
                        // Feature spans two bytes
                        uint8_t bitsInFirstByte = 8 - bitOffset;
                        packedBuffer[byteIndex] |= (feature_value << bitOffset);
                        packedBuffer[byteIndex + 1] |= (feature_value >> bitsInFirstByte);
                    }
                }

                if (!write_exact(file, packedBuffer.data(), packedFeatureBytes)) {
                    eml_debug_2(0, "❌ Write features failed at sample ", i, ": ", file_path);
                    break;
                }

                written++;
            }

            file.flush();
            file.close();

            std::error_code ec;
            std::filesystem::resize_file(file_path, newFileSize, ec);
            if (ec) {
                eml_debug(0, "❌ Failed to resize dataset file after append: ", file_path);
                return deletedLabels;
            }

            if (modernFormat) {
                const uint64_t dataSize = static_cast<uint64_t>(newNumSamples) * static_cast<uint64_t>(sampleDataSize);
                std::fstream checksumFile(file_path, std::ios::in | std::ios::out | std::ios::binary);
                if (!checksumFile.is_open()) {
                    eml_debug(0, "❌ Failed to open file for checksum update: ", file_path);
                    return deletedLabels;
                }
                if (!write_modern_checksum(checksumFile, headerSize, dataSize)) {
                    eml_debug(0, "❌ Failed to update dataset checksum: ", file_path);
                    checksumFile.close();
                    return deletedLabels;
                }
                checksumFile.close();
            }

            // Update internal size if data is loaded in memory
            if (isLoaded) {
                size_ = newNumSamples;
                eml_debug(1, "ℹ️ Data is loaded in memory. Consider reloading for consistency.");
            }

            eml_debug_2(1, "✅ Successfully wrote ", written, "samples to: ", file_path);

            return deletedLabels;
        }

        size_t memory_usage() const {
            size_t total = sizeof(eml_data);
            total += allLabels.capacity() * sizeof(label_type);
            for (const auto& chunk : sampleChunks) {
                total += sizeof(packed_vector<8>);
                total += chunk.capacity() * sizeof(uint8_t); // stored in bytes regardless of bpv
            }
            return total;
        }
    };

} // namespace eml
