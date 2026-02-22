#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <filesystem>
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
                // Remove any existing file
                if (std::filesystem::exists(file_path)) {
                    std::filesystem::remove(file_path);
                }
                std::ofstream file(file_path, std::ios::binary);
                if (!file.is_open()) {
                    eml_debug(0, "❌ Failed to open binary file for writing: ", file_path);
                    return false;
                }
                eml_debug(2, "📂 Saving data to: ", file_path);

                // Write binary header
                uint32_t numSamples = static_cast<uint32_t>(size_);
                uint16_t numFeatures = static_cast<uint16_t>(bitsPerSample / quantization_coefficient);

                file.write(reinterpret_cast<const char*>(&numSamples), sizeof(numSamples));
                file.write(reinterpret_cast<const char*>(&numFeatures), sizeof(numFeatures));

                // Calculate packed bytes needed for features per sample
                uint32_t totalBits = static_cast<uint32_t>(numFeatures) * quantization_coefficient;
                uint16_t packedFeatureBytes = (totalBits + 7) / 8; // Round up to nearest byte

                // Record size = label + packed features
                uint16_t recordSize = sizeof(label_type) + packedFeatureBytes;

                // Use a heap-allocated write buffer to batch multiple samples
                static constexpr size_t WRITE_BUFFER_SIZE = 4096;
                uint8_t* writeBuffer = new (std::nothrow) uint8_t[WRITE_BUFFER_SIZE];
                if (!writeBuffer) {
                    eml_debug(0, "❌ Failed to allocate write buffer");
                    file.close();
                    return false;
                }
                size_t bufferPos = 0;

                // Calculate how many complete samples fit in buffer
                sample_idx_type samplesPerBuffer = static_cast<sample_idx_type>(WRITE_BUFFER_SIZE / recordSize);
                if (samplesPerBuffer == 0) samplesPerBuffer = 1; // At least one sample per write

                for (sample_idx_type i = 0; i < size_; i++) {
                    // Reconstruct sample from chunked packed storage
                    sample_type s = getSample(i);

                    // Write label to buffer
                    memcpy(&writeBuffer[bufferPos], &s.label, sizeof(label_type));
                    bufferPos += sizeof(label_type);

                    // Initialize packed feature area to 0
                    memset(&writeBuffer[bufferPos], 0, packedFeatureBytes);

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
                        file.write(reinterpret_cast<const char*>(writeBuffer), bufferPos);
                        bufferPos = 0;
                    }
                }
                delete[] writeBuffer;
                file.close();

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

            // Read binary header
            uint32_t numSamples;
            uint16_t numFeatures;

            if (!file.read(reinterpret_cast<char*>(&numSamples), sizeof(numSamples)) ||
                !file.read(reinterpret_cast<char*>(&numFeatures), sizeof(numFeatures))) {
                eml_debug(0, "❌ Failed to read data header: ", file_path);
                file.close();
                return false;
            }

            if (numFeatures * quantization_coefficient != bitsPerSample) {
                eml_debug_2(0, "❌ Feature count mismatch: expected ", bitsPerSample / quantization_coefficient, ",found ", numFeatures);
                file.close();
                return false;
            }
            size_ = numSamples;

            // Calculate sizes based on quantization_coefficient
            uint32_t totalBits = static_cast<uint32_t>(numFeatures) * quantization_coefficient;
            const uint16_t packedFeatureBytes = (totalBits + 7) / 8; // Round up to nearest byte
            const size_t recordSize = sizeof(label_type) + packedFeatureBytes; // label + packed features
            const size_t elementsPerSample = numFeatures; // each feature is one element in packed_vector

            // Prepare storage: labels and chunks pre-sized to avoid per-sample resizing
            allLabels.clear();
            allLabels.reserve(numSamples);
            sampleChunks.clear();
            ensureChunkCapacity(numSamples);
            // Pre-size each chunk's element count and explicitly initialize to zero
            size_t remaining = numSamples;
            for (size_t ci = 0; ci < sampleChunks.size(); ++ci) {
                size_t chunkSamples = remaining > samplesEachChunk ? samplesEachChunk : remaining;
                size_t reqElems = chunkSamples * elementsPerSample;
                sampleChunks[ci].resize(reqElems, 0);  // Explicitly pass 0 as value
                remaining -= chunkSamples;
                if (remaining == 0) break;
            }

            // Batch read to reduce file I/O calls
            const size_t MAX_BATCH_BYTES = 65536; // 64KB for Linux
            uint8_t* ioBuf = new (std::nothrow) uint8_t[MAX_BATCH_BYTES];
            if (!ioBuf) {
                eml_debug(1, "❌ Failed to allocate IO buffer");
                file.close();
                return false;
            }

            bool fallback_yet = false;
            size_t processed = 0;
            while (processed < numSamples) {
                size_t batchSamples;
                if (ioBuf) {
                    size_t maxSamplesByBuf = MAX_BATCH_BYTES / recordSize;
                    if (maxSamplesByBuf == 0) maxSamplesByBuf = 1;
                    batchSamples = (numSamples - processed) < maxSamplesByBuf ? (numSamples - processed) : maxSamplesByBuf;

                    size_t bytesToRead = batchSamples * recordSize;
                    file.read(reinterpret_cast<char*>(ioBuf), bytesToRead);
                    size_t bytesRead = static_cast<size_t>(file.gcount());
                    if (bytesRead < bytesToRead) {
                        eml_debug(0, "❌ Read batch failed: ", file_path);
                        delete[] ioBuf;
                        file.close();
                        return false;
                    }

                    // Process buffer
                    for (size_t bi = 0; bi < batchSamples; ++bi) {
                        size_t off = bi * recordSize;
                        label_type lbl;
                        memcpy(&lbl, ioBuf + off, sizeof(label_type));
                        allLabels.push_back(static_cast<uint32_t>(lbl));

                        const uint8_t* packed = ioBuf + off + sizeof(label_type);
                        size_t sampleIndex = processed + bi;

                        // Locate chunk and base element index for this sample
                        auto loc = getChunkLocation(sampleIndex);
                        size_t chunkIndex = loc.first;
                        size_t localIndex = loc.second;
                        size_t startElementIndex = localIndex * elementsPerSample;

                        // Unpack features directly into chunk storage using set_unsafe for pre-sized storage
                        for (uint16_t j = 0; j < numFeatures; ++j) {
                            uint32_t bitPosition = static_cast<uint32_t>(j) * quantization_coefficient;
                            uint16_t byteIndex = bitPosition / 8;
                            uint8_t bitOffset = bitPosition % 8;

                            uint8_t fv = 0;
                            if (bitOffset + quantization_coefficient <= 8) {
                                // Feature in single byte
                                uint8_t mask = ((1 << quantization_coefficient) - 1) << bitOffset;
                                fv = (packed[byteIndex] & mask) >> bitOffset;
                            } else {
                                // Feature spans two bytes
                                uint8_t bitsInFirstByte = 8 - bitOffset;
                                uint8_t bitsInSecondByte = quantization_coefficient - bitsInFirstByte;
                                uint8_t mask1 = ((1 << bitsInFirstByte) - 1) << bitOffset;
                                uint8_t mask2 = (1 << bitsInSecondByte) - 1;
                                fv = ((packed[byteIndex] & mask1) >> bitOffset) |
                                     ((packed[byteIndex + 1] & mask2) << bitsInFirstByte);
                            }

                            size_t elemIndex = startElementIndex + j;
                            if (elemIndex >= sampleChunks[chunkIndex].size()) {
                                eml_debug_2(0, "❌ Index out of bounds: elemIndex=", elemIndex, ", size=", sampleChunks[chunkIndex].size());
                            }
                            sampleChunks[chunkIndex].set_unsafe(elemIndex, fv);
                        }
                    }
                } else {
                    if (!fallback_yet) {
                        eml_debug(2, "⚠️ IO buffer allocation failed, falling back to per-sample read");
                        fallback_yet = true;
                    }
                    // Fallback: per-sample small buffer
                    batchSamples = 1;
                    label_type lbl;
                    if (!file.read(reinterpret_cast<char*>(&lbl), sizeof(lbl))) {
                        eml_debug_2(0, "❌ Read label failed at sample: ", processed, ": ", file_path);
                        delete[] ioBuf;
                        file.close();
                        return false;
                    }
                    allLabels.push_back(static_cast<uint32_t>(lbl));
                    std::vector<uint8_t> packed(packedFeatureBytes, 0);
                    if (!file.read(reinterpret_cast<char*>(packed.data()), packedFeatureBytes)) {
                        eml_debug_2(0, "❌ Read features failed at sample: ", processed, ": ", file_path);
                        delete[] ioBuf;
                        file.close();
                        return false;
                    }
                    auto loc = getChunkLocation(processed);
                    size_t chunkIndex = loc.first;
                    size_t localIndex = loc.second;
                    size_t startElementIndex = localIndex * elementsPerSample;

                    // Unpack features according to quantization_coefficient
                    for (uint16_t j = 0; j < numFeatures; ++j) {
                        uint32_t bitPosition = static_cast<uint32_t>(j) * quantization_coefficient;
                        uint16_t byteIndex = bitPosition / 8;
                        uint8_t bitOffset = bitPosition % 8;

                        uint8_t fv = 0;
                        if (bitOffset + quantization_coefficient <= 8) {
                            // Feature in single byte
                            uint8_t mask = ((1 << quantization_coefficient) - 1) << bitOffset;
                            fv = (packed[byteIndex] & mask) >> bitOffset;
                        } else {
                            // Feature spans two bytes
                            uint8_t bitsInFirstByte = 8 - bitOffset;
                            uint8_t bitsInSecondByte = quantization_coefficient - bitsInFirstByte;
                            uint8_t mask1 = ((1 << bitsInFirstByte) - 1) << bitOffset;
                            uint8_t mask2 = (1 << bitsInSecondByte) - 1;
                            fv = ((packed[byteIndex] & mask1) >> bitOffset) |
                                 ((packed[byteIndex + 1] & mask2) << bitsInFirstByte);
                        }

                        size_t elemIndex = startElementIndex + j;
                        if (elemIndex < sampleChunks[chunkIndex].size()) {
                            sampleChunks[chunkIndex].set(elemIndex, fv);
                        }
                    }
                }
                processed += batchSamples;
            }

            delete[] ioBuf;

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

            // Read binary header
            uint32_t numSamples;
            uint16_t numFeatures;

            if (!file.read(reinterpret_cast<char*>(&numSamples), sizeof(numSamples)) ||
                !file.read(reinterpret_cast<char*>(&numFeatures), sizeof(numFeatures))) {
                eml_debug(0, "❌ Failed to read source header: ", source.file_path);
                file.close();
                return false;
            }

            // Clear current data and initialize parameters
            sampleChunks.clear();
            allLabels.clear();
            bitsPerSample = static_cast<uint16_t>(numFeatures * source.quantization_coefficient);
            quantization_coefficient = source.quantization_coefficient;
            updateSamplesEachChunk();

            // Calculate packed bytes needed for features
            uint32_t totalBits = static_cast<uint32_t>(numFeatures) * quantization_coefficient;
            uint16_t packedFeatureBytes = (totalBits + 7) / 8; // Round up to nearest byte
            size_t sampleDataSize = sizeof(label_type) + packedFeatureBytes; // label + packed features

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
                size_t headerSize = sizeof(uint32_t) + sizeof(uint16_t);
                size_t sampleFilePos = headerSize + (sampleIdx * sampleDataSize);

                // Seek to the sample position
                file.seekg(static_cast<std::streamoff>(sampleFilePos));
                if (!file.good()) {
                    eml_debug_2(2, "⚠️ Failed to seek to sample ", sampleIdx, "position ", sampleFilePos);
                    continue;
                }

                sample_type s;

                // Read label
                if (!file.read(reinterpret_cast<char*>(&s.label), sizeof(s.label))) {
                    eml_debug(2, "⚠️ Failed to read label for sample ", sampleIdx);
                    continue;
                }

                // Read packed features
                s.features.clear();
                s.features.reserve(numFeatures);

                std::vector<uint8_t> packedBuffer(packedFeatureBytes, 0);
                if (!file.read(reinterpret_cast<char*>(packedBuffer.data()), packedFeatureBytes)) {
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
                        uint32_t testNumSamples;
                        uint16_t testNumFeatures;
                        bool headerValid = false;
                        if (testFile.read(reinterpret_cast<char*>(&testNumSamples), sizeof(testNumSamples)) &&
                            testFile.read(reinterpret_cast<char*>(&testNumFeatures), sizeof(testNumFeatures))) {
                            headerValid = (testNumSamples > 0 && testNumFeatures > 0);
                        }
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
                allLabels = other.allLabels; // b_vector has its own copy semantics
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

            // Read current file header to get existing info
            uint32_t currentNumSamples;
            uint16_t numFeatures;
            {
                std::ifstream file(file_path, std::ios::binary);
                if (!file.is_open()) {
                    eml_debug(0, "❌ Failed to open file for adding new data: ", file_path);
                    return deletedLabels;
                }

                if (!file.read(reinterpret_cast<char*>(&currentNumSamples), sizeof(currentNumSamples)) ||
                    !file.read(reinterpret_cast<char*>(&numFeatures), sizeof(numFeatures))) {
                    eml_debug(0, "❌ Failed to read file header: ", file_path);
                    file.close();
                    return deletedLabels;
                }
                file.close();
            }

            // Validate feature count compatibility
            if (!samples.empty() && samples[0].features.size() != numFeatures) {
                eml_debug_2(0, "❌ Feature count mismatch: expected ", numFeatures, ", found ", samples[0].features.size());
                return deletedLabels;
            }

            // Calculate packed bytes needed for features
            uint32_t totalBits = static_cast<uint32_t>(numFeatures) * quantization_coefficient;
            uint16_t packedFeatureBytes = (totalBits + 7) / 8; // Round up to nearest byte
            size_t sampleDataSize = sizeof(label_type) + packedFeatureBytes; // label + packed features
            size_t headerSize = sizeof(uint32_t) + sizeof(uint16_t);

            uint32_t newNumSamples;
            size_t writePosition;

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
                            if (readFile.read(reinterpret_cast<char*>(&label), sizeof(label))) {
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
                    vector<uint8_t> temp_buffer;
                    temp_buffer.reserve(sampleDataSize);

                    std::fstream shiftFile(file_path, std::ios::in | std::ios::out | std::ios::binary);
                    if (shiftFile.is_open()) {
                        // Read and shift each sample
                        for (sample_idx_type i = 0; i < samples_to_keep; i++) {
                            size_t read_pos = headerSize + (samples_to_remove + i) * sampleDataSize;
                            size_t write_pos = headerSize + i * sampleDataSize;

                            shiftFile.seekg(static_cast<std::streamoff>(read_pos));
                            temp_buffer.clear();
                            for (size_t b = 0; b < sampleDataSize; b++) {
                                int byte_val = shiftFile.get();
                                if (byte_val == std::char_traits<char>::eof()) break;
                                temp_buffer.push_back(static_cast<uint8_t>(byte_val));
                            }

                            if (temp_buffer.size() == sampleDataSize) {
                                shiftFile.seekp(static_cast<std::streamoff>(write_pos));
                                shiftFile.write(reinterpret_cast<const char*>(temp_buffer.data()), temp_buffer.size());
                            }
                        }
                        shiftFile.close();
                    }

                    currentNumSamples = samples_to_keep;
                    eml_debug_2(1, "♻️  Removed ", samples_to_remove, " oldest samples, kept ", samples_to_keep);
                }
            }

            size_t newFileSize = headerSize + (static_cast<size_t>(newNumSamples) * sampleDataSize);
            if (newFileSize > MAX_DATASET_BYTES) {
                size_t maxSamplesBySize = (MAX_DATASET_BYTES - headerSize) / sampleDataSize;
                eml_debug(2, "⚠️ Limiting samples by file size to ", maxSamplesBySize);
                newNumSamples = static_cast<uint32_t>(maxSamplesBySize);
            }

            writePosition = headerSize + (currentNumSamples * sampleDataSize);

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
            file.seekp(0);
            file.write(reinterpret_cast<const char*>(&newNumSamples), sizeof(newNumSamples));
            file.write(reinterpret_cast<const char*>(&numFeatures), sizeof(numFeatures));

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
                if (!file.write(reinterpret_cast<const char*>(&sample.label), sizeof(sample.label))) {
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

                if (!file.write(reinterpret_cast<const char*>(packedBuffer.data()), packedFeatureBytes)) {
                    eml_debug_2(0, "❌ Write features failed at sample ", i, ": ", file_path);
                    break;
                }

                written++;
            }

            file.close();

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
