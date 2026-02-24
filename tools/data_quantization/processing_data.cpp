#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include "../../embedded_phase/core/containers/STL_MCU.h"
#include "../../embedded_phase/core/base/eml_base.h"

#include <utility>
#include <cstring>
#include <cerrno>
#include <filesystem>
#include <cstdint>
#include <cctype>
#include <set>

using namespace eml;

static uint8_t quantization_coefficient = 2; // Coefficient for quantization (bits per feature value)

// Helper functions to calculate derived values from quantization coefficient
static uint16_t getGroupsPerFeature() {
    if (quantization_coefficient >= 8) {
        return 256u;
    }
    return static_cast<uint16_t>(1u << quantization_coefficient); // 2^quantization_coefficient
}

static uint8_t getMaxFeatureValue() {
    if (quantization_coefficient >= 8) {
        return 255u;
    }
    return static_cast<uint8_t>((1u << quantization_coefficient) - 1u); // 2^quantization_coefficient - 1
}

static uint16_t getFeatureMask() {
    if (quantization_coefficient >= 8) {
        return 0xFFu;
    }
    return static_cast<uint16_t>((1u << quantization_coefficient) - 1u); // Bit mask for extracting feature values
}

static uint8_t getFeaturesPerByte() {
    return quantization_coefficient == 0 ? 0 : static_cast<uint8_t>(8 / quantization_coefficient); // How many whole features fit in one byte
}

static uint16_t getPackedFeatureBytes(uint16_t featureCount) {
    uint32_t totalBits = static_cast<uint32_t>(featureCount) * quantization_coefficient;
    return static_cast<uint16_t>((totalBits + 7u) / 8u);
}

struct QuantizationConfig {
    std::string inputPath;
    std::string modelName;
    std::string headerMode = "auto"; // auto|yes|no
    int quantBits = quantization_coefficient;
    bool removeOutliers = true;
    problem_type problemType = problem_type::ISOLATION;
};

static std::string trimWhitespace(const std::string& in) {
    size_t start = in.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = in.find_last_not_of(" \t\r\n");
    return in.substr(start, end - start + 1);
}

static std::string toLowerCopy(std::string value) {
    for (char& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

// Tiny JSON extractor supporting both flat and nested {"value": ...} format
static bool extractValue(const std::string& content, const std::string& key, std::string& out) {
    const std::string pattern = "\"" + key + "\"";
    size_t keyPos = content.find(pattern);
    if (keyPos == std::string::npos) return false;

    size_t colon = content.find(':', keyPos + pattern.size());
    if (colon == std::string::npos) return false;

    size_t valueStart = content.find_first_not_of(" \t\r\n", colon + 1);
    if (valueStart == std::string::npos) return false;

    // Check if value is a nested object
    if (content[valueStart] == '{') {
        // Find the nested "value" field
        size_t objEnd = content.find('}', valueStart);
        if (objEnd == std::string::npos) return false;
        std::string objContent = content.substr(valueStart, objEnd - valueStart + 1);
        
        size_t valueKeyPos = objContent.find("\"value\"");
        if (valueKeyPos == std::string::npos) return false;
        
        size_t valueColon = objContent.find(':', valueKeyPos + 7);
        if (valueColon == std::string::npos) return false;
        
        size_t nestedStart = objContent.find_first_not_of(" \t\r\n", valueColon + 1);
        if (nestedStart == std::string::npos) return false;
        
        if (objContent[nestedStart] == '"') {
            size_t end = objContent.find('"', nestedStart + 1);
            if (end == std::string::npos) return false;
            out = objContent.substr(nestedStart + 1, end - nestedStart - 1);
        } else {
            size_t end = objContent.find_first_of(",}\n\r", nestedStart);
            out = objContent.substr(nestedStart, end - nestedStart);
            out = trimWhitespace(out);
        }
        return true;
    }
    
    // Flat format (backward compatibility)
    if (content[valueStart] == '"') {
        size_t end = content.find('"', valueStart + 1);
        if (end == std::string::npos) return false;
        out = content.substr(valueStart + 1, end - valueStart - 1);
    } else {
        size_t end = content.find_first_of(",}\n\r", valueStart);
        out = content.substr(valueStart, end - valueStart);
        out = trimWhitespace(out);
    }
    return true;
}

static QuantizationConfig loadQuantizationConfig(const std::string& configPath) {
    QuantizationConfig cfg;
    std::ifstream fin(configPath);
    if (!fin) {
        throw std::runtime_error(std::string("Cannot open config file: ") + configPath);
    }

    std::ostringstream ss;
    ss << fin.rdbuf();
    std::string content = ss.str();

    std::string raw;
    if (extractValue(content, "input_path", raw)) {
        cfg.inputPath = trimWhitespace(raw);
    }
    if (extractValue(content, "model_name", raw)) {
        cfg.modelName = trimWhitespace(raw);
    }
    if (extractValue(content, "header", raw)) {
        cfg.headerMode = toLowerCopy(trimWhitespace(raw));
    }
    if (extractValue(content, "quantization_bits", raw)) {
        cfg.quantBits = std::stoi(raw);
    }
    if (extractValue(content, "problem_type", raw)) {
        cfg.problemType = problemTypeFromString(raw);
    }
    if (extractValue(content, "remove_outliers", raw)) {
        std::string lowered = toLowerCopy(raw);
        cfg.removeOutliers = (lowered == "true" || lowered == "1" || lowered == "yes");
    }

    // Validation and defaults fallback
    if (cfg.inputPath.empty()) {
        throw std::runtime_error("Config missing required field: input_path");
    }
    if (cfg.quantBits < 1 || cfg.quantBits > 8) {
        cfg.quantBits = quantization_coefficient;
    }
    if (cfg.problemType == problem_type::UNKNOWN) {
        throw std::runtime_error("Config has invalid problem_type. Supported values: classification, regression, isolation");
    }

    return cfg;
}

// Split a line on commas (naïve; assumes no embedded commas/quotes)
static vector<std::string> split(const std::string &line) {
    vector<std::string> out;
    std::istringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
        // Trim leading/trailing whitespace from cell
        size_t first = cell.find_first_not_of(" \t\r\n");
        if (std::string::npos == first) {
            out.push_back("");
            continue;
        }
        size_t last = cell.find_last_not_of(" \t\r\n");
        out.push_back(cell.substr(first, (last - first + 1)));
    }
    return out;
}

// Structure to store feature statistics for Z-score calculation
struct FeatureStats {
    float mean = 0.0f;
    float stdDev = 0.0f;
    float min = std::numeric_limits<float>::infinity();
    float max = -std::numeric_limits<float>::infinity();
    bool isDiscrete = false;
};

float clipOutlier(float value, float mean, float stdDev, float minVal, float maxVal);

// Structure for building QTZ4 format (per-feature rules, includes per-feature min/max)
class Rf_quantizer{
    uint16_t numFeatures = 0;
    uint16_t groupsPerFeature = 0;
    
    // Feature type definitions
    enum FeatureType : uint8_t { 
        FT_DF = 0,  // Discrete Full: values from 0 to groupsPerFeature-1   
        FT_DC = 1,  // Discrete Custom: specific discrete values defined in discreteValues vector
        FT_CU = 2   // Continuous with quantile-based edges defined in edgesScaled vector
    };
    
    struct FeatureInfo {
        FeatureType type;
        float minValue;
        float maxValue;
        vector<float> discreteValues;    // For FT_DC (sorted unique values)
        vector<uint16_t> edgesScaled;    // For FT_CU (scaled edges)
        int64_t baselineScaled;               // Baseline offset (scaled)
        uint64_t scaleFactor;                 // Per-feature scaling factor
        
        FeatureInfo() : type(FT_DF), minValue(0.0f), maxValue(0.0f), baselineScaled(0), scaleFactor(1) {}
    };
    
    vector<FeatureInfo> features;
    vector<pair<std::string, uint8_t>> labelMapping;
    bool removeOutliers = true; // Whether outlier filtering was applied during quantization
    vector<float> featureMeans;
    vector<float> featureStdDevs;

    static int64_t scaleFloatToInt64(double value, uint64_t scale) {
        long double scaled = static_cast<long double>(value) * static_cast<long double>(scale);
        if (scaled >= static_cast<long double>(std::numeric_limits<int64_t>::max())) {
            return std::numeric_limits<int64_t>::max();
        }
        if (scaled <= static_cast<long double>(std::numeric_limits<int64_t>::min())) {
            return std::numeric_limits<int64_t>::min();
        }
        if (scaled >= 0.0L) {
            scaled += 0.5L;
        } else {
            scaled -= 0.5L;
        }
        return static_cast<int64_t>(scaled);
    }

public:
    // Default constructor
    Rf_quantizer() = default;
    
    // Constructor for building quantizer
    Rf_quantizer(uint16_t featureCount, uint16_t gpf)
        : numFeatures(featureCount), groupsPerFeature(gpf) {
        features.resize(numFeatures);
    }
    
    // Constructor with label mapping
    Rf_quantizer(uint16_t featureCount, uint16_t gpf, const vector<pair<std::string, uint8_t>>& labelMap, bool enableOutlierRemoval = true)
        : numFeatures(featureCount), groupsPerFeature(gpf), labelMapping(labelMap), removeOutliers(enableOutlierRemoval) {
        features.resize(numFeatures);
    }

    // Set outlier statistics for Z-score clipping
    void setOutlierStatistics(const vector<float>& means, const vector<float>& stdDevs) {
        featureMeans = means;
        featureStdDevs = stdDevs;
    }
    
    // Set feature as discrete full range (0..groupsPerFeature-1)
    void setDiscreteFullFeature(uint16_t featureIdx) {
        if (featureIdx < numFeatures) {
            FeatureInfo& info = features[featureIdx];
            info.type = FT_DF;
            info.minValue = 0.0f;
            info.maxValue = (groupsPerFeature > 0) ? static_cast<float>(groupsPerFeature - 1) : 0.0f;
            info.baselineScaled = 0;
            info.scaleFactor = 1;
            info.discreteValues.clear();
            info.edgesScaled.clear();
        }
    }
    
    // Set feature as discrete with custom values
    void setDiscreteCustomFeature(uint16_t featureIdx, const vector<float>& values) {
        if (featureIdx < numFeatures) {
            FeatureInfo& info = features[featureIdx];
            info.type = FT_DC;
            info.discreteValues = values;
            if (!info.discreteValues.empty()) {
                std::sort(info.discreteValues.begin(), info.discreteValues.end());
                info.minValue = info.discreteValues.front();
                info.maxValue = info.discreteValues.back();
            } else {
                info.minValue = 0.0f;
                info.maxValue = 0.0f;
            }
            info.baselineScaled = 0;
            info.scaleFactor = 1;
            info.edgesScaled.clear();
        }
    }
    
    // Set feature as continuous with quantile edges
    void setContinuousFeature(uint16_t featureIdx, const vector<float>& edges, float minValue, float maxValue) {
        if (featureIdx >= numFeatures) return;

        FeatureInfo& info = features[featureIdx];
        info.discreteValues.clear();
        info.edgesScaled.clear();
        info.type = FT_CU;
        info.minValue = minValue;
        info.maxValue = maxValue;

        const double baselineValue = static_cast<double>(minValue);
        double range = static_cast<double>(maxValue) - baselineValue;
        if (!std::isfinite(range) || range < 0.0) {
            range = 0.0;
        }

        long double rawScale = 1.0L;
        if (range > 0.0) {
            rawScale = static_cast<long double>(std::numeric_limits<uint16_t>::max()) / static_cast<long double>(range);
        }
        if (rawScale < 1.0L) {
            rawScale = 1.0L;
        }
        if (rawScale > static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
            rawScale = static_cast<long double>(std::numeric_limits<uint64_t>::max());
        }
        uint64_t scaleValue = static_cast<uint64_t>(rawScale);
        if (scaleValue == 0) {
            scaleValue = 1;
        }

        info.scaleFactor = scaleValue;
        info.baselineScaled = scaleFloatToInt64(baselineValue, scaleValue);

        // Create scaled edge representation relative to baseline
        info.edgesScaled.reserve(edges.size());
        for (float edge : edges) {
            double diff = static_cast<double>(edge) - baselineValue;
            if (diff < 0.0) {
                diff = 0.0;
            }
            long double scaledEdge = static_cast<long double>(diff) * static_cast<long double>(scaleValue);
            if (scaledEdge >= static_cast<long double>(std::numeric_limits<uint16_t>::max())) {
                scaledEdge = static_cast<long double>(std::numeric_limits<uint16_t>::max());
            }
            if (scaledEdge < 0.0L) {
                scaledEdge = 0.0L;
            }
            info.edgesScaled.push_back(static_cast<uint16_t>(scaledEdge + 0.5L));
        }
    }
    
    // Set label mapping
    void setLabelMapping(const vector<pair<std::string, uint8_t>>& labelMap) {
        labelMapping = labelMap;
    }
    
    // Quantize a single feature value
    uint8_t quantizeFeature(uint16_t featureIdx, float value) const {
        if (featureIdx >= numFeatures) return 0;
        
        const FeatureInfo& info = features[featureIdx];
        switch (info.type) {
            case FT_DF:
                // Full discrete range: assume value is already quantized 0..groupsPerFeature-1
                {
                    int intValue = static_cast<int>(value);
                    if (intValue < 0) {
                        intValue = 0;
                    } else if (intValue >= static_cast<int>(groupsPerFeature)) {
                        intValue = static_cast<int>(groupsPerFeature - 1);
                    }
                    return static_cast<uint8_t>(intValue);
                }
                
            case FT_DC:
                // Discrete custom values (supports values outside uint8 range)
                {
                    const size_t count = info.discreteValues.size();
                    if (count == 0) {
                        return 0;
                    }

                    // Fast-path exact/near match
                    for (size_t i = 0; i < count; ++i) {
                        if (std::fabs(info.discreteValues[i] - value) <= 1e-6f) {
                            return static_cast<uint8_t>(i);
                        }
                    }

                    // If unseen (drift), pick nearest known category
                    size_t bestIdx = 0;
                    float bestDist = std::fabs(info.discreteValues[0] - value);
                    for (size_t i = 1; i < count; ++i) {
                        float d = std::fabs(info.discreteValues[i] - value);
                        if (d < bestDist) {
                            bestDist = d;
                            bestIdx = i;
                        }
                    }
                    return static_cast<uint8_t>(bestIdx);
                }
                
            case FT_CU:
                // Continuous unique edges
                {
                    int64_t scaledValue = scaleFloatToInt64(static_cast<double>(value), info.scaleFactor);
                    int64_t adjusted = scaledValue - info.baselineScaled;
                    if (adjusted <= 0) {
                        return 0;
                    }
                    const uint32_t limited = static_cast<uint32_t>(std::min<int64_t>(adjusted, std::numeric_limits<uint32_t>::max()));
                    const uint8_t edgeCount = static_cast<uint8_t>(info.edgesScaled.size());
                    for (uint8_t bin = 0; bin < edgeCount; ++bin) {
                        if (limited < info.edgesScaled[bin]) {
                            return bin;
                        }
                    }
                    return edgeCount;
                }
        }
        return 0;
    }
    
    // Quantize an entire sample
    vector<uint8_t> quantizeSample(const vector<float>& sample) const {
        vector<uint8_t> result;
        result.reserve(numFeatures);
        
        for (uint16_t i = 0; i < numFeatures && i < sample.size(); ++i) {
            result.push_back(quantizeFeature(i, sample[i]));
        }
        
        return result;
    }

    // Quantize with optional training-stat-based outlier clipping.
    vector<uint8_t> quantizeSampleWithTrainingStats(const vector<float>& sample) const {
        vector<uint8_t> result;
        result.reserve(numFeatures);

        for (uint16_t i = 0; i < numFeatures && i < sample.size(); ++i) {
            float value = sample[i];

            if (removeOutliers && i < featureMeans.size() && i < featureStdDevs.size()) {
                value = clipOutlier(
                    value,
                    featureMeans[i],
                    featureStdDevs[i],
                    features[i].minValue,
                    features[i].maxValue
                );
            }

            result.push_back(quantizeFeature(i, value));
        }

        return result;
    }
    
    // Save quantizer to binary format for ESP32 transfer
    void saveQuantizer(const char* filename) const {
        std::ofstream fout(filename, std::ios::binary);
        if (!fout) {
            throw std::runtime_error(std::string("Cannot open quantizer binary file: ") + filename);
        }

        // Write header: magic number "QTZ4" (per-feature rules + per-feature min/max)
        const char magic[4] = {'Q', 'T', 'Z', '4'};
        fout.write(magic, 4);

        // Write basic parameters
        fout.write(reinterpret_cast<const char*>(&numFeatures), sizeof(uint16_t));
        fout.write(reinterpret_cast<const char*>(&groupsPerFeature), sizeof(uint16_t));
        
        uint8_t numLabelsU8 = static_cast<uint8_t>(labelMapping.size());
        fout.write(reinterpret_cast<const char*>(&numLabelsU8), sizeof(uint8_t));
        
        // Write outlier filtering flag
        uint8_t outlierFlag = removeOutliers ? 1 : 0;
        fout.write(reinterpret_cast<const char*>(&outlierFlag), sizeof(uint8_t));

        // Write outlier statistics if enabled
        if (removeOutliers) {
            for (uint16_t i = 0; i < numFeatures; ++i) {
                float mean = (i < featureMeans.size()) ? featureMeans[i] : 0.0f;
                float stdDev = (i < featureStdDevs.size()) ? featureStdDevs[i] : 0.0f;
                fout.write(reinterpret_cast<const char*>(&mean), sizeof(float));
                fout.write(reinterpret_cast<const char*>(&stdDev), sizeof(float));
            }
        }

        // Write label mappings
        for (const auto& mapping : labelMapping) {
            fout.write(reinterpret_cast<const char*>(&mapping.second), sizeof(uint8_t));
            uint8_t labelLen = static_cast<uint8_t>(mapping.first.length());
            fout.write(reinterpret_cast<const char*>(&labelLen), sizeof(uint8_t));
            fout.write(mapping.first.c_str(), labelLen);
        }

        // Write feature definitions
        for (uint16_t i = 0; i < numFeatures; ++i) {
            const FeatureInfo& info = features[i];
            
            uint8_t typeU8 = static_cast<uint8_t>(info.type);
            fout.write(reinterpret_cast<const char*>(&typeU8), sizeof(uint8_t));

            // Store min/max per feature to support drift-aware updates later
            fout.write(reinterpret_cast<const char*>(&info.minValue), sizeof(float));
            fout.write(reinterpret_cast<const char*>(&info.maxValue), sizeof(float));
            
            fout.write(reinterpret_cast<const char*>(&info.baselineScaled), sizeof(int64_t));
            fout.write(reinterpret_cast<const char*>(&info.scaleFactor), sizeof(uint64_t));
            
            switch (info.type) {
                case FT_DF:
                    // No additional data
                    break;
                    
                case FT_DC:
                    {
                        uint8_t count = static_cast<uint8_t>(info.discreteValues.size());
                        fout.write(reinterpret_cast<const char*>(&count), sizeof(uint8_t));
                        for (float val : info.discreteValues) {
                            fout.write(reinterpret_cast<const char*>(&val), sizeof(float));
                        }
                    }
                    break;
                    
                case FT_CU:
                    {
                        uint8_t edgeCount = static_cast<uint8_t>(info.edgesScaled.size());
                        fout.write(reinterpret_cast<const char*>(&edgeCount), sizeof(uint8_t));
                        for (uint16_t edge : info.edgesScaled) {
                            fout.write(reinterpret_cast<const char*>(&edge), sizeof(uint16_t));
                        }
                    }
                    break;
            }
        }

        fout.close();
    }

    // Load quantizer from QTZ4 binary format.
    bool loadQuantizer(const char* filename) {
        std::ifstream fin(filename, std::ios::binary);
        if (!fin) {
            return false;
        }

        char magic[4] = {0, 0, 0, 0};
        fin.read(magic, 4);
        if (!fin || std::strncmp(magic, "QTZ4", 4) != 0) {
            return false;
        }

        uint16_t loadedNumFeatures = 0;
        uint16_t loadedGroupsPerFeature = 0;
        uint8_t numLabelsU8 = 0;
        uint8_t outlierFlag = 0;

        fin.read(reinterpret_cast<char*>(&loadedNumFeatures), sizeof(uint16_t));
        fin.read(reinterpret_cast<char*>(&loadedGroupsPerFeature), sizeof(uint16_t));
        fin.read(reinterpret_cast<char*>(&numLabelsU8), sizeof(uint8_t));
        fin.read(reinterpret_cast<char*>(&outlierFlag), sizeof(uint8_t));
        if (!fin) {
            return false;
        }

        numFeatures = loadedNumFeatures;
        groupsPerFeature = loadedGroupsPerFeature;
        removeOutliers = (outlierFlag != 0);

        features.clear();
        features.resize(numFeatures);
        labelMapping.clear();
        featureMeans.clear();
        featureStdDevs.clear();

        if (removeOutliers) {
            featureMeans.resize(numFeatures, 0.0f);
            featureStdDevs.resize(numFeatures, 0.0f);
            for (uint16_t i = 0; i < numFeatures; ++i) {
                fin.read(reinterpret_cast<char*>(&featureMeans[i]), sizeof(float));
                fin.read(reinterpret_cast<char*>(&featureStdDevs[i]), sizeof(float));
                if (!fin) {
                    return false;
                }
            }
        }

        for (uint8_t i = 0; i < numLabelsU8; ++i) {
            uint8_t id = 0;
            uint8_t labelLen = 0;
            fin.read(reinterpret_cast<char*>(&id), sizeof(uint8_t));
            fin.read(reinterpret_cast<char*>(&labelLen), sizeof(uint8_t));
            if (!fin) {
                return false;
            }

            std::string label;
            label.resize(labelLen);
            if (labelLen > 0) {
                fin.read(&label[0], labelLen);
                if (!fin) {
                    return false;
                }
            }

            labelMapping.push_back({label, id});
        }

        for (uint16_t i = 0; i < numFeatures; ++i) {
            uint8_t typeU8 = 0;
            fin.read(reinterpret_cast<char*>(&typeU8), sizeof(uint8_t));
            fin.read(reinterpret_cast<char*>(&features[i].minValue), sizeof(float));
            fin.read(reinterpret_cast<char*>(&features[i].maxValue), sizeof(float));
            fin.read(reinterpret_cast<char*>(&features[i].baselineScaled), sizeof(int64_t));
            fin.read(reinterpret_cast<char*>(&features[i].scaleFactor), sizeof(uint64_t));
            if (!fin) {
                return false;
            }

            if (typeU8 > static_cast<uint8_t>(FT_CU)) {
                return false;
            }
            features[i].type = static_cast<FeatureType>(typeU8);

            features[i].discreteValues.clear();
            features[i].edgesScaled.clear();

            if (features[i].type == FT_DC) {
                uint8_t count = 0;
                fin.read(reinterpret_cast<char*>(&count), sizeof(uint8_t));
                if (!fin) {
                    return false;
                }
                features[i].discreteValues.resize(count, 0.0f);
                for (uint8_t k = 0; k < count; ++k) {
                    fin.read(reinterpret_cast<char*>(&features[i].discreteValues[k]), sizeof(float));
                    if (!fin) {
                        return false;
                    }
                }
            } else if (features[i].type == FT_CU) {
                uint8_t edgeCount = 0;
                fin.read(reinterpret_cast<char*>(&edgeCount), sizeof(uint8_t));
                if (!fin) {
                    return false;
                }
                features[i].edgesScaled.resize(edgeCount, 0);
                for (uint8_t k = 0; k < edgeCount; ++k) {
                    fin.read(reinterpret_cast<char*>(&features[i].edgesScaled[k]), sizeof(uint16_t));
                    if (!fin) {
                        return false;
                    }
                }
            }
        }

        return true;
    }
    
    // Accessors
    uint16_t getNumFeatures() const { return numFeatures; }
    uint16_t getGroupsPerFeature() const { return groupsPerFeature; }
    const vector<pair<std::string, uint8_t>>& getLabelMapping() const { return labelMapping; }
    
    // Estimate memory usage for QTZ4 format
    size_t estimateQTZ3MemoryUsage() const {
        size_t usage = 0;
        
        // Basic header data
        usage += sizeof(numFeatures) + sizeof(groupsPerFeature);
        
        // Feature references (packed into uint16_t each)
        usage += numFeatures * sizeof(uint16_t);

        // Baseline offsets per feature
        usage += numFeatures * sizeof(int64_t);

        // Scale factors per feature
        usage += numFeatures * sizeof(uint64_t);

        // Outlier statistics
        if (removeOutliers) {
            usage += numFeatures * sizeof(float) * 2;
        }
        
        // Unique edges (estimated)
        size_t uniqueEdgeCount = 0;
        for (const auto& info : features) {
            if (info.type == FT_CU) {
                uniqueEdgeCount += info.edgesScaled.size();
            }
        }
        usage += uniqueEdgeCount * sizeof(uint16_t);
        
        // Discrete custom values (estimated)
        size_t discreteValueCount = 0;
        for (const auto& info : features) {
            if (info.type == FT_DC) {
                discreteValueCount += info.discreteValues.size();
            }
        }
        usage += discreteValueCount * sizeof(float);

        // Per-feature min/max
        usage += numFeatures * sizeof(float) * 2;
        
        // Label mappings (optional)
        for (const auto& mapping : labelMapping) {
            usage += mapping.first.length() + 1; // +1 for null terminator
        }
        
        return usage;
    }
};

// Helper: compute quantile bin edges for a feature
vector<float> computeQuantileBinEdges(vector<float> values, int numBins) {
    vector<float> edges;
    if (values.empty() || numBins < 2) return edges;

    std::sort(values.begin(), values.end());
    
    // Generate numBins - 1 edges
    for (int b = 1; b < numBins; ++b) {
        float q_idx = b * ((float)(values.size() - 1) / (float)numBins);
        int idx = static_cast<int>(q_idx);
        
        // Basic linear interpolation
        float fraction = q_idx - idx;
        float edge_val;
        if (static_cast<size_t>(idx + 1) < values.size()) {
            edge_val = values[idx] + fraction * (values[idx+1] - values[idx]);
        } else {
            edge_val = values.back();
        }
        edges.push_back(edge_val);
    }

    // Post-check for constant edges due to low variance
    bool all_same = true;
    for(size_t i = 1; i < edges.size(); ++i) {
        if (std::abs(edges[i] - edges[0]) > 1e-6f) {
            all_same = false;
            break;
        }
    }

    if (all_same && !edges.empty()) {
        float min_val = values.front();
        float max_val = values.back();
        float range = max_val - min_val;

        // If there's some range, create simple linear bins
        if (range > 1e-6f) {
            for (int b = 1; b < numBins; ++b) {
                edges[b-1] = min_val + b * (range / numBins);
            }
        }
        // If no range, all edges will be the same, which is acceptable.
        // The categorization logic will place all values in the last bin.
    }
    
    return edges;
}

// Replace both instances of std::set usage with this helper function
// Helper function to collect unique values using vector
vector<float> collectUniqueValues(const vector<vector<float>>& data,
                                      int featureIdx,
                                      int numSamples,
                                      size_t maxValues = 0) {
    vector<float> unique;
    if (maxValues > 0) {
        unique.reserve(maxValues + 1);
    }
    
    for (int i = 0; i < numSamples; ++i) {
        float value = data[i][featureIdx];
        
        // Check if value already exists in unique vector
        bool found = false;
        for (size_t j = 0; j < unique.size(); ++j) {
            if (unique[j] == value) {
                found = true;
                break;
            }
        }
        
        // Add value if not found
        if (!found) {
            unique.push_back(value);
            if (maxValues > 0 && unique.size() > maxValues) {
                return unique;
            }
        }
    }
    
    return unique;
}
// Apply Z-score outlier detection and clipping
float clipOutlier(float value, float mean, float stdDev, float minVal, float maxVal) {
    // Suppress unused parameter warnings
    (void)minVal;
    (void)maxVal;
    
    // Only apply if we have enough standard deviation to avoid division by near-zero
    if (stdDev > 1e-6f) {
        float zScore = (value - mean) / stdDev;
        const float THRESHOLD = 3.0f; // Z-score threshold for outliers
        
        if (zScore > THRESHOLD) {
            return mean + THRESHOLD * stdDev; // Upper bound
        } else if (zScore < -THRESHOLD) {
            return mean - THRESHOLD * stdDev; // Lower bound
        }
    }
    return value; // No clipping needed
}

// Helper function to check if a string is likely numeric (including scientific notation)
bool isLikelyNumeric(const std::string& str) {
    if (str.empty()) return false;
    
    // Trim whitespace
    std::string trimmed = str;
    size_t start = trimmed.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return false;
    size_t end = trimmed.find_last_not_of(" \t\r\n");
    trimmed = trimmed.substr(start, end - start + 1);
    
    if (trimmed.empty()) return false;
    
    try {
        // Try to parse as float - this handles integers, decimals, and scientific notation
        std::stof(trimmed);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// Function to automatically detect if CSV has header by analyzing first two rows
bool detectCSVHeader(const char* inputFilePath, problem_type problemType) {
    std::ifstream fin(inputFilePath);
    if (!fin) {
        throw std::runtime_error(std::string("Cannot open input file for header detection: ") + inputFilePath);
    }
    
    std::string firstLine, secondLine;
    if (!std::getline(fin, firstLine) || !std::getline(fin, secondLine)) {
        fin.close();
        // If we can't read two lines, assume no header
        return false;
    }
    fin.close();
    
    if (firstLine.empty() || secondLine.empty()) {
        return false;
    }
    
    auto firstCols = split(firstLine);
    auto secondCols = split(secondLine);
    
    if (firstCols.size() != secondCols.size() || firstCols.empty()) {
        return false;
    }

    const int featureStartColumn = (problemType == problem_type::ISOLATION) ? 0 : 1;
    if (static_cast<int>(firstCols.size()) <= featureStartColumn) {
        return false;
    }
    
    // Check if first row looks like headers (non-numeric) and second row looks like data (numeric)
    int firstRowNumeric = 0;
    int secondRowNumeric = 0;
    int totalCols = firstCols.size();
    
    for (int i = featureStartColumn; i < totalCols; ++i) {
        if (isLikelyNumeric(firstCols[i])) {
            firstRowNumeric++;
        }
        if (isLikelyNumeric(secondCols[i])) {
            secondRowNumeric++;
        }
    }
    
    int featureCols = totalCols - featureStartColumn;
    if (featureCols <= 0) {
        return false;
    }
    
    // Heuristic: if first row has significantly fewer numeric values than second row,
    // and second row is mostly numeric, then first row is likely a header
    float firstRowNumericRatio = static_cast<float>(firstRowNumeric) / featureCols;
    float secondRowNumericRatio = static_cast<float>(secondRowNumeric) / featureCols;
    
    // Header detection criteria:
    // 1. Second row should be mostly numeric (>= 80% of feature columns)
    // 2. First row should be significantly less numeric than second row
    bool hasHeader = (secondRowNumericRatio >= 0.8f) && 
                     (firstRowNumericRatio < 0.5f) && 
                     (secondRowNumericRatio - firstRowNumericRatio >= 0.3f);
    
    return hasHeader;
}

// Forward declaration
uint8_t getNormalizedLabel(const std::string& originalLabel, const vector<pair<std::string, uint8_t>>& labelMapping);

Rf_quantizer quantizeCSVFeatures(const char* inputFilePath,
                                const char* outputFilePath,
                                int groupsPerFeature,
                                const vector<pair<std::string, uint8_t>>& labelMapping,
                                bool skipHeader = false,
                                bool enableOutlierClipping = true,
                                problem_type problemType = problem_type::ISOLATION){
    if (groupsPerFeature < 1) {
        throw std::runtime_error("groupsPerFeature must be >= 1");
    }

    std::ifstream fin(inputFilePath);
    if (!fin) {
        throw std::runtime_error(std::string("Cannot open input file: ") + inputFilePath);
    }

    // Read first line to determine structure
    std::string firstLine;
    std::getline(fin, firstLine);
    auto cols = split(firstLine);
    int n_cols = (int)cols.size();
    if ((problemType == problem_type::ISOLATION && n_cols < 1) ||
        (problemType != problem_type::ISOLATION && n_cols < 2)) {
        fin.close();
        throw std::runtime_error(problemType == problem_type::ISOLATION
            ? "Input CSV needs at least one feature column"
            : "Input CSV needs at least one label + one feature");
    }

    const int featureStartColumn = (problemType == problem_type::ISOLATION) ? 0 : 1;
    int n_feats = n_cols - featureStartColumn;
    
    // First pass: collect data and calculate statistics for Z-score
    vector<FeatureStats> featureStats(n_feats);
    vector<std::string> labels;
    vector<vector<float>> data;
    
    // If skipHeader=false, process the first line as data
    // If skipHeader=true, skip the first line (treat it as header)
    bool processFirstLine = !skipHeader;
    
    if (processFirstLine) {
        // Process the first line as data
        if ((int)cols.size() == n_cols) {
            if (problemType != problem_type::ISOLATION) {
                labels.push_back(cols[0]);
            }
            vector<float> feats;
            feats.reserve(n_feats);
            
            for (int j = featureStartColumn; j < n_cols; ++j) {
                try {
                    float val = std::stof(cols[j]);
                    feats.push_back(val);
                    
                    int idx = (int)feats.size() - 1;
                    featureStats[idx].min = std::min(featureStats[idx].min, val);
                    featureStats[idx].max = std::max(featureStats[idx].max, val);
                    featureStats[idx].mean += val;
                } catch (const std::exception& e) {
                    // Handle conversion error with a default value
                    feats.push_back(0.0f);
                }
            }
            
            data.push_back(std::move(feats));
        }
    }
    
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        auto cells = split(line);
        if ((int)cells.size() != n_cols) {
            // Silently skip rows with incorrect column count
            continue; 
        }
        
        if (problemType != problem_type::ISOLATION) {
            labels.push_back(cells[0]);
        }
        vector<float> feats;
        feats.reserve(n_feats);
        
        for (int j = featureStartColumn; j < n_cols; ++j) {
            try {
                float val = std::stof(cells[j]);
                feats.push_back(val);
                
                int idx = (int)feats.size() - 1;
                featureStats[idx].min = std::min(featureStats[idx].min, val);
                featureStats[idx].max = std::max(featureStats[idx].max, val);
                featureStats[idx].mean += val;
            } catch (const std::exception& e) {
                // Handle conversion error with a default value
                feats.push_back(0.0f);
            }
        }
        
        data.push_back(std::move(feats));
    }
    
    fin.close();
    
    int n_samples = data.size();
    if (n_samples == 0) {
        throw std::runtime_error("No data rows found in file");
    }
    
    // Calculate mean and standard deviation
    for (int j = 0; j < n_feats; ++j) {
        if (n_samples > 0) featureStats[j].mean /= n_samples;
    }
    
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_feats; ++j) {
            float diff = data[i][j] - featureStats[j].mean;
            featureStats[j].stdDev += diff * diff;
        }
    }
    
    for (int j = 0; j < n_feats; ++j) {
        if (n_samples > 0) featureStats[j].stdDev = std::sqrt(featureStats[j].stdDev / n_samples);
    }
    
    // Initial check for discrete features to avoid clipping them
    for (int j = 0; j < n_feats; ++j) {
        vector<float> distinct = collectUniqueValues(data, j, n_samples, static_cast<size_t>(groupsPerFeature));
        if (distinct.size() <= static_cast<size_t>(groupsPerFeature)) {
            featureStats[j].isDiscrete = true;
        }
    }

    // Apply Z-score outlier clipping to continuous features (optional)
    if (enableOutlierClipping) {
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_feats; ++j) {
                if (!featureStats[j].isDiscrete) {
                    data[i][j] = clipOutlier(
                        data[i][j], 
                        featureStats[j].mean, 
                        featureStats[j].stdDev,
                        featureStats[j].min,
                        featureStats[j].max
                    );
                }
            }
        }
    }

    // Update feature min/max values after clipping
    for (int j = 0; j < n_feats; ++j) {
        float updatedMin = std::numeric_limits<float>::infinity();
        float updatedMax = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < n_samples; ++i) {
            float v = data[i][j];
            if (v < updatedMin) {
                updatedMin = v;
            }
            if (v > updatedMax) {
                updatedMax = v;
            }
        }
        featureStats[j].min = updatedMin;
        featureStats[j].max = updatedMax;
    }

    // Setup quantizer with QTZ3 format and outlier removal flag
    Rf_quantizer ctg(n_feats, groupsPerFeature, labelMapping, enableOutlierClipping);
    
    // Pass outlier statistics to quantizer
    if (enableOutlierClipping) {
        vector<float> means(n_feats);
        vector<float> stdDevs(n_feats);
        for (int j = 0; j < n_feats; ++j) {
            means[j] = featureStats[j].mean;
            stdDevs[j] = featureStats[j].stdDev;
        }
        ctg.setOutlierStatistics(means, stdDevs);
    }

    for (int j = 0; j < n_feats; ++j) {
    vector<float> distinct_after_clip = collectUniqueValues(data, j, n_samples, static_cast<size_t>(groupsPerFeature));
        
        if (distinct_after_clip.size() <= static_cast<size_t>(groupsPerFeature)) {
            // Check if it's a full discrete range (0..groupsPerFeature-1)
            bool isFullRange = (distinct_after_clip.size() == static_cast<size_t>(groupsPerFeature));
            if (isFullRange) {
                std::sort(distinct_after_clip.begin(), distinct_after_clip.end());
                for (size_t k = 0; k < distinct_after_clip.size(); ++k) {
                    if (static_cast<int>(distinct_after_clip[k]) != static_cast<int>(k)) {
                        isFullRange = false;
                        break;
                    }
                }
            }
            
            if (isFullRange) {
                ctg.setDiscreteFullFeature(j);
            } else {
                ctg.setDiscreteCustomFeature(j, distinct_after_clip);
            }
        } else {
            // Treat as continuous feature with quantile bins
            vector<float> values;
            values.reserve(n_samples);
            for (int i = 0; i < n_samples; ++i) {
                values.push_back(data[i][j]);
            }
            vector<float> edges = computeQuantileBinEdges(values, static_cast<int>(groupsPerFeature));
            ctg.setContinuousFeature(j, edges, featureStats[j].min, featureStats[j].max);
        }
    }
    
    // Encode into uint8_t categories
    vector<vector<uint8_t>> encoded(n_samples, vector<uint8_t>(n_feats));
    for (int i = 0; i < n_samples; ++i) {
        encoded[i] = ctg.quantizeSample(data[i]);
    }
    
    // Write output CSV (quantized data without headers)
    std::ofstream fout(outputFilePath);
    if (!fout) {
        throw std::runtime_error(std::string("Cannot open output file: ") + outputFilePath);
    }
    
    // Write quantized data (no headers needed for quantized output)
    for (int i = 0; i < n_samples; ++i) {
        if (problemType == problem_type::ISOLATION) {
            for (int j = 0; j < n_feats; ++j) {
                if (j > 0) {
                    fout << ",";
                }
                fout << static_cast<int>(encoded[i][j]);
            }
        } else {
            uint8_t normalizedLabel = getNormalizedLabel(labels[i], labelMapping);
            fout << static_cast<int>(normalizedLabel);
            for (int j = 0; j < n_feats; ++j) {
                fout << "," << static_cast<int>(encoded[i][j]);
            }
        }
        fout << "\n";
    }
    fout.close();
    
    return ctg;
}

void quantizeCSVFeaturesWithLoadedQuantizer(const char* inputFilePath,
                                            const char* outputFilePath,
                                            const Rf_quantizer& quantizer,
                                            const vector<pair<std::string, uint8_t>>& labelMapping,
                                            bool skipHeader = false,
                                            problem_type problemType = problem_type::ISOLATION) {
    std::ifstream fin(inputFilePath);
    if (!fin) {
        throw std::runtime_error(std::string("Cannot open input file: ") + inputFilePath);
    }

    std::string firstLine;
    std::getline(fin, firstLine);
    auto cols = split(firstLine);
    int n_cols = static_cast<int>(cols.size());
    if ((problemType == problem_type::ISOLATION && n_cols < 1) ||
        (problemType != problem_type::ISOLATION && n_cols < 2)) {
        fin.close();
        throw std::runtime_error(problemType == problem_type::ISOLATION
            ? "Input CSV needs at least one feature column"
            : "Input CSV needs at least one label + one feature");
    }

    const int featureStartColumn = (problemType == problem_type::ISOLATION) ? 0 : 1;
    const int n_feats = n_cols - featureStartColumn;
    if (n_feats != static_cast<int>(quantizer.getNumFeatures())) {
        fin.close();
        throw std::runtime_error("Feature count mismatch between input CSV and loaded quantizer");
    }

    vector<std::string> labels;
    vector<vector<float>> data;

    bool processFirstLine = !skipHeader;
    if (processFirstLine && static_cast<int>(cols.size()) == n_cols) {
        if (problemType != problem_type::ISOLATION) {
            labels.push_back(cols[0]);
        }
        vector<float> feats;
        feats.reserve(n_feats);
        for (int j = featureStartColumn; j < n_cols; ++j) {
            try {
                feats.push_back(std::stof(cols[j]));
            } catch (...) {
                feats.push_back(0.0f);
            }
        }
        data.push_back(std::move(feats));
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        auto cells = split(line);
        if (static_cast<int>(cells.size()) != n_cols) {
            continue;
        }
        if (problemType != problem_type::ISOLATION) {
            labels.push_back(cells[0]);
        }

        vector<float> feats;
        feats.reserve(n_feats);
        for (int j = featureStartColumn; j < n_cols; ++j) {
            try {
                feats.push_back(std::stof(cells[j]));
            } catch (...) {
                feats.push_back(0.0f);
            }
        }
        data.push_back(std::move(feats));
    }
    fin.close();

    const int n_samples = static_cast<int>(data.size());
    if (n_samples == 0) {
        throw std::runtime_error("No data rows found in file");
    }

    std::ofstream fout(outputFilePath);
    if (!fout) {
        throw std::runtime_error(std::string("Cannot open output file: ") + outputFilePath);
    }

    for (int i = 0; i < n_samples; ++i) {
        const vector<uint8_t> encoded = quantizer.quantizeSampleWithTrainingStats(data[i]);

        if (problemType == problem_type::ISOLATION) {
            for (int j = 0; j < n_feats; ++j) {
                if (j > 0) {
                    fout << ",";
                }
                fout << static_cast<int>(encoded[j]);
            }
        } else {
            uint8_t normalizedLabel = getNormalizedLabel(labels[i], labelMapping);
            fout << static_cast<int>(normalizedLabel);
            for (int j = 0; j < n_feats; ++j) {
                fout << "," << static_cast<int>(encoded[j]);
            }
        }
        fout << "\n";
    }

    fout.close();
}

// Dataset scanner to check features and collect labels
struct DatasetInfo {
    int numFeatures;
    int numSamples;
    vector<pair<std::string, uint8_t>> labelMapping; // original -> normalized
    problem_type problemType;
    
    DatasetInfo() : numFeatures(0), numSamples(0), 
                   problemType(problem_type::ISOLATION) {}
};

// Scan dataset to get info and create label mapping
DatasetInfo scanDataset(const char* inputFilePath, problem_type configuredProblemType) {
    DatasetInfo info;
    info.problemType = configuredProblemType;
    std::ifstream fin(inputFilePath);
    if (!fin) {
        throw std::runtime_error(std::string("Cannot open input file for scanning: ") + inputFilePath);
    }

    // Detect if CSV has header
    fin.close(); // Close and reopen for header detection
    bool hasHeader = detectCSVHeader(inputFilePath, configuredProblemType);
    
    fin.open(inputFilePath);
    if (!fin) {
        throw std::runtime_error(std::string("Cannot reopen input file for scanning: ") + inputFilePath);
    }

    // Read first line to get number of columns
    std::string firstLine;
    std::getline(fin, firstLine);
    auto cols = split(firstLine);
    int n_cols = (int)cols.size();
    if ((configuredProblemType == problem_type::ISOLATION && n_cols < 1) ||
        (configuredProblemType != problem_type::ISOLATION && n_cols < 2)) {
        fin.close();
        throw std::runtime_error(configuredProblemType == problem_type::ISOLATION
            ? "Input CSV needs at least one feature column"
            : "Input CSV needs at least one label + one feature");
    }

    const int featureStartColumn = (configuredProblemType == problem_type::ISOLATION) ? 0 : 1;
    info.numFeatures = n_cols - featureStartColumn;
    
    // Collect unique labels
    vector<std::string> uniqueLabels;
    std::string line;
    int lineCount = 0;
    
    // If first line is not a header, process it as data
    if (!hasHeader) {
        auto cells = split(firstLine);
        if ((int)cells.size() == n_cols) {
            lineCount++;
            if (configuredProblemType != problem_type::ISOLATION) {
                std::string label = cells[0].c_str();
                uniqueLabels.push_back(label);
            }
        }
    }
    
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        auto cells = split(line);
        if ((int)cells.size() != n_cols) continue; // Skip malformed rows
        
        lineCount++;
        if (configuredProblemType != problem_type::ISOLATION) {
            std::string label = cells[0].c_str();

            bool found = false;
            for (size_t i = 0; i < uniqueLabels.size(); ++i) {
                if (uniqueLabels[i] == label) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                uniqueLabels.push_back(label);
            }
        }
    }
    
    fin.close();
    info.numSamples = lineCount;
    if (configuredProblemType != problem_type::ISOLATION) {
        uniqueLabels.sort();
        info.labelMapping.reserve(uniqueLabels.size());
        for (size_t i = 0; i < uniqueLabels.size(); ++i) {
            info.labelMapping.push_back({uniqueLabels[i], static_cast<uint8_t>(i)});
        }
    }

    return info;
}

// Get normalized label from mapping
uint8_t getNormalizedLabel(const std::string& originalLabel, const vector<pair<std::string, uint8_t>>& labelMapping) {
    for (const auto& mapping : labelMapping) {
        if (mapping.first == originalLabel) {
            return mapping.second;
        }
    }
    return 0; // Default if not found (shouldn't happen after scanning)
}

// Dataset parameter synthesis for ESP32 transfer
// Output format changed from CSV to TXT for better readability
void generateDatasetParams(std::string path, const DatasetInfo& datasetInfo, const char* outputFile = "dataset_params.txt") {
    std::ofstream fout(outputFile);
    if (!fout) {
        throw std::runtime_error(std::string("Cannot create dataset params file: ") + outputFile);
    }
    
    vector<uint32_t> samplesPerLabel;
    uint32_t actualTotalSamples = 0;

    if (datasetInfo.problemType == problem_type::ISOLATION) {
        samplesPerLabel = vector<uint32_t>(1, 0);
        std::ifstream csvFile(path);
        if (csvFile) {
            std::string line;
            while (std::getline(csvFile, line)) {
                if (line.empty()) continue;
                auto cells = split(line);
                if ((int)cells.size() != datasetInfo.numFeatures) continue;
                samplesPerLabel[0]++;
                actualTotalSamples++;
            }
            csvFile.close();
        }
    } else {
        samplesPerLabel = vector<uint32_t>(datasetInfo.labelMapping.size(), 0);
        std::ifstream csvFile(path);
        if (csvFile) {
            std::string line;
            while (std::getline(csvFile, line)) {
                if (line.empty()) continue;
                auto cells = split(line);
                if (cells.empty()) continue;

                try {
                    int labelValue = std::stoi(cells[0]);
                    if (labelValue >= 0 && static_cast<size_t>(labelValue) < samplesPerLabel.size()) {
                        samplesPerLabel[labelValue]++;
                        actualTotalSamples++;
                    }
                } catch (...) {
                    // Skip invalid labels
                }
            }
            csvFile.close();
        }
    }

    uint16_t actualFeatures = static_cast<uint16_t>(datasetInfo.numFeatures);
    
    // Write TXT header
    fout << "parameter,value\n";
    
    // Write core parameters
    fout << "quantization_coefficient," << static_cast<int>(quantization_coefficient) << "\n";
    fout << "num_features," << actualFeatures << "\n";
    fout << "num_samples," << actualTotalSamples << "\n";
    fout << "num_labels," << samplesPerLabel.size() << "\n";
    
    // Write samples per label
    for (size_t i = 0; i < samplesPerLabel.size(); ++i) {
        fout << "samples_label_" << i << "," << samplesPerLabel[i] << "\n";
    }
    
    // Write problem type (classification/regression) at the end
    fout << "problem_type," << problemTypeToString(datasetInfo.problemType) << "\n";
    
    fout.close();
}

// Add binary conversion structures and functions before main()
// ESP32-compatible sample structure for binary conversion
struct ESP32_Sample {
    vector<uint8_t> features;
    uint8_t label;
    
    ESP32_Sample() : label(0) {}
    
    bool validate() const {
        for (uint8_t feature : features) {
            if (feature > getMaxFeatureValue()) {
                return false;
            }
        }
        return true;
    }
};

// Load CSV data for binary conversion
vector<ESP32_Sample> loadCSVForBinary(const std::string& csvFilename, uint16_t expectedFeatures, problem_type problemType) {
    std::ifstream file(csvFilename);
    if (!file) {
        throw std::runtime_error("Cannot open CSV file: " + csvFilename);
    }
    
    vector<ESP32_Sample> samples;
    std::string line;
    int lineCount = 0;
    int errorCount = 0;
    int validSamples = 0;
    
    while (std::getline(file, line)) {
        lineCount++;
        
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        
        if (line.empty()) continue;
        
        auto fields = split(line);
        
        ESP32_Sample sample;
        sample.features.reserve(expectedFeatures);
        
        try {
            bool parseError = false;

            if (problemType == problem_type::ISOLATION) {
                if (fields.size() != static_cast<size_t>(expectedFeatures)) {
                    errorCount++;
                    continue;
                }
                sample.label = 0;
                for (size_t i = 0; i < fields.size(); ++i) {
                    int featureValue = std::stoi(fields[i]);
                    if (featureValue < 0 || featureValue > getMaxFeatureValue()) {
                        parseError = true;
                        break;
                    }
                    sample.features.push_back(static_cast<uint8_t>(featureValue));
                }
            } else {
                if (fields.size() != static_cast<size_t>(expectedFeatures + 1)) {
                    errorCount++;
                    continue;
                }

                int labelValue = std::stoi(fields[0]);
                if (labelValue < 0 || labelValue > 255) {
                    errorCount++;
                    continue;
                }
                sample.label = static_cast<uint8_t>(labelValue);

                for (size_t i = 1; i < fields.size(); ++i) {
                    int featureValue = std::stoi(fields[i]);

                    if (featureValue < 0 || featureValue > getMaxFeatureValue()) {
                        parseError = true;
                        break;
                    }

                    sample.features.push_back(static_cast<uint8_t>(featureValue));
                }
            }

            if (parseError) {
                errorCount++;
                continue;
            }
            
            // Final validation
            if (!sample.validate()) {
                errorCount++;
                continue;
            }
            
            samples.push_back(sample);
            validSamples++;
            
        } catch (const std::exception& e) {
            errorCount++;
            continue;
        }
    }
    
    file.close();
    
    if (errorCount > 0) {
        std::cout << "⚠️  Warning: " << errorCount << " invalid rows skipped during loading\n";
    }
    
    return samples;
}

// Convert samples to ESP32-compatible binary format
void saveBinaryDataset(const vector<ESP32_Sample>& samples, 
                      const std::string& binaryFilename, 
                      uint16_t numFeatures) {
    std::ofstream file(binaryFilename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot create binary file: " + binaryFilename);
    }
    
    // Write ESP32-compatible header
    uint32_t numSamples = static_cast<uint32_t>(samples.size());
    uint16_t numFeatures_header = numFeatures;
    
    // Write header (exactly like ESP32 Rf_data)
    file.write(reinterpret_cast<const char*>(&numSamples), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&numFeatures_header), sizeof(uint16_t));
    
    // Calculate packed bytes needed for features
    uint16_t packedFeatureBytes = getPackedFeatureBytes(numFeatures);
    
    // Write samples (exactly like ESP32 Rf_data)
    for (size_t i = 0; i < samples.size(); ++i) {
        const ESP32_Sample& sample = samples[i];
        
        // Write label
        file.write(reinterpret_cast<const char*>(&sample.label), sizeof(uint8_t));
        
        // Pack and write features
        vector<uint8_t> packedBuffer(packedFeatureBytes, 0);
        
        uint16_t mask = getFeatureMask();
        size_t bitPosition = 0;
        
        // Pack features into bytes
        for (size_t f = 0; f < sample.features.size(); ++f) {
            uint16_t featureValue = static_cast<uint16_t>(sample.features[f]) & mask;
            size_t byteIndex = bitPosition / 8u;
            uint8_t bitOffset = static_cast<uint8_t>(bitPosition % 8u);
            uint16_t shifted = static_cast<uint16_t>(featureValue << bitOffset);
            
            if (byteIndex < packedBuffer.size()) {
                packedBuffer[byteIndex] |= static_cast<uint8_t>(shifted & 0xFFu);
            }
            
            if (bitOffset + quantization_coefficient > 8 && (byteIndex + 1) < packedBuffer.size()) {
                packedBuffer[byteIndex + 1] |= static_cast<uint8_t>(shifted >> 8);
            }
            
            bitPosition += quantization_coefficient;
        }
        
        file.write(reinterpret_cast<const char*>(packedBuffer.data()), packedFeatureBytes);
    }
    
    file.close();
    
    // Verify file size
    std::ifstream check(binaryFilename, std::ios::binary | std::ios::ate);
    if (check) {
        size_t fileSize = check.tellg();
        check.close();
        
        size_t expectedSize = 6 + samples.size() * (1 + packedFeatureBytes);
        
        if (fileSize != expectedSize) {
            std::cout << "❌ Binary file size mismatch: " << fileSize << " bytes (expected " << expectedSize << " bytes)\n";
        }
    }
}

// Integrated CSV to binary conversion function
void convertCSVToBinary(const std::string& inputCSV, const std::string& outputBinary, uint16_t numFeatures, problem_type problemType) {
    // Load CSV data
    auto samples = loadCSVForBinary(inputCSV, numFeatures, problemType);
    
    if (samples.empty()) {
        throw std::runtime_error("No valid samples found in CSV file");
    }
    
    // Convert to binary format
    saveBinaryDataset(samples, outputBinary, numFeatures);
}

int main(int argc, char* argv[]) {
    try {
        std::string configPath = "quantization_config.json";
        std::string existingQuantizerPath;
        
        // First pass: Find config path and check for help
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-c" || arg == "--config") {
                if (i + 1 < argc) configPath = argv[++i];
            } else if (arg == "-h" || arg == "--help") {
                std::cout << "Usage: " << argv[0] << " [options]\n\n";
                std::cout << "Options:\n";
                std::cout << "  -c, --config <path>       Path to quantization_config.json (default: quantization_config.json)\n";
                std::cout << "  -ip, --input_path <path>  Path to input CSV dataset\n";
                std::cout << "  -mn, --model_name <name>  Unified name for result files (e.g., 'iris')\n";
                std::cout << "  -hd, --header <auto|yes|no> Whether CSV has header (default: auto)\n";
                std::cout << "  -pt, --problem_type <classification|regression|isolation> Problem mode (default: isolation)\n";
                std::cout << "  -qb, --quantization_bits <1-8> Bits per feature (default: 2)\n";
                std::cout << "  -ro, --remove_outliers <bool> Enable Z-score outlier clipping (default: true)\n";
                std::cout << "  -qp, --quantizer_path <path> Reuse existing quantizer (transform mode; skips fitting/saving quantizer)\n";
                std::cout << "\nExample usage:\n";
                std::cout << "  " << argv[0] << " -ip dataset.csv -qb 4 -mn my_model\n";
                std::cout << "  " << argv[0] << " -ip val.csv -mn iforest_ben_val -qp quantized_datasets/iforest_qtz.bin\n";
                std::cout << "  " << argv[0] << " -c config.json -pt isolation -qb 2\n";
                return 0;
            }
        }

        // Load base config
        QuantizationConfig config;
        try {
            config = loadQuantizationConfig(configPath);
        } catch (...) {
            // If config doesn't exist, we'll rely on CLI args or defaults
            // But we need at least input_path from CLI if config fails
        }

        // Second pass: Override fields from CLI
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-ip" || arg == "--input_path") {
                if (i + 1 < argc) config.inputPath = argv[++i];
            } else if (arg == "-mn" || arg == "--model_name") {
                if (i + 1 < argc) config.modelName = argv[++i];
            } else if (arg == "-hd" || arg == "--header") {
                if (i + 1 < argc) config.headerMode = toLowerCopy(argv[++i]);
            } else if (arg == "-pt" || arg == "--problem_type") {
                if (i + 1 < argc) config.problemType = problemTypeFromString(argv[++i]);
            } else if (arg == "-qb" || arg == "--quantization_bits") {
                if (i + 1 < argc) config.quantBits = std::stoi(argv[++i]);
            } else if (arg == "-ro" || arg == "--remove_outliers") {
                if (i + 1 < argc) {
                    std::string val = toLowerCopy(argv[++i]);
                    config.removeOutliers = (val == "true" || val == "1" || val == "yes");
                }
            } else if (arg == "-qp" || arg == "--quantizer_path") {
                if (i + 1 < argc) {
                    existingQuantizerPath = argv[++i];
                }
            }
        }

        if (config.inputPath.empty()) {
            std::cerr << "Error: No input_path specified (use -ip or -c config.json)\n";
            return 1;
        }
        if (config.problemType == problem_type::UNKNOWN) {
            std::cerr << "Error: Invalid problem_type. Use classification, regression, or isolation\n";
            return 1;
        }

        // Apply configuration to globals used elsewhere
        quantization_coefficient = static_cast<uint8_t>(config.quantBits);

        bool skipHeader = false;
        bool headerSpecified = false;
        std::string headerLower = toLowerCopy(config.headerMode);
        if (headerLower == "yes") {
            skipHeader = true;
            headerSpecified = true;
        } else if (headerLower == "no") {
            skipHeader = false;
            headerSpecified = true;
        }

        const std::string inputFile = config.inputPath;
        const std::string modelName = config.modelName;

        // Generate output file names based on inputFile or modelName
        std::string baseName;
        
        // extract base name if modelName are empty or "auto"
        if (modelName.empty() || modelName == "auto") {
            // Extract base name from input file
            std::string inputName(inputFile);
            size_t slash = inputName.find_last_of("/\\");
            if (slash != std::string::npos) {
                inputName = inputName.substr(slash + 1);
            }
            baseName = inputName;
            size_t dot = baseName.find_last_of('.');
            if (dot != std::string::npos) {
                baseName = baseName.substr(0, dot);
            }
        } else {
            // Use the provided model name
            baseName = modelName;
        }

        std::filesystem::path configDir = std::filesystem::path(configPath).parent_path();
        if (configDir.empty()) {
            configDir = std::filesystem::current_path();
        }
        std::filesystem::path resultDirPath = configDir / "quantized_datasets";
        if (!std::filesystem::exists(resultDirPath)) {
            std::filesystem::create_directories(resultDirPath);
        }
        std::string quantizerFile = (resultDirPath / (baseName + "_qtz.bin")).string();
        std::string dataParamsFile = (resultDirPath / (baseName + "_dp.txt")).string();
        std::string normalizedFile = (resultDirPath / (baseName + "_nml.csv")).string();
        std::string binaryFile = (resultDirPath / (baseName + "_nml.bin")).string();

        // Step 1: Scan dataset to get info and create label mapping
        DatasetInfo datasetInfo = scanDataset(inputFile.c_str(), config.problemType);

        // Auto-detect header if not explicitly specified by user
        if (!headerSpecified) {
            bool hasHeader = detectCSVHeader(inputFile.c_str(), config.problemType);
            skipHeader = hasHeader; // Skip header if detected
            (void)hasHeader; // Header detection is silent for clean CLI output
        }

        uint16_t resolvedNumFeatures = static_cast<uint16_t>(datasetInfo.numFeatures);

        if (!existingQuantizerPath.empty()) {
            Rf_quantizer loadedQuantizer;
            if (!loadedQuantizer.loadQuantizer(existingQuantizerPath.c_str())) {
                throw std::runtime_error(std::string("Failed to load quantizer: ") + existingQuantizerPath);
            }
            if (loadedQuantizer.getNumFeatures() != static_cast<uint16_t>(datasetInfo.numFeatures)) {
                throw std::runtime_error("Loaded quantizer feature count does not match input dataset");
            }

            quantizeCSVFeaturesWithLoadedQuantizer(
                inputFile.c_str(),
                normalizedFile.c_str(),
                loadedQuantizer,
                datasetInfo.labelMapping,
                skipHeader,
                datasetInfo.problemType
            );

            resolvedNumFeatures = loadedQuantizer.getNumFeatures();
        } else {
            // Step 2: Fit quantizer on this dataset and quantize (fit mode)
            Rf_quantizer fittedQuantizer = quantizeCSVFeatures(inputFile.c_str(),
                                   normalizedFile.c_str(),
                                   getGroupsPerFeature(),
                                   datasetInfo.labelMapping,
                                   skipHeader,
                                   config.removeOutliers,
                                   datasetInfo.problemType);

            // Save quantizer for deployment/runtime transfer
            fittedQuantizer.saveQuantizer(quantizerFile.c_str());
            resolvedNumFeatures = fittedQuantizer.getNumFeatures();
        }

        // Step 3: Generate dataset parameters TXT for ESP32 transfer
        generateDatasetParams(normalizedFile, datasetInfo, dataParamsFile.c_str());

        // Step 4: Convert CSV to binary format
        convertCSVToBinary(normalizedFile, binaryFile, resolvedNumFeatures, datasetInfo.problemType);

        // Calculate file size compression ratio
        size_t inputFileSize = 0;
        size_t outputFileSize = 0;
        std::ifstream inputFileCheck(inputFile, std::ios::binary | std::ios::ate);
        if (inputFileCheck) {
            inputFileSize = inputFileCheck.tellg();
            inputFileCheck.close();
        }
        std::ifstream outputFileCheck(binaryFile, std::ios::binary | std::ios::ate);
        if (outputFileCheck) {
            outputFileSize = outputFileCheck.tellg();
            outputFileCheck.close();
        }
        
        std::cout << "\n=== Processing Complete ===\n";
        std::cout << "✅ Dataset quantized and compressed:\n";
        size_t numLabels = datasetInfo.problemType == problem_type::ISOLATION ? 1 : datasetInfo.labelMapping.size();
        std::cout << "   📊 Samples: " << datasetInfo.numSamples << " | Features: " << resolvedNumFeatures
              << " | Labels: " << numLabels << "\n";
        std::cout << "   🧩 Problem type: " << problemTypeToString(datasetInfo.problemType) << "\n";
        std::cout << "   🗜️  Quantization: " << static_cast<int>(quantization_coefficient) << std::endl;
        if (!existingQuantizerPath.empty()) {
            std::cout << "   🔁 Quantizer mode: transform-only (reused " << existingQuantizerPath << ")\n";
        } else {
            std::cout << "   🧪 Quantizer mode: fit-and-transform\n";
        }
        
        if (inputFileSize > 0 && outputFileSize > 0) {
            float compressionRatio = static_cast<float>(inputFileSize) / static_cast<float>(outputFileSize);
            float compressionPercent = (1.0f - static_cast<float>(outputFileSize) / static_cast<float>(inputFileSize)) * 100.0f;
            std::cout << "   📉 Compression: " << std::fixed << std::setprecision(2) 
                      << compressionRatio << "x (" << compressionPercent << "% size reduction)\n";
            std::cout << "      Input: " << inputFileSize << " bytes → Output: " << outputFileSize << " bytes\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}