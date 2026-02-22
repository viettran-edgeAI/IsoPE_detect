#pragma once

#include "../containers/STL_MCU.h"
#include <type_traits>
#include "eml_debug.h"

// #define EML_STATIC_MODEL   // Uncomment to disable on-device training, saving disk & memory, speed up inference 
// #define DEV_STAGE  // Uncomment for development stage (extra debug info, testing features and test_data)

namespace eml {

    using sample_idx_type = uint32_t;
    using node_idx_type   = uint32_t;

    static constexpr uint8_t         EML_MAX_LABEL_LENGTH    = 32;     // max label length
    static constexpr uint8_t         EML_PATH_BUFFER         = 64;     // buffer for file_path
    static constexpr uint8_t         EML_MAX_LABELS          = 255;    // maximum number of unique labels supported 

    // Enum defines the classification/regression problem.
    enum class problem_type : uint8_t {
        CLASSIFICATION = 0,
        REGRESSION = 1,
        ISOLATION = 2,
        UNKNOWN = 255
        // Add more problem types as needed
    };

    inline std::string problemTypeToString(problem_type type) {
        switch (type) {
            case problem_type::CLASSIFICATION: return "classification";
            case problem_type::REGRESSION: return "regression";
            case problem_type::ISOLATION: return "isolation";
            default: return "unknown";
        }
    }

    inline problem_type problemTypeFromString(const std::string& value) {
        size_t start = value.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) {
            return problem_type::UNKNOWN;
        }
        size_t end = value.find_last_not_of(" \t\r\n");
        std::string lowered = value.substr(start, end - start + 1);
        for (char& c : lowered) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        if (lowered == "classification") return problem_type::CLASSIFICATION;
        if (lowered == "regression") return problem_type::REGRESSION;
        if (lowered == "isolation") return problem_type::ISOLATION;
        return problem_type::UNKNOWN;
    }
    // Helper function to determine desired bits for integer value
    /**
     * @brief Calculate the number of bits required to represent a given unsigned integer (in 1 instruction).
     * @param x The unsigned integer value.
     * @return The number of bits required to represent the value.
     */
    static inline __attribute__((always_inline))
    uint8_t desired_bits(uint32_t x) noexcept
    {
        return x ? (32u - static_cast<uint8_t>(__builtin_clz(x))) : 0u;
    }

    /**
     * @brief Convert a floating-point value to a quantized integer representation with rounding.
     * @param value The floating-point value to be quantized.
     * @param quant_bits The number of bits to use for quantization.
     * @return The quantized integer value.
     */
    template<typename IntType = int16_t>
    static inline __attribute__((always_inline))
    IntType float_to_quantized(float value, uint8_t quant_bits) noexcept
    {
        const float scale = static_cast<float>(1u << quant_bits);
        return static_cast<IntType>(value * scale + (value >= 0.0f ? 0.5f : -0.5f));                                            
    }

    /**
     * @brief Convert a floating-point value to a quantized integer representation by truncation.
     * @param value The floating-point value to be quantized.
     * @param quant_bits The number of bits to use for quantization.
     * @return The quantized integer value.
     * @note This function truncates the value instead of rounding it, but faster.
     */
    template<typename IntType = int16_t>
    static inline __attribute__((always_inline))
    IntType float_to_quantized_trunc(float value, uint8_t quant_bits) noexcept
    {
        return static_cast<IntType>(value * static_cast<float>(1u << quant_bits));                                     
    }

    /**
     * @brief Convert a quantized integer value back to its floating-point representation.
     * @param qvalue The quantized integer value.
     * @param quant_bits The number of bits used for quantization.
     * @return The dequantized floating-point value.
     */
    template<typename IntType = int16_t>
    static inline __attribute__((always_inline))
    float quantized_to_float(IntType qvalue, uint8_t quant_bits) noexcept
    {
        return static_cast<float>(qvalue) * (1.0f / static_cast<float>(1u << quant_bits));
    }


    
}   // namespace eml