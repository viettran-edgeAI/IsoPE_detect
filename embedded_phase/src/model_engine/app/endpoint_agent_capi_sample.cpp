#include "model_engine_c.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string model_name = "iforest";
    std::filesystem::path resource_dir = "embedded_phase/core/models/isolation_forest/resources";
};

void print_usage(const char* program) {
    std::cout << "Usage: " << program
              << " [--model-name NAME] [--resource-dir PATH]" << std::endl;
}

bool parse_args(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string token = argv[i];
        if (token == "--model-name") {
            if (i + 1 >= argc) {
                return false;
            }
            options.model_name = argv[++i];
        } else if (token == "--resource-dir") {
            if (i + 1 >= argc) {
                return false;
            }
            options.resource_dir = std::filesystem::path(argv[++i]);
        } else if (token == "--help" || token == "-h") {
            return false;
        } else {
            std::cerr << "Unknown argument: " << token << std::endl;
            return false;
        }
    }

    options.resource_dir = std::filesystem::absolute(options.resource_dir);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_args(argc, argv, options)) {
        print_usage(argv[0]);
        return 1;
    }

    pe_model_engine_handle* handle = pe_model_engine_create();
    if (!handle) {
        std::cerr << "Failed to create model-engine handle" << std::endl;
        return 2;
    }

    const int loaded = pe_model_engine_load(
        handle,
        options.model_name.c_str(),
        options.resource_dir.string().c_str());
    if (!loaded) {
        std::cerr << "Load failed: " << pe_model_engine_last_error(handle) << std::endl;
        pe_model_engine_destroy(handle);
        return 3;
    }

    const uint16_t num_features = pe_model_engine_num_features(handle);
    if (num_features == 0u) {
        std::cerr << "Invalid feature count" << std::endl;
        pe_model_engine_destroy(handle);
        return 4;
    }

    std::vector<uint8_t> endpoint_quantized_features(num_features, 0u);
    pe_model_engine_result result{};
    const int ok = pe_model_engine_infer_quantized(
        handle,
        endpoint_quantized_features.data(),
        num_features,
        &result);

    std::cout << "endpoint_agent_capi_sample"
              << " model=" << options.model_name
              << " features=" << num_features
              << " success=" << (ok ? "true" : "false")
              << " anomaly=" << (result.is_anomaly ? "true" : "false")
              << " score=" << result.anomaly_score
              << " threshold=" << result.threshold
              << " status_code=" << result.status_code
              << " latency_us=" << result.prediction_time_us
              << std::endl;

    if (!ok) {
        std::cerr << "Inference failed: " << pe_model_engine_last_error(handle) << std::endl;
    }

    pe_model_engine_destroy(handle);
    return ok ? 0 : 5;
}
