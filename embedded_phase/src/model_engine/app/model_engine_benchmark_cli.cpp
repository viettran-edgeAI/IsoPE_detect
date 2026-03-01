#include "model_engine.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

struct BenchmarkOptions {
    std::filesystem::path resource_dir = eml::IsoForest::default_resource_dir();
    std::string model_name = "iforest";
    std::filesystem::path benign_test_nml_path;
    std::filesystem::path malware_test_nml_path;
    std::filesystem::path json_output_path;
    bool help = false;
};

void print_usage() {
    std::cout
        << "Usage: pe_model_engine_benchmark_cli [options]\n"
        << "Options:\n"
        << "  --resource-dir <path>   Resource directory\n"
        << "  --model-name <name>     Model prefix (default: iforest)\n"
        << "  --benign-test <path>    Benign test NML file\n"
        << "  --malware-test <path>   Malware test NML file\n"
        << "  --json-output <path>    Optional JSON output file\n"
        << "  --help                  Show this message\n";
}

bool parse_arguments(int argc, char** argv, BenchmarkOptions& options, std::string& error) {
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        const auto require_value = [&](const std::string& option, std::string& value) -> bool {
            if (index + 1 >= argc) {
                error = "missing value for option: " + option;
                return false;
            }
            value = argv[++index];
            return true;
        };

        if (arg == "--help") {
            options.help = true;
            return true;
        }

        std::string value;
        if (arg == "--resource-dir") {
            if (!require_value(arg, value)) return false;
            options.resource_dir = value;
        } else if (arg == "--model-name") {
            if (!require_value(arg, value)) return false;
            options.model_name = value;
        } else if (arg == "--benign-test") {
            if (!require_value(arg, value)) return false;
            options.benign_test_nml_path = value;
        } else if (arg == "--malware-test") {
            if (!require_value(arg, value)) return false;
            options.malware_test_nml_path = value;
        } else if (arg == "--json-output") {
            if (!require_value(arg, value)) return false;
            options.json_output_path = value;
        } else {
            error = "unknown option: " + arg;
            return false;
        }
    }

    return true;
}

bool write_json(const std::filesystem::path& output_path,
                const eml::model_engine::EvaluationSummary& summary,
                const eml::model_engine::EngineMetadata& metadata,
                std::string& error) {
    std::ofstream output(output_path, std::ios::out | std::ios::trunc);
    if (!output.is_open()) {
        error = "failed to open json output path: " + output_path.string();
        return false;
    }

    output << std::fixed << std::setprecision(6)
           << "{\n"
            << "  \"model_name\": \"" << metadata.model_name << "\",\n"
            << "  \"resource_dir\": \"" << metadata.resource_dir.string() << "\",\n"
            << "  \"num_features\": " << metadata.num_features << ",\n"
            << "  \"quantization_bits\": " << static_cast<int>(metadata.quantization_bits) << ",\n"
           << "  \"threshold\": " << summary.threshold << ",\n"
           << "  \"fpr\": " << summary.fpr << ",\n"
           << "  \"tpr\": " << summary.tpr << ",\n"
           << "  \"roc_auc\": " << summary.roc_auc << ",\n"
           << "  \"average_precision\": " << summary.average_precision << ",\n"
           << "  \"benign_samples\": " << summary.benign_samples << ",\n"
           << "  \"malware_samples\": " << summary.malware_samples << ",\n"
            << "  \"tp\": " << summary.true_positive << ",\n"
            << "  \"fp\": " << summary.false_positive << ",\n"
            << "  \"tn\": " << summary.true_negative << ",\n"
            << "  \"fn\": " << summary.false_negative << ",\n"
            << "  \"status\": \"" << eml::eml_status_to_string(summary.status_code) << "\",\n"
            << "  \"success\": " << (summary.success ? "true" : "false") << "\n"
           << "}\n";

    if (!output.good()) {
        error = "failed to write json output";
        return false;
    }

    return true;
}

}  // namespace

int main(int argc, char** argv) {
    BenchmarkOptions options;
    std::string error;

    if (!parse_arguments(argc, argv, options, error)) {
        std::cerr << "Argument error: " << error << "\n";
        print_usage();
        return 1;
    }

    if (options.help) {
        print_usage();
        return 0;
    }

    const std::filesystem::path benign_path = options.benign_test_nml_path.empty()
        ? (options.resource_dir / (options.model_name + "_ben_test_nml.bin"))
        : options.benign_test_nml_path;

    const std::filesystem::path malware_path = options.malware_test_nml_path.empty()
        ? (options.resource_dir / (options.model_name + "_mal_test_nml.bin"))
        : options.malware_test_nml_path;

    eml::model_engine::IsolationForestModelEngine engine;
    if (!engine.load_model(options.model_name, options.resource_dir, &error)) {
        std::cerr << "Load error: " << error << "\n";
        return 1;
    }

    eml::model_engine::EvaluationSummary summary;
    if (!eml::model_engine::evaluate_test_splits(engine, benign_path, malware_path, summary, &error)) {
        std::cerr << "Evaluation error: " << error << "\n";
        return 1;
    }

    const auto metadata = engine.metadata();

    std::cout << std::fixed << std::setprecision(6)
              << "model=" << metadata.model_name
              << " features=" << metadata.num_features
              << " qbits=" << static_cast<int>(metadata.quantization_bits)
              << " threshold=" << summary.threshold
              << " fpr=" << summary.fpr
              << " tpr=" << summary.tpr
              << " roc_auc=" << summary.roc_auc
              << " ap=" << summary.average_precision
              << " tp=" << summary.true_positive
              << " fp=" << summary.false_positive
              << " tn=" << summary.true_negative
              << " fn=" << summary.false_negative
              << "\n";

    if (!options.json_output_path.empty()) {
        if (!write_json(options.json_output_path, summary, metadata, error)) {
            std::cerr << "JSON output error: " << error << "\n";
            return 1;
        }
        std::cout << "json_output=" << options.json_output_path << "\n";
    }

    return 0;
}
