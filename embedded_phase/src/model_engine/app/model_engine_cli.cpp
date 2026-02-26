#include "model_engine.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct CliOptions {
    std::filesystem::path resource_dir = eml::IsoForest::default_resource_dir();
    std::string model_name = "iforest";
    bool evaluate_validation = false;
    std::filesystem::path benign_val_nml_path;
    std::filesystem::path malware_val_nml_path;
    std::string quantized_vector_csv;
    std::filesystem::path quantized_vector_file;
    std::string raw_vector_csv;
    std::filesystem::path raw_vector_file;
    std::filesystem::path json_output_path;
    bool help = false;
};

void print_usage() {
    std::cout
        << "Usage: pe_model_engine_cli [options]\n"
        << "Options:\n"
        << "  --resource-dir <path>      Resource directory (default: embedded_phase/core/models/isolation_forest/resources)\n"
        << "  --model-name <name>        Model prefix (default: iforest)\n"
        << "  --quantized-vector <csv>   Quantized feature values as CSV\n"
        << "  --quantized-file <path>    File containing quantized CSV values\n"
        << "  --raw-vector <csv>         Raw float feature values as CSV\n"
        << "  --raw-file <path>          File containing raw float CSV values\n"
        << "  --evaluate-validation      Evaluate benign/malware validation NML splits\n"
        << "  --benign-val <path>        Override benign validation NML path\n"
        << "  --malware-val <path>       Override malware validation NML path\n"
        << "  --json-output <path>       Write evaluation summary JSON\n"
        << "  --help                     Show this message\n";
}

bool read_text_file(const std::filesystem::path& path, std::string& out) {
    std::ifstream input(path);
    if (!input.is_open()) {
        return false;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    out = buffer.str();
    return true;
}

bool parse_uint8_csv(const std::string& text, std::vector<uint8_t>& out_values) {
    out_values.clear();

    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        int value = 0;
        try {
            value = std::stoi(token);
        } catch (...) {
            return false;
        }
        if (value < 0 || value > 255) {
            return false;
        }
        out_values.push_back(static_cast<uint8_t>(value));
    }

    return !out_values.empty();
}

bool parse_float_csv(const std::string& text, std::vector<float>& out_values) {
    out_values.clear();

    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        try {
            out_values.push_back(std::stof(token));
        } catch (...) {
            return false;
        }
    }

    return !out_values.empty();
}

bool parse_arguments(int argc, char** argv, CliOptions& options, std::string& error) {
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
        } else if (arg == "--quantized-vector") {
            if (!require_value(arg, value)) return false;
            options.quantized_vector_csv = value;
        } else if (arg == "--quantized-file") {
            if (!require_value(arg, value)) return false;
            options.quantized_vector_file = value;
        } else if (arg == "--raw-vector") {
            if (!require_value(arg, value)) return false;
            options.raw_vector_csv = value;
        } else if (arg == "--raw-file") {
            if (!require_value(arg, value)) return false;
            options.raw_vector_file = value;
        } else if (arg == "--evaluate-validation") {
            options.evaluate_validation = true;
        } else if (arg == "--benign-val") {
            if (!require_value(arg, value)) return false;
            options.benign_val_nml_path = value;
        } else if (arg == "--malware-val") {
            if (!require_value(arg, value)) return false;
            options.malware_val_nml_path = value;
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

void print_inference_result(const eml::eml_isolation_result_t& result) {
    std::cout << std::fixed << std::setprecision(6)
              << "success=" << (result.success ? "true" : "false")
              << " score=" << result.anomaly_score
              << " threshold=" << result.threshold
              << " anomaly=" << (result.is_anomaly ? "true" : "false")
              << " status=" << eml::eml_status_to_string(result.status_code)
              << " latency_us=" << result.prediction_time
              << "\n";
}

bool write_evaluation_json(const std::filesystem::path& path,
                           const eml::model_engine::EvaluationSummary& summary,
                           const eml::model_engine::EngineMetadata& metadata,
                           std::string& error) {
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output.is_open()) {
        error = "failed to open json output path: " + path.string();
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
    CliOptions options;
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

    eml::model_engine::IsolationForestModelEngine engine;
    if (!engine.load_model(options.model_name, options.resource_dir, &error)) {
        std::cerr << "Load error: " << error << "\n";
        return 1;
    }

    const auto metadata = engine.metadata();

    if (options.evaluate_validation) {
        const std::filesystem::path benign_path = options.benign_val_nml_path.empty()
            ? (options.resource_dir / (options.model_name + "_ben_val_nml.bin"))
            : options.benign_val_nml_path;

        const std::filesystem::path malware_path = options.malware_val_nml_path.empty()
            ? (options.resource_dir / (options.model_name + "_mal_val_nml.bin"))
            : options.malware_val_nml_path;

        eml::model_engine::EvaluationSummary summary;
        if (!eml::model_engine::evaluate_validation_splits(engine, benign_path, malware_path, summary, &error)) {
            std::cerr << "Evaluation error: " << error << "\n";
            return 1;
        }

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
            if (!write_evaluation_json(options.json_output_path, summary, metadata, error)) {
                std::cerr << "JSON output error: " << error << "\n";
                return 1;
            }
            std::cout << "json_output=" << options.json_output_path << "\n";
        }

        return 0;
    }

    eml::eml_isolation_result_t result;

    if (!options.quantized_vector_csv.empty() || !options.quantized_vector_file.empty()) {
        std::string csv_text = options.quantized_vector_csv;
        if (!options.quantized_vector_file.empty()) {
            if (!read_text_file(options.quantized_vector_file, csv_text)) {
                std::cerr << "Unable to read quantized vector file: " << options.quantized_vector_file << "\n";
                return 1;
            }
        }

        std::vector<uint8_t> features;
        if (!parse_uint8_csv(csv_text, features)) {
            std::cerr << "Invalid quantized vector CSV\n";
            return 1;
        }

        if (!engine.infer_quantized(features, result, &error)) {
            std::cerr << "Inference error: " << error << "\n";
            return 1;
        }

        print_inference_result(result);
        return 0;
    }

    if (!options.raw_vector_csv.empty() || !options.raw_vector_file.empty()) {
        std::string csv_text = options.raw_vector_csv;
        if (!options.raw_vector_file.empty()) {
            if (!read_text_file(options.raw_vector_file, csv_text)) {
                std::cerr << "Unable to read raw vector file: " << options.raw_vector_file << "\n";
                return 1;
            }
        }

        std::vector<float> features;
        if (!parse_float_csv(csv_text, features)) {
            std::cerr << "Invalid raw vector CSV\n";
            return 1;
        }

        if (!engine.infer_raw(features, result, &error)) {
            std::cerr << "Inference error: " << error << "\n";
            return 1;
        }

        print_inference_result(result);
        return 0;
    }

    const std::filesystem::path benign_path = options.resource_dir / (options.model_name + "_ben_val_nml.bin");
    std::vector<uint8_t> matrix;
    size_t samples = 0u;
    if (!eml::model_engine::load_quantized_nml_dataset(
            benign_path,
            metadata.num_features,
            metadata.quantization_bits,
            matrix,
            samples,
            &error)) {
        std::cerr << "Demo fallback failed: " << error << "\n";
        return 1;
    }

    if (samples == 0u) {
        std::cerr << "Demo fallback failed: benign validation dataset is empty\n";
        return 1;
    }

    if (!engine.infer_quantized(matrix.data(), metadata.num_features, result, &error)) {
        std::cerr << "Inference error: " << error << "\n";
        return 1;
    }

    print_inference_result(result);
    return 0;
}
