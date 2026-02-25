#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../embedded_phase/core/models/isolation_forest/if_model.h"
#include "../../embedded_phase/src/feature_extractor/include/extractor/extractor.hpp"

namespace {

    struct EvalStats {
        size_t total_files = 0u;
        size_t success = 0u;
        size_t failed = 0u;
        size_t flagged = 0u;
        std::vector<float> scores;
    };

    struct CliArgs {
        std::filesystem::path repo_root = std::filesystem::path(".");
        std::string model_name = "iforest";
        std::filesystem::path output_path;
        std::filesystem::path benign_test_dir;
        std::filesystem::path malware_test_dir;
    };

    void print_usage(const char* prog) {
        std::cerr << "Usage: " << prog << " [--repo-root PATH] [--model-name NAME] [--output PATH]"
                  << " [--benign-test-dir PATH] [--malware-test-dir PATH]" << std::endl;
    }

    bool parse_args(int argc, char** argv, CliArgs& args) {
        for (int i = 1; i < argc; ++i) {
            const std::string token = argv[i];
            if (token == "--repo-root") {
                if (i + 1 >= argc) {
                    return false;
                }
                args.repo_root = std::filesystem::path(argv[++i]);
            } else if (token == "--model-name") {
                if (i + 1 >= argc) {
                    return false;
                }
                args.model_name = argv[++i];
            } else if (token == "--output") {
                if (i + 1 >= argc) {
                    return false;
                }
                args.output_path = std::filesystem::path(argv[++i]);
            } else if (token == "--benign-test-dir") {
                if (i + 1 >= argc) {
                    return false;
                }
                args.benign_test_dir = std::filesystem::path(argv[++i]);
            } else if (token == "--malware-test-dir") {
                if (i + 1 >= argc) {
                    return false;
                }
                args.malware_test_dir = std::filesystem::path(argv[++i]);
            } else if (token == "--help" || token == "-h") {
                return false;
            } else {
                std::cerr << "Unknown argument: " << token << std::endl;
                return false;
            }
        }

        args.repo_root = std::filesystem::absolute(args.repo_root);
        if (args.output_path.empty()) {
            args.output_path = args.repo_root / "development_phase/reports/if_quantized_cpp_raw_pe_eval.json";
        }
        if (args.benign_test_dir.empty()) {
            args.benign_test_dir = args.repo_root / "datasets/BENIGN_TEST_DATASET";
        }
        if (args.malware_test_dir.empty()) {
            args.malware_test_dir = args.repo_root / "datasets/MALWARE_TEST_DATASET";
        }

        return true;
    }

    std::vector<std::filesystem::path> collect_files(const std::filesystem::path& root_dir) {
        std::vector<std::filesystem::path> out;
        if (!std::filesystem::exists(root_dir) || !std::filesystem::is_directory(root_dir)) {
            return out;
        }

        for (const auto& entry : std::filesystem::recursive_directory_iterator(root_dir)) {
            if (entry.is_regular_file()) {
                out.push_back(entry.path());
            }
        }

        std::sort(out.begin(), out.end());
        return out;
    }

    EvalStats evaluate_dataset(const std::vector<std::filesystem::path>& files,
                               const eml::IsoForest& model,
                               float threshold) {
        EvalStats stats;
        stats.total_files = files.size();

        for (const auto& pe_path : files) {
            const auto infer_result = model.infer_pe_path(pe_path, threshold);
            if (!infer_result.success) {
                ++stats.failed;
                continue;
            }

            ++stats.success;
            if (infer_result.is_anomaly) {
                ++stats.flagged;
            }
            stats.scores.push_back(infer_result.anomaly_score);
        }

        return stats;
    }

    bool write_report(const std::filesystem::path& output_path,
                      const std::filesystem::path& repo_root,
                      const std::string& model_name,
                      float threshold,
                      const EvalStats& benign,
                      const EvalStats& malware,
                      double roc_auc) {
        const float fpr = benign.success > 0u
            ? static_cast<float>(benign.flagged) / static_cast<float>(benign.success)
            : 0.0f;
        const float tpr = malware.success > 0u
            ? static_cast<float>(malware.flagged) / static_cast<float>(malware.success)
            : 0.0f;

        std::filesystem::create_directories(output_path.parent_path());
        std::ofstream out(output_path, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            return false;
        }

        out << std::fixed << std::setprecision(6);
        out << "{\n";
        out << "  \"repo_root\": \"" << repo_root.string() << "\",\n";
        out << "  \"model_name\": \"" << model_name << "\",\n";
        out << "  \"threshold\": " << threshold << ",\n";
        out << "  \"raw_pe_eval\": {\n";
        out << "    \"benign_total\": " << benign.total_files << ",\n";
        out << "    \"benign_success\": " << benign.success << ",\n";
        out << "    \"benign_failed\": " << benign.failed << ",\n";
        out << "    \"benign_flagged\": " << benign.flagged << ",\n";
        out << "    \"malware_total\": " << malware.total_files << ",\n";
        out << "    \"malware_success\": " << malware.success << ",\n";
        out << "    \"malware_failed\": " << malware.failed << ",\n";
        out << "    \"malware_flagged\": " << malware.flagged << ",\n";
        out << "    \"fpr\": " << fpr << ",\n";
        out << "    \"tpr\": " << tpr << ",\n";
        out << "    \"roc_auc\": " << roc_auc << "\n";
        out << "  }\n";
        out << "}\n";

        return true;
    }

} // namespace

int main(int argc, char** argv) {
    CliArgs args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    const std::filesystem::path resource_dir =
        args.repo_root / "embedded_phase/core/models/isolation_forest/resources";

    eml::IsoForest model;
    if (!model.init(args.model_name, resource_dir)) {
        std::cerr << "Failed to initialize IsoForest resources" << std::endl;
        return 2;
    }

    bool model_ready = model.load();
    if (!model_ready) {
        model_ready = model.train_from_quantized_dataset();
    }
    if (!model_ready) {
        std::cerr << "Failed to prepare IsoForest model (load/train)" << std::endl;
        return 3;
    }

    const float threshold = model.config().decision_threshold;
    extractor::PEExtractor pe_extractor;
    const std::vector<std::string> compiled_names = extractor::compiled_feature_names();
    std::unordered_map<std::string, size_t> compiled_index;
    compiled_index.reserve(compiled_names.size());
    for (size_t i = 0; i < compiled_names.size(); ++i) {
        compiled_index.emplace(compiled_names[i], i);
    }

    model.set_extract_callback(
        [&pe_extractor, &compiled_index](const std::filesystem::path& pe_path,
                                         const eml::vector<std::string>& feature_names,
                                         eml::vector<float>& out_features) -> bool {
            const extractor::ExtractionReport report = pe_extractor.extract_with_metadata(pe_path.string());
            if (!report.metadata.parse_ok) {
                return false;
            }

            const auto& values = report.feature_vector.values;
            out_features.resize(feature_names.size(), 0.0f);
            for (size_t i = 0; i < feature_names.size(); ++i) {
                const auto it = compiled_index.find(feature_names[i]);
                if (it == compiled_index.end()) {
                    return false;
                }
                const size_t idx = it->second;
                if (idx >= values.size()) {
                    return false;
                }
                out_features[i] = static_cast<float>(values[idx]);
            }
            return true;
        }
    );

    const std::vector<std::filesystem::path> benign_files = collect_files(args.benign_test_dir);
    const std::vector<std::filesystem::path> malware_files = collect_files(args.malware_test_dir);
    if (benign_files.empty() || malware_files.empty()) {
        std::cerr << "Test directories are empty or missing" << std::endl;
        return 4;
    }

    const EvalStats benign_stats = evaluate_dataset(
        benign_files,
        model,
        threshold
    );

    const EvalStats malware_stats = evaluate_dataset(
        malware_files,
        model,
        threshold
    );

    std::vector<float> y_scores;
    std::vector<uint8_t> y_true;
    y_scores.reserve(benign_stats.scores.size() + malware_stats.scores.size());
    y_true.reserve(benign_stats.scores.size() + malware_stats.scores.size());

    for (float score : benign_stats.scores) {
        y_scores.push_back(-score);
        y_true.push_back(0u);
    }
    for (float score : malware_stats.scores) {
        y_scores.push_back(-score);
        y_true.push_back(1u);
    }

    const double roc_auc = eml::if_roc_auc(y_scores, y_true);

    if (!write_report(
            args.output_path,
            args.repo_root,
            args.model_name,
            threshold,
            benign_stats,
            malware_stats,
            roc_auc)) {
        std::cerr << "Failed to write report: " << args.output_path << std::endl;
        return 5;
    }

    const float fpr = benign_stats.success > 0u
        ? static_cast<float>(benign_stats.flagged) / static_cast<float>(benign_stats.success)
        : 0.0f;
    const float tpr = malware_stats.success > 0u
        ? static_cast<float>(malware_stats.flagged) / static_cast<float>(malware_stats.success)
        : 0.0f;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Raw-PE quantized evaluation complete\n";
    std::cout << "Threshold: " << threshold << "\n";
    std::cout << "Benign: success=" << benign_stats.success << ", failed=" << benign_stats.failed << "\n";
    std::cout << "Malware: success=" << malware_stats.success << ", failed=" << malware_stats.failed << "\n";
    std::cout << "FPR=" << fpr << ", TPR=" << tpr << ", ROC-AUC=" << roc_auc << "\n";
    std::cout << "Report: " << args.output_path << std::endl;

    return 0;
}
