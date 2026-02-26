#include <algorithm>
#include <cstdint>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../../embedded_phase/core/models/isolation_forest/if_model.h"

namespace {

    struct EvalStats {
        size_t total_files = 0u;
        size_t success = 0u;
        size_t failed = 0u;
        size_t flagged = 0u;
        std::vector<float> scores;
        double total_time_sec = 0.0;            // cumulative inference time
        uint64_t total_bytes = 0u;              // sum of file sizes processed
    };

    struct CliArgs {
        std::filesystem::path repo_root = std::filesystem::path(".");
        std::string model_name = "iforest";
        std::filesystem::path output_path;
        std::filesystem::path benign_test_dir;
        std::filesystem::path malware_test_dir;
        bool no_calibration = false; // when true, build_model is called with enable_calibration=false
    };

    void print_usage(const char* prog) {
        std::cerr << "Usage: " << prog << " [--repo-root PATH] [--model-name NAME] [--output PATH]"
                  << " [--benign-test-dir PATH] [--malware-test-dir PATH]" \
                  << " [--no-calibration]" << std::endl;
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
            } else if (token == "--no-calibration") {
                args.no_calibration = true;
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
                               const eml::IsoForest& model) {
        EvalStats stats;
        stats.total_files = files.size();

        for (const auto& pe_path : files) {
            auto start = std::chrono::steady_clock::now();
            const auto infer_result = model.infer_pe_path(pe_path);
            auto end = std::chrono::steady_clock::now();
            stats.total_time_sec += std::chrono::duration<double>(end - start).count();

            std::error_code ec;
            uint64_t fsz = std::filesystem::file_size(pe_path, ec);
            if (!ec) {
                stats.total_bytes += fsz;
            }

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
        out << "    \"roc_auc\": " << roc_auc << ",\n";
        out << "    \"benign_total_time_sec\": " << benign.total_time_sec << ",\n";
        out << "    \"malware_total_time_sec\": " << malware.total_time_sec << ",\n";
        double combined_files = static_cast<double>(benign.total_files + malware.total_files);
        double combined_time = benign.total_time_sec + malware.total_time_sec;
        double combined_bytes = static_cast<double>(benign.total_bytes + malware.total_bytes);
        out << "    \"total_inference_time_sec\": " << combined_time << ",\n";
        out << "    \"avg_time_per_file_sec\": " << (combined_files>0? combined_time/combined_files : 0.0) << ",\n";
        out << "    \"avg_time_per_mb_sec\": " << (combined_bytes>0? combined_time/(combined_bytes/(1024.0*1024.0)) : 0.0) << "\n";
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

    // alway rebuild model at each run 
    model.build_model(!args.no_calibration);
    bool model_ready = model.loaded();
    if (!model_ready) {
        std::cerr << "Failed to prepare IsoForest model (load/train)" << std::endl;
        return 3;
    }

    const float threshold = model.config().decision_threshold;

    const std::vector<std::filesystem::path> benign_files = collect_files(args.benign_test_dir);
    const std::vector<std::filesystem::path> malware_files = collect_files(args.malware_test_dir);
    if (benign_files.empty() || malware_files.empty()) {
        std::cerr << "Test directories are empty or missing" << std::endl;
        return 4;
    }

    const EvalStats benign_stats = evaluate_dataset(benign_files, model);

    const EvalStats malware_stats = evaluate_dataset(malware_files, model);

    eml::eml_metrics_calc_t<eml::problem_type::ISOLATION> auc_metrics;
    auc_metrics.init(eml::eval_metric::ROC_AUC);

    for (float score : benign_stats.scores) {
        auc_metrics.update_score(-score, false);
    }
    for (float score : malware_stats.scores) {
        auc_metrics.update_score(-score, true);
    }

    const double roc_auc = static_cast<double>(auc_metrics.roc_auc());

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

    double combined_time = benign_stats.total_time_sec + malware_stats.total_time_sec;
    size_t combined_files = benign_stats.total_files + malware_stats.total_files;
    double combined_bytes = static_cast<double>(benign_stats.total_bytes + malware_stats.total_bytes);
    std::cout << "Total inference time (sec): " << combined_time << "\n";
    std::cout << "Avg time per file (sec): " << (combined_files? combined_time/combined_files : 0.0) << "\n";
    std::cout << "Avg time per MB (sec): " << (combined_bytes? combined_time/(combined_bytes/(1024.0*1024.0)) : 0.0) << "\n";

    std::cout << "Report: " << args.output_path << std::endl;

    return 0;
}
