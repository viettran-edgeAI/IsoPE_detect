#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../../embedded_phase/core/models/isolation_forest/if_model.h"

namespace {

    constexpr double BYTES_PER_MB = 1024.0 * 1024.0;

    struct EvalStats {
        size_t total_files = 0u;
        size_t success = 0u;
        size_t failed = 0u;
        size_t flagged = 0u;
        std::vector<float> scores;
        double total_time_sec = 0.0;
        uint64_t total_bytes = 0u;
    };

    struct DatasetCounts {
        size_t train_samples = 0u;
        size_t validation_samples = 0u;
        size_t validation_benign_samples = 0u;
        size_t validation_malware_samples = 0u;
        size_t test_samples = 0u;
        size_t test_benign_samples = 0u;
        size_t test_malware_samples = 0u;
    };

    struct CurvePoint {
        float x = 0.0f;
        float y = 0.0f;
    };

    struct CliArgs {
        std::filesystem::path repo_root = std::filesystem::path(".");
        std::string model_name = "iforest";
        std::filesystem::path output_txt_path;
        std::filesystem::path benign_test_dir;
        std::filesystem::path malware_test_dir;
        bool no_calibration = false;
    };

    void print_usage(const char* prog) {
        std::cerr << "Usage: " << prog << " [--repo-root PATH] [--model-name NAME] [--output-txt PATH]"
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
            } else if (token == "--output-txt" || token == "--output") {
                if (i + 1 >= argc) {
                    return false;
                }
                args.output_txt_path = std::filesystem::path(argv[++i]);
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
        if (args.output_txt_path.empty()) {
            args.output_txt_path = args.repo_root / "development_phase/reports/if_quantized_cpp_raw_pe_eval.txt";
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

    bool read_nml_sample_count(const std::filesystem::path& nml_path,
                               size_t& out_num_samples) {
        out_num_samples = 0u;

        std::ifstream fin(nml_path, std::ios::binary);
        if (!fin.is_open()) {
            return false;
        }

        uint32_t num_samples = 0u;
        uint16_t num_features = 0u;
        fin.read(reinterpret_cast<char*>(&num_samples), sizeof(num_samples));
        fin.read(reinterpret_cast<char*>(&num_features), sizeof(num_features));
        if (!fin.good() && !fin.eof()) {
            return false;
        }
        (void)num_features;

        out_num_samples = static_cast<size_t>(num_samples);
        return true;
    }

    DatasetCounts collect_dataset_counts(const eml::IsoForest& model,
                                         const EvalStats& benign,
                                         const EvalStats& malware) {
        DatasetCounts counts;
        counts.test_benign_samples = benign.total_files;
        counts.test_malware_samples = malware.total_files;
        counts.test_samples = counts.test_benign_samples + counts.test_malware_samples;

        size_t benign_train = 0u;
        if (read_nml_sample_count(model.base().get_benign_train_nml_path(), benign_train)) {
            counts.train_samples = benign_train;
        }

        size_t benign_val = 0u;
        size_t malware_val = 0u;
        if (read_nml_sample_count(model.base().get_benign_val_nml_path(), benign_val)) {
            counts.validation_benign_samples = benign_val;
        }
        if (read_nml_sample_count(model.base().get_malware_val_nml_path(), malware_val)) {
            counts.validation_malware_samples = malware_val;
        }
        counts.validation_samples = counts.validation_benign_samples + counts.validation_malware_samples;

        return counts;
    }

    std::vector<CurvePoint> compute_precision_recall_curve(const std::vector<float>& benign_scores,
                                                           const std::vector<float>& malware_scores) {
        struct ScoredLabel {
            float score = 0.0f;
            bool is_malware = false;
        };

        std::vector<ScoredLabel> samples;
        samples.reserve(benign_scores.size() + malware_scores.size());
        for (float score : benign_scores) {
            samples.push_back(ScoredLabel{-score, false});
        }
        for (float score : malware_scores) {
            samples.push_back(ScoredLabel{-score, true});
        }

        std::sort(samples.begin(), samples.end(), [](const ScoredLabel& left, const ScoredLabel& right) {
            return left.score > right.score;
        });

        size_t positives = 0u;
        for (const auto& sample : samples) {
            if (sample.is_malware) {
                ++positives;
            }
        }

        std::vector<CurvePoint> curve;
        curve.push_back(CurvePoint{0.0f, 1.0f});
        if (samples.empty() || positives == 0u) {
            curve.push_back(CurvePoint{1.0f, 0.0f});
            return curve;
        }

        size_t tp = 0u;
        size_t fp = 0u;
        for (size_t index = 0; index < samples.size(); ++index) {
            if (samples[index].is_malware) {
                ++tp;
            } else {
                ++fp;
            }

            if (index + 1 == samples.size() || samples[index].score != samples[index + 1u].score) {
                const float recall = static_cast<float>(tp) / static_cast<float>(positives);
                const float precision = (tp + fp) > 0u
                    ? static_cast<float>(tp) / static_cast<float>(tp + fp)
                    : 0.0f;
                curve.push_back(CurvePoint{recall, precision});
            }
        }

        return curve;
    }

    std::vector<CurvePoint> compute_roc_curve(const std::vector<float>& benign_scores,
                                              const std::vector<float>& malware_scores) {
        struct ScoredLabel {
            float score = 0.0f;
            bool is_malware = false;
        };

        std::vector<ScoredLabel> samples;
        samples.reserve(benign_scores.size() + malware_scores.size());
        for (float score : benign_scores) {
            samples.push_back(ScoredLabel{-score, false});
        }
        for (float score : malware_scores) {
            samples.push_back(ScoredLabel{-score, true});
        }

        std::sort(samples.begin(), samples.end(), [](const ScoredLabel& left, const ScoredLabel& right) {
            return left.score > right.score;
        });

        size_t positives = malware_scores.size();
        size_t negatives = benign_scores.size();

        std::vector<CurvePoint> curve;
        curve.push_back(CurvePoint{0.0f, 0.0f});
        if (samples.empty() || positives == 0u || negatives == 0u) {
            curve.push_back(CurvePoint{1.0f, 1.0f});
            return curve;
        }

        size_t tp = 0u;
        size_t fp = 0u;
        for (size_t index = 0; index < samples.size(); ++index) {
            if (samples[index].is_malware) {
                ++tp;
            } else {
                ++fp;
            }

            if (index + 1 == samples.size() || samples[index].score != samples[index + 1u].score) {
                const float fpr = static_cast<float>(fp) / static_cast<float>(negatives);
                const float tpr = static_cast<float>(tp) / static_cast<float>(positives);
                curve.push_back(CurvePoint{fpr, tpr});
            }
        }

        if (curve.back().x < 1.0f || curve.back().y < 1.0f) {
            curve.push_back(CurvePoint{1.0f, 1.0f});
        }

        return curve;
    }

    uint64_t file_size_bytes(const std::filesystem::path& path) {
        std::error_code ec;
        const auto size = std::filesystem::file_size(path, ec);
        return ec ? 0u : static_cast<uint64_t>(size);
    }

    bool write_report_txt(const std::filesystem::path& output_path,
                      const std::filesystem::path& repo_root,
                      const std::string& model_name,
                      float threshold,
                      size_t model_ram_size_bytes,
                      uint64_t model_file_size_bytes,
                      const std::filesystem::path& model_file_path,
                      const DatasetCounts& counts,
                      const EvalStats& benign,
                      const EvalStats& malware,
                      double roc_auc,
                      double average_precision,
                      const std::vector<CurvePoint>& pr_curve,
                      const std::vector<CurvePoint>& roc_curve) {
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
        double combined_files = static_cast<double>(benign.total_files + malware.total_files);
        double combined_time = benign.total_time_sec + malware.total_time_sec;
        double combined_bytes = static_cast<double>(benign.total_bytes + malware.total_bytes);
        const double avg_ms_per_file = combined_files > 0.0 ? (combined_time * 1000.0) / combined_files : 0.0;
        const double avg_ms_per_mb = combined_bytes > 0.0 ? (combined_time * 1000.0) / (combined_bytes / BYTES_PER_MB) : 0.0;

        out << "[metadata]\n";
        out << "repo_root=" << repo_root.string() << "\n";
        out << "model_name=" << model_name << "\n";
        out << "threshold=" << threshold << "\n\n";

        out << "[sample_counts]\n";
        out << "train_samples=" << counts.train_samples << "\n";
        out << "validation_samples=" << counts.validation_samples << "\n";
        out << "validation_benign_samples=" << counts.validation_benign_samples << "\n";
        out << "validation_malware_samples=" << counts.validation_malware_samples << "\n";
        out << "test_samples=" << counts.test_samples << "\n";
        out << "test_benign_samples=" << counts.test_benign_samples << "\n";
        out << "test_malware_samples=" << counts.test_malware_samples << "\n\n";

        out << "[model]\n";
        out << "model_ram_size_bytes=" << model_ram_size_bytes << "\n";
        out << "model_file_size_bytes=" << model_file_size_bytes << "\n";
        out << "model_file_path=" << model_file_path.string() << "\n\n";

        out << "[metrics]\n";
        out << "fpr=" << fpr << "\n";
        out << "tpr=" << tpr << "\n";
        out << "roc_auc=" << roc_auc << "\n";
        out << "average_precision=" << average_precision << "\n";
        out << "prc_auc=" << average_precision << "\n\n";

        out << "[speed]\n";
        out << "benign_total_time_sec=" << benign.total_time_sec << "\n";
        out << "malware_total_time_sec=" << malware.total_time_sec << "\n";
        out << "total_inference_time_sec=" << combined_time << "\n";
        out << "avg_inference_ms_per_file=" << avg_ms_per_file << "\n";
        out << "avg_inference_ms_per_mb=" << avg_ms_per_mb << "\n\n";

        out << "[pr_curve]\n";
        out << "recall,precision\n";
        for (const auto& point : pr_curve) {
            out << point.x << "," << point.y << "\n";
        }
        out << "\n";

        out << "[roc_curve]\n";
        out << "fpr,tpr\n";
        for (const auto& point : roc_curve) {
            out << point.x << "," << point.y << "\n";
        }
        out << "\n";

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
    const double average_precision = static_cast<double>(auc_metrics.average_precision());
    const std::vector<CurvePoint> pr_curve = compute_precision_recall_curve(benign_stats.scores, malware_stats.scores);
    const std::vector<CurvePoint> roc_curve = compute_roc_curve(benign_stats.scores, malware_stats.scores);

    const DatasetCounts sample_counts = collect_dataset_counts(model, benign_stats, malware_stats);
    const size_t model_ram_size_bytes = model.memory_usage();
    const std::filesystem::path model_file_path = model.base().get_model_path();
    const uint64_t model_file_size = file_size_bytes(model_file_path);

    if (!write_report_txt(
            args.output_txt_path,
            args.repo_root,
            args.model_name,
            threshold,
            model_ram_size_bytes,
            model_file_size,
            model_file_path,
            sample_counts,
            benign_stats,
            malware_stats,
            roc_auc,
            average_precision,
            pr_curve,
            roc_curve)) {
        std::cerr << "Failed to write report: " << args.output_txt_path << std::endl;
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
    std::cout << "FPR=" << fpr << ", TPR=" << tpr << ", ROC-AUC=" << roc_auc << ", AP=" << average_precision << "\n";
    std::cout << "Samples train=" << sample_counts.train_samples
              << " validation=" << sample_counts.validation_samples
              << " test=" << sample_counts.test_samples << "\n";
    std::cout << "Model RAM (bytes): " << model_ram_size_bytes << "\n";
    std::cout << "Model file (bytes): " << model_file_size << " [" << model_file_path.string() << "]\n";

    double combined_time = benign_stats.total_time_sec + malware_stats.total_time_sec;
    size_t combined_files = benign_stats.total_files + malware_stats.total_files;
    double combined_bytes = static_cast<double>(benign_stats.total_bytes + malware_stats.total_bytes);
    std::cout << "Total inference time (sec): " << combined_time << "\n";
    std::cout << "Avg time per file (ms): " << (combined_files ? (combined_time * 1000.0) / static_cast<double>(combined_files) : 0.0) << "\n";
    std::cout << "Avg time per MB (ms): " << (combined_bytes ? (combined_time * 1000.0) / (combined_bytes / BYTES_PER_MB) : 0.0) << "\n";

    std::cout << "Report: " << args.output_txt_path << std::endl;

    return 0;
}
