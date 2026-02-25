#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "model_engine.hpp"

namespace {

    std::string get_arg_value(int argc, char** argv, const std::string& key, const std::string& fallback) {
        for (int i = 1; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == key) {
                return std::string(argv[i + 1]);
            }
        }
        return fallback;
    }

    void print_usage() {
        std::cout
            << "Usage: pe_model_engine_cli [--config PATH] [--dp PATH] [--quantized-dir PATH] [--model-name NAME] [--output PATH] [--save-scores PATH]\n";
    }

    std::string infer_model_name(const std::filesystem::path& config_path) {
        const std::string stem = config_path.stem().string();
        const std::string suffix = "_optimized_config";
        if (stem.size() > suffix.size() &&
            stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) == 0) {
            return stem.substr(0, stem.size() - suffix.size());
        }
        return "iforest";
    }

} // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }
    }

    const std::filesystem::path config_path = get_arg_value(
        argc,
        argv,
        "--config",
        "development_phase/results/iforest_optimized_config.json"
    );

    const std::filesystem::path quantized_dir = get_arg_value(
        argc,
        argv,
        "--quantized-dir",
        "embedded_phase/tools/data_quantization/quantized_datasets"
    );

    const std::string model_name = get_arg_value(
        argc,
        argv,
        "--model-name",
        infer_model_name(config_path)
    );

    const std::filesystem::path default_dp_path = quantized_dir / (model_name + "_ben_train_dp.txt");

    const std::filesystem::path dp_path = get_arg_value(
        argc,
        argv,
        "--dp",
        default_dp_path.string()
    );

    const std::filesystem::path output_path = get_arg_value(
        argc,
        argv,
        "--output",
        "embedded_phase/src/model_engine/results/if_evaluation_summary.json"
    );

    const std::string save_scores_path = get_arg_value(argc, argv, "--save-scores", "");
    const bool do_save_scores = !save_scores_path.empty();

    eml::model_engine::DatasetBundlePaths datasets;
    datasets.benign_train = quantized_dir / (model_name + "_ben_train_nml.bin");
    datasets.benign_val = quantized_dir / (model_name + "_ben_val_nml.bin");
    datasets.benign_test = quantized_dir / (model_name + "_ben_test_nml.bin");
    datasets.malware_val = quantized_dir / (model_name + "_mal_val_nml.bin");
    datasets.malware_test = quantized_dir / (model_name + "_mal_test_nml.bin");

    const eml::model_engine::EvaluationSummary summary =
        eml::model_engine::train_and_evaluate(config_path, dp_path, datasets, do_save_scores);

    if (!summary.ok) {
        std::cerr << "model_engine evaluation failed: " << summary.message << "\n";
        return 1;
    }

    std::filesystem::create_directories(output_path.parent_path());
    std::ofstream fout(output_path, std::ios::out | std::ios::trunc);
    if (!fout.is_open()) {
        std::cerr << "failed to open output: " << output_path << "\n";
        return 2;
    }

    fout << "{\n";
    fout << "  \"selected_threshold\": " << summary.selected_threshold << ",\n";
    fout << "  \"embedded\": {\n";
    fout << "    \"validation\": {\"fpr\": " << summary.validation.fpr()
         << ", \"tpr\": " << summary.validation.tpr()
         << ", \"roc_auc\": " << summary.validation.roc_auc()
         << ", \"ap\": " << summary.validation.average_precision() << "},\n";
    fout << "    \"test\": {\"fpr\": " << summary.test.fpr()
         << ", \"tpr\": " << summary.test.tpr()
         << ", \"roc_auc\": " << summary.test.roc_auc()
         << ", \"ap\": " << summary.test.average_precision() << "}\n";
    fout << "  },\n";
    fout << "  \"development\": {\n";
    fout << "    \"validation\": {\"fpr\": " << summary.development.val_fpr
         << ", \"tpr\": " << summary.development.val_tpr
         << ", \"roc_auc\": " << summary.development.val_roc_auc << "},\n";
    fout << "    \"test\": {\"fpr\": " << summary.development.test_fpr
         << ", \"tpr\": " << summary.development.test_tpr
         << ", \"roc_auc\": " << summary.development.test_roc_auc << "}\n";
    fout << "  },\n";
    fout << "  \"delta\": {\n";
    fout << "    \"validation\": {\"fpr\": " << (summary.validation.fpr() - summary.development.val_fpr)
         << ", \"tpr\": " << (summary.validation.tpr() - summary.development.val_tpr)
         << ", \"roc_auc\": " << (summary.validation.roc_auc() - summary.development.val_roc_auc) << "},\n";
    fout << "    \"test\": {\"fpr\": " << (summary.test.fpr() - summary.development.test_fpr)
         << ", \"tpr\": " << (summary.test.tpr() - summary.development.test_tpr)
         << ", \"roc_auc\": " << (summary.test.roc_auc() - summary.development.test_roc_auc) << "}\n";
    fout << "  }\n";
    fout << "}\n";
    fout.close();

    // Optionally write per-sample test scores for ROC curve plotting
    if (do_save_scores) {
        std::filesystem::create_directories(
            std::filesystem::path(save_scores_path).parent_path());
        std::ofstream sfout(save_scores_path, std::ios::out | std::ios::trunc);
        if (sfout.is_open()) {
            sfout << "{\n";
            sfout << "  \"threshold\": " << summary.selected_threshold << ",\n";
            sfout << "  \"benign_test_scores\": [";
            for (size_t i = 0; i < summary.test_benign_scores.size(); ++i) {
                if (i > 0) sfout << ", ";
                sfout << summary.test_benign_scores[i];
            }
            sfout << "],\n";
            sfout << "  \"malware_test_scores\": [";
            for (size_t i = 0; i < summary.test_malware_scores.size(); ++i) {
                if (i > 0) sfout << ", ";
                sfout << summary.test_malware_scores[i];
            }
            sfout << "]\n";
            sfout << "}\n";
            sfout.close();
            std::cout << "  scores: " << save_scores_path << "\n";
        }
    }

    std::cout << "Embedded IF evaluation complete\n";
    std::cout << "  threshold: " << summary.selected_threshold << "\n";
    std::cout << "  val: fpr=" << summary.validation.fpr()
              << ", tpr=" << summary.validation.tpr()
              << ", auc=" << summary.validation.roc_auc() << "\n";
    std::cout << "  test: fpr=" << summary.test.fpr()
              << ", tpr=" << summary.test.tpr()
              << ", auc=" << summary.test.roc_auc() << "\n";
    std::cout << "  report: " << output_path << "\n";

    return 0;
}
