#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#endif

#include "model_engine.hpp"
#include "models/isolation_forest/if_scaler_transform.h"
#include "ml/eml_quantize.h"

namespace {

using Clock = std::chrono::high_resolution_clock;

struct Args {
    std::filesystem::path config_path = "development_phase/results/iforest_optimized_config.json";
    std::filesystem::path quantized_dir = "embedded_phase/tools/data_quantization/quantized_datasets";
    std::filesystem::path benign_dir = "datasets/BENIGN_TEST_DATASET";
    std::filesystem::path malware_dir = "datasets/MALWARE_TEST_DATASET";
    std::filesystem::path extractor_bin = "embedded_phase/src/feature_extractor/lief_feature_extractor";
    std::filesystem::path output_md = "embedded_phase/src/model_engine/results/if_benchmark_report.md";
    size_t samples_per_class = 10;
};

struct SampleBenchmark {
    std::string split;
    std::string file_name;
    uint64_t file_size_bytes = 0;
    double extract_ms = 0.0;
    double inference_ms = 0.0;
    double ram_rss_mb = 0.0;
    float score = 0.0f;
    bool anomaly = false;
};

std::string trim(const std::string& text) {
    if (text.empty()) {
        return {};
    }

    size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }

    if (begin >= text.size()) {
        return {};
    }

    size_t end = text.size() - 1;
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end])) != 0) {
        --end;
    }

    return text.substr(begin, end - begin + 1);
}

bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];

        auto require_value = [&](std::filesystem::path& dst) -> bool {
            if (i + 1 >= argc) {
                return false;
            }
            dst = argv[++i];
            return true;
        };

        if (key == "--config") {
            if (!require_value(args.config_path)) return false;
        } else if (key == "--quantized-dir") {
            if (!require_value(args.quantized_dir)) return false;
        } else if (key == "--benign-dir") {
            if (!require_value(args.benign_dir)) return false;
        } else if (key == "--malware-dir") {
            if (!require_value(args.malware_dir)) return false;
        } else if (key == "--extractor-bin") {
            if (!require_value(args.extractor_bin)) return false;
        } else if (key == "--output") {
            if (!require_value(args.output_md)) return false;
        } else if (key == "--samples-per-class") {
            if (i + 1 >= argc) {
                return false;
            }
            args.samples_per_class = static_cast<size_t>(std::max(1, std::atoi(argv[++i])));
        } else if (key == "--help" || key == "-h") {
            return false;
        } else {
            return false;
        }
    }

    return true;
}

void print_usage() {
    std::cout
        << "Usage: pe_model_engine_benchmark_cli [--config PATH] [--quantized-dir PATH] [--benign-dir PATH] [--malware-dir PATH] [--extractor-bin PATH] [--samples-per-class N] [--output PATH]\n";
}

bool read_text_file(const std::filesystem::path& path, std::string& out) {
    std::ifstream fin(path, std::ios::in);
    if (!fin.is_open()) {
        return false;
    }
    out.assign((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    return true;
}

bool parse_number_at(const std::string& text, size_t start_pos, double& out) {
    const char* begin = text.c_str() + start_pos;
    char* end_ptr = nullptr;
    out = std::strtod(begin, &end_ptr);
    return end_ptr != begin;
}

bool extract_scalar_number(const std::string& jsonl, const std::string& key, double& out) {
    const std::string needle = "\"" + key + "\":";
    const size_t key_pos = jsonl.find(needle);
    if (key_pos == std::string::npos) {
        return false;
    }

    size_t pos = key_pos + needle.size();
    while (pos < jsonl.size() && (jsonl[pos] == ' ' || jsonl[pos] == '\t')) {
        ++pos;
    }

    return parse_number_at(jsonl, pos, out);
}

bool extract_parse_ok(const std::string& jsonl, bool& parse_ok) {
    const std::string needle = "\"parse_ok\":";
    const size_t key_pos = jsonl.find(needle);
    if (key_pos == std::string::npos) {
        return false;
    }

    size_t pos = key_pos + needle.size();
    while (pos < jsonl.size() && (jsonl[pos] == ' ' || jsonl[pos] == '\t')) {
        ++pos;
    }

    if (jsonl.compare(pos, 4, "true") == 0) {
        parse_ok = true;
        return true;
    }
    if (jsonl.compare(pos, 5, "false") == 0) {
        parse_ok = false;
        return true;
    }

    return false;
}

bool extract_feature_values(const std::string& jsonl, size_t expected_count, std::vector<float>& out_values) {
    out_values.clear();
    out_values.reserve(expected_count);

    const std::string needle = "\"features\":{";
    const size_t begin_pos = jsonl.find(needle);
    if (begin_pos == std::string::npos) {
        return false;
    }

    size_t cursor = begin_pos + needle.size();
    while (cursor < jsonl.size() && out_values.size() < expected_count) {
        const size_t colon = jsonl.find(':', cursor);
        if (colon == std::string::npos) {
            break;
        }

        size_t value_pos = colon + 1;
        while (value_pos < jsonl.size() && (jsonl[value_pos] == ' ' || jsonl[value_pos] == '\t')) {
            ++value_pos;
        }

        double value = 0.0;
        if (!parse_number_at(jsonl, value_pos, value)) {
            return false;
        }

        out_values.push_back(static_cast<float>(value));

        const size_t comma = jsonl.find(',', value_pos);
        const size_t close = jsonl.find('}', value_pos);
        if (close == std::string::npos) {
            return false;
        }
        if (comma == std::string::npos || close < comma) {
            cursor = close + 1;
            break;
        }

        cursor = comma + 1;
    }

    return out_values.size() == expected_count;
}

std::string quote_for_shell(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 2);
    out.push_back('"');
    for (char ch : text) {
        if (ch == '"' || ch == '\\' || ch == '$' || ch == '`') {
            out.push_back('\\');
        }
        out.push_back(ch);
    }
    out.push_back('"');
    return out;
}

bool run_command_capture_stdout(const std::string& command, std::string& stdout_text, int& exit_code) {
    stdout_text.clear();
    exit_code = -1;

    std::array<char, 4096> buffer{};
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        return false;
    }

    while (true) {
        const size_t n = fread(buffer.data(), 1u, buffer.size(), pipe);
        if (n == 0u) {
            break;
        }
        stdout_text.append(buffer.data(), n);
    }

    const int status = pclose(pipe);
    if (status == -1) {
        return false;
    }

#ifdef _WIN32
    exit_code = status;
#else
    if (WIFEXITED(status) != 0) {
        exit_code = WEXITSTATUS(status);
    } else {
        exit_code = status;
    }
#endif
    return true;
}

bool current_rss_mb(double& out_mb) {
    out_mb = 0.0;

#ifdef _WIN32
    return false;
#else
    std::ifstream fin("/proc/self/status");
    if (!fin.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            const std::string value_part = trim(line.substr(6));
            std::istringstream iss(value_part);
            double kb = 0.0;
            iss >> kb;
            out_mb = kb / 1024.0;
            return true;
        }
    }

    return false;
#endif
}

std::vector<std::filesystem::path> collect_sample_files(const std::filesystem::path& dir, size_t count) {
    std::vector<std::filesystem::path> files;
    if (!std::filesystem::exists(dir)) {
        return files;
    }

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end());
    if (files.size() > count) {
        files.resize(count);
    }
    return files;
}

std::string make_markdown_table(const std::vector<SampleBenchmark>& rows) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out << std::setprecision(3);

    out << "| Split | File | File Size (KB) | Feature Extraction Time (ms) | Inference Time (ms) | RAM Usage (MB RSS) | Score | Verdict |\n";
    out << "|---|---|---:|---:|---:|---:|---:|---|\n";

    for (const auto& row : rows) {
        const double size_kb = static_cast<double>(row.file_size_bytes) / 1024.0;
        out << "| " << row.split
            << " | " << row.file_name
            << " | " << size_kb
            << " | " << row.extract_ms
            << " | " << row.inference_ms
            << " | " << row.ram_rss_mb
            << " | " << row.score
            << " | " << (row.anomaly ? "anomaly" : "benign")
            << " |\n";
    }

    if (!rows.empty()) {
        double sum_extract = 0.0;
        double sum_infer = 0.0;
        double max_ram = 0.0;
        for (const auto& row : rows) {
            sum_extract += row.extract_ms;
            sum_infer += row.inference_ms;
            max_ram = std::max(max_ram, row.ram_rss_mb);
        }

        out << "\n";
        out << "- Average feature extraction time: `" << (sum_extract / static_cast<double>(rows.size())) << " ms`\n";
        out << "- Average inference time (standardization + quantization + inference): `" << (sum_infer / static_cast<double>(rows.size())) << " ms`\n";
        out << "- Peak observed RSS during benchmark loop: `" << max_ram << " MB`\n";
    }

    return out.str();
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        print_usage();
        return 1;
    }

    std::string err;
    eml::model_engine::IsolationForestModelEngine engine;
    const std::filesystem::path dp_path = args.quantized_dir / "benign_train_optimized_dp.txt";

    eml::model_engine::DatasetBundlePaths datasets;
    datasets.benign_train = args.quantized_dir / "benign_train_optimized_nml.bin";
    datasets.benign_val = args.quantized_dir / "benign_val_optimized_nml.bin";
    datasets.benign_test = args.quantized_dir / "benign_test_optimized_nml.bin";
    datasets.malware_val = args.quantized_dir / "malware_val_optimized_nml.bin";
    datasets.malware_test = args.quantized_dir / "malware_test_optimized_nml.bin";

    const eml::model_engine::EvaluationSummary eval_summary =
        eml::model_engine::train_and_evaluate(args.config_path, dp_path, datasets);
    if (!eval_summary.ok) {
        std::cerr << "Failed to compute calibrated threshold: " << eval_summary.message << "\n";
        return 2;
    }
    const float decision_threshold = eval_summary.selected_threshold;

    if (!engine.load_config(args.config_path, dp_path, &err)) {
        std::cerr << "Failed to load model config: " << err << "\n";
        return 3;
    }

    std::vector<uint8_t> train_matrix;
    size_t n_train = 0;
    if (!eml::model_engine::load_quantized_nml_dataset(
            args.quantized_dir / "benign_train_optimized_nml.bin",
            engine.config().num_features,
            engine.config().quantization_bits,
            train_matrix,
            n_train,
            &err)) {
        std::cerr << "Failed to load train dataset: " << err << "\n";
        return 4;
    }

    if (!engine.train_on_quantized_matrix(train_matrix, n_train, &err)) {
        std::cerr << "Failed to train model: " << err << "\n";
        return 5;
    }

    eml::If_scaler_transform scaler;
    if (!scaler.init(engine.config())) {
        std::cerr << "Failed to initialize scaler\n";
        return 6;
    }

    eml::eml_quantizer<eml::problem_type::ISOLATION> quantizer;
    const std::filesystem::path quantizer_path = args.quantized_dir / "benign_train_optimized_qtz.bin";
    if (!quantizer.loadQuantizer(quantizer_path.string().c_str())) {
        std::cerr << "Failed to load quantizer: " << quantizer_path << "\n";
        return 7;
    }

    const auto benign_files = collect_sample_files(args.benign_dir, args.samples_per_class);
    const auto malware_files = collect_sample_files(args.malware_dir, args.samples_per_class);
    if (benign_files.size() < args.samples_per_class || malware_files.size() < args.samples_per_class) {
        std::cerr << "Not enough samples. benign=" << benign_files.size() << ", malware=" << malware_files.size() << "\n";
        return 8;
    }

    std::vector<SampleBenchmark> rows;
    rows.reserve(args.samples_per_class * 2);

    auto process_one = [&](const std::filesystem::path& file_path, const std::string& split_name) -> bool {
        std::string command = quote_for_shell(args.extractor_bin.string()) + " --format jsonl " + quote_for_shell(file_path.string()) + " 2>/dev/null";

        std::string stdout_text;
        int code = -1;
        if (!run_command_capture_stdout(command, stdout_text, code) || code != 0) {
            std::cerr << "Extractor command failed for: " << file_path << "\n";
            return false;
        }

        bool parse_ok = false;
        if (!extract_parse_ok(stdout_text, parse_ok) || !parse_ok) {
            std::cerr << "Extractor parse failed for: " << file_path << "\n";
            return false;
        }

        double extract_ms = 0.0;
        if (!extract_scalar_number(stdout_text, "processing_time_ms", extract_ms)) {
            std::cerr << "Missing processing_time_ms for: " << file_path << "\n";
            return false;
        }

        std::vector<float> raw_features;
        if (!extract_feature_values(stdout_text, engine.config().num_features, raw_features)) {
            std::cerr << "Failed to parse feature vector for: " << file_path << "\n";
            return false;
        }

        std::vector<float> scaled(engine.config().num_features, 0.0f);
        eml::packed_vector<8> quantized(engine.config().num_features, static_cast<uint8_t>(0));
        std::vector<uint8_t> q_values(engine.config().num_features, 0u);

        const auto t0 = Clock::now();
        if (!scaler.transform(raw_features.data(), engine.config().num_features, scaled.data())) {
            std::cerr << "Scaler transform failed for: " << file_path << "\n";
            return false;
        }

        const bool drift = quantizer.quantizeFeatures(scaled.data(), quantized, nullptr, nullptr);
        (void)drift;
        for (uint16_t i = 0; i < engine.config().num_features; ++i) {
            q_values[i] = static_cast<uint8_t>(quantized.get(i));
        }

        const float score = engine.decision_function_quantized(q_values.data(), engine.config().num_features);
        const auto t1 = Clock::now();
        const double inference_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        double rss_mb = 0.0;
        (void)current_rss_mb(rss_mb);

        SampleBenchmark row;
        row.split = split_name;
        row.file_name = file_path.filename().string();
        row.file_size_bytes = static_cast<uint64_t>(std::filesystem::file_size(file_path));
        row.extract_ms = extract_ms;
        row.inference_ms = inference_ms;
        row.ram_rss_mb = rss_mb;
        row.score = score;
        row.anomaly = (score < decision_threshold);
        rows.push_back(row);
        return true;
    };

    for (const auto& path : benign_files) {
        if (!process_one(path, "benign_test")) {
            return 9;
        }
    }

    for (const auto& path : malware_files) {
        if (!process_one(path, "malware_test")) {
            return 10;
        }
    }

    const std::string table_md = make_markdown_table(rows);

    std::filesystem::create_directories(args.output_md.parent_path());
    std::ofstream fout(args.output_md, std::ios::out | std::ios::trunc);
    if (!fout.is_open()) {
        std::cerr << "Failed to open output file: " << args.output_md << "\n";
        return 11;
    }

    fout << "# Embedded Inference Benchmark Report\n\n";
    fout << "Benchmark scope: 10 files from `BENIGN_TEST_DATASET` and 10 files from `MALWARE_TEST_DATASET`.\n\n";
    fout << "Measured stages:\n";
    fout << "- File size\n";
    fout << "- Feature extraction time from `lief_feature_extractor` (`processing_time_ms`)\n";
    fout << "- Model inference time: standardization + quantization + IF scoring\n";
    fout << "- RAM usage (current process RSS in MB)\n\n";
    fout << "Decision threshold used for verdict: `" << decision_threshold << "`\n\n";
    fout << table_md;
    fout.close();

    std::cout << "Embedded benchmark completed\n";
    std::cout << "  output: " << args.output_md << "\n";
    std::cout << "  samples: " << rows.size() << "\n";
    std::cout << table_md;

    return 0;
}
