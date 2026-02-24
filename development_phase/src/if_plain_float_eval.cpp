#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "models/isolation_forest/if_plain_float.h"

namespace fs = std::filesystem;

struct CsvDataset {
    std::vector<std::string> header;
    std::vector<float> matrix;
    size_t rows = 0u;
    uint16_t cols = 0u;
};

struct PlainEvalConfig {
    uint16_t n_estimators = 200u;
    float max_samples = 1.0f;
    float max_features = 1.0f;
    bool bootstrap = false;
    uint32_t random_state = 42u;
    float target_fpr = 0.05f;
};

static std::string read_text_file(const fs::path& file_path) {
    std::ifstream fin(file_path);
    if (!fin.is_open()) {
        return {};
    }
    return std::string((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
}

static bool extract_number(const std::string& json, const std::string& key, double& out) {
    const std::string pattern = "\"" + key + "\"";
    const size_t key_pos = json.find(pattern);
    if (key_pos == std::string::npos) {
        return false;
    }

    const size_t colon = json.find(':', key_pos + pattern.size());
    if (colon == std::string::npos) {
        return false;
    }

    size_t pos = colon + 1u;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos])) != 0) {
        ++pos;
    }
    if (pos >= json.size()) {
        return false;
    }

    const char* begin = json.c_str() + pos;
    char* end = nullptr;
    out = std::strtod(begin, &end);
    return end != begin;
}

static bool extract_bool(const std::string& json, const std::string& key, bool& out) {
    const std::string pattern = "\"" + key + "\"";
    const size_t key_pos = json.find(pattern);
    if (key_pos == std::string::npos) {
        return false;
    }

    const size_t colon = json.find(':', key_pos + pattern.size());
    if (colon == std::string::npos) {
        return false;
    }

    size_t pos = colon + 1u;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos])) != 0) {
        ++pos;
    }

    if (json.compare(pos, 4u, "true") == 0) {
        out = true;
        return true;
    }
    if (json.compare(pos, 5u, "false") == 0) {
        out = false;
        return true;
    }
    return false;
}

static std::vector<std::string> parse_string_array(const std::string& json) {
    std::vector<std::string> out;
    const size_t open = json.find('[');
    const size_t close = json.find_last_of(']');
    if (open == std::string::npos || close == std::string::npos || close <= open) {
        return out;
    }

    size_t pos = open + 1u;
    while (pos < close) {
        const size_t q1 = json.find('"', pos);
        if (q1 == std::string::npos || q1 >= close) {
            break;
        }
        const size_t q2 = json.find('"', q1 + 1u);
        if (q2 == std::string::npos || q2 >= close) {
            break;
        }
        out.push_back(json.substr(q1 + 1u, q2 - q1 - 1u));
        pos = q2 + 1u;
    }

    return out;
}

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> parts;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        parts.push_back(token);
    }
    return parts;
}

static bool parse_float_row(const std::vector<std::string>& tokens,
                            uint16_t expected_cols,
                            std::vector<float>& row_out) {
    if (tokens.size() != expected_cols) {
        return false;
    }

    row_out.clear();
    row_out.reserve(expected_cols);
    for (const std::string& token : tokens) {
        const char* begin = token.c_str();
        char* end = nullptr;
        const float value = std::strtof(begin, &end);
        if (end == begin) {
            return false;
        }
        row_out.push_back(value);
    }
    return true;
}

static bool load_csv_dataset(const fs::path& csv_path, CsvDataset& dataset) {
    dataset = CsvDataset{};

    std::ifstream fin(csv_path);
    if (!fin.is_open()) {
        return false;
    }

    std::string line;
    if (!std::getline(fin, line)) {
        return false;
    }

    std::vector<std::string> first_tokens = split_csv_line(line);
    if (first_tokens.empty()) {
        return false;
    }

    std::vector<float> first_row;
    bool first_is_data = true;
    for (const std::string& token : first_tokens) {
        const char* begin = token.c_str();
        char* end = nullptr;
        (void)std::strtof(begin, &end);
        if (end == begin) {
            first_is_data = false;
            break;
        }
    }

    if (!first_is_data) {
        dataset.header = first_tokens;
        dataset.cols = static_cast<uint16_t>(dataset.header.size());
    } else {
        dataset.cols = static_cast<uint16_t>(first_tokens.size());
        first_row.reserve(dataset.cols);
        for (const std::string& token : first_tokens) {
            first_row.push_back(std::strtof(token.c_str(), nullptr));
        }
        dataset.matrix.insert(dataset.matrix.end(), first_row.begin(), first_row.end());
        dataset.rows = 1u;
    }

    std::vector<float> row_values;
    while (std::getline(fin, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> tokens = split_csv_line(line);
        if (!parse_float_row(tokens, dataset.cols, row_values)) {
            return false;
        }
        dataset.matrix.insert(dataset.matrix.end(), row_values.begin(), row_values.end());
        ++dataset.rows;
    }

    if (dataset.rows == 0u) {
        return false;
    }

    return true;
}

static bool load_plain_config(const fs::path& config_path, PlainEvalConfig& config_out) {
    const std::string json = read_text_file(config_path);
    if (json.empty()) {
        return false;
    }

    PlainEvalConfig cfg;
    double value = 0.0;
    bool bool_value = false;

    if (extract_number(json, "n_estimators", value) && value > 0.0) {
        cfg.n_estimators = static_cast<uint16_t>(value);
    }
    if (extract_number(json, "max_samples", value) && value > 0.0) {
        cfg.max_samples = static_cast<float>(value);
    }
    if (extract_number(json, "max_features", value) && value > 0.0) {
        cfg.max_features = static_cast<float>(value);
    }
    if (extract_bool(json, "bootstrap", bool_value)) {
        cfg.bootstrap = bool_value;
    }
    if (extract_number(json, "random_state", value) && value >= 0.0) {
        cfg.random_state = static_cast<uint32_t>(value);
    }

    if (extract_number(json, "fpr_threshold", value) && value > 0.0 && value < 1.0) {
        cfg.target_fpr = static_cast<float>(value);
    }

    config_out = cfg;
    return true;
}

int main() {
    const fs::path root = "/home/viettran/Documents/visual_code/EDR_AGENT";
    const fs::path config_path = root / "development_phase/results/iforest_optimized_config.json";
    const fs::path features_path = root / "development_phase/results/iforest_optimized_features.json";

    const fs::path ben_train_csv = root / "development_phase/data/optimized/iforest_ben_train.csv";
    const fs::path ben_val_csv = root / "development_phase/data/optimized/iforest_ben_val.csv";
    const fs::path ben_test_csv = root / "development_phase/data/optimized/iforest_ben_test.csv";
    const fs::path mal_val_csv = root / "development_phase/data/optimized/iforest_mal_val.csv";
    const fs::path mal_test_csv = root / "development_phase/data/optimized/iforest_mal_test.csv";

    PlainEvalConfig eval_config;
    if (!load_plain_config(config_path, eval_config)) {
        std::cerr << "CONFIG_LOAD_FAIL\n";
        return 1;
    }

    const std::vector<std::string> optimized_features = parse_string_array(read_text_file(features_path));
    if (optimized_features.empty()) {
        std::cerr << "FEATURE_LIST_FAIL\n";
        return 2;
    }

    CsvDataset ben_train;
    CsvDataset ben_val;
    CsvDataset ben_test;
    CsvDataset mal_val;
    CsvDataset mal_test;

    if (!load_csv_dataset(ben_train_csv, ben_train) ||
        !load_csv_dataset(ben_val_csv, ben_val) ||
        !load_csv_dataset(ben_test_csv, ben_test) ||
        !load_csv_dataset(mal_val_csv, mal_val) ||
        !load_csv_dataset(mal_test_csv, mal_test)) {
        std::cerr << "CSV_LOAD_FAIL\n";
        return 3;
    }

    const uint16_t expected_features = static_cast<uint16_t>(optimized_features.size());
    if (ben_train.cols != expected_features ||
        ben_val.cols != expected_features ||
        ben_test.cols != expected_features ||
        mal_val.cols != expected_features ||
        mal_test.cols != expected_features) {
        std::cerr << "FEATURE_COUNT_MISMATCH\n";
        return 4;
    }

    eml::PlainIsoForest model;
    if (!model.train(ben_train.matrix,
                     ben_train.rows,
                     ben_train.cols,
                     eval_config.n_estimators,
                     eval_config.max_samples,
                     eval_config.max_features,
                     eval_config.bootstrap,
                     eval_config.random_state)) {
        std::cerr << "TRAIN_FAIL\n";
        return 5;
    }

    const std::vector<float> val_b_scores = model.score_matrix(ben_val.matrix, ben_val.rows);
    const std::vector<float> val_m_scores = model.score_matrix(mal_val.matrix, mal_val.rows);
    if (val_b_scores.empty() || val_m_scores.empty()) {
        std::cerr << "VAL_SCORE_FAIL\n";
        return 6;
    }

    float val_cal_fpr = 0.0f;
    float val_cal_tpr = 0.0f;
    const float threshold = eml::PlainIsoForest::select_threshold_tpr_with_fpr_cap(
        val_b_scores,
        val_m_scores,
        eval_config.target_fpr,
        val_cal_fpr,
        val_cal_tpr
    );
    model.set_threshold(threshold);

    const std::vector<float> test_b_scores = model.score_matrix(ben_test.matrix, ben_test.rows);
    const std::vector<float> test_m_scores = model.score_matrix(mal_test.matrix, mal_test.rows);
    if (test_b_scores.empty() || test_m_scores.empty()) {
        std::cerr << "TEST_SCORE_FAIL\n";
        return 7;
    }

    float val_fpr = 0.0f;
    float val_tpr = 0.0f;
    eml::PlainIsoForest::compute_metrics(val_b_scores, val_m_scores, threshold, val_fpr, val_tpr);

    float test_fpr = 0.0f;
    float test_tpr = 0.0f;
    eml::PlainIsoForest::compute_metrics(test_b_scores, test_m_scores, threshold, test_fpr, test_tpr);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "PLAIN_IF_CONFIG "
              << "n_estimators=" << eval_config.n_estimators
              << " max_samples=" << eval_config.max_samples
              << " max_features=" << eval_config.max_features
              << " bootstrap=" << (eval_config.bootstrap ? 1 : 0)
              << " random_state=" << eval_config.random_state
              << " target_fpr=" << eval_config.target_fpr
              << " n_features=" << expected_features
              << "\n";

    std::cout << "PLAIN_IF_THRESHOLD "
              << "threshold=" << threshold
              << " val_cal_fpr=" << val_cal_fpr
              << " val_cal_tpr=" << val_cal_tpr
              << "\n";

    std::cout << "PLAIN_IF_VAL "
              << "ben_rows=" << ben_val.rows
              << " mal_rows=" << mal_val.rows
              << " fpr=" << val_fpr
              << " tpr=" << val_tpr
              << "\n";

    std::cout << "PLAIN_IF_TEST "
              << "ben_rows=" << ben_test.rows
              << " mal_rows=" << mal_test.rows
              << " fpr=" << test_fpr
              << " tpr=" << test_tpr
              << "\n";

    return 0;
}
