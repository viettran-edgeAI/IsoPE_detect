// embedded_phase/tools/data_quantization/processing_data.cpp
//
// Batch orchestrator for the dataset quantization pipeline.
//
// Reads quantization_config.json (model_name + input_dir + shared settings),
// then invokes the single-file quantization module at
//   ../../../tools/data_quantization/processing_data
// for each of the five optimized CSV splits:
//   <model_name>_ben_train.csv
//   <model_name>_ben_test.csv
//   <model_name>_ben_val.csv
//   <model_name>_mal_test.csv
//   <model_name>_mal_val.csv
//
// All output artifacts (*.bin, *.csv, *_dp.txt) are placed in
//   <config_dir>/quantized_datasets/
// which mirrors the layout produced by the single-split workflow.

#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Minimal JSON helpers  (nested {"value":...} and flat formats)
// ─────────────────────────────────────────────────────────────────────────────

static std::string trimWs(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static std::string toLower(std::string s) {
    for (char& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static bool extractJsonValue(const std::string& content,
                             const std::string& key,
                             std::string& out) {
    const std::string pattern = "\"" + key + "\"";
    size_t kPos = content.find(pattern);
    if (kPos == std::string::npos) return false;

    size_t colon = content.find(':', kPos + pattern.size());
    if (colon == std::string::npos) return false;

    size_t vStart = content.find_first_not_of(" \t\r\n", colon + 1);
    if (vStart == std::string::npos) return false;

    // Nested {"value": ...}
    if (content[vStart] == '{') {
        size_t objEnd = content.find('}', vStart);
        if (objEnd == std::string::npos) return false;
        std::string obj = content.substr(vStart, objEnd - vStart + 1);
        return extractJsonValue(obj, "value", out);
    }

    // Quoted string
    if (content[vStart] == '"') {
        size_t end = content.find('"', vStart + 1);
        if (end == std::string::npos) return false;
        out = content.substr(vStart + 1, end - vStart - 1);
        return true;
    }

    // Scalar (number / bool / null)
    size_t end = content.find_first_of(",}\n\r", vStart);
    out = trimWs(content.substr(vStart, end - vStart));
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

struct OrchestratorConfig {
    std::string modelName;        // e.g. "iforest"
    std::string inputDir;         // directory containing the five optimized CSV splits
    int         quantBits    = 2;
    std::string headerMode   = "auto";
    std::string problemType  = "isolation";
    bool        removeOutliers = false;
};

static OrchestratorConfig loadConfig(const std::string& configPath) {
    std::ifstream fin(configPath);
    if (!fin) throw std::runtime_error("Cannot open config file: " + configPath);

    std::ostringstream ss;
    ss << fin.rdbuf();
    const std::string content = ss.str();

    OrchestratorConfig cfg;
    std::string raw;

    if (extractJsonValue(content, "model_name", raw))
        cfg.modelName = trimWs(raw);
    if (extractJsonValue(content, "input_dir", raw))
        cfg.inputDir = trimWs(raw);
    if (extractJsonValue(content, "quantization_bits", raw) && !raw.empty())
        cfg.quantBits = std::stoi(raw);
    if (extractJsonValue(content, "header", raw))
        cfg.headerMode = toLower(trimWs(raw));
    if (extractJsonValue(content, "problem_type", raw))
        cfg.problemType = toLower(trimWs(raw));
    if (extractJsonValue(content, "remove_outliers", raw)) {
        std::string lower = toLower(raw);
        cfg.removeOutliers = (lower == "true" || lower == "1" || lower == "yes");
    }

    if (cfg.modelName.empty())
        throw std::runtime_error("Config missing required field: model_name");
    if (cfg.inputDir.empty())
        throw std::runtime_error("Config missing required field: input_dir");

    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// Shell quoting
// ─────────────────────────────────────────────────────────────────────────────

// Wrap path in single-quotes and escape any embedded single-quotes.
static std::string shellQuote(const std::string& s) {
    std::string result = "'";
    for (char c : s) {
        if (c == '\'') result += "'\\''";
        else            result += c;
    }
    result += "'";
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    try {
        // ── Parse CLI arguments ───────────────────────────────────────────
        std::string configPath = "quantization_config.json";
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
                configPath = argv[++i];
            } else if (arg == "-h" || arg == "--help") {
                std::cout
                    << "Usage: " << argv[0] << " [-c quantization_config.json]\n\n"
                    << "Batch orchestrator: reads model_name and input_dir from config,\n"
                    << "then calls ../../../tools/data_quantization/processing_data\n"
                    << "for each of the five optimized CSV splits:\n"
                    << "  <model_name>_ben_train.csv\n"
                    << "  <model_name>_ben_test.csv\n"
                    << "  <model_name>_ben_val.csv\n"
                    << "  <model_name>_mal_test.csv\n"
                    << "  <model_name>_mal_val.csv\n\n"
                    << "Config fields:\n"
                    << "  model_name        e.g. \"iforest\"\n"
                    << "  input_dir         path to development_phase/data/optimized\n"
                    << "  quantization_bits 1-8 (default 2)\n"
                    << "  header            auto|yes|no\n"
                    << "  problem_type      isolation|classification|regression\n"
                    << "  remove_outliers   true|false\n";
                return 0;
            }
        }

        // ── Load config ───────────────────────────────────────────────────
        const OrchestratorConfig cfg = loadConfig(configPath);

        // Resolve all paths relative to the config file's directory.
        std::filesystem::path cfgParent = std::filesystem::path(configPath).parent_path();
        if (cfgParent.empty()) {
            cfgParent = std::filesystem::current_path();
        }
        const std::filesystem::path cfgDir = std::filesystem::absolute(cfgParent);
        if (cfgDir.empty())
            throw std::runtime_error("Cannot determine config directory.");

        const std::filesystem::path inputDirAbs =
            std::filesystem::weakly_canonical(cfgDir / cfg.inputDir);

        // Path to the single-file processor binary.
        // Convention: this orchestrator lives at
        //   embedded_phase/tools/data_quantization/
        // and the single-file processor at
        //   tools/data_quantization/
        const std::filesystem::path singleProcAbs =
            std::filesystem::weakly_canonical(
                cfgDir / "../../../tools/data_quantization/processing_data");

        if (!std::filesystem::exists(singleProcAbs)) {
            throw std::runtime_error(
                "Single-file processor not found: " + singleProcAbs.string() +
                "\nBuild it first:  cd tools/data_quantization && make build");
        }
        if (!std::filesystem::is_directory(inputDirAbs)) {
            throw std::runtime_error(
                "input_dir not found or not a directory: " + inputDirAbs.string());
        }

        // ── Five CSV splits ───────────────────────────────────────────────
        const std::string& mn = cfg.modelName;
        const std::string trainStem = mn + "_ben_train";
        const std::vector<std::string> splitStems = {
            trainStem,
            mn + "_ben_test",
            mn + "_ben_val",
            mn + "_mal_test",
            mn + "_mal_val",
        };

        // Ensure output directory exists next to this config.
        const std::filesystem::path outDir = cfgDir / "quantized_datasets";
        std::filesystem::create_directories(outDir);

        // ── Orchestrate ───────────────────────────────────────────────────
        std::cout << "=== Batch Quantization Orchestrator ===\n"
                  << "  Model     : " << mn                      << "\n"
                  << "  Input dir : " << inputDirAbs.string()    << "\n"
                  << "  Output dir: " << outDir.string()         << "\n"
                  << "  Quant bits: " << cfg.quantBits           << "\n"
                  << "  Problem   : " << cfg.problemType         << "\n\n";

        int failures = 0;

        // Validate required inputs first.
        for (const auto& stem : splitStems) {
            const std::filesystem::path csvPath = inputDirAbs / (stem + ".csv");
            if (!std::filesystem::exists(csvPath)) {
                std::cerr << "  MISSING  " << csvPath.string() << "\n";
                ++failures;
            }
        }
        if (failures > 0) {
            std::cout << "=== Batch Complete ===\n"
                      << "  Processed : 0 / " << splitStems.size() << "\n"
                      << "  Failures  : " << failures << "\n";
            return 1;
        }

        // Remove stale per-split quantizer files to enforce one quantizer per model.
        for (const auto& stem : splitStems) {
            const std::filesystem::path staleQtz = outDir / (stem + "_qtz.bin");
            if (std::filesystem::exists(staleQtz)) {
                std::error_code ec;
                std::filesystem::remove(staleQtz, ec);
            }
        }

        // 1) Fit quantizer once on benign-train with model-level artifact names.
        const std::filesystem::path trainCsvPath = inputDirAbs / (trainStem + ".csv");
        std::cout << "[1/" << splitStems.size() << "]  " << trainStem << ".csv (fit quantizer once) ...\n";

        const std::string fitCmd =
            "cd " + shellQuote(cfgDir.string()) +
            " && " + shellQuote(singleProcAbs.string()) +
            " -ip " + shellQuote(trainCsvPath.string()) +
            " -mn " + shellQuote(mn) +
            " -qb " + std::to_string(cfg.quantBits) +
            " -pt " + shellQuote(cfg.problemType) +
            " -hd " + shellQuote(cfg.headerMode) +
            " -ro " + (cfg.removeOutliers ? "true" : "false");

        if (std::system(fitCmd.c_str()) != 0) {
            std::cerr << "  FAILED  quantizer fit on benign train\n";
            return 1;
        }
        std::cout << "  OK\n\n";

        // Copy model-level train outputs to split-scoped train outputs for eval loaders.
        const std::vector<std::pair<std::string, std::string>> trainCopies = {
            {mn + "_nml.bin", trainStem + "_nml.bin"},
            {mn + "_nml.csv", trainStem + "_nml.csv"},
            {mn + "_dp.txt",  trainStem + "_dp.txt"},
        };
        for (const auto& [srcName, dstName] : trainCopies) {
            const std::filesystem::path src = outDir / srcName;
            const std::filesystem::path dst = outDir / dstName;
            if (!std::filesystem::exists(src)) {
                throw std::runtime_error("Missing train artifact: " + src.string());
            }
            std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);
        }

        const std::filesystem::path modelQuantizerPath = outDir / (mn + "_qtz.bin");
        if (!std::filesystem::exists(modelQuantizerPath)) {
            throw std::runtime_error("Expected model quantizer not found: " + modelQuantizerPath.string());
        }

        // 2) Transform remaining splits using the single fitted quantizer.
        size_t progress = 2;
        for (const auto& stem : splitStems) {
            if (stem == trainStem) {
                continue;
            }

            const std::filesystem::path inputPath = inputDirAbs / (stem + ".csv");
            std::cout << "[" << progress << "/" << splitStems.size() << "]  "
                      << stem << ".csv (transform with shared quantizer) ...\n";

            const std::string transformCmd =
                "cd " + shellQuote(cfgDir.string()) +
                " && " + shellQuote(singleProcAbs.string()) +
                " -ip " + shellQuote(inputPath.string()) +
                " -mn " + shellQuote(stem) +
                " -qb " + std::to_string(cfg.quantBits) +
                " -pt " + shellQuote(cfg.problemType) +
                " -hd " + shellQuote(cfg.headerMode) +
                " -ro " + (cfg.removeOutliers ? "true" : "false") +
                " -qp " + shellQuote(modelQuantizerPath.string());

            const int ret = std::system(transformCmd.c_str());
            if (ret != 0) {
                std::cerr << "  FAILED  exit=" << ret
                          << " for: " << stem << ".csv\n";
                ++failures;
            } else {
                std::cout << "  OK\n";
            }
            std::cout << "\n";
            ++progress;
        }

        // ── Summary ───────────────────────────────────────────────────────
        const int processed = static_cast<int>(splitStems.size()) - failures;
        std::cout << "=== Batch Complete ===\n"
                  << "  Processed : " << processed
                  << " / " << splitStems.size() << "\n";
        if (failures > 0) {
            std::cout << "  Failures  : " << failures << "\n";
            return 1;
        }

        std::cout << "  Artifacts : " << outDir.string() << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
