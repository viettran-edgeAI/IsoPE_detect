#include <LIEF/PE.hpp>
#include "hash_stream.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr size_t MAX_SECTIONS = 10;

const std::vector<std::string> kFallbackFeatureNames = {
  "coff_machine",
  "opt_sizeof_init_data",
  "has_resources",
  "sec_name_hash_23",
  "sec3_is_write",
  "imp_dll_hash_61",
  "num_certificates",
  "sec_name_hash_8",
  "sec_mean_entropy",
  "dd_DEBUG_DIR_rva",
  "imp_func_hash_149",
  "opt_checksum",
  "sec2_entropy",
  "has_relocations",
  "dd_CERTIFICATE_TABLE_rva",
  "coff_characteristics",
  "sec1_entropy",
  "rich_max_build_id",
  "overlay_size",
  "has_overlay",
  "opt_dllchar_NO_SEH",
  "overlay_entropy",
  "rsrc_has_version",
  "opt_dllchar_HIGH_ENTROPY_VA",
  "has_repro",
  "has_debug",
  "opt_section_alignment",
  "sec2_vsize",
  "imp_dll_hash_43",
  "opt_imagebase",
  "coff_char_DLL",
  "dd_CLR_RUNTIME_HEADER_rva",
  "sec_max_entropy",
  "opt_dllchar_DYNAMIC_BASE",
  "dd_CERTIFICATE_TABLE_size",
  "coff_char_LARGE_ADDRESS_AWARE",
  "sig_verified",
  "opt_subsystem",
  "opt_dllchar_NX_COMPAT",
  "checksum_matches",
};

struct ExtractResult {
  std::string filepath;
  bool parse_ok = false;
  std::string error;
  double processing_time_ms = 0.0;
  std::unordered_map<std::string, double> features;
};

struct Args {
  std::string feature_names_path = "development_phase/results/feature_names.json";
  std::string output_path;
  std::string format = "csv";
  std::vector<std::string> files;
};

std::string lower_ascii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

double entropy_from_span(LIEF::span<const uint8_t> data, size_t max_bytes = std::numeric_limits<size_t>::max()) {
  const size_t usable = std::min(data.size(), max_bytes);
  if (usable == 0) {
    return 0.0;
  }

  std::array<size_t, 256> counts{};
  for (size_t index = 0; index < usable; ++index) {
    counts[data[index]] += 1;
  }

  double entropy = 0.0;
  for (size_t count : counts) {
    if (count == 0) {
      continue;
    }
    const double probability = static_cast<double>(count) / static_cast<double>(usable);
    entropy -= probability * std::log2(probability);
  }
  return entropy;
}

std::vector<double> feature_hash(const std::vector<std::string>& strings, size_t n_features) {
  std::vector<double> result(n_features, 0.0);

  for (const std::string& value : strings) {
    if (value.empty()) {
      continue;
    }

    LIEF::hashstream md5_stream(LIEF::hashstream::HASH::MD5);
    LIEF::hashstream sha1_stream(LIEF::hashstream::HASH::SHA1);

    md5_stream.write(reinterpret_cast<const uint8_t*>(value.data()), value.size());
    sha1_stream.write(reinterpret_cast<const uint8_t*>(value.data()), value.size());

    const auto& md5_digest = md5_stream.raw();
    const auto& sha1_digest = sha1_stream.raw();
    if (md5_digest.size() < sizeof(uint64_t) || sha1_digest.size() < sizeof(uint64_t)) {
      continue;
    }

    uint64_t h1 = 0;
    uint64_t h2 = 0;
    std::memcpy(&h1, md5_digest.data(), sizeof(uint64_t));
    std::memcpy(&h2, sha1_digest.data(), sizeof(uint64_t));

    const size_t bucket = static_cast<size_t>(h1 % n_features);
    const double sign = (h2 % 2 == 0) ? 1.0 : -1.0;
    result[bucket] += sign;
  }

  return result;
}

std::vector<std::string> load_feature_names(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    return kFallbackFeatureNames;
  }

  std::stringstream buffer;
  buffer << input.rdbuf();
  const std::string text = buffer.str();

  std::regex string_regex("\"([^\"]+)\"");
  std::sregex_iterator iter(text.begin(), text.end(), string_regex);
  std::sregex_iterator end;

  std::vector<std::string> names;
  for (; iter != end; ++iter) {
    names.push_back((*iter)[1].str());
  }

  if (names.empty()) {
    return kFallbackFeatureNames;
  }
  return names;
}

double feature_or_zero(const std::unordered_map<std::string, double>& features, const std::string& name) {
  const auto it = features.find(name);
  return (it == features.end()) ? 0.0 : it->second;
}

std::string json_escape(const std::string& text) {
  std::ostringstream out;
  for (char ch : text) {
    switch (ch) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << ch; break;
    }
  }
  return out.str();
}

ExtractResult extract_features(const std::string& filepath, const std::vector<std::string>& feature_names) {
  ExtractResult result;
  result.filepath = filepath;

  for (const std::string& name : feature_names) {
    result.features[name] = 0.0;
  }

  const auto started = std::chrono::steady_clock::now();

  try {
    std::unique_ptr<LIEF::PE::Binary> pe = LIEF::PE::Parser::parse(filepath);
    if (!pe) {
      result.error = "parse_failed";
    } else {
      result.parse_ok = true;

      const auto& header = pe->header();
      const auto& opt = pe->optional_header();

      result.features["coff_machine"] = static_cast<double>(static_cast<uint32_t>(header.machine()));
      result.features["coff_characteristics"] = static_cast<double>(header.characteristics());
      result.features["coff_char_DLL"] = header.has_characteristic(LIEF::PE::Header::CHARACTERISTICS::DLL) ? 1.0 : 0.0;
      result.features["coff_char_LARGE_ADDRESS_AWARE"] = header.has_characteristic(LIEF::PE::Header::CHARACTERISTICS::LARGE_ADDRESS_AWARE) ? 1.0 : 0.0;

      result.features["opt_sizeof_init_data"] = static_cast<double>(opt.sizeof_initialized_data());
      result.features["opt_checksum"] = static_cast<double>(opt.checksum());
      result.features["opt_section_alignment"] = static_cast<double>(opt.section_alignment());
      result.features["opt_imagebase"] = static_cast<double>(opt.imagebase());
      result.features["opt_subsystem"] = static_cast<double>(static_cast<uint32_t>(opt.subsystem()));
      result.features["opt_dllchar_NO_SEH"] = opt.has(LIEF::PE::OptionalHeader::DLL_CHARACTERISTICS::NO_SEH) ? 1.0 : 0.0;
      result.features["opt_dllchar_HIGH_ENTROPY_VA"] = opt.has(LIEF::PE::OptionalHeader::DLL_CHARACTERISTICS::HIGH_ENTROPY_VA) ? 1.0 : 0.0;
      result.features["opt_dllchar_DYNAMIC_BASE"] = opt.has(LIEF::PE::OptionalHeader::DLL_CHARACTERISTICS::DYNAMIC_BASE) ? 1.0 : 0.0;
      result.features["opt_dllchar_NX_COMPAT"] = opt.has(LIEF::PE::OptionalHeader::DLL_CHARACTERISTICS::NX_COMPAT) ? 1.0 : 0.0;

      try {
        result.features["checksum_matches"] = (opt.checksum() == pe->compute_checksum()) ? 1.0 : 0.0;
      } catch (...) {
        result.features["checksum_matches"] = 0.0;
      }

      std::vector<double> entropies;
      std::vector<std::string> section_names;
      size_t section_index = 0;
      for (const LIEF::PE::Section& sec : pe->sections()) {
        if (section_index >= MAX_SECTIONS) {
          break;
        }
        const double entropy = sec.entropy();
        entropies.push_back(entropy);
        section_names.push_back(sec.name());

        if (section_index == 1) {
          result.features["sec1_entropy"] = entropy;
        }
        if (section_index == 2) {
          result.features["sec2_entropy"] = entropy;
          result.features["sec2_vsize"] = static_cast<double>(sec.virtual_size());
        }
        if (section_index == 3) {
          result.features["sec3_is_write"] = sec.has_characteristic(LIEF::PE::Section::CHARACTERISTICS::MEM_WRITE) ? 1.0 : 0.0;
        }
        section_index += 1;
      }

      if (!entropies.empty()) {
        const double sum_entropy = std::accumulate(entropies.begin(), entropies.end(), 0.0);
        result.features["sec_mean_entropy"] = sum_entropy / static_cast<double>(entropies.size());
        result.features["sec_max_entropy"] = *std::max_element(entropies.begin(), entropies.end());
      }

      const std::vector<double> sec_hash = feature_hash(section_names, 32);
      result.features["sec_name_hash_8"] = sec_hash.at(8);
      result.features["sec_name_hash_23"] = sec_hash.at(23);

      result.features["has_resources"] = pe->has_resources() ? 1.0 : 0.0;
      if (pe->has_resources()) {
        if (auto manager = pe->resources_manager()) {
          result.features["rsrc_has_version"] = manager->has_version() ? 1.0 : 0.0;
        }
      }

      result.features["has_relocations"] = pe->has_relocations() ? 1.0 : 0.0;
      result.features["has_debug"] = pe->has_debug() ? 1.0 : 0.0;
      if (pe->has_debug()) {
        bool has_repro = false;
        for (const LIEF::PE::Debug& debug_entry : pe->debug()) {
          if (debug_entry.type() == LIEF::PE::Debug::TYPES::REPRO) {
            has_repro = true;
            break;
          }
        }
        result.features["has_repro"] = has_repro ? 1.0 : 0.0;
      }

      if (pe->has_rich_header()) {
        if (const LIEF::PE::RichHeader* rich = pe->rich_header()) {
          uint32_t max_build_id = 0;
          for (const LIEF::PE::RichEntry& entry : rich->entries()) {
            max_build_id = std::max(max_build_id, static_cast<uint32_t>(entry.build_id()));
          }
          result.features["rich_max_build_id"] = static_cast<double>(max_build_id);
        }
      }

      const auto overlay = pe->overlay();
      result.features["has_overlay"] = overlay.empty() ? 0.0 : 1.0;
      result.features["overlay_size"] = static_cast<double>(overlay.size());
      result.features["overlay_entropy"] = overlay.empty() ? 0.0 : entropy_from_span(overlay, 8192);

      if (pe->has_imports()) {
        std::vector<std::string> dll_names;
        std::vector<std::string> func_names;

        for (const LIEF::PE::Import& imp : pe->imports()) {
          dll_names.push_back(lower_ascii(imp.name()));
          for (const LIEF::PE::ImportEntry& entry : imp.entries()) {
            if (!entry.is_ordinal()) {
              const std::string& name = entry.name();
              if (!name.empty()) {
                func_names.push_back(lower_ascii(name));
              }
            }
          }
        }

        const std::vector<double> dll_hash = feature_hash(dll_names, 64);
        result.features["imp_dll_hash_43"] = dll_hash.at(43);
        result.features["imp_dll_hash_61"] = dll_hash.at(61);

        const std::vector<double> func_hash = feature_hash(func_names, 256);
        result.features["imp_func_hash_149"] = func_hash.at(149);
      }

      if (pe->has_signatures()) {
        size_t certificate_count = 0;
        for (const LIEF::PE::Signature& signature : pe->signatures()) {
          certificate_count += signature.certificates().size();
        }
        result.features["num_certificates"] = static_cast<double>(certificate_count);

        try {
          const auto verified = pe->verify_signature();
          result.features["sig_verified"] = (verified == LIEF::PE::Signature::VERIFICATION_FLAGS::OK) ? 1.0 : 0.0;
        } catch (...) {
          result.features["sig_verified"] = 0.0;
        }
      }

      for (const LIEF::PE::DataDirectory& directory : pe->data_directories()) {
        switch (directory.type()) {
          case LIEF::PE::DataDirectory::TYPES::DEBUG_DIR:
            result.features["dd_DEBUG_DIR_rva"] = static_cast<double>(directory.RVA());
            break;
          case LIEF::PE::DataDirectory::TYPES::CERTIFICATE_TABLE:
            result.features["dd_CERTIFICATE_TABLE_rva"] = static_cast<double>(directory.RVA());
            result.features["dd_CERTIFICATE_TABLE_size"] = static_cast<double>(directory.size());
            break;
          case LIEF::PE::DataDirectory::TYPES::CLR_RUNTIME_HEADER:
            result.features["dd_CLR_RUNTIME_HEADER_rva"] = static_cast<double>(directory.RVA());
            break;
          default:
            break;
        }
      }
    }
  } catch (const std::exception& ex) {
    result.error = ex.what();
  } catch (...) {
    result.error = "unknown_exception";
  }

  const auto ended = std::chrono::steady_clock::now();
  result.processing_time_ms = std::chrono::duration<double, std::milli>(ended - started).count();
  return result;
}

void print_usage(const char* prog) {
  std::cerr << "Usage: " << prog << " [--feature-names PATH] [--output PATH] [--format csv|jsonl] <pe_file> [<pe_file> ...]\n";
}

bool parse_args(int argc, char** argv, Args& args) {
  for (int index = 1; index < argc; ++index) {
    const std::string token(argv[index]);

    if (token == "--feature-names") {
      if (index + 1 >= argc) {
        return false;
      }
      args.feature_names_path = argv[++index];
    } else if (token == "--output") {
      if (index + 1 >= argc) {
        return false;
      }
      args.output_path = argv[++index];
    } else if (token == "--format") {
      if (index + 1 >= argc) {
        return false;
      }
      args.format = lower_ascii(argv[++index]);
      if (args.format != "csv" && args.format != "jsonl") {
        return false;
      }
    } else if (token == "--help" || token == "-h") {
      return false;
    } else if (!token.empty() && token[0] == '-') {
      return false;
    } else {
      args.files.push_back(token);
    }
  }
  return !args.files.empty();
}

int run(const Args& args) {
  const std::vector<std::string> feature_names = load_feature_names(args.feature_names_path);

  std::vector<ExtractResult> rows;
  rows.reserve(args.files.size());
  for (const std::string& path : args.files) {
    rows.push_back(extract_features(path, feature_names));
  }

  std::ostream* out = &std::cout;
  std::ofstream file_out;
  if (!args.output_path.empty()) {
    file_out.open(args.output_path, std::ios::out | std::ios::trunc);
    if (!file_out) {
      std::cerr << "Failed to open output path: " << args.output_path << "\n";
      return 2;
    }
    out = &file_out;
  }

  out->setf(std::ios::fixed);
  *out << std::setprecision(9);

  if (args.format == "jsonl") {
    for (const ExtractResult& row : rows) {
      *out << "{\"filepath\":\"" << json_escape(row.filepath)
           << "\",\"parse_ok\":" << (row.parse_ok ? "true" : "false")
           << ",\"processing_time_ms\":" << row.processing_time_ms
           << ",\"error\":\"" << json_escape(row.error)
           << "\",\"features\":{";

      for (size_t i = 0; i < feature_names.size(); ++i) {
        const std::string& feature = feature_names[i];
        *out << "\"" << json_escape(feature) << "\":" << feature_or_zero(row.features, feature);
        if (i + 1 < feature_names.size()) {
          *out << ',';
        }
      }
      *out << "}}\n";
    }
  } else {
    *out << "filepath,parse_ok,processing_time_ms,error";
    for (const std::string& feature : feature_names) {
      *out << ',' << feature;
    }
    *out << '\n';

    for (const ExtractResult& row : rows) {
      *out << '"' << json_escape(row.filepath) << '"'
           << ',' << (row.parse_ok ? 1 : 0)
           << ',' << row.processing_time_ms
           << ',' << '"' << json_escape(row.error) << '"';

      for (const std::string& feature : feature_names) {
        *out << ',' << feature_or_zero(row.features, feature);
      }
      *out << '\n';
    }
  }

  return 0;
}

} // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, args)) {
    print_usage(argv[0]);
    return 1;
  }
  return run(args);
}
