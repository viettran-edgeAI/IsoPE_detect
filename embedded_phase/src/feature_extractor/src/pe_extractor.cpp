#include "extractor/extractor.hpp"

#include <LIEF/PE.hpp>
#include <LIEF/PE/signature/Signature.hpp>

#include "compiled_feature_config.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

using embedded_feature_config::CompiledFeatureKind;
using embedded_feature_config::CompiledFeatureSpec;
using embedded_feature_config::kCompiledFeatureCount;
using embedded_feature_config::kCompiledFeatureNames;
using embedded_feature_config::kCompiledFeatureSource;
using embedded_feature_config::kCompiledFeatureSpecs;
using embedded_feature_config::kDataDirectorySlots;
using embedded_feature_config::kDirectFeatureCount;

constexpr size_t MAX_SECTIONS = 10;

struct ExtractResult {
  bool parse_ok = false;
  std::string error;
  double processing_time_ms = 0.0;

  std::array<double, kDirectFeatureCount> direct{};
  uint32_t coff_characteristics = 0;
  uint32_t opt_dll_characteristics = 0;
  std::array<double, MAX_SECTIONS> sec_entropy{};
  std::array<double, MAX_SECTIONS> sec_vsize{};
  std::array<double, MAX_SECTIONS> sec_is_write{};
  std::array<double, 32> sec_name_hash{};
  std::array<double, 64> imp_dll_hash{};
  std::array<double, 256> imp_func_hash{};
  std::array<double, kDataDirectorySlots> dd_rva{};
  std::array<double, kDataDirectorySlots> dd_size{};
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

template <size_t N>
void hash_update(std::array<double, N>& buckets, const std::string& input, bool lower) {
  if (input.empty()) {
    return;
  }

  const std::string payload = lower ? lower_ascii(input) : input;
  const auto* data = reinterpret_cast<const uint8_t*>(payload.data());

  const auto md5_digest = LIEF::PE::Signature::hash(data, payload.size(), LIEF::PE::ALGORITHMS::MD5);
  const auto sha1_digest = LIEF::PE::Signature::hash(data, payload.size(), LIEF::PE::ALGORITHMS::SHA_1);
  if (md5_digest.size() < sizeof(uint64_t) || sha1_digest.size() < sizeof(uint64_t)) {
    return;
  }

  uint64_t h1 = 0;
  uint64_t h2 = 0;
  std::memcpy(&h1, md5_digest.data(), sizeof(uint64_t));
  std::memcpy(&h2, sha1_digest.data(), sizeof(uint64_t));

  const size_t bucket = static_cast<size_t>(h1 % N);
  const double sign = (h2 % 2 == 0) ? 1.0 : -1.0;
  buckets[bucket] += sign;
}

ExtractResult extract_row(const std::string& filepath) {
  ExtractResult result;
  const auto started = std::chrono::steady_clock::now();

  try {
    std::unique_ptr<LIEF::PE::Binary> pe = LIEF::PE::Parser::parse(filepath);
    if (!pe) {
      result.error = "parse_failed";
    } else {
      result.parse_ok = true;
      const auto& header = pe->header();
      const auto& opt = pe->optional_header();

      result.direct[embedded_feature_config::D_COFF_MACHINE] = static_cast<double>(static_cast<uint32_t>(header.machine()));
      result.direct[embedded_feature_config::D_COFF_NUM_SECTIONS] = static_cast<double>(header.numberof_sections());
      result.direct[embedded_feature_config::D_COFF_TIMESTAMP] = static_cast<double>(header.time_date_stamp());
      result.direct[embedded_feature_config::D_COFF_SIZEOF_OPT_HEADER] = static_cast<double>(header.sizeof_optional_header());
      result.direct[embedded_feature_config::D_COFF_CHARACTERISTICS] = static_cast<double>(header.characteristics());
      result.direct[embedded_feature_config::D_OPT_SIZEOF_INIT_DATA] = static_cast<double>(opt.sizeof_initialized_data());
      result.direct[embedded_feature_config::D_OPT_CHECKSUM] = static_cast<double>(opt.checksum());
      result.direct[embedded_feature_config::D_OPT_SECTION_ALIGNMENT] = static_cast<double>(opt.section_alignment());
      result.direct[embedded_feature_config::D_OPT_IMAGEBASE] = static_cast<double>(opt.imagebase());
      result.direct[embedded_feature_config::D_OPT_SUBSYSTEM] = static_cast<double>(static_cast<uint32_t>(opt.subsystem()));

      result.coff_characteristics = header.characteristics();
      result.opt_dll_characteristics = opt.dll_characteristics();

      try {
        result.direct[embedded_feature_config::D_CHECKSUM_MATCHES] = (opt.checksum() == pe->compute_checksum()) ? 1.0 : 0.0;
      } catch (...) {
        result.direct[embedded_feature_config::D_CHECKSUM_MATCHES] = 0.0;
      }

      size_t section_count = 0;
      double entropy_sum = 0.0;
      double entropy_max = 0.0;

      for (const LIEF::PE::Section& sec : pe->sections()) {
        if (section_count >= MAX_SECTIONS) {
          break;
        }

        const double entropy = sec.entropy();
        result.sec_entropy[section_count] = entropy;
        result.sec_vsize[section_count] = static_cast<double>(sec.virtual_size());
        result.sec_is_write[section_count] = sec.has_characteristic(LIEF::PE::Section::CHARACTERISTICS::MEM_WRITE) ? 1.0 : 0.0;

        hash_update(result.sec_name_hash, sec.name(), false);

        entropy_sum += entropy;
        if (section_count == 0 || entropy > entropy_max) {
          entropy_max = entropy;
        }
        section_count += 1;
      }

      if (section_count > 0) {
        result.direct[embedded_feature_config::D_SEC_MEAN_ENTROPY] = entropy_sum / static_cast<double>(section_count);
        result.direct[embedded_feature_config::D_SEC_MAX_ENTROPY] = entropy_max;
      }

      result.direct[embedded_feature_config::D_HAS_RESOURCES] = pe->has_resources() ? 1.0 : 0.0;
      if (pe->has_resources()) {
        if (auto manager = pe->resources_manager()) {
          result.direct[embedded_feature_config::D_RSRC_HAS_VERSION] = manager->has_version() ? 1.0 : 0.0;
        }
      }

      result.direct[embedded_feature_config::D_HAS_RELOCATIONS] = pe->has_relocations() ? 1.0 : 0.0;
      result.direct[embedded_feature_config::D_HAS_DEBUG] = pe->has_debug() ? 1.0 : 0.0;

      if (pe->has_debug()) {
        bool has_repro = false;
        for (const LIEF::PE::Debug& debug_entry : pe->debug()) {
          if (debug_entry.type() == LIEF::PE::Debug::TYPES::REPRO) {
            has_repro = true;
            break;
          }
        }
        result.direct[embedded_feature_config::D_HAS_REPRO] = has_repro ? 1.0 : 0.0;
      }

      if (pe->has_rich_header()) {
        if (const LIEF::PE::RichHeader* rich = pe->rich_header()) {
          uint32_t max_build_id = 0;
          for (const LIEF::PE::RichEntry& entry : rich->entries()) {
            max_build_id = std::max(max_build_id, static_cast<uint32_t>(entry.build_id()));
          }
          result.direct[embedded_feature_config::D_RICH_MAX_BUILD_ID] = static_cast<double>(max_build_id);
        }
      }

      const auto overlay = pe->overlay();
      result.direct[embedded_feature_config::D_HAS_OVERLAY] = overlay.empty() ? 0.0 : 1.0;
      result.direct[embedded_feature_config::D_OVERLAY_SIZE] = static_cast<double>(overlay.size());
      result.direct[embedded_feature_config::D_OVERLAY_ENTROPY] = overlay.empty() ? 0.0 : entropy_from_span(overlay, 8192);

      if (pe->has_imports()) {
        for (const LIEF::PE::Import& imp : pe->imports()) {
          hash_update(result.imp_dll_hash, imp.name(), true);
          for (const LIEF::PE::ImportEntry& entry : imp.entries()) {
            if (!entry.is_ordinal() && !entry.name().empty()) {
              hash_update(result.imp_func_hash, entry.name(), true);
            }
          }
        }
      }

      if (pe->has_signatures()) {
        size_t certificate_count = 0;
        for (const LIEF::PE::Signature& signature : pe->signatures()) {
          certificate_count += signature.certificates().size();
        }
        result.direct[embedded_feature_config::D_NUM_CERTIFICATES] = static_cast<double>(certificate_count);

        try {
          const auto verified = pe->verify_signature();
          result.direct[embedded_feature_config::D_SIG_VERIFIED] =
              (verified == LIEF::PE::Signature::VERIFICATION_FLAGS::OK) ? 1.0 : 0.0;
        } catch (...) {
          result.direct[embedded_feature_config::D_SIG_VERIFIED] = 0.0;
        }
      }

      for (const LIEF::PE::DataDirectory& directory : pe->data_directories()) {
        const size_t idx = static_cast<size_t>(directory.type());
        if (idx < kDataDirectorySlots) {
          result.dd_rva[idx] = static_cast<double>(directory.RVA());
          result.dd_size[idx] = static_cast<double>(directory.size());
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

double resolve_value(const ExtractResult& row, const CompiledFeatureSpec& spec) {
  switch (spec.kind) {
    case CompiledFeatureKind::Direct:
      return (spec.index < row.direct.size()) ? row.direct[spec.index] : 0.0;
    case CompiledFeatureKind::CoffChar:
      return (row.coff_characteristics & spec.value) ? 1.0 : 0.0;
    case CompiledFeatureKind::OptDllChar:
      return (row.opt_dll_characteristics & spec.value) ? 1.0 : 0.0;
    case CompiledFeatureKind::SecEntropy:
      return (spec.index < row.sec_entropy.size()) ? row.sec_entropy[spec.index] : 0.0;
    case CompiledFeatureKind::SecVsize:
      return (spec.index < row.sec_vsize.size()) ? row.sec_vsize[spec.index] : 0.0;
    case CompiledFeatureKind::SecIsWrite:
      return (spec.index < row.sec_is_write.size()) ? row.sec_is_write[spec.index] : 0.0;
    case CompiledFeatureKind::SecNameHash:
      return (spec.index < row.sec_name_hash.size()) ? row.sec_name_hash[spec.index] : 0.0;
    case CompiledFeatureKind::ImpDllHash:
      return (spec.index < row.imp_dll_hash.size()) ? row.imp_dll_hash[spec.index] : 0.0;
    case CompiledFeatureKind::ImpFuncHash:
      return (spec.index < row.imp_func_hash.size()) ? row.imp_func_hash[spec.index] : 0.0;
    case CompiledFeatureKind::DataDirectoryRva:
      return (spec.index < row.dd_rva.size()) ? row.dd_rva[spec.index] : 0.0;
    case CompiledFeatureKind::DataDirectorySize:
      return (spec.index < row.dd_size.size()) ? row.dd_size[spec.index] : 0.0;
    default:
      return 0.0;
  }
}

extractor::FeatureVector make_feature_vector(const ExtractResult& row) {
  extractor::FeatureVector vector;
  vector.values.resize(kCompiledFeatureCount, 0.0);
  for (size_t i = 0; i < kCompiledFeatureCount; ++i) {
    vector.values[i] = resolve_value(row, kCompiledFeatureSpecs[i]);
  }
  return vector;
}

} // namespace

namespace extractor {

FeatureVector PEExtractor::extract(const std::string& path) {
  const ExtractionReport report = extract_with_metadata(path);
  if (!report.metadata.parse_ok) {
    throw std::runtime_error(
      report.metadata.error.empty() ? "extractor_parse_failed" : report.metadata.error
    );
  }
  return report.feature_vector;
}

ExtractionReport PEExtractor::extract_with_metadata(const std::string& path) const {
  const ExtractResult row = extract_row(path);

  ExtractionReport report;
  report.feature_vector = make_feature_vector(row);
  report.metadata.parse_ok = row.parse_ok;
  report.metadata.error = row.error;
  report.metadata.processing_time_ms = row.processing_time_ms;
  return report;
}

size_t compiled_feature_count() {
  return kCompiledFeatureCount;
}

std::vector<std::string> compiled_feature_names() {
  std::vector<std::string> names;
  names.reserve(kCompiledFeatureCount);
  for (const char* name : kCompiledFeatureNames) {
    names.emplace_back(name);
  }
  return names;
}

const std::string& compiled_feature_source() {
  static const std::string source = kCompiledFeatureSource;
  return source;
}

} // namespace extractor
