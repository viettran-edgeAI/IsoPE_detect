#include "extractor/extractor.hpp"
#include "extractor/resource_limits.hpp"

#include <LIEF/PE.hpp>
#include <LIEF/PE/signature/Signature.hpp>

#include "compiled_feature_config.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace {

using embedded_feature_config::CompiledFeatureKind;
using embedded_feature_config::CompiledFeatureSpec;
using embedded_feature_config::kCompiledFeatureCount;
using embedded_feature_config::kCompiledFeatureNames;
using embedded_feature_config::kCompiledFeatureSource;
using embedded_feature_config::kCompiledFeatureSpecs;
using embedded_feature_config::kDataDirectorySlots;
using embedded_feature_config::kDirectFeatureCount;

constexpr size_t MAX_SECTIONS = static_cast<size_t>(EDR_PE_MAX_SECTIONS);

const std::unordered_set<std::string> kSuspiciousApis = {
  "virtualalloc", "virtualallocex", "virtualprotect", "virtualprotectex",
  "createremotethread", "ntcreatethreadex",
  "writeprocessmemory", "readprocessmemory", "openprocess",
  "ntunmapviewofsection", "ntmapviewofsection",
  "winexec", "shellexecutea", "shellexecutew", "shellexecuteex",
  "urldownloadtofilea", "urldownloadtofilew",
  "internetopena", "internetopenw", "internetopenurla",
  "httpsendrequesta", "httpopenrequesta",
  "cryptencrypt", "cryptdecrypt", "cryptgenkey",
  "regsetvalueexa", "regsetvalueexw",
  "regcreatekeyexa", "regcreatekeyexw",
  "adjusttokenprivileges", "lookupprivilegevaluea",
  "getasynckeystate", "setwindowshookexa", "setwindowshookexw",
  "isdebuggerpresent", "checkremotedebuggerpresent",
  "ntqueryinformationprocess", "ntsetinformationthread",
  "loadlibrarya", "loadlibraryw", "loadlibraryexa",
  "getprocaddress", "ldrloaddll",
  "createservicea", "createservicew",
};

const std::unordered_map<std::string, uint16_t> kDirectFeatureNameToIndex = {
  {"coff_machine", embedded_feature_config::D_COFF_MACHINE},
  {"coff_num_sections", embedded_feature_config::D_COFF_NUM_SECTIONS},
  {"coff_timestamp", embedded_feature_config::D_COFF_TIMESTAMP},
  {"coff_sizeof_opt_header", embedded_feature_config::D_COFF_SIZEOF_OPT_HEADER},
  {"coff_characteristics", embedded_feature_config::D_COFF_CHARACTERISTICS},
  {"opt_sizeof_init_data", embedded_feature_config::D_OPT_SIZEOF_INIT_DATA},
  {"opt_checksum", embedded_feature_config::D_OPT_CHECKSUM},
  {"opt_section_alignment", embedded_feature_config::D_OPT_SECTION_ALIGNMENT},
  {"opt_imagebase", embedded_feature_config::D_OPT_IMAGEBASE},
  {"opt_subsystem", embedded_feature_config::D_OPT_SUBSYSTEM},
  {"checksum_matches", embedded_feature_config::D_CHECKSUM_MATCHES},
  {"sec_mean_entropy", embedded_feature_config::D_SEC_MEAN_ENTROPY},
  {"sec_max_entropy", embedded_feature_config::D_SEC_MAX_ENTROPY},
  {"has_resources", embedded_feature_config::D_HAS_RESOURCES},
  {"rsrc_has_version", embedded_feature_config::D_RSRC_HAS_VERSION},
  {"has_relocations", embedded_feature_config::D_HAS_RELOCATIONS},
  {"has_debug", embedded_feature_config::D_HAS_DEBUG},
  {"has_repro", embedded_feature_config::D_HAS_REPRO},
  {"rich_max_build_id", embedded_feature_config::D_RICH_MAX_BUILD_ID},
  {"has_overlay", embedded_feature_config::D_HAS_OVERLAY},
  {"overlay_size", embedded_feature_config::D_OVERLAY_SIZE},
  {"overlay_entropy", embedded_feature_config::D_OVERLAY_ENTROPY},
  {"num_certificates", embedded_feature_config::D_NUM_CERTIFICATES},
  {"sig_verified", embedded_feature_config::D_SIG_VERIFIED},
  {"dos_e_lfanew", embedded_feature_config::D_DOS_E_LFANEW},
  {"opt_sizeof_headers", embedded_feature_config::D_OPT_SIZEOF_HEADERS},
  {"num_suspicious_imports", embedded_feature_config::D_NUM_SUSPICIOUS_IMPORTS},
  {"has_pdb", embedded_feature_config::D_HAS_PDB},
  {"opt_sizeof_image", embedded_feature_config::D_OPT_SIZEOF_IMAGE},
  {"opt_major_linker", embedded_feature_config::D_OPT_MAJOR_LINKER},
  {"overlay_ratio", embedded_feature_config::D_OVERLAY_RATIO},
  {"num_write_sections", embedded_feature_config::D_NUM_WRITE_SECTIONS},
  {"num_sections", embedded_feature_config::D_COFF_NUM_SECTIONS},
};

const std::unordered_map<std::string, uint32_t> kCoffCharFlags = {
  {"EXECUTABLE_IMAGE", 0x0002u},
  {"DLL", 0x2000u},
  {"LARGE_ADDRESS_AWARE", 0x0020u},
  {"RELOCS_STRIPPED", 0x0001u},
  {"DEBUG_STRIPPED", 0x0200u},
  {"SYSTEM", 0x1000u},
};

const std::unordered_map<std::string, uint32_t> kOptDllCharFlags = {
  {"DYNAMIC_BASE", 0x0040u},
  {"NX_COMPAT", 0x0100u},
  {"GUARD_CF", 0x4000u},
  {"HIGH_ENTROPY_VA", 0x0020u},
  {"NO_SEH", 0x0400u},
  {"NO_BIND", 0x0800u},
  {"APPCONTAINER", 0x1000u},
  {"TERMINAL_SERVER_AWARE", 0x8000u},
  {"FORCE_INTEGRITY", 0x0080u},
  {"NO_ISOLATION", 0x0200u},
  {"WDM_DRIVER", 0x2000u},
};

const std::unordered_map<std::string, uint16_t> kDataDirectoryNameToIndex = {
  {"EXPORT_TABLE", 0u},
  {"IMPORT_TABLE", 1u},
  {"RESOURCE_TABLE", 2u},
  {"EXCEPTION_TABLE", 3u},
  {"CERTIFICATE_TABLE", 4u},
  {"BASE_RELOCATION_TABLE", 5u},
  {"DEBUG_DIR", 6u},
  {"ARCHITECTURE", 7u},
  {"GLOBAL_PTR", 8u},
  {"TLS_TABLE", 9u},
  {"LOAD_CONFIG_TABLE", 10u},
  {"BOUND_IMPORT", 11u},
  {"IAT", 12u},
  {"DELAY_IMPORT_DESCRIPTOR", 13u},
  {"CLR_RUNTIME_HEADER", 14u},
  {"RESERVED", 15u},
};

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

bool parse_index_suffix(const std::string& name,
                        const std::string& prefix,
                        uint16_t max_allowed,
                        uint16_t& out_index) {
  if (name.rfind(prefix, 0) != 0 || name.size() <= prefix.size()) {
    return false;
  }

  const std::string suffix = name.substr(prefix.size());
  try {
    const unsigned long parsed = std::stoul(suffix);
    if (parsed > static_cast<unsigned long>(max_allowed)) {
      return false;
    }
    out_index = static_cast<uint16_t>(parsed);
    return true;
  } catch (...) {
    return false;
  }
}

bool try_resolve_runtime_feature_spec(const std::string& feature_name,
                                      CompiledFeatureSpec& out_spec) {
  const auto direct_it = kDirectFeatureNameToIndex.find(feature_name);
  if (direct_it != kDirectFeatureNameToIndex.end()) {
    out_spec = CompiledFeatureSpec{CompiledFeatureKind::Direct, direct_it->second, 0u};
    return true;
  }

  if (feature_name.rfind("coff_char_", 0) == 0) {
    const std::string flag = feature_name.substr(std::strlen("coff_char_"));
    const auto coff_it = kCoffCharFlags.find(flag);
    if (coff_it == kCoffCharFlags.end()) {
      return false;
    }
    out_spec = CompiledFeatureSpec{CompiledFeatureKind::CoffChar, 0u, coff_it->second};
    return true;
  }

  if (feature_name.rfind("opt_dllchar_", 0) == 0) {
    const std::string flag = feature_name.substr(std::strlen("opt_dllchar_"));
    const auto opt_it = kOptDllCharFlags.find(flag);
    if (opt_it == kOptDllCharFlags.end()) {
      return false;
    }
    out_spec = CompiledFeatureSpec{CompiledFeatureKind::OptDllChar, 0u, opt_it->second};
    return true;
  }

  uint16_t index = 0u;
  if (parse_index_suffix(feature_name, "sec_name_hash_", 31u, index)) {
    out_spec = CompiledFeatureSpec{CompiledFeatureKind::SecNameHash, index, 0u};
    return true;
  }
  if (parse_index_suffix(feature_name, "imp_dll_hash_", 63u, index)) {
    out_spec = CompiledFeatureSpec{CompiledFeatureKind::ImpDllHash, index, 0u};
    return true;
  }
  if (parse_index_suffix(feature_name, "imp_func_hash_", 255u, index)) {
    out_spec = CompiledFeatureSpec{CompiledFeatureKind::ImpFuncHash, index, 0u};
    return true;
  }

  if (feature_name.rfind("sec", 0) == 0) {
    const size_t underscore = feature_name.find('_', 3u);
    if (underscore != std::string::npos && underscore > 3u) {
      try {
        const unsigned long section_index = std::stoul(feature_name.substr(3u, underscore - 3u));
        if (section_index <= 9u) {
          const std::string suffix = feature_name.substr(underscore + 1u);
          if (suffix == "entropy") {
            out_spec = CompiledFeatureSpec{CompiledFeatureKind::SecEntropy, static_cast<uint16_t>(section_index), 0u};
            return true;
          }
          if (suffix == "vsize") {
            out_spec = CompiledFeatureSpec{CompiledFeatureKind::SecVsize, static_cast<uint16_t>(section_index), 0u};
            return true;
          }
          if (suffix == "is_write") {
            out_spec = CompiledFeatureSpec{CompiledFeatureKind::SecIsWrite, static_cast<uint16_t>(section_index), 0u};
            return true;
          }
        }
      } catch (...) {
        return false;
      }
    }
  }

  if (feature_name.rfind("dd_", 0) == 0) {
    constexpr const char* kRvaSuffix = "_rva";
    constexpr const char* kSizeSuffix = "_size";

    if (feature_name.size() > 3u + std::strlen(kRvaSuffix) &&
        feature_name.compare(feature_name.size() - std::strlen(kRvaSuffix), std::strlen(kRvaSuffix), kRvaSuffix) == 0) {
      const std::string dd_name = feature_name.substr(3u, feature_name.size() - 3u - std::strlen(kRvaSuffix));
      const auto dd_it = kDataDirectoryNameToIndex.find(dd_name);
      if (dd_it == kDataDirectoryNameToIndex.end()) {
        return false;
      }
      out_spec = CompiledFeatureSpec{CompiledFeatureKind::DataDirectoryRva, dd_it->second, 0u};
      return true;
    }

    if (feature_name.size() > 3u + std::strlen(kSizeSuffix) &&
        feature_name.compare(feature_name.size() - std::strlen(kSizeSuffix), std::strlen(kSizeSuffix), kSizeSuffix) == 0) {
      const std::string dd_name = feature_name.substr(3u, feature_name.size() - 3u - std::strlen(kSizeSuffix));
      const auto dd_it = kDataDirectoryNameToIndex.find(dd_name);
      if (dd_it == kDataDirectoryNameToIndex.end()) {
        return false;
      }
      out_spec = CompiledFeatureSpec{CompiledFeatureKind::DataDirectorySize, dd_it->second, 0u};
      return true;
    }
  }

  return false;
}

std::string clamp_error(const std::string& error) {
  const size_t max_error_bytes = static_cast<size_t>(EDR_PE_MAX_ERROR_TEXT_BYTES);
  if (max_error_bytes == 0 || error.size() <= max_error_bytes) {
    return error;
  }
  return error.substr(0, max_error_bytes);
}

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
void hash_update(
    std::array<double, N>& buckets,
    const std::string& input,
    bool lower,
    size_t& hash_updates,
    bool& hash_budget_exhausted) {
  if (input.empty()) {
    return;
  }

  const size_t max_hash_updates = static_cast<size_t>(EDR_PE_MAX_HASH_UPDATES_PER_FILE);
  if (hash_updates >= max_hash_updates) {
    hash_budget_exhausted = true;
    return;
  }

  const std::string payload = lower ? lower_ascii(input) : input;
  const size_t max_hashable_name_bytes = static_cast<size_t>(EDR_PE_MAX_HASHABLE_NAME_BYTES);
  const size_t input_len = std::min(payload.size(), max_hashable_name_bytes);
  if (input_len == 0) {
    return;
  }
  const auto* data = reinterpret_cast<const uint8_t*>(payload.data());

  const auto md5_digest = LIEF::PE::Signature::hash(data, input_len, LIEF::PE::ALGORITHMS::MD5);
  const auto sha1_digest = LIEF::PE::Signature::hash(data, input_len, LIEF::PE::ALGORITHMS::SHA_1);
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
  hash_updates += 1;
}

ExtractResult extract_row(const std::string& filepath) {
  ExtractResult result;
  const auto started = std::chrono::steady_clock::now();
  bool truncated = false;
  size_t hash_updates = 0;

  try {
    const size_t max_path_bytes = static_cast<size_t>(EDR_PE_MAX_PATH_BYTES);
    if (filepath.empty() || filepath.size() > max_path_bytes) {
      result.error = "path_limit";
      const auto ended = std::chrono::steady_clock::now();
      result.processing_time_ms = std::chrono::duration<double, std::milli>(ended - started).count();
      return result;
    }

    const std::filesystem::path fs_path(filepath);
    std::error_code stat_ec;
    const bool exists = std::filesystem::exists(fs_path, stat_ec);
    if (stat_ec || !exists) {
      result.error = "file_not_found";
      const auto ended = std::chrono::steady_clock::now();
      result.processing_time_ms = std::chrono::duration<double, std::milli>(ended - started).count();
      return result;
    }

    const uintmax_t file_size_raw = std::filesystem::file_size(fs_path, stat_ec);
    if (stat_ec) {
      result.error = "file_stat_failed";
      const auto ended = std::chrono::steady_clock::now();
      result.processing_time_ms = std::chrono::duration<double, std::milli>(ended - started).count();
      return result;
    }

    const uintmax_t min_input_bytes = static_cast<uintmax_t>(EDR_PE_MIN_INPUT_FILE_BYTES);
    const uintmax_t max_input_bytes = static_cast<uintmax_t>(EDR_PE_MAX_INPUT_FILE_BYTES);
    const uintmax_t max_working_set_bytes = static_cast<uintmax_t>(EDR_PE_MAX_WORKING_SET_BYTES);
    if (file_size_raw < min_input_bytes) {
      result.error = "file_too_small";
      const auto ended = std::chrono::steady_clock::now();
      result.processing_time_ms = std::chrono::duration<double, std::milli>(ended - started).count();
      return result;
    }
    if (file_size_raw > max_input_bytes) {
      result.error = "file_too_large";
      const auto ended = std::chrono::steady_clock::now();
      result.processing_time_ms = std::chrono::duration<double, std::milli>(ended - started).count();
      return result;
    }
    if (file_size_raw > max_working_set_bytes) {
      result.error = "memory_budget_exceeded";
      const auto ended = std::chrono::steady_clock::now();
      result.processing_time_ms = std::chrono::duration<double, std::milli>(ended - started).count();
      return result;
    }

    std::unique_ptr<LIEF::PE::Binary> pe = LIEF::PE::Parser::parse(filepath);
    if (!pe) {
      result.error = "parse_failed";
    } else {
      result.parse_ok = true;
      const auto& header = pe->header();
      const auto& dos = pe->dos_header();
      const auto& opt = pe->optional_header();

      result.direct[embedded_feature_config::D_COFF_MACHINE] = static_cast<double>(static_cast<uint32_t>(header.machine()));
      result.direct[embedded_feature_config::D_COFF_NUM_SECTIONS] = static_cast<double>(header.numberof_sections());
      result.direct[embedded_feature_config::D_COFF_TIMESTAMP] = static_cast<double>(header.time_date_stamp());
      result.direct[embedded_feature_config::D_COFF_SIZEOF_OPT_HEADER] = static_cast<double>(header.sizeof_optional_header());
      result.direct[embedded_feature_config::D_COFF_CHARACTERISTICS] = static_cast<double>(header.characteristics());
      result.direct[embedded_feature_config::D_OPT_SIZEOF_INIT_DATA] = static_cast<double>(opt.sizeof_initialized_data());
      result.direct[embedded_feature_config::D_DOS_E_LFANEW] = static_cast<double>(dos.addressof_new_exeheader());
      result.direct[embedded_feature_config::D_OPT_SIZEOF_HEADERS] = static_cast<double>(opt.sizeof_headers());
      result.direct[embedded_feature_config::D_OPT_CHECKSUM] = static_cast<double>(opt.checksum());
      result.direct[embedded_feature_config::D_OPT_SECTION_ALIGNMENT] = static_cast<double>(opt.section_alignment());
      result.direct[embedded_feature_config::D_OPT_IMAGEBASE] = static_cast<double>(opt.imagebase());
      result.direct[embedded_feature_config::D_OPT_SUBSYSTEM] = static_cast<double>(static_cast<uint32_t>(opt.subsystem()));
      result.direct[embedded_feature_config::D_OPT_SIZEOF_IMAGE] = static_cast<double>(opt.sizeof_image());
      result.direct[embedded_feature_config::D_OPT_MAJOR_LINKER] = static_cast<double>(opt.major_linker_version());

      result.coff_characteristics = header.characteristics();
      result.opt_dll_characteristics = opt.dll_characteristics();

      try {
        result.direct[embedded_feature_config::D_CHECKSUM_MATCHES] = (opt.checksum() == pe->compute_checksum()) ? 1.0 : 0.0;
      } catch (...) {
        result.direct[embedded_feature_config::D_CHECKSUM_MATCHES] = 0.0;
      }

      size_t section_count = 0;
      size_t write_section_count = 0;
      double entropy_sum = 0.0;
      double entropy_max = 0.0;

      for (const LIEF::PE::Section& sec : pe->sections()) {
        if (section_count >= MAX_SECTIONS) {
          truncated = true;
          break;
        }

        const double entropy = sec.entropy();
        result.sec_entropy[section_count] = entropy;
        result.sec_vsize[section_count] = static_cast<double>(sec.virtual_size());
        const bool is_write_section = sec.has_characteristic(LIEF::PE::Section::CHARACTERISTICS::MEM_WRITE);
        result.sec_is_write[section_count] = is_write_section ? 1.0 : 0.0;
        if (is_write_section) {
          write_section_count += 1;
        }

        bool hash_budget_exhausted = false;
        hash_update(result.sec_name_hash, sec.name(), false, hash_updates, hash_budget_exhausted);
        if (hash_budget_exhausted) {
          truncated = true;
        }

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
      result.direct[embedded_feature_config::D_NUM_WRITE_SECTIONS] = static_cast<double>(write_section_count);

      result.direct[embedded_feature_config::D_HAS_RESOURCES] = pe->has_resources() ? 1.0 : 0.0;
      if (pe->has_resources()) {
        if (auto manager = pe->resources_manager()) {
          result.direct[embedded_feature_config::D_RSRC_HAS_VERSION] = manager->has_version() ? 1.0 : 0.0;
        }
      }

      result.direct[embedded_feature_config::D_HAS_RELOCATIONS] = pe->has_relocations() ? 1.0 : 0.0;
      result.direct[embedded_feature_config::D_HAS_DEBUG] = pe->has_debug() ? 1.0 : 0.0;
      result.direct[embedded_feature_config::D_HAS_PDB] = (pe->codeview_pdb() != nullptr) ? 1.0 : 0.0;

      if (pe->has_debug()) {
        bool has_repro = false;
        size_t debug_count = 0;
        const size_t max_debug_entries = static_cast<size_t>(EDR_PE_MAX_DEBUG_ENTRIES);
        for (const LIEF::PE::Debug& debug_entry : pe->debug()) {
          if (debug_count >= max_debug_entries) {
            truncated = true;
            break;
          }
          if (debug_entry.type() == LIEF::PE::Debug::TYPES::REPRO) {
            has_repro = true;
            break;
          }
          debug_count += 1;
        }
        result.direct[embedded_feature_config::D_HAS_REPRO] = has_repro ? 1.0 : 0.0;
      }

      if (pe->has_rich_header()) {
        if (const LIEF::PE::RichHeader* rich = pe->rich_header()) {
          uint32_t max_build_id = 0;
          size_t rich_count = 0;
          const size_t max_rich_entries = static_cast<size_t>(EDR_PE_MAX_RICH_ENTRIES);
          for (const LIEF::PE::RichEntry& entry : rich->entries()) {
            if (rich_count >= max_rich_entries) {
              truncated = true;
              break;
            }
            max_build_id = std::max(max_build_id, static_cast<uint32_t>(entry.build_id()));
            rich_count += 1;
          }
          result.direct[embedded_feature_config::D_RICH_MAX_BUILD_ID] = static_cast<double>(max_build_id);
        }
      }

      const auto overlay = pe->overlay();
      result.direct[embedded_feature_config::D_HAS_OVERLAY] = overlay.empty() ? 0.0 : 1.0;
      result.direct[embedded_feature_config::D_OVERLAY_SIZE] = static_cast<double>(overlay.size());
      result.direct[embedded_feature_config::D_OVERLAY_ENTROPY] =
          overlay.empty() ? 0.0 : entropy_from_span(overlay, static_cast<size_t>(EDR_PE_MAX_OVERLAY_ENTROPY_BYTES));
        result.direct[embedded_feature_config::D_OVERLAY_RATIO] =
          (file_size_raw > 0u) ? (static_cast<double>(overlay.size()) / static_cast<double>(file_size_raw)) : 0.0;

      if (pe->has_imports()) {
        size_t dll_count = 0;
        size_t import_func_count = 0;
        size_t suspicious_import_count = 0;
        const size_t max_import_dlls = static_cast<size_t>(EDR_PE_MAX_IMPORT_DLLS);
        const size_t max_import_funcs_total = static_cast<size_t>(EDR_PE_MAX_IMPORT_FUNCS_TOTAL);
        for (const LIEF::PE::Import& imp : pe->imports()) {
          if (dll_count >= max_import_dlls) {
            truncated = true;
            break;
          }

          bool hash_budget_exhausted = false;
          hash_update(result.imp_dll_hash, imp.name(), true, hash_updates, hash_budget_exhausted);
          if (hash_budget_exhausted) {
            truncated = true;
          }
          dll_count += 1;

          for (const LIEF::PE::ImportEntry& entry : imp.entries()) {
            if (import_func_count >= max_import_funcs_total) {
              truncated = true;
              break;
            }
            if (!entry.is_ordinal() && !entry.name().empty()) {
              const std::string lowered_name = lower_ascii(entry.name());
              if (kSuspiciousApis.find(lowered_name) != kSuspiciousApis.end()) {
                suspicious_import_count += 1;
              }

              hash_budget_exhausted = false;
              hash_update(result.imp_func_hash, entry.name(), true, hash_updates, hash_budget_exhausted);
              if (hash_budget_exhausted) {
                truncated = true;
              }
              import_func_count += 1;
            }
          }

          if (import_func_count >= max_import_funcs_total) {
            break;
          }
        }

        result.direct[embedded_feature_config::D_NUM_SUSPICIOUS_IMPORTS] = static_cast<double>(suspicious_import_count);
      }

      if (pe->has_signatures()) {
        size_t certificate_count = 0;
        size_t signature_count = 0;
        const size_t max_signatures = static_cast<size_t>(EDR_PE_MAX_SIGNATURES);
        const size_t max_certificates_total = static_cast<size_t>(EDR_PE_MAX_CERTIFICATES_TOTAL);
        for (const LIEF::PE::Signature& signature : pe->signatures()) {
          if (signature_count >= max_signatures) {
            truncated = true;
            break;
          }
          certificate_count += signature.certificates().size();
          if (certificate_count >= max_certificates_total) {
            certificate_count = max_certificates_total;
            truncated = true;
            break;
          }
          signature_count += 1;
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

      const size_t max_data_directories = std::min(
          static_cast<size_t>(kDataDirectorySlots),
          static_cast<size_t>(EDR_PE_MAX_DATA_DIRECTORIES));
      size_t data_directory_count = 0;
      for (const LIEF::PE::DataDirectory& directory : pe->data_directories()) {
        if (data_directory_count >= max_data_directories) {
          truncated = true;
          break;
        }
        const size_t idx = static_cast<size_t>(directory.type());
        if (idx < kDataDirectorySlots) {
          result.dd_rva[idx] = static_cast<double>(directory.RVA());
          result.dd_size[idx] = static_cast<double>(directory.size());
        }
        data_directory_count += 1;
      }

      if (truncated) {
        result.error = "resource_limit";
      }
    }
  } catch (const std::exception& ex) {
    result.error = ex.what();
  } catch (...) {
    result.error = "unknown_exception";
  }

  result.error = clamp_error(result.error);

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

extractor::FeatureVector make_feature_vector_dynamic(const ExtractResult& row,
                                                     const std::vector<std::string>& feature_names,
                                                     bool& out_ok,
                                                     std::string& out_error) {
  extractor::FeatureVector vector;
  vector.values.resize(feature_names.size(), 0.0);
  out_ok = true;
  out_error.clear();

  for (size_t i = 0; i < feature_names.size(); ++i) {
    CompiledFeatureSpec spec{};
    if (!try_resolve_runtime_feature_spec(feature_names[i], spec)) {
      out_ok = false;
      out_error = "unsupported_feature:" + feature_names[i];
      vector.values.clear();
      return vector;
    }
    vector.values[i] = resolve_value(row, spec);
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

FeatureVector PEExtractor::extract_selected(const std::string& path,
                                            const std::vector<std::string>& feature_names) const {
  const ExtractionReport report = extract_selected_with_metadata(path, feature_names);
  if (!report.metadata.parse_ok) {
    throw std::runtime_error(
      report.metadata.error.empty() ? "extractor_parse_failed" : report.metadata.error
    );
  }
  return report.feature_vector;
}

ExtractionReport PEExtractor::extract_selected_with_metadata(
    const std::string& path,
    const std::vector<std::string>& feature_names) const {
  const ExtractResult row = extract_row(path);

  ExtractionReport report;
  report.metadata.parse_ok = row.parse_ok;
  report.metadata.error = row.error;
  report.metadata.processing_time_ms = row.processing_time_ms;

  if (!row.parse_ok) {
    report.feature_vector.values.assign(feature_names.size(), 0.0);
    return report;
  }

  bool features_ok = false;
  std::string feature_error;
  report.feature_vector = make_feature_vector_dynamic(row, feature_names, features_ok, feature_error);
  if (!features_ok) {
    report.metadata.parse_ok = false;
    report.metadata.error = feature_error;
  }

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
