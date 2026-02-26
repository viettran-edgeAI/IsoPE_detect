#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <LIEF/PE.hpp>
#include <LIEF/PE/signature/Signature.hpp>

#include "../../base/eml_base.h"

namespace eml {

namespace detail {

// --- Resource Limits ---
#ifndef EDR_PE_MAX_INPUT_FILE_BYTES
#define EDR_PE_MAX_INPUT_FILE_BYTES (128ULL * 1024ULL * 1024ULL)
#endif
#ifndef EDR_PE_MAX_WORKING_SET_BYTES
#define EDR_PE_MAX_WORKING_SET_BYTES (256ULL * 1024ULL * 1024ULL)
#endif
#ifndef EDR_PE_MIN_INPUT_FILE_BYTES
#define EDR_PE_MIN_INPUT_FILE_BYTES 512ULL
#endif
#ifndef EDR_PE_MAX_PATH_BYTES
#define EDR_PE_MAX_PATH_BYTES 4096ULL
#endif
#ifndef EDR_PE_MAX_SECTIONS
#define EDR_PE_MAX_SECTIONS 10ULL
#endif
#ifndef EDR_PE_MAX_IMPORT_DLLS
#define EDR_PE_MAX_IMPORT_DLLS 512ULL
#endif
#ifndef EDR_PE_MAX_IMPORT_FUNCS_TOTAL
#define EDR_PE_MAX_IMPORT_FUNCS_TOTAL 8192ULL
#endif
#ifndef EDR_PE_MAX_HASHABLE_NAME_BYTES
#define EDR_PE_MAX_HASHABLE_NAME_BYTES 256ULL
#endif
#ifndef EDR_PE_MAX_HASH_UPDATES_PER_FILE
#define EDR_PE_MAX_HASH_UPDATES_PER_FILE 10000ULL
#endif
#ifndef EDR_PE_MAX_DEBUG_ENTRIES
#define EDR_PE_MAX_DEBUG_ENTRIES 16ULL
#endif
#ifndef EDR_PE_MAX_RICH_ENTRIES
#define EDR_PE_MAX_RICH_ENTRIES 1024ULL
#endif
#ifndef EDR_PE_MAX_OVERLAY_ENTROPY_BYTES
#define EDR_PE_MAX_OVERLAY_ENTROPY_BYTES (1ULL * 1024ULL * 1024ULL)
#endif
#ifndef EDR_PE_MAX_SIGNATURES
#define EDR_PE_MAX_SIGNATURES 16ULL
#endif
#ifndef EDR_PE_MAX_CERTIFICATES_TOTAL
#define EDR_PE_MAX_CERTIFICATES_TOTAL 64ULL
#endif
#ifndef EDR_PE_MAX_DATA_DIRECTORIES
#define EDR_PE_MAX_DATA_DIRECTORIES 16ULL
#endif
#ifndef EDR_PE_MAX_ERROR_TEXT_BYTES
#define EDR_PE_MAX_ERROR_TEXT_BYTES 128ULL
#endif

// --- Feature Configuration ---
enum class CompiledFeatureKind : uint8_t {
    Direct = 0,
    CoffChar = 1,
    OptDllChar = 2,
    SecEntropy = 3,
    SecVsize = 4,
    SecIsWrite = 5,
    SecNameHash = 6,
    ImpDllHash = 7,
    ImpFuncHash = 8,
    DataDirectoryRva = 9,
    DataDirectorySize = 10,
};

struct CompiledFeatureSpec {
    CompiledFeatureKind kind;
    uint16_t index;
    uint32_t value;
};

enum DirectFeatureId : uint16_t {
    D_COFF_MACHINE = 0,
    D_COFF_NUM_SECTIONS = 1,
    D_COFF_TIMESTAMP = 2,
    D_COFF_SIZEOF_OPT_HEADER = 3,
    D_COFF_CHARACTERISTICS = 4,
    D_OPT_SIZEOF_INIT_DATA = 5,
    D_OPT_CHECKSUM = 6,
    D_OPT_SECTION_ALIGNMENT = 7,
    D_OPT_IMAGEBASE = 8,
    D_OPT_SUBSYSTEM = 9,
    D_CHECKSUM_MATCHES = 10,
    D_SEC_MEAN_ENTROPY = 11,
    D_SEC_MAX_ENTROPY = 12,
    D_HAS_RESOURCES = 13,
    D_RSRC_HAS_VERSION = 14,
    D_HAS_RELOCATIONS = 15,
    D_HAS_DEBUG = 16,
    D_HAS_REPRO = 17,
    D_RICH_MAX_BUILD_ID = 18,
    D_HAS_OVERLAY = 19,
    D_OVERLAY_SIZE = 20,
    D_OVERLAY_ENTROPY = 21,
    D_NUM_CERTIFICATES = 22,
    D_SIG_VERIFIED = 23,
    D_DOS_E_LFANEW = 24,
    D_OPT_SIZEOF_HEADERS = 25,
    D_NUM_SUSPICIOUS_IMPORTS = 26,
    D_HAS_PDB = 27,
    D_OPT_SIZEOF_IMAGE = 28,
    D_OPT_MAJOR_LINKER = 29,
    D_OVERLAY_RATIO = 30,
    D_NUM_WRITE_SECTIONS = 31,
    DIRECT_FEATURE_COUNT
};

constexpr size_t kDataDirectorySlots = 16;
constexpr size_t kDirectFeatureCount = static_cast<size_t>(DIRECT_FEATURE_COUNT);

// --- Static Data ---
inline const std::unordered_set<std::string>& get_suspicious_apis() {
    static const std::unordered_set<std::string> apis = {
        "virtualalloc", "virtualallocex", "virtualprotect", "virtualprotectex",
        "createremotethread", "ntcreatethreadex", "writeprocessmemory", "readprocessmemory",
        "openprocess", "ntunmapviewofsection", "ntmapviewofsection", "winexec",
        "shellexecutea", "shellexecutew", "shellexecuteex", "urldownloadtofilea",
        "urldownloadtofilew", "internetopena", "internetopenw", "internetopenurla",
        "httpsendrequesta", "httpopenrequesta", "cryptencrypt", "cryptdecrypt",
        "cryptgenkey", "regsetvalueexa", "regsetvalueexw", "regcreatekeyexa",
        "regcreatekeyexw", "adjusttokenprivileges", "lookupprivilegevaluea",
        "getasynckeystate", "setwindowshookexa", "setwindowshookexw", "isdebuggerpresent",
        "checkremotedebuggerpresent", "ntqueryinformationprocess", "ntsetinformationthread",
        "loadlibrarya", "loadlibraryw", "loadlibraryexa", "getprocaddress", "ldrloaddll",
        "createservicea", "createservicew",
    };
    return apis;
}

inline const std::unordered_map<std::string, uint16_t>& get_direct_feature_map() {
    static const std::unordered_map<std::string, uint16_t> m = {
        {"coff_machine", D_COFF_MACHINE}, {"coff_num_sections", D_COFF_NUM_SECTIONS},
        {"coff_timestamp", D_COFF_TIMESTAMP}, {"coff_sizeof_opt_header", D_COFF_SIZEOF_OPT_HEADER},
        {"coff_characteristics", D_COFF_CHARACTERISTICS}, {"opt_sizeof_init_data", D_OPT_SIZEOF_INIT_DATA},
        {"opt_checksum", D_OPT_CHECKSUM}, {"opt_section_alignment", D_OPT_SECTION_ALIGNMENT},
        {"opt_imagebase", D_OPT_IMAGEBASE}, {"opt_subsystem", D_OPT_SUBSYSTEM},
        {"checksum_matches", D_CHECKSUM_MATCHES}, {"sec_mean_entropy", D_SEC_MEAN_ENTROPY},
        {"sec_max_entropy", D_SEC_MAX_ENTROPY}, {"has_resources", D_HAS_RESOURCES},
        {"rsrc_has_version", D_RSRC_HAS_VERSION}, {"has_relocations", D_HAS_RELOCATIONS},
        {"has_debug", D_HAS_DEBUG}, {"has_repro", D_HAS_REPRO}, {"rich_max_build_id", D_RICH_MAX_BUILD_ID},
        {"has_overlay", D_HAS_OVERLAY}, {"overlay_size", D_OVERLAY_SIZE}, {"overlay_entropy", D_OVERLAY_ENTROPY},
        {"num_certificates", D_NUM_CERTIFICATES}, {"sig_verified", D_SIG_VERIFIED},
        {"dos_e_lfanew", D_DOS_E_LFANEW}, {"opt_sizeof_headers", D_OPT_SIZEOF_HEADERS},
        {"num_suspicious_imports", D_NUM_SUSPICIOUS_IMPORTS}, {"has_pdb", D_HAS_PDB},
        {"opt_sizeof_image", D_OPT_SIZEOF_IMAGE}, {"opt_major_linker", D_OPT_MAJOR_LINKER},
        {"overlay_ratio", D_OVERLAY_RATIO}, {"num_write_sections", D_NUM_WRITE_SECTIONS},
        {"num_sections", D_COFF_NUM_SECTIONS},
    };
    return m;
}

inline const std::unordered_map<std::string, uint32_t>& get_coff_char_map() {
    static const std::unordered_map<std::string, uint32_t> m = {
        {"EXECUTABLE_IMAGE", 0x0002u}, {"DLL", 0x2000u}, {"LARGE_ADDRESS_AWARE", 0x0020u},
        {"RELOCS_STRIPPED", 0x0001u}, {"DEBUG_STRIPPED", 0x0200u}, {"SYSTEM", 0x1000u},
    };
    return m;
}

inline const std::unordered_map<std::string, uint32_t>& get_opt_dll_char_map() {
    static const std::unordered_map<std::string, uint32_t> m = {
        {"DYNAMIC_BASE", 0x0040u}, {"NX_COMPAT", 0x0100u}, {"GUARD_CF", 0x4000u},
        {"HIGH_ENTROPY_VA", 0x0020u}, {"NO_SEH", 0x0400u}, {"NO_BIND", 0x0800u},
        {"APPCONTAINER", 0x1000u}, {"TERMINAL_SERVER_AWARE", 0x8000u}, {"FORCE_INTEGRITY", 0x0080u},
        {"NO_ISOLATION", 0x0200u}, {"WDM_DRIVER", 0x2000u},
    };
    return m;
}

inline const std::unordered_map<std::string, uint32_t>& get_data_dir_map() {
    static const std::unordered_map<std::string, uint32_t> m = {
        {"EXPORT_TABLE", 0u}, {"IMPORT_TABLE", 1u}, {"RESOURCE_TABLE", 2u}, {"EXCEPTION_TABLE", 3u},
        {"CERTIFICATE_TABLE", 4u}, {"BASE_RELOCATION_TABLE", 5u}, {"DEBUG_DIR", 6u},
        {"ARCHITECTURE", 7u}, {"GLOBAL_PTR", 8u}, {"TLS_TABLE", 9u}, {"LOAD_CONFIG_TABLE", 10u},
        {"BOUND_IMPORT", 11u}, {"IAT", 12u}, {"DELAY_IMPORT_DESCRIPTOR", 13u},
        {"CLR_RUNTIME_HEADER", 14u}, {"RESERVED", 15u},
    };
    return m;
}

// --- Utils ---
inline std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

inline double entropy_from_span(LIEF::span<const uint8_t> data, size_t max_bytes = std::numeric_limits<size_t>::max()) {
    const size_t usable = std::min(data.size(), max_bytes);
    if (usable == 0) return 0.0;
    std::array<size_t, 256> counts{};
    for (size_t index = 0; index < usable; ++index) counts[data[index]] += 1;
    double entropy = 0.0;
    for (size_t count : counts) {
        if (count == 0) continue;
        const double probability = static_cast<double>(count) / static_cast<double>(usable);
        entropy -= probability * std::log2(probability);
    }
    return entropy;
}

template <size_t N>
inline void hash_update(std::array<double, N>& buckets, const std::string& input, bool lower,
                        size_t& hash_updates, bool& hash_budget_exhausted) {
    if (input.empty()) return;
    const size_t max_hash_updates = static_cast<size_t>(EDR_PE_MAX_HASH_UPDATES_PER_FILE);
    if (hash_updates >= max_hash_updates) { hash_budget_exhausted = true; return; }
    const std::string payload = lower ? lower_ascii(input) : input;
    const size_t max_hashable_name_bytes = static_cast<size_t>(EDR_PE_MAX_HASHABLE_NAME_BYTES);
    const size_t input_len = std::min(payload.size(), max_hashable_name_bytes);
    if (input_len == 0) return;
    const auto* data = reinterpret_cast<const uint8_t*>(payload.data());
    const auto md5_digest = LIEF::PE::Signature::hash(data, input_len, LIEF::PE::ALGORITHMS::MD5);
    const auto sha1_digest = LIEF::PE::Signature::hash(data, input_len, LIEF::PE::ALGORITHMS::SHA_1);
    if (md5_digest.size() < sizeof(uint64_t) || sha1_digest.size() < sizeof(uint64_t)) return;
    uint64_t h1 = 0, h2 = 0;
    std::memcpy(&h1, md5_digest.data(), sizeof(uint64_t));
    std::memcpy(&h2, sha1_digest.data(), sizeof(uint64_t));
    const size_t bucket = static_cast<size_t>(h1 % N);
    const double sign = (h2 % 2 == 0) ? 1.0 : -1.0;
    buckets[bucket] += sign;
    hash_updates += 1;
}

struct ExtractResult {
    bool parse_ok = false;
    std::string error;
    double processing_time_ms = 0.0;
    std::array<double, kDirectFeatureCount> direct{};
    uint32_t coff_characteristics = 0;
    uint32_t opt_dll_characteristics = 0;
    std::array<double, static_cast<size_t>(EDR_PE_MAX_SECTIONS)> sec_entropy{};
    std::array<double, static_cast<size_t>(EDR_PE_MAX_SECTIONS)> sec_vsize{};
    std::array<double, static_cast<size_t>(EDR_PE_MAX_SECTIONS)> sec_is_write{};
    std::array<double, 32> sec_name_hash{};
    std::array<double, 64> imp_dll_hash{};
    std::array<double, 256> imp_func_hash{};
    std::array<double, kDataDirectorySlots> dd_rva{};
    std::array<double, kDataDirectorySlots> dd_size{};
};

inline ExtractResult extract_row_from_lief_binary(LIEF::PE::Binary* pe, uintmax_t file_size_raw) {
    ExtractResult result;
    if (!pe) { result.error = "parse_failed"; return result; }
    result.parse_ok = true;
    bool truncated = false;
    size_t hash_updates = 0;

    const auto& header = pe->header();
    const auto& dos = pe->dos_header();
    const auto& opt = pe->optional_header();

    result.direct[D_COFF_MACHINE] = static_cast<double>(static_cast<uint32_t>(header.machine()));
    result.direct[D_COFF_NUM_SECTIONS] = static_cast<double>(header.numberof_sections());
    result.direct[D_COFF_TIMESTAMP] = static_cast<double>(header.time_date_stamp());
    result.direct[D_COFF_SIZEOF_OPT_HEADER] = static_cast<double>(header.sizeof_optional_header());
    result.direct[D_COFF_CHARACTERISTICS] = static_cast<double>(header.characteristics());
    result.direct[D_OPT_SIZEOF_INIT_DATA] = static_cast<double>(opt.sizeof_initialized_data());
    result.direct[D_DOS_E_LFANEW] = static_cast<double>(dos.addressof_new_exeheader());
    result.direct[D_OPT_SIZEOF_HEADERS] = static_cast<double>(opt.sizeof_headers());
    result.direct[D_OPT_CHECKSUM] = static_cast<double>(opt.checksum());
    result.direct[D_OPT_SECTION_ALIGNMENT] = static_cast<double>(opt.section_alignment());
    result.direct[D_OPT_IMAGEBASE] = static_cast<double>(opt.imagebase());
    result.direct[D_OPT_SUBSYSTEM] = static_cast<double>(static_cast<uint32_t>(opt.subsystem()));
    result.direct[D_OPT_SIZEOF_IMAGE] = static_cast<double>(opt.sizeof_image());
    result.direct[D_OPT_MAJOR_LINKER] = static_cast<double>(opt.major_linker_version());
    result.coff_characteristics = header.characteristics();
    result.opt_dll_characteristics = opt.dll_characteristics();

    try { result.direct[D_CHECKSUM_MATCHES] = (opt.checksum() == pe->compute_checksum()) ? 1.0 : 0.0; } catch (...) {}

    size_t section_count = 0, write_section_count = 0;
    double entropy_sum = 0.0, entropy_max = 0.0;
    for (const LIEF::PE::Section& sec : pe->sections()) {
        if (section_count >= static_cast<size_t>(EDR_PE_MAX_SECTIONS)) { truncated = true; break; }
        const double entropy = sec.entropy();
        result.sec_entropy[section_count] = entropy;
        result.sec_vsize[section_count] = static_cast<double>(sec.virtual_size());
        const bool is_write = sec.has_characteristic(LIEF::PE::Section::CHARACTERISTICS::MEM_WRITE);
        result.sec_is_write[section_count] = is_write ? 1.0 : 0.0;
        if (is_write) write_section_count++;
        bool hash_budget_exhausted = false;
        hash_update(result.sec_name_hash, sec.name(), false, hash_updates, hash_budget_exhausted);
        if (hash_budget_exhausted) truncated = true;
        entropy_sum += entropy;
        if (section_count == 0 || entropy > entropy_max) entropy_max = entropy;
        section_count++;
    }
    if (section_count > 0) {
        result.direct[D_SEC_MEAN_ENTROPY] = entropy_sum / static_cast<double>(section_count);
        result.direct[D_SEC_MAX_ENTROPY] = entropy_max;
    }
    result.direct[D_NUM_WRITE_SECTIONS] = static_cast<double>(write_section_count);
    result.direct[D_HAS_RESOURCES] = pe->has_resources() ? 1.0 : 0.0;
    if (pe->has_resources()) if (auto manager = pe->resources_manager()) result.direct[D_RSRC_HAS_VERSION] = manager->has_version() ? 1.0 : 0.0;
    result.direct[D_HAS_RELOCATIONS] = pe->has_relocations() ? 1.0 : 0.0;
    result.direct[D_HAS_DEBUG] = pe->has_debug() ? 1.0 : 0.0;
    result.direct[D_HAS_PDB] = (pe->codeview_pdb() != nullptr) ? 1.0 : 0.0;
    if (pe->has_debug()) {
        bool has_repro = false; size_t debug_count = 0;
        for (const LIEF::PE::Debug& entry : pe->debug()) {
            if (debug_count >= static_cast<size_t>(EDR_PE_MAX_DEBUG_ENTRIES)) { truncated = true; break; }
            if (entry.type() == LIEF::PE::Debug::TYPES::REPRO) has_repro = true;
            debug_count++;
        }
        result.direct[D_HAS_REPRO] = has_repro ? 1.0 : 0.0;
    }
    if (pe->has_rich_header()) if (const LIEF::PE::RichHeader* rich = pe->rich_header()) {
        uint32_t max_build_id = 0; size_t rich_count = 0;
        for (const LIEF::PE::RichEntry& entry : rich->entries()) {
            if (rich_count >= static_cast<size_t>(EDR_PE_MAX_RICH_ENTRIES)) { truncated = true; break; }
            max_build_id = std::max(max_build_id, static_cast<uint32_t>(entry.build_id()));
            rich_count++;
        }
        result.direct[D_RICH_MAX_BUILD_ID] = static_cast<double>(max_build_id);
    }
    const auto overlay = pe->overlay();
    result.direct[D_HAS_OVERLAY] = overlay.empty() ? 0.0 : 1.0;
    result.direct[D_OVERLAY_SIZE] = static_cast<double>(overlay.size());
    result.direct[D_OVERLAY_ENTROPY] = overlay.empty() ? 0.0 : entropy_from_span(overlay, static_cast<size_t>(EDR_PE_MAX_OVERLAY_ENTROPY_BYTES));
    result.direct[D_OVERLAY_RATIO] = (file_size_raw > 0u) ? (static_cast<double>(overlay.size()) / static_cast<double>(file_size_raw)) : 0.0;

    if (pe->has_imports()) {
        size_t dll_count = 0, import_func_count = 0, suspicious_count = 0;
        const auto& suspicious_apis = get_suspicious_apis();
        for (const LIEF::PE::Import& imp : pe->imports()) {
            if (dll_count >= static_cast<size_t>(EDR_PE_MAX_IMPORT_DLLS)) { truncated = true; break; }
            bool hash_budget_exhausted = false;
            hash_update(result.imp_dll_hash, imp.name(), true, hash_updates, hash_budget_exhausted);
            if (hash_budget_exhausted) truncated = true;
            dll_count++;
            for (const LIEF::PE::ImportEntry& entry : imp.entries()) {
                if (import_func_count >= static_cast<size_t>(EDR_PE_MAX_IMPORT_FUNCS_TOTAL)) { truncated = true; break; }
                if (!entry.is_ordinal() && !entry.name().empty()) {
                    if (suspicious_apis.count(lower_ascii(entry.name()))) suspicious_count++;
                    hash_budget_exhausted = false;
                    hash_update(result.imp_func_hash, entry.name(), true, hash_updates, hash_budget_exhausted);
                    if (hash_budget_exhausted) truncated = true;
                    import_func_count++;
                }
            }
            if (import_func_count >= static_cast<size_t>(EDR_PE_MAX_IMPORT_FUNCS_TOTAL)) break;
        }
        result.direct[D_NUM_SUSPICIOUS_IMPORTS] = static_cast<double>(suspicious_count);
    }
    if (pe->has_signatures()) {
        size_t certificate_count = 0, signature_count = 0;
        for (const LIEF::PE::Signature& signature : pe->signatures()) {
            if (signature_count >= static_cast<size_t>(EDR_PE_MAX_SIGNATURES)) { truncated = true; break; }
            certificate_count += signature.certificates().size();
            if (certificate_count >= static_cast<size_t>(EDR_PE_MAX_CERTIFICATES_TOTAL)) { certificate_count = static_cast<size_t>(EDR_PE_MAX_CERTIFICATES_TOTAL); truncated = true; break; }
            signature_count++;
        }
        result.direct[D_NUM_CERTIFICATES] = static_cast<double>(certificate_count);
        try { if (pe->verify_signature() == LIEF::PE::Signature::VERIFICATION_FLAGS::OK) result.direct[D_SIG_VERIFIED] = 1.0; } catch (...) {}
    }
    size_t data_directory_count = 0;
    for (const LIEF::PE::DataDirectory& directory : pe->data_directories()) {
        if (data_directory_count >= static_cast<size_t>(EDR_PE_MAX_DATA_DIRECTORIES)) { truncated = true; break; }
        const size_t idx = static_cast<size_t>(directory.type());
        if (idx < kDataDirectorySlots) { result.dd_rva[idx] = static_cast<double>(directory.RVA()); result.dd_size[idx] = static_cast<double>(directory.size()); }
        data_directory_count++;
    }
    if (truncated) result.error = "resource_limit";
    return result;
}

inline double resolve_spec_value(const ExtractResult& row, const CompiledFeatureSpec& spec) {
    switch (spec.kind) {
        case CompiledFeatureKind::Direct: return (spec.index < row.direct.size()) ? row.direct[spec.index] : 0.0;
        case CompiledFeatureKind::CoffChar: return (row.coff_characteristics & spec.value) ? 1.0 : 0.0;
        case CompiledFeatureKind::OptDllChar: return (row.opt_dll_characteristics & spec.value) ? 1.0 : 0.0;
        case CompiledFeatureKind::SecEntropy: return (spec.index < row.sec_entropy.size()) ? row.sec_entropy[spec.index] : 0.0;
        case CompiledFeatureKind::SecVsize: return (spec.index < row.sec_vsize.size()) ? row.sec_vsize[spec.index] : 0.0;
        case CompiledFeatureKind::SecIsWrite: return (spec.index < row.sec_is_write.size()) ? row.sec_is_write[spec.index] : 0.0;
        case CompiledFeatureKind::SecNameHash: return (spec.index < row.sec_name_hash.size()) ? row.sec_name_hash[spec.index] : 0.0;
        case CompiledFeatureKind::ImpDllHash: return (spec.index < row.imp_dll_hash.size()) ? row.imp_dll_hash[spec.index] : 0.0;
        case CompiledFeatureKind::ImpFuncHash: return (spec.index < row.imp_func_hash.size()) ? row.imp_func_hash[spec.index] : 0.0;
        case CompiledFeatureKind::DataDirectoryRva: return (spec.index < row.dd_rva.size()) ? row.dd_rva[spec.index] : 0.0;
        case CompiledFeatureKind::DataDirectorySize: return (spec.index < row.dd_size.size()) ? row.dd_size[spec.index] : 0.0;
        default: return 0.0;
    }
}

inline bool parse_index_suffix_internal(const std::string& name, const std::string& prefix, uint16_t max_allowed, uint16_t& out_index) {
    if (name.rfind(prefix, 0) != 0 || name.size() <= prefix.size()) return false;
    try { const unsigned long parsed = std::stoul(name.substr(prefix.size())); if (parsed > static_cast<unsigned long>(max_allowed)) return false; out_index = static_cast<uint16_t>(parsed); return true; } catch (...) { return false; }
}

inline bool try_resolve_feature_spec(const std::string& name, CompiledFeatureSpec& out_spec) {
    const auto& direct = get_direct_feature_map();
    if (direct.count(name)) { out_spec = {CompiledFeatureKind::Direct, direct.at(name), 0u}; return true; }
    if (name.rfind("coff_char_", 0) == 0) { const auto& m = get_coff_char_map(); std::string f = name.substr(10); if (m.count(f)) { out_spec = {CompiledFeatureKind::CoffChar, 0u, m.at(f)}; return true; } }
    if (name.rfind("opt_dllchar_", 0) == 0) { const auto& m = get_opt_dll_char_map(); std::string f = name.substr(12); if (m.count(f)) { out_spec = {CompiledFeatureKind::OptDllChar, 0u, m.at(f)}; return true; } }
    uint16_t idx = 0;
    if (parse_index_suffix_internal(name, "sec_name_hash_", 31, idx)) { out_spec = {CompiledFeatureKind::SecNameHash, idx, 0u}; return true; }
    if (parse_index_suffix_internal(name, "imp_dll_hash_", 63, idx)) { out_spec = {CompiledFeatureKind::ImpDllHash, idx, 0u}; return true; }
    if (parse_index_suffix_internal(name, "imp_func_hash_", 255, idx)) { out_spec = {CompiledFeatureKind::ImpFuncHash, idx, 0u}; return true; }
    if (name.rfind("sec", 0) == 0) {
        size_t u = name.find('_', 3);
        if (u != std::string::npos && u > 3) try {
            uint16_t sid = static_cast<uint16_t>(std::stoul(name.substr(3, u - 3)));
            if (sid <= 9) {
                std::string s = name.substr(u + 1);
                if (s == "entropy") { out_spec = {CompiledFeatureKind::SecEntropy, sid, 0u}; return true; }
                if (s == "vsize") { out_spec = {CompiledFeatureKind::SecVsize, sid, 0u}; return true; }
                if (s == "is_write") { out_spec = {CompiledFeatureKind::SecIsWrite, sid, 0u}; return true; }
            }
        } catch (...) {}
    }
    if (name.rfind("dd_", 0) == 0) {
        const auto& m = get_data_dir_map();
        if (name.size() > 7 && name.substr(name.size() - 4) == "_rva") { std::string d = name.substr(3, name.size() - 7); if (m.count(d)) { out_spec = {CompiledFeatureKind::DataDirectoryRva, static_cast<uint16_t>(m.at(d)), 0u}; return true; } }
        if (name.size() > 8 && name.substr(name.size() - 5) == "_size") { std::string d = name.substr(3, name.size() - 8); if (m.count(d)) { out_spec = {CompiledFeatureKind::DataDirectorySize, static_cast<uint16_t>(m.at(d)), 0u}; return true; } }
    }
    return false;
}

} // namespace detail

class If_feature_extractor {
public:
    If_feature_extractor() = default;

    bool init(const vector<std::string>& feature_list, uint16_t expected_feature_count = 0) {
        feature_config_path_.clear();
        return init_from_feature_names(feature_list, expected_feature_count);
    }

    bool init(const std::filesystem::path& feature_config_path, uint16_t expected_feature_count = 0) {
        feature_config_path_ = feature_config_path;
        std::string json;
        if (!read_text_file(feature_config_path_, json)) {
            set_status(eml_status_code::file_open_failed);
            return false;
        }

        vector<std::string> parsed_feature_names;
        if (!parse_feature_list_json(json, parsed_feature_names)) {
            set_status(eml_status_code::json_parse_failed);
            return false;
        }

        return init_from_feature_names(parsed_feature_names, expected_feature_count);
    }

    bool extract_from_pe(const std::filesystem::path& pe_path, vector<float>& out_features) const {
        if (!loaded_) { set_status(eml_status_code::not_loaded); return false; }
        
        std::error_code ec;
        if (!std::filesystem::exists(pe_path, ec) || ec) { set_status(eml_status_code::file_open_failed); return false; }
        uintmax_t size = std::filesystem::file_size(pe_path, ec);
        if (ec || size < EDR_PE_MIN_INPUT_FILE_BYTES) { set_status(eml_status_code::size_mismatch); return false; }
        if (size > EDR_PE_MAX_INPUT_FILE_BYTES) { set_status(eml_status_code::size_mismatch); return false; }

        try {
            std::unique_ptr<LIEF::PE::Binary> pe = LIEF::PE::Parser::parse(pe_path.string());
            if (!pe) { set_status(eml_status_code::callback_failed); return false; }
            
            detail::ExtractResult res = detail::extract_row_from_lief_binary(pe.get(), size);
            if (!res.parse_ok) { set_status(eml_status_code::callback_failed); return false; }

            out_features.assign(resolved_specs_.size(), 0.0f);
            for (size_t i = 0; i < resolved_specs_.size(); ++i) {
                out_features[i] = static_cast<float>(detail::resolve_spec_value(res, resolved_specs_[i]));
            }

            set_status(eml_status_code::ok);
            return true;
        } catch (...) {
            set_status(eml_status_code::callback_failed);
            return false;
        }
    }

    bool extract_from_pe_content(const uint8_t* content, size_t size, vector<float>& out_features) const {
        if (!loaded_) { set_status(eml_status_code::not_loaded); return false; }
        if (!content || size < EDR_PE_MIN_INPUT_FILE_BYTES) { set_status(eml_status_code::size_mismatch); return false; }
        if (size > EDR_PE_MAX_INPUT_FILE_BYTES) { set_status(eml_status_code::size_mismatch); return false; }

        try {
            std::unique_ptr<LIEF::PE::Binary> pe = LIEF::PE::Parser::parse(content, size);
            if (!pe) { set_status(eml_status_code::callback_failed); return false; }

            detail::ExtractResult res = detail::extract_row_from_lief_binary(pe.get(), size);
            if (!res.parse_ok) { set_status(eml_status_code::callback_failed); return false; }

            out_features.assign(resolved_specs_.size(), 0.0f);
            for (size_t i = 0; i < resolved_specs_.size(); ++i) {
                out_features[i] = static_cast<float>(detail::resolve_spec_value(res, resolved_specs_[i]));
            }

            set_status(eml_status_code::ok);
            return true;
        } catch (...) {
            set_status(eml_status_code::callback_failed);
            return false;
        }
    }

    void release() {
        feature_names_.clear();
        resolved_specs_.clear();
        loaded_ = false;
    }

    bool loaded() const { return loaded_; }
    eml_status_code last_status() const { return last_status_code_; }
    
    const vector<std::string>& feature_names() const { return feature_names_; }
    
    size_t memory_usage() const {
        size_t total = sizeof(*this);
        for (const auto& name : feature_names_) total += name.capacity();
        total += resolved_specs_.capacity() * sizeof(detail::CompiledFeatureSpec);
        return total;
    }

private:
    std::filesystem::path feature_config_path_;
    vector<std::string> feature_names_;
    vector<detail::CompiledFeatureSpec> resolved_specs_;
    bool loaded_ = false;
    mutable eml_status_code last_status_code_ = eml_status_code::ok;

    void set_status(eml_status_code status) const { last_status_code_ = status; }

    bool init_from_feature_names(const vector<std::string>& feature_names, uint16_t expected_feature_count) {
        feature_names_ = feature_names;
        resolved_specs_.clear();

        if (feature_names_.empty()) {
            set_status(eml_status_code::json_parse_failed);
            return false;
        }

        if (expected_feature_count > 0 && feature_names_.size() != expected_feature_count) {
             set_status(eml_status_code::invalid_argument);
             return false;
        }

        // Validate that all features are resolvable at init time
        resolved_specs_.reserve(feature_names_.size());
        for (const auto& name : feature_names_) {
            detail::CompiledFeatureSpec spec{};
            if (!detail::try_resolve_feature_spec(name, spec)) {
                set_status(eml_status_code::invalid_configuration);
                return false;
            }
            resolved_specs_.push_back(spec);
        }

        loaded_ = true;
        set_status(eml_status_code::ok);
        return true;
    }

    static bool read_text_file(const std::filesystem::path& path, std::string& out) {
        std::ifstream fin(path);
        if (!fin.is_open()) return false;
        out.assign((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
        return true;
    }

    static bool parse_feature_list_json(const std::string& json, vector<std::string>& out) {
        out.clear();
        const size_t open = json.find('[');
        if (open == std::string::npos) return false;
        size_t close = json.find(']', open);
        if (close == std::string::npos) return false;
        std::string_view payload(json.c_str() + open + 1, close - open - 1);
        size_t pos = 0;
        while (pos < payload.size()) {
            size_t q1 = payload.find('"', pos);
            if (q1 == std::string_view::npos) break;
            size_t q2 = payload.find('"', q1 + 1);
            if (q2 == std::string::npos) break;
            out.push_back(std::string(payload.substr(q1 + 1, q2 - q1 - 1)));
            pos = q2 + 1;
        }
        return !out.empty();
    }
};

} // namespace eml
