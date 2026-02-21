#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace embedded_feature_config {
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
  DIRECT_FEATURE_COUNT
};

static constexpr size_t kDataDirectorySlots = 16;
static constexpr size_t kDirectFeatureCount = static_cast<size_t>(DIRECT_FEATURE_COUNT);
static constexpr size_t kCompiledFeatureCount = 40;
static constexpr const char* kCompiledFeatureSource = "/home/viettran/Documents/visual_code/EDR_AGENT/development_phase/results/feature_names.json";

static constexpr std::array<const char*, kCompiledFeatureCount> kCompiledFeatureNames = {
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

static constexpr std::array<CompiledFeatureSpec, kCompiledFeatureCount> kCompiledFeatureSpecs = {
  CompiledFeatureSpec{CompiledFeatureKind::Direct, 0, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::Direct, 5, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::Direct, 13, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::SecNameHash, 23, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::SecIsWrite, 3, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::ImpDllHash, 61, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::Direct, 22, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::SecNameHash, 8, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::Direct, 11, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::DataDirectoryRva, 6, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::ImpFuncHash, 149, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::Direct, 6, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::SecEntropy, 2, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::Direct, 15, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::DataDirectoryRva, 4, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::Direct, 4, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::SecEntropy, 1, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::Direct, 18, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::Direct, 20, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::Direct, 19, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::OptDllChar, 0, 1024u},
  CompiledFeatureSpec{CompiledFeatureKind::Direct, 21, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::Direct, 14, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::OptDllChar, 0, 32u},
  CompiledFeatureSpec{CompiledFeatureKind::Direct, 17, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::Direct, 16, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::Direct, 7, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::SecVsize, 2, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::ImpDllHash, 43, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::Direct, 8, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::CoffChar, 0, 8192u},
  CompiledFeatureSpec{CompiledFeatureKind::DataDirectoryRva, 14, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::Direct, 12, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::OptDllChar, 0, 64u},
  CompiledFeatureSpec{CompiledFeatureKind::DataDirectorySize, 4, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::CoffChar, 0, 32u},
  CompiledFeatureSpec{CompiledFeatureKind::Direct, 23, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::Direct, 9, 0u},
  CompiledFeatureSpec{CompiledFeatureKind::OptDllChar, 0, 256u},
  CompiledFeatureSpec{CompiledFeatureKind::Direct, 10, 0u},
};
} // namespace embedded_feature_config
