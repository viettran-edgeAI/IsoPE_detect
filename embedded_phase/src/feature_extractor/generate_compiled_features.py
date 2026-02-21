#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

DIRECT_FEATURES = {
    "coff_machine": 0,
    "coff_num_sections": 1,
    "coff_timestamp": 2,
    "coff_sizeof_opt_header": 3,
    "coff_characteristics": 4,
    "opt_sizeof_init_data": 5,
    "opt_checksum": 6,
    "opt_section_alignment": 7,
    "opt_imagebase": 8,
    "opt_subsystem": 9,
    "checksum_matches": 10,
    "sec_mean_entropy": 11,
    "sec_max_entropy": 12,
    "has_resources": 13,
    "rsrc_has_version": 14,
    "has_relocations": 15,
    "has_debug": 16,
    "has_repro": 17,
    "rich_max_build_id": 18,
    "has_overlay": 19,
    "overlay_size": 20,
    "overlay_entropy": 21,
    "num_certificates": 22,
    "sig_verified": 23,
}

COFF_CHAR_FLAGS = {
    "EXECUTABLE_IMAGE": 0x0002,
    "DLL": 0x2000,
    "LARGE_ADDRESS_AWARE": 0x0020,
    "RELOCS_STRIPPED": 0x0001,
    "DEBUG_STRIPPED": 0x0200,
    "SYSTEM": 0x1000,
}

OPT_DLL_FLAGS = {
    "HIGH_ENTROPY_VA": 0x0020,
    "DYNAMIC_BASE": 0x0040,
    "FORCE_INTEGRITY": 0x0080,
    "NX_COMPAT": 0x0100,
    "NO_ISOLATION": 0x0200,
    "NO_SEH": 0x0400,
    "NO_BIND": 0x0800,
    "APPCONTAINER": 0x1000,
    "WDM_DRIVER": 0x2000,
    "GUARD_CF": 0x4000,
    "TERMINAL_SERVER_AWARE": 0x8000,
}

DD_TYPES = {
    "EXPORT_TABLE": 0,
    "IMPORT_TABLE": 1,
    "RESOURCE_TABLE": 2,
    "EXCEPTION_TABLE": 3,
    "CERTIFICATE_TABLE": 4,
    "BASE_RELOCATION_TABLE": 5,
    "DEBUG_DIR": 6,
    "ARCHITECTURE": 7,
    "GLOBAL_PTR": 8,
    "TLS_TABLE": 9,
    "LOAD_CONFIG_TABLE": 10,
    "BOUND_IMPORT": 11,
    "IAT": 12,
    "DELAY_IMPORT_DESCRIPTOR": 13,
    "CLR_RUNTIME_HEADER": 14,
    "RESERVED": 15,
}

KIND_DIRECT = "Direct"
KIND_COFF_CHAR = "CoffChar"
KIND_OPT_DLL = "OptDllChar"
KIND_SEC_ENTROPY = "SecEntropy"
KIND_SEC_VSIZE = "SecVsize"
KIND_SEC_IS_WRITE = "SecIsWrite"
KIND_SEC_NAME_HASH = "SecNameHash"
KIND_IMP_DLL_HASH = "ImpDllHash"
KIND_IMP_FUNC_HASH = "ImpFuncHash"
KIND_DD_RVA = "DataDirectoryRva"
KIND_DD_SIZE = "DataDirectorySize"


def parse_feature(feature: str):
    if feature in DIRECT_FEATURES:
        return (KIND_DIRECT, DIRECT_FEATURES[feature], 0)

    if feature.startswith("coff_char_"):
        key = feature[len("coff_char_"):]
        if key in COFF_CHAR_FLAGS:
            return (KIND_COFF_CHAR, 0, COFF_CHAR_FLAGS[key])

    if feature.startswith("opt_dllchar_"):
        key = feature[len("opt_dllchar_"):]
        if key in OPT_DLL_FLAGS:
            return (KIND_OPT_DLL, 0, OPT_DLL_FLAGS[key])

    m = re.match(r"^sec_name_hash_(\d+)$", feature)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < 32:
            return (KIND_SEC_NAME_HASH, idx, 0)

    m = re.match(r"^imp_dll_hash_(\d+)$", feature)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < 64:
            return (KIND_IMP_DLL_HASH, idx, 0)

    m = re.match(r"^imp_func_hash_(\d+)$", feature)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < 256:
            return (KIND_IMP_FUNC_HASH, idx, 0)

    m = re.match(r"^sec(\d+)_(entropy|vsize|is_write)$", feature)
    if m:
        sec_idx = int(m.group(1))
        metric = m.group(2)
        if 0 <= sec_idx < 10:
            if metric == "entropy":
                return (KIND_SEC_ENTROPY, sec_idx, 0)
            if metric == "vsize":
                return (KIND_SEC_VSIZE, sec_idx, 0)
            if metric == "is_write":
                return (KIND_SEC_IS_WRITE, sec_idx, 0)

    m = re.match(r"^dd_([A-Z0-9_]+)_(rva|size)$", feature)
    if m:
        dd_name = m.group(1)
        suffix = m.group(2)
        if dd_name in DD_TYPES:
            if suffix == "rva":
                return (KIND_DD_RVA, DD_TYPES[dd_name], 0)
            return (KIND_DD_SIZE, DD_TYPES[dd_name], 0)

    return None


def cxx_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def main():
    parser = argparse.ArgumentParser(description="Generate compile-time feature config header")
    parser.add_argument("--feature-names", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    raw = json.loads(args.feature_names.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("feature_names.json must be a non-empty JSON list")

    deduped = []
    seen = set()
    for item in raw:
        if not isinstance(item, str):
            raise RuntimeError(f"feature name must be string, got: {item!r}")
        if item not in seen:
            deduped.append(item)
            seen.add(item)

    specs = []
    errors = []
    for feature in deduped:
        parsed = parse_feature(feature)
        if parsed is None:
            errors.append(feature)
        else:
            specs.append((feature, *parsed))

    if errors:
        msg = "Unsupported feature(s) for compile-time extractor:\n" + "\n".join(f"  - {x}" for x in errors)
        raise RuntimeError(msg)

    lines = []
    lines.append("#pragma once")
    lines.append("#include <array>")
    lines.append("#include <cstddef>")
    lines.append("#include <cstdint>")
    lines.append("")
    lines.append("namespace embedded_feature_config {")
    lines.append("enum class CompiledFeatureKind : uint8_t {")
    lines.append("  Direct = 0,")
    lines.append("  CoffChar = 1,")
    lines.append("  OptDllChar = 2,")
    lines.append("  SecEntropy = 3,")
    lines.append("  SecVsize = 4,")
    lines.append("  SecIsWrite = 5,")
    lines.append("  SecNameHash = 6,")
    lines.append("  ImpDllHash = 7,")
    lines.append("  ImpFuncHash = 8,")
    lines.append("  DataDirectoryRva = 9,")
    lines.append("  DataDirectorySize = 10,")
    lines.append("};")
    lines.append("")
    lines.append("struct CompiledFeatureSpec {")
    lines.append("  CompiledFeatureKind kind;")
    lines.append("  uint16_t index;")
    lines.append("  uint32_t value;")
    lines.append("};")
    lines.append("")
    lines.append("enum DirectFeatureId : uint16_t {")
    for name, idx in sorted(DIRECT_FEATURES.items(), key=lambda kv: kv[1]):
        lines.append(f"  D_{name.upper()} = {idx},")
    lines.append("  DIRECT_FEATURE_COUNT")
    lines.append("};")
    lines.append("")
    lines.append("static constexpr size_t kDataDirectorySlots = 16;")
    lines.append("static constexpr size_t kDirectFeatureCount = static_cast<size_t>(DIRECT_FEATURE_COUNT);")
    lines.append(f"static constexpr size_t kCompiledFeatureCount = {len(specs)};")
    lines.append(f"static constexpr const char* kCompiledFeatureSource = \"{cxx_escape(str(args.feature_names))}\";")
    lines.append("")

    lines.append("static constexpr std::array<const char*, kCompiledFeatureCount> kCompiledFeatureNames = {")
    for feature, *_ in specs:
        lines.append(f"  \"{cxx_escape(feature)}\",")
    lines.append("};")
    lines.append("")

    lines.append("static constexpr std::array<CompiledFeatureSpec, kCompiledFeatureCount> kCompiledFeatureSpecs = {")
    for _feature, kind, index, value in specs:
        lines.append(f"  CompiledFeatureSpec{{CompiledFeatureKind::{kind}, {index}, {value}u}},")
    lines.append("};")
    lines.append("} // namespace embedded_feature_config")

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[generate_compiled_features] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
