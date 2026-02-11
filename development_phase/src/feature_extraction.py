#!/usr/bin/env python3
"""
Stage 1+2: Data Preprocessing and Feature Selection
====================================================
Extract raw PE features with LIEF, build consistent raw datasets,
then apply feature filtering to create cleaned datasets.

Usage:
    python feature_extraction.py [--config PATH] [--workers N] [--corr-threshold X]

Outputs:
    ../data/raw/benign_train_raw.parquet
    ../data/raw/benign_val_raw.parquet
    ../data/raw/benign_test_raw.parquet
    ../data/raw/malware_val_raw.parquet
    ../data/raw/malware_test_raw.parquet
    ../data/cleaned/benign_train_clean.parquet
    ../data/cleaned/benign_val_clean.parquet
    ../data/cleaned/benign_test_clean.parquet
    ../data/cleaned/malware_val_clean.parquet
    ../data/cleaned/malware_test_clean.parquet
    ../schemas/feature_group_mapping.json
    ../schemas/feature_schema.json
    ../schemas/feature_schema_selected.json
    ../reports/feature_selection_report.md
"""

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lief
import numpy as np
import pandas as pd
from scipy.stats import kurtosis as sp_kurtosis

# ── Constants ──
SEED = 42
PARSE_TIMEOUT = 60
MAX_SECTIONS = 10

HASH_SIZES = {
    "imp_func": 256,
    "imp_dll": 64,
    "exp_name": 32,
    "sec_name": 32,
}

SUSPICIOUS_APIS = frozenset({
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
})


# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def calculate_entropy(data: bytes) -> float:
    """Shannon entropy of byte data (0.0–8.0)."""
    if not data or len(data) == 0:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    length = len(data)
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / length
            ent -= p * math.log2(p)
    return ent


def feature_hash(strings: list, n_features: int = 256) -> list:
    """Hashing trick: fixed-size vector from variable-length string list.
    Uses first 8 bytes of MD5 for bucket index, first 8 bytes of SHA1 for sign.
    Byte-identical reproducible between Python and C++.
    """
    vector = [0.0] * n_features
    for s in strings:
        if not s:
            continue
        encoded = s.encode("utf-8", errors="replace")
        h1 = int.from_bytes(hashlib.md5(encoded).digest()[:8], "little")
        h2 = int.from_bytes(hashlib.sha1(encoded).digest()[:8], "little")
        idx = h1 % n_features
        sign = 1.0 if (h2 % 2 == 0) else -1.0
        vector[idx] += sign
    return vector


def safe_div(a, b):
    return a / b if b != 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION ENGINE
# ═══════════════════════════════════════════════════════════════

def extract_features(filepath: str) -> dict | None:
    """Extract all PE features from a single file. Returns dict or None."""
    try:
        pe = lief.PE.parse(filepath)
        if pe is None:
            return None
    except Exception:
        return None

    f = {}

    # ── FILE METADATA ──
    try:
        f["file_size"] = os.path.getsize(filepath)
    except Exception:
        f["file_size"] = 0

    # ── DOS HEADER ──
    try:
        dos = pe.dos_header
        f["dos_magic"] = dos.magic
        f["dos_e_lfanew"] = dos.addressof_new_exeheader
        f["dos_checksum"] = dos.checksum
        f["dos_num_relocs"] = dos.numberof_relocation
    except Exception:
        f.update(dos_magic=0, dos_e_lfanew=0, dos_checksum=0, dos_num_relocs=0)

    # ── COFF HEADER ──
    try:
        hdr = pe.header
        f["coff_machine"] = hdr.machine.value
        f["coff_num_sections"] = hdr.numberof_sections
        f["coff_timestamp"] = hdr.time_date_stamps
        f["coff_sizeof_opt_header"] = hdr.sizeof_optional_header
        f["coff_characteristics"] = hdr.characteristics
        for name, flag in [
            ("EXECUTABLE_IMAGE", lief.PE.Header.CHARACTERISTICS.EXECUTABLE_IMAGE),
            ("DLL", lief.PE.Header.CHARACTERISTICS.DLL),
            ("LARGE_ADDRESS_AWARE", lief.PE.Header.CHARACTERISTICS.LARGE_ADDRESS_AWARE),
            ("RELOCS_STRIPPED", lief.PE.Header.CHARACTERISTICS.RELOCS_STRIPPED),
            ("DEBUG_STRIPPED", lief.PE.Header.CHARACTERISTICS.DEBUG_STRIPPED),
            ("SYSTEM", lief.PE.Header.CHARACTERISTICS.SYSTEM),
        ]:
            f[f"coff_char_{name}"] = int(hdr.has_characteristic(flag))
    except Exception:
        f.update(
            coff_machine=0, coff_num_sections=0, coff_timestamp=0,
            coff_sizeof_opt_header=0, coff_characteristics=0,
        )

    # ── OPTIONAL HEADER ──
    try:
        opt = pe.optional_header
        f["opt_magic"] = opt.magic.value
        f["opt_major_linker"] = opt.major_linker_version
        f["opt_minor_linker"] = opt.minor_linker_version
        f["opt_sizeof_code"] = opt.sizeof_code
        f["opt_sizeof_init_data"] = opt.sizeof_initialized_data
        f["opt_sizeof_uninit_data"] = opt.sizeof_uninitialized_data
        f["opt_entrypoint"] = opt.addressof_entrypoint
        f["opt_baseof_code"] = opt.baseof_code
        f["opt_imagebase"] = opt.imagebase
        f["opt_section_alignment"] = opt.section_alignment
        f["opt_file_alignment"] = opt.file_alignment
        f["opt_major_os_ver"] = opt.major_operating_system_version
        f["opt_minor_os_ver"] = opt.minor_operating_system_version
        f["opt_major_image_ver"] = opt.major_image_version
        f["opt_minor_image_ver"] = opt.minor_image_version
        f["opt_major_subsys_ver"] = opt.major_subsystem_version
        f["opt_minor_subsys_ver"] = opt.minor_subsystem_version
        f["opt_sizeof_image"] = opt.sizeof_image
        f["opt_sizeof_headers"] = opt.sizeof_headers
        f["opt_checksum"] = opt.checksum
        f["opt_subsystem"] = opt.subsystem.value
        f["opt_dll_characteristics"] = opt.dll_characteristics
        f["opt_sizeof_stack_reserve"] = opt.sizeof_stack_reserve
        f["opt_sizeof_stack_commit"] = opt.sizeof_stack_commit
        f["opt_sizeof_heap_reserve"] = opt.sizeof_heap_reserve
        f["opt_sizeof_heap_commit"] = opt.sizeof_heap_commit
        f["opt_num_rva_and_sizes"] = opt.numberof_rva_and_size
        for name, flag in [
            ("DYNAMIC_BASE", lief.PE.OptionalHeader.DLL_CHARACTERISTICS.DYNAMIC_BASE),
            ("NX_COMPAT", lief.PE.OptionalHeader.DLL_CHARACTERISTICS.NX_COMPAT),
            ("GUARD_CF", lief.PE.OptionalHeader.DLL_CHARACTERISTICS.GUARD_CF),
            ("HIGH_ENTROPY_VA", lief.PE.OptionalHeader.DLL_CHARACTERISTICS.HIGH_ENTROPY_VA),
            ("NO_SEH", lief.PE.OptionalHeader.DLL_CHARACTERISTICS.NO_SEH),
            ("NO_BIND", lief.PE.OptionalHeader.DLL_CHARACTERISTICS.NO_BIND),
            ("APPCONTAINER", lief.PE.OptionalHeader.DLL_CHARACTERISTICS.APPCONTAINER),
            ("TERMINAL_SERVER_AWARE", lief.PE.OptionalHeader.DLL_CHARACTERISTICS.TERMINAL_SERVER_AWARE),
            ("FORCE_INTEGRITY", lief.PE.OptionalHeader.DLL_CHARACTERISTICS.FORCE_INTEGRITY),
            ("NO_ISOLATION", lief.PE.OptionalHeader.DLL_CHARACTERISTICS.NO_ISOLATION),
            ("WDM_DRIVER", lief.PE.OptionalHeader.DLL_CHARACTERISTICS.WDM_DRIVER),
        ]:
            f[f"opt_dllchar_{name}"] = int(opt.has(flag))
        try:
            f["checksum_matches"] = int(opt.checksum == pe.compute_checksum())
        except Exception:
            f["checksum_matches"] = 0
    except Exception:
        pass

    # ── SECTIONS ──
    try:
        sections = list(pe.sections)
        f["num_sections"] = len(sections)
        entropies, raw_sizes, virt_sizes, sec_names = [], [], [], []

        for i, sec in enumerate(sections[:MAX_SECTIONS]):
            name = sec.name if isinstance(sec.name, str) else ""
            sec_names.append(name)
            ent = sec.entropy
            entropies.append(ent)
            vs, rs = sec.virtual_size, sec.sizeof_raw_data
            raw_sizes.append(rs)
            virt_sizes.append(vs)
            p = f"sec{i}"
            f[f"{p}_entropy"] = ent
            f[f"{p}_vsize"] = vs
            f[f"{p}_rsize"] = rs
            f[f"{p}_vr_ratio"] = safe_div(vs, rs)
            f[f"{p}_characteristics"] = sec.characteristics
            f[f"{p}_is_exec"] = int(sec.has_characteristic(lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE))
            f[f"{p}_is_write"] = int(sec.has_characteristic(lief.PE.Section.CHARACTERISTICS.MEM_WRITE))
            f[f"{p}_has_code"] = int(sec.has_characteristic(lief.PE.Section.CHARACTERISTICS.CNT_CODE))

        for i in range(len(sections), MAX_SECTIONS):
            p = f"sec{i}"
            for suf in ["entropy", "vsize", "rsize", "vr_ratio", "characteristics", "is_exec", "is_write", "has_code"]:
                f[f"{p}_{suf}"] = 0

        f["sec_mean_entropy"] = float(np.mean(entropies)) if entropies else 0
        f["sec_max_entropy"] = max(entropies) if entropies else 0
        f["sec_min_entropy"] = min(entropies) if entropies else 0
        f["sec_std_entropy"] = float(np.std(entropies)) if entropies else 0
        f["sec_total_raw_size"] = sum(raw_sizes)
        f["sec_total_virtual_size"] = sum(virt_sizes)
        f["sec_mean_vr_ratio"] = float(np.mean([safe_div(v, r) for v, r in zip(virt_sizes, raw_sizes)])) if entropies else 0

        for i, val in enumerate(feature_hash(sec_names, HASH_SIZES["sec_name"])):
            f[f"sec_name_hash_{i}"] = val

        f["num_exec_sections"] = sum(1 for s in sections if s.has_characteristic(lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE))
        f["num_write_sections"] = sum(1 for s in sections if s.has_characteristic(lief.PE.Section.CHARACTERISTICS.MEM_WRITE))
        f["num_rwx_sections"] = sum(1 for s in sections if s.has_characteristic(lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE) and s.has_characteristic(lief.PE.Section.CHARACTERISTICS.MEM_WRITE))
    except Exception:
        f["num_sections"] = 0

    # ── IMPORTS ──
    try:
        f["has_imports"] = int(pe.has_imports)
        if pe.has_imports:
            dll_names, func_names, total = [], [], 0
            for imp in pe.imports:
                dname = imp.name.lower() if isinstance(imp.name, str) else ""
                dll_names.append(dname)
                for entry in imp.entries:
                    total += 1
                    if not entry.is_ordinal and isinstance(entry.name, str):
                        func_names.append(entry.name.lower())
            f["num_import_dlls"] = len(dll_names)
            f["num_import_functions"] = total
            f["num_import_by_name"] = len(func_names)
            f["num_import_by_ordinal"] = total - len(func_names)
            for i, v in enumerate(feature_hash(dll_names, HASH_SIZES["imp_dll"])):
                f[f"imp_dll_hash_{i}"] = v
            for i, v in enumerate(feature_hash(func_names, HASH_SIZES["imp_func"])):
                f[f"imp_func_hash_{i}"] = v
            try:
                f["imphash"] = int(lief.PE.get_imphash(pe, lief.PE.IMPHASH_MODE.PEFILE)[:8], 16)
            except Exception:
                f["imphash"] = 0
            f["num_suspicious_imports"] = sum(1 for fn in func_names if fn in SUSPICIOUS_APIS)
        else:
            f["num_import_dlls"] = f["num_import_functions"] = 0
            f["num_import_by_name"] = f["num_import_by_ordinal"] = 0
            for i in range(HASH_SIZES["imp_dll"]):
                f[f"imp_dll_hash_{i}"] = 0
            for i in range(HASH_SIZES["imp_func"]):
                f[f"imp_func_hash_{i}"] = 0
            f["imphash"] = f["num_suspicious_imports"] = 0
    except Exception:
        f["has_imports"] = 0

    # ── EXPORTS ──
    try:
        f["has_exports"] = int(pe.has_exports)
        if pe.has_exports:
            exp = pe.get_export()
            entries = list(exp.entries)
            f["num_exports"] = len(entries)
            f["export_name_count"] = exp.names_addr_table_cnt
            f["num_forwarded_exports"] = sum(1 for e in entries if e.is_forwarded)
            exp_names = [e.name.lower() if isinstance(e.name, str) else "" for e in entries]
            for i, v in enumerate(feature_hash(exp_names, HASH_SIZES["exp_name"])):
                f[f"exp_hash_{i}"] = v
        else:
            f["num_exports"] = f["export_name_count"] = f["num_forwarded_exports"] = 0
            for i in range(HASH_SIZES["exp_name"]):
                f[f"exp_hash_{i}"] = 0
    except Exception:
        f["has_exports"] = 0

    # ── RESOURCES ──
    try:
        f["has_resources"] = int(pe.has_resources)
        if pe.has_resources:
            def _count_rsrc(node):
                d, da, sz = 0, 0, 0
                if node.is_directory:
                    d += 1
                elif node.is_data:
                    da += 1
                    sz += len(node.content)
                for ch in node.childs:
                    a, b, c = _count_rsrc(ch)
                    d += a
                    da += b
                    sz += c
                return d, da, sz

            nd, nda, tsz = _count_rsrc(pe.resources)
            f["rsrc_num_directories"] = nd
            f["rsrc_num_data_entries"] = nda
            f["rsrc_total_size"] = tsz
            f["rsrc_size_ratio"] = safe_div(tsz, f.get("file_size", 1))
            rm = pe.resources_manager
            if not isinstance(rm, lief.lief_errors):
                f["rsrc_num_types"] = len(rm.types)
                f["rsrc_has_manifest"] = int(rm.has_manifest)
                f["rsrc_has_version"] = int(rm.has_version)
                f["rsrc_has_icons"] = int(rm.has_icons)
                f["rsrc_has_dialogs"] = int(rm.has_dialogs)
                f["rsrc_has_string_table"] = int(rm.has_string_table)
            else:
                for k in ["rsrc_num_types", "rsrc_has_manifest", "rsrc_has_version",
                           "rsrc_has_icons", "rsrc_has_dialogs", "rsrc_has_string_table"]:
                    f[k] = 0

            def _rsrc_ent(node, mx=20):
                es = []
                if node.is_data and len(node.content) > 0:
                    es.append(calculate_entropy(bytes(node.content[:4096])))
                if len(es) < mx:
                    for ch in node.childs:
                        es.extend(_rsrc_ent(ch, mx - len(es)))
                        if len(es) >= mx:
                            break
                return es

            re = _rsrc_ent(pe.resources)
            f["rsrc_mean_entropy"] = float(np.mean(re)) if re else 0
            f["rsrc_max_entropy"] = max(re) if re else 0
        else:
            for k in ["rsrc_num_directories", "rsrc_num_data_entries", "rsrc_total_size",
                       "rsrc_size_ratio", "rsrc_num_types", "rsrc_has_manifest", "rsrc_has_version",
                       "rsrc_has_icons", "rsrc_has_dialogs", "rsrc_has_string_table",
                       "rsrc_mean_entropy", "rsrc_max_entropy"]:
                f[k] = 0
    except Exception:
        f["has_resources"] = 0

    # ── TLS ──
    try:
        f["has_tls"] = int(pe.has_tls)
        if pe.has_tls:
            tls = pe.tls
            f["tls_num_callbacks"] = len(tls.callbacks)
            f["tls_sizeof_zero_fill"] = tls.sizeof_zero_fill
            f["tls_characteristics"] = tls.characteristics
            f["tls_data_size"] = len(tls.data_template)
        else:
            f["tls_num_callbacks"] = f["tls_sizeof_zero_fill"] = f["tls_characteristics"] = f["tls_data_size"] = 0
    except Exception:
        f["has_tls"] = 0

    # ── DEBUG ──
    try:
        f["has_debug"] = int(pe.has_debug)
        if pe.has_debug:
            dbgs = list(pe.debug)
            f["num_debug_entries"] = len(dbgs)
            f["has_codeview"] = int(any(d.type == lief.PE.Debug.TYPES.CODEVIEW for d in dbgs))
            f["has_pogo"] = int(any(d.type == lief.PE.Debug.TYPES.POGO for d in dbgs))
            f["has_repro"] = int(any(d.type == lief.PE.Debug.TYPES.REPRO for d in dbgs))
            f["is_reproducible"] = int(pe.is_reproducible_build)
            f["has_pdb"] = int(pe.codeview_pdb is not None)
        else:
            for k in ["num_debug_entries", "has_codeview", "has_pogo", "has_repro", "is_reproducible", "has_pdb"]:
                f[k] = 0
    except Exception:
        f["has_debug"] = 0

    # ── SIGNATURES ──
    try:
        f["has_signatures"] = int(pe.has_signatures)
        if pe.has_signatures:
            sigs = list(pe.signatures)
            f["num_signatures"] = len(sigs)
            f["num_certificates"] = sum(len(list(s.certificates)) for s in sigs)
            try:
                f["sig_verified"] = int(pe.verify_signature() == lief.PE.Signature.VERIFICATION_FLAGS.OK)
            except Exception:
                f["sig_verified"] = 0
        else:
            f["num_signatures"] = f["num_certificates"] = f["sig_verified"] = 0
    except Exception:
        f["has_signatures"] = 0

    # ── RELOCATIONS ──
    try:
        f["has_relocations"] = int(pe.has_relocations)
        if pe.has_relocations:
            rels = list(pe.relocations)
            f["num_relocation_blocks"] = len(rels)
            f["num_relocation_entries"] = sum(len(list(r.entries)) for r in rels)
        else:
            f["num_relocation_blocks"] = f["num_relocation_entries"] = 0
    except Exception:
        f["has_relocations"] = 0

    # ── RICH HEADER ──
    try:
        f["has_rich_header"] = int(pe.has_rich_header)
        if pe.has_rich_header:
            rich = pe.rich_header
            ents = list(rich.entries)
            f["rich_key"] = rich.key
            f["rich_num_entries"] = len(ents)
            if ents:
                ids = [e.id for e in ents]
                bids = [e.build_id for e in ents]
                cnts = [e.count for e in ents]
                f["rich_mean_id"] = float(np.mean(ids))
                f["rich_max_build_id"] = max(bids)
                f["rich_total_count"] = sum(cnts)
            else:
                f["rich_mean_id"] = f["rich_max_build_id"] = f["rich_total_count"] = 0
        else:
            for k in ["rich_key", "rich_num_entries", "rich_mean_id", "rich_max_build_id", "rich_total_count"]:
                f[k] = 0
    except Exception:
        f["has_rich_header"] = 0

    # ── LOAD CONFIGURATION ──
    try:
        f["has_load_config"] = int(pe.has_configuration)
        if pe.has_configuration:
            lc = pe.load_configuration
            f["lc_size"] = lc.size
            f["lc_security_cookie"] = lc.security_cookie
            f["lc_seh_count"] = lc.se_handler_count or 0
            f["lc_guard_flags"] = lc.guard_flags or 0
            f["lc_cf_function_count"] = lc.guard_cf_function_count or 0
        else:
            for k in ["lc_size", "lc_security_cookie", "lc_seh_count", "lc_guard_flags", "lc_cf_function_count"]:
                f[k] = 0
    except Exception:
        f["has_load_config"] = 0

    # ── OVERLAY ──
    try:
        ovl = pe.overlay
        f["has_overlay"] = int(len(ovl) > 0)
        f["overlay_size"] = len(ovl)
        f["overlay_ratio"] = safe_div(len(ovl), f.get("file_size", 1))
        f["overlay_entropy"] = calculate_entropy(bytes(ovl[:8192])) if len(ovl) > 0 else 0
    except Exception:
        f["has_overlay"] = f["overlay_size"] = 0
        f["overlay_ratio"] = f["overlay_entropy"] = 0.0

    # ── DATA DIRECTORIES ──
    try:
        for dd in pe.data_directories:
            nm = dd.type.name if hasattr(dd.type, "name") else str(dd.type)
            f[f"dd_{nm}_rva"] = dd.rva
            f[f"dd_{nm}_size"] = dd.size
    except Exception:
        pass

    # ── DELAY IMPORTS ──
    try:
        f["has_delay_imports"] = int(pe.has_delay_imports)
        f["num_delay_imports"] = len(list(pe.delay_imports)) if pe.has_delay_imports else 0
    except Exception:
        f["has_delay_imports"] = f["num_delay_imports"] = 0

    return f


def extract_features_worker(filepath: str):
    """Wrapper for multiprocessing."""
    try:
        result = extract_features(filepath)
        if result is not None:
            result["_filepath"] = os.path.basename(filepath)
        return result
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# GROUP MAPPING
# ═══════════════════════════════════════════════════════════════

def build_group_mapping(columns: list) -> dict:
    """Map each feature column to its logical group."""
    mapping = {}
    for col in columns:
        if col.startswith("_"):
            continue
        if col.startswith("dos_"):
            grp = "dos_header"
        elif col.startswith("coff_"):
            grp = "coff_header"
        elif col.startswith("opt_") or col == "checksum_matches":
            grp = "optional_header"
        elif col.startswith("sec") and "_" in col and col[3:4].isdigit():
            grp = "sections_per_slot"
        elif col.startswith("sec_name_hash"):
            grp = "sections_name_hash"
        elif col.startswith("sec_"):
            grp = "sections_aggregate"
        elif col.startswith("num_") and "section" in col:
            grp = "sections_counts"
        elif col.startswith("imp_dll_hash"):
            grp = "imports_dll_hash"
        elif col.startswith("imp_func_hash"):
            grp = "imports_func_hash"
        elif col.startswith("imp_") or col.startswith("has_import") or col.startswith("num_import") or col == "imphash" or col == "num_suspicious_imports":
            grp = "imports_scalar"
        elif col.startswith("exp_hash"):
            grp = "exports_hash"
        elif col.startswith("exp_") or col.startswith("has_export") or col.startswith("num_export") or col == "export_name_count" or col == "num_forwarded_exports":
            grp = "exports_scalar"
        elif col.startswith("rsrc_") or col == "has_resources":
            grp = "resources"
        elif col.startswith("tls_") or col == "has_tls":
            grp = "tls"
        elif col in ["has_debug", "num_debug_entries", "has_codeview", "has_pogo", "has_repro", "is_reproducible", "has_pdb"]:
            grp = "debug"
        elif col.startswith("sig_") or col == "has_signatures" or col.startswith("num_sig") or col.startswith("num_cert"):
            grp = "signatures"
        elif col.startswith("reloc") or col == "has_relocations" or col.startswith("num_reloc"):
            grp = "relocations"
        elif col.startswith("rich_") or col == "has_rich_header":
            grp = "rich_header"
        elif col.startswith("lc_") or col == "has_load_config":
            grp = "load_config"
        elif col.startswith("overlay") or col == "has_overlay":
            grp = "overlay"
        elif col.startswith("dd_"):
            grp = "data_directories"
        elif col.startswith("dimp") or col == "has_delay_imports" or col == "num_delay_imports":
            grp = "delay_imports"
        elif col == "file_size":
            grp = "file_metadata"
        elif col == "num_sections":
            grp = "sections_counts"
        else:
            grp = "other"
        mapping[col] = grp
    return mapping


# ═══════════════════════════════════════════════════════════════
# BATCH EXTRACTION
# ═══════════════════════════════════════════════════════════════

def scan_pe_files(directory: Path) -> list:
    """Find all PE files in a directory (any extension)."""
    files = sorted(directory.glob("*"))
    return [str(f) for f in files if f.is_file()]


def batch_extract(file_list: list, label: str, max_workers: int) -> tuple:
    """Extract features from a list of PE files with multiprocessing."""
    results, failed = [], []
    t0 = time.time()
    print(f"\n{'=' * 60}")
    print(f"Extracting [{label}]: {len(file_list)} files, {max_workers} workers")
    print(f"{'=' * 60}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_features_worker, fp): fp for fp in file_list}
        for i, future in enumerate(as_completed(futures)):
            fp = futures[future]
            try:
                res = future.result(timeout=PARSE_TIMEOUT)
                if res is not None:
                    results.append(res)
                else:
                    failed.append(fp)
            except Exception:
                failed.append(fp)
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                print(
                    f"  [{label}] {i + 1}/{len(file_list)} "
                    f"({len(results)} ok, {len(failed)} fail) "
                    f"[{elapsed:.1f}s]"
                )

    elapsed = time.time() - t0
    print(f"  Done: {len(results)} success, {len(failed)} failed in {elapsed:.1f}s")
    return results, failed


def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    if "_filepath" in df.columns:
        df = df.drop(columns=["_filepath"])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.float32)
    return df


def _build_canonical_feature_order(dfs_raw: dict) -> list:
    base_df = dfs_raw.get("benign_train")
    if base_df is None:
        base_df = next(iter(dfs_raw.values()))
    base_cols = list(base_df.columns)
    all_cols = set()
    for df in dfs_raw.values():
        all_cols.update(df.columns)
    extra_cols = sorted(all_cols - set(base_cols))
    return base_cols + extra_cols


def _align_dataframe(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    aligned = df.reindex(columns=feature_cols, fill_value=0.0)
    return aligned.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════

def variance_filter(df: pd.DataFrame, group_mapping: dict,
                    norm_var_threshold: float = 1e-7) -> tuple:
    """Remove features with near-zero variance.

    Rules:
    - Binary flags: minority class ratio < 0.5%
    - Hash vector buckets: nonzero fraction < 1%
    - Continuous: normalized variance < norm_var_threshold

    Returns (kept_cols, removed_log).
    """
    removed = []
    kept = []

    for col in df.columns:
        vals = df[col]
        unique = vals.nunique()
        is_binary = set(vals.unique()).issubset({0, 1, 0.0, 1.0})
        is_hash = "hash_" in col

        if is_binary:
            minority = min(vals.mean(), 1 - vals.mean())
            if minority < 0.005:
                removed.append({
                    "name": col,
                    "group": group_mapping.get(col, "?"),
                    "reason": "near_constant_binary",
                    "value": round(minority, 6),
                })
                continue
        elif is_hash:
            nonzero_ratio = (vals != 0).mean()
            if nonzero_ratio < 0.01:
                removed.append({
                    "name": col,
                    "group": group_mapping.get(col, "?"),
                    "reason": "inactive_hash_bucket",
                    "value": round(nonzero_ratio, 6),
                })
                continue
        else:
            v = vals.var()
            rng = vals.max() - vals.min()
            norm_var = v / (rng**2 + 1e-15) if rng > 0 else 0
            if norm_var < norm_var_threshold or unique <= 1:
                removed.append({
                    "name": col,
                    "group": group_mapping.get(col, "?"),
                    "reason": "near_zero_variance",
                    "value": round(norm_var, 10),
                })
                continue

        kept.append(col)

    return kept, removed


def correlation_pruning(df: pd.DataFrame, threshold: float = 0.95,
                        group_mapping: dict | None = None) -> tuple:
    """Remove one feature from each correlated pair (|r| > threshold).

    Strategy: iteratively remove the feature with the highest degree
    in the correlation graph (most correlated partners).

    Returns (kept_cols, removed_log).
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    pairs = []
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            num = pd.to_numeric(val, errors="coerce")
            if not pd.isna(num) and num > threshold:
                pairs.append((idx, col, float(num)))

    pairs.sort(key=lambda x: -x[2])

    to_remove = set()
    removed_log = []
    for f1, f2, corr_val in pairs:
        if f1 in to_remove or f2 in to_remove:
            continue
        deg1 = sum(1 for a, b, _ in pairs if (b == f1 or a == f1) and b not in to_remove and a not in to_remove)
        deg2 = sum(1 for a, b, _ in pairs if (b == f2 or a == f2) and b not in to_remove and a not in to_remove)
        var1, var2 = df[f1].var(), df[f2].var()

        if deg1 > deg2 or (deg1 == deg2 and var1 < var2):
            drop, keep = f1, f2
        else:
            drop, keep = f2, f1
        to_remove.add(drop)
        removed_log.append({
            "dropped": drop,
            "kept": keep,
            "correlation": round(corr_val, 4),
            "dropped_group": group_mapping.get(drop, "?") if group_mapping else "?",
        })

    kept = [c for c in df.columns if c not in to_remove]
    return kept, removed_log


def stability_filter(df: pd.DataFrame, group_mapping: dict) -> tuple:
    """Remove unstable features.

    Checks: CV > 50, outlier fraction > 5%, kurtosis > 100, fill-rate < 10%.
    Log-transform rescue: keep if log1p(|x|) stabilizes the feature.

    Returns (kept_cols, log_transform_cols, removed_log).
    """
    removed = []
    kept = []
    log_transform = []

    for col in df.columns:
        vals = df[col]
        is_binary = set(vals.unique()).issubset({0, 1, 0.0, 1.0})

        if is_binary:
            fill = vals.mean()
            if fill < 0.001 and (1 - fill) < 0.001:
                removed.append({
                    "name": col,
                    "group": group_mapping.get(col, "?"),
                    "reason": "binary_empty",
                    "value": 0,
                })
                continue
            kept.append(col)
            continue

        mu = vals.mean()
        sigma = vals.std()
        cv = sigma / (abs(mu) + 1e-10)

        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        outlier_frac = ((vals > q3 + 5 * iqr) | (vals < q1 - 5 * iqr)).mean()

        kurt = sp_kurtosis(vals, fisher=True, nan_policy="omit")

        fill_rate = (vals != 0).mean()

        reasons = []
        if cv > 50:
            reasons.append("high_cv")
        if outlier_frac > 0.05:
            reasons.append("outlier_dominated")
        if kurt > 100:
            reasons.append("extreme_kurtosis")
        if fill_rate < 0.10:
            reasons.append("sparse_fill")

        if reasons:
            log_vals = np.log1p(np.abs(vals))
            log_cv = log_vals.std() / (abs(log_vals.mean()) + 1e-10)
            log_kurt = sp_kurtosis(log_vals, fisher=True, nan_policy="omit")

            if log_cv < 10 and log_kurt < 20:
                kept.append(col)
                log_transform.append(col)
                continue
            removed.append({
                "name": col,
                "group": group_mapping.get(col, "?"),
                "reason": "|".join(reasons),
                "cv": round(cv, 2),
                "outlier_frac": round(outlier_frac, 4),
                "kurtosis": round(float(kurt), 1),
                "fill_rate": round(fill_rate, 4),
            })
            continue

        kept.append(col)

    return kept, log_transform, removed


def apply_transforms(df: pd.DataFrame, selected_cols: list,
                     log_cols: list) -> pd.DataFrame:
    """Select columns and apply log1p transforms."""
    missing = [c for c in selected_cols if c not in df.columns]
    for c in missing:
        df[c] = 0.0
    out = df[selected_cols].copy()
    for c in log_cols:
        if c in out.columns:
            out[c] = np.log1p(np.abs(out[c].astype(np.float64))).astype(np.float32)
    return out


def generate_report(
    all_raw_cols, kept_after_var, var_removed,
    kept_after_corr, corr_removed,
    kept_after_stab, stab_removed,
    log_transform_cols, group_mapping,
) -> str:
    """Generate a markdown feature selection report."""
    selected_features = kept_after_stab
    grp_before = Counter(group_mapping.get(c, "?") for c in all_raw_cols if c in group_mapping)
    selected_groups = Counter(group_mapping.get(c, "?") for c in selected_features)

    lines = [
        "# Feature Selection Report\n",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}\n",
        "## Summary\n",
        f"- Raw features extracted: **{len(all_raw_cols)}**",
        f"- After variance filtering: **{len(kept_after_var)}** (removed {len(var_removed)})",
        f"- After correlation pruning: **{len(kept_after_corr)}** (removed {len(corr_removed)})",
        f"- After stability filtering: **{len(kept_after_stab)}** (removed {len(stab_removed)})",
        f"- Log-transformed features: **{len(log_transform_cols)}**",
        f"- **Final selected: {len(selected_features)} features in {len(selected_groups)} groups**\n",
        "## Per-Group Summary\n",
        "| Group | Raw | After Var | After Corr | After Stab |",
        "|-------|-----|-----------|------------|------------|",
    ]

    for grp in sorted(grp_before.keys()):
        raw_n = grp_before[grp]
        var_n = Counter(group_mapping.get(c, "?") for c in kept_after_var).get(grp, 0)
        cor_n = Counter(group_mapping.get(c, "?") for c in kept_after_corr).get(grp, 0)
        stb_n = selected_groups.get(grp, 0)
        lines.append(f"| {grp} | {raw_n} | {var_n} | {cor_n} | {stb_n} |")

    lines.append(f"\n## Selected Features ({len(selected_features)} total)\n")
    for grp in sorted(selected_groups.keys()):
        cols = [c for c in selected_features if group_mapping.get(c, "?") == grp]
        lines.append(f"### {grp} ({len(cols)} features)")
        for c in cols:
            tag = " [log1p]" if c in log_transform_cols else ""
            lines.append(f"- `{c}`{tag}")
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    script_dir = Path(__file__).resolve().parent
    default_config_path = script_dir / "feature_extraction_config.json"
    default_workers = max(1, (os.cpu_count() or 1) - 1)

    parser = argparse.ArgumentParser(description="Stage 1+2: Data Preprocessing and Feature Selection")
    parser.add_argument("--config", type=str, default=None,
                        help=f"Path to JSON config file (defaults to {default_config_path})")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (overrides config)")
    parser.add_argument("--corr-threshold", type=float, default=None,
                        help="Correlation pruning threshold (overrides config)")
    parser.add_argument("--norm-var-threshold", type=float, default=None,
                        help="Normalized variance threshold for continuous features (overrides config)")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path
    config = {}
    if config_path.is_file():
        try:
            with open(config_path, "r") as cf:
                config = json.load(cf)
            print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Warning: Failed to parse config file {config_path}: {e}")
    else:
        print(f"No config file found at {config_path}, using defaults and CLI args")

    base_dir = script_dir.parent.parent

    dataset_dirs_cfg = config.get("dataset_dirs")
    if isinstance(dataset_dirs_cfg, dict):
        # Config paths are relative to the script directory (where config file is)
        dataset_dirs = OrderedDict([(k, (script_dir / v).resolve()) for k, v in dataset_dirs_cfg.items()])
    else:
        dataset_dirs = OrderedDict([
            ("benign_train", base_dir / "BENIGN_TRAIN_DATASET"),
            ("benign_val", base_dir / "BENIGN_VALIDATION_DATASET"),
            ("benign_test", base_dir / "BENIGN_TEST_DATASET"),
            ("malware_val", base_dir / "MALWARE_VALIDATION_DATASET"),
            ("malware_test", base_dir / "MALWARE_TEST_DATASET"),
        ])

    # Merge CLI args with config (CLI overrides config)
    workers = args.workers if args.workers is not None else config.get("workers", default_workers)
    corr_threshold = args.corr_threshold if args.corr_threshold is not None else config.get("corr_threshold", 0.95)
    norm_var_threshold = args.norm_var_threshold if args.norm_var_threshold is not None else config.get("norm_var_threshold", 1e-7)

    # Override module-level constants if specified in config
    global PARSE_TIMEOUT, MAX_SECTIONS, HASH_SIZES
    PARSE_TIMEOUT = int(config.get("parse_timeout", PARSE_TIMEOUT))
    MAX_SECTIONS = int(config.get("max_sections", MAX_SECTIONS))
    if "hash_sizes" in config and isinstance(config["hash_sizes"], dict):
        HASH_SIZES.update(config["hash_sizes"])

    print(f"Configuration: workers={workers}, corr_threshold={corr_threshold}, norm_var_threshold={norm_var_threshold}, parse_timeout={PARSE_TIMEOUT}, max_sections={MAX_SECTIONS}")

    data_raw_dir = script_dir.parent / "data" / "raw"
    data_clean_dir = script_dir.parent / "data" / "cleaned"
    schema_dir = script_dir.parent / "schemas"
    report_dir = script_dir.parent / "reports"
    for d in [data_raw_dir, data_clean_dir, schema_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 1: PE FEATURE EXTRACTION")
    print("=" * 70)

    datasets = {}
    for name, directory in dataset_dirs.items():
        files = scan_pe_files(directory)
        datasets[name] = files
        print(f"  {name}: {len(files)} files from {directory}")

    all_results = {}
    all_failed = {}
    for name, files in datasets.items():
        results, failed = batch_extract(files, name, workers)
        all_results[name] = results
        all_failed[name] = failed

    dfs_raw = {}
    for name, results in all_results.items():
        df = _sanitize_dataframe(pd.DataFrame(results))
        dfs_raw[name] = df

    feature_cols = _build_canonical_feature_order(dfs_raw)
    for name, df in dfs_raw.items():
        df_aligned = _align_dataframe(df, feature_cols)
        dfs_raw[name] = df_aligned
        out_path = data_raw_dir / f"{name}_raw.parquet"
        df_aligned.to_parquet(out_path, engine="pyarrow", compression="snappy")
        print(f"Saved {name}: {df_aligned.shape} -> {out_path}")

    group_mapping = build_group_mapping(feature_cols)
    with open(schema_dir / "feature_group_mapping.json", "w") as fh:
        json.dump(group_mapping, fh, indent=2)

    schema = {
        "schema_version": "1.0",
        "total_raw_features": len(feature_cols),
        "feature_columns": feature_cols,
        "group_mapping": group_mapping,
        "hash_sizes": HASH_SIZES,
    }
    with open(schema_dir / "feature_schema.json", "w") as fh:
        json.dump(schema, fh, indent=2)

    grp_counts = Counter(group_mapping.values())
    print(f"\n{'=' * 70}")
    print("EXTRACTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total features: {len(feature_cols)}")
    print(f"Feature groups ({len(grp_counts)}):")
    for grp, cnt in sorted(grp_counts.items(), key=lambda x: -x[1]):
        print(f"  {grp:30s} {cnt:4d}")

    total_failed = sum(len(f) for f in all_failed.values())
    if total_failed > 0:
        print(f"\nTotal failed: {total_failed}")
        for name, failed in all_failed.items():
            if failed:
                print(f"  {name}: {len(failed)} failures")

    print("\n" + "=" * 70)
    print("STAGE 2: PRE-TRAINING FEATURE SELECTION")
    print("=" * 70)

    df_train = dfs_raw["benign_train"]
    all_raw_cols = list(df_train.columns)
    print(f"\nTotal raw features: {len(all_raw_cols)}")

    print("\n--- Variance Filtering ---")
    kept_after_var, var_removed = variance_filter(
        df_train,
        group_mapping,
        norm_var_threshold=norm_var_threshold,
    )
    print(f"  {len(all_raw_cols)} -> {len(kept_after_var)} features (removed {len(var_removed)})")

    if var_removed:
        var_df = pd.DataFrame(var_removed)
        print("  Removals by group:")
        for grp, cnt in var_df.groupby("group")["name"].count().sort_values(ascending=False).items():
            print(f"    {grp}: {cnt}")

    print("\n--- Correlation Pruning ---")
    df_var_filtered = df_train[kept_after_var]
    kept_after_corr, corr_removed = correlation_pruning(
        df_var_filtered,
        threshold=corr_threshold,
        group_mapping=group_mapping,
    )
    print(f"  {len(kept_after_var)} -> {len(kept_after_corr)} features (removed {len(corr_removed)})")

    if corr_removed:
        corr_df = pd.DataFrame(corr_removed)
        print("  Removals by group:")
        for grp, cnt in corr_df.groupby("dropped_group")["dropped"].count().sort_values(ascending=False).items():
            print(f"    {grp}: {cnt}")

    print("\n--- Stability Filtering ---")
    df_corr_filtered = df_train[kept_after_corr]
    kept_after_stab, log_transform_cols, stab_removed = stability_filter(
        df_corr_filtered,
        group_mapping,
    )
    print(f"  {len(kept_after_corr)} -> {len(kept_after_stab)} features (removed {len(stab_removed)})")
    print(f"  Log-transform rescued: {len(log_transform_cols)}")

    if stab_removed:
        stab_df = pd.DataFrame(stab_removed)
        print("  Removals by group:")
        for grp, cnt in stab_df.groupby("group")["name"].count().sort_values(ascending=False).items():
            print(f"    {grp}: {cnt}")

    selected_features = kept_after_stab

    print(f"\n--- Saving cleaned datasets ({len(selected_features)} features) ---")
    for name, df in dfs_raw.items():
        df_clean = apply_transforms(df, selected_features, log_transform_cols)
        out_path = data_clean_dir / f"{name}_clean.parquet"
        df_clean.to_parquet(out_path, engine="pyarrow", compression="snappy")
        print(f"  Saved {name}: {df_clean.shape} -> {out_path}")

    feature_status = {}
    for col in all_raw_cols:
        feature_status[col] = {"status": "removed", "stage": "initial"}

    var_removed_names = {r["name"] for r in var_removed}
    for col in all_raw_cols:
        if col in var_removed_names:
            feature_status[col] = {"status": "removed", "stage": "variance_filter"}

    corr_removed_names = {r["dropped"] for r in corr_removed}
    for col in kept_after_var:
        if col in corr_removed_names:
            feature_status[col] = {"status": "removed", "stage": "correlation_prune"}

    stab_removed_names = {r["name"] for r in stab_removed}
    for col in kept_after_corr:
        if col in stab_removed_names:
            feature_status[col] = {"status": "removed", "stage": "stability_filter"}

    for col in selected_features:
        if col in log_transform_cols:
            feature_status[col] = {"status": "selected_log_transform", "stage": "passed_all"}
        else:
            feature_status[col] = {"status": "selected", "stage": "passed_all"}

    schema = {
        "schema_version": "1.0",
        "created_date": time.strftime("%Y-%m-%d"),
        "total_raw_features": len(all_raw_cols),
        "selected_features": len(selected_features),
        "feature_order": selected_features,
        "log_transform_features": log_transform_cols,
        "feature_status": feature_status,
        "group_mapping": {c: group_mapping.get(c, "?") for c in selected_features},
    }
    schema_path = schema_dir / "feature_schema_selected.json"
    with open(schema_path, "w") as fh:
        json.dump(schema, fh, indent=2)
    print(f"\n  Feature schema saved to {schema_path}")

    report = generate_report(
        all_raw_cols, kept_after_var, var_removed,
        kept_after_corr, corr_removed,
        kept_after_stab, stab_removed,
        log_transform_cols, group_mapping,
    )
    report_path = report_dir / "feature_selection_report.md"
    report_path.write_text(report)
    print(f"  Report saved to {report_path}")

    selected_groups = Counter(group_mapping.get(c, "?") for c in selected_features)
    print(f"\n{'=' * 70}")
    print("FEATURE SELECTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Raw:      {len(all_raw_cols)} features")
    print(f"  Selected: {len(selected_features)} features in {len(selected_groups)} groups")
    print(f"  Log-transformed: {len(log_transform_cols)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
