# Isolation Forest Model Binary Format (Portable Contract)

Date: 2026-02-26
Scope: `embedded_phase/core/models/isolation_forest/if_components.h` (`If_tree_container`)

## 1) Purpose

This document defines the model binary architecture used by `If_tree_container::save_model_binary` and `load_model_binary`.

Design goals:
- Cross-platform persistence with explicit endianness.
- Versioned and extensible header.
- Payload integrity checking.
- Backward load compatibility for older model binaries.

---

## 2) File Layout

Canonical modern format:

```text
[MAGIC: 4 bytes = "IFMQ"]
[VERSION: uint16 LE]
[ENDIAN FLAG: uint8]
[HEADER SIZE: uint16 LE]
[DATA SIZE: uint64 LE]
[HEADER PADDING to HEADER SIZE]
[DATA PAYLOAD]
[CHECKSUM: uint32 LE]
```

Current values:
- `VERSION = 1`
- `ENDIAN FLAG = 1` (little-endian canonical storage)
- `HEADER SIZE = 32`

Payload schema:

```text
[threshold_bits: uint8]
[feature_bits: uint8]
[child_bits: uint8]
[leaf_size_bits: uint8]
[depth_bits: uint8]
[samples_per_tree: uint32 LE]
[threshold_offset: float32 LE]
[tree_count: uint32 LE]
for each tree:
  [tree_depth: uint16 LE]
  [node_count: uint32 LE]
  [packed_node_0: uint64 LE]
  ...
  [packed_node_n: uint64 LE]
```

---

## 3) Integrity & Validation

Checksum:
- Algorithm: FNV-1a 32-bit
- Seed: `2166136261`
- Domain: exact payload bytes
- Trailer: uint32 LE after payload

Load-time validation:
1. Verify header magic/version/endian/header-size.
2. Validate file-size coverage (`header + payload + checksum`).
3. Decode payload with LE readers.
4. Recompute and compare checksum.
5. Validate non-empty trees and per-tree node counts.

---

## 4) Compatibility

`load_model_binary` behavior:
- First tries modern `IFMQ` format.
- Falls back to legacy `IFR1` model format for backward compatibility.

`save_model_binary` always emits modern `IFMQ` format.

---

## 5) RAII/Atomicity Notes

- Save path writes `target.tmp` first, then renames to final path.
- No native-struct binary dump is used.
- All metadata/payload values are serialized with explicit LE helper functions.

---

## 6) Endpoint Portability

This contract is independent of host struct packing and host endianness.
The explicit little-endian, size-tagged header and payload checksum make binaries portable across endpoint platforms and toolchains.
