# eml_data Binary File Format (Portable RAII Contract)

Date: 2026-02-26
Scope: `embedded_phase/core/ml/eml_data.h`

## 1) Purpose

This document defines the on-disk dataset format used by `eml_data` for cross-platform persistence.

Design goals:
- Stable across CPU endianness and platform ABIs.
- Explicit header metadata for forward compatibility.
- Integrity check for payload corruption detection.
- RAII-safe I/O implementation (no raw heap buffers for serialization I/O paths).

---

## 2) File Layout

Canonical modern format:

```text
[MAGIC: 4 bytes = "EMLD"]
[VERSION: uint16 LE]
[ENDIAN FLAG: uint8]
[HEADER SIZE: uint16 LE]
[DATA SIZE: uint64 LE]
[NUM SAMPLES: uint32 LE]
[NUM FEATURES: uint16 LE]
[QUANT BITS: uint8]
[LABEL SIZE BYTES: uint8]
[LABEL BITS: uint8]
[RESERVED: uint8]
[HEADER PADDING: remaining bytes up to HEADER SIZE]
[DATA PAYLOAD]
[CHECKSUM: uint32 LE]
```

Current values:
- `VERSION = 1`
- `ENDIAN FLAG = 1` (little-endian canonical storage)
- `HEADER SIZE = 32`

`DATA PAYLOAD` is a packed sequence of records:
- Record = `[label][packed quantized features]`
- `record_size = label_size + packed_feature_bytes`
- `packed_feature_bytes = ceil(num_features * quant_bits / 8)`

---

## 3) Integrity & Validation

Checksum:
- Algorithm: FNV-1a 32-bit
- Seed: `2166136261`
- Domain: exact `DATA PAYLOAD` bytes (header excluded)
- Stored at the end of file as uint32 LE

Validation on load:
1. Parse and validate header fields.
2. Validate `data_size == num_samples * record_size`.
3. Validate file length covers header + payload + checksum.
4. Recompute payload checksum and compare with stored checksum.

---

## 4) Compatibility Behavior

`eml_data` supports both:
- **Modern format** (`EMLD` + checksum)
- **Legacy format** (historical `[num_samples][num_features]` header, no checksum)

Write path (`releaseData(false)`) always emits modern format.
Read path accepts modern first, then falls back to legacy parsing.

---

## 5) RAII/Robustness Notes

- Serialization buffers are `std::vector<uint8_t>` based.
- Save path writes to `*.tmp` then atomically renames to target file.
- Append/update operations refresh modern header metadata and recompute checksum.
- Resize/truncation uses filesystem APIs to keep file size consistent with metadata.

---

## 6) Multi-Platform Contract

All multibyte numeric metadata uses **little-endian encoding** (explicit read/write helpers).
No host-native struct dump is used, preventing ABI/packing/endianness drift across endpoints.
