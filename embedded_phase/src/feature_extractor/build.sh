#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

LIEF_DIR="${ROOT_DIR}/embedded_phase/third_party/LIEF"
LIEF_BUILD_DIR="${LIEF_DIR}/build"

OUT="${SCRIPT_DIR}/lief_feature_extractor"
OPTIMIZED_FEATURE_LIST_PATH="${OPTIMIZED_FEATURE_LIST_PATH:-${FEATURE_NAMES_PATH:-${ROOT_DIR}/development_phase/results/optimized_feature_list.json}}"
COMPILED_CFG="${SCRIPT_DIR}/compiled_feature_config.hpp"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILD_DIR="${SCRIPT_DIR}/build"

if [[ ! -f "${LIEF_BUILD_DIR}/libLIEF.a" ]]; then
  echo "Missing ${LIEF_BUILD_DIR}/libLIEF.a"
  echo "Build LIEF first:"
  echo "  cd ${LIEF_DIR}"
  echo "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLIEF_PYTHON_API=OFF -DLIEF_EXAMPLES=OFF -DLIEF_TESTS=OFF -DLIEF_DOC=OFF -DLIEF_INSTALL=OFF"
  echo "  cmake --build build -j\$(nproc)"
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/generate_compiled_features.py" \
  --feature-names "${OPTIMIZED_FEATURE_LIST_PATH}" \
  --output "${COMPILED_CFG}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFEATURE_EXTRACTOR_FETCH_LIEF=OFF \
  -DLIEF_DIR="${LIEF_BUILD_DIR}"

cmake --build "${BUILD_DIR}" --target lief_feature_extractor -j"$(nproc)"

BIN_PATH="$(find "${BUILD_DIR}" -maxdepth 4 -type f \( -name 'lief_feature_extractor' -o -name 'lief_feature_extractor.exe' \) | head -n1)"
if [[ -z "${BIN_PATH}" ]]; then
  echo "Build succeeded but could not locate lief_feature_extractor binary in ${BUILD_DIR}"
  exit 1
fi

cp "${BIN_PATH}" "${OUT}"
chmod +x "${OUT}" || true

echo "Built: ${OUT}"
