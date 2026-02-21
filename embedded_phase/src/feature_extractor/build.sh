#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

LIEF_DIR="${ROOT_DIR}/embedded_phase/third_party/LIEF"
LIEF_BUILD_DIR="${LIEF_DIR}/build"

SRC="${SCRIPT_DIR}/lief_feature_extractor.cpp"
OUT="${SCRIPT_DIR}/lief_feature_extractor"

if [[ ! -f "${LIEF_BUILD_DIR}/libLIEF.a" ]]; then
  echo "Missing ${LIEF_BUILD_DIR}/libLIEF.a"
  echo "Build LIEF first:"
  echo "  cd ${LIEF_DIR}"
  echo "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLIEF_PYTHON_API=OFF -DLIEF_EXAMPLES=OFF -DLIEF_TESTS=OFF -DLIEF_DOC=OFF -DLIEF_INSTALL=OFF"
  echo "  cmake --build build -j\$(nproc)"
  exit 1
fi

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
  -I"${LIEF_DIR}/include" \
  -I"${LIEF_DIR}/src" \
  -I"${LIEF_BUILD_DIR}/include" \
  -I"${LIEF_BUILD_DIR}" \
  "${SRC}" \
  "${LIEF_BUILD_DIR}/libLIEF.a" \
  -ldl -lpthread -lz \
  -o "${OUT}"

echo "Built: ${OUT}"
