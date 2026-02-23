#!/bin/bash

# Dataset Batch Quantization Script
# Compiles the orchestrator (if needed) and runs it.
# The orchestrator reads model_name + input_dir from quantization_config.json
# and invokes tools/data_quantization/processing_data for each of the five
# optimized CSV splits.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

CONFIG_PATH="$SCRIPT_DIR/quantization_config.json"
HELP=false

show_usage() {
    echo -e "${CYAN}Dataset Batch Quantization Tool${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    echo "Usage: $0 [-c quantization_config.json]"
    echo ""
    echo "Reads model_name and input_dir from the config file, then quantizes:"
    echo "  <model_name>_ben_train.csv"
    echo "  <model_name>_ben_test.csv"
    echo "  <model_name>_ben_val.csv"
    echo "  <model_name>_mal_test.csv"
    echo "  <model_name>_mal_val.csv"
    echo ""
    echo "Config fields:"
    echo "  model_name        (required) e.g. \"iforest\""
    echo "  input_dir         (required) path to development_phase/data/optimized"
    echo "  quantization_bits 1-8 (default 2)"
    echo "  header            auto|yes|no"
    echo "  problem_type      isolation|classification|regression"
    echo "  remove_outliers   true|false"
    echo ""
    echo "Before running, ensure the single-file processor is built:"
    echo "  cd tools/data_quantization && make build"
    echo ""
    echo "Options:"
    echo "  -c, --config <file>   Path to configuration JSON (default: ./quantization_config.json)"
    echo "  -h, --help            Show this help message"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config) CONFIG_PATH="$2"; shift 2 ;;
        -h|--help)   HELP=true; shift ;;
        *) echo -e "${RED}Error: Unknown option $1${NC}"; show_usage; exit 1 ;;
    esac
done

if [ "$HELP" = true ]; then show_usage; exit 0; fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo -e "${RED}Error: Config file '$CONFIG_PATH' not found${NC}"
    exit 1
fi

# ── Ensure single-file processor is built ─────────────────────────────────────
SINGLE_PROC="$(realpath "$SCRIPT_DIR/../../../tools/data_quantization/processing_data" 2>/dev/null)"
if [[ ! -f "$SINGLE_PROC" ]]; then
    echo -e "${YELLOW}Single-file processor not found — building it now...${NC}"
    SINGLE_PROC_DIR="$(realpath "$SCRIPT_DIR/../../../tools/data_quantization")"
    if ! (cd "$SINGLE_PROC_DIR" && make build); then
        echo -e "${RED}❌ Failed to build single-file processor${NC}"
        exit 1
    fi
fi

# ── Compile orchestrator if needed ────────────────────────────────────────────
if [[ ! -f "$SCRIPT_DIR/processing_data" || \
      "$SCRIPT_DIR/processing_data.cpp" -nt "$SCRIPT_DIR/processing_data" ]]; then
    echo -e "${YELLOW}Compiling orchestrator...${NC}"
    if ! g++ -std=c++17 -O2 -o "$SCRIPT_DIR/processing_data" "$SCRIPT_DIR/processing_data.cpp"; then
        echo -e "${RED}❌ Failed to compile orchestrator${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Orchestrator compiled${NC}"
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo -e "${CYAN}=== Dataset Batch Quantization ===${NC}"
echo -e "${BLUE}Config: ${CONFIG_PATH}${NC}"
echo ""

if ! "$SCRIPT_DIR/processing_data" -c "$CONFIG_PATH"; then
    echo -e "${RED}❌ Batch quantization failed${NC}"
    exit 1
fi

# ── Report generated files ────────────────────────────────────────────────────
RESULT_DIR="$SCRIPT_DIR/quantized_datasets"
if [[ -d "$RESULT_DIR" ]]; then
    COUNT=$(find "$RESULT_DIR" -type f | wc -l)
    echo -e "\n${CYAN}Generated files in ${RESULT_DIR} (${COUNT} total):${NC}"
    find "$RESULT_DIR" -maxdepth 1 -type f | sort | while read -r f; do
        echo -e "  ${GREEN}$(basename "$f")${NC}"
    done
fi

