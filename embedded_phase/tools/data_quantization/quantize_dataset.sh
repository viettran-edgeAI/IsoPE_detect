#!/bin/bash

# ESP32 Dataset Quantization Script
# This script quantizes CSV datasets and exports artifacts for isolation workflows

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

CONFIG_PATH="$SCRIPT_DIR/quantization_config.json"
HELP=false

# Function to show usage
show_usage() {
    echo -e "${CYAN}ESP32 Dataset Quantization Tool${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    echo "Usage: $0 [-c quantization_config.json]"
    echo ""
    echo "Configuration is provided in JSON (default: quantization_config.json in this folder)."
    echo "Fields: input_path, model_name, header, problem_type, quantization_bits,"
    echo "        remove_outliers"
    echo ""
    echo "Options:"
    echo "  -c, --config <file>   Path to configuration JSON (default: ./quantization_config.json)"
    echo "  -h, --help            Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -h|--help)
            HELP=true
            shift
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Show help and exit if requested
if [ "$HELP" = true ]; then
    show_usage
    exit 0
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo -e "${RED}Error: Config file '$CONFIG_PATH' not found${NC}"
    exit 1
fi

# Load configuration values once via Python (stdlib only)
# Supports both nested {"value": ...} and flat formats
readarray -t CFG < <(python3 - "$CONFIG_PATH" <<'PY'
import json, sys
cfg_path = sys.argv[1]
with open(cfg_path) as f:
    cfg = json.load(f)

def get_value(obj, key, default):
    """Extract value from either nested {value: ...} or flat format"""
    val = obj.get(key)
    if val is None:
        return default
    if isinstance(val, dict) and "value" in val:
        return val["value"]
    return val

def emit(val):
    print(val)

emit(get_value(cfg, "input_path", ""))
emit(get_value(cfg, "model_name", ""))
emit(str(get_value(cfg, "header", "auto")).lower())
emit(str(get_value(cfg, "problem_type", "isolation")).lower())
emit(get_value(cfg, "quantization_bits", 2))
PY
)

CSV_PATH="${CFG[0]}"
MODEL_NAME="${CFG[1]}"
HEADER="${CFG[2]}"
PROBLEM_TYPE="${CFG[3]}"
QUANT_BITS="${CFG[4]}"

if [[ -z "$CSV_PATH" ]]; then
    echo -e "${RED}Error: 'input_path' is missing in $CONFIG_PATH${NC}"
    exit 1
fi

if [[ ! -f "$CSV_PATH" ]]; then
    echo -e "${RED}Error: File '$CSV_PATH' not found${NC}"
    exit 1
fi

echo -e "${CYAN}=== ESP32 Dataset Quantization ===${NC}"
echo -e "${BLUE}Configuration (from ${CONFIG_PATH}):${NC}"
echo -e "  📁 Input file: ${GREEN}$CSV_PATH${NC}"
if [[ -n "$MODEL_NAME" ]]; then
    echo -e "  📛 Model name: ${GREEN}$MODEL_NAME${NC}"
else
    echo -e "  📛 Model name: ${GREEN}(derived from input)${NC}"
fi
echo -e "  📋 Header mode: ${GREEN}${HEADER:-auto}${NC}"
echo -e "  🧩 Problem type: ${GREEN}${PROBLEM_TYPE:-isolation}${NC}"
echo -e "  🧮 Quantization bits: ${GREEN}${QUANT_BITS:-2}${NC}"
echo ""

# Compile processing program if needed (quiet unless failure)
if [[ ! -f "processing_data" || "processing_data.cpp" -nt "processing_data" ]]; then
    if ! g++ -std=c++17 -I../../src -o processing_data processing_data.cpp; then
        echo -e "${RED}❌ Failed to compile processing program${NC}"
        exit 1
    fi
fi

# Run dataset processing with config-driven executable
if ! ./processing_data -c "$CONFIG_PATH"; then
    echo -e "${RED}❌ Dataset processing failed${NC}"
    exit 1
fi

# Extract base name for further operations
if [[ -n "$MODEL_NAME" && "$MODEL_NAME" != "auto" ]]; then
    BASENAME="$MODEL_NAME"
else
    BASENAME=$(basename "$CSV_PATH" .csv)
fi
CONFIG_DIR="$(cd "$(dirname "$CONFIG_PATH")" && pwd)"
RESULT_DIR="$CONFIG_DIR/quantized_datasets"

echo -e "\n${CYAN}Generated Files:${NC}"
if [[ -f "$RESULT_DIR/${BASENAME}_nml.csv" ]]; then
    echo -e "  📊 Normalized CSV: ${GREEN}$RESULT_DIR/${BASENAME}_nml.csv${NC}"
fi
if [[ -f "$RESULT_DIR/${BASENAME}_nml.bin" ]]; then
    echo -e "  💾 Binary dataset: ${GREEN}$RESULT_DIR/${BASENAME}_nml.bin${NC}"
fi
if [[ -f "$RESULT_DIR/${BASENAME}_qtz.bin" ]]; then
    echo -e "  📋 Quantizer: ${GREEN}$RESULT_DIR/${BASENAME}_qtz.bin${NC}"
fi
if [[ -f "$RESULT_DIR/${BASENAME}_dp.txt" ]]; then
    echo -e "  ⚙️  Parameters: ${GREEN}$RESULT_DIR/${BASENAME}_dp.txt${NC}"
fi

echo ""
