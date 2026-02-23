# Batch Dataset Quantization Makefile
# Builds the orchestrator that processes all five optimized CSV splits.
# The single-file quantization module is at tools/data_quantization/;
# build it first with:  cd ../../../tools/data_quantization && make build

CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
TARGET   = processing_data
SOURCE   = processing_data.cpp

CONFIG ?= quantization_config.json
OUTPUT_DIR = quantized_datasets

.PHONY: all build process test clean clean-all help status

all: build

build:
	@echo "🔧 Compiling $(TARGET) (orchestrator)..."
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE)
	@echo "✅ Compilation successful"

setup-dirs:
	@echo "📁 Creating output directory..."
	mkdir -p $(OUTPUT_DIR)
	@echo "✅ Directory ready"

process: build
	@echo "🚀 Running batch quantization..."
	chmod +x quantize_dataset.sh
	./quantize_dataset.sh -c $(CONFIG)

test: process
	@echo "✅ Test run completed"

clean:
	@echo "🧹 Removing binary..."
	rm -f $(TARGET)
	@echo "✅ Clean completed"

clean-all: clean
	@echo "🧹 Removing generated quantized files..."
	rm -rf $(OUTPUT_DIR)/*
	@echo "✅ Deep clean completed"

status:
	@echo "📊 Status"
	@echo "=========="
	@if [ -f "$(TARGET)" ]; then echo "✅ Binary: $(TARGET)"; else echo "❌ Binary: not built"; fi
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		COUNT=$$(find $(OUTPUT_DIR) -type f | wc -l); \
		echo "✅ Output dir: $(OUTPUT_DIR) ($$COUNT files)"; \
	else \
		echo "❌ Output dir missing: $(OUTPUT_DIR)"; \
	fi

help:
	@echo "Batch Dataset Quantization Makefile"
	@echo "===================================="
	@echo "make build                 Compile the orchestrator"
	@echo "make process [CONFIG=...]  Build + run batch quantization"
	@echo "make test                  Alias for process"
	@echo "make status                Show current status"
	@echo "make clean                 Remove compiled binary"
	@echo "make clean-all             Remove binary and all quantized outputs"
	@echo ""
	@echo "Prerequisite: build the single-file processor first:"
	@echo "  cd ../../../tools/data_quantization && make build"

.DEFAULT_GOAL := help
