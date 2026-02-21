# Dataset Quantization Makefile (Isolation-focused)

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
INCLUDES = -I../../src
TARGET = processing_data
SOURCE = processing_data.cpp

CONFIG ?= quantization_config.json
OUTPUT_DIR = quantized_datasets

.PHONY: all build setup-dirs process test clean clean-all help status

all: build

build:
	@echo "🔧 Compiling $(TARGET)..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCE)
	@echo "✅ Compilation successful"

setup-dirs:
	@echo "📁 Creating directories..."
	mkdir -p datasets $(OUTPUT_DIR)
	@echo "✅ Directories ready"

process: build setup-dirs
	@echo "🚀 Running quantization workflow..."
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
	@echo "Dataset Quantization Makefile"
	@echo "============================"
	@echo "make build                 Compile processing_data"
	@echo "make process [CONFIG=...]  Run quantization script"
	@echo "make test                  Build + run default workflow"
	@echo "make status                Show current status"
	@echo "make clean                 Remove compiled binary"
	@echo "make clean-all             Remove compiled binary and quantized outputs"

.DEFAULT_GOAL := help
