# Single-File Dataset Quantization Tool
# Processes one CSV file per invocation.
# Include path points to shared headers in embedded_phase/core/.

CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
INCLUDES = -I../../embedded_phase/src
TARGET   = processing_data
SOURCE   = processing_data.cpp

CONFIG ?= quantization_config.json
OUTPUT_DIR = quantized_datasets

.PHONY: all build setup-dirs process clean clean-all help status

all: build

build:
	@echo "🔧 Compiling $(TARGET)..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCE)
	@echo "✅ Compilation successful"

setup-dirs:
	@echo "📁 Creating output directory..."
	mkdir -p $(OUTPUT_DIR)
	@echo "✅ Directory ready"

process: build setup-dirs
	@echo "🚀 Running single-file quantization..."
	./$(TARGET) -c $(CONFIG)

clean:
	@echo "🧹 Removing binary..."
	rm -f $(TARGET)
	@echo "✅ Clean completed"

clean-all: clean
	@echo "🧹 Removing quantized outputs..."
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
	@echo "Single-File Dataset Quantization Makefile"
	@echo "========================================="
	@echo "make build                 Compile processing_data"
	@echo "make process [CONFIG=...]  Run quantization on the file in config"
	@echo "make status                Show current status"
	@echo "make clean                 Remove compiled binary"
	@echo "make clean-all             Remove binary and quantized outputs"

.DEFAULT_GOAL := help
