#pragma once

/**
 * @file xgboost_mcu.h
 * @brief XGBoost Implementation for ESP32 Microcontrollers
 * 
 * This file provides a memory-efficient XGBoost classifier for ESP32 devices.
 * It is designed to load and run models trained using the PC-based trainer
 * (xgboost_pc.cpp).
 * 
 * Features:
 * - Multi-class classification using softmax
 * - Memory-efficient tree representation (64-bit packed nodes)
 * - Support for quantized features (1-8 bits per feature)
 * - Optional internal feature preprocessing for raw inputs
 * 
 * Usage:
 *   XGBoost xgb("model_name");
 *   xgb.init();
 *   
 *   // With raw quantized features
 *   uint8_t features[NUM_FEATURES] = {...};
 *   auto result = xgb.predict(features, NUM_FEATURES);
 *   
 *   // With packed_vector
 *   packed_vector<8> features;
 *   auto result = xgb.predict(features);
 */

#include "xg_components.h"
#include "../../base/eml_base.h"
#include "../../base/eml_random.h"
#include "../../ml/eml_data.h"
#include "../../ml/eml_samples.h"
#include "../../ml/eml_quantize.h"
#include "../../ml/eml_logger.h"
#include "../../ml/eml_predict_result.h"
#include "../../ml/eml_metrics.h"
#include <cstdio>
#include <cmath>
#include <limits>

// Optional LUT-based softmax (avoids expf in inference)
// Define XG_SOFTMAX_USE_LUT to enable.
#ifndef XG_SOFTMAX_LUT_SIZE
#define XG_SOFTMAX_LUT_SIZE 128
#endif
#ifndef XG_SOFTMAX_LUT_MIN
#define XG_SOFTMAX_LUT_MIN (-8.0f)
#endif
#ifndef XG_SOFTMAX_LUT_MAX
#define XG_SOFTMAX_LUT_MAX (8.0f)
#endif

namespace eml {

    using xg_label_map_t = eml_label_t<problem_type::CLASSIFICATION>;
    using XG_data = eml_data<problem_type::CLASSIFICATION>;
    using XG_sample = eml_sample_t<problem_type::CLASSIFICATION>;
    using XG_random = eml_random;
    using XG_logger = eml_logger_t<XG_base>;
    using xg_predict_result_t = eml_predict_result_t<problem_type::CLASSIFICATION>;

    // ============================================================================
    // XGBoost Classifier
    // ============================================================================
    
    class XGBoost {
    private:
        #ifndef EML_STATIC_MODEL
        struct TrainingContext {
            XG_data train_data;
            XG_data valid_data;
            XG_data test_data;
            XG_random random_generator;
            
            // Current model predictions (raw scores before softmax)
            // Stored as [sample1_class1, sample1_class2, ..., sample1_classK, sample2_class1, ...]
            vector<float> predictions; 
            
            // Gradients and Hessians
            vector<float> gradients;
            vector<float> hessians;

            // RAM tracking
            XG_RAM_Tracker ram_tracker;
            
            // Two-phase training: calibration data
            float global_min_weight = std::numeric_limits<float>::max();
            float global_max_weight = std::numeric_limits<float>::lowest();
            bool calibration_complete = false;

            bool build_model;
            bool data_prepared;

            TrainingContext() : data_prepared(false), build_model(true) {}
            
            void release() {
                train_data.purgeData();
                valid_data.purgeData();
                test_data.purgeData();
                predictions.clear();
                gradients.clear();
                hessians.clear();
            }
        };

        TrainingContext* training_ctx = nullptr;
        #endif

        XG_base base;
        XG_config config;
        XG_tree_container forest;
        // Quantizer support (optional). Internally loaded from the model's *_qtz.bin
        // to enable predictRaw() and label string mapping, without exposing it publicly.
        eml_quantizer<problem_type::CLASSIFICATION> quantizer;

        // Inference buffers
        mutable vector<float> score_buffer;           // Per-class scores (float, for softmax)
        mutable vector<int32_t> score_buffer_q;       // Per-class scores (quantized)
        mutable packed_vector<8> feature_buffer;      // Quantized feature buffer
        mutable char path_buffer[XG_PATH_BUFFER] = {'\0'};

        // Model state
        bool is_initialized = false;
        XG_logger logger;

        #ifndef EML_STATIC_MODEL
        inline TrainingContext* ensure_training_context() {
            if (!training_ctx) {
                training_ctx = new TrainingContext();
                training_ctx->random_generator.seed(1234); // Default seed
            }
            return training_ctx;
        }

        inline void destroy_training_context() {
            if (training_ctx) {
                training_ctx->release();
                delete training_ctx;
                training_ctx = nullptr;
            }
        }
        #endif

    public:
        // ======================== Constructors ========================

        XGBoost() {}

        XGBoost(const char* model_name) {
            init(model_name);
        }

        ~XGBoost() {
            release();
        }

        // ======================== Optional Quantizer (internal) ========================

        // ======================== Initialization ========================

        /**
         * @brief Initialize the XGBoost model
         * @param model_name Name of the model (folder name in filesystem)
         * @return true if initialization successful
         */
        bool init(const char* model_name) {
            eml_debug(1, "🚀 Initializing XGBoost: ", model_name);

            #if defined(ESP32) && (RF_DEBUG_LEVEL > 0)
            // Check stack size
            UBaseType_t stackRemaining = uxTaskGetStackHighWaterMark(NULL);
            size_t stackBytes = stackRemaining * sizeof(StackType_t);
            if (stackBytes < 2048) {
                eml_debug(0, "⚠️ WARNING: Low stack space: ", (uint32_t)stackBytes);
            }
            #endif

            // 1) Base must be initialized first (scans for model files and validates resources)
            base.init(model_name);
            
            // Check if base initialization was successful
            if (!base.ready_to_use()) {
                eml_debug(0, "❌ Model resource initialization failed. Cannot proceed.");
                return false;
            }

            // 2) Logger next (shared MCU init ordering)
            logger.init(&base);

            // 3) Check if config file exists before attempting to load
            if (!base.config_file_exists()) {
                eml_debug(0, "❌ Configuration file not found. Cannot initialize model.");
                return false;
            }

            // 4) Load configuration (includes dataset metadata)
            base.get_config_path(path_buffer);
            if (!config.loadConfig(path_buffer)) {
                eml_debug(0, "❌ Failed to load configuration from: ", path_buffer);
                return false;
            }
            eml_debug(1, "✅ Configuration loaded successfully");
            eml_debug(2, "   Features: ", config.num_features, ", Labels: ", config.num_labels, ", Boost rounds: ", config.num_boost_rounds);

            // 5) Optional: try to load quantizer for raw input support and label mapping
            if (base.qtz_file_exists()) {
                char qtz_path[XG_PATH_BUFFER];
                base.get_qtz_path(qtz_path);
                if (quantizer.loadQuantizer(qtz_path)) {
                    eml_debug(1, "✅ Quantizer loaded successfully");
                    // Sync quantization coefficient if quantizer provides it
                    if (quantizer.getQuantizationCoefficient() > 0 && 
                        config.quantization_coefficient != quantizer.getQuantizationCoefficient()) {
                        config.quantization_coefficient = quantizer.getQuantizationCoefficient();
                        eml_debug(1, "   Synchronized quantization_coefficient from quantizer: ", (uint32_t)config.quantization_coefficient);
                    }
                }
            }

            // 6) Initialize forest container (initializes node_resource layout)
            if (!forest.init(&config)) {
                eml_debug(0, "❌ Failed to initialize forest container");
                return false;
            }

            // 7) Model is NOT loaded here - call loadModel() explicitly when needed
            // This saves RAM by deferring model loading until inference is required
            if (base.model_file_exists()) {
                eml_debug(1, "📁 Model file found. Call loadModel() to load into RAM for inference.");
            } else {
                eml_debug(1, "⚠️ Model file not found. Training will be required.");
            }

            // 8) Allocate inference buffers
            score_buffer.resize(config.num_labels, 0.0f);
            score_buffer_q.resize(config.num_labels, 0);
            feature_buffer.set_bits_per_value(config.quantization_coefficient);
            feature_buffer.resize(config.num_features, 0);

            is_initialized = true;
            eml_debug(1, "✅ XGBoost initialized successfully");
            printModelInfo();

            // Report resource status
            if (base.able_to_inference()) {
                eml_debug(1, "📊 Model is ready for inference");
            }
            if (base.able_to_training()) {
                eml_debug(1, "🎯 Model is ready for training/re-training");
            }

            return true;
        }

        /**
         * @brief Load/reload the forest model from file into RAM
         * @param force_reload If true, reload even if already loaded. Default: false.
         * @return true if successful
         * 
         * Call this method explicitly when you need the model for inference.
         * Model is not auto-loaded during init() to save RAM.
         */
        bool loadModel(bool force_reload = false) {
            // Skip if model already in memory (forest is loaded) unless force reload
            if (forest.loaded() && !force_reload) {
                eml_debug(1, "ℹ️ Model already in memory. Use loadModel(true) to force reload.");
                return true;
            }

            if (!base.ready_to_use()) {
                eml_debug(0, "❌ Model base not initialized. Call init() first.");
                return false;
            }

            base.get_model_path(path_buffer);
            if (!RF_FS_EXISTS(path_buffer)) {
                eml_debug(0, "❌ Model file not found: ", path_buffer);
                return false;
            }

            eml_debug(1, "📥 Loading model into RAM...");
            if (!forest.loadForest(path_buffer)) {
                eml_debug(0, "❌ Failed to load forest from: ", path_buffer);
                return false;
            }

            // Update base status after successful load
            base.set_model_file_status(true);
            
            // Sync quantization bits from model header if available
            const uint8_t model_quant_bits = forest.getQuantBits();
            if (model_quant_bits > 0 && config.quantization_coefficient != model_quant_bits) {
                config.quantization_coefficient = model_quant_bits;
                eml_debug(1, "   Synchronized quantization_coefficient from model: ", (uint32_t)config.quantization_coefficient);
            }

            // Resize score buffers to match loaded model
            if (score_buffer.size() != config.num_labels) {
                score_buffer.resize(config.num_labels, 0.0f);
                score_buffer_q.resize(config.num_labels, 0);
            }

            eml_debug(1, "✅ Model loaded into RAM - ready for inference");
            return true;
        }

        /**
         * @brief Release model from RAM (keeps resources for potential retraining)
         */
        void releaseModel() {
            forest.releaseForest();
            eml_debug(1, "🗑️ Model released from RAM");
        }

        /**
         * @brief Full release of all resources
         */
        void release() {
            forest.releaseForest();
            score_buffer.clear();
            score_buffer_q.clear();
            feature_buffer.clear();

            // Release internal quantizer memory.
            quantizer.releaseQuantizer(true);
            
            #ifndef EML_STATIC_MODEL
            destroy_training_context();
            #endif
            
            is_initialized = false;
            eml_debug(1, "🗑️ All XGBoost resources released");
        }

        /**
         * @brief Check if model is loaded and ready for inference
         */
        bool ready_for_inference() const {
            return is_initialized && forest.loaded();
        }

        // ======================== On-Device Training ========================
        #ifndef EML_STATIC_MODEL

        /**
         * @brief Start a training session
         * @return true if successful
         */
        bool begin_training_session() {
            ensure_training_context();
            logger.init(&base, true); // Keep old logs
            return true;
        }

        /**
         * @brief End training session and free memory
         */
        void end_training_session() {
            cleanup_training_data();
            destroy_training_context();
        }

        /**
         * @brief Clean up temporary training files
         */
        void cleanup_training_data() {
            #ifndef EML_STATIC_MODEL
            if (!training_ctx) return;
            
            eml_debug(1, "🧹 Cleaning up training files...");
            base.build_file_path(path_buffer, "_train.bin");
            RF_FS_REMOVE(path_buffer);
            base.build_file_path(path_buffer, "_valid.bin");
            RF_FS_REMOVE(path_buffer);
            base.build_file_path(path_buffer, "_test.bin");
            RF_FS_REMOVE(path_buffer);
            
            training_ctx->release();
            #endif
        }

        /**
         * @brief Prepare training/validation/testing data from base data file (_nml.bin)
         * @param train_ratio Ratio of samples for training (0.0 to 1.0). If < 0, uses config.train_ratio.
         * @param valid_ratio Ratio of samples for validation (0.0 to 1.0). If < 0, uses config.valid_ratio.
         * @param test_ratio  Ratio of samples for testing (0.0 to 1.0). If < 0, uses config.test_ratio.
         * @return true if successful
         */
        bool prepare_data(float train_ratio = -1.0f, float valid_ratio = -1.0f, float test_ratio = -1.0f) {
            #ifndef EML_STATIC_MODEL
            // Check if base data exists
            if (!base.base_data_exists()) {
                eml_debug(0, "❌ Base data file not found. Cannot prepare training data.");
                return false;
            }

            auto* ctx = ensure_training_context();

            float tr = (train_ratio < 0.0f) ? config.train_ratio : train_ratio;
            float vr = (valid_ratio < 0.0f) ? config.valid_ratio : valid_ratio;
            float ter = (test_ratio < 0.0f) ? config.test_ratio : test_ratio;

            #ifndef DEV_STAGE
            ter = 0.0f;
            #endif

            float sum = tr + vr + ter;
            if (sum <= 0.0f) {
                tr = 1.0f; vr = 0.0f; ter = 0.0f;
            } else if (sum > 1.0f) {
                tr /= sum; vr /= sum; ter /= sum;
            }
            
            char nml_path[XG_PATH_BUFFER];
            base.build_file_path(nml_path, "_nml.bin");
            
            if (!RF_FS_EXISTS(nml_path)) {
                eml_debug(0, "❌ Base data not found: ", nml_path);
                return false;
            }

            XG_data base_data;
            if (!base_data.init(nml_path, config.num_features, static_cast<xg_label_map_t>(config.num_labels), config.quantization_coefficient, config.num_samples)) return false;
            if (!base_data.loadData()) return false;
            
            // Define destinations
            base.build_file_path(path_buffer, "_train.bin");
            ctx->train_data.init(path_buffer, config.num_features, static_cast<xg_label_map_t>(config.num_labels), config.quantization_coefficient, config.num_samples);

            if (vr > 0.0f) {
                char valid_path[XG_PATH_BUFFER];
                base.build_file_path(valid_path, "_valid.bin");
                ctx->valid_data.init(valid_path, config.num_features, static_cast<xg_label_map_t>(config.num_labels), config.quantization_coefficient, config.num_samples);
            }
            
            #ifdef DEV_STAGE
            if (ter > 0.0f) {
                char test_path[XG_PATH_BUFFER];
                base.build_file_path(test_path, "_test.bin");
                ctx->test_data.init(test_path, config.num_features, static_cast<xg_label_map_t>(config.num_labels), config.quantization_coefficient, config.num_samples);
            }
            #endif

            // Stratified split logic
            eml_debug(0, "🔀 Stratified splitting data (train/valid/test): ", tr, "/", vr, "/", ter);
            
            size_t num_samples = base_data.size();
            unordered_map_s<xg_label_map_t, vector<sample_idx_type>> label_indices;
            label_indices.reserve(config.num_labels);
            
            for (sample_idx_type id = 0; id < (sample_idx_type)num_samples; id++) {
                label_indices[base_data.getLabel(id)].push_back(id);
            }
            
            for (auto& kv : label_indices) {
                auto& indices = kv.second;
                // Shuffle indices
                for (size_t i = indices.size() - 1; i > 0; i--) {
                    size_t j = ctx->random_generator.bounded(static_cast<uint32_t>(i + 1));
                    std::swap(indices[i], indices[j]);
                }
                
                const size_t total = indices.size();
                size_t valid_count = static_cast<size_t>(total * vr);
                size_t test_count = static_cast<size_t>(total * ter);
                if (valid_count + test_count > total) {
                    const size_t overflow = (valid_count + test_count) - total;
                    if (test_count >= overflow) test_count -= overflow;
                    else valid_count = (valid_count > overflow) ? (valid_count - overflow) : 0;
                }
                const size_t train_count = total - valid_count - test_count;

                XG_sample_id_set train_ids(train_count);
                for (size_t i = 0; i < train_count; i++) train_ids.push_back(indices[i]);
                if (train_ids.size() > 0) ctx->train_data.loadData(base_data, train_ids, true);

                if (valid_count > 0 && vr > 0.0f) {
                    XG_sample_id_set valid_ids(valid_count);
                    for (size_t i = train_count; i < train_count + valid_count; i++) valid_ids.push_back(indices[i]);
                    if (valid_ids.size() > 0) ctx->valid_data.loadData(base_data, valid_ids, true);
                }

                #ifdef DEV_STAGE
                if (test_count > 0 && ter > 0.0f) {
                    XG_sample_id_set test_ids(test_count);
                    for (size_t i = train_count + valid_count; i < total; i++) test_ids.push_back(indices[i]);
                    if (test_ids.size() > 0) ctx->test_data.loadData(base_data, test_ids, true);
                }
                #endif
            }
            
            ctx->train_data.releaseData(false);
            if (vr > 0.0f) ctx->valid_data.releaseData(false);
            #ifdef DEV_STAGE
            if (ter > 0.0f) ctx->test_data.releaseData(false);
            #endif
            base_data.releaseData(); 
            eml_debug(1, "✅ Data split complete. Train: ", ctx->train_data.size(), " Valid: ", ctx->valid_data.size(), " Test: ", ctx->test_data.size());
            return true;
            #else
            return false;
            #endif
        }

        /**
         * @brief Build/Train the XGBoost model on-device using Two-Phase Training with Per-Round Streaming
         * 
         * Memory-Optimized Training Strategy:
         * - Phase 1 (Calibration): Build trees to collect global min/max leaf weights for quantization
         * - Phase 2 (Streaming): Rebuild with streaming - quantize and save per-round, then release
         * 
         * This approach dramatically reduces peak RAM by:
         * 1. Never holding all trees in RAM simultaneously
         * 2. Streaming trees to file as they're built
         * 3. Using ID_vector for compact sample index storage
         * 
         * @return true if successful
         */
        bool build_model() {
            // Check if model is ready for training
            if (!base.able_to_training()) {
                eml_debug(0, "❌ Model is not ready for training.");
                eml_debug(0, "   Required: base data (_nml.bin), quantizer (_qtz.bin), config (_xgb_config.json)");
                if (!base.base_data_exists()) {
                    eml_debug(0, "   - Missing: base data file");
                }
                if (!base.qtz_file_exists()) {
                    eml_debug(0, "   - Missing: quantizer file");
                }
                if (!base.config_file_exists()) {
                    eml_debug(0, "   - Missing: config file");
                }
                eml_debug(0, "⛔ Training session aborted safely.");
                return false;
            }

            // Match MCU model ergonomics: allow calling build_model() directly.
            if (!training_ctx) {
                (void)begin_training_session();
            }

            auto* ctx = training_ctx;
            if (!ctx) {
                eml_debug(0, "❌ Training session not started");
                return false;
            }

            // Initialize RAM tracker
            ctx->ram_tracker.reset();
            size_t start_time = logger.drop_anchor();
            eml_debug(0, "🏗️ Starting XGBoost two-phase training (memory-optimized)...");

            // 1. Prepare training data
            base.build_file_path(path_buffer, "_train.bin");
            if (!prepare_data()) {
                eml_debug(0, "❌ Failed to prepare training data.");
                eml_debug(0, "⛔ Training session aborted safely.");
                end_training_session();
                return false;
            }
            
            if (!ctx->train_data.init(path_buffer, config.num_features, static_cast<xg_label_map_t>(config.num_labels), config.quantization_coefficient, config.num_samples)) {
                eml_debug(0, "❌ Error initializing training data");
                return false;
            }

            if (!ctx->train_data.loadData()) {
                eml_debug(0, "❌ Error loading training data to RAM");
                return false;
            }

            // Track training data RAM
            xg_sample_type num_samples = ctx->train_data.size();
            xg_label_type num_classes = config.num_labels;
            size_t train_data_bytes = ctx->train_data.memoryUsage();
            ctx->ram_tracker.add(train_data_bytes);

            // Load validation data (for early stopping) if enabled
            bool has_valid = false;
            if (config.valid_ratio > 0.0f) {
                base.build_file_path(path_buffer, "_valid.bin");
                if (RF_FS_EXISTS(path_buffer)) {
                    if (ctx->valid_data.init(path_buffer, config.num_features, static_cast<xg_label_map_t>(config.num_labels), config.quantization_coefficient, config.num_samples) &&
                        ctx->valid_data.loadData()) {
                        has_valid = (ctx->valid_data.size() > 0);
                        if (has_valid) {
                            ctx->ram_tracker.add(ctx->valid_data.memoryUsage());
                        } else {
                            ctx->valid_data.releaseData();
                        }
                    }
                }
            }

            // 2. Initialize predictions, gradients, and hessians
            size_t pred_buffer_bytes = static_cast<size_t>(num_samples) * num_classes * sizeof(float);
            ctx->predictions.assign(num_samples * num_classes, 0.0f);
            ctx->gradients.resize(num_samples * num_classes);
            ctx->hessians.resize(num_samples * num_classes);
            ctx->ram_tracker.add(pred_buffer_bytes * 3);  // predictions + gradients + hessians

            // 3. Prepare forest for training
            forest.releaseForest();

            // Early stopping setup (validation only)
            eval_metric eval_metric_type = stringToEvalMetric(config.eval_metric);
            if (eval_metric_type == eval_metric::UNKNOWN || isRegressionMetric(eval_metric_type)) {
                eml_debug(1, "⚠️ Unsupported eval_metric for classification. Falling back to mlogloss.");
                eval_metric_type = eval_metric::MLOGLOSS;
            }
            const bool use_early_stop = has_valid && config.early_stopping && (config.early_stopping_rounds > 0);
            float best_score = lowerIsBetter(eval_metric_type) ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
            uint16_t best_round = 0;
            uint16_t no_improve = 0;
            uint16_t actual_rounds = config.num_boost_rounds;

            // ==================================================================
            // PHASE 1: CALIBRATION PASS - Collect global min/max leaf weights
            // Note: No early stopping in calibration - we need all rounds to 
            // collect accurate weight statistics. Early stopping applies in Phase 2.
            // ==================================================================
            eml_debug(0, "📊 Phase 1: Calibration pass (collecting weight statistics)...");
            ctx->global_min_weight = std::numeric_limits<float>::max();
            ctx->global_max_weight = std::numeric_limits<float>::lowest();
            ctx->calibration_complete = false;

            // Reset predictions for calibration pass
            std::fill(ctx->predictions.begin(), ctx->predictions.end(), 0.0f);

            for (uint16_t round = 0; round < config.num_boost_rounds; ++round) {
                eml_debug_2(1, "🔵 [Calibration] Round: ", round + 1, "/", config.num_boost_rounds);

                // Update gradients and hessians
                compute_gradients_hessians();

                // Train one tree for each class (collect weight stats but don't save)
                for (xg_label_type k = 0; k < num_classes; ++k) {
                    XG_tree tree(static_cast<uint8_t>(round % 256));
                    tree.setResource(&forest.getNodeResource());
                    
                    if (!build_tree(tree, k)) {
                        eml_debug_2(0, "❌ Failed to build tree for class ", k, " in round ", round);
                        return false;
                    }

                    // Collect min/max weights from this tree
                    for (const auto& node : tree.build_nodes) {
                        if (node.is_leaf) {
                            if (node.weight < ctx->global_min_weight) ctx->global_min_weight = node.weight;
                            if (node.weight > ctx->global_max_weight) ctx->global_max_weight = node.weight;
                        }
                    }

                    // Update predictions using the tree
                    update_predictions(tree, k);
                    
                    // Release tree immediately after use (don't accumulate in RAM)
                    tree.clear();
                }

                logger.m_log("calibration round");
            }

            ctx->calibration_complete = true;
            eml_debug(1, "📊 Weight range: [", ctx->global_min_weight, ", ", ctx->global_max_weight, "]");

            // ==================================================================
            // PHASE 2: STREAMING PASS - Rebuild and stream to file
            // ==================================================================
            eml_debug(0, "💾 Phase 2: Streaming pass (quantize and save per-round)...");

            // Reset predictions for streaming pass
            std::fill(ctx->predictions.begin(), ctx->predictions.end(), 0.0f);
            
            // Reset early stopping for streaming pass
            best_score = lowerIsBetter(eval_metric_type) ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
            no_improve = 0;

            // Begin streaming save
            base.build_file_path(path_buffer, "_xgboost.bin");
            if (!forest.beginStreamingSave(path_buffer)) {
                eml_debug(0, "❌ Failed to begin streaming save");
                return false;
            }

            // Validation predictions for early stopping (if validation set exists)
            vector<float> valid_predictions;
            xg_sample_type valid_size = has_valid ? ctx->valid_data.size() : 0;
            if (has_valid && use_early_stop) {
                valid_predictions.assign(static_cast<size_t>(valid_size) * num_classes, 0.0f);
                ctx->ram_tracker.add(valid_predictions.size() * sizeof(float));
            }

            // Temporary storage for one round's trees
            vector<XG_tree> round_trees;
            round_trees.reserve(num_classes);
            size_t round_trees_bytes = 0;

            for (uint16_t round = 0; round < actual_rounds; ++round) {
                eml_debug_2(1, "🔵 [Streaming] Round: ", round + 1, "/", actual_rounds);

                // Update gradients and hessians
                compute_gradients_hessians();

                // Clear round trees buffer
                round_trees.clear();
                round_trees_bytes = 0;

                // Build trees for this round
                for (xg_label_type k = 0; k < num_classes; ++k) {
                    XG_tree tree(static_cast<uint8_t>(round % 256));
                    tree.setResource(&forest.getNodeResource());
                    
                    if (!build_tree(tree, k)) {
                        eml_debug_2(0, "❌ Failed to build tree for class ", k, " in round ", round);
                        return false;
                    }

                    // Update training predictions
                    update_predictions(tree, k);

                    // Update validation predictions for early stopping
                    if (has_valid && use_early_stop) {
                        for (xg_sample_type i = 0; i < valid_size; ++i) {
                            XG_sample sample = ctx->valid_data.getSample(i);
                            float weight = tree.predictSample(sample);
                            valid_predictions[i * num_classes + k] += config.learning_rate * weight;
                        }
                    }

                    // Track RAM for this tree
                    round_trees_bytes += tree.memoryUsage();
                    ctx->ram_tracker.add(tree.memoryUsage());

                    round_trees.push_back(std::move(tree));
                }

                // Stream this round's trees to file (quantizes and releases them)
                if (!forest.appendRoundToStream(round_trees)) {
                    eml_debug(0, "❌ Failed to append round to stream");
                    return false;
                }

                // Release RAM tracked for this round's trees
                ctx->ram_tracker.release(round_trees_bytes);

                // Early stopping evaluation on validation set
                if (has_valid && use_early_stop) {
                    float val_score = evaluate_predictions_as_metric(
                        valid_predictions, ctx->valid_data, eval_metric_type);
                    
                    eml_debug_2(1, "📊 Validation ", config.eval_metric, ": ", val_score);

                    const float delta = lowerIsBetter(eval_metric_type)
                        ? (best_score - val_score)
                        : (val_score - best_score);

                    if (delta > config.early_stopping_threshold) {
                        best_score = val_score;
                        best_round = round + 1;
                        no_improve = 0;
                    } else {
                        no_improve++;
                    }

                    if (no_improve >= config.early_stopping_rounds) {
                        eml_debug_2(0, "⏹️ Early stopping at round ", round + 1, ". Best round: ", best_round);
                        actual_rounds = static_cast<uint16_t>(round + 1);
                        break;
                    }
                }

                logger.m_log("streaming round");
            }

            // Release validation predictions
            if (has_valid && use_early_stop) {
                ctx->ram_tracker.release(valid_predictions.size() * sizeof(float));
                valid_predictions.clear();
            }

            // Finalize streaming save
            if (!forest.finalizeStreamingSave()) {
                eml_debug(0, "❌ Failed to finalize streaming save");
                return false;
            }

            // Update config
            config.num_boost_rounds = actual_rounds;
            config.total_trees = static_cast<uint32_t>(actual_rounds) * static_cast<uint32_t>(num_classes);
            base.set_model_file_status(true);
            eml_debug(1, "💾 Model saved to: ", path_buffer);

            // Release training buffers
            ctx->ram_tracker.release(pred_buffer_bytes * 3);
            ctx->predictions.clear();
            ctx->gradients.clear();
            ctx->hessians.clear();

            ctx->ram_tracker.release(train_data_bytes);
            eml_debug_2(1, "✅ Data saved(", num_samples, " samples trained)", "");
            ctx->train_data.releaseData();

            if (has_valid) {
                ctx->ram_tracker.release(ctx->valid_data.memoryUsage());
                ctx->valid_data.releaseData();
            }

            // Clean up temporary training files
            cleanup_training_data();

            // Post-training evaluation on test set (DEV_STAGE only)
            #ifdef DEV_STAGE
            if (config.test_ratio > 0.0f) {
                base.build_file_path(path_buffer, "_test.bin");
                if (RF_FS_EXISTS(path_buffer)) {
                    if (ctx->test_data.init(path_buffer, config.num_features, static_cast<xg_label_map_t>(config.num_labels), config.quantization_coefficient, config.num_samples) &&
                        ctx->test_data.loadData()) {
                        float test_score = 0.0f;
                        if (evaluate_dataset(ctx->test_data, eval_metric_type, test_score)) {
                            eml_debug_2(0, "✅ Test ", config.eval_metric, ": ", test_score);
                        }
                        ctx->test_data.releaseData();
                    }
                }
            }
            #endif
            
            size_t end_time = logger.drop_anchor();
            long unsigned dur = logger.t_log("Total training time", start_time, end_time, "s");
            
            // Print peak RAM usage
            size_t peak_ram = ctx->ram_tracker.getPeak();
            eml_debug(0, "📈 Peak RAM usage during training: ", peak_ram, " bytes (", peak_ram / 1024, " KB)");
            eml_debug(0, "✅ XGBoost two-phase training complete in ", dur, "s", "");

            is_initialized = true;
            
            // Note: Model is NOT in RAM after training. 
            // Call loadModel() to load the saved model for inference.
            eml_debug(1, "ℹ️ Call loadModel() to load the trained model for inference.");
            
            return true;
        }

    private:
        /**
         * @brief Compute gradients and hessians for multiclass softmax
         */
        void compute_gradients_hessians() {
            auto* ctx = training_ctx;
            xg_sample_type num_samples = ctx->train_data.size();
            xg_label_type num_classes = config.num_labels;

            vector<float> prob(num_classes);

            for (xg_sample_type i = 0; i < num_samples; ++i) {
                // Get raw scores for this sample
                for (xg_label_type k = 0; k < num_classes; ++k) {
                    prob[k] = ctx->predictions[i * num_classes + k];
                }

                // Apply softmax to get probabilities
                softmax(prob);

                // Get true label
                xg_label_type y = static_cast<xg_label_type>(ctx->train_data.getLabel(i));

                // Compute G and H
                for (xg_label_type k = 0; k < num_classes; ++k) {
                    float p = prob[k];
                    float target = (k == y) ? 1.0f : 0.0f;
                    
                    ctx->gradients[i * num_classes + k] = p - target;
                    // Standard second deriv for Softmax is p(1-p), 
                    // though some impls use a constant or simplified H.
                    // We'll use p*(1-p) for better convergence and safety.
                    ctx->hessians[i * num_classes + k] = std::max(p * (1.0f - p), 1e-4f);
                }
            }
        }

        /**
         * @brief Evaluate current model on a dataset using the configured eval metric
         */
        bool evaluate_dataset(const XG_data& data, eval_metric metric, float& out_score) const {
            if (data.size() == 0) return false;

            eml_classification_metrics metrics;
            metrics.init(static_cast<xg_label_map_t>(config.num_labels), metric);

            vector<int32_t> scores_q(config.num_labels, 0);
            vector<float> scores(config.num_labels, 0.0f);

            for (xg_sample_type i = 0; i < data.size(); ++i) {
                const XG_sample sample = data.getSample(i);

                // Reset scores
                for (xg_label_type c = 0; c < config.num_labels; ++c) {
                    scores_q[c] = 0;
                }

                // Compute raw scores from current forest
                // Note: During training, trees_by_class is used (build mode with float weights)
                for (xg_label_type c = 0; c < config.num_labels; ++c) {
                    const size_t n = forest.numTreesForClass(c);
                    for (size_t t = 0; t < n; ++t) {
                        scores[c] += forest.getTree(c, t).predictSample(sample.features);
                    }
                    scores[c] *= config.learning_rate;
                }

                // Convert to probabilities
                softmax(scores);

                const xg_label_type label = static_cast<xg_label_type>(data.getLabel(i));
                metrics.update_with_probabilities(static_cast<xg_label_map_t>(label), scores.data());
            }

            out_score = metrics.calculate_metric(metric);
            return true;
        }

        /**
         * @brief Evaluate a predictions array against a dataset using the configured eval metric
         * Used for early stopping during training when trees aren't stored in forest.
         * 
         * @param predictions Accumulated predictions array [samples * num_classes]
         * @param data Dataset to evaluate against (for labels)
         * @param metric Evaluation metric to compute
         * @return Metric score
         */
        float evaluate_predictions_as_metric(const vector<float>& predictions, 
                                             const XG_data& data, 
                                             eval_metric metric) const {
            if (data.size() == 0) return 0.0f;

            eml_classification_metrics metrics;
            metrics.init(static_cast<xg_label_map_t>(config.num_labels), metric);

            xg_label_type num_classes = config.num_labels;
            vector<float> probs(num_classes);

            for (xg_sample_type i = 0; i < data.size(); ++i) {
                // Get raw scores for this sample
                for (xg_label_type k = 0; k < num_classes; ++k) {
                    probs[k] = predictions[i * num_classes + k];
                }

                // Convert to probabilities
                softmax(probs);

                const xg_label_type label = static_cast<xg_label_type>(data.getLabel(i));
                metrics.update_with_probabilities(static_cast<xg_label_map_t>(label), probs.data());
            }

            return metrics.calculate_metric(metric);
        }

        /**
         * @brief Update predictions for all samples using the newly trained tree
         */
        void update_predictions(const XG_tree& tree, xg_label_type class_idx) {
            auto* ctx = training_ctx;
            xg_sample_type num_samples = ctx->train_data.size();
            xg_label_type num_classes = config.num_labels;

            for (xg_sample_type i = 0; i < num_samples; ++i) {
                XG_sample sample = ctx->train_data.getSample(i);
                float weight = tree.predictSample(sample);
                ctx->predictions[i * num_classes + class_idx] += config.learning_rate * weight;
            }
        }

        /**
         * @brief Build a single regression tree for class k
         */
        bool build_tree(XG_tree& tree, xg_label_type class_idx) {
            auto* ctx = training_ctx;
            xg_sample_type num_samples = ctx->train_data.size();
            
            // Initial sample indices for root
            ID_vector<xg_sample_type, 3> root_indices; // 32-bit indices
            root_indices.reserve(num_samples);
            for (xg_sample_type i = 0; i < num_samples; ++i) {
                root_indices.push_back(i);
            }

            // Queue for breadth-first construction
            struct NodeBuildInfo {
                xg_node_type node_idx;
                ID_vector<xg_sample_type, 3> sample_indices;
                uint8_t depth;
            };

            vector<NodeBuildInfo> queue;
            queue.push_back({0, std::move(root_indices), 0});

            // Build-time node buffer (float weights during training)
            vector<XG_Building_node> build_nodes;
            build_nodes.push_back(XG_Building_node());

            uint32_t head = 0;
            while (head < queue.size()) {
                NodeBuildInfo current = std::move(queue[head++]);
                
                float G = 0, H = 0;
                for (size_t i = 0; i < current.sample_indices.size(); ++i) {
                    xg_sample_type idx = current.sample_indices[i];
                    G += ctx->gradients[idx * config.num_labels + class_idx];
                    H += ctx->hessians[idx * config.num_labels + class_idx];
                }

                // Check termination criteria
                bool is_leaf = (current.depth >= config.max_depth) || 
                               (current.sample_indices.size() < config.min_child_weight);

                uint16_t best_feature = 0;
                uint16_t best_threshold = 0;
                float best_gain = -1.0f;

                if (!is_leaf) {
                    // Search for best split
                    if (!find_best_split(current.sample_indices, class_idx, best_feature, best_threshold, best_gain)) {
                        is_leaf = true;
                    } else if (best_gain < config.gamma) {
                        is_leaf = true;
                    }
                }

                if (is_leaf) {
                    // Set as leaf node (float weight during training)
                    float weight = -G / (H + config.lambda);
                    build_nodes[current.node_idx] = XG_Building_node::makeLeaf(weight);
                } else {
                    // Set as split node
                    uint32_t left_child_idx = build_nodes.size();
                    build_nodes[current.node_idx] = XG_Building_node::makeSplit(best_feature, best_threshold, left_child_idx);
                    
                    // Add child nodes placeholders
                    build_nodes.push_back(XG_Building_node()); // Left
                    build_nodes.push_back(XG_Building_node()); // Right

                    ID_vector<xg_sample_type, 3> left_indices;
                    ID_vector<xg_sample_type, 3> right_indices;

                    for (size_t i = 0; i < current.sample_indices.size(); ++i) {
                        xg_sample_type idx = current.sample_indices[i];
                        if (ctx->train_data.getSample(idx).features.get(best_feature) <= best_threshold) {
                            left_indices.push_back(idx);
                        } else {
                            right_indices.push_back(idx);
                        }
                    }

                    queue.push_back({left_child_idx, std::move(left_indices), (uint8_t)(current.depth + 1)});
                    queue.push_back({left_child_idx + 1, std::move(right_indices), (uint8_t)(current.depth + 1)});
                }
            }

            // Move float-based nodes directly to tree (no quantization during training)
            tree.setResource(&forest.getNodeResource());
            tree.build_nodes = std::move(build_nodes);
            tree.in_build_mode = true;
            tree.is_loaded = true;
            return true;
        }

        /**
         * @brief Find best split for a set of samples
         */
        bool find_best_split(const ID_vector<xg_sample_type, 3>& samples, xg_label_type class_idx, 
                             uint16_t& best_feature, uint16_t& best_threshold, float& best_gain) {
            auto* ctx = training_ctx;
            best_gain = -1.0f;

            float G_total = 0, H_total = 0;
            for (size_t i = 0; i < samples.size(); ++i) {
                xg_sample_type idx = samples[i];
                G_total += ctx->gradients[idx * config.num_labels + class_idx];
                H_total += ctx->hessians[idx * config.num_labels + class_idx];
            }

            uint16_t max_val = (1 << config.quantization_coefficient);

            // Iterate over all features
            for (uint16_t f = 0; f < config.num_features; ++f) {
                // For quantized features, we can iterate over all possible thresholds
                vector<float> G_bins(max_val, 0.0f);
                vector<float> H_bins(max_val, 0.0f);

                for (size_t i = 0; i < samples.size(); ++i) {
                    xg_sample_type idx = samples[i];
                    uint8_t val = ctx->train_data.getSample(idx).features.get(f);
                    if (val < max_val) {
                        G_bins[val] += ctx->gradients[idx * config.num_labels + class_idx];
                        H_bins[val] += ctx->hessians[idx * config.num_labels + class_idx];
                    }
                }

                float G_left = 0, H_left = 0;
                for (uint16_t t = 0; t < max_val - 1; ++t) {
                    G_left += G_bins[t];
                    H_left += H_bins[t];

                    float G_right = G_total - G_left;
                    float H_right = H_total - H_left;

                    if (H_left > 0 && H_right > 0) {
                        float gain = 0.5f * ( (G_left * G_left) / (H_left + config.lambda) + 
                                              (G_right * G_right) / (H_right + config.lambda) -
                                              (G_total * G_total) / (H_total + config.lambda) );
                        
                        if (gain > best_gain) {
                            best_gain = gain;
                            best_feature = f;
                            best_threshold = t;
                        }
                    }
                }
            }

            return best_gain > 0;
        }

        #endif

        // ======================== Prediction ========================
        public:

        /**
         * @brief Predict class from raw quantized features
         * @param features Pointer to quantized feature array
         * @param num_features Number of features
         * @param skip_softmax If true, skip softmax and use argmax of raw scores
         * @return Prediction result
         */
        xg_predict_result_t predict(const uint8_t* features, uint16_t num_features, bool skip_softmax = false) {
            xg_predict_result_t result;
            result.success = false;

            if (!is_initialized) {
                eml_debug(0, "❌ Model not initialized. Call init() first.");
                return result;
            }

            if (!forest.loaded()) {
                eml_debug(0, "❌ Model not loaded. Call loadModel() first.");
                return result;
            }

            if (!features || num_features != config.num_features) {
                eml_debug(0, "❌ Invalid features");
                return result;
            }

            size_t start_time = micros();

            // Reset score buffers
            for (size_t i = 0; i < score_buffer_q.size(); ++i) {
                score_buffer_q[i] = 0;
            }

            // Choose prediction path based on model format
            if (forest.isPackedStream()) {
                // New packed format: trees stored round-major (tree[t] = class t % num_classes)
                // Use float accumulation for better precision with low scale_factor_bits
                const float lr = forest.getLearningRate();
                const uint32_t total_trees = static_cast<uint32_t>(forest.numTreesTotal());
                const uint16_t num_classes = static_cast<uint16_t>(forest.numClasses());
                
                // Reset float buffer for accumulation
                for (size_t i = 0; i < score_buffer.size(); ++i) {
                    score_buffer[i] = 0.0f;
                }
                
                for (uint32_t t = 0; t < total_trees; ++t) {
                    uint16_t class_idx = static_cast<uint16_t>(t % num_classes);
                    float weight = forest.predictTreePacked(t, features, num_features);
                    score_buffer[class_idx] += lr * weight;
                }
                // Scores are already in float, no dequantization needed
            } else {
                // Build mode: trees stored per-class with float weights
                for (xg_label_type c = 0; c < config.num_labels; ++c) {
                    size_t n = forest.numTreesForClass(c);
                    const bool use_limit = (config.early_exit_score_limit > 0.0f);
                    for (size_t t = 0; t < n; ++t) {
                        float weight = forest.getTree(c, t).predictSample(features, num_features);
                        score_buffer[c] += weight;
                        if (use_limit && (score_buffer[c] >= config.early_exit_score_limit || score_buffer[c] <= -config.early_exit_score_limit)) {
                            break;
                        }
                    }
                }

                // Apply learning rate
                for (size_t i = 0; i < score_buffer.size(); ++i) {
                    score_buffer[i] *= config.learning_rate;
                }
            }

            // Apply softmax (optional) and find best class
            if (!skip_softmax) {
                softmax(score_buffer);
            }
            
            xg_label_type best_class = 0;
            float max_prob = score_buffer[0];
            for (xg_label_type k = 1; k < config.num_labels; ++k) {
                if (score_buffer[k] > max_prob) {
                    max_prob = score_buffer[k];
                    best_class = k;
                }
            }

            result.predicted_class = best_class;
            result.confidence = max_prob;
            result.prediction_time = micros() - start_time;
            result.success = true;

            // Get label string if quantizer available
            if (quantizer.loaded()) {
                if (best_class <= std::numeric_limits<xg_label_map_t>::max()) {
                    quantizer.getOriginalLabel(static_cast<xg_label_map_t>(best_class), result.label, XG_MAX_LABEL_LENGTH);
                } else {
                    snprintf(result.label, XG_MAX_LABEL_LENGTH, "class_%u", best_class);
                }
            } else {
                snprintf(result.label, XG_MAX_LABEL_LENGTH, "class_%u", best_class);
            }

            return result;
        }

        /**
         * @brief Predict class from packed_vector features
         * @param features Packed vector of quantized features
         * @param skip_softmax If true, skip softmax and use argmax of raw scores
         * @return Prediction result
         */
        template<uint8_t bits>
        xg_predict_result_t predict(const packed_vector<bits>& features, bool skip_softmax = false) {
            xg_predict_result_t result;
            result.success = false;

            if (!is_initialized) {
                eml_debug(0, "❌ Model not initialized. Call init() first.");
                return result;
            }

            if (!forest.loaded()) {
                eml_debug(0, "❌ Model not loaded. Call loadModel() first.");
                return result;
            }

            if (features.size() != config.num_features) {
                eml_debug(0, "❌ Feature count mismatch");
                return result;
            }

            size_t start_time = micros();

            // Reset score buffers
            for (size_t i = 0; i < score_buffer_q.size(); ++i) {
                score_buffer_q[i] = 0;
            }

            // Choose prediction path based on model format
            if (forest.isPackedStream()) {
                // New packed format: trees stored round-major (tree[t] = class t % num_classes)
                // Use float accumulation for better precision with low scale_factor_bits
                const float lr = forest.getLearningRate();
                const uint32_t total_trees = static_cast<uint32_t>(forest.numTreesTotal());
                const uint16_t num_classes = static_cast<uint16_t>(forest.numClasses());
                
                // Reset float buffer for accumulation
                for (size_t i = 0; i < score_buffer.size(); ++i) {
                    score_buffer[i] = 0.0f;
                }
                
                for (uint32_t t = 0; t < total_trees; ++t) {
                    uint16_t class_idx = static_cast<uint16_t>(t % num_classes);
                    float weight = forest.predictTreePacked(t, features);
                    score_buffer[class_idx] += lr * weight;
                }
                // Scores are already in float, no dequantization needed
            } else {
                // Build mode: trees stored per-class with float weights
                for (xg_label_type c = 0; c < config.num_labels; ++c) {
                    size_t n = forest.numTreesForClass(c);
                    const bool use_limit = (config.early_exit_score_limit > 0.0f);
                    for (size_t t = 0; t < n; ++t) {
                        float weight = forest.getTree(c, t).predictSample(features);
                        score_buffer[c] += weight;
                        if (use_limit && (score_buffer[c] >= config.early_exit_score_limit || score_buffer[c] <= -config.early_exit_score_limit)) {
                            break;
                        }
                    }
                }

                // Apply learning rate
                for (size_t i = 0; i < score_buffer.size(); ++i) {
                    score_buffer[i] *= config.learning_rate;
                }
            }

            // Apply softmax (optional) and find best class
            if (!skip_softmax) {
                softmax(score_buffer);
            }
            
            xg_label_type best_class = 0;
            float max_prob = score_buffer[0];
            for (xg_label_type k = 1; k < config.num_labels; ++k) {
                if (score_buffer[k] > max_prob) {
                    max_prob = score_buffer[k];
                    best_class = k;
                }
            }

            result.predicted_class = best_class;
            result.confidence = max_prob;
            result.prediction_time = micros() - start_time;
            result.success = true;

            // Get label string
            if (quantizer.loaded()) {
                if (best_class <= std::numeric_limits<xg_label_map_t>::max()) {
                    quantizer.getOriginalLabel(static_cast<xg_label_map_t>(best_class), result.label, XG_MAX_LABEL_LENGTH);
                } else {
                    snprintf(result.label, XG_MAX_LABEL_LENGTH, "class_%u", best_class);
                }
            } else {
                snprintf(result.label, XG_MAX_LABEL_LENGTH, "class_%u", best_class);
            }

            return result;
        }

        /**
         * @brief Predict with raw float features (requires quantizer)
         * @param raw_features Pointer to raw float feature values
         * @param num_features Number of features
         * @param skip_softmax If true, skip softmax and use argmax of raw scores
         * @return Prediction result
         */
        xg_predict_result_t predictRaw(const float* raw_features, uint16_t num_features, bool skip_softmax = false) {
            xg_predict_result_t result;
            result.success = false;

            if (!quantizer.loaded()) {
                eml_debug(0, "❌ Quantizer required for raw prediction");
                return result;
            }

            if (num_features != config.num_features) {
                eml_debug(0, "❌ Feature count mismatch");
                return result;
            }

            // Quantize features using quantizer
            feature_buffer.resize(num_features, 0);
            quantizer.quantizeFeatures(raw_features, feature_buffer, nullptr, nullptr);

            return predict(feature_buffer, skip_softmax);
        }

        /**
         * @brief Get class probabilities for last prediction
         * @param probs Output array (must have num_labels elements)
         */
        void getLastProbabilities(float* probs) const {
            if (probs && score_buffer.size() > 0) {
                for (size_t i = 0; i < score_buffer.size(); ++i) {
                    probs[i] = score_buffer[i];
                }
            }
        }

        // ======================== Setters ========================

        /**
         * @brief Set training ratio for dataset splitting
         * @param ratio Float value in range (0.0, 1.0)
         */
        void set_train_ratio(float ratio) {
            if (ratio > 0.0f && ratio <= 1.0f) {
                config.train_ratio = ratio;
            }
        }

        /**
         * @brief Set test ratio for dataset splitting
         * @param ratio Float value in range (0.0, 1.0)
         */
        void set_test_ratio(float ratio) {
            if (ratio >= 0.0f && ratio <= 1.0f) {
                config.test_ratio = ratio;
            }
        }

        /**
         * @brief Set validation ratio for dataset splitting
         * @param ratio Float value in range (0.0, 1.0)
         */
        void set_valid_ratio(float ratio) {
            if (ratio >= 0.0f && ratio <= 1.0f) {
                config.valid_ratio = ratio;
            }
        }

        /**
         * @brief Set random seed for reproducibility
         * @param seed Random seed value (default is 42)
         */
        void set_random_seed(uint32_t seed) {
            config.random_seed = seed;
            #ifndef EML_STATIC_MODEL
            if (training_ctx) {
                training_ctx->random_generator.seed(seed);
            }
            #endif
        }

        /**
         * @brief Use default random seed (42)
         */
        void use_default_seed() {
            set_random_seed(42);
        }

        /**
         * @brief Set number of boosting rounds
         * @param rounds Number of boosting rounds (trees per class)
         */
        void set_num_boost_rounds(uint16_t rounds) {
            if (rounds > 0 && rounds <= XG_MAX_BOOST_ROUNDS) {
                config.num_boost_rounds = rounds;
            }
        }

        /**
         * @brief Set learning rate (eta)
         * @param rate Learning rate value in range (0.0, 1.0]
         */
        void set_learning_rate(float rate) {
            if (rate > 0.0f && rate <= 1.0f) {
                config.learning_rate = rate;
            }
        }

        /**
         * @brief Set maximum tree depth
         * @param depth Maximum depth of each tree
         */
        void set_max_depth(uint8_t depth) {
            if (depth > 0 && depth <= XG_MAX_DEPTH) {
                config.max_depth = depth;
            }
        }

        /**
         * @brief Set L2 regularization term (lambda)
         * @param lambda Regularization value >= 0
         */
        void set_lambda(float lambda) {
            if (lambda >= 0.0f) {
                config.lambda = lambda;
            }
        }

        /**
         * @brief Set L1 regularization term (alpha)
         * @param alpha Regularization value >= 0
         */
        void set_alpha(float alpha) {
            if (alpha >= 0.0f) {
                config.alpha = alpha;
            }
        }

        /**
         * @brief Set minimum loss reduction for split (gamma)
         * @param gamma Minimum loss reduction value >= 0
         */
        void set_gamma(float gamma) {
            if (gamma >= 0.0f) {
                config.gamma = gamma;
            }
        }

        /**
         * @brief Set subsample ratio for training data
         * @param ratio Subsample ratio in range (0.0, 1.0]
         */
        void set_subsample(float ratio) {
            if (ratio > 0.0f && ratio <= 1.0f) {
                config.subsample = ratio;
            }
        }

        /**
         * @brief Set column sample ratio by tree
         * @param ratio Column sample ratio in range (0.0, 1.0]
         */
        void set_colsample_bytree(float ratio) {
            if (ratio > 0.0f && ratio <= 1.0f) {
                config.colsample_bytree = ratio;
            }
        }

        /**
         * @brief Set minimum child weight
         * @param weight Minimum sum of instance weight needed in a child
         */
        void set_min_child_weight(uint16_t weight) {
            config.min_child_weight = weight;
        }

        /**
         * @brief Set max delta step for weight updates
         * @param step Max delta step value >= 0
         */
        void set_max_delta_step(float step) {
            if (step >= 0.0f) {
                config.max_delta_step = step;
            }
        }

        /**
         * @brief Set maximum samples to use for training (0 = unlimited)
         * @param max_samples Maximum number of samples
         */
        void set_max_samples(xg_sample_type max_samples) {
            config.max_samples = max_samples;
        }

        /**
         * @brief Enable or disable on-device retraining
         * @param enable True to enable retraining
         */
        void set_enable_retrain(bool enable) {
            config.enable_retrain = enable;
        }

        /**
         * @brief Enable early stopping during training
         * @param enable True to enable early stopping
         * @param rounds Number of rounds without improvement before stopping
         * @param threshold Minimum improvement threshold
         */
        void set_early_stopping(bool enable, uint16_t rounds = 10, float threshold = 0.001f) {
            config.early_stopping = enable;
            config.early_stopping_rounds = rounds;
            config.early_stopping_threshold = threshold;
        }

        // ======================== Getters ========================

        bool isInitialized() const { return is_initialized; }
        uint16_t getNumFeatures() const { return config.num_features; }
        xg_label_type getNumLabels() const { return config.num_labels; }
        uint16_t getNumBoostRounds() const { return config.num_boost_rounds; }
        size_t getNumTrees() const { return forest.numTreesTotal(); }
        float getLearningRate() const { return config.learning_rate; }

        uint8_t getMaxDepth() const { return config.max_depth; }
        float getLambda() const { return config.lambda; }
        float getAlpha() const { return config.alpha; }
        float getGamma() const { return config.gamma; }
        float getSubsample() const { return config.subsample; }
        float getColsampleBytree() const { return config.colsample_bytree; }
        uint16_t getMinChildWeight() const { return config.min_child_weight; }
        float getMaxDeltaStep() const { return config.max_delta_step; }
        uint32_t getRandomSeed() const { return config.random_seed; }
        uint8_t getQuantizationCoefficient() const { return config.quantization_coefficient; }

        float getTrainRatio() const { return config.train_ratio; }
        float getTestRatio() const { return config.test_ratio; }
        float getValidRatio() const { return config.valid_ratio; }

        bool isEarlyStoppingEnabled() const { return config.early_stopping; }
        bool isRetrainEnabled() const { return config.enable_retrain; }
        
        /**
         * @brief Check if model is ready for inference
         */
        bool able_to_inference() const { return base.able_to_inference(); }

        /**
         * @brief Check if model is ready for training
         */
        bool able_to_training() const { return base.able_to_training(); }

        /**
         * @brief Get model name
         * @param name Output buffer for model name
         * @param length Buffer length
         */
        void get_model_name(char* name, size_t length) const {
            base.get_model_name(name, length);
        }

        /**
         * @brief Get original label string from internal normalized label
         * @param normalizedLabel Internal label index
         * @param outLabel Output pointer to label string
         * @param outLength Optional output for label length
         * @return true if label found
         */
        bool get_label_view(xg_label_type normalizedLabel, const char** outLabel, uint16_t* outLength = nullptr) const {
            return quantizer.getOriginalLabelView(normalizedLabel, outLabel, outLength);
        }

        const XG_config& getConfig() const { return config; }
        const XG_tree_container& getForest() const { return forest; }

        /**
         * @brief Get total memory usage of the model
         */
        size_t memoryUsage() const {
            return sizeof(XGBoost) + 
                   config.memoryUsage() + 
                   forest.memoryUsage() +
                   score_buffer.size() * sizeof(float) +
                   score_buffer_q.size() * sizeof(int32_t) +
                   feature_buffer.memory_usage();
        }

        /**
         * @brief Print model information
         */
        void printModelInfo() const {
            eml_debug(0, "📊 XGBoost Model Info:");
            eml_debug(0, "   Features: ", (uint32_t)config.num_features);
            eml_debug(0, "   Labels: ", (uint32_t)config.num_labels);
            eml_debug(0, "   Boost rounds: ", (uint32_t)config.num_boost_rounds);
            eml_debug(0, "   Learning rate: ", config.learning_rate);
            
            if (forest.loaded()) {
                eml_debug(0, "   Model status: ✅ Loaded in RAM");
                eml_debug(0, "   Total trees: ", (uint32_t)forest.numTreesTotal());
                eml_debug(0, "   Total nodes: ", forest.getTotalNodes());
                eml_debug(0, "   Max depth: ", (uint32_t)forest.getMaxDepth());
                eml_debug(0, "   Format: ", forest.isPackedStream() ? "Packed stream" : "Tree structure");
                eml_debug(0, "   Memory (bytes): ", (uint32_t)memoryUsage());
            } else {
                eml_debug(0, "   Model status: ⚪ Not loaded (call loadModel())");
            }
        }

    private:
        // ======================== Internal utilities ========================

#if defined(XG_SOFTMAX_USE_LUT)
        static inline float exp_lut(float x) {
            static bool inited = false;
            static float lut[XG_SOFTMAX_LUT_SIZE];
            const float min_x = XG_SOFTMAX_LUT_MIN;
            const float max_x = XG_SOFTMAX_LUT_MAX;
            const float step = (max_x - min_x) / static_cast<float>(XG_SOFTMAX_LUT_SIZE - 1);

            if (!inited) {
                for (int i = 0; i < (int)XG_SOFTMAX_LUT_SIZE; ++i) {
                    lut[i] = expf(min_x + step * static_cast<float>(i));
                }
                inited = true;
            }

            if (x <= min_x) return lut[0];
            if (x >= max_x) return lut[XG_SOFTMAX_LUT_SIZE - 1];

            const float f = (x - min_x) / step;
            const int idx = static_cast<int>(f);
            const float frac = f - static_cast<float>(idx);
            const float a = lut[idx];
            const float b = lut[idx + 1];
            return a + (b - a) * frac;
        }
#endif

        /**
         * @brief Apply softmax transformation to score vector
         */
        void softmax(vector<float>& x) const {
            if (x.empty()) return;

            // Find max for numerical stability
            float max_val = x[0];
            for (size_t i = 1; i < x.size(); ++i) {
                if (x[i] > max_val) max_val = x[i];
            }

            // Compute exp and sum
            float sum = 0.0f;
            for (size_t i = 0; i < x.size(); ++i) {
#if defined(XG_SOFTMAX_USE_LUT)
                x[i] = exp_lut(x[i] - max_val);
#else
                x[i] = expf(x[i] - max_val);
#endif
                sum += x[i];
            }

            // Normalize
            if (sum > 0.0f) {
                float inv_sum = 1.0f / sum;
                for (size_t i = 0; i < x.size(); ++i) {
                    x[i] *= inv_sum;
                }
            }
        }
    };

    // ============================================================================
    // Utility functions
    // ============================================================================
    inline size_t estimate_memory(uint32_t num_trees, uint32_t avg_nodes_per_tree, uint16_t num_labels) {
        // Each node is 8 bytes (64-bit packed)
        size_t tree_memory = num_trees * avg_nodes_per_tree * sizeof(XG_node);
        
        // Score buffer
        size_t score_memory = num_labels * sizeof(float);
        
        // Overhead for tree container and config
        size_t overhead = sizeof(XGBoost) + num_trees * sizeof(XG_tree);
        
        return tree_memory + score_memory + overhead;
    }

} // namespace eml
