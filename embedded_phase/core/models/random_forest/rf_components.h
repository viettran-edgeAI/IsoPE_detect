#pragma once
#include "rf_base.h"

#include "../../containers/STL_MCU.h"
#include "../../../Rf_file_manager.h"
#include "../../base/eml_base.h"
#include "../../models/random_forest/rf_base.h"
#include "../../models/random_forest/rf_components.h"
#include "../../ml/eml_quantize.h"
#include "../../ml/eml_data.h"
#include "../../ml/eml_logger.h"
#include "../../ml/eml_metrics.h"

#if defined(ESP_PLATFORM)
    #include "esp_system.h"

    #if RF_BOARD_SUPPORTS_PSRAM
    #include <esp_psram.h>
    #endif
#endif
#include <cstdlib>
#include <cstring>
#include <limits>


namespace mcu {

    // Core dataset types are now provided by eml_data (framework core).
    using Rf_sample = eml_sample_t<problem_type::CLASSIFICATION>;
    using Rf_drift_sample = eml_drift_sample;
    using Rf_data = eml_data<problem_type::CLASSIFICATION>;


    inline size_t rf_max_dataset_size() {
        return rf_storage_max_dataset_bytes();
    }

    static constexpr rf_label_type RF_ERROR_LABEL = EML_ERROR_LABEL<problem_type::CLASSIFICATION>;


    /*
    ------------------------------------------------------------------------------------------------------------------
    ------------------------------------------------ RF_COMPONENTS ---------------------------------------------------
    ------------------------------------------------------------------------------------------------------------------
    */
    // struct Rf_sample;           // single data sample
    // struct Building_node;           // single tree node
    // ...

    // class Rf_data;              // dataset object
    class Rf_tree;              // decision tree
    // class Rf_config;            // forest configuration & dataset parameters
    class Rf_base;              // Manage and monitor the status of forest components and resources
    class Rf_node_predictor;    // estimate node per tree based on dataset & config.
    // class Rf_random;            // random generator (for stability across platforms and runs)
    class Rf_tree_container;    // manages all trees at forest level
    class Rf_pending_data;      // manage pending data waiting for true labels from feedback action



    /*
    ------------------------------------------------------------------------------------------------------------------
    ---------------------------------------------------- RF_CONFIG ---------------------------------------------------
    ------------------------------------------------------------------------------------------------------------------
    */

    // enum metric_score;      // flags for training process/score calculation (accuracy, precision, recall, f1_score)
    // enum Rf_training_score;     // score types for training process (oob, validation, k-fold)
    // ...


    typedef enum Rf_training_score : uint8_t {
        OOB_SCORE    = 0x00,   // default 
        VALID_SCORE  = 0x01,
        K_FOLD_SCORE = 0x02
    } Rf_training_score;

    // Configuration class : model configuration and dataset parameters
    // handle 2 files: model_name_rf_config.json (config file) and model_name_dp.csv (dp file)
    class Rf_config {
        const Rf_base* base_ptr = nullptr;

        bool has_base() const {
            return base_ptr != nullptr && base_ptr->ready_to_use();
        }
    public:
        bool isLoaded = false; 

        // Core model configuration
        uint8_t     num_trees;
        uint32_t    random_seed;
        uint8_t     min_split;
        uint8_t     min_leaf;
        uint8_t     max_depth;
        bool        use_boostrap;
        bool        use_gini;
        uint8_t     k_folds;
        float       boostrap_ratio; 
        float       impurity_threshold;
        float       train_ratio;
        float       test_ratio;
        float       valid_ratio;
        eval_metric metric_score;
        float       result_score;
        uint32_t    estimatedRAM;
        Rf_training_score training_score;

        bool enable_retrain;
        bool enable_auto_config;   // change config based on dataset parameters (when base_data expands)
        bool allow_new_labels;     // allow new labels to be added to the dataset (default: false)


        // runtime parameters
        pair<uint8_t, uint8_t> min_split_range;
        pair<uint8_t, uint8_t> min_leaf_range;
        pair<uint16_t, uint16_t> max_depth_range; 

        // Dataset parameters 
        sample_idx_type num_samples;
        uint16_t num_features;
        rf_label_type  num_labels;
        uint8_t  quantization_coefficient; // Bits per feature value (1-8)
        float lowest_distribution; 
        b_vector<sample_idx_type,8> samples_per_label; // index = label, value = count

        // MCU node layout bits (loaded from PC-trained model config)
        uint8_t threshold_bits = 0;
        uint8_t feature_bits = 0;
        uint8_t label_bits = 0;
        uint8_t child_bits = 0;

        void init(Rf_base* base) {
            base_ptr = base;
            isLoaded = false;

            // Set default values
            num_trees           = 20;
            random_seed         = 37;
            min_split           = 2;
            min_leaf            = 1;
            max_depth           = 250;
            use_boostrap        = true;
            boostrap_ratio      = 0.632f; 
            use_gini            = false;
            k_folds             = 4;
            impurity_threshold  = 0.0f;
            train_ratio         = 0.8f;
            test_ratio          = 0.0f;
            valid_ratio         = 0.0f;
            training_score      = OOB_SCORE;
            metric_score         = eval_metric::ACCURACY;
            result_score        = 0.0;
            estimatedRAM        = 0;
            enable_retrain      = true;
            enable_auto_config  = false;
            allow_new_labels    = false;
            quantization_coefficient = 2; 
        }
        
        Rf_config() {
            init(nullptr);
        }
        Rf_config(Rf_base* base) {
            init(base);
        }

        ~Rf_config() {
            releaseConfig();
            base_ptr = nullptr;
        }

    private:
        //  scan base_data file to get dataset parameters (when no dp file found)
        bool scan_base_data(){
            char base_file_path[RF_PATH_BUFFER];
            base_ptr->get_base_data_path(base_file_path);
            eml_debug(1, "📊 Scanning base data: ", base_file_path);

            File file = RF_FS_OPEN(base_file_path, RF_FILE_READ);
            if (!file) {
                eml_debug(0, "❌ Failed to open base data file for scanning: ", base_file_path);
                return false;
            }

            // Read binary header
            uint32_t numSamples;
            uint16_t numFeatures;
            
            if(file.read((uint8_t*)&numSamples, sizeof(numSamples)) != sizeof(numSamples) ||
               file.read((uint8_t*)&numFeatures, sizeof(numFeatures)) != sizeof(numFeatures)) {
                eml_debug(0, "❌ Failed to read dataset header during scan", base_file_path);
                file.close();
                return false;
            }

            // Set basic parameters
            num_samples = numSamples;
            num_features = numFeatures;

            // Calculate packed feature bytes per sample (using existing quantization_coefficient)
            uint32_t totalBits = static_cast<uint32_t>(numFeatures) * quantization_coefficient;
            const uint16_t packedFeatureBytes = (totalBits + 7) / 8; // Round up to nearest byte

            // Track unique labels and their counts
            unordered_map_s<rf_label_type, sample_idx_type> label_counts;
            rf_label_type max_label = 0;

            // Scan through all samples to collect label statistics
            for(sample_idx_type i = 0; i < numSamples; i++) {
                rf_label_type label;
                if(file.read(&label, sizeof(label)) != sizeof(label)) {
                    eml_debug_2(0, "❌ Failed to read label of sample", i, ": ", base_file_path);
                    file.close();
                    return false;
                }

                // Track label statistics
                auto it = label_counts.find(label);
                if(it != label_counts.end()) {
                    it->second++;
                } else {
                    label_counts[label] = 1;
                }

                if(label > max_label) {
                    max_label = label;
                }

                // Skip packed features for this sample
                if(file.seek(file.position() + packedFeatureBytes) == false) {
                    eml_debug_2(0, "❌ Failed to skip features of sample", i, ": ", base_file_path);
                    file.close();
                    return false;
                }
            }

            file.close();

            // Set number of labels
            num_labels = label_counts.size();

            // Initialize samples_per_label vector with proper size
            samples_per_label.clear();
            samples_per_label.resize(max_label + 1, 0);

            // Fill samples_per_label with counts
            for(auto& pair : label_counts) {
                samples_per_label[pair.first] = pair.second;
            }

            eml_debug(1, "✅ Base data scan complete.");
            eml_debug(1, "   📊 Samples: ", num_samples);
            eml_debug(1, "   🔢 Features: ", num_features);
            eml_debug(1, "   🏷️ Labels: ", num_labels);
            eml_debug(1, "   📈 Samples per label: ");
            for (size_t i = 0; i < samples_per_label.size(); i++) {
                if(samples_per_label[i] > 0) {
                    eml_debug_2(1, "   Lable ", i, ": ", samples_per_label[i]);
                }
            }
            return true;
        }
        
        // generate optimal ranges for min_split and min_leaf based on dataset parameters
        void generate_ranges(bool force = false) {
            int baseline_minsplit_ratio = 100 * (num_samples / 500 + 1);
            if (baseline_minsplit_ratio > 500) baseline_minsplit_ratio = 500;
            uint8_t min_minSplit = 2;

            int dynamic_max_split = min(min_minSplit + 6, (int)(log2(num_samples) / 4 + num_features / 25.0f));
            uint8_t max_minSplit = min<uint8_t>(16, dynamic_max_split);
            if (max_minSplit <= min_minSplit) {
                max_minSplit = static_cast<uint8_t>(min_minSplit + 4);
            }

            float samples_per_label = (num_labels > 0)
                                          ? static_cast<float>(num_samples) / static_cast<float>(num_labels)
                                          : static_cast<float>(num_samples);
            float density_factor = samples_per_label / 600.0f;
            if (density_factor < 0.3f) density_factor = 0.3f;
            if (density_factor > 3.0f) density_factor = 3.0f;

            float expected_min_pct = (num_labels > 0) ? (100.0f / static_cast<float>(num_labels)) : 100.0f;
            float deficit_pct = max(0.0f, expected_min_pct - lowest_distribution);
            float imbalance_ratio = (expected_min_pct > 0.0f) ? (deficit_pct / expected_min_pct) : 0.0f;
            if (imbalance_ratio > 0.5f) imbalance_ratio = 0.5f;
            float imbalance_factor = 1.0f - imbalance_ratio; // 0.5 .. 1.0

            float min_ratio = 0.12f + 0.05f * density_factor * imbalance_factor;
            if (min_ratio < 0.1f) min_ratio = 0.1f;
            if (min_ratio > 0.35f) min_ratio = 0.35f;

            float max_ratio = min_ratio + (0.12f + 0.04f * density_factor);
            float min_allowed = min_ratio + 0.1f;
            if (max_ratio < min_allowed) max_ratio = min_allowed;
            if (max_ratio > 0.6f) max_ratio = 0.6f;

            uint8_t max_cap = (max_minSplit > 1) ? static_cast<uint8_t>(max_minSplit - 1) : static_cast<uint8_t>(1);
            uint8_t min_minLeaf = static_cast<uint8_t>(floorf(static_cast<float>(min_minSplit) * min_ratio));
            if (min_minLeaf < 1) min_minLeaf = 1;
            if (min_minLeaf > max_cap) min_minLeaf = max_cap;

            uint8_t max_minLeaf = static_cast<uint8_t>(ceilf(static_cast<float>(max_minSplit) * max_ratio));
            if (max_minLeaf > max_cap) max_minLeaf = max_cap;
            if (max_minLeaf < min_minLeaf) {
                max_minLeaf = min_minLeaf;
            }

            int base_maxDepth = (int)(log2(num_samples) + log2(num_features)) + 1;
            uint16_t max_maxDepth = max(8, base_maxDepth);
            uint16_t min_maxDepth = max_maxDepth > 18 ? max_maxDepth - 6 : max_maxDepth > 12 ? max_maxDepth - 4 : max_maxDepth > 8 ? max_maxDepth - 2 : 4;

            if (min_split == 0 || force) {
                min_split = min_minSplit;
                eml_debug_2(1, "Setting minSplit to ", min_split, " (auto)", "");
            }
            if (min_leaf == 0 || force) {
                min_leaf = min_minLeaf;
                eml_debug_2(1, "Setting minLeaf to ", min_leaf, " (auto)", "");
            }

            if (max_depth == 0 || force) {
                max_depth = max_maxDepth;
                eml_debug_2(1, "Setting maxDepth to ", max_depth, " (auto)", "");
            } 
            
            eml_debug_2(1, "⚙️ Setting minSplit range: ", min_minSplit, "to ", max_minSplit);
            eml_debug_2(1, "⚙️ Setting minLeaf range: ", min_minLeaf, "to ", max_minLeaf);
            eml_debug_2(1, "⚙️ Setting maxDepth range: ", min_maxDepth, "to ", max_maxDepth);

            min_split_range = make_pair(min_minSplit, max_minSplit);
            min_leaf_range = make_pair(min_minLeaf, max_minLeaf);
            max_depth_range = make_pair(min_maxDepth, max_maxDepth);
        }

        void generate_impurity_threshold(){
            // find lowest distribution
            if (samples_per_label.size() == 0) {
                impurity_threshold = 0.0f;
                return;
            }
            int K = max(2, static_cast<int>(num_labels));
            float expected_min_pct = 100.0f / static_cast<float>(K);
            float deficit = max(0.0f, expected_min_pct - lowest_distribution);
            float imbalance = expected_min_pct > 0.0f ? min(1.0f, deficit / expected_min_pct) : 0.0f; // 0..1

            double log_samples = log2(max(2.0, static_cast<double>(num_samples)));
            double adjusted = max(0.0, log_samples - 10.0); // keep small datasets unaffected
            float sample_factor = static_cast<float>(1.0 / (1.0 + adjusted / 2.5));
            sample_factor = max(0.25f, min(1.15f, sample_factor));
            // Imbalance factor: reduce threshold for imbalanced data to allow splitting on rare classes
            float imbalance_factor = 1.0f - 0.5f * imbalance; // 0.5..1.0
            // Feature factor: with many features, weak splits are common; require slightly higher gain
            float feature_factor = 0.9f + 0.1f * min(1.0f, static_cast<float>(log2(max(2, static_cast<int>(num_features)))) / 8.0f);

            if (use_gini) {
                float max_gini = 1.0f - 1.0f / static_cast<float>(K);
                float base = 0.003f * max_gini; // very small base for Gini
                float thr = base * sample_factor * imbalance_factor * feature_factor;
                impurity_threshold = max(0.0003f, min(0.02f, thr));
            } else { // entropy
                float max_entropy = log2(static_cast<float>(K));
                float base = 0.02f * (max_entropy > 0.0f ? max_entropy : 1.0f); // larger than gini
                float thr = base * sample_factor * imbalance_factor * feature_factor;
                impurity_threshold = max(0.002f, min(0.2f, thr));
            }
            eml_debug(1, "⚙️ Setting impurity_threshold to ", impurity_threshold);
        }
        
        // setup config manually (when no config file)
        void auto_config(){
            // set metric_score based on dataset balance
            if(samples_per_label.size() > 0){
                sample_idx_type minorityCount = num_samples;
                sample_idx_type majorityCount = 0;

                for (auto& count : samples_per_label) {
                    if (count > majorityCount) {
                        majorityCount = count;
                    }
                    if (count < minorityCount) {
                        minorityCount = count;
                    }
                }

                float maxImbalanceRatio = 1/lowest_distribution * 100.0f; 

                if (maxImbalanceRatio > 10.0f) {
                    metric_score = eval_metric::RECALL;
                    eml_debug_2(1, "⚠️ Highly imbalanced dataset: ", maxImbalanceRatio, "Setting metric_score to RECALL.", "");
                } else if (maxImbalanceRatio > 3.0f) {
                    metric_score = eval_metric::F1_SCORE;
                    eml_debug_2(1, "⚠️ Moderately imbalanced dataset: ", maxImbalanceRatio, "Setting metric_score to F1_SCORE.", "");
                } else if (maxImbalanceRatio > 1.5f) {
                    metric_score = eval_metric::PRECISION;
                    eml_debug_2(1, "⚠️ Slightly imbalanced dataset: ", maxImbalanceRatio, "Setting metric_score to PRECISION.", "");
                } else {
                    metric_score = eval_metric::ACCURACY;
                    eml_debug_2(1, "✅ Balanced dataset (ratio: ", maxImbalanceRatio, "). Setting metric_score to ACCURACY.", "");
                }
            }

            sample_idx_type avg_samples_per_label = num_samples / max(1, static_cast<int>(num_labels));
        
            // set training_score method
            if (avg_samples_per_label < 200){
                training_score = K_FOLD_SCORE;
            }else if (avg_samples_per_label < 500){
                training_score = OOB_SCORE;
            }else{
                training_score = VALID_SCORE;
            }
            validate_ratios();
            generate_ranges(true); // force generate min_split, min_leaf, max_depth
            generate_impurity_threshold(); // no prior distribution info
        }
        
    public:
        // Load configuration from JSON file
        bool loadConfig() {
            if (isLoaded) return true;
            if (!has_base()) {
                eml_debug(0, "❌ Base pointer is null or base not ready", "load config");
                return false;
            }

            // 1. Try to load config file (which contains dataset parameters)
            bool config_loaded = false;
            String jsonString = "";
            if(base_ptr->config_file_exists()){
                char file_path[RF_PATH_BUFFER];
                base_ptr->get_config_path(file_path);
                File file = RF_FS_OPEN(file_path, RF_FILE_READ);
                if (file) {
                    jsonString = file.readString();
                    file.close();
                    parseJSONConfig(jsonString); // This should populate num_samples, etc.
                    config_loaded = true;
                } else {
                    eml_debug(1, "⚠️ Failed to open config file: ", file_path);
                }
            } else {
                 eml_debug(1, "⚠️ No config file found");
            }

            // 2. Validate/Fallback: Check if dataset params are loaded
            if (num_samples == 0) {
                 eml_debug(1, "⚠️ Dataset parameters not found in config, scanning base data...");
                 if (!scan_base_data()) {
                     eml_debug(1, "❌ Cannot load dataset parameters (config missing/old and scan failed)");
                     return false; 
                 }
                 enable_auto_config = true;
            }

            // 3. Process loaded data
            lowest_distribution = 100.0f;
            for(auto & count : samples_per_label) {
                if (count > 0) {
                    float pct = 100.0f * static_cast<float>(count) / static_cast<float>(num_samples);
                    if (pct < lowest_distribution) {
                        lowest_distribution = pct;
                    }
                }
            }
            
            if (config_loaded) {
                validate_ratios();
                generate_ranges();
            } else {
                enable_auto_config = true;
            }
            
            // Now decide loading strategy based on enable_auto_config
            if(enable_auto_config){
                eml_debug(1, "🔧 Auto-config enabled: generating settings from dataset parameters");
                auto_config();
            } 
            if constexpr (RF_DEBUG_LEVEL >1) print_config();
            isLoaded = true;
            return true;
        }
    
        // Save configuration to JSON file 
        bool releaseConfig() {
            if (!isLoaded || !has_base()){
                eml_debug(0, "❌ Save config failed: Config not loaded or base not ready");
                return false;
            }
            char file_path[RF_PATH_BUFFER];
            base_ptr->get_config_path(file_path);
            String existingTimestamp = "";
            String existingAuthor = "Viettran";
            
            if (RF_FS_EXISTS(file_path)) {
                File readFile = RF_FS_OPEN(file_path, RF_FILE_READ);
                if (readFile) {
                    String jsonContent = readFile.readString();
                    readFile.close();
                    existingTimestamp = extractStringValue(jsonContent, "timestamp");
                    existingAuthor = extractStringValue(jsonContent, "author");
                }
                RF_FS_REMOVE(file_path);
            }

            File file = RF_FS_OPEN(file_path, RF_FILE_WRITE);
            if (!file) {
                eml_debug(0, "❌ Failed to create config file: ", file_path);
                return false;
            }

            file.println("{");
            file.printf("  \"numTrees\": %d,\n", num_trees);
            file.printf("  \"randomSeed\": %d,\n", random_seed);
            
            file.printf("  \"quantization_coefficient\": %d,\n", quantization_coefficient);
            file.printf("  \"num_features\": %d,\n", num_features);
            file.printf("  \"num_samples\": %u,\n", num_samples);
            file.printf("  \"num_labels\": %d,\n", num_labels);
            for(size_t i=0; i<samples_per_label.size(); i++){
                if(samples_per_label[i] > 0)
                    file.printf("  \"samples_label_%u\": %u,\n", i, samples_per_label[i]);
            }
            
            file.printf("  \"train_ratio\": %.1f,\n", train_ratio);
            file.printf("  \"test_ratio\": %.2f,\n", test_ratio);
            file.printf("  \"valid_ratio\": %.2f,\n", valid_ratio);
            file.printf("  \"minSplit\": %d,\n", min_split);
            file.printf("  \"minLeaf\": %d,\n", min_leaf);
            file.printf("  \"maxDepth\": %d,\n", max_depth);
            file.printf("  \"useBootstrap\": %s,\n", use_boostrap ? "true" : "false");
            file.printf("  \"boostrapRatio\": %.3f,\n", boostrap_ratio);
            file.printf("  \"criterion\": \"%s\",\n", use_gini ? "gini" : "entropy");
            file.printf("  \"trainingScore\": \"%s\",\n", getTrainingScoreString(training_score).c_str());
            file.printf("  \"k_folds\": %d,\n", k_folds);
            file.printf("  \"impurityThreshold\": %.4f,\n", impurity_threshold);
            file.printf("  \"eval_metric\": \"%s\",\n", getEvalMetricStr(metric_score).c_str());
            file.printf("  \"resultScore\": %.4f,\n", result_score);
            file.printf("  \"threshold_bits\": %d,\n", threshold_bits);
            file.printf("  \"feature_bits\": %d,\n", feature_bits);
            file.printf("  \"label_bits\": %d,\n", label_bits);
            file.printf("  \"child_bits\": %d,\n", child_bits);
            file.printf("  \"enableRetrain\": %s,\n", enable_retrain ? "true" : "false");
            file.printf("  \"enableAutoConfig\": %s,\n", enable_auto_config ? "true" : "false");
            file.printf("  \"Estimated RAM (bytes)\": %d,\n", estimatedRAM);

            if (existingTimestamp.length() > 0) {
                file.printf("  \"timestamp\": \"%s\",\n", existingTimestamp.c_str());
            }
            if (existingAuthor.length() > 0) {
                file.printf("  \"author\": \"%s\"\n", existingAuthor.c_str());
            } else {
                file.seek(file.position() - 2); // Go back to remove ",\n"
                file.println("");
            }
            
            file.println("}");
            file.close();
            isLoaded = false;
            base_ptr->set_config_status(true);
            eml_debug(1, "✅ Configuration saved to: ", file_path);
            return true;
        }

        void purgeConfig() {
            isLoaded = false;
        }

    private:
        // Simple JSON parser for configuration
        void parseJSONConfig(const String& jsonStr) {
            // Use the actual keys from digit_data_rf_config.json
            num_trees = extractIntValue(jsonStr, "numTrees");              
            random_seed = extractIntValue(jsonStr, "randomSeed");

            // Dataset parameters (integrated from dp file)
            quantization_coefficient = extractIntValue(jsonStr, "quantization_coefficient");
            num_features = extractIntValue(jsonStr, "num_features");
            num_samples = extractIntValue(jsonStr, "num_samples");
            num_labels = extractIntValue(jsonStr, "num_labels");
            
            if (num_labels > 0) {
                samples_per_label.clear();
                samples_per_label.resize(num_labels, 0);
                for (int i = 0; i < num_labels; i++) {
                    String key = "samples_label_" + String(i);
                    samples_per_label[i] = extractIntValue(jsonStr, key);
                }
            }

            min_split = extractIntValue(jsonStr, "minSplit");             
            min_leaf = extractIntValue(jsonStr, "minLeaf");
            if (min_leaf == 0) {
                min_leaf = 1;
            }
            max_depth = extractIntValue(jsonStr, "maxDepth");            
            use_boostrap = extractBoolValue(jsonStr, "useBootstrap");     
            boostrap_ratio = extractFloatValue(jsonStr, "boostrapRatio"); 
            
            String criterion = extractStringValue(jsonStr, "criterion");
            use_gini = (criterion == "gini");  // true for "gini", false for "entropy"
            
            k_folds = extractIntValue(jsonStr, "k_folds");                  
            impurity_threshold = extractFloatValue(jsonStr, "impurityThreshold");
            train_ratio = extractFloatValue(jsonStr, "train_ratio");      
            test_ratio = extractFloatValue(jsonStr, "test_ratio");        
            valid_ratio = extractFloatValue(jsonStr, "valid_ratio");     
            training_score = parseTrainingScore(extractStringValue(jsonStr, "trainingScore")); 
            metric_score = parseEvalMetric(extractStringValue(jsonStr, "eval_metric"));
            enable_retrain = extractBoolValue(jsonStr, "enableRetrain");
            enable_auto_config = extractBoolValue(jsonStr, "enableAutoConfig");
            result_score = extractFloatValue(jsonStr, "resultScore");      
            estimatedRAM = extractIntValue(jsonStr, "Estimated RAM (bytes)");
            
            // Load layout bits if available (from PC-trained model)
            threshold_bits = extractIntValue(jsonStr, "threshold_bits");
            feature_bits = extractIntValue(jsonStr, "feature_bits");
            label_bits = extractIntValue(jsonStr, "label_bits");
            child_bits = extractIntValue(jsonStr, "child_bits");
            if(num_trees == 1){     // decision tree mode 
                use_boostrap = false;
                boostrap_ratio = 1.0f; // disable bootstrap for single tree
                if(training_score == OOB_SCORE){
                    training_score = VALID_SCORE; // use validation score for single tree
                }
            }
        }

        // Convert string to metric_score enum
        eval_metric parseEvalMetric(const String& str) {
            if (str == "ACCURACY" || str == "accuracy") return eval_metric::ACCURACY;
            if (str == "PRECISION" || str == "precision") return eval_metric::PRECISION;
            if (str == "RECALL" || str == "recall") return eval_metric::RECALL;
            if (str == "F1_SCORE" || str == "f1_score" || str == "f1") return eval_metric::F1_SCORE;
            if (str == "LOGLOSS" || str == "logloss") return eval_metric::LOGLOSS;
            if (str == "MLOGLOSS" || str == "mlogloss") return eval_metric::MLOGLOSS;
            return eval_metric::ACCURACY; // Default
        }

        // Convert metric_score enum to string
        String getEvalMetricStr(eval_metric m) const {
            switch(m) {
                case eval_metric::ACCURACY:  return "ACCURACY";
                case eval_metric::PRECISION: return "PRECISION";
                case eval_metric::RECALL:    return "RECALL";
                case eval_metric::F1_SCORE:  return "F1_SCORE";
                case eval_metric::LOGLOSS:   return "LOGLOSS";
                case eval_metric::MLOGLOSS:  return "MLOGLOSS";
                default: return "ACCURACY";
            }
        }

        // Convert string to Rf_training_score enum
        Rf_training_score parseTrainingScore(const String& scoreStr) {
            if (scoreStr == "oob_score") return OOB_SCORE;
            if (scoreStr == "valid_score") return VALID_SCORE;
            if (scoreStr == "k_fold_score") return K_FOLD_SCORE;
            return OOB_SCORE; 
        }

        // Convert Rf_training_score enum to string
        String getTrainingScoreString(Rf_training_score score) const {
            switch(score) {
                case OOB_SCORE: return "oob_score";
                case VALID_SCORE: return "valid_score";
                case K_FOLD_SCORE: return "k_fold_score";
                default: return "oob_score";
            }
        }

        uint32_t extractIntValue(const String& json, const String& key) {
            int keyIndex = json.indexOf("\"" + key + "\"");
            if (keyIndex == -1) return 0;
            
            int colonIndex = json.indexOf(": ", keyIndex);
            if (colonIndex == -1) return 0;
            
            int commaIndex = json.indexOf(",", colonIndex);
            if (commaIndex == -1) commaIndex = json.indexOf("}", colonIndex);
            
            String valueStr = json.substring(colonIndex + 1, commaIndex);
            valueStr.trim();
            return valueStr.toInt();
        }

        float extractFloatValue(const String& json, const String& key) {
            int keyIndex = json.indexOf("\"" + key + "\"");
            if (keyIndex == -1) return 0.0;
            
            int colonIndex = json.indexOf(": ", keyIndex);
            if (colonIndex == -1) return 0.0;
            
            int commaIndex = json.indexOf(",", colonIndex);
            if (commaIndex == -1) commaIndex = json.indexOf("}", colonIndex);
            
            String valueStr = json.substring(colonIndex + 1, commaIndex);
            valueStr.trim();
            return valueStr.toFloat();
        }

        bool extractBoolValue(const String& json, const String& key) {
            int keyIndex = json.indexOf("\"" + key + "\"");
            if (keyIndex == -1) return false;
            
            int colonIndex = json.indexOf(": ", keyIndex);
            if (colonIndex == -1) return false;
            
            int commaIndex = json.indexOf(",", colonIndex);
            if (commaIndex == -1) commaIndex = json.indexOf("}", colonIndex);
            
            String valueStr = json.substring(colonIndex + 1, commaIndex);
            valueStr.trim();
            return valueStr.indexOf("true") != -1;
        }

        String extractStringValue(const String& json, const String& key) {
            int keyIndex = json.indexOf("\"" + key + "\"");
            if (keyIndex == -1) return "";
            
            int colonIndex = json.indexOf(": ", keyIndex);
            if (colonIndex == -1) return "";
            
            int firstQuoteIndex = json.indexOf("\"", colonIndex);
            if (firstQuoteIndex == -1) return "";
            
            int secondQuoteIndex = json.indexOf("\"", firstQuoteIndex + 1);
            if (secondQuoteIndex == -1) return "";
            
            return json.substring(firstQuoteIndex + 1, secondQuoteIndex);
        }
        
        String getEvalMetricString() const {
            return getEvalMetricStr(metric_score);
        }
        
        public:
        bool use_validation() const {
            // return valid_ratio > 0.0f;
            return training_score == VALID_SCORE;
        }

        // Method to validate that samples_per_label data is consistent
        bool validateSamplesPerLabel() const {
            if (samples_per_label.size() != num_labels) {
                return false;
            }
            sample_idx_type totalSamples = 0;
            for (sample_idx_type count : samples_per_label) {
                totalSamples += count;
            }
            return totalSamples == num_samples;
        }
        
        // make sure train, test and valid ratios valid and optimal 
        void validate_ratios(){;
            sample_idx_type rarest_class = RF_MAX_SAMPLES;
            for(auto & count : samples_per_label){
                if (count < rarest_class){
                    rarest_class = count;
                }
            }
            if(enable_auto_config){
                if(rarest_class > 750){
                    train_ratio = 0.8f;
                    test_ratio = 0.1f;
                    valid_ratio = 0.1f;
                } 
                else if(rarest_class > 150){
                    train_ratio = 0.7f;
                    test_ratio = 0.15f;
                    valid_ratio = 0.15f;
                }else{
                    train_ratio = 0.6f;
                    test_ratio = 0.2f;
                    valid_ratio = 0.2f;
                }
            }
            if (training_score != VALID_SCORE){
                train_ratio += valid_ratio;
                valid_ratio = 0.0f;
            } else {
                if (valid_ratio < 0.1f){
                    if(rarest_class > 750)      valid_ratio = 0.1f;
                    else if(rarest_class > 150) valid_ratio = 0.15f;
                    else                        valid_ratio = 0.2f;
                    train_ratio -= valid_ratio;
                }
            }
            if (!ENABLE_TEST_DATA){
                train_ratio += test_ratio;
                test_ratio = 0.0f;
            }else{
                if (test_ratio < 0.1f){
                    if(rarest_class > 750)      test_ratio = 0.1f;
                    else if(rarest_class > 150) test_ratio = 0.15f;
                    else                        test_ratio = 0.2f;
                    train_ratio -= test_ratio;
                }
            }
            // ensure ratios sum to 1.0
            float total_ratio = train_ratio + test_ratio + valid_ratio;
            if (total_ratio > 1.0f) {
                train_ratio /= total_ratio;
                test_ratio /= total_ratio;
                valid_ratio /= total_ratio;
            }else{
                if(total_ratio < 1.0f){
                    train_ratio *= (1.0f / total_ratio);
                    test_ratio *= (1.0f / total_ratio);
                    valid_ratio *= (1.0f / total_ratio);
                }
            }
        }
          
        void print_config() const {
            eml_debug(1, "🛠️ Model configuration: ");
            eml_debug(1, "   - Trees: ", num_trees);
            eml_debug(1, "   - Random seed: ", random_seed);
            eml_debug(1, "   - max_depth: ", max_depth);
            eml_debug(1, "   - min_split: ", min_split);
            eml_debug(1, "   - min_leaf: ", min_leaf);
            eml_debug(1, "   - train_ratio: ", train_ratio);
            eml_debug(1, "   - test_ratio: ", test_ratio);
            eml_debug(1, "   - valid_ratio: ", valid_ratio);
            eml_debug(1, "   - use_bootstrap: ", use_boostrap ? "true" : "false");
            eml_debug(1, "   - bootstrap_ratio: ", boostrap_ratio);
            eml_debug(1, "   - criterion: ", use_gini ? "gini" : "entropy");
            eml_debug(1, "   - k_folds: ", k_folds);
            eml_debug(1, "   - impurity_threshold: ", impurity_threshold);
            eml_debug(1, "   - threshold_bits: ", threshold_bits);
            eml_debug(1, "   - feature_bits: ", feature_bits);
            eml_debug(1, "   - label_bits: ", label_bits);
            eml_debug(1, "   - child_bits: ", child_bits);
            eml_debug(1, "   - training_score: ", getTrainingScoreString(training_score).c_str());
            eml_debug(1, "   - metric_score: ", getEvalMetricStr(metric_score).c_str());
            eml_debug(1, "   - enable_retrain: ", enable_retrain ? "true" : "false");
            eml_debug(1, "   - enable_auto_config: ", enable_auto_config ? "true" : "false");

            eml_debug(1, "📊 Dataset Parameters: ");
            eml_debug(1, "   - Samples: ", num_samples);
            eml_debug(1, "   - Features: ", num_features);
            eml_debug(1, "   - Labels: ", num_labels);
            eml_debug(1, "   - Samples per label: ");

            for (size_t i = 0; i < samples_per_label.size(); i++) {
                if(samples_per_label[i] > 0) {
                    eml_debug_2(1, "   🏷️ Label ", i, ": ", samples_per_label[i]);
                }
            }
        }
        
        size_t memory_usage() const {
            size_t total = sizeof(Rf_config);
            total += 4;   
            total += samples_per_label.size() * sizeof(sample_idx_type); 
            return total;
        }
        
    };


    // enum metric_score;      // flags for training process/score calculation (accuracy, precision, recall, f1_score)
    // enum Rf_training_score;     // score types for training process (oob, validation, k-fold)
    // ...


    /*
    ------------------------------------------------------------------------------------------------------------------
    -------------------------------------------------- RF_DATA ------------------------------------------------------
    ------------------------------------------------------------------------------------------------------------------
    */

    
    /*
    ------------------------------------------------------------------------------------------------------------------
    ---------------------------------------------------- RF_TREE -----------------------------------------------------
    ------------------------------------------------------------------------------------------------------------------
    */

    // node_resource: holds dynamic bit-width configuration and provides layout accessors for
    // build-time Building_node and compact inference-time node formats.
    struct node_resource {
        uint8_t threshold_bits = 3;
        uint8_t feature_bits = 8;
        uint8_t label_bits = 3;
        uint8_t child_bits = 10;

        // Building_node layout: [is_leaf:1][threshold][feature][label][left_child]
        pair<uint8_t, uint8_t> build_threshold_layout;
        pair<uint8_t, uint8_t> build_featureID_layout;
        pair<uint8_t, uint8_t> build_label_layout;
        pair<uint8_t, uint8_t> build_left_child_layout;

        // Internal_node: [child_is_leaf:1][threshold][feature][left_child]
        pair<uint8_t, uint8_t> in_threshold_layout;
        pair<uint8_t, uint8_t> in_featureID_layout;
        pair<uint8_t, uint8_t> in_left_child_layout;

        // Mixed_node: [left_is_leaf:1][threshold][feature][left_child][right_child]
        pair<uint8_t, uint8_t> mx_threshold_layout;
        pair<uint8_t, uint8_t> mx_featureID_layout;
        pair<uint8_t, uint8_t> mx_left_child_layout;
        pair<uint8_t, uint8_t> mx_right_child_layout;

        void set_bits(uint8_t featureBits, uint8_t labelBits, uint8_t childBits, uint8_t thresholdBits) {
            // Clamp to supported ranges
            if (thresholdBits < 1) thresholdBits = 1;
            if (thresholdBits > 8) thresholdBits = 8;
            if (labelBits < 1) labelBits = 1;
            if (labelBits > 8) labelBits = 8;
            if (featureBits < 1) featureBits = 1;
            if (featureBits > 10) featureBits = 10;
            if (childBits < 1) childBits = 1;

            threshold_bits = thresholdBits;
            feature_bits = featureBits;
            label_bits = labelBits;
            child_bits = childBits;

            // Building_node layout (for training)
            build_threshold_layout = make_pair(static_cast<uint8_t>(1), threshold_bits);
            build_featureID_layout = make_pair(static_cast<uint8_t>(build_threshold_layout.first + build_threshold_layout.second), feature_bits);
            build_label_layout = make_pair(static_cast<uint8_t>(build_featureID_layout.first + build_featureID_layout.second), label_bits);
            build_left_child_layout = make_pair(static_cast<uint8_t>(build_label_layout.first + build_label_layout.second), child_bits);

            // Internal_node layout (compact inference)
            in_threshold_layout = make_pair(static_cast<uint8_t>(1), threshold_bits);
            in_featureID_layout = make_pair(static_cast<uint8_t>(in_threshold_layout.first + in_threshold_layout.second), feature_bits);
            in_left_child_layout = make_pair(static_cast<uint8_t>(in_featureID_layout.first + in_featureID_layout.second), child_bits);

            // Mixed_node layout (compact inference)
            mx_threshold_layout = make_pair(static_cast<uint8_t>(1), threshold_bits);
            mx_featureID_layout = make_pair(static_cast<uint8_t>(mx_threshold_layout.first + mx_threshold_layout.second), feature_bits);
            mx_left_child_layout = make_pair(static_cast<uint8_t>(mx_featureID_layout.first + mx_featureID_layout.second), child_bits);
            mx_right_child_layout = make_pair(static_cast<uint8_t>(mx_left_child_layout.first + mx_left_child_layout.second), child_bits);
        }

        // Building_node layout accessors
        inline pair<uint8_t, uint8_t> get_Building_node_threshold_layout() const { return build_threshold_layout; }
        inline pair<uint8_t, uint8_t> get_Building_node_featureID_layout() const { return build_featureID_layout; }
        inline pair<uint8_t, uint8_t> get_Building_node_label_layout() const { return build_label_layout; }
        inline pair<uint8_t, uint8_t> get_Building_node_left_child_layout() const { return build_left_child_layout; }

        // Internal_node layout accessors
        inline pair<uint8_t, uint8_t> get_Internal_node_threshold_layout() const { return in_threshold_layout; }
        inline pair<uint8_t, uint8_t> get_Internal_node_featureID_layout() const { return in_featureID_layout; }
        inline pair<uint8_t, uint8_t> get_Internal_node_left_child_layout() const { return in_left_child_layout; }

        // Mixed_node layout accessors
        inline pair<uint8_t, uint8_t> get_Mixed_node_threshold_layout() const { return mx_threshold_layout; }
        inline pair<uint8_t, uint8_t> get_Mixed_node_featureID_layout() const { return mx_featureID_layout; }
        inline pair<uint8_t, uint8_t> get_Mixed_node_left_child_layout() const { return mx_left_child_layout; }
        inline pair<uint8_t, uint8_t> get_Mixed_node_right_child_layout() const { return mx_right_child_layout; }

        // Bits-per-node calculations
        inline uint8_t bits_per_internal_node() const {
            return static_cast<uint8_t>(1 + threshold_bits + feature_bits + child_bits);
        }

        inline uint8_t bits_per_mixed_node() const {
            return static_cast<uint8_t>(1 + threshold_bits + feature_bits + child_bits + child_bits);
        }

        inline uint8_t bits_per_leaf_node() const {
            return label_bits;
        }

        inline uint8_t bits_per_building_node() const {
            return static_cast<uint8_t>(1 + threshold_bits + feature_bits + label_bits + child_bits);
        }

        // Utility methods (formerly in node_layout)
        inline uint16_t max_features() const {
            uint16_t mf = 1 << feature_bits;
            return mf > RF_MAX_FEATURES ? mf : RF_MAX_FEATURES;
        }

        inline uint8_t max_labels() const {
            return (1 << label_bits);
        }

        inline node_idx_type max_nodes() const {
            if (child_bits >= 64) return RF_MAX_NODES;
            if (child_bits >= 32) return RF_MAX_NODES;
            uint32_t mn = 1u << child_bits;
            return mn > RF_MAX_NODES ? mn : RF_MAX_NODES;
        }
    };

    struct Internal_node {
        size_t packed_data;
        Internal_node() : packed_data(0) {}

        inline bool childrenAreLeaf() const {
            return (packed_data >> 0) & 0x01;
        }

        inline uint16_t getThresholdSlot(const node_resource& res) const noexcept {
            const auto& layout = res.in_threshold_layout;
            if (layout.second == 0) {
                return 0;
            }
            const uint16_t mask = static_cast<uint16_t>((1u << layout.second) - 1u);
            return static_cast<uint16_t>((packed_data >> layout.first) & mask);
        }

        inline uint16_t getFeatureID(const node_resource& res) const noexcept {
            const auto& layout = res.in_featureID_layout;
            return (packed_data >> layout.first) & ((1u << layout.second) - 1u);
        }

        inline node_idx_type getLeftChildIndex(const node_resource& res) const noexcept {
            const auto& layout = res.in_left_child_layout;
            return (packed_data >> layout.first) & ((static_cast<size_t>(1) << layout.second) - 1u);
        }

        inline node_idx_type getRightChildIndex(const node_resource& res) const noexcept {
            (void)res;
            // Breadth-first property: right = left + 1 when both children are in the same vector space
            return getLeftChildIndex(res) + 1;
        }

        inline void setChildrenAreLeaf(bool v) {
            packed_data &= ~(static_cast<size_t>(0x01u));
            packed_data |= (v ? 1u : 0u) << 0;
        }

        inline void setThresholdSlot(uint16_t slot, const node_resource& res) noexcept {
            const auto& layout = res.in_threshold_layout;
            if (layout.second == 0) {
                return;
            }
            const size_t mask = (((static_cast<size_t>(1) << layout.second) - 1u) << layout.first);
            packed_data &= ~mask;
            packed_data |= (static_cast<size_t>(slot) & ((static_cast<size_t>(1) << layout.second) - 1u)) << layout.first;
        }

        inline void setFeatureID(uint16_t featureID, const node_resource& res) noexcept {
            const auto& layout = res.in_featureID_layout;
            const size_t mask = (((static_cast<size_t>(1) << layout.second) - 1u) << layout.first);
            packed_data &= ~mask;
            packed_data |= (static_cast<size_t>(featureID) & ((static_cast<size_t>(1) << layout.second) - 1u)) << layout.first;
        }

        inline void setLeftChildIndex(node_idx_type index, const node_resource& res) noexcept {
            const auto& layout = res.in_left_child_layout;
            const size_t mask = (((static_cast<size_t>(1) << layout.second) - 1u) << layout.first);
            packed_data &= ~mask;
            packed_data |= (static_cast<size_t>(index) & ((static_cast<size_t>(1) << layout.second) - 1u)) << layout.first;
        }
    };

    struct Mixed_node {
        size_t packed_data;
        Mixed_node() : packed_data(0) {}

        inline bool leftIsLeaf() const {
            return (packed_data >> 0) & 0x01;
        }

        inline uint16_t getThresholdSlot(const node_resource& res) const noexcept {
            const auto& layout = res.mx_threshold_layout;
            if (layout.second == 0) {
                return 0;
            }
            const uint16_t mask = static_cast<uint16_t>((1u << layout.second) - 1u);
            return static_cast<uint16_t>((packed_data >> layout.first) & mask);
        }

        inline uint16_t getFeatureID(const node_resource& res) const noexcept {
            const auto& layout = res.mx_featureID_layout;
            return (packed_data >> layout.first) & ((1u << layout.second) - 1u);
        }

        inline node_idx_type getLeftChildIndex(const node_resource& res) const noexcept {
            const auto& layout = res.mx_left_child_layout;
            return (packed_data >> layout.first) & ((static_cast<size_t>(1) << layout.second) - 1u);
        }

        inline node_idx_type getRightChildIndex(const node_resource& res) const noexcept {
            const auto& layout = res.mx_right_child_layout;
            return (packed_data >> layout.first) & ((static_cast<size_t>(1) << layout.second) - 1u);
        }

        inline void setLeftIsLeaf(bool v) {
            packed_data &= ~(static_cast<size_t>(0x01u));
            packed_data |= (v ? 1u : 0u) << 0;
        }

        inline void setThresholdSlot(uint16_t slot, const node_resource& res) noexcept {
            const auto& layout = res.mx_threshold_layout;
            if (layout.second == 0) {
                return;
            }
            const size_t mask = (((static_cast<size_t>(1) << layout.second) - 1u) << layout.first);
            packed_data &= ~mask;
            packed_data |= (static_cast<size_t>(slot) & ((static_cast<size_t>(1) << layout.second) - 1u)) << layout.first;
        }

        inline void setFeatureID(uint16_t featureID, const node_resource& res) noexcept {
            const auto& layout = res.mx_featureID_layout;
            const size_t mask = (((static_cast<size_t>(1) << layout.second) - 1u) << layout.first);
            packed_data &= ~mask;
            packed_data |= (static_cast<size_t>(featureID) & ((static_cast<size_t>(1) << layout.second) - 1u)) << layout.first;
        }

        inline void setLeftChildIndex(node_idx_type index, const node_resource& res) noexcept {
            const auto& layout = res.mx_left_child_layout;
            const size_t mask = (((static_cast<size_t>(1) << layout.second) - 1u) << layout.first);
            packed_data &= ~mask;
            packed_data |= (static_cast<size_t>(index) & ((static_cast<size_t>(1) << layout.second) - 1u)) << layout.first;
        }

        inline void setRightChildIndex(node_idx_type index, const node_resource& res) noexcept {
            const auto& layout = res.mx_right_child_layout;
            const size_t mask = (((static_cast<size_t>(1) << layout.second) - 1u) << layout.first);
            packed_data &= ~mask;
            packed_data |= (static_cast<size_t>(index) & ((static_cast<size_t>(1) << layout.second) - 1u)) << layout.first;
        }
    };

    struct Leaf_node {
        rf_label_type label;
        Leaf_node() : label(0) {}
    };

    struct Building_node{
        size_t packed_data; 

        Building_node() : packed_data(0) {}

        inline bool getIsLeaf() const {
            return (packed_data >> 0) & 0x01;  // Bit 0
        }
        inline uint16_t getThresholdSlot(const pair<uint8_t, uint8_t>& layout) const noexcept {
            if (layout.second == 0) {
                return 0;
            }
            const uint16_t mask = static_cast<uint16_t>((1u << layout.second) - 1u);
            return static_cast<uint16_t>((packed_data >> layout.first) & mask);
        }
        inline uint16_t getFeatureID(const pair<uint8_t, uint8_t>& layout) const noexcept{
            return (packed_data >> layout.first) & ((1u << layout.second) - 1u);  
        }
        
        inline rf_label_type getLabel(const pair<uint8_t, uint8_t>& layout) const noexcept{
            return (packed_data >> layout.first) & ((1u << layout.second) - 1u);  
        }
        inline node_idx_type getLeftChildIndex(const pair<uint8_t, uint8_t>& layout) const noexcept{
            return (packed_data >> layout.first) & ((1u << layout.second) - 1u);  
        }
        
        inline node_idx_type getRightChildIndex(const pair<uint8_t, uint8_t>& layout) const noexcept{
            return getLeftChildIndex(layout) + 1;  // Breadth-first property: right = left + 1
        }
        
        // Setter methods for packed data
        inline void setIsLeaf(bool isLeaf) {
            packed_data &= ~(0x01u); // Clear bit 0
            packed_data |= (isLeaf ? 1u : 0u) << 0; // Set bit 0
        }
        inline void setThresholdSlot(uint16_t slot, const pair<uint8_t, uint8_t>& layout) noexcept {
            if (layout.second == 0) {
                return;
            }
            const uint32_t mask = (((1u << layout.second) - 1u) << layout.first);
            packed_data &= ~mask;
            packed_data |= (static_cast<uint32_t>(slot) & ((1u << layout.second) - 1u)) << layout.first;
        }
        inline void setFeatureID(uint16_t featureID, const pair<uint8_t, uint8_t>& layout) noexcept{
            const uint32_t mask = (((1u << layout.second) - 1u) << layout.first);
            packed_data &= ~mask; // Clear featureID bits
            packed_data |= (featureID & ((1u << layout.second) - 1u)) << layout.first; // Set featureID bits
        }
        inline void setLabel(rf_label_type label, const pair<uint8_t, uint8_t>& layout) noexcept{
            const uint32_t mask = (((1u << layout.second) - 1u) << layout.first);
            packed_data &= ~mask; // Clear label bits
            packed_data |= (label & ((1u << layout.second) - 1u)) << layout.first; // Set label bits
        }
        inline void setLeftChildIndex(node_idx_type index, const pair<uint8_t, uint8_t>& layout) noexcept{
            const uint32_t mask = (((1u << layout.second) - 1u) << layout.first);
            packed_data &= ~mask; // Clear left child index bits
            packed_data |= (index & ((1u << layout.second) - 1u)) << layout.first; // Set left child index bits
        }
    };

    class Rf_tree {
        static constexpr uint8_t bits_per_node = sizeof(size_t) * 8;
    public:
        // Build-time representation (Building_node, breadth-first).
        packed_vector<bits_per_node, Building_node> nodes;

        // Compact inference-time representation.
        packed_vector<bits_per_node, Internal_node> internal_nodes;
        packed_vector<bits_per_node, Mixed_node> mixed_nodes;
        packed_vector<8, rf_label_type> leaf_nodes;
        packed_vector<1, uint8_t> branch_kind; // bpv=1; 0=internal, 1=mixed in branch-index space

        // Prefix sums over branch_kind words to map branch index -> internal/mixed local index in O(1)
        b_vector<uint16_t, 32> mixed_prefix;

        node_resource* resource = nullptr;      // Node layouts and bit widths

        // Root reference in compact form
        bool root_is_leaf = false;
        node_idx_type root_index = 0; // leaf index if root_is_leaf, else branch index

        uint16_t depth;
        uint8_t index;
        bool isLoaded;
        ID_vector<sample_idx_type, 3> bootstrapIDs;

        Rf_tree() : nodes(), internal_nodes(), mixed_nodes(), leaf_nodes(), branch_kind(), mixed_prefix(), resource(nullptr), index(255), isLoaded(false) {}

        explicit Rf_tree(uint8_t idx) : nodes(), internal_nodes(), mixed_nodes(), leaf_nodes(), branch_kind(), mixed_prefix(), resource(nullptr), index(idx), isLoaded(false) {}

        Rf_tree(const Rf_tree& other) : nodes(other.nodes),
            internal_nodes(other.internal_nodes),
            mixed_nodes(other.mixed_nodes),
            leaf_nodes(other.leaf_nodes),
            branch_kind(other.branch_kind),
            mixed_prefix(other.mixed_prefix),
            resource(other.resource),
            root_is_leaf(other.root_is_leaf),
            root_index(other.root_index),
            depth(other.depth),
            index(other.index),
            isLoaded(other.isLoaded),
            bootstrapIDs(other.bootstrapIDs) {}

        Rf_tree& operator=(const Rf_tree& other) {
            if (this != &other) {
                nodes = other.nodes;
                internal_nodes = other.internal_nodes;
                mixed_nodes = other.mixed_nodes;
                leaf_nodes = other.leaf_nodes;
                branch_kind = other.branch_kind;
                mixed_prefix = other.mixed_prefix;
                resource = other.resource;
                index = other.index;
                isLoaded = other.isLoaded;
                root_is_leaf = other.root_is_leaf;
                root_index = other.root_index;
                depth = other.depth;
                bootstrapIDs = other.bootstrapIDs;
            }
            return *this;
        }

        Rf_tree(Rf_tree&& other) noexcept
                : nodes(std::move(other.nodes)),
                    internal_nodes(std::move(other.internal_nodes)),
                    mixed_nodes(std::move(other.mixed_nodes)),
                    leaf_nodes(std::move(other.leaf_nodes)),
                    branch_kind(std::move(other.branch_kind)),
                    mixed_prefix(std::move(other.mixed_prefix)),
                    resource(other.resource),
                    root_is_leaf(other.root_is_leaf),
                    root_index(other.root_index),
                    depth(other.depth),
                    index(other.index),
                    isLoaded(other.isLoaded),
                    bootstrapIDs(std::move(other.bootstrapIDs)) {
            other.resource = nullptr;
            other.index = 255;
            other.isLoaded = false;
        }

        Rf_tree& operator=(Rf_tree&& other) noexcept {
            if (this != &other) {
                nodes = std::move(other.nodes);
                internal_nodes = std::move(other.internal_nodes);
                mixed_nodes = std::move(other.mixed_nodes);
                leaf_nodes = std::move(other.leaf_nodes);
                branch_kind = std::move(other.branch_kind);
                mixed_prefix = std::move(other.mixed_prefix);
                resource = other.resource;
                index = other.index;
                isLoaded = other.isLoaded;
                root_is_leaf = other.root_is_leaf;
                root_index = other.root_index;
                depth = other.depth;
                bootstrapIDs = std::move(other.bootstrapIDs);
                other.resource = nullptr;
                other.index = 255;
                other.isLoaded = false;
            }
            return *this;
        }

        void set_resource(node_resource* res_ptr, bool reset_storage = false) {
            resource = res_ptr;
            if (reset_storage) {
                reset_node_storage();
            }
        }

        void reset_node_storage(size_t reserveCount = 0) {
            const uint8_t desired = desired_bits_per_node();
            if (nodes.get_bits_per_value() != desired) {
                nodes.set_bits_per_value(desired);
            } else {
                nodes.clear();
            }
            if (reserveCount > 0) {
                nodes.reserve(reserveCount);
            }

            // Pre-allocate compact buffers too (heuristics: ~half leaves, ~half branch; mixed ~2%).
            if (resource) {
                const uint8_t inBits = resource->bits_per_internal_node();
                const uint8_t mxBits = resource->bits_per_mixed_node();
                const uint8_t lfBits = resource->bits_per_leaf_node();
                if (internal_nodes.get_bits_per_value() != inBits) {
                    internal_nodes.set_bits_per_value(inBits);
                } else {
                    internal_nodes.clear();
                }
                if (mixed_nodes.get_bits_per_value() != mxBits) {
                    mixed_nodes.set_bits_per_value(mxBits);
                } else {
                    mixed_nodes.clear();
                }
                if (leaf_nodes.get_bits_per_value() != lfBits) {
                    leaf_nodes.set_bits_per_value(lfBits);
                } else {
                    leaf_nodes.clear();
                }
                // branch_kind.set_bits_per_value(1);
                branch_kind.clear();
                mixed_prefix.clear();

                if (reserveCount > 0) {
                    const size_t half = reserveCount / 2;
                    internal_nodes.reserve(half);
                    leaf_nodes.reserve(reserveCount - half);
                    const size_t mx = (reserveCount > 50) ? (reserveCount * 2 / 100) : 1;
                    mixed_nodes.reserve(mx);
                    branch_kind.reserve(half + (reserveCount - half));
                }
            }
        }

        node_idx_type countNodes() const {
            // Prefer compact representation if present.
            const size_t compact_total = internal_nodes.size() + mixed_nodes.size() + leaf_nodes.size();
            if (compact_total > 0) {
                return static_cast<node_idx_type>(compact_total);
            }
            return static_cast<node_idx_type>(nodes.size());
        }

        size_t memory_usage() const {
            return nodes.memory_usage() + sizeof(*this);
        }

        node_idx_type countLeafNodes() const {
            if (leaf_nodes.size() > 0) {
                return static_cast<node_idx_type>(leaf_nodes.size());
            }
            node_idx_type leafCount = 0;
            for (size_t i = 0; i < nodes.size(); ++i) {
                if (nodes.get(i).getIsLeaf()) {
                    ++leafCount;
                }
            }
            return leafCount;
        }

        uint16_t getTreeDepth() const {
            return depth;
        }

        // Convert build-time Building_node storage into compact storage.
        // After successful conversion, build nodes are cleared to free RAM.
        bool convert_to_compact() {
            if (!resource) {
                return false;
            }
            if (nodes.empty()) {
                return false;
            }

            // Ensure compact buffers are reset (do not assume prior reserve state)
            internal_nodes.clear();
            mixed_nodes.clear();
            leaf_nodes.clear();
            branch_kind.clear();
            mixed_prefix.clear();

            internal_nodes.set_bits_per_value(resource->bits_per_internal_node());
            mixed_nodes.set_bits_per_value(resource->bits_per_mixed_node());
            leaf_nodes.set_bits_per_value(resource->bits_per_leaf_node());
            branch_kind.set_bits_per_value(1);

            const auto& featureLayout = resource->get_Building_node_featureID_layout();
            const auto& labelLayout = resource->get_Building_node_label_layout();
            const auto& childLayout = resource->get_Building_node_left_child_layout();
            const auto& thresholdLayout = resource->get_Building_node_threshold_layout();

            const node_idx_type nodeCount = static_cast<node_idx_type>(nodes.size());
            if (nodeCount == 0) {
                return false;
            }

            // Map old indices -> leaf index / branch index (branch index is in old-order filtering)
            vector<node_idx_type> old_to_leaf;
            vector<node_idx_type> old_to_branch;
            old_to_leaf.resize(nodeCount, static_cast<node_idx_type>(0xFFFFFFFFu));
            old_to_branch.resize(nodeCount, static_cast<node_idx_type>(0xFFFFFFFFu));

            node_idx_type branchCount = 0;
            for (node_idx_type i = 0; i < nodeCount; ++i) {
                const Building_node n = nodes.get(i);
                if (n.getIsLeaf()) {
                    old_to_leaf[i] = static_cast<node_idx_type>(leaf_nodes.size());
                    leaf_nodes.push_back(n.getLabel(labelLayout));
                } else {
                    old_to_branch[i] = branchCount;
                    branchCount++;
                }
            }

            // Root
            const Building_node root = nodes.get(0);
            root_is_leaf = root.getIsLeaf();
            root_index = root_is_leaf ? old_to_leaf[0] : old_to_branch[0];

            // Build branch nodes in old-order filtering; branch_kind indicates which compact vector to use.
            for (node_idx_type i = 0; i < nodeCount; ++i) {
                const Building_node n = nodes.get(i);
                if (n.getIsLeaf()) {
                    continue;
                }
                const node_idx_type bidx = old_to_branch[i];
                const node_idx_type left_old = n.getLeftChildIndex(childLayout);
                const node_idx_type right_old = static_cast<node_idx_type>(left_old + 1);
                if (left_old >= nodeCount || right_old >= nodeCount) {
                    return false;
                }
                const Building_node left_n = nodes.get(left_old);
                const Building_node right_n = nodes.get(right_old);
                const bool left_leaf = left_n.getIsLeaf();
                const bool right_leaf = right_n.getIsLeaf();

                const uint16_t featureID = n.getFeatureID(featureLayout);
                const uint16_t threshold = n.getThresholdSlot(thresholdLayout);

                if (left_leaf == right_leaf) {
                    Internal_node inode;
                    inode.setChildrenAreLeaf(left_leaf);
                    inode.setThresholdSlot(threshold, *resource);
                    inode.setFeatureID(featureID, *resource);
                    const node_idx_type left_new = left_leaf ? old_to_leaf[left_old] : old_to_branch[left_old];
                    inode.setLeftChildIndex(left_new, *resource);

                    // Append internal node and mark kind=0
                    (void)bidx; // bidx used in branch_kind ordering only
                    internal_nodes.push_back(inode);
                    branch_kind.push_back(0);
                } else {
                    Mixed_node mnode;
                    mnode.setLeftIsLeaf(left_leaf);
                    mnode.setThresholdSlot(threshold, *resource);
                    mnode.setFeatureID(featureID, *resource);
                    const node_idx_type left_new = left_leaf ? old_to_leaf[left_old] : old_to_branch[left_old];
                    const node_idx_type right_new = right_leaf ? old_to_leaf[right_old] : old_to_branch[right_old];
                    mnode.setLeftChildIndex(left_new, *resource);
                    mnode.setRightChildIndex(right_new, *resource);

                    mixed_nodes.push_back(mnode);
                    branch_kind.push_back(1);
                }
            }

            // Build prefix sums for rank mapping
            build_mixed_prefix();

            // Drop build nodes to reclaim RAM
            nodes.clear();
            nodes.shrink_to_fit();

            return true;
        }

        // Rebuild auxiliary indices (rank prefix) after loading compact data.
        void rebuild_compact_index() {
            if (branch_kind.size() > 0) {
                build_mixed_prefix();
            } else {
                mixed_prefix.clear();
            }
        }


        bool releaseTree(const char* path, bool re_use = false) {
            if (!re_use) {
                if (index > RF_MAX_TREES || nodes.empty()) {
                    // In compact mode nodes may be empty; allow save if compact buffers exist.
                    const bool hasCompact = (internal_nodes.size() + mixed_nodes.size() + leaf_nodes.size()) > 0;
                    if (!hasCompact) {
                        eml_debug(0, "❌ save tree failed, invalid tree index: ", index);
                        return false;
                    }
                }
                if (path == nullptr || strlen(path) == 0) {
                    eml_debug(0, "❌ save tree failed, invalid path: ", path);
                    return false;
                }
                if (RF_FS_EXISTS(path)) {
                    if (!RF_FS_REMOVE(path)) {
                        eml_debug(0, "❌ Failed to remove existing tree file: ", path);
                        return false;
                    }
                }
                File file = RF_FS_OPEN(path, FILE_WRITE);
                if (!file) {
                    eml_debug(0, "❌ Failed to open tree file for writing: ", path);
                    return false;
                }

                // Prefer compact format; if build nodes exist, convert first.
                if ((internal_nodes.size() + mixed_nodes.size() + leaf_nodes.size()) == 0) {
                    (void)convert_to_compact();
                }
                if (!resource) {
                    eml_debug(0, "❌ save tree failed: node_resource not set");
                    file.close();
                    return false;
                }

                // Compact tree format: TRC3 (portable: fixed-width counters + byte-packed branch_kind)
                const uint32_t magic = 0x33524354; // 'T''R''C''3'
                file.write(reinterpret_cast<const uint8_t*>(&magic), sizeof(magic));

                const uint8_t version = 3;
                file.write(reinterpret_cast<const uint8_t*>(&version), sizeof(version));

                auto write_u32 = [&](uint32_t v) {
                    file.write(reinterpret_cast<const uint8_t*>(&v), sizeof(v));
                };
                auto write_le = [&](uint64_t v, uint8_t bytes) {
                    for (uint8_t b = 0; b < bytes; ++b) {
                        const uint8_t byte = static_cast<uint8_t>((v >> (8u * b)) & 0xFFu);
                        file.write(reinterpret_cast<const uint8_t*>(&byte), 1);
                    }
                };

                // Persist bit widths for robustness
                file.write(reinterpret_cast<const uint8_t*>(&resource->threshold_bits), sizeof(uint8_t));
                file.write(reinterpret_cast<const uint8_t*>(&resource->feature_bits), sizeof(uint8_t));
                file.write(reinterpret_cast<const uint8_t*>(&resource->label_bits), sizeof(uint8_t));
                file.write(reinterpret_cast<const uint8_t*>(&resource->child_bits), sizeof(uint8_t));

                const uint8_t rootLeaf = root_is_leaf ? 1 : 0;
                file.write(reinterpret_cast<const uint8_t*>(&rootLeaf), sizeof(rootLeaf));
                write_u32(static_cast<uint32_t>(root_index));

                const uint32_t branchCount = static_cast<uint32_t>(branch_kind.size());
                const uint32_t internalCount = static_cast<uint32_t>(internal_nodes.size());
                const uint32_t mixedCount = static_cast<uint32_t>(mixed_nodes.size());
                const uint32_t leafCount = static_cast<uint32_t>(leaf_nodes.size());
                write_u32(branchCount);
                write_u32(internalCount);
                write_u32(mixedCount);
                write_u32(leafCount);

                // Write packed payloads as raw little-endian bytes per element
                const uint8_t inBits = internal_nodes.get_bits_per_value();
                const uint8_t mxBits = mixed_nodes.get_bits_per_value();
                const uint8_t lfBits = leaf_nodes.get_bits_per_value();
                const uint8_t inBytes = static_cast<uint8_t>((inBits + 7) / 8);
                const uint8_t mxBytes = static_cast<uint8_t>((mxBits + 7) / 8);
                const uint8_t lfBytes = static_cast<uint8_t>((lfBits + 7) / 8);
                file.write(reinterpret_cast<const uint8_t*>(&inBits), sizeof(inBits));
                file.write(reinterpret_cast<const uint8_t*>(&mxBits), sizeof(mxBits));
                file.write(reinterpret_cast<const uint8_t*>(&lfBits), sizeof(lfBits));

                // branch_kind bits (bpv=1) as raw bytes (portable)
                const uint32_t kindBytes = (branchCount + 7u) / 8u;
                write_u32(kindBytes);
                for (uint32_t byteIndex = 0; byteIndex < kindBytes; ++byteIndex) {
                    uint8_t out = 0;
                    const uint32_t base = byteIndex * 8u;
                    for (uint8_t bit = 0; bit < 8; ++bit) {
                        const uint32_t i = base + static_cast<uint32_t>(bit);
                        if (i < branchCount) {
                            out |= (static_cast<uint8_t>(branch_kind.get(i) & 1u) << bit);
                        }
                    }
                    file.write(reinterpret_cast<const uint8_t*>(&out), 1);
                }

                // Internal nodes
                for (uint32_t i = 0; i < internalCount; ++i) {
                    const Internal_node n = internal_nodes.get(static_cast<node_idx_type>(i));
                    write_le(static_cast<uint64_t>(n.packed_data), inBytes);
                }

                // Mixed nodes
                for (uint32_t i = 0; i < mixedCount; ++i) {
                    const Mixed_node n = mixed_nodes.get(static_cast<node_idx_type>(i));
                    write_le(static_cast<uint64_t>(n.packed_data), mxBytes);
                }

                // Leaf nodes (labels)
                for (uint32_t i = 0; i < leafCount; ++i) {
                    const rf_label_type lbl = leaf_nodes.get(static_cast<node_idx_type>(i));
                    write_le(static_cast<uint64_t>(lbl), lfBytes);
                }
                file.close();
            }
            nodes.clear();
            nodes.shrink_to_fit();
            internal_nodes.clear();
            internal_nodes.shrink_to_fit();
            mixed_nodes.clear();
            mixed_nodes.shrink_to_fit();
            leaf_nodes.clear();
            leaf_nodes.shrink_to_fit();
            branch_kind.clear();
            branch_kind.shrink_to_fit();
            mixed_prefix.clear();
            mixed_prefix.shrink_to_fit();
            isLoaded = false;
            eml_debug(2, "✅ Tree saved to file system: ", index);
            return true;
        }

        bool loadTree(const char* path, bool re_use = false) {
            if (isLoaded) {
                return true;
            }

            if (index >= RF_MAX_TREES) {
                eml_debug(0, "❌ Invalid tree index: ", index);
                return false;
            }
            if (path == nullptr || strlen(path) == 0) {
                eml_debug(0, "❌ Invalid path for loading tree: ", path);
                return false;
            }
            if (!RF_FS_EXISTS(path)) {
                eml_debug(0, "❌ Tree file does not exist: ", path);
                return false;
            }
            File file = RF_FS_OPEN(path, RF_FILE_READ);
            if (!file) {
                eml_debug(2, "❌ Failed to open tree file: ", path);
                return false;
            }

            uint32_t magic = 0;
            if (file.read(reinterpret_cast<uint8_t*>(&magic), sizeof(magic)) != sizeof(magic)) {
                file.close();
                return false;
            }

            if (magic != 0x33524354) { // "TRC3"
                eml_debug(0, "❌ Invalid tree file format (expected TRC3): ", path);
                file.close();
                return false;
            }

            {
                uint8_t version = 0;
                if (file.read(reinterpret_cast<uint8_t*>(&version), sizeof(version)) != sizeof(version) || version != 3) {
                    file.close();
                    return false;
                }

                auto read_u32 = [&](uint32_t& out) -> bool {
                    return file.read(reinterpret_cast<uint8_t*>(&out), sizeof(out)) == sizeof(out);
                };
                auto read_le = [&](uint64_t& out, uint8_t bytes) -> bool {
                    out = 0;
                    for (uint8_t b = 0; b < bytes; ++b) {
                        uint8_t byte = 0;
                        if (file.read(reinterpret_cast<uint8_t*>(&byte), 1) != 1) {
                            return false;
                        }
                        out |= (static_cast<uint64_t>(byte) << (8u * b));
                    }
                    return true;
                };

                uint8_t tBits = 0, fBits = 0, lBits = 0, cBits = 0;
                if (file.read(reinterpret_cast<uint8_t*>(&tBits), 1) != 1 ||
                    file.read(reinterpret_cast<uint8_t*>(&fBits), 1) != 1 ||
                    file.read(reinterpret_cast<uint8_t*>(&lBits), 1) != 1 ||
                    file.read(reinterpret_cast<uint8_t*>(&cBits), 1) != 1) {
                    file.close();
                    return false;
                }
                if (resource) {
                    resource->set_bits(fBits, lBits, cBits, tBits);
                }

                uint8_t rootLeaf = 0;
                if (file.read(reinterpret_cast<uint8_t*>(&rootLeaf), sizeof(rootLeaf)) != sizeof(rootLeaf)) {
                    file.close();
                    return false;
                }
                root_is_leaf = (rootLeaf != 0);

                uint32_t rootIndexU32 = 0;
                if (!read_u32(rootIndexU32)) {
                    file.close();
                    return false;
                }
                root_index = static_cast<node_idx_type>(rootIndexU32);

                uint32_t branchCountU32 = 0, internalCountU32 = 0, mixedCountU32 = 0, leafCountU32 = 0;
                if (!read_u32(branchCountU32) || !read_u32(internalCountU32) || !read_u32(mixedCountU32) || !read_u32(leafCountU32)) {
                    file.close();
                    return false;
                }

                uint8_t inBits = 0, mxBits = 0, lfBits = 0;
                if (file.read(reinterpret_cast<uint8_t*>(&inBits), 1) != 1 ||
                    file.read(reinterpret_cast<uint8_t*>(&mxBits), 1) != 1 ||
                    file.read(reinterpret_cast<uint8_t*>(&lfBits), 1) != 1) {
                    file.close();
                    return false;
                }
                const uint8_t inBytes = static_cast<uint8_t>((inBits + 7) / 8);
                const uint8_t mxBytes = static_cast<uint8_t>((mxBits + 7) / 8);
                const uint8_t lfBytes = static_cast<uint8_t>((lfBits + 7) / 8);

                internal_nodes.set_bits_per_value(inBits);
                mixed_nodes.set_bits_per_value(mxBits);
                leaf_nodes.set_bits_per_value(lfBits);
                branch_kind.set_bits_per_value(1);

                internal_nodes.clear();
                mixed_nodes.clear();
                leaf_nodes.clear();
                branch_kind.clear();
                mixed_prefix.clear();

                uint32_t kindBytes = 0;
                if (!read_u32(kindBytes)) {
                    file.close();
                    return false;
                }

                branch_kind.resize(static_cast<node_idx_type>(branchCountU32), 0);
                for (uint32_t byteIndex = 0; byteIndex < kindBytes; ++byteIndex) {
                    uint8_t in = 0;
                    if (file.read(reinterpret_cast<uint8_t*>(&in), 1) != 1) {
                        file.close();
                        return false;
                    }
                    const uint32_t base = byteIndex * 8u;
                    for (uint8_t bit = 0; bit < 8; ++bit) {
                        const uint32_t i = base + static_cast<uint32_t>(bit);
                        if (i < branchCountU32) {
                            branch_kind.set(static_cast<node_idx_type>(i), static_cast<uint8_t>((in >> bit) & 1u));
                        }
                    }
                }

                internal_nodes.reserve(static_cast<node_idx_type>(internalCountU32));
                for (uint32_t i = 0; i < internalCountU32; ++i) {
                    uint64_t raw = 0;
                    if (!read_le(raw, inBytes)) {
                        file.close();
                        return false;
                    }
                    Internal_node n;
                    n.packed_data = static_cast<size_t>(raw);
                    internal_nodes.push_back(n);
                }

                mixed_nodes.reserve(static_cast<node_idx_type>(mixedCountU32));
                for (uint32_t i = 0; i < mixedCountU32; ++i) {
                    uint64_t raw = 0;
                    if (!read_le(raw, mxBytes)) {
                        file.close();
                        return false;
                    }
                    Mixed_node n;
                    n.packed_data = static_cast<size_t>(raw);
                    mixed_nodes.push_back(n);
                }

                leaf_nodes.reserve(static_cast<node_idx_type>(leafCountU32));
                for (uint32_t i = 0; i < leafCountU32; ++i) {
                    uint64_t raw = 0;
                    if (!read_le(raw, lfBytes)) {
                        file.close();
                        return false;
                    }
                    leaf_nodes.push_back(static_cast<rf_label_type>(raw));
                }

                file.close();
                rebuild_compact_index();
                isLoaded = true;
            }

            if (!re_use) {
                eml_debug(2, "♻️ Single-load mode: removing tree file after loading; ", path);
                RF_FS_REMOVE(path);
            }
            return true;
        }

        bool saveBootstrapIDs(const char* path) {
            if (bootstrapIDs.empty()) {
                if (RF_FS_EXISTS(path)) RF_FS_REMOVE(path);
                return true;
            }
            File file = RF_FS_OPEN(path, RF_FILE_WRITE);
            if (!file) return false;
            
            uint32_t magic = 0x42544944; // "BTID"
            file.write((uint8_t*)&magic, 4);
            
            sample_idx_type min_id = bootstrapIDs.minID();
            sample_idx_type max_id = bootstrapIDs.maxID();
            uint32_t size = bootstrapIDs.size();
            
            file.write((uint8_t*)&min_id, sizeof(sample_idx_type));
            file.write((uint8_t*)&max_id, sizeof(sample_idx_type));
            file.write((uint8_t*)&size, sizeof(uint32_t));

            for (sample_idx_type id = min_id; id <= max_id; ++id) {
                uint32_t c = bootstrapIDs.count(id);
                if (c > 0) {
                    file.write((uint8_t*)&id, sizeof(sample_idx_type));
                    file.write((uint8_t*)&c, sizeof(uint32_t));
                }
            }
            file.close();
            return true;
        }

        bool loadBootstrapIDs(const char* path) {
            bootstrapIDs.clear();
            if (!RF_FS_EXISTS(path)) return true;
            
            File file = RF_FS_OPEN(path, RF_FILE_READ);
            if (!file) return false;
            
            uint32_t magic;
            if (file.read((uint8_t*)&magic, 4) != 4 || magic != 0x42544944) {
                file.close();
                return false;
            }
            
            sample_idx_type min_id, max_id;
            uint32_t size;
            file.read((uint8_t*)&min_id, sizeof(sample_idx_type));
            file.read((uint8_t*)&max_id, sizeof(sample_idx_type));
            file.read((uint8_t*)&size, sizeof(uint32_t));
            
            bootstrapIDs.set_ID_range(min_id, max_id);
            
            while (file.available()) {
                sample_idx_type id;
                uint32_t c;
                if (file.read((uint8_t*)&id, sizeof(sample_idx_type)) != sizeof(sample_idx_type)) break;
                if (file.read((uint8_t*)&c, sizeof(uint32_t)) != sizeof(uint32_t)) break;
                for (uint32_t i = 0; i < c; ++i) {
                    bootstrapIDs.push_back(id);
                }
            }
            file.close();
            return true;
        }

        __attribute__((always_inline)) inline rf_label_type predict_features(
            const packed_vector<8>& packed_features) const {
            // Early exit for empty tree
            if (__builtin_expect(!resource || leaf_nodes.size() == 0, 0)) {
                return RF_ERROR_LABEL;
            }

            // Handle leaf root
            if (__builtin_expect(root_is_leaf, 0)) {
                return (root_index < leaf_nodes.size()) ? leaf_nodes.get(root_index) : RF_ERROR_LABEL;
            }

            // Cache resource reference to avoid repeated pointer dereference
            const node_resource& res = *resource;
            
            node_idx_type currentBranch = root_index;
            const node_idx_type branchCount = static_cast<node_idx_type>(branch_kind.size());
            const node_idx_type leafCount = static_cast<node_idx_type>(leaf_nodes.size());
            
            uint16_t maxDepth = 100;
            while (__builtin_expect(maxDepth-- > 0, 1)) {
                if (__builtin_expect(currentBranch >= branchCount, 0)) {
                    return RF_ERROR_LABEL;
                }
                
                const uint8_t kind = branch_kind.get(currentBranch);
                if (__builtin_expect(kind == 0, 1)) {
                    // Internal node (common case)
                    const node_idx_type mixedBefore = rank_mixed(currentBranch);
                    const node_idx_type internalIndex = currentBranch - mixedBefore;
                    const Internal_node node = internal_nodes.get(internalIndex);
                    
                    const uint16_t featureID = node.getFeatureID(res);
                    const uint16_t featureValue = static_cast<uint16_t>(packed_features[featureID]);
                    const uint16_t threshold = node.getThresholdSlot(res);
                    const node_idx_type left = node.getLeftChildIndex(res);
                    const node_idx_type chosen = (featureValue <= threshold) ? left : (left + 1);
                    
                    if (__builtin_expect(node.childrenAreLeaf(), 0)) {
                        return (chosen < leafCount) ? leaf_nodes.get(chosen) : RF_ERROR_LABEL;
                    }
                    currentBranch = chosen;
                } else {
                    // Mixed node (less common)
                    const node_idx_type mixedIndex = rank_mixed(currentBranch);
                    const Mixed_node node = mixed_nodes.get(mixedIndex);
                    
                    const uint16_t featureID = node.getFeatureID(res);
                    const uint16_t featureValue = static_cast<uint16_t>(packed_features[featureID]);
                    const uint16_t threshold = node.getThresholdSlot(res);
                    const bool goLeft = (featureValue <= threshold);
                    const bool leftIsLeaf = node.leftIsLeaf();
                    
                    if (goLeft) {
                        const node_idx_type idx = node.getLeftChildIndex(res);
                        if (leftIsLeaf) {
                            return (idx < leafCount) ? leaf_nodes.get(idx) : RF_ERROR_LABEL;
                        }
                        currentBranch = idx;
                    } else {
                        const node_idx_type idx = node.getRightChildIndex(res);
                        if (!leftIsLeaf) {
                            return (idx < leafCount) ? leaf_nodes.get(idx) : RF_ERROR_LABEL;
                        }
                        currentBranch = idx;
                    }
                }
            }

            return RF_ERROR_LABEL;
        }

        void clearTree(bool freeMemory = false) {
            (void)freeMemory;
            nodes.clear();
            nodes.shrink_to_fit();
            internal_nodes.clear();
            internal_nodes.shrink_to_fit();
            mixed_nodes.clear();
            mixed_nodes.shrink_to_fit();
            leaf_nodes.clear();
            leaf_nodes.shrink_to_fit();
            branch_kind.clear();
            branch_kind.shrink_to_fit();
            mixed_prefix.clear();
            mixed_prefix.shrink_to_fit();
            isLoaded = false;
        }

        void purgeTree(const char* path, bool rmf = true) {
            nodes.clear();
            nodes.shrink_to_fit();
            internal_nodes.clear();
            internal_nodes.shrink_to_fit();
            mixed_nodes.clear();
            mixed_nodes.shrink_to_fit();
            leaf_nodes.clear();
            leaf_nodes.shrink_to_fit();
            branch_kind.clear();
            branch_kind.shrink_to_fit();
            mixed_prefix.clear();
            mixed_prefix.shrink_to_fit();
            if (rmf && index < RF_MAX_TREES) {
                if (RF_FS_EXISTS(path)) {
                    RF_FS_REMOVE(path);
                    eml_debug(2, "🗑️ Tree file removed: ", path);
                }
            }
            index = 255;
            isLoaded = false;
        }

    private:
        // Build mixed_prefix for rank queries
        inline void build_mixed_prefix() {
            mixed_prefix.clear();
            const size_t wcount = branch_kind.words();
            mixed_prefix.reserve(wcount + 1);
            mixed_prefix.push_back(0);
            uint16_t acc = 0;
            const auto* w = branch_kind.raw_data();
            for (size_t i = 0; i < wcount; ++i) {
                const size_t word = w ? w[i] : 0;
                if constexpr (sizeof(size_t) == 8) {
                    acc = static_cast<uint16_t>(acc + static_cast<uint16_t>(__builtin_popcountll(static_cast<unsigned long long>(word))));
                } else {
                    acc = static_cast<uint16_t>(acc + static_cast<uint16_t>(__builtin_popcount(static_cast<unsigned int>(word))));
                }
                mixed_prefix.push_back(acc);
            }
        }

        // Rank query: number of mixed nodes before branchIndex
        inline node_idx_type rank_mixed(node_idx_type branchIndex) const {
            // Number of mixed nodes strictly before branchIndex
            const size_t WORD_BITS = sizeof(size_t) * 8;
            const size_t wi = static_cast<size_t>(branchIndex) / WORD_BITS;
            const size_t bi = static_cast<size_t>(branchIndex) % WORD_BITS;
            if (wi >= mixed_prefix.size()) {
                return 0;
            }
            const uint16_t base = mixed_prefix[wi];
            const size_t* w = branch_kind.raw_data();
            if (!w) {
                return base;
            }
            size_t mask = 0;
            if (bi == 0) {
                mask = 0;
            } else if (bi >= WORD_BITS) {
                mask = static_cast<size_t>(~static_cast<size_t>(0));
            } else {
                mask = (static_cast<size_t>(1) << bi) - 1u;
            }
            const size_t word = w[wi] & mask;
            uint16_t pc = 0;
            if constexpr (sizeof(size_t) == 8) {
                pc = static_cast<uint16_t>(__builtin_popcountll(static_cast<unsigned long long>(word)));
            } else {
                pc = static_cast<uint16_t>(__builtin_popcount(static_cast<unsigned int>(word)));
            }
            return static_cast<node_idx_type>(base + pc);
        }

        inline uint8_t desired_bits_per_node() const noexcept {
            uint8_t bits = resource ? resource->bits_per_building_node() : static_cast<uint8_t>(32);
            if (bits == 0 || bits > 32) {
                bits = 32;
            }
            return bits;
        }
    };
    
    
    /*
    ------------------------------------------------------------------------------------------------------------------
    -------------------------------------------- RF_NODE_PREDCITOR ---------------------------------------------------
    ------------------------------------------------------------------------------------------------------------------
    */
    struct node_data {
        uint32_t total_nodes;
        uint8_t min_leaf;
        uint8_t min_split;
        uint16_t max_depth;

        node_data() : total_nodes(0), min_leaf(0), min_split(0), max_depth(250) {}
        node_data(uint8_t min_split, uint8_t min_leaf, uint16_t max_depth = 250)
            : total_nodes(0), min_leaf(min_leaf), min_split(min_split), max_depth(max_depth) {}
        node_data(uint8_t min_split, uint8_t min_leaf, uint16_t max_depth, uint32_t total_nodes)
            : total_nodes(total_nodes), min_leaf(min_leaf), min_split(min_split), max_depth(max_depth) {}
    };

    class Rf_node_predictor {
    public:
        float coefficients[4];  // bias, min_split_coeff, min_leaf_coeff, max_depth_coeff
        bool is_trained;
        b_vector<node_data, 12> buffer;
    private:
        const Rf_base* base_ptr = nullptr;
        const Rf_config* config_ptr = nullptr;
        uint32_t trained_sample_count = 0;   // Samples present when coefficients were derived
        bool dataset_warning_emitted = false;
        bool dataset_drift_emitted = false;
        
        bool has_base() const {
            return base_ptr != nullptr && base_ptr->ready_to_use();
        }

        float evaluate_formula(const node_data& data) const {
            if (!is_trained) {
                return manual_estimate(data); // Use manual estimate if not trained
            }
            
            float result = coefficients[0]; // bias
            result += coefficients[1] * static_cast<float>(data.min_split);
            result += coefficients[2] * static_cast<float>(data.min_leaf);
            result += coefficients[3] * static_cast<float>(data.max_depth);
            
            return result > 10.0f ? result : 10.0f; // ensure reasonable minimum
        }

        // if failed to load predictor, manual estimate will be used
        float manual_estimate(const node_data& data) const {
            if (data.min_split == 0) {
                return 100.0f; 
            }
            
            // Enhanced heuristic considering dataset complexity
            // Base estimate accounts for tree structure parameters
            float safe_leaf = max(1.0f, static_cast<float>(data.min_leaf));
            float leaf_adjustment = 60.0f / safe_leaf;
            float depth_factor = min(250.0f, static_cast<float>(data.max_depth)) / 50.0f;
            
            // Dataset complexity factors
            float sample_factor = 1.0f;
            float feature_factor = 1.0f;
            float label_factor = 1.0f;
            
            if (config_ptr) {
                // More samples → more potential nodes (logarithmic growth)
                if (config_ptr->num_samples > 100) {
                    sample_factor = 1.0f + 0.5f * log2(static_cast<float>(config_ptr->num_samples) / 100.0f);
                    sample_factor = min(2.5f, sample_factor); // Cap at 2.5x
                }
                
                // More features → more splitting opportunities (sublinear)
                if (config_ptr->num_features > 10) {
                    feature_factor = 1.0f + 0.3f * log2(static_cast<float>(config_ptr->num_features) / 10.0f);
                    feature_factor = min(2.0f, feature_factor); // Cap at 2.0x
                }
                
                // More labels → more complex decision boundaries (linear)
                if (config_ptr->num_labels > 2) {
                    label_factor = 0.8f + 0.2f * static_cast<float>(config_ptr->num_labels) / 10.0f;
                    label_factor = min(1.5f, label_factor); // Cap at 1.5x
                }
            }
            
            float estimate = 120.0f - data.min_split * 10.0f + leaf_adjustment + depth_factor * 15.0f;
            estimate *= (sample_factor * feature_factor * label_factor);
            
            return estimate < 10.0f ? 10.0f : estimate; // ensure reasonable minimum
        }

        // Predict number of nodes for given parameters
        float raw_estimate(const node_data& data) {
            if(!is_trained) {
                if(!loadPredictor()){
                    return manual_estimate(data); // Use manual estimate if predictor is disabled 
                }
            }
            float prediction = evaluate_formula(data);
            if (is_trained && config_ptr) {
                uint32_t current_samples = config_ptr->num_samples;
                if (trained_sample_count > 0 && current_samples > 0) {
                    float ratio = static_cast<float>(current_samples) / static_cast<float>(trained_sample_count);
                    if (ratio > 1.75f || ratio < 0.5f) {
                        if (!dataset_drift_emitted) {
                            eml_debug_2(1, "⚠️ Node predictor dataset drift detected. Trained on ", trained_sample_count, ", current samples: ", current_samples);
                            eml_debug(1, "   Recommendation: retrain node predictor to refresh coefficients.");
                            dataset_drift_emitted = true;
                        }
                        return manual_estimate(data);
                    }

                    if ((ratio > 1.05f || ratio < 0.95f) && !dataset_warning_emitted) {
                        eml_debug(1, "ℹ️ Adjusting node estimate for sample count change.");
                        eml_debug_2(1, "   factor: ", ratio, "", "");
                        dataset_warning_emitted = true;
                    }

                    float clamped_ratio = ratio;
                    if (clamped_ratio > 1.35f) clamped_ratio = 1.35f;
                    if (clamped_ratio < 0.75f) clamped_ratio = 0.75f;
                    prediction *= clamped_ratio;
                }
            }
            return prediction; 
        }
        

    public:
        uint8_t accuracy;      // in percentage
        uint8_t peak_percent;  // number of nodes at depth with maximum number of nodes / total number of nodes in tree
        
        Rf_node_predictor() : is_trained(false), accuracy(0), peak_percent(0) {
            for (int i = 0; i < 4; i++) {
                coefficients[i] = 0.0f;
            }
            trained_sample_count = 0;
            dataset_warning_emitted = false;
            dataset_drift_emitted = false;
            base_ptr = nullptr;
            config_ptr = nullptr;
        }

        Rf_node_predictor(Rf_base* base) : base_ptr(base), is_trained(false), accuracy(0), peak_percent(0) {
            eml_debug(2, "🔧 Initializing node predictor");
            for (int i = 0; i < 4; i++) {
                coefficients[i] = 0.0f;
            }
            trained_sample_count = 0;
            dataset_warning_emitted = false;
            dataset_drift_emitted = false;
            config_ptr = nullptr;
        }

        ~Rf_node_predictor() {
            base_ptr = nullptr;
            config_ptr = nullptr;
            is_trained = false;
            buffer.clear();
            trained_sample_count = 0;
            dataset_warning_emitted = false;
            dataset_drift_emitted = false;
        }

        void init(Rf_base* base, const Rf_config* config = nullptr) {
            base_ptr = base;
            config_ptr = config;
            is_trained = false;
            trained_sample_count = 0;
            dataset_warning_emitted = false;
            dataset_drift_emitted = false;
            for (int i = 0; i < 4; i++) {
                coefficients[i] = 0.0f;
            }
            char node_predictor_log[RF_PATH_BUFFER] = {0};
            if (base_ptr) {
                base_ptr->get_node_log_path(node_predictor_log);
            }
            // Create new file with correct header if it doesn't exist
            if (node_predictor_log[0] != '\0' && !RF_FS_EXISTS(node_predictor_log)) {
                File logFile = RF_FS_OPEN(node_predictor_log, FILE_WRITE);
                if (logFile) {
                    logFile.println("min_split,min_leaf,max_depth,max_nodes");
                    logFile.close();
                }
            }
        }
        
        // Load trained model from file system (updated format without version)
        bool loadPredictor() {
            if (!has_base()){
                eml_debug(0, "❌ Load Predictor failed: base pointer not ready");
                return false;
            }
            char file_path[RF_PATH_BUFFER];
            base_ptr->get_node_pred_path(file_path);
            eml_debug(2, "🔍 Loading node predictor from file: ", file_path);
            if(is_trained) return true;
            dataset_warning_emitted = false;
            dataset_drift_emitted = false;
            if (!RF_FS_EXISTS(file_path)) {
                eml_debug(1, "⚠️  No predictor file found, using default predictor.");
                return false;
            }
            
            File file = RF_FS_OPEN(file_path, RF_FILE_READ);
            if (!file) {
                eml_debug(0, "❌ Failed to open predictor file: ", file_path);
                return false;
            }
            
            // Read and verify magic number
            uint32_t magic;
            if (file.read((uint8_t*)&magic, sizeof(magic)) != sizeof(magic) || magic != 0x4E4F4445) {
                eml_debug(0, "❌ Invalid predictor file format: ", file_path);
                file.close();
                return false;
            }
            
            // Read training status (but don't use it to set is_trained - that's set after successful loading)
            bool file_is_trained;
            if (file.read((uint8_t*)&file_is_trained, sizeof(file_is_trained)) != sizeof(file_is_trained)) {
                eml_debug(0, "❌ Failed to read training status");
                file.close();
                return false;
            }
            
            // Read accuracy and peak_percent
            if (file.read((uint8_t*)&accuracy, sizeof(accuracy)) != sizeof(accuracy)) {
                eml_debug(2, "⚠️ Failed to read accuracy, using manual estimate node.");
            }
            
            if (file.read((uint8_t*)&peak_percent, sizeof(peak_percent)) != sizeof(peak_percent)) {
                eml_debug(2, "⚠️ Failed to read peak_percent, using manual estimate node.");
            }
            
            // Read number of coefficients
            uint8_t num_coefficients;
            if (file.read((uint8_t*)&num_coefficients, sizeof(num_coefficients)) != sizeof(num_coefficients)) {
                eml_debug(0, "❌ Failed to read coefficient count");
                file.close();
                return false;
            }

            if (num_coefficients == 3) {
                if (file.read((uint8_t*)coefficients, sizeof(float) * 3) != sizeof(float) * 3) {
                    eml_debug(0, "❌ Failed to read legacy coefficients");
                    file.close();
                    return false;
                }
                coefficients[3] = 0.0f; // Legacy files omitted depth coefficient
            } else if (num_coefficients == 4) {
                if (file.read((uint8_t*)coefficients, sizeof(float) * 4) != sizeof(float) * 4) {
                    eml_debug(0, "❌ Failed to read coefficients");
                    file.close();
                    return false;
                }
            } else {
                eml_debug_2(2, "❌ Unsupported coefficient count: ", num_coefficients, "", "");
                file.close();
                return false;
            }

            // Optional sample count metadata (available in new format)
            trained_sample_count = 0;
            uint32_t stored_samples = 0;
            size_t bytes_read = file.read((uint8_t*)&stored_samples, sizeof(stored_samples));
            if (bytes_read == sizeof(stored_samples)) {
                trained_sample_count = stored_samples;
            }
            
            file.close();
            
            // Only set is_trained to true if the file was actually trained
            if (file_is_trained) {
                is_trained = true;
                if (peak_percent == 0) {
                    peak_percent = 30; // Use reasonable default for binary trees
                    eml_debug(2, "⚠️  Fixed peak_percent from 0% to 30%");
                }
                eml_debug(1, "✅ Node predictor loaded : ", file_path);
                eml_debug(2, "bias: ", this->coefficients[0]);
                eml_debug(2, "min_split effect: ", this->coefficients[1]);
                eml_debug(2, "min_leaf effect: ", this->coefficients[2]);
                eml_debug(2, "accuracy: ", accuracy);
                dataset_warning_emitted = false;
                dataset_drift_emitted = false;
                if (trained_sample_count == 0) {
                    eml_debug(2, "ℹ️ Predictor file missing sample count metadata (legacy format).");
                } else {
                    eml_debug_2(2, "   Predictor trained on samples: ", trained_sample_count, "", "");
                }
            } else {
                eml_debug(1, "⚠️  Predictor file exists but is not trained, using default predictor.");
                is_trained = false;
                trained_sample_count = 0;
                dataset_warning_emitted = false;
                dataset_drift_emitted = false;
            }
            return file_is_trained;
        }
        
        // Save trained predictor to file system
        bool releasePredictor() {
            if (!has_base()){
                eml_debug(0, "❌ Release Predictor failed: base pointer not ready");
                return false;
            }
            if (!is_trained) {
                eml_debug(1, "❌ Predictor is not trained, cannot save.");
                return false;
            }
            char file_path[RF_PATH_BUFFER];
            base_ptr->get_node_pred_path(file_path);
            if (RF_FS_EXISTS(file_path)) RF_FS_REMOVE(file_path);

            File file = RF_FS_OPEN(file_path, FILE_WRITE);
            if (!file) {
                eml_debug(0, "❌ Failed to create predictor file: ", file_path);
                return false;
            }
            
            if (config_ptr) {
                trained_sample_count = static_cast<uint32_t>(config_ptr->num_samples);
            }

            // Write magic number
            uint32_t magic = 0x4E4F4445; // "NODE" in hex
            file.write((uint8_t*)&magic, sizeof(magic));
            
            // Write training status
            file.write((uint8_t*)&is_trained, sizeof(is_trained));
            
            // Write accuracy and peak_percent
            file.write((uint8_t*)&accuracy, sizeof(accuracy));
            file.write((uint8_t*)&peak_percent, sizeof(peak_percent));
            
            // Write number of coefficients
            uint8_t num_coefficients = 4;
            file.write((uint8_t*)&num_coefficients, sizeof(num_coefficients));
            
            // Write coefficients
            file.write((uint8_t*)coefficients, sizeof(float) * 4);

            // Write dataset sample count metadata
            file.write((uint8_t*)&trained_sample_count, sizeof(trained_sample_count));
            
            file.close();
            dataset_warning_emitted = false;
            dataset_drift_emitted = false;
            eml_debug(1, "✅ Node predictor saved: ", file_path);
            return true;
        }
        
        // Add new training samples to buffer
        void add_new_samples(uint8_t min_split, uint8_t min_leaf, uint16_t max_depth, uint32_t total_nodes) {
            if (min_split == 0 || min_leaf == 0) return; // invalid sample
            if (buffer.size() >= 100) {
                eml_debug(2, "⚠️ Node_pred buffer full, consider retraining soon.");
                return;
            }
            buffer.push_back(node_data(min_split, min_leaf, max_depth, total_nodes));
        }
        // Retrain the predictor using data from rf_tree_log.csv (synchronized with PC version)
        bool re_train(bool save_after_retrain = true) {
            if (!has_base()){
                eml_debug(0, "❌ Base pointer is null, cannot retrain predictor.");
                return false;
            }
            if(buffer.size() > 0){
                flush_buffer();
            }
            buffer.clear();
            buffer.shrink_to_fit();

            if(!can_retrain()) {
                eml_debug(2, "❌ No training data available for retraining.");
                return false;
            }

            char node_predictor_log[RF_PATH_BUFFER];
            base_ptr->get_node_log_path(node_predictor_log);
            eml_debug(2, "🔂 Starting retraining of node predictor...");
            File file = RF_FS_OPEN(node_predictor_log, RF_FILE_READ);
            if (!file) {
                eml_debug(1, "❌ Failed to open node_predictor log file: ", node_predictor_log);
                return false;
            }
            eml_debug(2, "🔄 Retraining node predictor from CSV data...");
            
            b_vector<node_data> training_data;
            training_data.reserve(50); // Reserve space for training samples
            
            String line;
            bool first_line = true;
            
            // Read CSV data
            while (file.available()) {
                line = file.readStringUntil('\n');
                line.trim();
                
                if (line.length() == 0 || first_line) {
                    first_line = false;
                    continue; // Skip header line
                }
                
                // Parse CSV line: min_split,min_leaf,max_depth,total_nodes
                node_data sample;
                int comma1 = line.indexOf(',');
                int comma2 = line.indexOf(',', comma1 + 1);
                int comma3 = line.indexOf(',', comma2 + 1);
                
                if (comma1 != -1 && comma2 != -1 && comma3 != -1) {
                    String min_split_str = line.substring(0, comma1);
                    String min_leaf_str = line.substring(comma1 + 1, comma2);
                    String max_depth_str = line.substring(comma2 + 1, comma3);
                    String total_nodes_str = line.substring(comma3 + 1);

                    int parsed_split = min_split_str.toInt();
                    if (parsed_split < 0) parsed_split = 0;
                    if (parsed_split > 255) parsed_split = 255;
                    sample.min_split = static_cast<uint8_t>(parsed_split);

                    int parsed_leaf = min_leaf_str.toInt();
                    if (parsed_leaf < 0) parsed_leaf = 0;
                    if (parsed_leaf > 255) parsed_leaf = 255;
                    sample.min_leaf = static_cast<uint8_t>(parsed_leaf);
                    
                    int parsed_depth = max_depth_str.toInt();
                    if (parsed_depth < 0) parsed_depth = 0;
                    if (parsed_depth > 65535) parsed_depth = 65535;
                    sample.max_depth = static_cast<uint16_t>(parsed_depth);
                    
                    sample.total_nodes = total_nodes_str.toInt();
                    
                    // skip invalid samples
                    if (sample.min_split > 0 && sample.min_leaf > 0 && sample.max_depth > 0 && sample.total_nodes > 0) {
                        training_data.push_back(sample);
                    }
                }
            }
            file.close();
            
            if (training_data.size() < 3) {
                return false;
            }
            
            // Collect all unique min_split and min_leaf values
            b_vector<uint8_t> unique_min_splits;
            b_vector<uint8_t> unique_min_leafs;
            
            for (const auto& sample : training_data) {
                // Add unique min_split values
                bool found_split = false;
                for (const auto& existing_split : unique_min_splits) {
                    if (existing_split == sample.min_split) {
                        found_split = true;
                        break;
                    }
                }
                if (!found_split) {
                    unique_min_splits.push_back(sample.min_split);
                }
                
                // Add unique min_leaf values
                bool found_leaf = false;
                for (const auto& existing_leaf : unique_min_leafs) {
                    if (existing_leaf == sample.min_leaf) {
                        found_leaf = true;
                        break;
                    }
                }
                if (!found_leaf) {
                    unique_min_leafs.push_back(sample.min_leaf);
                }
            }
            
            // Sort vectors for easier processing
            unique_min_splits.sort();
            unique_min_leafs.sort();
            
            // Calculate effects using a simpler approach without large intermediate vectors
            // Calculate min_split effect directly
            float split_effect = 0.0f;
            if (unique_min_splits.size() >= 2) {
                // Calculate average nodes for first and last min_split values
                float first_split_avg = 0.0f;
                float last_split_avg = 0.0f;
                int first_split_count = 0;
                int last_split_count = 0;
                
                uint8_t first_split = unique_min_splits[0];
                uint8_t last_split = unique_min_splits[unique_min_splits.size() - 1];
                
                // Calculate averages directly from training data
                for (const auto& sample : training_data) {
                    if (sample.min_split == first_split) {
                        first_split_avg += sample.total_nodes;
                        first_split_count++;
                    } else if (sample.min_split == last_split) {
                        last_split_avg += sample.total_nodes;
                        last_split_count++;
                    }
                }
                
                if (first_split_count > 0 && last_split_count > 0) {
                    first_split_avg /= first_split_count;
                    last_split_avg /= last_split_count;
                    
                    float split_range = static_cast<float>(last_split - first_split);
                    if (split_range > 0.01f) {
                        split_effect = (last_split_avg - first_split_avg) / split_range;
                    }
                }
            }
            
            // Calculate min_leaf effect directly
            float leaf_effect = 0.0f;
            if (unique_min_leafs.size() >= 2) {
                // Calculate average nodes for first and last min_leaf values
                float first_leaf_avg = 0.0f;
                float last_leaf_avg = 0.0f;
                int first_leaf_count = 0;
                int last_leaf_count = 0;

                uint8_t first_leaf = unique_min_leafs[0];
                uint8_t last_leaf = unique_min_leafs[unique_min_leafs.size() - 1];

                // Calculate averages directly from training data
                for (const auto& sample : training_data) {
                    if (sample.min_leaf == first_leaf) {
                        first_leaf_avg += sample.total_nodes;
                        first_leaf_count++;
                    } else if (sample.min_leaf == last_leaf) {
                        last_leaf_avg += sample.total_nodes;
                        last_leaf_count++;
                    }
                }

                if (first_leaf_count > 0 && last_leaf_count > 0) {
                    first_leaf_avg /= first_leaf_count;
                    last_leaf_avg /= last_leaf_count;

                    float leaf_range = static_cast<float>(last_leaf - first_leaf);
                    if (leaf_range > 0.01f) {
                        leaf_effect = (last_leaf_avg - first_leaf_avg) / leaf_range;
                    }
                }
            }
            
            // Calculate overall average as baseline
            float overall_avg = 0.0f;
            for (const auto& sample : training_data) {
                overall_avg += sample.total_nodes;
            }
            overall_avg /= training_data.size();
            
            // Build the simple linear predictor: nodes = bias + split_coeff * min_split + leaf_coeff * min_leaf
            // Calculate bias to center the predictor around the overall average
            float reference_split = unique_min_splits.empty() ? 3.0f : static_cast<float>(unique_min_splits[0]);
            float reference_leaf = unique_min_leafs.empty() ? 2.0f : static_cast<float>(unique_min_leafs[0]);
            
            coefficients[0] = overall_avg - (split_effect * reference_split) - (leaf_effect * reference_leaf); // bias
            coefficients[1] = split_effect; // min_split coefficient
            coefficients[2] = leaf_effect; // min_leaf coefficient
            
            // Calculate accuracy using PC version's approach exactly
            float total_error = 0.0f;
            float total_actual = 0.0f;
            
            for (const auto& sample : training_data) {
                node_data data(sample.min_split, sample.min_leaf);
                float predicted = evaluate_formula(data);
                float actual = static_cast<float>(sample.total_nodes);
                float error = fabs(predicted - actual);
                total_error += error;
                total_actual += actual;
            }
            
            float mae = total_error / training_data.size(); // Mean Absolute Error
            float mape = (total_error / total_actual) * 100.0f; // Mean Absolute Percentage Error
            
            float get_accuracy_result = fmax(0.0f, 100.0f - mape);
            accuracy = static_cast<uint8_t>(fmin(255.0f, get_accuracy_result * 100.0f / 100.0f)); // Simplify to match intent
            
            // Actually, let's just match the logical intent rather than the bug
            accuracy = static_cast<uint8_t>(fmin(100.0f, fmax(0.0f, 100.0f - mape)));
            
            peak_percent = 30; // A reasonable default for binary tree structures
            
            is_trained = true;
            if (config_ptr) {
                trained_sample_count = static_cast<uint32_t>(config_ptr->num_samples);
            }
            dataset_warning_emitted = false;
            dataset_drift_emitted = false;
            eml_debug(2, "✅ Node predictor retraining complete!");
            eml_debug_2(2, "   Accuracy: ", accuracy, "%, Peak (%): ", peak_percent);
            if(save_after_retrain) {
                releasePredictor(); // Save the new predictor
            }
            return true;
        }
        
        uint16_t estimate_nodes(uint8_t min_split, uint8_t min_leaf, uint16_t max_depth = 50) {
            if (min_leaf == 0) {
                min_leaf = 1;
            }
            node_data data(min_split, min_leaf, max_depth);
            float raw_est = raw_estimate(data);
            float acc = accuracy;
            if(acc < 90.0f) acc = 90.0f;
            uint16_t estimate = static_cast<uint16_t>(raw_est * 100 / acc);
            uint16_t safe_estimate;
            if(config_ptr->num_samples < 2024) safe_estimate = 512;
            else safe_estimate = max(static_cast<sample_idx_type>((config_ptr->num_samples / config_ptr->min_leaf)), RF_MAX_NODES);
            return estimate < RF_MAX_NODES ? estimate : safe_estimate;       
        }

        uint16_t estimate_nodes(const Rf_config& config) {
            uint8_t min_split = config.min_split;
            uint8_t min_leaf = config.min_leaf > 0 ? config.min_leaf : static_cast<uint8_t>(1);
            uint16_t max_depth = config.max_depth > 0 ? config.max_depth : 25;
            return estimate_nodes(min_split, min_leaf, max_depth);
        }

        uint16_t queue_peak_size(uint8_t min_split, uint8_t min_leaf, uint16_t max_depth = 250) {
            return min(120, estimate_nodes(min_split, min_leaf, max_depth) * peak_percent / 100);
        }

        uint16_t queue_peak_size(const Rf_config& config) {
            uint16_t est_nodes = estimate_nodes(config);
            if(config.training_score == K_FOLD_SCORE){
                est_nodes = est_nodes * config.k_folds / (config.k_folds + 1);
            }
            est_nodes = static_cast<uint16_t>(est_nodes * peak_percent / 100);
            uint16_t max_peak_theory = max(static_cast<sample_idx_type>((config_ptr->num_samples / config_ptr->min_leaf)), RF_MAX_NODES) * 0.3; // 30% of theoretical max nodes
            uint16_t min_peak_theory = 30;
            if (est_nodes > max_peak_theory) return max_peak_theory;
            if (est_nodes < min_peak_theory) return min_peak_theory;
            return est_nodes;
        }

        void flush_buffer() {
            if (!has_base()){
                eml_debug(0, "❌Failed to flush_buffer : base pointer is null");
                return;
            }
            char node_predictor_log[RF_PATH_BUFFER];
            base_ptr->get_node_log_path(node_predictor_log);
            if (buffer.size() == 0) return;
            // Read all existing lines
            b_vector<String> lines;
            File file = RF_FS_OPEN(node_predictor_log, RF_FILE_READ);
            if (file) {
                while (file.available()) {
                    String line = file.readStringUntil('\n');
                    line.trim();
                    if (line.length() > 0) lines.push_back(line);
                }
                file.close();
            }
            // Ensure header is present
            String header = "min_split,min_leaf,max_depth,total_nodes";
            if (lines.empty() || lines[0] != header) {
                lines.insert(0, header);
            }
            // Remove header for easier manipulation and discard legacy headers
            b_vector<String> data_lines;
            for (size_t i = 1; i < lines.size(); ++i) {
                const String& existing = lines[i];
                if (existing == header) {
                    continue;
                }
                if (existing.length() == 0) {
                    continue;
                }
                char first_char = existing.charAt(0);
                if (first_char < '0' || first_char > '9') {
                    continue;
                }
                data_lines.push_back(existing);
            }
            // Prepend new samples
            for (int i = buffer.size() - 1; i >= 0; --i) {
                const node_data& nd = buffer[i];
                String row = String(nd.min_split) + "," + String(nd.min_leaf) + "," + String(nd.max_depth) + "," + String(nd.total_nodes);
                data_lines.insert(0, row);
            }
            // Limit to 50 rows
            while (data_lines.size() > 50) {
                data_lines.pop_back();
            }
            // Write back to file
            RF_FS_REMOVE(node_predictor_log);
            file = RF_FS_OPEN(node_predictor_log, FILE_WRITE);
            if (file) {
                file.println(header);
                for (const auto& row : data_lines) {
                    file.println(row);
                }
                file.close();
            }
        }

        bool can_retrain() const {
            if (!has_base()) {
                eml_debug(0, "❌ can_retrain check failed: base pointer not ready");
                return false;
            }
            char node_predictor_log[RF_PATH_BUFFER];
            base_ptr->get_node_log_path(node_predictor_log);
            if (!RF_FS_EXISTS(node_predictor_log)) {
                eml_debug(2, "❌ No log file found for retraining.");
                return false;
            }
            File file = RF_FS_OPEN(node_predictor_log, RF_FILE_READ);
            bool result = file && file.size() > 0;
            // only retrain if log file has more 4 samples (excluding header)
            if (result) {
                size_t line_count = 0;
                while (file.available()) {
                    String line = file.readStringUntil('\n');
                    line.trim();
                    if (line.length() > 0) line_count++;
                }
                result = line_count > 4; // more than header + 3 samples
            }
            if (file) file.close();
            if(!result){
                eml_debug(2, "❌ Not enough data for retraining (need > 3 samples).");
            }
            return result;
        }

        size_t memory_usage() const {
            size_t total = sizeof(Rf_node_predictor);
            total += buffer.capacity() * sizeof(node_data);
            return total + 4;
        }
    };

    /*
    ------------------------------------------------------------------------------------------------------------------------------
    ------------------------------------------------ TREE_CONTAINER ------------------------------------------------------------
    ------------------------------------------------------------------------------------------------------------------------------
    */

    struct NodeToBuild {
        sample_idx_type begin;   // inclusive
        sample_idx_type end;     // exclusive
        node_idx_type nodeIndex;
        uint16_t depth;
        
        NodeToBuild() : begin(0), end(0), nodeIndex(0), depth(0) {}
        NodeToBuild(node_idx_type idx, sample_idx_type b, sample_idx_type e, uint16_t d) 
            : nodeIndex(idx), begin(b), end(e), depth(d) {}
    };


    class Rf_tree_container{
        private:
            // String model_name;
            const Rf_base* base_ptr = nullptr;
            Rf_config* config_ptr = nullptr;
            Rf_node_predictor* node_pred_ptr = nullptr;
            char tree_path_buffer[RF_PATH_BUFFER] = {0}; // Buffer for tree file paths

            vector<Rf_tree> trees;        // stores tree slots and manages file system file_paths
            node_resource resources;
            size_t   total_depths;       // store total depth of all trees
            node_idx_type   total_nodes;        // store total nodes of all trees
            node_idx_type   total_leaves;       // store total leaves of all trees
            vector<NodeToBuild> queue_nodes; // Queue for breadth-first tree building
            unordered_map_s<rf_label_type, sample_idx_type> predictClass; // Map to count predictions per class during inference
            bool is_unified = true;  // Default to unified form (used at the end of training and inference)

            inline bool has_base() const { 
                return config_ptr!= nullptr && base_ptr != nullptr && base_ptr->ready_to_use(); 
            }

            void rebuild_tree_slots(uint8_t count, bool reset_storage = true) {
                trees.clear();
                trees.reserve(count);
                for (uint8_t i = 0; i < count; ++i) {
                    Rf_tree tree(i);
                    tree.set_resource(&resources, reset_storage);
                    trees.push_back(std::move(tree));
                }
            }

            void ensure_tree_slot(uint8_t index) {
                if (index < trees.size()) {
                    if (trees[index].resource != &resources) {
                        trees[index].set_resource(&resources);
                    }
                    if (trees[index].index == 255) {
                        trees[index].index = index;
                    }
                    return;
                }
                const size_t desired = static_cast<size_t>(index) + 1;
                trees.reserve(desired);
                while (trees.size() < desired) {
                    uint8_t new_index = static_cast<uint8_t>(trees.size());
                    Rf_tree tree(new_index);
                    tree.set_resource(&resources, true);
                    trees.push_back(std::move(tree));
                }
            }

        public:
            void calculate_layout(rf_label_type num_label, uint16_t num_feature, node_idx_type max_node){
                const node_idx_type fallback_node_index = 8191;  // safety fallback 13 bits 

                const rf_label_type max_label_id   = (num_label  > 0) ? static_cast<rf_label_type>(num_label  - 1) : 0;
                const uint32_t max_feature_id = (num_feature > 0) ? static_cast<uint32_t>(num_feature - 1) : 0;

                node_idx_type max_node_index = (max_node   > 0) ? static_cast<uint32_t>(max_node   - 1) : 0;
                if (max_node_index > fallback_node_index) {
                    max_node_index = fallback_node_index;
                }

                uint8_t label_bits       = desired_bits(max_label_id);
                uint8_t feature_bits     = desired_bits(max_feature_id);
                uint8_t child_index_bits = desired_bits(max_node_index);
                uint8_t threshold_bits   = config_ptr ? config_ptr->quantization_coefficient : 1;
                if (threshold_bits < 1) {
                    threshold_bits = 1;
                } else if (threshold_bits > 8) {
                    threshold_bits = 8;
                }

                if (label_bits > 8) {
                    label_bits = 8;
                }
                if (feature_bits > 10) {
                    feature_bits = 10;
                }

                const uint8_t max_child_bits_limit = desired_bits(fallback_node_index);

                auto compute_available_child_bits = [&](uint8_t tb) -> uint8_t {
                    if ((1 + tb + feature_bits + label_bits) >= 32) {
                        return static_cast<uint8_t>(0);
                    }
                    return static_cast<uint8_t>(32 - (1 + tb + feature_bits + label_bits));
                };

                uint8_t desired_child_bits = child_index_bits;
                if (desired_child_bits > max_child_bits_limit) {
                    desired_child_bits = max_child_bits_limit;
                }
                if (desired_child_bits == 0) {
                    desired_child_bits = 1;
                }

                uint8_t available_child_bits = compute_available_child_bits(threshold_bits);
                while (threshold_bits > 1 && available_child_bits < desired_child_bits) {
                    --threshold_bits;
                    available_child_bits = compute_available_child_bits(threshold_bits);
                }
                if (available_child_bits == 0) {
                    threshold_bits = 1;
                    available_child_bits = compute_available_child_bits(threshold_bits);
                }

                if (config_ptr && threshold_bits < config_ptr->quantization_coefficient) {
                    eml_debug_2(2, "⚙️ Adjusted threshold bits from ", static_cast<int>(config_ptr->quantization_coefficient),
                             " to ", static_cast<int>(threshold_bits));
                }

                const uint8_t max_child_bits_word = available_child_bits;

                if (max_child_bits_word == 0) {
                    child_index_bits = 1;
                } else {
                    if (child_index_bits > max_child_bits_word) {
                        child_index_bits = max_child_bits_word;
                    }
                    if (child_index_bits > max_child_bits_limit) {
                        child_index_bits = max_child_bits_limit;
                    }
                    if (child_index_bits == 0) {
                        child_index_bits = 1;
                    }
                }

                eml_debug(1, "📐 Calculated node resources :");
                eml_debug(1, "   - Threshold bits : ", static_cast<int>(threshold_bits));
                eml_debug(1, "   - Feature bits   : ", static_cast<int>(feature_bits));
                eml_debug(1, "   - Label bits     : ", static_cast<int>(label_bits));
                eml_debug(1, "   - Child index bits: ", static_cast<int>(child_index_bits));

                resources.set_bits(feature_bits, label_bits, child_index_bits, threshold_bits);
            }
            
            bool is_loaded = false;

            Rf_tree_container(){};
            Rf_tree_container(Rf_base* base, Rf_config* config, Rf_node_predictor* node_pred){
                init(base, config, node_pred);
            }

            bool init(Rf_base* base, Rf_config* config, Rf_node_predictor* node_pred){
                base_ptr = base;
                config_ptr = config;
                node_pred_ptr = node_pred;
                if (!config_ptr) {
                    trees.clear();
                    eml_debug(0, "❌ Cannot initialize tree container: config pointer is null.");
                    return false;
                }
                
                // Check if layout bits are provided in config 
                if (config_ptr->threshold_bits > 0 && config_ptr->feature_bits > 0 && 
                    config_ptr->label_bits > 0 && config_ptr->child_bits > 0) {
                    eml_debug(2, "📐 Setting node layout from config file");
                    resources.set_bits(config_ptr->feature_bits, config_ptr->label_bits,
                                       config_ptr->child_bits, config_ptr->threshold_bits);
                } else {
                    eml_debug(0, "❌ Cannot initialize tree container: layout bits missing in config.");
                    return false; 
                }

                rebuild_tree_slots(config_ptr->num_trees, true);
                predictClass.reserve(config_ptr->num_trees);
                queue_nodes.clear();
                total_depths = 0;
                total_nodes = 0;
                total_leaves = 0;
                is_loaded = false; // Initially in individual form
                return true;
            }
            
            ~Rf_tree_container(){
                // save to file system in unified form 
                releaseForest();
                trees.clear();
                base_ptr = nullptr;
                config_ptr = nullptr;
                node_pred_ptr = nullptr;
            }


            // Clear all trees, old forest file and reset state to individual form (ready for rebuilding)
            void clearForest() {
                eml_debug(1, "🧹 Clearing forest..");
                if(!has_base()) {
                    eml_debug(0, "❌ Cannot clear forest: base or config pointer is null.");
                    return;
                }
                for (size_t i = 0; i < trees.size(); i++) {
                    base_ptr->build_tree_file_path(tree_path_buffer, trees[i].index);
                    trees[i].purgeTree(tree_path_buffer); 
                    yield();        
                    delay(10);
                }
                if (config_ptr) {
                    // Use predictor only if it's trained; otherwise use safe default of 2046 nodes
                    uint16_t est_nodes;
                    if (node_pred_ptr && node_pred_ptr->is_trained) {
                        est_nodes = node_pred_ptr->estimate_nodes(*config_ptr);
                    } else {
                        eml_debug(2, "⚠️ Node predictor not available or not trained, using safe default for layout calculation.");
                        est_nodes = static_cast<uint16_t>(2046); // Safe default when predictor not available/trained
                    }
                    calculate_layout(config_ptr->num_labels, config_ptr->num_features, est_nodes);
                }
                rebuild_tree_slots(config_ptr->num_trees, true);
                is_loaded = false;
                // Remove old forest file to ensure clean slate
                char oldForestFile[RF_PATH_BUFFER];
                base_ptr->get_forest_path(oldForestFile);
                if(RF_FS_EXISTS(oldForestFile)) {
                    RF_FS_REMOVE(oldForestFile);
                    eml_debug(2, "🗑️ Removed old forest file: ", oldForestFile);
                }
                is_unified = false; // Now in individual form
                total_depths = 0;
                total_nodes = 0;
                total_leaves = 0;
            }
            
            bool add_tree(Rf_tree&& tree){
                if(!tree.isLoaded) eml_debug(2, "🟡 Warning: Adding an unloaded tree to the container.");
                if(tree.index != 255 && tree.index < config_ptr->num_trees) {
                    uint8_t index = tree.index;
                    ensure_tree_slot(index);
                    uint16_t d = tree.getTreeDepth();
                    uint16_t n = tree.countNodes();
                    uint16_t l = tree.countLeafNodes();

                    total_depths += d;
                    total_nodes  += n;
                    total_leaves += l;

                    base_ptr->build_tree_file_path(tree_path_buffer, index);
                    // Ensure the tree has access to resource before saving.
                    tree.set_resource(&resources);
                    tree.releaseTree(tree_path_buffer); // Release tree nodes from memory after adding to container
                    trees[tree.index] = std::move(tree);
                    eml_debug_2(1, "🌲 Added tree index: ", index, "- nodes: ", n);
                    // slot.set_layout(&layout);
                } else {
                    eml_debug(0, "❌ Invalid tree index: ",tree.index);
                    return false;
                }
                return true;
            }

            rf_label_type predict_features(const packed_vector<8>& features) {
                if(__builtin_expect(trees.empty() || !is_loaded, 0)) {
                    eml_debug(2, "❌ Forest not loaded or empty, cannot predict.");
                    return RF_ERROR_LABEL; // Unknown class
                }
                
                const rf_label_type numLabels = config_ptr->num_labels;
                
                // Use stack array only for small label sets to avoid stack overflow
                // For larger label sets, use heap-allocated map
                if(__builtin_expect(numLabels <= 32, 1)) {
                    // Fast path: small label count - use stack array (32 bytes max)
                    uint8_t votes[32] = {0};
                    
                    const uint8_t numTrees = trees.size();
                    for(uint8_t t = 0; t < numTrees; ++t) {
                        rf_label_type predict = trees[t].predict_features(features);
                        if(__builtin_expect(predict < numLabels, 1)) {
                            votes[predict]++;
                        }
                    }
                    
                    uint8_t maxVotes = 0;
                    uint8_t mostPredict = 0;
                    for(uint8_t label = 0; label < numLabels; ++label) {
                        if(votes[label] > maxVotes) {
                            maxVotes = votes[label];
                            mostPredict = label;
                        }
                    }
                    
                    return (maxVotes > 0) ? mostPredict : RF_ERROR_LABEL;
                } else {
                    // Slow path: large label count - use map to avoid stack overflow
                    predictClass.clear();
                    
                    const uint8_t numTrees = trees.size();
                    for(uint8_t t = 0; t < numTrees; ++t) {
                        rf_label_type predict = trees[t].predict_features(features);
                        if(__builtin_expect(predict < numLabels, 1)) {
                            predictClass[predict]++;
                        }
                    }
                    
                    uint8_t maxVotes = 0;
                    uint8_t mostPredict = RF_ERROR_LABEL;
                    for(const auto& entry : predictClass) {
                        if(entry.second > maxVotes) {
                            maxVotes = entry.second;
                            mostPredict = entry.first;
                        }
                    }
                    
                    return (maxVotes > 0) ? mostPredict : RF_ERROR_LABEL;
                }
            }

            class iterator {
                using self_type = iterator;
                using value_type = Rf_tree;
                using reference = Rf_tree&;
                using pointer = Rf_tree*;
                using difference_type = std::ptrdiff_t;
                using iterator_category = std::forward_iterator_tag;
                
            public:
                iterator(Rf_tree_container* parent = nullptr, size_t idx = 0) : parent(parent), idx(idx) {}
                reference operator*() const { return parent->trees[idx]; }
                pointer operator->() const { return &parent->trees[idx]; }

                // Prefix ++
                self_type& operator++() { ++idx; return *this; }
                // Postfix ++
                self_type operator++(int) { self_type tmp = *this; ++(*this); return tmp; }

                bool operator==(const self_type& other) const {
                    return parent == other.parent && idx == other.idx;
                }
                bool operator!=(const self_type& other) const {
                    return !(*this == other);
                }

            private:
                Rf_tree_container* parent;
                size_t idx;
            };

            // begin / end to support range-based for and STL-style iteration
            iterator begin() { return iterator(this, 0); }
            iterator end()   { return iterator(this, size()); }

            // Forest loading functionality - dispatcher based on is_unified flag
            bool loadForest() {
                if (is_loaded) {
                    eml_debug(2, "✅ Forest already loaded, skipping load.");
                    return true;
                }
                if(!has_base()) {
                    eml_debug(0, "❌ Base pointer is null", "load forest");
                    return false;
                }
                // Ensure container is properly sized before loading
                if(trees.size() != config_ptr->num_trees) {
                    eml_debug_2(2, "🔧 Adjusting container size from", trees.size(), "to", config_ptr->num_trees);
                    if(config_ptr->num_trees > 0) {
                        ensure_tree_slot(static_cast<uint8_t>(config_ptr->num_trees - 1));
                    } else {
                        trees.clear();
                    }
                }
                // Memory safety check
                size_t freeMemory = eml_memory_status().first;
                if(freeMemory < config_ptr->estimatedRAM + 8000) {
                    eml_debug_2(1, "❌ Insufficient memory to load forest (need", 
                                config_ptr->estimatedRAM + 8000, "bytes, have", freeMemory);
                    return false;
                }
                if (is_unified) {
                    return loadForestUnified();
                } else {
                    return loadForestIndividual();
                }
            }

        private:
            bool check_valid_after_load(){
                // Verify trees are actually loaded
                uint8_t loadedTrees = 0;
                total_depths = 0;
                total_nodes = 0;
                total_leaves = 0;
                for(const auto& tree : trees) {
                    if(tree.isLoaded && (tree.leaf_nodes.size() > 0 || tree.branch_kind.size() > 0 || !tree.nodes.empty())) {
                        loadedTrees++;
                        total_depths += tree.getTreeDepth();
                        total_nodes += tree.countNodes();
                        total_leaves += tree.countLeafNodes();
                    }
                }
                
                if(loadedTrees != config_ptr->num_trees) {
                    eml_debug_2(1, "❌ Loaded trees mismatch: ", loadedTrees, "expected: ", config_ptr->num_trees);
                    is_loaded = false;
                    return false;
                }
                
                is_loaded = true;
                return true;
            }

            // Load forest from unified format (single file containing all trees)
            bool loadForestUnified() {
                char unifiedfile_path[RF_PATH_BUFFER];
                base_ptr->get_forest_path(unifiedfile_path);
                if(unifiedfile_path[0] == '\0' || !RF_FS_EXISTS(unifiedfile_path)) {
                    eml_debug(0, "❌ Unified forest file not found: ", unifiedfile_path);
                    return false;
                }
                
                // Load from unified file (optimized format)
                File file = RF_FS_OPEN(unifiedfile_path, RF_FILE_READ);
                if (!file) {
                    eml_debug(0, "❌ Failed to open unified forest file: ", unifiedfile_path);
                    return false;
                }
                
                // Read forest header with error checking
                uint32_t magic;
                if(file.read((uint8_t*)&magic, sizeof(magic)) != sizeof(magic)) {
                    eml_debug(0, "❌ Failed to read magic number from: ", unifiedfile_path);
                    file.close();
                    return false;
                }
                
                if (magic != 0x33435246) { // "FRC3"
                    eml_debug(0, "❌ Invalid forest file format (expected FRC3): ", unifiedfile_path);
                    file.close();
                    return false;
                }

                {
                    uint8_t version = 0;
                    if (file.read(reinterpret_cast<uint8_t*>(&version), 1) != 1 || version != 3) {
                        file.close();
                        return false;
                    }

                    auto read_u32 = [&](uint32_t& out) -> bool {
                        return file.read(reinterpret_cast<uint8_t*>(&out), sizeof(out)) == sizeof(out);
                    };
                    auto read_le = [&](uint64_t& out, uint8_t bytes) -> bool {
                        out = 0;
                        for (uint8_t b = 0; b < bytes; ++b) {
                            uint8_t byte = 0;
                            if (file.read(reinterpret_cast<uint8_t*>(&byte), 1) != 1) {
                                return false;
                            }
                            out |= (static_cast<uint64_t>(byte) << (8u * b));
                        }
                        return true;
                    };

                    uint8_t treeCount = 0;
                    if (file.read(reinterpret_cast<uint8_t*>(&treeCount), sizeof(treeCount)) != sizeof(treeCount)) {
                        file.close();
                        return false;
                    }
                    if (treeCount != config_ptr->num_trees) {
                        eml_debug_2(1, "⚠️ Tree count mismatch in unified file: ", treeCount, "expected: ", config_ptr->num_trees);
                        file.close();
                        return false;
                    }

                    // Bit widths for container
                    uint8_t tBits = 0, fBits = 0, lBits = 0, cBits = 0;
                    if (file.read(reinterpret_cast<uint8_t*>(&tBits), 1) != 1 ||
                        file.read(reinterpret_cast<uint8_t*>(&fBits), 1) != 1 ||
                        file.read(reinterpret_cast<uint8_t*>(&lBits), 1) != 1 ||
                        file.read(reinterpret_cast<uint8_t*>(&cBits), 1) != 1) {
                        file.close();
                        return false;
                    }
                    resources.set_bits(fBits, lBits, cBits, tBits);

                    eml_debug(1, "📁 Loading from unified compact forest file", unifiedfile_path);

                    uint8_t successfullyLoaded = 0;
                    for (uint8_t i = 0; i < treeCount; ++i) {
                        uint8_t treeIndex = 0;
                        if (file.read(reinterpret_cast<uint8_t*>(&treeIndex), 1) != 1) {
                            break;
                        }
                        ensure_tree_slot(treeIndex);
                        auto& tree = trees[treeIndex];
                        tree.set_resource(&resources);

                        uint8_t rootLeaf = 0;
                        if (file.read(reinterpret_cast<uint8_t*>(&rootLeaf), 1) != 1) {
                            break;
                        }
                        tree.root_is_leaf = (rootLeaf != 0);
                        uint32_t rootIndexU32 = 0;
                        if (!read_u32(rootIndexU32)) {
                            break;
                        }
                        tree.root_index = static_cast<node_idx_type>(rootIndexU32);

                        uint32_t branchCountU32 = 0, internalCountU32 = 0, mixedCountU32 = 0, leafCountU32 = 0;
                        if (!read_u32(branchCountU32) || !read_u32(internalCountU32) || !read_u32(mixedCountU32) || !read_u32(leafCountU32)) {
                            break;
                        }

                        uint8_t inBits = 0, mxBits = 0, lfBits = 0;
                        if (file.read(reinterpret_cast<uint8_t*>(&inBits), 1) != 1 ||
                            file.read(reinterpret_cast<uint8_t*>(&mxBits), 1) != 1 ||
                            file.read(reinterpret_cast<uint8_t*>(&lfBits), 1) != 1) {
                            break;
                        }
                        const uint8_t inBytes = static_cast<uint8_t>((inBits + 7) / 8);
                        const uint8_t mxBytes = static_cast<uint8_t>((mxBits + 7) / 8);
                        const uint8_t lfBytes = static_cast<uint8_t>((lfBits + 7) / 8);

                        tree.internal_nodes.set_bits_per_value(inBits);
                        tree.mixed_nodes.set_bits_per_value(mxBits);
                        tree.leaf_nodes.set_bits_per_value(lfBits);
                        tree.branch_kind.set_bits_per_value(1);

                        tree.internal_nodes.clear();
                        tree.mixed_nodes.clear();
                        tree.leaf_nodes.clear();
                        tree.branch_kind.clear();
                        tree.mixed_prefix.clear();

                        uint32_t kindBytes = 0;
                        if (!read_u32(kindBytes)) {
                            break;
                        }
                        tree.branch_kind.resize(static_cast<node_idx_type>(branchCountU32), 0);
                        for (uint32_t byteIndex = 0; byteIndex < kindBytes; ++byteIndex) {
                            uint8_t in = 0;
                            if (file.read(reinterpret_cast<uint8_t*>(&in), 1) != 1) {
                                break;
                            }
                            const uint32_t base = byteIndex * 8u;
                            for (uint8_t bit = 0; bit < 8; ++bit) {
                                const uint32_t idx = base + static_cast<uint32_t>(bit);
                                if (idx < branchCountU32) {
                                    tree.branch_kind.set(static_cast<node_idx_type>(idx), static_cast<uint8_t>((in >> bit) & 1u));
                                }
                            }
                        }

                        tree.internal_nodes.reserve(static_cast<node_idx_type>(internalCountU32));
                        for (uint32_t k = 0; k < internalCountU32; ++k) {
                            uint64_t raw = 0;
                            if (!read_le(raw, inBytes)) {
                                break;
                            }
                            Internal_node n;
                            n.packed_data = static_cast<size_t>(raw);
                            tree.internal_nodes.push_back(n);
                        }

                        tree.mixed_nodes.reserve(static_cast<node_idx_type>(mixedCountU32));
                        for (uint32_t k = 0; k < mixedCountU32; ++k) {
                            uint64_t raw = 0;
                            if (!read_le(raw, mxBytes)) {
                                break;
                            }
                            Mixed_node n;
                            n.packed_data = static_cast<size_t>(raw);
                            tree.mixed_nodes.push_back(n);
                        }

                        tree.leaf_nodes.reserve(static_cast<node_idx_type>(leafCountU32));
                        for (uint32_t k = 0; k < leafCountU32; ++k) {
                            uint64_t raw = 0;
                            if (!read_le(raw, lfBytes)) {
                                break;
                            }
                            tree.leaf_nodes.push_back(static_cast<rf_label_type>(raw));
                        }

                        tree.isLoaded = true;
                        tree.nodes.clear();
                        tree.nodes.shrink_to_fit();
                        tree.rebuild_compact_index();
                        successfullyLoaded++;
                    }
                    file.close();
                    (void)successfullyLoaded;
                    return check_valid_after_load();
                }
            }

            // Load forest from individual tree files (used during training)
            bool loadForestIndividual() {
                eml_debug(1, "📁 Loading from individual tree files...");
                
                char model_name[RF_PATH_BUFFER];
                base_ptr->get_model_name(model_name, RF_PATH_BUFFER);
                
                uint8_t successfullyLoaded = 0;
                for (auto& tree : trees) {
                    if (!tree.isLoaded) {
                        try {
                            tree.set_resource(&resources);
                            // Construct tree file path
                            base_ptr->build_tree_file_path(tree_path_buffer, tree.index);
                            tree.loadTree(tree_path_buffer);
                            if(tree.isLoaded) successfullyLoaded++;
                        } catch (...) {
                            eml_debug(1, "❌ Exception loading tree: ", tree.index);
                            tree.isLoaded = false;
                        }
                    }
                }
                return check_valid_after_load();
            }

        public:
            // Release forest to unified format (single file containing all trees)
            bool releaseForest() {
                if(!is_loaded || trees.empty()) {
                    eml_debug(2, "✅ Forest is not loaded in memory, nothing to release.");
                    return true; // Nothing to do
                }    
                // Count loaded trees
                uint8_t loadedCount = 0;
                node_idx_type totalNodes = 0;
                for(auto& tree : trees) {
                    if (tree.isLoaded && (tree.leaf_nodes.size() > 0 || tree.branch_kind.size() > 0 || !tree.nodes.empty())) {
                        loadedCount++;
                        totalNodes += tree.countNodes();
                    }
                }
                
                if(loadedCount == 0) {
                    eml_debug(1, "❌ No loaded trees to release");
                    is_loaded = false;
                    return false;
                }
                
                // Check available file system space before writing
                size_t totalFS = RF_TOTAL_BYTES();
                size_t usedFS = RF_USED_BYTES();
                size_t freeFS = totalFS - usedFS;
                // Estimate size using compact structure (rough): assume internal nodes ~half, leaf nodes ~half.
                uint8_t estLeafBits = resources.bits_per_leaf_node();
                uint8_t estInBits = resources.bits_per_internal_node();
                uint8_t estLeafBytes = static_cast<uint8_t>((estLeafBits + 7) / 8);
                uint8_t estInBytes = static_cast<uint8_t>((estInBits + 7) / 8);
                size_t estimatedSize = static_cast<size_t>(totalNodes / 2) * estInBytes + static_cast<size_t>(totalNodes / 2) * estLeafBytes + 256;
                
                if(freeFS < estimatedSize) {
                    eml_debug_2(1, "❌ Insufficient file system space to release forest (need ~", 
                                estimatedSize, "bytes, have", freeFS);
                    return false;
                }
                
                // Single file approach - write all trees to unified forest file
                char unifiedfile_path[RF_PATH_BUFFER];
                base_ptr->get_forest_path(unifiedfile_path);
                if(unifiedfile_path[0] == '\0') {
                    eml_debug(0, "❌ Cannot release forest: no base reference for file management");
                    return false;
                }
                
                unsigned long fileStart = eml_time_now(MILLISECONDS);
                File file = RF_FS_OPEN(unifiedfile_path, FILE_WRITE);
                if (!file) {
                    eml_debug(0, "❌ Failed to create unified forest file: ", unifiedfile_path);
                    return false;
                }
                
                // Write forest header: FRC3 (portable)
                uint32_t magic = 0x33435246; // 'F''R''C''3'
                if(file.write((uint8_t*)&magic, sizeof(magic)) != sizeof(magic)) {
                    eml_debug(0, "❌ Failed to write magic number to: ", unifiedfile_path);
                    file.close();
                    RF_FS_REMOVE(unifiedfile_path);
                    return false;
                }

                const uint8_t version = 3;
                if (file.write(reinterpret_cast<const uint8_t*>(&version), 1) != 1) {
                    file.close();
                    RF_FS_REMOVE(unifiedfile_path);
                    return false;
                }

                auto write_u32 = [&](uint32_t v) {
                    file.write(reinterpret_cast<const uint8_t*>(&v), sizeof(v));
                };
                auto write_le = [&](uint64_t v, uint8_t bytes) {
                    for (uint8_t b = 0; b < bytes; ++b) {
                        const uint8_t byte = static_cast<uint8_t>((v >> (8u * b)) & 0xFFu);
                        file.write(reinterpret_cast<const uint8_t*>(&byte), 1);
                    }
                };

                if(file.write((uint8_t*)&loadedCount, sizeof(loadedCount)) != sizeof(loadedCount)) {
                    eml_debug(0, "❌ Failed to write tree count to: ", unifiedfile_path);
                    file.close();
                    RF_FS_REMOVE(unifiedfile_path);
                    return false;
                }

                // Persist bit widths once per forest
                file.write(reinterpret_cast<const uint8_t*>(&resources.threshold_bits), 1);
                file.write(reinterpret_cast<const uint8_t*>(&resources.feature_bits), 1);
                file.write(reinterpret_cast<const uint8_t*>(&resources.label_bits), 1);
                file.write(reinterpret_cast<const uint8_t*>(&resources.child_bits), 1);
                
                size_t totalBytes = 0;
                
                // Write all trees in sequence with error checking
                uint8_t savedCount = 0;
                for(auto& tree : trees) {
                    if (tree.isLoaded && tree.index != 255 && (tree.leaf_nodes.size() > 0 || tree.branch_kind.size() > 0 || !tree.nodes.empty())) {
                        tree.set_resource(&resources);
                        if ((tree.internal_nodes.size() + tree.mixed_nodes.size() + tree.leaf_nodes.size()) == 0) {
                            (void)tree.convert_to_compact();
                        }
                        // Write tree header
                        if(file.write((uint8_t*)&tree.index, sizeof(tree.index)) != sizeof(tree.index)) {
                            eml_debug(1, "❌ Failed to write tree index: ", tree.index);
                            break;
                        }

                        const uint8_t rootLeaf = tree.root_is_leaf ? 1 : 0;
                        file.write(reinterpret_cast<const uint8_t*>(&rootLeaf), 1);
                        write_u32(static_cast<uint32_t>(tree.root_index));

                        const uint32_t branchCount = static_cast<uint32_t>(tree.branch_kind.size());
                        const uint32_t internalCount = static_cast<uint32_t>(tree.internal_nodes.size());
                        const uint32_t mixedCount = static_cast<uint32_t>(tree.mixed_nodes.size());
                        const uint32_t leafCount = static_cast<uint32_t>(tree.leaf_nodes.size());
                        write_u32(branchCount);
                        write_u32(internalCount);
                        write_u32(mixedCount);
                        write_u32(leafCount);

                        const uint8_t inBits = tree.internal_nodes.get_bits_per_value();
                        const uint8_t mxBits = tree.mixed_nodes.get_bits_per_value();
                        const uint8_t lfBits = tree.leaf_nodes.get_bits_per_value();
                        const uint8_t inBytes = static_cast<uint8_t>((inBits + 7) / 8);
                        const uint8_t mxBytes = static_cast<uint8_t>((mxBits + 7) / 8);
                        const uint8_t lfBytes = static_cast<uint8_t>((lfBits + 7) / 8);
                        file.write(reinterpret_cast<const uint8_t*>(&inBits), 1);
                        file.write(reinterpret_cast<const uint8_t*>(&mxBits), 1);
                        file.write(reinterpret_cast<const uint8_t*>(&lfBits), 1);

                        const uint32_t kindBytes = (branchCount + 7u) / 8u;
                        write_u32(kindBytes);
                        for (uint32_t byteIndex = 0; byteIndex < kindBytes; ++byteIndex) {
                            uint8_t out = 0;
                            const uint32_t base = byteIndex * 8u;
                            for (uint8_t bit = 0; bit < 8; ++bit) {
                                const uint32_t i = base + static_cast<uint32_t>(bit);
                                if (i < branchCount) {
                                    out |= (static_cast<uint8_t>(tree.branch_kind.get(i) & 1u) << bit);
                                }
                            }
                            file.write(reinterpret_cast<const uint8_t*>(&out), 1);
                        }
                        totalBytes += kindBytes;

                        for (uint32_t i = 0; i < internalCount; ++i) {
                            const Internal_node n = tree.internal_nodes.get(static_cast<node_idx_type>(i));
                            write_le(static_cast<uint64_t>(n.packed_data), inBytes);
                            totalBytes += inBytes;
                        }
                        for (uint32_t i = 0; i < mixedCount; ++i) {
                            const Mixed_node n = tree.mixed_nodes.get(static_cast<node_idx_type>(i));
                            write_le(static_cast<uint64_t>(n.packed_data), mxBytes);
                            totalBytes += mxBytes;
                        }
                        for (uint32_t i = 0; i < leafCount; ++i) {
                            const rf_label_type lbl = tree.leaf_nodes.get(static_cast<node_idx_type>(i));
                            write_le(static_cast<uint64_t>(lbl), lfBytes);
                            totalBytes += lfBytes;
                        }

                        savedCount++;
                    }
                }
                file.close();
                
                // Verify file was written correctly
                if(savedCount != loadedCount) {
                    eml_debug_2(1, "❌ Save incomplete: ", savedCount, "/", loadedCount);
                    RF_FS_REMOVE(unifiedfile_path);
                    return false;
                }
                
                // Only clear trees from RAM after successful save
                uint8_t clearedCount = 0;
                for(auto& tree : trees) {
                    if (tree.isLoaded) {
                        tree.clearTree();
                        tree.isLoaded = false;
                        clearedCount++;
                    }
                }
                
                is_loaded = false;
                is_unified = true; // forest always in unified form after first time release
                
                eml_debug_2(1, "✅ Released ", clearedCount, "trees to unified format: ", unifiedfile_path);
                return true;
            }

            void end_training_phase() {
                queue_nodes.clear();
                queue_nodes.shrink_to_fit();
            }

            Rf_tree& operator[](uint8_t index){
                return trees[index];
            }

            node_resource* resource_ptr() {
                return &resources;
            }

            const node_resource* resource_ptr() const {
                return &resources;
            }

            const node_resource& get_resource() const {
                return resources;
            }

            size_t get_total_nodes() const {
                return total_nodes;
            }

            size_t get_total_leaves() const {
                return total_leaves;
            }

            float avg_depth() const {
                return static_cast<float>(total_depths) / config_ptr->num_trees;
            }

            float avg_nodes() const {
                return static_cast<float>(total_nodes) / config_ptr->num_trees;
            }

            float avg_leaves() const {
                return static_cast<float>(total_leaves) / config_ptr->num_trees;
            }

            // Get the number of trees
            size_t size() const {
                if(config_ptr){
                    return config_ptr->num_trees;
                }else{
                    return trees.size();
                }
            }

            uint8_t bits_per_node() const {
                return resources.bits_per_building_node();
            }

            //  model size in ram 
            size_t size_in_ram() const {     
                size_t size = 0;
                size += sizeof(*this);                           
                size += config_ptr->num_trees * sizeof(Rf_tree);    
                // Approximate: internal nodes + mixed nodes + leaf nodes are already packed.
                size += (total_nodes * resources.bits_per_internal_node() + 7) / 8;
                size += predictClass.memory_usage();
                size += queue_nodes.memory_usage();
                return size;
            }

            // Check if container is empty
            bool empty() const {
                return trees.empty();
            }

            // Get queue_nodes for tree building
            vector<NodeToBuild>& getQueueNodes() {
                return queue_nodes;
            }

            void set_to_unified_form() {
                is_unified = true;
            }

            void set_to_individual_form() {
                is_unified = false;
            }

            // Get the maximum depth among all trees
            uint16_t max_depth_tree() const {
                uint16_t maxDepth = 0;
                for (const auto& tree : trees) {
                    uint16_t depth = tree.getTreeDepth();
                    if (depth > maxDepth) {
                        maxDepth = depth;
                    }
                }
                return maxDepth;
            }
    };

    /*
    ------------------------------------------------------------------------------------------------------------------------------
    ------------------------------------------------ RF_PENDING_DATA ------------------------------------------------------------
    ------------------------------------------------------------------------------------------------------------------------------
    */

    class Rf_pending_data{
        static size_t max_infer_logfile_size() {
            return rf_storage_max_infer_log_bytes();
        }
        vector<Rf_sample> pending_samples; // buffer for pending samples
        vector<rf_label_type> actual_labels; // true labels of the samples
        vector<Rf_drift_sample> drift_samples; // concept-drift samples recorded during inference
        uint16_t max_pending_samples; // max number of pending samples in buffer

        // interval between 2 inferences. If after this interval the actual label is not provided, the currently labeled waiting sample will be skipped.
        long unsigned max_wait_time; // max wait time for true label in ms 
        long unsigned last_time_received_actual_label;   
        bool first_label_received = false; // flag to indicate if the first actual label has been received 

        const Rf_base* base_ptr = nullptr; // pointer to base data, used for auto-flush
        Rf_config* config_ptr = nullptr; // pointer to config, used for auto-flush
        eml_quantizer<problem_type::CLASSIFICATION>* quantizer_ptr = nullptr; // optional pointer for drift-aware auto-flush

        inline bool ptr_ready() const {
            return base_ptr != nullptr && config_ptr != nullptr && base_ptr->ready_to_use();
        }

        public:
        Rf_pending_data() {
            init(nullptr, nullptr, nullptr);
        }
        // destructor
        ~Rf_pending_data() {
            base_ptr = nullptr;
            config_ptr = nullptr;
            pending_samples.clear();
            actual_labels.clear();
            drift_samples.clear();
        }

        void init(Rf_base* base, Rf_config* config, eml_quantizer<problem_type::CLASSIFICATION>* quantizer = nullptr){
            base_ptr = base;
            config_ptr = config;
            quantizer_ptr = quantizer;
            pending_samples.clear();
            actual_labels.clear();
            drift_samples.clear();
            set_max_pending_samples(100);
            max_wait_time = 2147483647; // ~24 days
        }

        // add pending sample to buffer, including the label predicted by the model
        void add_pending_sample(const Rf_sample& sample, Rf_data& base_data, bool drift_sample = false, uint16_t drift_feature = 0, float drift_value = 0.0f){
            pending_samples.push_back(sample);
            if (drift_sample) {
                Rf_drift_sample ds;
                ds.pending_index = static_cast<sample_idx_type>(pending_samples.size() - 1);
                ds.feature_index = drift_feature;
                ds.value = drift_value;
                drift_samples.push_back(ds);
            }
            if(pending_samples.size() > max_pending_samples){
                // Auto-flush if parameters are provided
                if(ptr_ready()){
                    if (quantizer_ptr) {
                        flush_pending_data(base_data, *quantizer_ptr);
                    } else {
                        flush_pending_data(base_data);
                    }
                } else {
                    pending_samples.clear();
                    actual_labels.clear();
                    drift_samples.clear();
                }
            }
        }

        void add_actual_label(rf_label_type true_label){
            uint16_t ignore_index = (eml_time_now(MILLISECONDS) - last_time_received_actual_label) / max_wait_time;
            if(!first_label_received){
                ignore_index = 0;
                first_label_received = true;
            }
            while(ignore_index-- > 0) actual_labels.push_back(RF_ERROR_LABEL); // push error label for ignored samples

            // all pending samples have been labeled, ignore this label
            if(actual_labels.size() >= pending_samples.size()) return;

            actual_labels.push_back(true_label);
            last_time_received_actual_label = eml_time_now(MILLISECONDS);
        }

        void set_max_pending_samples(sample_idx_type max_samples){
            max_pending_samples = max_samples;
        }

        void set_max_wait_time(long unsigned wait_time_ms){
            max_wait_time = wait_time_ms;
        }

        // write valid samples to base_data file
        bool write_to_base_data(Rf_data& base_data){
            if(pending_samples.empty()) {
                eml_debug(1, "⚠️ No pending samples to write to base data");
                return false;
            }
            if (!ptr_ready()) {
                eml_debug(1, "❌ Cannot write to base data: data pointers not ready");
                return false;
            }
            // first scan 
            sample_idx_type valid_samples_count = 0;
            b_vector<Rf_sample> valid_samples;
            for(sample_idx_type i = 0; i < pending_samples.size() && i < actual_labels.size(); i++) {
                if(actual_labels[i] < RF_ERROR_LABEL) { // Valid actual label provided
                    valid_samples_count++;
                    Rf_sample sample(pending_samples[i].features, actual_labels[i]);
                    valid_samples.push_back(sample);
                }
            }
            
            if(valid_samples_count == 0) {
                return false; // No valid samples to add
            }

            auto deleted_labels = base_data.addNewData(valid_samples, config_ptr->num_samples);

            // update config
            config_ptr->num_samples += valid_samples_count;

            for(sample_idx_type i = 0; i < pending_samples.size() && i < actual_labels.size(); i++) {
                if(actual_labels[i] < RF_ERROR_LABEL) { // Valid actual label provided
                    config_ptr->samples_per_label[actual_labels[i]]++;
                }
            }

            for(auto& lbl : deleted_labels) {
                if(lbl < RF_ERROR_LABEL && lbl < config_ptr->num_labels && config_ptr->samples_per_label[lbl] > 0) {
                    config_ptr->samples_per_label[lbl]--;
                }
            }
            char data_path[RF_PATH_BUFFER];
            base_ptr->get_base_data_path(data_path);
            eml_debug_2(1, "✅ Added", valid_samples_count, "new samples to base data", data_path);
            return true;
        }

        // Write prediction which given actual label (0 < actual_label < RF_ERROR_LABEL) to the inference log file
        bool write_to_infer_log(){
            if(pending_samples.empty()) return false;
            if(!ptr_ready()){
                eml_debug(1, "❌ Cannot write to inference log: data pointers not ready");
                return false;
            };
            char infer_log_path[RF_PATH_BUFFER];
            base_ptr->get_infer_log_path(infer_log_path);
            if(infer_log_path[0] == '\0') {
                eml_debug(1, "❌ Cannot write to inference log: no base reference for file management");
                return false;
            }
            bool file_exists = RF_FS_EXISTS(infer_log_path);
            uint32_t current_prediction_count = 0;
            
            // If file exists, read current prediction count from header
            if(file_exists) {
                File read_file = RF_FS_OPEN(infer_log_path, RF_FILE_READ);
                if(read_file && read_file.size() >= 8) {
                    uint8_t magic_bytes[4];
                    read_file.read(magic_bytes, 4);
                    // Verify magic number
                    if(magic_bytes[0] == 0x49 && magic_bytes[1] == 0x4E && 
                       magic_bytes[2] == 0x46 && magic_bytes[3] == 0x4C) {
                        read_file.read((uint8_t*)&current_prediction_count, 4);
                    }
                }
                read_file.close();
            }
            
            File file = RF_FS_OPEN(infer_log_path, file_exists ? FILE_APPEND : FILE_WRITE);
            if(!file) {
                eml_debug(1, "❌ Failed to open inference log file: ", infer_log_path);
                return false;
            }
            
            // Write header if new file
            if(!file_exists) {
                // Magic number (4 bytes): 'I', 'N', 'F', 'L' (INFL)
                uint8_t magic_bytes[4] = {0x49, 0x4E, 0x46, 0x4C};
                size_t written = file.write(magic_bytes, 4);
                if(written != 4) {
                    eml_debug(1, "❌ Failed to write magic number to inference log");
                }
                
                // Write initial prediction count (4 bytes)
                uint32_t initial_count = 0;
                written = file.write((uint8_t*)&initial_count, 4);
                if(written != 4) {
                    eml_debug(1, "❌ Failed to write initial prediction count to inference log");
                }
                
                file.flush();
            }
            
            // Collect and write prediction pairs for valid samples
            b_vector<rf_label_type> prediction_pairs;
            uint32_t new_predictions = 0;
            
            for(sample_idx_type i = 0; i < pending_samples.size() && i < actual_labels.size(); i++) {
                if(actual_labels[i] != RF_ERROR_LABEL) { // Valid actual label provided
                    rf_label_type predicted_label = pending_samples[i].label;
                    rf_label_type actual_label = actual_labels[i];
                    
                    // Write predicted_label followed by actual_label
                    prediction_pairs.push_back(predicted_label);
                    prediction_pairs.push_back(actual_label);
                    new_predictions++;
                }
            }
            
            if(!prediction_pairs.empty()) {
                // Write prediction pairs to end of file
                size_t written = file.write(prediction_pairs.data(), prediction_pairs.size());
                if(written != prediction_pairs.size()) {
                    eml_debug_2(1, "❌ Failed to write all prediction pairs to inference log: ", 
                                 written, "/", prediction_pairs.size());
                }
                
                file.flush();
                file.close();
                
                // Update prediction count in header - read entire file and rewrite
                File read_file = RF_FS_OPEN(infer_log_path, RF_FILE_READ);
                if(read_file) {
                    size_t file_size = read_file.size();
                    b_vector<uint8_t> file_data(file_size);
                    read_file.read(file_data.data(), file_size);
                    read_file.close();
                    
                    // Update prediction count in the header (bytes 4-7)
                    uint32_t updated_count = current_prediction_count + new_predictions;
                    memcpy(&file_data[4], &updated_count, 4);
                    
                    // Write back the entire file
                    File write_file = RF_FS_OPEN(infer_log_path, FILE_WRITE);
                    if(write_file) {
                        write_file.write(file_data.data(), file_data.size());
                        write_file.flush();
                        write_file.close();

                        eml_debug_2(1, "✅ Added", new_predictions, "prediction pairs to log: ", updated_count);
                    }
                }
            } else {
                file.close();
            }
            // Trim file if it exceeds max size
            return trim_log_file(infer_log_path);
        }

        // Public method to flush pending data when buffer is full or on demand
        void flush_pending_data(Rf_data& base_data) {
            if(pending_samples.empty()) return;
            
            write_to_base_data(base_data);
            write_to_infer_log();
            
            // Clear buffers after processing
            pending_samples.clear();
            actual_labels.clear();
            drift_samples.clear();
        }

        // Drift-aware flush: update quantizer ranges and create mapping filter before writing.
        void flush_pending_data(Rf_data& base_data, eml_quantizer<problem_type::CLASSIFICATION>& quantizer) {
            if (pending_samples.empty()) {
                return;
            }

            if (!drift_samples.empty()) {
                // Update quantizer and create mapping rule.
                auto& uf = base_data.get_update_filter();
                if (quantizer.apply_concept_drift_update(drift_samples, uf)) {
                    // Apply mapping immediately to the currently-buffered pending samples
                    // (these samples were quantized with the old bins).
                    for (size_t si = 0; si < pending_samples.size(); ++si) {
                        Rf_sample& s = pending_samples[si];
                        for (uint16_t f = 0; f < s.features.size(); ++f) {
                            uint8_t oldBin = static_cast<uint8_t>(s.features[f]);
                            uint8_t newBin = uf.map(f, oldBin);
                            s.features.set(f, newBin);
                        }
                    }
                }
            }

            write_to_base_data(base_data);
            write_to_infer_log();
            pending_samples.clear();
            actual_labels.clear();
            drift_samples.clear();
        }

    private:
        // trim log file if it exceeds the storage-dependent limit
        bool trim_log_file(const char* infer_log_path) {
            if(!RF_FS_EXISTS(infer_log_path)) return false;
            
            File file = RF_FS_OPEN(infer_log_path, RF_FILE_READ);
            if(!file) return false;
            
            size_t file_size = file.size();
            file.close();
            
            const size_t limit = max_infer_logfile_size();
            if(file_size <= limit) return true; // No trimming needed;
            
            // File is too large, trim from the beginning (keep most recent data)
            file = RF_FS_OPEN(infer_log_path, RF_FILE_READ);
            if(!file) return false;
            
            // Read and verify header
            uint8_t magic_bytes[4];
            uint32_t total_predictions;
            
            if(file.read(magic_bytes, 4) != 4 || 
               magic_bytes[0] != 0x49 || magic_bytes[1] != 0x4E || 
               magic_bytes[2] != 0x46 || magic_bytes[3] != 0x4C) {
                file.close();
                eml_debug(1, "❌ Invalid magic number in infer log file: ", infer_log_path);
                return false;
            }
            
            if(file.read((uint8_t*)&total_predictions, 4) != 4) {
                file.close();
                eml_debug(1, "❌ Failed to read prediction count from infer log file: ", infer_log_path);
                return false;
            }
            
            size_t header_size = 8; // magic (4) + prediction_count (4)
            size_t data_size = file_size - header_size;
            size_t prediction_pairs_count = data_size / 2; // Each prediction is 2 bytes (predicted + actual)
            
            size_t max_data_size = limit > header_size ? (limit - header_size) : 0;
            size_t max_pairs_to_keep = max_data_size / 2;
            
            if(prediction_pairs_count <= max_pairs_to_keep) {
                file.close();
                return true; // No trimming needed
            }
            
            // Keep the most recent prediction pairs
            size_t pairs_to_keep = max_pairs_to_keep / 2; // Keep half to allow room for growth
            size_t pairs_to_skip = prediction_pairs_count - pairs_to_keep;
            size_t bytes_to_skip = pairs_to_skip * 2;
            
            // Skip to the position we want to keep from
            file.seek(header_size + bytes_to_skip);
            
            // Read remaining prediction pairs
            size_t remaining_data_size = pairs_to_keep * 2;
            b_vector<uint8_t> remaining_data(remaining_data_size);
            size_t bytes_read = file.read(remaining_data.data(), remaining_data_size);
            file.close();
            
            if(bytes_read != remaining_data_size) {
                eml_debug(1, "❌ Failed to read remaining data from infer log file: ", infer_log_path);
                return false;
            }
            
            // Rewrite file with header and trimmed data
            file = RF_FS_OPEN(infer_log_path, FILE_WRITE);
            if(!file) {
                eml_debug(1, "❌ Failed to reopen log file for writing: ", infer_log_path);
                return false;
            }
            
            // Write header with updated prediction count
            file.write(magic_bytes, 4);
            uint32_t new_prediction_count = pairs_to_keep;
            file.write((uint8_t*)&new_prediction_count, 4);
            
            // Write remaining prediction pairs
            file.write(remaining_data.data(), remaining_data.size());
            file.flush();
            file.close();
            return true;
        }

    };

    /*
    ------------------------------------------------------------------------------------------------------------------
    ------------------------------------------------ RF_LOGGER -------------------------------------------------------
    ------------------------------------------------------------------------------------------------------------------
    */

    using Rf_logger = eml_logger_t<Rf_base>;

}   // namespace mcu