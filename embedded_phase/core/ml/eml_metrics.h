#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <limits>
#include <cctype>
#include <cstring>

#include "../base/eml_base.h"
#include "eml_samples.h"

/**
 * @file eml_metrics.h
 * @brief Comprehensive metrics calculation framework for ML models on MCU
 * 
 * This file provides metrics calculation for both classification and regression tasks.
 * 
 * CLASSIFICATION METRICS:
 * - Confusion matrix-based: accuracy, precision, recall, f1_score
 * - Probability-based: logloss (binary), mlogloss (multi-class)
 * 
 * REGRESSION METRICS:
 * - MAE (Mean Absolute Error)
 * - MSE (Mean Squared Error)
 * - RMSE (Root Mean Squared Error)
 * - MAPE (Mean Absolute Percentage Error)
 * - R2 (Coefficient of Determination)
 * 
 * UPDATE METHODS:
 * Classification:
 * - update(actual_label, predicted_label) - for hard predictions
 * - update_logloss(actual_label, predicted_prob) - for binary log loss
 * - update_mlogloss(actual_label, probs[]) - for multi-class log loss
 * - update_with_probabilities(actual_label, probs[]) - updates both confusion matrix and log loss
 * 
 * Regression:
 * - update(actual, predicted) - updates all regression metrics simultaneously
 * 
 * CALCULATION METHODS:
 * All calculation methods are on-demand (lazy evaluation).
 * Use calculate_metric(eval_metric) for specific metric calculation.
 */

namespace eml {

    // -------------------------------------------------------------------------
    // Eval metric selection 
    // -------------------------------------------------------------------------
    /**
     * @brief Unified evaluation metric enum for the entire framework.
     * 
     * Classification metrics:
     * - ACCURACY, PRECISION, RECALL, F1_SCORE: Confusion matrix-based
     * - LOGLOSS: Binary log loss
     * - MLOGLOSS: Multi-class log loss
     * 
     * Regression metrics:
     * - MAE, MSE, RMSE, MAPE, R2
     */
    enum class eval_metric : uint8_t {
        // Classification metrics
        ACCURACY = 0,
        PRECISION = 1,
        RECALL = 2,
        F1_SCORE = 3,
        LOGLOSS = 4,
        MLOGLOSS = 5,
        // Regression metrics
        MAE = 6,
        MSE = 7,
        RMSE = 8,
        MAPE = 9,
        R2 = 10,
        // Anomaly detection metrics (isolation forest)
        ANOMALY_SCORE = 11,
        ROC_AUC = 12,
        PRC_AUC = 13,
        AVERAGE_PRECISION = 14,
        FPR = 15,
        TPR = 16,
        // Unknown
        UNKNOWN = 255
    };

    inline const char* evalMetricToString(eval_metric metric) {
        switch (metric) {
            case eval_metric::ACCURACY:  return "accuracy";
            case eval_metric::PRECISION: return "precision";
            case eval_metric::RECALL:    return "recall";
            case eval_metric::F1_SCORE:  return "f1_score";
            case eval_metric::LOGLOSS:   return "logloss";
            case eval_metric::MLOGLOSS:  return "mlogloss";
            case eval_metric::MAE:       return "mae";
            case eval_metric::MSE:       return "mse";
            case eval_metric::RMSE:      return "rmse";
            case eval_metric::MAPE:      return "mape";
            case eval_metric::R2:            return "r2";
            case eval_metric::ANOMALY_SCORE:  return "anomaly_score";
            case eval_metric::ROC_AUC:       return "roc_auc";
            case eval_metric::PRC_AUC:       return "prc_auc";
            case eval_metric::AVERAGE_PRECISION: return "average_precision";
            case eval_metric::FPR:           return "fpr";
            case eval_metric::TPR:           return "tpr";
            default: return "unknown";
        }
    }

    inline bool isSpaceOrQuote(char c) {
        return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '"';
    }

    inline void normalizeToken(const char* in, char* out, size_t out_len) {
        if (!out || out_len == 0) return;
        out[0] = '\0';
        if (!in) return;

        while (*in && isSpaceOrQuote(*in)) ++in;

        const char* end = in;
        while (*end) ++end;
        while (end > in && isSpaceOrQuote(*(end - 1))) --end;

        size_t len = static_cast<size_t>(end - in);
        if (len >= out_len) len = out_len - 1;

        for (size_t i = 0; i < len; ++i) {
            out[i] = static_cast<char>(::tolower(static_cast<unsigned char>(in[i])));
        }
        out[len] = '\0';
    }

    inline eval_metric stringToEvalMetric(const char* str) {
        char lower[24];
        normalizeToken(str, lower, sizeof(lower));

        if (lower[0] == '\0') return eval_metric::UNKNOWN;
        if (strcmp(lower, "accuracy") == 0)  return eval_metric::ACCURACY;
        if (strcmp(lower, "precision") == 0) return eval_metric::PRECISION;
        if (strcmp(lower, "recall") == 0)    return eval_metric::RECALL;
        if (strcmp(lower, "f1_score") == 0 || strcmp(lower, "f1") == 0) return eval_metric::F1_SCORE;
        if (strcmp(lower, "logloss") == 0)   return eval_metric::LOGLOSS;
        if (strcmp(lower, "mlogloss") == 0)  return eval_metric::MLOGLOSS;
        if (strcmp(lower, "mae") == 0)       return eval_metric::MAE;
        if (strcmp(lower, "mse") == 0)       return eval_metric::MSE;
        if (strcmp(lower, "rmse") == 0)      return eval_metric::RMSE;
        if (strcmp(lower, "mape") == 0)      return eval_metric::MAPE;
        if (strcmp(lower, "r2") == 0)                                        return eval_metric::R2;
        if (strcmp(lower, "anomaly_score") == 0 || strcmp(lower, "anomaly") == 0) return eval_metric::ANOMALY_SCORE;
        if (strcmp(lower, "roc_auc") == 0 || strcmp(lower, "auc") == 0) return eval_metric::ROC_AUC;
        if (strcmp(lower, "prc_auc") == 0 || strcmp(lower, "auprc") == 0) return eval_metric::PRC_AUC;
        if (strcmp(lower, "average_precision") == 0 || strcmp(lower, "ap") == 0) return eval_metric::AVERAGE_PRECISION;
        if (strcmp(lower, "fpr") == 0) return eval_metric::FPR;
        if (strcmp(lower, "tpr") == 0) return eval_metric::TPR;
        return eval_metric::UNKNOWN;
    }

    inline bool isClassificationMetric(eval_metric metric) {
        return metric == eval_metric::ACCURACY ||
               metric == eval_metric::PRECISION ||
               metric == eval_metric::RECALL ||
               metric == eval_metric::F1_SCORE ||
               metric == eval_metric::LOGLOSS ||
               metric == eval_metric::MLOGLOSS ||
               metric == eval_metric::ROC_AUC ||
               metric == eval_metric::PRC_AUC ||
               metric == eval_metric::AVERAGE_PRECISION;
    }

    inline bool isRegressionMetric(eval_metric metric) {
        return metric == eval_metric::MAE ||
               metric == eval_metric::MSE ||
               metric == eval_metric::RMSE ||
               metric == eval_metric::MAPE ||
               metric == eval_metric::R2;
    }

    inline bool lowerIsBetter(eval_metric metric) {
        return metric == eval_metric::LOGLOSS ||
               metric == eval_metric::MLOGLOSS ||
               metric == eval_metric::MAE ||
               metric == eval_metric::MSE ||
               metric == eval_metric::RMSE ||
               metric == eval_metric::MAPE ||
               metric == eval_metric::FPR;
    }

    inline eval_metric getDefaultMetric(problem_type type) {
        switch (type) {
            case problem_type::CLASSIFICATION: return eval_metric::ACCURACY;
            case problem_type::REGRESSION:     return eval_metric::RMSE;
            case problem_type::ISOLATION:      return eval_metric::ROC_AUC;
            default: return eval_metric::ACCURACY;
        }
    }

    inline const char* getAvailableMetrics(problem_type type) {
        if (type == problem_type::CLASSIFICATION) {
            return "accuracy, precision, recall, f1_score, logloss, mlogloss, roc_auc, prc_auc, ap";
        }
        if (type == problem_type::REGRESSION) {
            return "mae, mse, rmse, mape, r2";
        }
        if (type == problem_type::ISOLATION) {
            return "anomaly_score, accuracy, precision, recall, f1_score, roc_auc, prc_auc, ap, fpr, tpr";
        }
        return "accuracy";
    }

    enum class eml_average : uint8_t {
        MICRO = 0,
        MACRO = 1,
        WEIGHTED = 2
    };

    template<problem_type ProblemType>
    class eml_metrics_calc_t;

    // ---------------- Classification metrics ----------------
    template<>
    class eml_metrics_calc_t<problem_type::CLASSIFICATION> {
    public:
        using label_type = eml_label_t<problem_type::CLASSIFICATION>;
        using count_type = sample_idx_type;
        using real_t = float;

    private:
        // ===== Confusion matrix attributes (for accuracy, precision, recall, f1) =====
        vector<count_type> tp;  // True positives per class
        vector<count_type> fp;  // False positives per class
        vector<count_type> fn;  // False negatives per class

        count_type total_predict = 0;
        count_type correct_predict = 0;

        // ===== Log loss attributes (for logloss, mlogloss) =====
        real_t sum_log_loss = 0.0f;         // Accumulated log loss
        count_type log_loss_count = 0;      // Number of samples for log loss
        
        // ===== Configuration =====
        label_type num_labels = 0;
        eval_metric metric = eval_metric::ACCURACY;
        float log_loss_epsilon = 1e-7f;     // Small value to prevent log(0)

        void ensure_buffers() {
            if (tp.size() != num_labels) {
                tp.clear(); tp.reserve(num_labels);
                for (label_type i = 0; i < num_labels; ++i) tp.push_back(0);
            }
            if (fp.size() != num_labels) {
                fp.clear(); fp.reserve(num_labels);
                for (label_type i = 0; i < num_labels; ++i) fp.push_back(0);
            }
            if (fn.size() != num_labels) {
                fn.clear(); fn.reserve(num_labels);
                for (label_type i = 0; i < num_labels; ++i) fn.push_back(0);
            }
        }

        inline float safe_div(float num, float den) const {
            return den == 0.0f ? 0.0f : (num / den);
        }

        inline count_type support(label_type label) const {
            return tp[label] + fn[label];
        }

        inline float clip_probability(float prob) const {
            if (prob < log_loss_epsilon) return log_loss_epsilon;
            if (prob > 1.0f - log_loss_epsilon) return 1.0f - log_loss_epsilon;
            return prob;
        }

    public:
        eml_metrics_calc_t() = default;

        eml_metrics_calc_t(label_type num_labels, eval_metric metric = eval_metric::ACCURACY)
            : num_labels(num_labels), metric(metric) {
            ensure_buffers();
            reset();
        }

        void init(label_type labels, eval_metric m = eval_metric::ACCURACY) {
            num_labels = labels;
            metric = m;
            ensure_buffers();
            reset();
        }

        void set_metric(eval_metric m) { metric = m; }
        eval_metric get_metric() const { return metric; }

        void set_log_loss_epsilon(float eps) { log_loss_epsilon = (eps > 0.0f ? eps : 1e-7f); }
        float get_log_loss_epsilon() const { return log_loss_epsilon; }

        label_type labels() const { return num_labels; }
        count_type total() const { return total_predict; }
        count_type correct() const { return correct_predict; }

        void reset() {
            // Reset confusion matrix attributes
            total_predict = 0;
            correct_predict = 0;
            ensure_buffers();
            if (num_labels > 0) {
                tp.fill(0);
                fp.fill(0);
                fn.fill(0);
            }

            // Reset log loss attributes
            sum_log_loss = 0.0f;
            log_loss_count = 0;
        }

        // ===== Update methods for confusion matrix-based metrics (accuracy, precision, recall, f1) =====
        void update(label_type actual_label, label_type predicted_label) {
            if (num_labels == 0) return;
            if (actual_label >= num_labels || predicted_label >= num_labels) return;

            total_predict++;
            if (predicted_label == actual_label) {
                correct_predict++;
                tp[actual_label]++;
            } else {
                fn[actual_label]++;
                fp[predicted_label]++;
            }
        }

        void update_prediction(label_type actual_label, label_type predicted_label) {
            update(actual_label, predicted_label);
        }

        void update_batch(const label_type* actual, const label_type* predicted, size_t count) {
            if (!actual || !predicted || count == 0) return;
            for (size_t i = 0; i < count; ++i) {
                update(actual[i], predicted[i]);
            }
        }

        // ===== Update methods for log loss metrics (logloss, mlogloss) =====
        
        // Binary log loss: update with actual label (0 or 1) and probability for class 1
        void update_logloss(label_type actual_label, float predicted_prob) {
            if (num_labels != 2) return;  // Binary classification only
            if (actual_label >= 2) return;

            const float prob = clip_probability(predicted_prob);
            const float loss = actual_label == 1 
                ? -std::log(prob) 
                : -std::log(1.0f - prob);
            
            sum_log_loss += loss;
            log_loss_count++;
        }

        // Multi-class log loss: update with actual label and probability distribution
        // probs should be an array of probabilities for each class (length = num_labels)
        void update_mlogloss(label_type actual_label, const float* probs) {
            if (num_labels == 0 || !probs) return;
            if (actual_label >= num_labels) return;

            const float prob = clip_probability(probs[actual_label]);
            const float loss = -std::log(prob);
            
            sum_log_loss += loss;
            log_loss_count++;
        }

        // Combined update: updates both confusion matrix and log loss
        // For models that output probabilities, this is the most efficient
        void update_with_probabilities(label_type actual_label, const float* probs) {
            if (num_labels == 0 || !probs) return;
            if (actual_label >= num_labels) return;

            // Find predicted class (argmax)
            label_type predicted_label = 0;
            float max_prob = probs[0];
            for (label_type i = 1; i < num_labels; ++i) {
                if (probs[i] > max_prob) {
                    max_prob = probs[i];
                    predicted_label = i;
                }
            }

            // Update confusion matrix
            update(actual_label, predicted_label);

            // Update log loss
            update_mlogloss(actual_label, probs);
        }

        void update_batch_with_probabilities(const label_type* actual, const float* const* probs, size_t count) {
            if (!actual || !probs || count == 0) return;
            for (size_t i = 0; i < count; ++i) {
                update_with_probabilities(actual[i], probs[i]);
            }
        }

        bool merge(const eml_metrics_calc_t& other) {
            if (other.num_labels != num_labels) return false;
            
            // Merge confusion matrix attributes
            total_predict += other.total_predict;
            correct_predict += other.correct_predict;
            for (label_type i = 0; i < num_labels; ++i) {
                tp[i] += other.tp[i];
                fp[i] += other.fp[i];
                fn[i] += other.fn[i];
            }

            // Merge log loss attributes
            sum_log_loss += other.sum_log_loss;
            log_loss_count += other.log_loss_count;

            return true;
        }

        // ===== Calculation methods (on-demand) =====

        // --- Confusion matrix-based metrics ---
        float accuracy() const {
            return safe_div(static_cast<float>(correct_predict), static_cast<float>(total_predict));
        }

        float precision(label_type label) const {
            if (label >= num_labels) return 0.0f;
            return safe_div(static_cast<float>(tp[label]), static_cast<float>(tp[label] + fp[label]));
        }

        float recall(label_type label) const {
            if (label >= num_labels) return 0.0f;
            return safe_div(static_cast<float>(tp[label]), static_cast<float>(tp[label] + fn[label]));
        }

        float f1(label_type label) const {
            const float p = precision(label);
            const float r = recall(label);
            return safe_div(2.0f * p * r, p + r);
        }

        float precision(eml_average avg) const {
            if (num_labels == 0) return 0.0f;
            if (avg == eml_average::MICRO) {
                count_type tp_sum = 0, fp_sum = 0;
                for (label_type i = 0; i < num_labels; ++i) {
                    tp_sum += tp[i];
                    fp_sum += fp[i];
                }
                return safe_div(static_cast<float>(tp_sum), static_cast<float>(tp_sum + fp_sum));
            }

            float acc = 0.0f;
            count_type denom = 0;
            for (label_type i = 0; i < num_labels; ++i) {
                const count_type s = support(i);
                const float p = precision(i);
                if (avg == eml_average::WEIGHTED) {
                    acc += p * static_cast<float>(s);
                    denom += s;
                } else if (tp[i] + fp[i] > 0) {
                    acc += p;
                    denom++;
                }
            }
            return denom == 0 ? 0.0f : acc / static_cast<float>(denom);
        }

        float recall(eml_average avg) const {
            if (num_labels == 0) return 0.0f;
            if (avg == eml_average::MICRO) {
                count_type tp_sum = 0, fn_sum = 0;
                for (label_type i = 0; i < num_labels; ++i) {
                    tp_sum += tp[i];
                    fn_sum += fn[i];
                }
                return safe_div(static_cast<float>(tp_sum), static_cast<float>(tp_sum + fn_sum));
            }

            float acc = 0.0f;
            count_type denom = 0;
            for (label_type i = 0; i < num_labels; ++i) {
                const count_type s = support(i);
                const float r = recall(i);
                if (avg == eml_average::WEIGHTED) {
                    acc += r * static_cast<float>(s);
                    denom += s;
                } else if (tp[i] + fn[i] > 0) {
                    acc += r;
                    denom++;
                }
            }
            return denom == 0 ? 0.0f : acc / static_cast<float>(denom);
        }

        float f1(eml_average avg) const {
            if (num_labels == 0) return 0.0f;
            if (avg == eml_average::MICRO) {
                count_type tp_sum = 0, fp_sum = 0, fn_sum = 0;
                for (label_type i = 0; i < num_labels; ++i) {
                    tp_sum += tp[i];
                    fp_sum += fp[i];
                    fn_sum += fn[i];
                }
                return safe_div(2.0f * static_cast<float>(tp_sum),
                                static_cast<float>(2 * tp_sum + fp_sum + fn_sum));
            }

            float acc = 0.0f;
            count_type denom = 0;
            for (label_type i = 0; i < num_labels; ++i) {
                const count_type s = support(i);
                const float f = f1(i);
                if (avg == eml_average::WEIGHTED) {
                    acc += f * static_cast<float>(s);
                    denom += s;
                } else if (tp[i] + fp[i] > 0 || tp[i] + fn[i] > 0) {
                    acc += f;
                    denom++;
                }
            }
            return denom == 0 ? 0.0f : acc / static_cast<float>(denom);
        }

        // --- Log loss-based metrics ---
        float logloss() const {
            if (num_labels != 2) return 0.0f;  // Binary classification only
            return safe_div(sum_log_loss, static_cast<float>(log_loss_count));
        }

        float mlogloss() const {
            return safe_div(sum_log_loss, static_cast<float>(log_loss_count));
        }

        count_type log_loss_samples() const {
            return log_loss_count;
        }

        // ===== Helper methods to get per-class metrics =====
        vector<pair<label_type, float>> get_precisions() const {
            vector<pair<label_type, float>> precisions;
            precisions.reserve(num_labels);
            for (label_type label = 0; label < num_labels; ++label) {
                precisions.push_back(make_pair(label, precision(label)));
            }
            return precisions;
        }

        vector<pair<label_type, float>> get_recalls() const {
            vector<pair<label_type, float>> recalls;
            recalls.reserve(num_labels);
            for (label_type label = 0; label < num_labels; ++label) {
                recalls.push_back(make_pair(label, recall(label)));
            }
            return recalls;
        }

        vector<pair<label_type, float>> get_f1_scores() const {
            vector<pair<label_type, float>> f1s;
            f1s.reserve(num_labels);
            for (label_type label = 0; label < num_labels; ++label) {
                f1s.push_back(make_pair(label, f1(label)));
            }
            return f1s;
        }

        vector<pair<label_type, float>> get_accuracies() const {
            vector<pair<label_type, float>> accuracies;
            accuracies.reserve(num_labels);
            const float acc = accuracy();
            for (label_type label = 0; label < num_labels; ++label) {
                accuracies.push_back(make_pair(label, acc));
            }
            return accuracies;
        }

        // ===== Unified score calculation =====
        float calculate_score() const {
            return calculate_metric(metric);
        }

        // Calculate specific metric by enum
        float calculate_metric(eval_metric metric) const {
            switch (metric) {
                case eval_metric::ACCURACY:  return accuracy();
                case eval_metric::PRECISION: return precision(eml_average::MACRO);
                case eval_metric::RECALL:    return recall(eml_average::MACRO);
                case eval_metric::F1_SCORE:  return f1(eml_average::MACRO);
                case eval_metric::LOGLOSS:   return logloss();
                case eval_metric::MLOGLOSS:  return mlogloss();
                default: return 0.0f;
            }
        }

        // ===== Accessors =====

        const vector<count_type>& true_positives() const { return tp; }
        const vector<count_type>& false_positives() const { return fp; }
        const vector<count_type>& false_negatives() const { return fn; }

        size_t memory_usage() const {
            size_t usage = sizeof(total_predict) + sizeof(correct_predict) + sizeof(num_labels) + sizeof(metric);
            usage += sizeof(sum_log_loss) + sizeof(log_loss_count) + sizeof(log_loss_epsilon);
            usage += tp.size() * sizeof(count_type);
            usage += fp.size() * sizeof(count_type);
            usage += fn.size() * sizeof(count_type);
            return usage;
        }
    };

    // ---------------- Regression metrics ----------------
    template<>
    class eml_metrics_calc_t<problem_type::REGRESSION> {
    public:
        using count_type = sample_idx_type;
        using real_t = float;   // change to double if using 64-bit cpus or higher precision is needed

    private:
        // ===== Regression attributes (for MAE, MSE, RMSE, MAPE, R2) =====
        count_type total_count = 0;
        real_t sum_abs_error = 0.0;        // For MAE
        real_t sum_sq_error = 0.0;         // For MSE, RMSE
        real_t sum_error = 0.0;            // For mean error (bias)
        real_t sum_actual = 0.0;           // For R2, mean calculation
        real_t sum_actual_sq = 0.0;        // For R2 (total sum of squares)
        real_t sum_abs_pct_error = 0.0;    // For MAPE
        real_t max_abs_error = 0.0;        // For max error

        // ===== Configuration =====
        float mape_epsilon = 1e-6f;        // Small value to prevent division by zero in MAPE

        inline real_t safe_div(real_t num, real_t den) const {
            return den == 0.0 ? 0.0 : (num / den);
        }

    public:
        eml_metrics_calc_t() = default;

        void reset() {
            // Reset all regression attributes
            total_count = 0;
            sum_abs_error = 0.0;
            sum_sq_error = 0.0;
            sum_error = 0.0;
            sum_actual = 0.0;
            sum_actual_sq = 0.0;
            sum_abs_pct_error = 0.0;
            max_abs_error = 0.0;
        }

        void set_mape_epsilon(float eps) { mape_epsilon = (eps > 0.0f ? eps : 1e-6f); }
        float get_mape_epsilon() const { return mape_epsilon; }

        // ===== Update methods for regression metrics =====
        void update(float actual, float predicted) {
            const real_t err = static_cast<real_t>(predicted) - static_cast<real_t>(actual);
            const real_t abs_err = std::fabs(err);
            
            // Update all accumulators
            total_count++;
            sum_error += err;                                   // For mean error
            sum_abs_error += abs_err;                           // For MAE
            sum_sq_error += err * err;                          // For MSE, RMSE
            sum_actual += actual;                               // For R2
            sum_actual_sq += static_cast<real_t>(actual) * static_cast<real_t>(actual);  // For R2
            
            // Update max error
            if (abs_err > max_abs_error) {
                max_abs_error = abs_err;
            }
            
            // Update MAPE accumulator
            const real_t denom = std::fabs(static_cast<real_t>(actual)) + static_cast<real_t>(mape_epsilon);
            sum_abs_pct_error += abs_err / denom;
        }

        void update_batch(const float* actual, const float* predicted, size_t count) {
            if (!actual || !predicted || count == 0) return;
            for (size_t i = 0; i < count; ++i) {
                update(actual[i], predicted[i]);
            }
        }

        bool merge(const eml_metrics_calc_t& other) {
            // Merge all regression attributes
            total_count += other.total_count;
            sum_abs_error += other.sum_abs_error;
            sum_sq_error += other.sum_sq_error;
            sum_error += other.sum_error;
            sum_actual += other.sum_actual;
            sum_actual_sq += other.sum_actual_sq;
            sum_abs_pct_error += other.sum_abs_pct_error;
            if (other.max_abs_error > max_abs_error) {
                max_abs_error = other.max_abs_error;
            }
            return true;
        }

        count_type count() const { return total_count; }

        // ===== Calculation methods (on-demand) =====
        
        // Mean Absolute Error
        real_t mean_absolute_error() const { 
            return safe_div(sum_abs_error, static_cast<real_t>(total_count)); 
        }
        
        real_t mae() const { return mean_absolute_error(); }

        // Mean Squared Error
        real_t mean_squared_error() const { 
            return safe_div(sum_sq_error, static_cast<real_t>(total_count)); 
        }
        
        real_t mse() const { return mean_squared_error(); }

        // Root Mean Squared Error
        real_t root_mean_squared_error() const { 
            return std::sqrt(mean_squared_error()); 
        }
        
        real_t rmse() const { return root_mean_squared_error(); }

        // Mean Error (bias)
        real_t mean_error() const { 
            return safe_div(sum_error, static_cast<real_t>(total_count)); 
        }
        
        // Maximum Absolute Error
        real_t max_error() const { return max_abs_error; }

        // Mean Absolute Percentage Error
        real_t mean_absolute_percentage_error() const {
            return safe_div(sum_abs_pct_error, static_cast<real_t>(total_count));
        }
        
        real_t mape() const { return mean_absolute_percentage_error(); }

        // R² Score (coefficient of determination)
        real_t r2_score() const {
            if (total_count == 0) return 0.0;
            const real_t mean = sum_actual / static_cast<real_t>(total_count);
            const real_t ss_tot = sum_actual_sq - (sum_actual * mean);
            if (ss_tot <= 0.0) return 0.0;
            return 1.0 - (sum_sq_error / ss_tot);
        }
        
        real_t r2() const { return r2_score(); }

        // Calculate specific metric by enum
        real_t calculate_metric(eval_metric metric) const {
            switch (metric) {
                case eval_metric::MAE:   return mae();
                case eval_metric::MSE:   return mse();
                case eval_metric::RMSE:  return rmse();
                case eval_metric::MAPE:  return mape();
                case eval_metric::R2:    return r2();
                default: return 0.0;
            }
        }

        size_t memory_usage() const {
            return sizeof(*this);
        }
    };

    // ---------------- Isolation forest / anomaly detection metrics ----------------
    template<>
    class eml_metrics_calc_t<problem_type::ISOLATION> {
    public:
        using count_type = sample_idx_type;
        using real_t = float;

    private:
        // Binary detection counters (treat anomaly=true as positive class)
        count_type tp_ = 0;   // true anomalies detected as anomalies
        count_type fp_ = 0;   // normals detected as anomalies
        count_type tn_ = 0;   // normals detected as normals
        count_type fn_ = 0;   // true anomalies detected as normals

        // Anomaly score distribution tracking
        real_t sum_scores_  = 0.0f;
        real_t min_score_   = std::numeric_limits<real_t>::max();
        real_t max_score_   = std::numeric_limits<real_t>::lowest();
        count_type score_count_ = 0;

        // Score tracking for curve-based metrics (AUC, AP, etc.)
        struct sample_score {
            real_t score;
            bool is_anomaly;
            
            bool operator<(const sample_score& other) const {
                return score < other.score;
            }
        };
        vector<sample_score> scores_cache_;

        eval_metric metric_ = eval_metric::ROC_AUC;

        inline real_t safe_div(real_t num, real_t den) const {
            return den == 0.0f ? 0.0f : (num / den);
        }

    public:
        eml_metrics_calc_t() = default;

        void init(eval_metric m = eval_metric::ROC_AUC) {
            metric_ = m;
            reset();
        }

        void set_metric(eval_metric m) { metric_ = m; }
        eval_metric get_metric() const { return metric_; }

        void reset() {
            tp_ = fp_ = tn_ = fn_ = 0;
            sum_scores_  = 0.0f;
            min_score_   = std::numeric_limits<real_t>::max();
            max_score_   = std::numeric_limits<real_t>::lowest();
            score_count_ = 0;
            scores_cache_.clear();
        }

        // Update with binary ground truth (actual_is_anomaly=true means truly anomalous sample)
        // Note: For Isolation Forest binary prediction, thresholding is needed beforehand.
        void update(bool actual_is_anomaly, bool predicted_is_anomaly) {
            if (actual_is_anomaly  && predicted_is_anomaly)  tp_++;
            else if (!actual_is_anomaly && predicted_is_anomaly)  fp_++;
            else if (!actual_is_anomaly && !predicted_is_anomaly) tn_++;
            else                                                   fn_++;
        }

        // Update with anomaly score + ground truth
        // score: should be converted such that HIGHER score means MORE anomalous
        // (For isolation forest, often -anomaly_score is used)
        void update_score(real_t anomaly_score, bool actual_is_anomaly) {
            sum_scores_ += anomaly_score;
            if (anomaly_score < min_score_) min_score_ = anomaly_score;
            if (anomaly_score > max_score_) max_score_ = anomaly_score;
            score_count_++;
            scores_cache_.push_back({anomaly_score, actual_is_anomaly});
        }

        // Combined update: ground truth binary prediction + anomaly score
        void update(bool actual_is_anomaly, bool predicted_is_anomaly, real_t anomaly_score) {
            update(actual_is_anomaly, predicted_is_anomaly);
            update_score(anomaly_score, actual_is_anomaly);
        }

        void update_batch(const bool* actual, const bool* predicted, size_t count) {
            if (!actual || !predicted || count == 0) return;
            for (size_t i = 0; i < count; ++i) update(actual[i], predicted[i]);
        }

        void update_batch_scores(const bool* actual, const real_t* scores, size_t count) {
            if (!actual || !scores || count == 0) return;
            scores_cache_.reserve(scores_cache_.size() + count);
            for (size_t i = 0; i < count; ++i) update_score(scores[i], actual[i]);
        }

        bool merge(const eml_metrics_calc_t& other) {
            tp_ += other.tp_; fp_ += other.fp_;
            tn_ += other.tn_; fn_ += other.fn_;
            sum_scores_ += other.sum_scores_;
            if (other.score_count_ > 0) {
                if (other.min_score_ < min_score_) min_score_ = other.min_score_;
                if (other.max_score_ > max_score_) max_score_ = other.max_score_;
            }
            score_count_ += other.score_count_;
            
            // Merge cache
            scores_cache_.reserve(scores_cache_.size() + other.scores_cache_.size());
            for (const auto& s : other.scores_cache_) scores_cache_.push_back(s);
            
            return true;
        }

        // ===== Counters =====
        count_type total()        const { return tp_ + fp_ + tn_ + fn_; }
        count_type score_samples() const { return score_count_; }
        count_type true_positives()  const { return tp_; }
        count_type false_positives() const { return fp_; }
        count_type true_negatives()  const { return tn_; }
        count_type false_negatives() const { return fn_; }

        // ===== Detection metrics (Point-based) =====

        // Accuracy: (TP + TN) / total
        real_t accuracy() const {
            return safe_div(static_cast<real_t>(tp_ + tn_), static_cast<real_t>(total()));
        }

        // True Positive Rate = sensitivity = recall
        real_t true_positive_rate() const {
            return safe_div(static_cast<real_t>(tp_), static_cast<real_t>(tp_ + fn_));
        }
        real_t recall()      const { return true_positive_rate(); }
        real_t tpr()         const { return true_positive_rate(); }
        real_t sensitivity() const { return true_positive_rate(); }

        // False Positive Rate = FP / (FP + TN)
        real_t false_positive_rate() const {
            return safe_div(static_cast<real_t>(fp_), static_cast<real_t>(fp_ + tn_));
        }
        real_t fpr() const { return false_positive_rate(); }

        // Specificity = TN / (TN + FP)
        real_t specificity() const {
            return safe_div(static_cast<real_t>(tn_), static_cast<real_t>(tn_ + fp_));
        }

        // Precision = TP / (TP + FP)
        real_t precision() const {
            return safe_div(static_cast<real_t>(tp_), static_cast<real_t>(tp_ + fp_));
        }

        // F1 = 2 * precision * recall / (precision + recall)
        real_t f1() const {
            const real_t p = precision();
            const real_t r = recall();
            return safe_div(2.0f * p * r, p + r);
        }

        // ===== Curve-based metrics (AUC, AP) =====
        
        real_t roc_auc() const {
            if (scores_cache_.empty()) return 0.5f;
            
            // Collect labels and scores
            vector<sample_score> sorted = scores_cache_;
            std::sort(sorted.begin(), sorted.end());
            
            size_t n = sorted.size();
            size_t n_pos = 0;
            for (const auto& s : sorted) if (s.is_anomaly) n_pos++;
            
            size_t n_neg = n - n_pos;
            if (n_pos == 0 || n_neg == 0) return 0.5f;

            double rank_sum_pos = 0.0;
            size_t i = 0;
            while (i < n) {
                size_t j = i + 1;
                while (j < n && sorted[j].score == sorted[i].score) j++;
                
                // Average rank for ties: (rank_start + rank_end) / 2
                // ranks are 1-based, so i+1 to j
                double avg_rank = 0.5 * (static_cast<double>(i + 1) + static_cast<double>(j));
                for (size_t k = i; k < j; k++) {
                    if (sorted[k].is_anomaly) rank_sum_pos += avg_rank;
                }
                i = j;
            }

            double u = rank_sum_pos - (static_cast<double>(n_pos) * (n_pos + 1) * 0.5);
            return static_cast<real_t>(u / (static_cast<double>(n_pos) * n_neg));
        }

        real_t average_precision() const {
            if (scores_cache_.empty()) return 0.0f;
            
            vector<sample_score> sorted = scores_cache_;
            // Sort by score descending for precision-recall integration
            std::sort(sorted.begin(), sorted.end(), [](const sample_score& a, const sample_score& b) {
                return a.score > b.score;
            });
            
            size_t n_pos = 0;
            for (const auto& s : sorted) if (s.is_anomaly) n_pos++;
            if (n_pos == 0) return 0.0f;

            double ap = 0.0;
            size_t current_tp = 0;
            size_t current_fp = 0;
            
            for (size_t i = 0; i < sorted.size(); i++) {
                if (sorted[i].is_anomaly) {
                    current_tp++;
                    double precision = static_cast<double>(current_tp) / (current_tp + current_fp);
                    ap += precision;
                } else {
                    current_fp++;
                }
            }
            
            return static_cast<real_t>(ap / n_pos);
        }

        real_t prc_auc() const {
            return average_precision();
        }

        // Returns vector of {Recall, Precision} pairs
        vector<pair<real_t, real_t>> precision_recall_curve() const {
            if (scores_cache_.empty()) return {};
            
            vector<sample_score> sorted = scores_cache_;
            std::sort(sorted.begin(), sorted.end(), [](const sample_score& a, const sample_score& b) {
                return a.score > b.score;
            });
            
            size_t n_pos = 0;
            for (const auto& s : sorted) if (s.is_anomaly) n_pos++;
            if (n_pos == 0) {
                vector<pair<real_t, real_t>> curve;
                curve.push_back({0.0f, 0.0f});
                curve.push_back({1.0f, 0.0f});
                return curve;
            }

            vector<pair<real_t, real_t>> curve;
            curve.reserve(sorted.size() + 1);
            
            size_t current_tp = 0;
            size_t current_fp = 0;
            
            // Add point at recall=0
            curve.push_back({0.0f, 1.0f});
            
            for (size_t i = 0; i < sorted.size(); i++) {
                if (sorted[i].is_anomaly) current_tp++;
                else current_fp++;
                
                // Only add point if score changes or it's the last sample
                if (i + 1 == sorted.size() || sorted[i].score != sorted[i+1].score) {
                    real_t recall = static_cast<real_t>(current_tp) / n_pos;
                    real_t precision = static_cast<real_t>(current_tp) / (current_tp + current_fp);
                    curve.push_back({recall, precision});
                }
            }
            
            return curve;
        }

        // ===== Anomaly score stats =====
        real_t mean_score()      const { return safe_div(sum_scores_, static_cast<real_t>(score_count_)); }
        real_t min_anomaly_score() const { return score_count_ > 0 ? min_score_ : 0.0f; }
        real_t max_anomaly_score() const { return score_count_ > 0 ? max_score_ : 0.0f; }

        // ===== Unified metric calculation =====
        real_t calculate_metric(eval_metric m) const {
            switch (m) {
                case eval_metric::ANOMALY_SCORE:   return mean_score();
                case eval_metric::ACCURACY:        return accuracy();
                case eval_metric::PRECISION:       return precision();
                case eval_metric::RECALL:          return recall();
                case eval_metric::F1_SCORE:        return f1();
                case eval_metric::ROC_AUC:         return roc_auc();
                case eval_metric::PRC_AUC:         return prc_auc();
                case eval_metric::AVERAGE_PRECISION: return average_precision();
                case eval_metric::FPR:             return fpr();
                case eval_metric::TPR:             return tpr();
                default: return 0.0f;
            }
        }

        real_t calculate_score() const { return calculate_metric(metric_); }

        size_t memory_usage() const { 
            return sizeof(*this) + scores_cache_.capacity() * sizeof(sample_score); 
        }
    };

    using eml_classification_metrics = eml_metrics_calc_t<problem_type::CLASSIFICATION>;
    using eml_regression_metrics     = eml_metrics_calc_t<problem_type::REGRESSION>;
    using eml_isolation_metrics      = eml_metrics_calc_t<problem_type::ISOLATION>;
    using eml_metrics = eml_metrics_calc_t<problem_type::CLASSIFICATION>; // default to classification

} // namespace eml
