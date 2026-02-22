#pragma once

#include "../base/eml_base.h"

namespace mcu {
    // ----------------------- Label types ---------------------------
    using c_label_type = uint8_t;           // sample for classification problem
    using r_label_type = int16_t;          // sample for regression problem (quantized from float)    

    // ----------------------- Error label ---------------------------
    // define error label template
    template<typename T>
    struct eml_err_label {
        static constexpr T value = static_cast<T>(~static_cast<T>(0));
    };

    // ----------------------- label traits ---------------------------
    template<problem_type ProblemType>
    struct eml_label_traits {};

    // classification 
    template<>
    struct eml_label_traits<problem_type::CLASSIFICATION> {
        using label_type = c_label_type;
        static constexpr label_type error_label = eml_err_label<label_type>::value;
    };
    // regression
    template<>
    struct eml_label_traits<problem_type::REGRESSION> {
        using label_type = r_label_type;
        static constexpr label_type error_label = eml_err_label<label_type>::value;
    };

    template<problem_type ProblemType>
    using eml_label_t = typename eml_label_traits<ProblemType>::label_type;

    template<problem_type ProblemType>
    static constexpr eml_label_t<ProblemType> EML_ERROR_LABEL = eml_label_traits<ProblemType>::error_label;

    // sample structures using CRTP pattern
    template<typename Derived, problem_type ProblemType>
    struct eml_sample_base {
        using label_type = eml_label_t<ProblemType>;
        label_type label;
        packed_vector<8> features;

        eml_sample_base() : label(0), features() {}

        // Convenience constructors for derived samples
        template<uint8_t BitsPerValue>
        eml_sample_base(label_type label,
                        const packed_vector<BitsPerValue>& source,
                        size_t start, size_t end) {
            this->label = label;
            features = packed_vector<BitsPerValue>(source, start, end);
        }

        template<uint8_t BitsPerValue>
        eml_sample_base(const packed_vector<BitsPerValue>& features,
                        label_type label) {
            this->label = label;
            this->features = features;
        }

        template<uint8_t BitsPerValue>
        eml_sample_base(label_type label,
                        packed_vector<BitsPerValue>&& features) {
            this->label = label;
            this->features = std::move(features);
        }

        // ------------------ REQUIRED INTERFACES ------------------

        // // construct from parent packed_vector in eml_data (for data slicing - maximum speed)
        // template<uint8_t BitsPerValue>
        // eml_sample_base(label_type label, const packed_vector<BitsPerValue>& source, size_t start, size_t end) {
        //     this->label = label;    
        //     features = packed_vector<8>(source, start, end);

        //     static_assert(std::is_constructible_v<Derived, label_type, const packed_vector<BitsPerValue>&, size_t, size_t>,
        //                   "Derived sample must implement the constructor: (label_type, const packed_vector<BitsPerValue>&, start, end)");
        // }

        // // standard constructor
        // template<uint8_t BitsPerValue>
        // eml_sample_base(const packed_vector<BitsPerValue>& features, label_type label) : features(features), label(label) {
        //     static_assert(std::is_constructible_v<Derived, const packed_vector<BitsPerValue>&, label_type>,
        //                   "Derived sample must implement the constructor: (const packed_vector<BitsPerValue>&, label_type)");
        // }

        // Note: using Factory CRTP instead of coercive constructor (above)

        // 1. slice constructor : construct from parent packed_vector in eml_data (for data slicing - maximum speed)
        template<uint8_t BitsPerValue>
        static Derived create_from_slice(label_type label,
                                         const packed_vector<BitsPerValue>& source, 
                                         size_t start, size_t end) 
        {
            return Derived::template create_from_slice_impl(BitsPerValue, label, source, start, end);
        }

        // 2. standard constructor
        template<uint8_t BitsPerValue>
        static Derived create_from_features(const packed_vector<BitsPerValue>& features,
                                            label_type label) 
        {
            return Derived::template create_from_features_impl(BitsPerValue, features, label);
        }

        // 3. move constructor
        template<uint8_t BitsPerValue>
        static Derived create_from_features_move(label_type label,
                                                 packed_vector<BitsPerValue>&& features) 
        {
            return Derived::template create_from_features_move_impl(BitsPerValue, label, std::move(features));
        }

        // 4. operator[] to access feature by index
        inline uint8_t operator[](size_t index) const {
            return static_cast<const Derived*>(this)->operator[](index);
        }

    protected:
        ~eml_sample_base() = default;
    };

    // ----------------------- Sample structures ---------------------------

    // 1. classification sample
    struct classifier_sample_t : public eml_sample_base<classifier_sample_t, problem_type::CLASSIFICATION> {
        using base_t = eml_sample_base<classifier_sample_t, problem_type::CLASSIFICATION>;
        using label_type = typename base_t::label_type;

        using base_t::base_t;
        classifier_sample_t() = default;

        template<uint8_t BitsPerValue>
        classifier_sample_t(label_type label, packed_vector<BitsPerValue>&& features) {
            this->label = label;
            this->features = std::move(features);
        }

        // ----------------- required implementations -----------------

        // 1. slice constructor implementation
        template<uint8_t BitsPerValue>
        static classifier_sample_t create_from_slice_impl(label_type label,
                                                        const packed_vector<BitsPerValue>& source, 
                                                        size_t start, size_t end) 
        {
            return classifier_sample_t(label, packed_vector<BitsPerValue>(source, start, end));
        }

        template<uint8_t BitsPerValue>
        static classifier_sample_t create_from_slice_impl(uint8_t /*bpv*/,
                                                        label_type label,
                                                        const packed_vector<BitsPerValue>& source,
                                                        size_t start, size_t end)
        {
            return create_from_slice_impl(label, source, start, end);
        }

        // 2. standard constructor implementation
        template<uint8_t BitsPerValue>
        static classifier_sample_t create_from_features_impl(const packed_vector<BitsPerValue>& features,
                                                            label_type label) 
        {
            return classifier_sample_t(features, label);
        }

        template<uint8_t BitsPerValue>
        static classifier_sample_t create_from_features_impl(uint8_t /*bpv*/,
                                                            const packed_vector<BitsPerValue>& features,
                                                            label_type label)
        {
            return create_from_features_impl(features, label);
        }

        // 3. move constructor implementation
        template<uint8_t BitsPerValue>
        static classifier_sample_t create_from_features_move_impl(label_type label,
                                                                packed_vector<BitsPerValue>&& features) 
        {
            return classifier_sample_t(label, std::move(features));
        }

        template<uint8_t BitsPerValue>
        static classifier_sample_t create_from_features_move_impl(uint8_t /*bpv*/,
                                                                label_type label,
                                                                packed_vector<BitsPerValue>&& features)
        {
            return create_from_features_move_impl(label, std::move(features));
        }

        // 4. operator[] implementation
        inline uint8_t operator[](size_t index) const {
            return static_cast<uint8_t>(features[index]);
        }
        
    };

    struct regression_sample_t : public eml_sample_base<regression_sample_t, problem_type::REGRESSION> {
        using base_t = eml_sample_base<regression_sample_t, problem_type::REGRESSION>;
        using label_type = typename base_t::label_type;

        using base_t::base_t;
        regression_sample_t() = default;
        template<uint8_t BitsPerValue>
        regression_sample_t(label_type label, packed_vector<BitsPerValue>&& features) {
            this->label = label;
            this->features = std::move(features);
        }
        // ----------------- required implementations -----------------
        // 1. slice constructor implementation
        template<uint8_t BitsPerValue>
        static regression_sample_t create_from_slice_impl(label_type label,
                                                        const packed_vector<BitsPerValue>& source, 
                                                        size_t start, size_t end) 
        {
            return regression_sample_t(label, packed_vector<BitsPerValue>(source, start, end));
        }

        template<uint8_t BitsPerValue>
        static regression_sample_t create_from_slice_impl(uint8_t /*bpv*/,
                                                        label_type label,
                                                        const packed_vector<BitsPerValue>& source,
                                                        size_t start, size_t end)
        {
            return create_from_slice_impl(label, source, start, end);
        }
        // 2. standard constructor implementation
        template<uint8_t BitsPerValue>
        static regression_sample_t create_from_features_impl(const packed_vector<BitsPerValue>& features,
                                                            label_type label) 
        {
            return regression_sample_t(features, label);
        }

        template<uint8_t BitsPerValue>
        static regression_sample_t create_from_features_impl(uint8_t /*bpv*/,
                                                            const packed_vector<BitsPerValue>& features,
                                                            label_type label)
        {
            return create_from_features_impl(features, label);
        }
        // 3. move constructor implementation
        template<uint8_t BitsPerValue>
        static regression_sample_t create_from_features_move_impl(label_type label,
                                                                packed_vector<BitsPerValue>&& features) 
        {
            return regression_sample_t(label, std::move(features));
        }

        template<uint8_t BitsPerValue>
        static regression_sample_t create_from_features_move_impl(uint8_t /*bpv*/,
                                                                label_type label,
                                                                packed_vector<BitsPerValue>&& features)
        {
            return create_from_features_move_impl(label, std::move(features));
        }
        // 4. operator[] implementation
        inline uint8_t operator[](size_t index) const {
            return static_cast<uint8_t>(features[index]);
        }   
        
    };

    // ----------------------- sample traits ---------------------------

    // global sample traits
    template<problem_type ProblemType>
    struct eml_sample_traits {};

    // classification 
    template<>
    struct eml_sample_traits<problem_type::CLASSIFICATION> {
        using sample_type = classifier_sample_t;
        using label_type  = c_label_type;

        static constexpr bool is_classification = true;
        static constexpr bool is_regression = false;
    };

    // regression
    template<>
    struct eml_sample_traits<problem_type::REGRESSION> {
        using sample_type = regression_sample_t;     
        using label_type  = r_label_type; 

        static constexpr bool is_classification = false;
        static constexpr bool is_regression = true;
    };


    // --------------------- Convenience aliases ------------------------
    template<problem_type ProblemType>
    using eml_sample_t = typename eml_sample_traits<ProblemType>::sample_type;


    // --------------------- drift sample ------------------------
    // Record of a detected concept-drift sample (value exceeded current per-feature min/max).
    struct eml_drift_sample {
        sample_idx_type pending_index = 0; // index inside pending_samples
        float value = 0.0f;
        uint16_t feature_index = 0; 
    };


}