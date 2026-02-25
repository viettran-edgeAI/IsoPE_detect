#include "if_feature_extractor.h"

// This file pulls in the optional PE extractor implementation.  It is built only
// when callers need the stock file-based extractor; the base header remains
// independent of extractor.hpp so that other runtimes can supply custom
// callbacks.

#include "../../../src/feature_extractor/include/extractor/extractor.hpp"

namespace eml {

If_feature_extractor::extract_callback_t If_feature_extractor::default_pe_path_callback() {
    return [](const std::filesystem::path& pe_path,
               const vector<std::string>& feature_names,
               vector<float>& out_features) -> bool {
        extractor::PEExtractor pe_extractor;
        const std::vector<std::string> selected_features(feature_names.begin(), feature_names.end());
        const extractor::ExtractionReport report =
            pe_extractor.extract_selected_with_metadata(pe_path.string(), selected_features);
        if (!report.metadata.parse_ok || report.feature_vector.values.empty()) {
            return false;
        }

        out_features.resize(report.feature_vector.values.size(), 0.0f);
        for (size_t index = 0; index < report.feature_vector.values.size(); ++index) {
            out_features[index] = static_cast<float>(report.feature_vector.values[index]);
        }
        return true;
    };
}

} // namespace eml
