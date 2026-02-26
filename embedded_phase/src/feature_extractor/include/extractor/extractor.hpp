#pragma once

#include <string>
#include <vector>

namespace extractor {

struct FeatureVector {
  std::vector<double> values;
};

class IExtractor {
  public:
  virtual FeatureVector extract(const std::string& path) = 0;
  virtual ~IExtractor() = default;
};

struct ExtractionMetadata {
  bool parse_ok = false;
  std::string error;
  double processing_time_ms = 0.0;
};

struct ExtractionReport {
  FeatureVector feature_vector;
  ExtractionMetadata metadata;
};

class PEExtractor : public IExtractor {
  public:
  FeatureVector extract(const std::string& path) override;
  ExtractionReport extract_with_metadata(const std::string& path) const;
  FeatureVector extract_selected(const std::string& path, const std::vector<std::string>& feature_names) const;
  ExtractionReport extract_selected_with_metadata(
      const std::string& path,
      const std::vector<std::string>& feature_names) const;
};

size_t compiled_feature_count();
std::vector<std::string> compiled_feature_names();
const std::string& compiled_feature_source();

} // namespace extractor
