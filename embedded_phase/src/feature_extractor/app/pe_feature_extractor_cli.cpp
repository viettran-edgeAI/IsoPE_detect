#include "extractor/extractor.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string output_path;
  std::string format = "csv";
  std::vector<std::string> files;
};

std::string lower_ascii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

std::string json_escape(const std::string& text) {
  std::ostringstream out;
  for (char ch : text) {
    switch (ch) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << ch; break;
    }
  }
  return out.str();
}

void print_usage(const char* prog) {
  std::cerr << "Usage: " << prog << " [--output PATH] [--format csv|jsonl] <pe_file> [<pe_file> ...]\n";
  std::cerr << "Compiled feature source: " << extractor::compiled_feature_source() << "\n";
}

bool parse_args(int argc, char** argv, Args& args) {
  for (int index = 1; index < argc; ++index) {
    const std::string token(argv[index]);

    if (token == "--output") {
      if (index + 1 >= argc) {
        return false;
      }
      args.output_path = argv[++index];
    } else if (token == "--format") {
      if (index + 1 >= argc) {
        return false;
      }
      args.format = lower_ascii(argv[++index]);
      if (args.format != "csv" && args.format != "jsonl") {
        return false;
      }
    } else if (token == "--help" || token == "-h") {
      return false;
    } else if (!token.empty() && token[0] == '-') {
      return false;
    } else {
      args.files.push_back(token);
    }
  }

  return !args.files.empty();
}

int run(const Args& args) {
  std::ostream* out = &std::cout;
  std::ofstream file_out;
  if (!args.output_path.empty()) {
    file_out.open(args.output_path, std::ios::out | std::ios::trunc);
    if (!file_out) {
      std::cerr << "Failed to open output path: " << args.output_path << "\n";
      return 2;
    }
    out = &file_out;
  }

  const std::vector<std::string> feature_names = extractor::compiled_feature_names();
  extractor::PEExtractor pe_extractor;

  out->setf(std::ios::fixed);
  *out << std::setprecision(9);

  if (args.format == "jsonl") {
    for (const std::string& path : args.files) {
      const extractor::ExtractionReport report = pe_extractor.extract_with_metadata(path);
      *out << "{\"filepath\":\"" << json_escape(path)
           << "\",\"parse_ok\":" << (report.metadata.parse_ok ? "true" : "false")
           << ",\"processing_time_ms\":" << report.metadata.processing_time_ms
           << ",\"error\":\"" << json_escape(report.metadata.error)
           << "\",\"features\":{";

      for (size_t i = 0; i < feature_names.size(); ++i) {
        const double value = (i < report.feature_vector.values.size()) ? report.feature_vector.values[i] : 0.0;
        *out << "\"" << json_escape(feature_names[i]) << "\":" << value;
        if (i + 1 < feature_names.size()) {
          *out << ',';
        }
      }
      *out << "}}\n";
    }
  } else {
    *out << "filepath,parse_ok,processing_time_ms,error";
    for (const std::string& feature_name : feature_names) {
      *out << ',' << feature_name;
    }
    *out << '\n';

    for (const std::string& path : args.files) {
      const extractor::ExtractionReport report = pe_extractor.extract_with_metadata(path);
      *out << '"' << json_escape(path) << '"'
           << ',' << (report.metadata.parse_ok ? 1 : 0)
           << ',' << report.metadata.processing_time_ms
           << ',' << '"' << json_escape(report.metadata.error) << '"';

      for (size_t i = 0; i < feature_names.size(); ++i) {
        const double value = (i < report.feature_vector.values.size()) ? report.feature_vector.values[i] : 0.0;
        *out << ',' << value;
      }
      *out << '\n';
    }
  }

  return 0;
}

} // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, args)) {
    print_usage(argv[0]);
    return 1;
  }
  return run(args);
}
