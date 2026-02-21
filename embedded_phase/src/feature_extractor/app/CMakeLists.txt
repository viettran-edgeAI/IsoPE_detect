add_executable(lief_feature_extractor
  pe_feature_extractor_cli.cpp
)

target_compile_features(lief_feature_extractor PRIVATE cxx_std_17)

target_link_libraries(lief_feature_extractor
  PRIVATE
    pe_feature_extractor
)
