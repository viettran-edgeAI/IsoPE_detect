add_library(pe_feature_extractor STATIC
  pe_extractor.cpp
)

target_compile_features(pe_feature_extractor PUBLIC cxx_std_17)

target_include_directories(pe_feature_extractor
  PUBLIC
    ${FEATURE_EXTRACTOR_MODULE_DIR}/include
  PRIVATE
    ${FEATURE_EXTRACTOR_MODULE_DIR}
)

target_link_libraries(pe_feature_extractor
  PUBLIC
    LIEF::LIEF
)
