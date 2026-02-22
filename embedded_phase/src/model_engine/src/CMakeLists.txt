add_library(pe_model_engine STATIC
  model_engine.cpp
)

target_compile_features(pe_model_engine PUBLIC cxx_std_17)

target_include_directories(pe_model_engine
  PUBLIC
    ${MODEL_ENGINE_MODULE_DIR}/include
    ${MODEL_ENGINE_MODULE_DIR}/../../core
)
