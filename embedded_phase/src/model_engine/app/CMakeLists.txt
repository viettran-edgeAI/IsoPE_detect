add_executable(pe_model_engine_cli
  model_engine_cli.cpp
)

target_compile_features(pe_model_engine_cli PRIVATE cxx_std_17)

target_link_libraries(pe_model_engine_cli
  PRIVATE
    pe_model_engine
)

add_executable(pe_model_engine_benchmark_cli
  model_engine_benchmark_cli.cpp
)

target_compile_features(pe_model_engine_benchmark_cli PRIVATE cxx_std_17)

target_link_libraries(pe_model_engine_benchmark_cli
  PRIVATE
    pe_model_engine
)
