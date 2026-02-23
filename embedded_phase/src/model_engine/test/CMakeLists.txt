add_executable(pe_model_engine_tests
  model_engine_tests.cpp
)

target_compile_features(pe_model_engine_tests PRIVATE cxx_std_17)

target_link_libraries(pe_model_engine_tests
  PRIVATE
    pe_model_engine
)

add_test(NAME pe_model_engine_tests COMMAND pe_model_engine_tests)
