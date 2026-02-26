add_library(pe_model_engine STATIC
  model_engine.cpp
)

target_include_directories(pe_model_engine
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../core
)

target_compile_features(pe_model_engine PUBLIC cxx_std_17)
target_link_libraries(pe_model_engine PUBLIC LIEF::LIEF)

add_library(pe_model_engine_c STATIC
  model_engine_c.cpp
)

target_include_directories(pe_model_engine_c
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_compile_features(pe_model_engine_c PUBLIC cxx_std_17)
target_link_libraries(pe_model_engine_c PUBLIC pe_model_engine)
