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

# LIEF's mbedtls sub-library uses inet_pton (ws2_32) and BCryptGenRandom (bcrypt)
# on Windows. Add these system libs when building with MinGW / any Windows toolchain.
if(WIN32)
  target_link_libraries(pe_model_engine PUBLIC ws2_32 bcrypt)
endif()

add_library(pe_model_engine_c STATIC
  model_engine_c.cpp
)

target_include_directories(pe_model_engine_c
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_compile_features(pe_model_engine_c PUBLIC cxx_std_17)
target_link_libraries(pe_model_engine_c PUBLIC pe_model_engine)
