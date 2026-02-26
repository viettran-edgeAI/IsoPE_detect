cmake_minimum_required(VERSION 3.18)

project(pe_model_engine LANGUAGES CXX)

option(PE_MODEL_ENGINE_BUILD_APPS "Build model engine CLI apps" ON)
option(PE_MODEL_ENGINE_BUILD_TESTS "Build model engine tests" ON)

find_package(LIEF QUIET CONFIG)

if(NOT TARGET LIEF::LIEF)
  set(PE_MODEL_ENGINE_LOCAL_LIEF_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/LIEF")
  set(PE_MODEL_ENGINE_LOCAL_LIEF_BUILD "${PE_MODEL_ENGINE_LOCAL_LIEF_ROOT}/build")

  find_library(PE_MODEL_ENGINE_LIEF_LIBRARY
    NAMES LIEF libLIEF
    PATHS
      ${PE_MODEL_ENGINE_LOCAL_LIEF_BUILD}
      ${PE_MODEL_ENGINE_LOCAL_LIEF_BUILD}/Release
      ${PE_MODEL_ENGINE_LOCAL_LIEF_BUILD}/lib
    NO_DEFAULT_PATH
  )

  if(PE_MODEL_ENGINE_LIEF_LIBRARY)
    add_library(LIEF::LIEF UNKNOWN IMPORTED)
    set_target_properties(LIEF::LIEF PROPERTIES
      IMPORTED_LOCATION "${PE_MODEL_ENGINE_LIEF_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES
        "${PE_MODEL_ENGINE_LOCAL_LIEF_ROOT}/include;${PE_MODEL_ENGINE_LOCAL_LIEF_ROOT}/src;${PE_MODEL_ENGINE_LOCAL_LIEF_BUILD}/include;${PE_MODEL_ENGINE_LOCAL_LIEF_BUILD}"
    )

    if(UNIX AND NOT APPLE)
      set_property(TARGET LIEF::LIEF APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES "dl;pthread;z"
      )
    endif()
  endif()
endif()

if(NOT TARGET LIEF::LIEF)
  message(FATAL_ERROR "LIEF not found. Build third_party/LIEF or provide LIEF package config.")
endif()

add_subdirectory(src)

if(PE_MODEL_ENGINE_BUILD_APPS)
  add_subdirectory(app)
endif()

if(PE_MODEL_ENGINE_BUILD_TESTS)
  add_subdirectory(test)
endif()
