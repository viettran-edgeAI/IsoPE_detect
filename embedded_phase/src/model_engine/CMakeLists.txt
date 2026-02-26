cmake_minimum_required(VERSION 3.18)

project(pe_model_engine LANGUAGES CXX)

option(PE_MODEL_ENGINE_BUILD_APPS "Build model engine CLI apps" ON)
option(PE_MODEL_ENGINE_BUILD_TESTS "Build model engine tests" ON)

add_subdirectory(src)

if(PE_MODEL_ENGINE_BUILD_APPS)
  add_subdirectory(app)
endif()

if(PE_MODEL_ENGINE_BUILD_TESTS)
  add_subdirectory(test)
endif()
