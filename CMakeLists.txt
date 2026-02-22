cmake_minimum_required(VERSION 3.18)

project(edr_agent LANGUAGES CXX)

enable_testing()

option(BUILD_EMBEDDED_MODEL_ENGINE "Build embedded model engine" ON)

if(BUILD_EMBEDDED_MODEL_ENGINE)
  add_subdirectory(embedded_phase/src/model_engine)
endif()
