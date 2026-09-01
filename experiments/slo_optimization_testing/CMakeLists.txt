cmake_minimum_required(VERSION 3.18)
project(quickserve_slo_optimization_experiment LANGUAGES CXX)
execute_process(COMMAND git rev-parse --short=12 HEAD WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../.. OUTPUT_VARIABLE QS_REV OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND git rev-parse --short=12 HEAD WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../../external/llama.cpp OUTPUT_VARIABLE LLAMA_REV OUTPUT_STRIP_TRAILING_WHITESPACE)
file(SHA256 "${CMAKE_CURRENT_LIST_DIR}/backend_cache.cpp" COLLECTOR_CPP_SHA)
file(SHA256 "${CMAKE_CURRENT_LIST_DIR}/backend_cache.hpp" COLLECTOR_HPP_SHA)
string(SHA256 COLLECTOR_ID
       "schema=3|backend_cache.cpp=${COLLECTOR_CPP_SHA}|backend_cache.hpp=${COLLECTOR_HPP_SHA}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Eigen3 CONFIG REQUIRED)
find_package(proxsuite CONFIG REQUIRED)

set(CLARABEL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../external/Clarabel.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/clarabel)

set(GGML_METAL ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../external/llama.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/llama)

add_library(slo_experiment_core
    experiment.cpp
    solvers.cpp
    backend_cache.cpp)
target_include_directories(slo_experiment_core PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(slo_experiment_core PUBLIC
    Eigen3::Eigen
    proxsuite::proxsuite
    libclarabel_c_shared
    llama)
target_compile_definitions(slo_experiment_core PUBLIC
    SLO_QS_REV="${QS_REV}" SLO_LLAMA_REV="${LLAMA_REV}"
    SLO_COLLECTOR_ID="${COLLECTOR_ID}"
    SLO_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

add_executable(slo_optimization_experiment main.cpp)
target_link_libraries(slo_optimization_experiment PRIVATE slo_experiment_core)

add_executable(slo_optimization_tests tests.cpp)
target_link_libraries(slo_optimization_tests PRIVATE slo_experiment_core)

enable_testing()
add_test(NAME slo_optimization_tests COMMAND slo_optimization_tests)
