cmake_minimum_required(VERSION 3.18)
project(custom_inference_scheduler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(QUICKSERVE_TEST_MODEL "" CACHE FILEPATH
    "Absolute path to a GGUF model used by the real-backend test")
set(QUICKSERVE_BENCHMARK_POLICY_SOURCE "" CACHE FILEPATH
    "C++ source defining quickserve_benchmark_policy::create")
set(QUICKSERVE_BENCHMARK_POLICY_HEADER "" CACHE FILEPATH
    "C++ header declaring quickserve_benchmark_policy::create")

if((QUICKSERVE_BENCHMARK_POLICY_SOURCE AND NOT QUICKSERVE_BENCHMARK_POLICY_HEADER) OR
   (QUICKSERVE_BENCHMARK_POLICY_HEADER AND NOT QUICKSERVE_BENCHMARK_POLICY_SOURCE))
    message(FATAL_ERROR "QUICKSERVE_BENCHMARK_POLICY_SOURCE and QUICKSERVE_BENCHMARK_POLICY_HEADER must be supplied together")
endif()

# Apple Silicon backend
set(GGML_METAL ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

add_subdirectory(external/llama.cpp)
find_package(Threads REQUIRED)

add_library(quickserve_runtime
    src/runtime/scheduler.cpp
    src/runtime/environment.cpp
    src/runtime/model_profile.cpp
    src/runtime/hardware_profile.cpp
)

target_link_libraries(quickserve_runtime PUBLIC
    llama
    Threads::Threads
)

target_include_directories(quickserve_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(quickserve_runtime PUBLIC cxx_std_17)

add_library(quickserve_policies src/policies/fifo.cpp)
target_link_libraries(quickserve_policies PUBLIC quickserve_runtime)
target_include_directories(quickserve_policies PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(quickserve_policies PUBLIC cxx_std_17)

add_library(quickserve_trace src/benchmark/trace.cpp)
target_include_directories(quickserve_trace PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(quickserve_trace PUBLIC cxx_std_17)

add_library(quickserve_benchmark_support
    src/benchmark/replay.cpp
    src/benchmark/results.cpp
)
target_link_libraries(quickserve_benchmark_support PUBLIC quickserve_trace)
target_include_directories(quickserve_benchmark_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(quickserve_benchmark_support PUBLIC cxx_std_17)

add_executable(quickserve_prepare_trace src/benchmark/prepare_trace_main.cpp)
target_link_libraries(quickserve_prepare_trace PRIVATE quickserve_trace)
target_compile_features(quickserve_prepare_trace PRIVATE cxx_std_17)

if(QUICKSERVE_BENCHMARK_POLICY_SOURCE AND QUICKSERVE_BENCHMARK_POLICY_HEADER)
    get_filename_component(QUICKSERVE_BENCHMARK_POLICY_SOURCE_ABSOLUTE
        "${QUICKSERVE_BENCHMARK_POLICY_SOURCE}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
    get_filename_component(QUICKSERVE_BENCHMARK_POLICY_HEADER_ABSOLUTE
        "${QUICKSERVE_BENCHMARK_POLICY_HEADER}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
    if(NOT EXISTS "${QUICKSERVE_BENCHMARK_POLICY_SOURCE_ABSOLUTE}")
        message(FATAL_ERROR "Benchmark policy source does not exist: ${QUICKSERVE_BENCHMARK_POLICY_SOURCE_ABSOLUTE}")
    endif()
    if(NOT EXISTS "${QUICKSERVE_BENCHMARK_POLICY_HEADER_ABSOLUTE}")
        message(FATAL_ERROR "Benchmark policy header does not exist: ${QUICKSERVE_BENCHMARK_POLICY_HEADER_ABSOLUTE}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${QUICKSERVE_BENCHMARK_POLICY_SOURCE_ABSOLUTE}"
        "${QUICKSERVE_BENCHMARK_POLICY_HEADER_ABSOLUTE}")

    file(SHA256 "${QUICKSERVE_BENCHMARK_POLICY_SOURCE_ABSOLUTE}" QUICKSERVE_POLICY_SOURCE_HASH)
    file(SHA256 "${QUICKSERVE_BENCHMARK_POLICY_HEADER_ABSOLUTE}" QUICKSERVE_POLICY_HEADER_HASH)
    get_filename_component(QUICKSERVE_POLICY_NAME_VALUE
        "${QUICKSERVE_BENCHMARK_POLICY_SOURCE_ABSOLUTE}" NAME_WE)
    execute_process(COMMAND git rev-parse --short=12 HEAD
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE QUICKSERVE_GIT_REV OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    execute_process(COMMAND git rev-parse --short=12 HEAD
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/external/llama.cpp"
        OUTPUT_VARIABLE QUICKSERVE_LLAMA_REV OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(NOT QUICKSERVE_GIT_REV)
        set(QUICKSERVE_GIT_REV "unknown")
    endif()
    if(NOT QUICKSERVE_LLAMA_REV)
        set(QUICKSERVE_LLAMA_REV "unknown")
    endif()
    configure_file(cmake/benchmark_policy_adapter.hpp.in
        "${CMAKE_CURRENT_BINARY_DIR}/generated/benchmark_policy_adapter.hpp" @ONLY)

    add_executable(quickserve_benchmark
        src/benchmark/benchmark_main.cpp
        "${QUICKSERVE_BENCHMARK_POLICY_SOURCE_ABSOLUTE}"
    )
    target_include_directories(quickserve_benchmark PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(quickserve_benchmark PRIVATE
        quickserve_benchmark_support quickserve_policies)
    target_compile_features(quickserve_benchmark PRIVATE cxx_std_17)
    target_compile_definitions(quickserve_benchmark PRIVATE
        QUICKSERVE_POLICY_NAME="${QUICKSERVE_POLICY_NAME_VALUE}"
        QUICKSERVE_POLICY_SOURCE_SHA256="${QUICKSERVE_POLICY_SOURCE_HASH}"
        QUICKSERVE_POLICY_HEADER_SHA256="${QUICKSERVE_POLICY_HEADER_HASH}"
        QUICKSERVE_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
        QUICKSERVE_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
        QUICKSERVE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        QUICKSERVE_GIT_REVISION="${QUICKSERVE_GIT_REV}"
        QUICKSERVE_LLAMA_REVISION="${QUICKSERVE_LLAMA_REV}")
endif()

enable_testing()
add_subdirectory(tests)
