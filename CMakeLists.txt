cmake_minimum_required(VERSION 3.18)
project(custom_inference_scheduler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Apple Silicon backend
set(GGML_METAL ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

add_subdirectory(external/llama.cpp)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_runtime
    src/runtime/scheduler.cpp
)

target_link_libraries(_runtime PRIVATE
    llama
    ggml
    ggml-base
    ggml-metal
)

target_include_directories(_runtime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

enable_testing()
add_subdirectory(tests)
