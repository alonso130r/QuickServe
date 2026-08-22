cmake_minimum_required(VERSION 3.18)
project(custom_inference_scheduler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(QUICKSERVE_TEST_MODEL "" CACHE FILEPATH
    "Absolute path to a GGUF model used by the real-backend test")

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

enable_testing()
add_subdirectory(tests)
