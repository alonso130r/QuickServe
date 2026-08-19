option(QUICKSERVE_TEST_SANITIZERS "Build tests with ASan/UBSan" ON)

# These targets need llama.h for llama_token but link nothing from llama.cpp,
# so `make test_handoff` does not pull in the model backend build.
add_executable(test_handoff test_handoff.cpp)

target_include_directories(test_handoff PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/external/llama.cpp/ggml/include
)

target_compile_features(test_handoff PRIVATE cxx_std_17)
target_compile_options(test_handoff PRIVATE -Wall -Wextra)

if(QUICKSERVE_TEST_SANITIZERS)
    target_compile_options(test_handoff PRIVATE -fsanitize=address,undefined -g)
    target_link_options(test_handoff PRIVATE -fsanitize=address,undefined)
endif()

add_test(NAME handoff COMMAND test_handoff)
