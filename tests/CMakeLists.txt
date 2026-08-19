option(QUICKSERVE_TEST_SANITIZERS "Build tests with ASan/UBSan" ON)

add_executable(test_protocol test_protocol.cpp)

target_include_directories(test_protocol PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(test_protocol PRIVATE cxx_std_17)
target_compile_options(test_protocol PRIVATE -Wall -Wextra)

# Model-free handoff tests link nothing from llama.cpp, so building them does
# not pull in the model backend.
add_executable(test_handoff test_handoff.cpp)

target_include_directories(test_handoff PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(test_handoff PRIVATE cxx_std_17)
target_compile_options(test_handoff PRIVATE -Wall -Wextra)

if(QUICKSERVE_TEST_SANITIZERS)
    target_compile_options(test_protocol PRIVATE -fsanitize=address,undefined -g)
    target_link_options(test_protocol PRIVATE -fsanitize=address,undefined)

    target_compile_options(test_handoff PRIVATE -fsanitize=address,undefined -g)
    target_link_options(test_handoff PRIVATE -fsanitize=address,undefined)
endif()

add_test(NAME protocol COMMAND test_protocol)
add_test(NAME handoff COMMAND test_handoff)
