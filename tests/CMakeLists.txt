option(QUICKSERVE_TEST_SANITIZERS "Build tests with ASan/UBSan" ON)

function(configure_model_free_test target test_name)
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
    )
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_options(${target} PRIVATE -Wall -Wextra)
    if(QUICKSERVE_TEST_SANITIZERS)
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined -g)
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
    add_test(NAME ${test_name} COMMAND ${target})
endfunction()

add_executable(test_protocol test_protocol.cpp)
configure_model_free_test(test_protocol protocol)

add_executable(test_model_profile
    test_model_profile.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/model_profile.cpp
)
configure_model_free_test(test_model_profile model_profile)

add_executable(test_hardware_profile
    test_hardware_profile.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/hardware_profile.cpp
)
configure_model_free_test(test_hardware_profile hardware_profile)

add_executable(test_heuristic_aimd
    test_heuristic_aimd.cpp
    ${CMAKE_SOURCE_DIR}/src/policies/heuristic_aimd.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/scheduler.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/model_profile.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/hardware_profile.cpp
)
configure_model_free_test(test_heuristic_aimd heuristic_aimd)

add_executable(test_trace
    test_trace.cpp
    ${CMAKE_SOURCE_DIR}/src/benchmark/trace.cpp
)
configure_model_free_test(test_trace trace)

add_executable(test_benchmark
    test_benchmark.cpp
    ${CMAKE_SOURCE_DIR}/src/benchmark/trace.cpp
    ${CMAKE_SOURCE_DIR}/src/benchmark/replay.cpp
    ${CMAKE_SOURCE_DIR}/src/benchmark/results.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/scheduler.cpp
    ${CMAKE_SOURCE_DIR}/src/policies/fifo.cpp
)
configure_model_free_test(test_benchmark benchmark)

# Model-free handoff tests link nothing from llama.cpp, so building them does
# not pull in the model backend.
add_executable(test_handoff
    test_handoff.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/scheduler.cpp
)
configure_model_free_test(test_handoff handoff)

add_executable(test_scheduler
    test_scheduler.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/scheduler.cpp
)
configure_model_free_test(test_scheduler scheduler)

add_executable(test_fifo
    test_fifo.cpp
    ${CMAKE_SOURCE_DIR}/src/policies/fifo.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/scheduler.cpp
)
configure_model_free_test(test_fifo fifo)

add_executable(test_continuous_batched_fifo
    test_continuous_batched_fifo.cpp
    ${CMAKE_SOURCE_DIR}/src/policies/continuous_batched_fifo.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/scheduler.cpp
)
configure_model_free_test(test_continuous_batched_fifo continuous_batched_fifo)

if(QUICKSERVE_TEST_MODEL)
    get_filename_component(QUICKSERVE_TEST_MODEL_ABSOLUTE
        "${QUICKSERVE_TEST_MODEL}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")

    add_executable(test_real_backend test_real_backend.cpp)
    target_link_libraries(test_real_backend PRIVATE quickserve_policies)
    target_compile_features(test_real_backend PRIVATE cxx_std_17)
    target_compile_options(test_real_backend PRIVATE -Wall -Wextra)
    if(QUICKSERVE_TEST_SANITIZERS)
        target_compile_options(test_real_backend PRIVATE
            -fsanitize=address,undefined -g)
        target_link_options(test_real_backend PRIVATE
            -fsanitize=address,undefined)
    endif()
    add_test(NAME real_backend COMMAND test_real_backend
        "${QUICKSERVE_TEST_MODEL_ABSOLUTE}")
endif()
