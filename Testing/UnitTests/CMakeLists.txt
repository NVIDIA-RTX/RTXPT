cmake_minimum_required(VERSION 3.18)

# Also supports a CPU-only build: cmake -S Testing/UnitTests -B build/omm-smoke
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(RTXPathTracingCpuTests LANGUAGES CXX)
    enable_testing()
endif()

add_executable(omm_bake_input_key_smoke
    omm_bake_input_key_smoke.cpp
    omm_cpu_input_identity_tests.cpp
)
target_compile_features(omm_bake_input_key_smoke PRIVATE cxx_std_20)
# The test exercises inline production code; no GPU libraries or shaders are needed.
target_include_directories(omm_bake_input_key_smoke PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../External/Omm/libraries/omm-gpu-nvrhi"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../External/Omm/libraries/omm-lib/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../External/Donut/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../External/Donut/nvrhi/include"
)
add_test(NAME omm_bake_input_key_smoke COMMAND omm_bake_input_key_smoke)
