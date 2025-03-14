# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


include(../External/Donut/compileshaders.cmake)
file(GLOB_RECURSE shaders "*.hlsl" "*.hlsli")
file(GLOB_RECURSE sources "*.cpp" "*.h" "*.md")

file(GLOB NRD_INTEGRATION_HEADERS "../External/Nrd/Integration/*.h" "../External/Nrd/Integration/*.hpp")
source_group("RayTracingDenoiser" FILES ${NRD_INTEGRATION_HEADERS})

set(project Rtxpt)
set(folder "!RTX Path Tracing")

set(rtxpt_shaders ${project}_shaders)

if (WIN32)
set(SHADERMAKE_OPTIONS_DXIL "-D ENABLE_DEBUG_PRINT")
set(SHADERMAKE_OPTIONS_SPIRV "-D ENABLE_DEBUG_PRINT")
else()
# DXIL is irrelevant on Linux
set(SHADERMAKE_OPTIONS_DXIL "")
# Don't enable DebugPrint with Linux DXC because that leads to segfaults in the compiler (as of v1.8.2407)
set(SHADERMAKE_OPTIONS_SPIRV "")
endif()

if(DONUT_WITH_AFTERMATH OR DONUT_EMBED_SHADER_PDBS)
    set(shadermake_options --embedPDB)
endif()

donut_compile_shaders(
    TARGET rtxpt_shaders
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg
    SOURCES ${shaders}
    FOLDER ${folder}
    DXIL ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/${project}/dxil
    SPIRV_DXC ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/${project}/spirv
    INCLUDES ${CMAKE_SOURCE_DIR}/External/Nrd/Shaders/Include ${CMAKE_SOURCE_DIR}/External/Rtxdi/Include
    SHADERMAKE_OPTIONS ${shadermake_options}
    SHADERMAKE_OPTIONS_DXIL ${SHADERMAKE_OPTIONS_DXIL}
    SHADERMAKE_OPTIONS_SPIRV ${SHADERMAKE_OPTIONS_SPIRV}
)

add_executable(${project} WIN32 ${sources})

add_dependencies(${project} rtxpt_shaders nrd_shaders omm_shaders)

target_link_libraries(${project} cxxopts donut_render donut_app donut_engine NRD Rtxdi omm-lib omm-gpu-nvrhi)

set_target_properties(${project} PROPERTIES FOLDER ${folder})

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${shaders})

target_include_directories(${project} PRIVATE "../External/Nrd/Include" ${CMAKE_SOURCE_DIR}/External/Rtxdi/Include) #"../External/Streamline/include")