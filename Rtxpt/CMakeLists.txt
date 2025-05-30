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

#set(shadermake_options --verbose)

donut_compile_shaders(
    TARGET rtxpt_shaders
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg
    SOURCES ${shaders}
    FOLDER ${folder}
    DXIL ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderPrecompiled/${project}/dxil
    SPIRV_DXC ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderPrecompiled/${project}/spirv
    INCLUDES ${CMAKE_SOURCE_DIR}/External/Nrd/Shaders/Include ${CMAKE_SOURCE_DIR}/External/Rtxdi/Include ${CMAKE_SOURCE_DIR}/External
    SHADERMAKE_OPTIONS ${shadermake_options}
    SHADERMAKE_OPTIONS_DXIL ${SHADERMAKE_OPTIONS_DXIL}
    SHADERMAKE_OPTIONS_SPIRV ${SHADERMAKE_OPTIONS_SPIRV}
)


##########################################################################
################ dynamic shader recompile dependencies ###################
set(DYNSH_COPY1_MARKER "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderDynamic/Tools/copyVK.marker")
set(DYNSH_COPY2_MARKER "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderDynamic/Tools/copyDX.marker")
set(DYNSH_COPY3_MARKER "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderDynamic/Source/copy.marker")

# vulkan SDK dxc (optional)
if (DXC_SPIRV_PATH)
get_filename_component(DXC_SPIRV_PATH_MINUS_EXE ${DXC_SPIRV_PATH} DIRECTORY)
file(GLOB FILES_TO_COPY_2 "${DXC_SPIRV_PATH_MINUS_EXE}/*.exe" "${DXC_SPIRV_PATH_MINUS_EXE}/*.dll" "${DXC_SPIRV_PATH_MINUS_EXE}/*.json")
set(DEST_DIR_2 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderDynamic/Tools/vk/x64")
add_custom_command(
    OUTPUT "${DYNSH_COPY1_MARKER}"
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderDynamic
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DEST_DIR_2}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FILES_TO_COPY_2} ${DEST_DIR_2}
    COMMAND ${CMAKE_COMMAND} -E touch "${DYNSH_COPY1_MARKER}"
    COMMENT "Copying ${DXC_SPIRV_PATH_MINUS_EXE} to ${DEST_DIR_2}"
)
endif()

# d3d12 dxc
set(SOURCE_DIR_1 "${DXC_CUSTOM_PATH}")
set(DEST_DIR_1 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderDynamic/Tools/d3d12/x64")
# d3d12 linux dxc
# ...

add_custom_command(
    OUTPUT "${DYNSH_COPY2_MARKER}"
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderDynamic
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DEST_DIR_1}
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${SOURCE_DIR_1} ${DEST_DIR_1}
    COMMAND ${CMAKE_COMMAND} -E touch "${DYNSH_COPY2_MARKER}"
    COMMENT "Copying ${SOURCE_DIR_1} to ${DEST_DIR_1}"
)

# shader files - main rtxpt
set(SOURCE_DIR_RTXPT_SHADERS "${PROJECT_SOURCE_DIR}/Rtxpt/Shaders")
set(DEST_DIR_RTXPT_SHADERS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderDynamic/Source/Rtxpt")
# shader files - donut shader dependencies
set(SOURCE_DIR_DONUT_SHADERS "${PROJECT_SOURCE_DIR}/External/Donut/include/donut/shaders")
set(DEST_DIR_DONUT_SHADERS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderDynamic/Source/External/donut/shaders")
# shader files - NVAPI shader dependencies
set(SOURCE_DIR_NVAPI_SHADERS "${PROJECT_SOURCE_DIR}/External/NVAPI")
set(DEST_DIR_NVAPI_SHADERS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderDynamic/Source/External/NVAPI")
file(GLOB SOURCE_DIR_NVAPI_SHADERS_FILES "${SOURCE_DIR_NVAPI_SHADERS}/nvHLSL*.h" "${SOURCE_DIR_NVAPI_SHADERS}/nvShader*.h")
#message("SourceDir=${SOURCE_DIR_RTXPT_SHADERS}, DestDir=${DEST_DIR_RTXPT_SHADERS}")
add_custom_target(ShaderDynamicAssets_CopyAlways
#    OUTPUT "${DYNSH_COPY3_MARKER}" <- let this run every build; it will only update changed
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DEST_DIR_RTXPT_SHADERS}
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${SOURCE_DIR_RTXPT_SHADERS} ${DEST_DIR_RTXPT_SHADERS}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DEST_DIR_DONUT_SHADERS}
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${SOURCE_DIR_DONUT_SHADERS} ${DEST_DIR_DONUT_SHADERS}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DEST_DIR_NVAPI_SHADERS}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SOURCE_DIR_NVAPI_SHADERS_FILES} ${DEST_DIR_NVAPI_SHADERS}
#    COMMAND ${CMAKE_COMMAND} -E touch "${DYNSH_COPY3_MARKER}" <- let this run every build; it will only update changed
    COMMENT "Copying shader sources"
)

add_custom_target(ShaderDynamicAssets ALL DEPENDS "${DYNSH_COPY1_MARKER}" "${DYNSH_COPY2_MARKER}")

##########################################################################

add_executable(${project} WIN32 ${sources})

add_dependencies(${project} rtxpt_shaders nrd_shaders omm_shaders ShaderDynamicAssets ShaderDynamicAssets_CopyAlways)

target_link_libraries(${project} cxxopts donut_render donut_app donut_engine NRD Rtxdi omm-lib omm-gpu-nvrhi)

set_target_properties(${project} PROPERTIES FOLDER ${folder})

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${shaders})

target_include_directories(${project} PRIVATE "../External/Nrd/Include" ${CMAKE_SOURCE_DIR}/External/Rtxdi/Include) 


##########################################################################
############################# AgilitySDK #################################

if (RTXPT_D3D_AGILITY_SDK_PATH AND RTXPT_D3D_AGILITY_SDK_VERSION)
    if (NOT DEFINED RTXPT_D3D_AGILITY_SDK_VERSION)
        message(FATAL_ERROR "RTXPT_D3D_AGILITY_SDK_VERSION must be defined when RTXPT_D3D_AGILITY_SDK_PATH is defined")
    endif()
    target_compile_definitions(${project} PUBLIC RTXPT_D3D_AGILITY_SDK_VERSION=${RTXPT_D3D_AGILITY_SDK_VERSION})
    add_custom_command(
        TARGET ${project} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${project}>/D3D12/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${RTXPT_D3D_AGILITY_SDK_PATH}/build/native/bin/x64/D3D12Core.dll "$<TARGET_FILE_DIR:${project}>/D3D12/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${RTXPT_D3D_AGILITY_SDK_PATH}/build/native/bin/x64/d3d12SDKLayers.dll "$<TARGET_FILE_DIR:${project}>/D3D12/"
    )
endif()

##########################################################################
