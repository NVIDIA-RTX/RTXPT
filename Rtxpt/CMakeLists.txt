# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


include(../External/Donut/compileshaders.cmake)
file(GLOB_RECURSE shaders "*.hlsl" "*.hlsli")
file(GLOB_RECURSE shared_sources "*.cpp" "*.h" "*.md")
# Separate out shared sources from entry point files
list(REMOVE_ITEM shared_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/IntroSample/IntroSample.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/AdvancedSample.cpp"
)

file(GLOB NRD_INTEGRATION_HEADERS "../External/Nrd/Integration/*.h" "../External/Nrd/Integration/*.hpp")
source_group("RayTracingDenoiser" FILES ${NRD_INTEGRATION_HEADERS})

set(project Rtxpt)
set(folder "!RTX Path Tracing")

set(rtxpt_shaders ${project}_shaders)

if (WIN32)
set(SHADERMAKE_OPTIONS_DXIL "-D ENABLE_DEBUG_PRINT")
#set(SHADERMAKE_OPTIONS_SPIRV "-D ENABLE_DEBUG_PRINT") # Disable SPIRV debug printf since it causes DXC to crash as of dxc 1.8.2505
set(SHADERMAKE_OPTIONS_SPIRV "")
else()
# DXIL is irrelevant on Linux
set(SHADERMAKE_OPTIONS_DXIL "")
# Don't enable DebugPrint with Linux DXC because that leads to segfaults in the compiler (as of v1.8.2407)
set(SHADERMAKE_OPTIONS_SPIRV "")
endif()

if(DONUT_WITH_AFTERMATH OR DONUT_EMBED_SHADER_PDBS)
    set(shadermake_options "--embedPDB")
#else()
#    set(shadermake_options "--PDB")
endif()

#set(shadermake_options --verbose)

if(RTXPT_D3D_AGILITY_SDK_VERSION AND (RTXPT_D3D_AGILITY_SDK_VERSION GREATER_EQUAL "619"))
    # Appended to the string, not set over it: overwriting would silently drop the --embedPDB above and
    # leave the precompiled shaders with no debug info. It has to stay ONE string rather than become a
    # list - donut_compile_shaders declares SHADERMAKE_OPTIONS as a one-value keyword, so a second list
    # element is dropped, and losing "-m 6_9" this way fails the build at SM 6.5 on AllocateRayQuery2.
    string(APPEND shadermake_options " -m 6_9")
endif()

donut_compile_shaders(
    TARGET rtxpt_shaders
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg
    SOURCES ${shaders}
    FOLDER ${folder}
    DXIL ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderPrecompiled/${project}/dxil
    SPIRV_DXC ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderPrecompiled/${project}/spirv
    INCLUDES ${CMAKE_SOURCE_DIR}/External/Nrd/Shaders ${CMAKE_SOURCE_DIR}/External/Rtxdi/Include ${CMAKE_SOURCE_DIR}/External
    SHADERMAKE_OPTIONS "${shadermake_options}"
    SHADERMAKE_OPTIONS_DXIL ${SHADERMAKE_OPTIONS_DXIL}
    SHADERMAKE_OPTIONS_SPIRV ${SHADERMAKE_OPTIONS_SPIRV}
)


##########################################################################
################ dynamic shader recompile dependencies ###################
set(DYNSH_COPY3_MARKER "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderDynamic/Source/copy.marker")

# The runtime compilers are copied on every build (copy_if_different only updates changed files), not once behind a
# marker - otherwise a DXC upgrade or a different DXC_SPIRV_PATH never reaches bin/ShaderDynamic/Tools.

# DXC_CUSTOM_PATH and DXC_SPIRV_PATH are the build host's compilers; a cross-build ships the embedded DXC for the target instead
set(DYNSH_CROSS_DXC_DIR "")
if (NOT NVRHI_TARGET_ARCH STREQUAL NVRHI_HOST_ARCH)
    set(DYNSH_CROSS_DXC_DIR "${CMAKE_SOURCE_DIR}/External/dxc/build/native/bin/${NVRHI_TARGET_ARCH}")
endif()

# vulkan SDK dxc (optional)
set(DYNSH_COPY_VK_COMMANDS "")
if (DXC_SPIRV_PATH)
get_filename_component(DXC_SPIRV_PATH_MINUS_EXE ${DXC_SPIRV_PATH} DIRECTORY)
if (DYNSH_CROSS_DXC_DIR)
    set(DXC_SPIRV_PATH_MINUS_EXE "${DYNSH_CROSS_DXC_DIR}")
endif()
file(GLOB FILES_TO_COPY_2 "${DXC_SPIRV_PATH_MINUS_EXE}/*.exe" "${DXC_SPIRV_PATH_MINUS_EXE}/*.dll" "${DXC_SPIRV_PATH_MINUS_EXE}/*.json")
set(DEST_DIR_2 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderDynamic/Tools/vk/${NVRHI_TARGET_ARCH}")
set(DYNSH_COPY_VK_COMMANDS
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DEST_DIR_2}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FILES_TO_COPY_2} ${DEST_DIR_2}
)
endif()

# d3d12 dxc
set(SOURCE_DIR_1 "${DXC_CUSTOM_PATH}")
if (DYNSH_CROSS_DXC_DIR)
    set(SOURCE_DIR_1 "${DYNSH_CROSS_DXC_DIR}")
endif()
set(DEST_DIR_1 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderDynamic/Tools/d3d12/${NVRHI_TARGET_ARCH}")
# d3d12 linux dxc
# ...

add_custom_target(ShaderDynamicAssets ALL
    ${DYNSH_COPY_VK_COMMANDS}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DEST_DIR_1}
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${SOURCE_DIR_1} ${DEST_DIR_1}
    COMMENT "Copying shader compilers"
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
# shader files - RTXDI shader dependencies
set(SOURCE_DIR_RTXDI_SHADERS "${PROJECT_SOURCE_DIR}/External/Rtxdi/Include")
set(DEST_DIR_RTXDI_SHADERS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderDynamic/Source/External/Rtxdi/Include")
# shader files - SHaRC shader dependencies
set(SOURCE_DIR_SHARC_SHADERS "${PROJECT_SOURCE_DIR}/External/SHARC/include")
set(DEST_DIR_SHARC_SHADERS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderDynamic/Source/External/SHARC/include")
#message("SourceDir=${SOURCE_DIR_RTXPT_SHADERS}, DestDir=${DEST_DIR_RTXPT_SHADERS}")
add_custom_target(ShaderDynamicAssets_CopyAlways
#    OUTPUT "${DYNSH_COPY3_MARKER}" <- let this run every build; it will only update changed
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DEST_DIR_RTXPT_SHADERS}
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${SOURCE_DIR_RTXPT_SHADERS} ${DEST_DIR_RTXPT_SHADERS}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DEST_DIR_DONUT_SHADERS}
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${SOURCE_DIR_DONUT_SHADERS} ${DEST_DIR_DONUT_SHADERS}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DEST_DIR_NVAPI_SHADERS}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SOURCE_DIR_NVAPI_SHADERS_FILES} ${DEST_DIR_NVAPI_SHADERS}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DEST_DIR_RTXDI_SHADERS}
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${SOURCE_DIR_RTXDI_SHADERS} ${DEST_DIR_RTXDI_SHADERS}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DEST_DIR_SHARC_SHADERS}
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${SOURCE_DIR_SHARC_SHADERS} ${DEST_DIR_SHARC_SHADERS}
#    COMMAND ${CMAKE_COMMAND} -E touch "${DYNSH_COPY3_MARKER}" <- let this run every build; it will only update changed
    COMMENT "Copying shader sources"
)

##########################################################################

# Create a static library for shared code (used by both samples)
add_library(RtxptCore STATIC ${shared_sources})
target_link_libraries(RtxptCore PUBLIC cxxopts donut_render donut_app donut_engine NRD Rtxdi omm-lib omm-gpu-nvrhi TracyClient)
target_include_directories(RtxptCore PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "../External/Nrd/Include" ${CMAKE_SOURCE_DIR}/External/Rtxdi/Include)
target_compile_definitions(RtxptCore PUBLIC RTXPT_D3D12_WITH_NVAPI=$<BOOL:${NVRHI_WITH_NVAPI}>)
add_dependencies(RtxptCore rtxpt_shaders nrd_shaders omm_shaders ShaderDynamicAssets ShaderDynamicAssets_CopyAlways)
set_target_properties(RtxptCore PROPERTIES FOLDER ${folder})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${shaders})
# AgilitySDK support for RtxptCore
if (RTXPT_D3D_AGILITY_SDK_PATH AND RTXPT_D3D_AGILITY_SDK_VERSION)
    if (NOT DEFINED RTXPT_D3D_AGILITY_SDK_VERSION)
        message(FATAL_ERROR "RTXPT_D3D_AGILITY_SDK_VERSION must be defined when RTXPT_D3D_AGILITY_SDK_PATH is defined")
    endif()
    target_compile_definitions(RtxptCore PUBLIC RTXPT_D3D_AGILITY_SDK_VERSION=${RTXPT_D3D_AGILITY_SDK_VERSION})
    target_compile_definitions(RtxptCore PRIVATE NVRHI_D3D12_WITH_DXR12_OPACITY_MICROMAP=$<BOOL:${NVRHI_D3D12_WITH_DXR12_OPACITY_MICROMAP}>)
    target_include_directories(RtxptCore PRIVATE ${RTXPT_D3D_AGILITY_SDK_PATH}/build/native/include)
endif()

# Macro to create sample executables with common settings
macro(add_rtxpt_sample TARGET_NAME MAIN_SOURCE)
    add_executable(${TARGET_NAME} WIN32 ${MAIN_SOURCE})
    target_link_libraries(${TARGET_NAME} PRIVATE RtxptCore)
    add_dependencies(${TARGET_NAME} rtxpt_shaders nrd_shaders omm_shaders ShaderDynamicAssets ShaderDynamicAssets_CopyAlways)
    set_target_properties(${TARGET_NAME} PROPERTIES 
        FOLDER ${folder}
        OUTPUT_NAME_DEBUG "${TARGET_NAME}D"
        OUTPUT_NAME_RELEASE "${TARGET_NAME}"
    )
    
    # AgilitySDK support for this target
    if (RTXPT_D3D_AGILITY_SDK_PATH AND RTXPT_D3D_AGILITY_SDK_VERSION)
        target_compile_definitions(${TARGET_NAME} PUBLIC RTXPT_D3D_AGILITY_SDK_VERSION=${RTXPT_D3D_AGILITY_SDK_VERSION})
        add_custom_command(
            TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${TARGET_NAME}>/D3D12/"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${RTXPT_D3D_AGILITY_SDK_PATH}/build/native/bin/${NVRHI_TARGET_ARCH}/D3D12Core.dll "$<TARGET_FILE_DIR:${TARGET_NAME}>/D3D12/"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${RTXPT_D3D_AGILITY_SDK_PATH}/build/native/bin/${NVRHI_TARGET_ARCH}/d3d12SDKLayers.dll "$<TARGET_FILE_DIR:${TARGET_NAME}>/D3D12/"
        )
        target_compile_definitions(${TARGET_NAME} PRIVATE NVRHI_D3D12_WITH_DXR12_OPACITY_MICROMAP=$<BOOL:${NVRHI_D3D12_WITH_DXR12_OPACITY_MICROMAP}>)
        target_include_directories(${TARGET_NAME} PRIVATE ${RTXPT_D3D_AGILITY_SDK_PATH}/build/native/include)
    endif()

    if (DONUT_WITH_AFTERMATH)
        add_custom_command(
            TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${TARGET_NAME}> "$<TARGET_FILE_DIR:${TARGET_NAME}>/"
            COMMAND_EXPAND_LISTS
        )
    endif()
endmacro()

# Build both sample executables
add_rtxpt_sample(IntroSample "IntroSample/IntroSample.cpp")
add_rtxpt_sample(Rtxpt "AdvancedSample.cpp")
