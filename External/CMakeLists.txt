#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

##########################################################################
################################ Donut ####################################
##########################################################################

include(Donut/compileshaders.cmake)
add_subdirectory(Donut)

##########################################################################
################################ UTIL ####################################
##########################################################################
function(util_get_shader_profile_from_name FILE_NAME DXC_PROFILE)
    get_filename_component(EXTENSION ${FILE_NAME} EXT)
    if ("${EXTENSION}" STREQUAL ".cs.hlsl")
        set(DXC_PROFILE "cs" PARENT_SCOPE)
    endif()
    if ("${EXTENSION}" STREQUAL ".vs.hlsl")
        set(DXC_PROFILE "vs" PARENT_SCOPE)
    endif()
    if ("${EXTENSION}" STREQUAL ".gs.hlsl")
        set(DXC_PROFILE "gs" PARENT_SCOPE)
    endif()
    if ("${EXTENSION}" STREQUAL ".ps.hlsl")
        set(DXC_PROFILE "ps" PARENT_SCOPE)
    endif()
endfunction()

function(util_generate_shader_config_file OUT_FILE_NAME DIR DEFINES)
    file(GLOB_RECURSE HLSL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${DIR}/*.hlsl")

    set(out_content "")
    foreach(FILE_NAME ${HLSL_FILES})
        get_filename_component(NAME_ONLY ${FILE_NAME} NAME)
        set(DXC_PROFILE "")
        util_get_shader_profile_from_name(${FILE_NAME} DXC_PROFILE)
        set(out_content "${out_content}${DIR}/${NAME_ONLY} -T ${DXC_PROFILE} -E main ${DEFINES}\n")
    endforeach()

    file(WRITE ${OUT_FILE_NAME} ${out_content})
endfunction()


##########################################################################
########################### Tracy ###########################
##########################################################################
option(TRACY_ENABLE "" ON)      # Tracy >= 0.14 defaults this to OFF
option(TRACY_ON_DEMAND "" ON)   # only profile when viewer connects
FetchContent_Declare (
    tracy
    GIT_REPOSITORY https://github.com/wolfpld/tracy.git
    GIT_TAG v0.14.1
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable ( tracy )


##########################################################################
########################### Nrd ###########################
##########################################################################

# see https://github.com/NVIDIA-RTX/NRD?tab=readme-ov-file#how-to-build
# set(NRD_DXC_CUSTOM_PATH ${DXC_CUSTOM_PATH})
set(NRD_DXC_PATH "${DXC_DXIL_EXECUTABLE}" CACHE STRING "DXC shader compiler path for NRD")
set(NRD_SHADER_OUTPUT_PATH "${SHADER_OUTPUT_PATH}" CACHE STRING "")
set(NRD_PROJECT_FOLDER "External/Nrd")
set(NRD_NORMAL_ENCODING "2" CACHE STRING "Normal encoding variant (0-4, matches nrd::NormalEncoding)")
set(NRD_ROUGHNESS_ENCODING "1" CACHE STRING "Roughness encoding variant (0-2, matches nrd::RoughnessEncoding)")
set(NRD_IS_SUBMODULE ON)
# NRD uses a custom output path for some reason
set(GLOBAL_BIN_OUTPUT_PATH ${CMAKE_BINARY_DIR} CACHE STRING "")
set(NRD_SHADERS_PATH "${CMAKE_BINARY_DIR}/NRDShaders" CACHE STRING "")
option(NRD_EMBEDS_SPIRV_SHADERS "NRD embeds SPIRV shaders" OFF)
option(NRD_EMBEDS_DXIL_SHADERS "NRD embeds DXIL shaders" OFF)
option(NRD_EMBEDS_DXBC_SHADERS "NRD embeds DXBC shaders" OFF)
option(NRD_DISABLE_SHADER_COMPILATION "" ON)
option(NRD_USE_PRECOMPILED_SHADERS "" OFF)

add_subdirectory("Nrd")
# Debug and Release share bin/, so give the Debug DLL its own name (as RtxptD) rather than overwrite the Release one
set_target_properties(NRD PROPERTIES DEBUG_POSTFIX "D")

# NRD's own Shaders.cfg already encodes the permutation macros (NRD_SIGNAL, NRD_MODE, ...) natively
# understood by ShaderMake, and NRD.hlsli auto-includes the CMake-generated NRDConfig.hlsli (which
# already bakes in NRD_NORMAL_ENCODING/NRD_ROUGHNESS_ENCODING) and auto-detects NRD_COMPILER_DXC,
# so no custom config generation or extra defines are needed here anymore. NRD_INTERNAL still needs
# to be passed explicitly (NRD's own CMakeLists.txt does the same for its embedded-shaders build,
# see SHADERMAKE_GENERAL_ARGS in External/Nrd/CMakeLists.txt) - it gates the DIFF/SPEC/BOTH and
# NRD_DIFF/NRD_SPEC permutation macros that Shaders.cfg's "-D NRD_SIGNAL={DIFF,SPEC,BOTH}" relies on.
set(nrd_shadermake_options "-D NRD_INTERNAL")

# ShaderMake emits -Zi only when --PDB or --embedPDB is given, so without this NRD's shaders carry no
# debug info at all and a profiler cannot resolve them - in Debug as well as Release, since nothing here
# varies by configuration. Embedded rather than separate PDBs so the symbols cannot be the wrong build.
# NRD's own CMakeLists never passes either flag, and its shader build is disabled above
# (NRD_DISABLE_SHADER_COMPILATION), so this target is the only place it can be set.
if(DONUT_WITH_AFTERMATH OR DONUT_EMBED_SHADER_PDBS)
    string(APPEND nrd_shadermake_options " --embedPDB")
endif()

donut_compile_shaders(
    TARGET nrd_shaders
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/Nrd/Shaders/Shaders.cfg
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Nrd/Shaders
    FOLDER "NRD"
    DXIL ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderPrecompiled/nrd/dxil
    SPIRV_DXC ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderPrecompiled/nrd/spirv
    SHADERMAKE_OPTIONS "${nrd_shadermake_options}"
    INCLUDES ${CMAKE_BINARY_DIR}/_deps/mathlib-src ${CMAKE_CURRENT_SOURCE_DIR}/Nrd/Shaders
)

##########################################################################
########################### Opacity Micro-Map SDK ########################
##########################################################################

option(OMM_INTEGRATION_LAYER_NVRHI "" ON)
option(OMM_ENABLE_PRECOMPILED_SHADERS_DXIL "" OFF)
option(OMM_ENABLE_PRECOMPILED_SHADERS_SPIRV "" OFF)
option(OMM_ENABLE_BENCHMARK "" OFF)
option(OMM_ENABLE_TESTS "" OFF)
option(OMM_BUILD_VIEWER "" OFF)
# OpenMP only serves the OMM CPU baker, which RTXPT doesn't use at the moment; FORCE as existing caches already hold ON
set(OMM_ENABLE_OPENMP OFF CACHE BOOL "" FORCE)

set(OMM_PROJECT_FOLDER "Omm")
add_subdirectory("Omm")
set_target_properties(omm-lib PROPERTIES DEBUG_POSTFIX "D")    # see NRD above

# Let CMake generate the omm_shaders.cfg file 
util_generate_shader_config_file(
    "Omm.cfg" 
    "Omm/libraries/omm-lib/shaders"
    "-D COMPILER_DXC=1"
)

# Compile all shaders in omm.cfg 
donut_compile_shaders(
    TARGET omm_shaders
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/Omm.cfg
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Omm/libraries/omm-lib/include
    FOLDER "Opacity Micro-Map SDK"
    DXIL ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderPrecompiled/omm/dxil
    SPIRV_DXC ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderPrecompiled/omm/spirv
)

##########################################################################
################################ RTXDI SDK ###############################
##########################################################################

set(RTXDI_SKIP_SHADER_VALIDATION ON CACHE STRING "Skip RTXDI shader validation")

add_subdirectory("Rtxdi")

##########################################################################
################################ cxxopts ###############################
##########################################################################

if (NOT TARGET cxxopts)
    add_subdirectory("cxxopts")
endif()