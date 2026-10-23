#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

cmake_minimum_required(VERSION 4.1)
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0175 OLD) # streamline error
# DirectX-Headers (fetched by nvrhi) calls include(CTest) unconditionally, which would add the Nightly/Continuous/
# Experimental CDash targets to the solution. We have no dashboard, so suppress them.
set(BUILD_TESTING OFF CACHE BOOL "Build the testing tree." FORCE)

project(RTXPathTracing)

# Sets NVRHI_TARGET_ARCH and NVRHI_HOST_ARCH (arm64, x64 or x86) the same way nvrhi and Donut do; ARM64EC is not supported
include("${CMAKE_CURRENT_SOURCE_DIR}/External/Donut/nvrhi/cmake/NvrhiTargetArch.cmake")
# Cross-building makes Donut build ShaderMake as a host tool; other generators build it with the ARM64 compiler, which an x64 host can't run
if (NVRHI_TARGET_ARCH STREQUAL "arm64" AND NOT NVRHI_HOST_ARCH STREQUAL "arm64" AND NOT CMAKE_GENERATOR MATCHES "^Visual Studio")
    message(FATAL_ERROR "Cross-building for ARM64 on a ${NVRHI_HOST_ARCH} host requires a Visual Studio generator (-A ARM64).")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)    # no C++20 modules here; CMake 3.28+ (CMP0155) would scan every source for them

set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

if (MSVC)
	set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

	# Workaround for a Microsoft "fix" to std::mutex that happened in VS2022 v17.10.
	# Using std::mutex with a DLL version of MSVCRT that is older than that VS version will lead to a crash.
	# See https://developercommunity.visualstudio.com/t/Access-violation-with-std::mutex::lock-a/10664660
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR")

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")

    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_ITERATOR_DEBUG_LEVEL=1")

    # Only affect Release configuration
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi" CACHE STRING "" FORCE)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /Zi" CACHE STRING "" FORCE)
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG" CACHE STRING "" FORCE)
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG" CACHE STRING "" FORCE)
endif()

# Register project CPU tests without include(CTest), keeping CDash targets disabled.
enable_testing()
add_subdirectory(Testing/UnitTests)

# we can't run with DX11
option(DONUT_WITH_DX11 OFF)

# default VULKAN to OFF
option(DONUT_WITH_VULKAN "Enable the Vulkan version of Donut" OFF)

# KTX2 loading defaults to ON upstream and pulls facebook/zstd via FetchContent at configure time; we ship no .ktx2
# assets, so opt out rather than take a network dependency the build does not otherwise need.
option(DONUT_WITH_KTX "" OFF)

# we want shader PDBs always embedded
option(DONUT_EMBED_SHADER_PDBS "" ON)

set (CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

set(RTXPT_LOCAL_CONFIG_ID "NONAME" CACHE STRING "Local user initials for user-specific settings")
add_definitions( -DRTXPT_LOCAL_CONFIG_ID_STRING=std::string\(\"${RTXPT_LOCAL_CONFIG_ID}\"\) )

#----------------------------------------------------------------------------------------------------------------------
# Helper to download and unzip a package from a URL
# Uses a zero-length file to identify the version of the package
function(CheckAndDownloadPackage NAME VERSION LOCAL_PATH URL ARCHIVE_TYPE)
    # Do we already have the correct version?
    if(NOT EXISTS ${LOCAL_PATH}/${VERSION}.ver)
        # Was there a previous version that we need to delete?
        if(EXISTS ${LOCAL_PATH})
            message(STATUS "Deleting old " ${NAME})
            file(REMOVE_RECURSE ${LOCAL_PATH})
        endif()
        message(STATUS "Obtaining " ${NAME} " " ${VERSION} " from " ${URL})
        file(DOWNLOAD ${URL} ${LOCAL_PATH}.${ARCHIVE_TYPE})
        message(STATUS "Extracting " ${NAME} " to " ${LOCAL_PATH})
        file(ARCHIVE_EXTRACT INPUT ${LOCAL_PATH}.${ARCHIVE_TYPE} DESTINATION ${LOCAL_PATH})
        file(REMOVE ${LOCAL_PATH}.${ARCHIVE_TYPE})
        # Create an empty file so we know which version we have
        file(WRITE ${LOCAL_PATH}/${VERSION}.ver)
    endif()
endfunction()
# use example:
# CheckAndDownloadPackage("DXC" "v1.7.2308" ${CMAKE_CURRENT_SOURCE_DIR}/external/dxc https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.7.2308/dxc_2023_08_14.zip "zip")
#----------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------------
# Streamline config
#set(FETCHCONTENT_QUIET OFF) <- use for a lot more detailed info from FetchContent
option(DONUT_WITH_STREAMLINE "Enable streamline, separate package required" ON)
set(STREAMLINE_DOWNLOAD_PATH "${CMAKE_SOURCE_DIR}/External/Streamline")

# Project-owned version pin. Plain variable plus FORCE on the cache sets: a plain CACHE set is ignored once the
# entry exists, so bumping the version here would otherwise lose silently to a stale build/CMakeCache.txt.
# Set RTXPT_STREAMLINE_USE_PINNED=OFF to supply DONUT_STREAMLINE_FETCH_URL/_SHA256 yourself.
set(RTXPT_STREAMLINE_VERSION "v2.14.16")
# ARM64 is a separate package, fetched to its own folder so that x64 and ARM64 build trees can coexist
if (NVRHI_TARGET_ARCH STREQUAL "arm64")
    set(RTXPT_STREAMLINE_PACKAGE "streamline-sdk-${RTXPT_STREAMLINE_VERSION}-aarch64.zip")
    set(RTXPT_STREAMLINE_SHA256 "66b6ee8eca0421ad7f58fb56858c949fac0cf4b67da12e2d0f9e5d33397ef44e")
    set(RTXPT_STREAMLINE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/External/Streamline-arm64")
else()
    set(RTXPT_STREAMLINE_PACKAGE "streamline-sdk-${RTXPT_STREAMLINE_VERSION}.zip")
    set(RTXPT_STREAMLINE_SHA256 "32041e51db0cab709a271b72a352281b690c2abbd669a827809401a5dd23956b")
    set(RTXPT_STREAMLINE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/External/Streamline")
endif()
option(RTXPT_STREAMLINE_USE_PINNED "Use the project-pinned Streamline package" ON)
if (RTXPT_STREAMLINE_USE_PINNED)
    set(DONUT_STREAMLINE_FETCH_URL "https://github.com/NVIDIA-RTX/Streamline/releases/download/${RTXPT_STREAMLINE_VERSION}/${RTXPT_STREAMLINE_PACKAGE}" CACHE STRING "URL to Streamline package to fetch from https://github.com/NVIDIA-RTX/Streamline/releases" FORCE)
    set(DONUT_STREAMLINE_FETCH_SHA256 "${RTXPT_STREAMLINE_SHA256}" CACHE STRING "SHA256 hash of Streamline package from https://github.com/NVIDIA-RTX/Streamline/releases" FORCE)
    # Also forced, or an ARM64 cache from before the per-arch folders would unpack the ARM64 package over the x64 one
    set(DONUT_STREAMLINE_FETCH_DIR "${RTXPT_STREAMLINE_DIR}" CACHE STRING "Directory to fetch Streamline to, empty uses build directory default" FORCE)
endif()
set(DONUT_STREAMLINE_FETCH_DIR "${RTXPT_STREAMLINE_DIR}" CACHE STRING "Directory to fetch Streamline to, empty uses build directory default")

option(STREAMLINE_FEATURE_DLSS_SR   "Include DLSS-SR dll"           ON  )
option(STREAMLINE_FEATURE_DLSS_RR   "Include DLSS-RR dll"           ON  )
option(STREAMLINE_FEATURE_NRD       "Include NRD dll"               OFF )
option(STREAMLINE_FEATURE_IMGUI     "Include Imgui dll"             ON  )
option(STREAMLINE_FEATURE_NVPERF    "Include NSight Perf SDK dll"   OFF )
option(STREAMLINE_FEATURE_REFLEX    "Include Reflex dll"            ON  )
option(STREAMLINE_FEATURE_NIS       "Include NIS dll"               OFF )
option(STREAMLINE_FEATURE_DLSS_FG   "Include DLSS-FG dll"           ON  )
option(STREAMLINE_FEATURE_DEEPDVC   "Include DEEPDVC dll"           OFF )
option(STREAMLINE_FEATURE_DIRECTSR  "Include DirectSR dll"          OFF )
#----------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------
# Get and unpack DXC for DirectX shader compilation
CheckAndDownloadPackage("DXC" "v1.9.2609.5" ${CMAKE_CURRENT_SOURCE_DIR}/external/dxc https://www.nuget.org/api/v2/package/Microsoft.Direct3D.DXC/1.9.2609.5 "zip")
set(DXC_CUSTOM_PATH "${CMAKE_SOURCE_DIR}/External/dxc/build/native/bin/${NVRHI_HOST_ARCH}/" CACHE STRING "Path to embedded dxc")   # runs on the build host
#----------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------
# Get the OptiX AI denoiser command line tool (Testing/RegressionTesting-GENERATE.bat, Sample::DenoisedScreenshot); too big
# to commit without LFS. Not needed to build, so a failed download only warns. Re-downloads if the hash doesn't match.
set(RTXPT_OPTIX_DENOISER_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Support/denoiser_OptiX/Denoiser.exe")
set(RTXPT_OPTIX_DENOISER_SHA256 "42dd0a099a3029f3271aa42a782aee06bbfd03bd2f466a2ecfa76c8666616058")
set(_denoiser_hash "")
if (EXISTS ${RTXPT_OPTIX_DENOISER_PATH})
    file(SHA256 ${RTXPT_OPTIX_DENOISER_PATH} _denoiser_hash)
endif()
if (NOT _denoiser_hash STREQUAL RTXPT_OPTIX_DENOISER_SHA256)
    message(STATUS "Obtaining OptiX denoiser 3.0 from https://github.com/DeclanRussell/NvidiaAIDenoiser")
    file(DOWNLOAD https://github.com/DeclanRussell/NvidiaAIDenoiser/releases/download/3.0/Denoiser.exe ${RTXPT_OPTIX_DENOISER_PATH} STATUS _denoiser_status)
    set(_denoiser_hash "")
    if (EXISTS ${RTXPT_OPTIX_DENOISER_PATH})
        file(SHA256 ${RTXPT_OPTIX_DENOISER_PATH} _denoiser_hash)
    endif()
    if (NOT _denoiser_hash STREQUAL RTXPT_OPTIX_DENOISER_SHA256)   # EXPECTED_HASH would make a mismatch fatal
        file(REMOVE ${RTXPT_OPTIX_DENOISER_PATH})
        message(WARNING "OptiX denoiser download failed (status: ${_denoiser_status}); regression testing and denoised screenshots won't work")
    endif()
endif()
#----------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------------
# Get and unpack the Agility SDK
set(RTXPT_D3D_AGILITY_SDK_VERSION_NAME "1.619.6" CACHE STRING "Optional Agility SDK version to download")
set_property(CACHE RTXPT_D3D_AGILITY_SDK_VERSION_NAME PROPERTY STRINGS "1.619.6" "1.717.0-preview" "1.719.0-preview" "")

if (RTXPT_D3D_AGILITY_SDK_VERSION_NAME)
    # Extract the numeric SDK version (e.g. "1.717.0-preview" -> "717")
    string(REGEX MATCH "^[0-9]+\\.([0-9]+)\\." _agility_ver_match "${RTXPT_D3D_AGILITY_SDK_VERSION_NAME}")
    set(_agility_numeric_ver "${CMAKE_MATCH_1}")
    if (NOT _agility_numeric_ver)
        message(FATAL_ERROR "Could not extract numeric version from: ${RTXPT_D3D_AGILITY_SDK_VERSION_NAME}")
    endif()

    set(RTXPT_D3D_AGILITY_SDK_PATH ${CMAKE_CURRENT_SOURCE_DIR}/external/AgilitySDK/${RTXPT_D3D_AGILITY_SDK_VERSION_NAME} CACHE STRING "AgilitySDKPath" FORCE)
    set(RTXPT_D3D_AGILITY_SDK_VERSION "${_agility_numeric_ver}" CACHE STRING "AgilitySDKVersion" FORCE)
    CheckAndDownloadPackage("Agility SDK" "${RTXPT_D3D_AGILITY_SDK_VERSION_NAME}"
        ${RTXPT_D3D_AGILITY_SDK_PATH} "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/${RTXPT_D3D_AGILITY_SDK_VERSION_NAME}" "zip")
else()
    set(RTXPT_D3D_AGILITY_SDK_PATH "" CACHE STRING "AgilitySDKPath" FORCE)
    set(RTXPT_D3D_AGILITY_SDK_VERSION "" CACHE STRING "AgilitySDKVersion" FORCE)
endif()

#----------------------------------------------------------------------------------------------------------------------
option(NVRHI_WITH_NVAPI "Include NVAPI support (requires NVAPI SDK)" ON)
set(NVAPI_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/External/nvapi/" CACHE STRING "Path to NVAPI include headers/shaders" )
if (NVRHI_TARGET_ARCH STREQUAL "arm64")
    set(NVAPI_LIBRARY "${CMAKE_SOURCE_DIR}/External/nvapi/aarch64/nvapia64.lib" CACHE STRING "Path to NVAPI .lib file")
else()
    set(NVAPI_LIBRARY "${CMAKE_SOURCE_DIR}/External/nvapi/amd64/nvapi64.lib" CACHE STRING "Path to NVAPI .lib file")
endif()

set(DXC_PATH "${DXC_CUSTOM_PATH}/dxc.exe" CACHE STRING "Path to embedded dxc file for DX12")
set(DXC_DXIL_EXECUTABLE "${DXC_CUSTOM_PATH}/dxc.exe" CACHE STRING "Path to embedded dxc file for DX12")
set(DXC_SPIRV_PATH "${DXC_CUSTOM_PATH}/dxc.exe" CACHE STRING "Path to embedded dxc file for SPIRV")

set(SHADERMAKE_FIND_DXC OFF CACHE BOOL  "" FORCE)
set(SHADERMAKE_FIND_DXC_VK OFF CACHE BOOL "" FORCE)
set(SHADERMAKE_DXC_PATH "${DXC_PATH}" CACHE STRING "" FORCE)
set(SHADERMAKE_DXC_VK_PATH "${DXC_PATH}" CACHE STRING "" FORCE)

# option(USE_RAY_TRACING_DENOISER "Include NRD as part of the sample." ON)

# Other architectures get their own output folder (bin_arm64), so they can be built and packaged alongside x64
if (NVRHI_TARGET_ARCH STREQUAL "x64")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin_${NVRHI_TARGET_ARCH}")
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(DONUT_SHADERS_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ShaderPrecompiled/framework")
option(DONUT_WITH_LZ4 "Include LZ4" OFF)

# Default to using DXR OMMs if we can; set before External, whose nvrhi reads it at configure time
if(RTXPT_D3D_AGILITY_SDK_VERSION AND (RTXPT_D3D_AGILITY_SDK_VERSION GREATER_EQUAL "619"))
    set(NVRHI_D3D12_WITH_DXR12_OPACITY_MICROMAP ON CACHE BOOL "" FORCE)
else()
    set(NVRHI_D3D12_WITH_DXR12_OPACITY_MICROMAP OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory("External")

if (NVRHI_WITH_VULKAN OR NVRHI_WITH_DX12)
	add_subdirectory(Rtxpt)
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Rtxpt)
endif()
#----------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------------
# Alternative way of obtaining assets (we're currently relying on git submodule but this is an option as well)
#include(FetchContent)
#set(RTXPT_ASSETS_FETCH_URL "https://gitlab-master.nvidia.com/rtx/path-tracing-sdk-assets")
#set(RTXPT_ASSETS_FETCH_TAG "main")
#set(RTXPT_ASSETS_FETCH_DIR "${CMAKE_SOURCE_DIR}/Assets")
#message(STATUS "Updating assets from " ${RTXPT_ASSETS_FETCH_URL} ", tag " ${RTXPT_ASSETS_FETCH_TAG} ", into folder " ${RTXPT_ASSETS_FETCH_DIR} )
#FetchContent_Declare(assets_download  GIT_REPOSITORY ${RTXPT_ASSETS_FETCH_URL}  GIT_TAG ${RTXPT_ASSETS_FETCH_TAG}  SOURCE_DIR ${RTXPT_ASSETS_FETCH_DIR}  GIT_PROGRESS TRUE)
#FetchContent_MakeAvailable(assets_download)
#----------------------------------------------------------------------------------------------------------------------
