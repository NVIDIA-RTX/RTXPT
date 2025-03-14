#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

cmake_minimum_required(VERSION 3.18)
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0175 OLD) # streamline error

project(RTXPathTracing)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

if (MSVC)
	set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

	# Workaround for a Microsoft "fix" to std::mutex that happened in VS2022 v17.10.
	# Using std::mutex with a DLL version of MSVCRT that is older than that VS version will lead to a crash.
	# See https://developercommunity.visualstudio.com/t/Access-violation-with-std::mutex::lock-a/10664660
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR")

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")

    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_ITERATOR_DEBUG_LEVEL=1")
endif()

# we can't run with DX11
option(DONUT_WITH_DX11 OFF)

# default VULKAN to OFF
option(DONUT_WITH_VULKAN OFF "Enable the Vulkan version of Donut")

# we want shader PDBs always embedded
option(DONUT_EMBED_SHADER_PDBS "" ON)

set (CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

set(RTXPT_LOCAL_CONFIG_ID "NONAME" CACHE STRING "Local user initials for user-specific settings")
add_definitions( -DRTXPT_LOCAL_CONFIG_ID_STRING=std::string\(\"${RTXPT_LOCAL_CONFIG_ID}\"\) )

#----------------------------------------------------------------------------------------------------------------------
# Streamline config
#set(FETCHCONTENT_QUIET OFF) <- use for a lot more detailed info from FetchContent
option(DONUT_WITH_STREAMLINE "Enable streamline, separate package required" ON)
set(STREAMLINE_DOWNLOAD_PATH "${CMAKE_SOURCE_DIR}/External/Streamline")

# Use this to optionally download Streamline (and not as a submodule)
#set(DONUT_STREAMLINE_FETCH_URL "https://github.com/NVIDIAGameWorks/Streamline.git" CACHE STRING "")
#set(DONUT_STREAMLINE_FETCH_TAG "v2.7.2" CACHE STRING "")
#set(DONUT_STREAMLINE_FETCH_DIR ${STREAMLINE_DOWNLOAD_PATH} CACHE STRING "Directory to fetch streamline to, empty uses build directory default")
set(DONUT_STREAMLINE_SEARCH_PATHS ${STREAMLINE_DOWNLOAD_PATH} CACHE STRING "Search paths for streamline package")

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


option(NVRHI_WITH_NVAPI "Include NVAPI support (requires NVAPI SDK)" ON)
set(NVAPI_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/External/nvapi/" CACHE STRING "Path to NVAPI include headers/shaders" )
set(NVAPI_LIBRARY "${CMAKE_SOURCE_DIR}/External/nvapi/amd64/nvapi64.lib" CACHE STRING "Path to NVAPI .lib file")

set(DXC_CUSTOM_PATH "${CMAKE_SOURCE_DIR}/External/dxc/bin/x64/" CACHE STRING "Path to embedded dxc")
set(DXC_PATH "${DXC_CUSTOM_PATH}/dxc.exe" CACHE STRING "Path to embedded dxc file for DX12")
set(DXC_DXIL_EXECUTABLE "${DXC_CUSTOM_PATH}/dxc.exe" CACHE STRING "Path to embedded dxc file for DX12")
#set(DXC_SPIRV_EXECUTABLE "${DXC_CUSTOM_PATH}/dxc.exe" CACHE STRING "Path to embedded dxc file for SPIRV") # default to using dxc from the Vulkan SDK; remove this to use embedded .dxc

option (SHADERMAKE_FIND_DXC_SPIRV "Toggles whether to search for DXC for SPIR-V" ON)

# option(USE_RAY_TRACING_DENOISER "Include NRD as part of the sample." ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(DONUT_SHADERS_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/framework")
option(DONUT_WITH_LZ4 "Include LZ4" OFF)

add_subdirectory("External")

if (NVRHI_WITH_VULKAN OR NVRHI_WITH_DX12)
	add_subdirectory(Rtxpt)
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Rtxpt)
endif()


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
