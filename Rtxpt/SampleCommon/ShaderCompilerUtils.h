/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <optional>
#include <nvrhi/nvrhi.h>
#include <donut/engine/ShaderFactory.h>

namespace ShaderCompilerUtils
{
    //////////////////////////////////////////////////////////////////////////
    // SHA256 hash computation for shader permutations
    //////////////////////////////////////////////////////////////////////////
    
    // Size of SHA256 digest in bytes (32 bytes = 256 bits)
    constexpr size_t k_Sha256DigestSize = 32;
    
    // Computes SHA256 hash of a string and returns it as a hex string
    std::string ComputeSha256Hex(const std::string& input);
    
    //////////////////////////////////////////////////////////////////////////
    // Shader compiler configuration
    //////////////////////////////////////////////////////////////////////////
    
    // Holds paths and configuration for the shader compiler
    struct ShaderCompilerConfig
    {
        std::filesystem::path   ShaderCompilerPath;             // Path to dxc.exe
        std::filesystem::path   ShadersPath;                    // Root path for shader sources
        std::filesystem::path   ShadersPathExternalIncludes1;   // First external include path
        std::filesystem::path   ShadersPathExternalIncludes2;   // Second external include path (can be empty)
        std::filesystem::path   ShaderBinariesPath;             // Output path for compiled binaries
        nvrhi::GraphicsAPI      GraphicsAPI;                    // D3D12 or Vulkan
        
        // Initializes the config by finding shader compiler and source paths
        // Returns true on success, false if required paths cannot be found
        bool Initialize(nvrhi::IDevice* device, const std::string& binarySubfolder);
        
        // Returns the shader compiler executable path as a quoted string for command-line use
        std::string GetCompilerPathQuoted() const;
    };
    
    //////////////////////////////////////////////////////////////////////////
    // DXC command-line builder
    //////////////////////////////////////////////////////////////////////////
    
    // Shader target profile
    enum class ShaderProfile
    {
        Library_6_6,    // lib_6_6 - for ray tracing shaders
        Library_6_9,    // lib_6_9 - for ray tracing shaders with newer features
        Compute_6_6,    // cs_6_6 - for compute shaders
        Compute_6_9,    // cs_6_9 - for compute shaders with newer features
    };
    
    // Options for building DXC command line
    struct DxcCommandOptions
    {
        std::filesystem::path   SourceFilePath;         // Full path to source .hlsl file
        ShaderProfile           Profile = ShaderProfile::Compute_6_6;
        std::string             EntryPoint;             // Entry point function name (empty for library targets)
        bool                    EnableDebugInfo = true;
        bool                    EmbedPdb = false;
        bool                    UseOptimizations = true;
        bool                    Enable16BitTypes = true;
        bool                    WarningsAsErrors = true;
        bool                    AllResourcesBound = true;
        bool                    EnableDebugPrint = true;
        
        // Additional macros to define
        std::vector<donut::engine::ShaderMacro> Macros;
        
        // Include paths (in addition to config's external includes)
        std::vector<std::filesystem::path> AdditionalIncludes;
    };
    
    // Result of building a DXC command line
    struct DxcCommandResult
    {
        std::string CommandBase;            // Command without output file specification
        std::string HashHex;                // SHA256 hash of CommandBase (for caching)
        std::string OutputFileNameNoExt;    // Suggested output filename without extension
    };
    
    // Builds a DXC command line for compiling a shader
    // Returns the command arguments (without the compiler path) and a hash for caching
    DxcCommandResult BuildDxcCommand(
        const ShaderCompilerConfig& config,
        const DxcCommandOptions& options);
    
    // Appends output file specification to a command
    // fullOutputPath: full path for the output .bin file
    // pdbPath: optional path for external PDB (if not embedding)
    std::string AppendOutputToCommand(
        const std::string& commandBase,
        const std::string& fullOutputPath,
        const std::string& pdbPath = "");
    
    //////////////////////////////////////////////////////////////////////////
    // File timestamp utilities (wrappers around SampleCommon functions)
    //////////////////////////////////////////////////////////////////////////
    
    // Checks if a compiled shader file is up-to-date
    // Returns true if compiledFile exists and is newer than lastSourceModified
    bool IsCompiledShaderUpToDate(
        const std::filesystem::path& compiledFile,
        std::filesystem::file_time_type lastSourceModified);
    
} // namespace ShaderCompilerUtils
