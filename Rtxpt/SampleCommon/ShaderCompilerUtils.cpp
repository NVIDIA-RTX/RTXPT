/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "ShaderCompilerUtils.h"
#include "SampleCommon.h"
#include "../Misc/picosha2.h"

#include <donut/app/ApplicationBase.h>
#include <donut/core/log.h>

namespace ShaderCompilerUtils
{
    //////////////////////////////////////////////////////////////////////////
    // SHA256 hash computation
    //////////////////////////////////////////////////////////////////////////
    
    static_assert(picosha2::k_digest_size == k_Sha256DigestSize, "SHA256 digest size mismatch");
    
    std::string ComputeSha256Hex(const std::string& input)
    {
        std::vector<unsigned char> hash(picosha2::k_digest_size);
        picosha2::hash256(input.begin(), input.end(), hash.begin(), hash.end());
        return picosha2::bytes_to_hex_string(hash.begin(), hash.end());
    }
    
    //////////////////////////////////////////////////////////////////////////
    // Shader compiler configuration
    //////////////////////////////////////////////////////////////////////////
    
    bool ShaderCompilerConfig::Initialize(nvrhi::IDevice* device, const std::string& binarySubfolder)
    {
        std::string graphicsAPIName;
        if (device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
            graphicsAPIName = "d3d12";
        else if (device->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN)
            graphicsAPIName = "vk";
        else
        {
            donut::log::error("Unsupported graphics API for shader compilation");
            return false;
        }
        
        GraphicsAPI = device->getGraphicsAPI();
        
        std::string platformName = "x64"; // add "arm64" for ARM
        
        ShaderCompilerPath = std::filesystem::absolute(
            donut::app::GetDirectoryWithExecutable() / "ShaderDynamic/Tools" / graphicsAPIName / platformName / "dxc.exe");
        
        if (!std::filesystem::exists(ShaderCompilerPath))
        {
            donut::log::error("Unable to find '%s' - required for runtime shader compilation", 
                ShaderCompilerPath.string().c_str());
            return false;
        }
        
        std::filesystem::path shaderSourcePathDevelopment = 
            donut::app::GetDirectoryWithExecutable() / "../Rtxpt/Shaders";
        std::filesystem::path shaderSourcePathRuntime = 
            donut::app::GetDirectoryWithExecutable() / "ShaderDynamic/Source/Rtxpt";
        
        if (!std::filesystem::exists(shaderSourcePathDevelopment))
        {
            donut::log::info("Shaders development folder '%s' not found, trying local '%s'...", 
                shaderSourcePathDevelopment.string().c_str(), 
                shaderSourcePathRuntime.string().c_str());
                
            if (!std::filesystem::exists(shaderSourcePathRuntime))
            {
                donut::log::error("Unable to find shaders folder '%s' - required for shader compilation", 
                    shaderSourcePathRuntime.string().c_str());
                return false;
            }
            
            ShadersPath = shaderSourcePathRuntime;
            ShadersPathExternalIncludes1 = 
                donut::app::GetDirectoryWithExecutable() / "ShaderDynamic/Source/External";
            ShadersPathExternalIncludes2 = std::filesystem::path();
        }
        else
        {
            ShadersPath = shaderSourcePathDevelopment;
            ShadersPathExternalIncludes1 = 
                donut::app::GetDirectoryWithExecutable() / "../External/Donut/Include";
            ShadersPathExternalIncludes2 = 
                donut::app::GetDirectoryWithExecutable() / "../External";
        }
        
        ShaderBinariesPath = donut::app::GetDirectoryWithExecutable() / binarySubfolder / 
            donut::app::GetShaderTypeName(device->getGraphicsAPI());
        
        // Convert all paths to absolute
        ShaderCompilerPath = std::filesystem::absolute(ShaderCompilerPath);
        ShadersPath = std::filesystem::absolute(ShadersPath);
        ShadersPathExternalIncludes1 = std::filesystem::absolute(ShadersPathExternalIncludes1);
        if (!ShadersPathExternalIncludes2.empty())
            ShadersPathExternalIncludes2 = std::filesystem::absolute(ShadersPathExternalIncludes2);
        ShaderBinariesPath = std::filesystem::absolute(ShaderBinariesPath);
        
        donut::log::info("Shader compiler: '%s'", ShaderCompilerPath.string().c_str());
        donut::log::info("Shader sources: '%s' (includes: '%s', '%s')", 
            ShadersPath.string().c_str(), 
            ShadersPathExternalIncludes1.string().c_str(), 
            ShadersPathExternalIncludes2.string().c_str());
        donut::log::info("Shader binaries output: '%s'", ShaderBinariesPath.string().c_str());
        
        return true;
    }
    
    std::string ShaderCompilerConfig::GetCompilerPathQuoted() const
    {
        return "\"" + ShaderCompilerPath.string() + "\"";
    }
    
    //////////////////////////////////////////////////////////////////////////
    // DXC command-line builder
    //////////////////////////////////////////////////////////////////////////
    
    static std::string GetProfileString(ShaderProfile profile)
    {
        switch (profile)
        {
        case ShaderProfile::Library_6_6:    return "lib_6_6";
        case ShaderProfile::Library_6_9:    return "lib_6_9";
        case ShaderProfile::Compute_6_6:    return "cs_6_6";
        case ShaderProfile::Compute_6_9:    return "cs_6_9";
        default:                            return "cs_6_6";
        }
    }
    
    static bool IsLibraryProfile(ShaderProfile profile)
    {
        return profile == ShaderProfile::Library_6_6 || profile == ShaderProfile::Library_6_9;
    }
    
    DxcCommandResult BuildDxcCommand(
        const ShaderCompilerConfig& config,
        const DxcCommandOptions& options)
    {
        DxcCommandResult result;
        
        auto srcFullPath = std::filesystem::absolute(options.SourceFilePath);
        
        // See https://simoncoenen.com/blog/programming/graphics/DxcCompiling for switch reference
        std::string command;
        
        // Source file
        command += " \"" + srcFullPath.string() + "\"";
        
        // Debug info
        if (options.EnableDebugInfo)
        {
            command += " -Zi";              // Enable debug information
            if (options.EmbedPdb)
                command += " -Qembed_debug"; // Embed PDB in shader container
        }
        
        // Hash based on binary output only
        command += " -Zsb";
        
        // Optimization level
        if (options.UseOptimizations)
            command += " -O3";
        else
            command += " -Od";
        
        // 16-bit types
        if (options.Enable16BitTypes)
            command += " -enable-16bit-types";
        
        // Warnings as errors
        if (options.WarningsAsErrors)
            command += " -WX";
        
        // All resources bound
        if (options.AllResourcesBound)
            command += " -all_resources_bound";
        
        // Shader profile/target
        command += " -T " + GetProfileString(options.Profile);
        
        // Entry point (only for non-library targets)
        if (!IsLibraryProfile(options.Profile) && !options.EntryPoint.empty())
            command += " -E " + options.EntryPoint;
        
        // Payload qualifiers for older library profiles
        if (options.Profile == ShaderProfile::Library_6_6)
            command += " -enable-payload-qualifiers";
        
        // Debug print macro
        if (options.EnableDebugPrint)
            command += " -D ENABLE_DEBUG_PRINT";
        
        // User-defined macros
        for (const auto& macro : options.Macros)
            command += " -D " + macro.name + "=" + macro.definition;
        
        // Include paths - config's external includes
        command += " -I \"" + config.ShadersPathExternalIncludes1.string() + "\"";
        if (!config.ShadersPathExternalIncludes2.empty())
            command += " -I \"" + config.ShadersPathExternalIncludes2.string() + "\"";
        
        // Additional include paths
        for (const auto& includePath : options.AdditionalIncludes)
            command += " -I \"" + includePath.string() + "\"";
        
        // Target API macro
        std::string targetMacro = " -D ";
        if (config.GraphicsAPI == nvrhi::GraphicsAPI::D3D12)
        {
            targetMacro += "TARGET_D3D12";
        }
        else if (config.GraphicsAPI == nvrhi::GraphicsAPI::VULKAN)
        {
            targetMacro += "TARGET_VULKAN";
        }
        command += targetMacro;
        
        // Vulkan-specific options
        if (config.GraphicsAPI == nvrhi::GraphicsAPI::VULKAN)
        {
            command += " -D SPIRV";
            command += " -spirv";
            command += " -fspv-target-env=vulkan1.2";
            command += " -fspv-extension=SPV_EXT_descriptor_indexing";
            command += " -fspv-extension=KHR";
            
            nvrhi::VulkanBindingOffsets cBindingOffsets;
            for (int i = 0; i < 7; i++)
            {
                // TODO: test with 'all' instead of the second %d - should work as well
                command += StringFormat(" -fvk-s-shift %d %d", cBindingOffsets.sampler, i);
                command += StringFormat(" -fvk-t-shift %d %d", cBindingOffsets.shaderResource, i);
                command += StringFormat(" -fvk-b-shift %d %d", cBindingOffsets.constantBuffer, i);
                command += StringFormat(" -fvk-u-shift %d %d", cBindingOffsets.unorderedAccess, i);
            }
        }
        
        result.CommandBase = command;
        
        // Compute hash of command (which includes all macros) but NOT the full file path
        // This avoids recompiling if just moving folders around
        result.HashHex = ComputeSha256Hex(command);
        
        // Build suggested output filename
        std::filesystem::path outFileName = options.SourceFilePath;
        outFileName.replace_extension(); // Remove .hlsl extension
        result.OutputFileNameNoExt = outFileName.filename().string() + "_" + result.HashHex;
        
        return result;
    }
    
    std::string AppendOutputToCommand(
        const std::string& commandBase,
        const std::string& fullOutputPath,
        const std::string& pdbPath)
    {
        std::string command = commandBase;
        
        if (!pdbPath.empty())
            command += " /Fd \"" + pdbPath + "\"";
        
        command += " -Fo \"" + fullOutputPath + "\"";
        
        return command;
    }
    
    //////////////////////////////////////////////////////////////////////////
    // File timestamp utilities
    //////////////////////////////////////////////////////////////////////////
    
    bool IsCompiledShaderUpToDate(
        const std::filesystem::path& compiledFile,
        std::filesystem::file_time_type lastSourceModified)
    {
        auto lastModifiedTime = GetFileModifiedTime(compiledFile);
        return lastModifiedTime.has_value() && (*lastModifiedTime) >= lastSourceModified;
    }
    
} // namespace ShaderCompilerUtils
