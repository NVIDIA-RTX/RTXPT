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

#include <mutex>
#include <memory>
#include <nvrhi/nvrhi.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/ThreadPool.h>
#include <unordered_map>

#include "ShaderCompilerUtils.h"

namespace donut::vfs
{
    class RootFileSystem;
}

class ComputeShaderVariant;

class ComputePipelineBaker : public std::enable_shared_from_this<ComputePipelineBaker>
{
public:
    // additionalMonitorPaths: extra directories to monitor for hot reload (in addition to the default ShadersPath)
    ComputePipelineBaker(nvrhi::IDevice* device, const std::vector<std::filesystem::path>& additionalMonitorPaths = {});
    ~ComputePipelineBaker();

    // Register a compute shader for hot reload management
    // Returns a variant handle that can be used to get the pipeline
    std::shared_ptr<ComputeShaderVariant> CreateVariant(
        const std::string& shaderSourcePath,
        const std::string& entryPoint,
        const std::vector<donut::engine::ShaderMacro>& macros,
        const nvrhi::BindingLayoutVector& bindingLayouts,
        const std::string& debugName);

    // Release a previously created variant
    void ReleaseVariant(std::shared_ptr<ComputeShaderVariant>& variant);

    // Called each frame - checks for changes and recompiles if needed
    // Set forceReload=true to force recompilation of all shaders
    void Update(bool forceReload = false);

    // Access to verbose mode for debugging
    bool IsVerbose() const { return m_verbose; }
    void SetVerbose(bool verbose) { m_verbose = verbose; }

private:
    friend class ComputeShaderVariant;

    nvrhi::DeviceHandle GetDevice() const { return m_device; }
    
    const ShaderCompilerUtils::ShaderCompilerConfig& GetCompilerConfig() const { return m_compilerConfig; }
    
    const std::filesystem::path& GetShaderBinariesPath() const { return m_compilerConfig.ShaderBinariesPath; }
    const std::filesystem::path& GetShaderCompilerPath() const { return m_compilerConfig.ShaderCompilerPath; }
    const std::filesystem::path& GetShadersPath() const { return m_compilerConfig.ShadersPath; }
    const std::filesystem::path& GetShadersPathExternalIncludes1() const { return m_compilerConfig.ShadersPathExternalIncludes1; }
    const std::filesystem::path& GetShadersPathExternalIncludes2() const { return m_compilerConfig.ShadersPathExternalIncludes2; }
    
    std::shared_ptr<donut::vfs::RootFileSystem> GetFS() { return m_shadersFS; }
    std::mutex& GetMutex() { return m_mutex; }

    int64_t GetVersion() const { return m_version; }

    // Enqueue a shader for compilation (thread-safe)
    void EnqueueShaderForCompilation(ComputeShaderVariant* variant);

private:
    nvrhi::DeviceHandle m_device;
    std::shared_ptr<donut::vfs::RootFileSystem> m_shadersFS;
    
    ShaderCompilerUtils::ShaderCompilerConfig m_compilerConfig;
    
    std::vector<std::shared_ptr<ComputeShaderVariant>> m_variants;
    
    // For parallel compilation
    std::vector<ComputeShaderVariant*> m_parallelCompileListAll;
    std::unordered_map<std::string, ComputeShaderVariant*> m_parallelCompileListUnique;
    
    donut::engine::ThreadPool m_threadPool;
    
    std::optional<std::filesystem::file_time_type> m_lastUpdatedSourceTimestamp;
    
    bool m_verbose = false;
    
    int64_t m_version = 0;
    
    std::mutex m_mutex;

    // Auto-reload via polling
    float m_autoReloadPollIntervalSeconds = 0.5f;
    std::optional<std::filesystem::file_time_type> m_cachedSourceTimestamp;
    
    // Additional directories to monitor for hot reload (besides ShadersPath)
    std::vector<std::filesystem::path> m_additionalMonitorPaths;
};

class ComputeShaderVariant
{
public:
    ~ComputeShaderVariant();

    // Get the compiled compute pipeline
    // Returns nullptr if pipeline is not yet compiled or compilation failed
    nvrhi::ComputePipelineHandle GetPipeline() const { return m_pipeline; }

    // Check if the variant needs to be updated (recompiled)
    bool NeedsUpdate() const;

    // Get the last compilation error (empty if no error)
    const std::string& GetCompileError() const { return m_compileError; }

    // Get debug name
    const std::string& GetDebugName() const { return m_debugName; }

private:
    friend class ComputePipelineBaker;

    ComputeShaderVariant(
        const std::string& shaderSourcePath,
        const std::string& entryPoint,
        const std::vector<donut::engine::ShaderMacro>& macros,
        const nvrhi::BindingLayoutVector& bindingLayouts,
        const std::string& debugName,
        const std::shared_ptr<ComputePipelineBaker>& baker);

    // Prepare compilation command (does not execute)
    void PrepareCompilation(std::filesystem::file_time_type lastModifiedSourceCode);

    // Execute the actual compilation (called from thread pool)
    void CompileIfNeeded();

    // Load shader blob and create pipeline
    void LoadShaderAndCreatePipeline();

    // Reset pipeline (forces recompile on next update)
    void ResetPipeline();

private:
    std::weak_ptr<ComputePipelineBaker> m_baker;
    std::shared_ptr<ComputePipelineBaker> m_lockedBaker; // temporary lock during update

    // Shader source info
    std::filesystem::path m_shaderSrcFileName;
    std::string m_entryPoint;
    std::vector<donut::engine::ShaderMacro> m_macros;
    nvrhi::BindingLayoutVector m_bindingLayouts;
    std::string m_debugName;

    // Compiled shader info
    std::string m_compiledHashHex;
    std::string m_compiledFileNameNoExt;
    std::string m_compiledFullPath;
    std::string m_compileCmdLine;  // empty if no recompile needed
    std::string m_compileError;

    // Pipeline
    nvrhi::ShaderHandle m_shader;
    nvrhi::ComputePipelineHandle m_pipeline;

    // Version tracking
    int64_t m_localVersion = -1;
};
