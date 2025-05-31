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

namespace donut::vfs
{
    class RootFileSystem;
}

class PTPipelineVariant
{
private:
    friend class PTPipelineBaker;
    PTPipelineVariant(const std::string & relativeSourcePath, const std::vector<donut::engine::ShaderMacro> & variantMacros, const std::shared_ptr<PTPipelineBaker> & baker);
public:
    ~PTPipelineVariant();

public:

    // TODO: this will invalidate content; make sure m_shaderTable is deleted (ResetPipeline()) if macros different, and assert if null on GetShaderTable here
    //void                                    SetMacrios(const std::vector<donut::engine::ShaderMacro> & variantMacros);

    const nvrhi::rt::ShaderTableHandle &    GetShaderTable() const { return m_shaderTable; }

private:
    void                                    UpdateStart(std::filesystem::file_time_type lastModifiedSourceCode);
    void                                    UpdateFinalize();
    int64_t                                 GetVersion() const { return m_localVersion; }

private:
    void                                    CompileIfNeededPrepare(std::filesystem::file_time_type lastModifiedSourceCode);
    void                                    CompileIfNeededExecute();
    void                                    ResetPipeline();
    const std::string &                     LastCompileError() const { assert( !m_updateStarted.load() ); return m_us_compileError; }

private:

    std::vector<donut::engine::ShaderMacro> m_macros;
    std::vector<donut::engine::ShaderMacro> m_combinedMacros;

    std::filesystem::path                   m_shaderFileName;
    std::filesystem::path                   m_shaderFileOutName;
    
    nvrhi::ShaderLibraryHandle              m_shaderLibrary;
    nvrhi::rt::ShaderTableHandle            m_shaderTable;
    nvrhi::rt::PipelineHandle               m_pipeline;
    std::weak_ptr<class PTPipelineBaker>    m_baker;
    int64_t                                 m_localVersion = -1;     // if it doesn't match Baker version, we're out of date
    bool                                    m_exportAnyHit = false;

    std::string                             m_compiledHashHex;         // picosha2::k_digest_size
    std::string                             m_compiledFileNameNoExt;
    std::string                             m_compiledFullPath;

    std::atomic_bool                        m_updateStarted;
    std::string                             m_us_compileCmdLine;
    std::shared_ptr<class PTPipelineBaker>  m_us_lockedBaker;
    std::string                             m_us_compileError;
};

struct HitGroupInfo
{
    std::string ExportName;
    std::string ClosestHitShader;
    std::string AnyHitShader;
};

class PTPipelineBaker : public std::enable_shared_from_this<PTPipelineBaker>
{
public:
    PTPipelineBaker(nvrhi::IDevice* device, std::shared_ptr<class MaterialsBaker> & materialsBaker, nvrhi::BindingLayoutHandle bindingLayout, nvrhi::BindingLayoutHandle bindlessLayout);
    ~PTPipelineBaker();
    
    void                                Update(const std::shared_ptr<class ExtendedScene> & scene, unsigned int subInstanceCount, const std::function<void(std::vector<donut::engine::ShaderMacro> & macros)>& globalMacrosGetter, bool forceShaderReload);
    std::shared_ptr<PTPipelineVariant>  CreateVariant(const std::string & relativeSourcePath, std::vector<donut::engine::ShaderMacro> variantMacros);
    void                                ReleaseVariant(std::shared_ptr<PTPipelineVariant> & variant);

private:
    friend class PTPipelineVariant;
    const std::vector<HitGroupInfo> &   GetPerSubInstanceHitGroup() const   { return m_perSubInstanceHitGroup; }
    const std::unordered_map<std::string, HitGroupInfo> &   
                                        GetUniqueHitGroups() const          { return m_uniqueHitGroups; }
    const std::vector<donut::engine::ShaderMacro> & 
                                        GetMacros() const                   { return m_macros; }
    int64_t                             GetVersion() const                  { return m_version; }

    nvrhi::BindingLayoutHandle          GetBindingLayout() const            { return m_bindingLayout; }
    nvrhi::BindingLayoutHandle          GetBindlessLayout() const           { return m_bindlessLayout; }
    nvrhi::DeviceHandle                 GetDevice() const                   { return m_device; }

    const std::filesystem::path &       GetShaderBinariesPath() const       { return m_shaderBinariesPath; }
    const std::filesystem::path &       GetShaderCompilerPath() const       { return m_shaderCompilerPath; }
    const std::filesystem::path &       GetShadersPath() const              { return m_shadersPath; }
    const std::filesystem::path &       GetShadersPathExternalIncludes1() const { return m_shadersPathExternalIncludes1; }
    const std::filesystem::path &       GetShadersPathExternalIncludes2() const { return m_shadersPathExternalIncludes2; }

    bool                                IsVerbose() const                       { return m_verbose; }
    bool                                IsNVAPIShaderExtensionEnabled() const   { return m_enableNVAPIShaderExtension; }

    std::shared_ptr<donut::vfs::RootFileSystem>     
                                        GetFS()                             { return m_shadersFS; }
    std::mutex &                        GetMutex()                          { return m_mutex; }

private:
    nvrhi::DeviceHandle                             m_device;
    std::shared_ptr<donut::vfs::RootFileSystem>     m_shadersFS;
    std::shared_ptr<class MaterialsBaker> &         m_materialsBaker;
    nvrhi::BindingLayoutHandle                      m_bindingLayout;
    nvrhi::BindingLayoutHandle                      m_bindlessLayout;

    std::filesystem::path                           m_shaderBinariesPath;
    std::filesystem::path                           m_shaderCompilerPath;
    std::filesystem::path                           m_shadersPath;
    std::filesystem::path                           m_shadersPathExternalIncludes1;
    std::filesystem::path                           m_shadersPathExternalIncludes2;

    std::vector<donut::engine::ShaderMacro>         m_macros;

    std::vector<std::shared_ptr<PTPipelineVariant>> m_variants;

    friend class Sample;
    std::vector<HitGroupInfo>                       m_perSubInstanceHitGroup;
    std::unordered_map<std::string, HitGroupInfo>   m_uniqueHitGroups;

    std::optional<std::filesystem::file_time_type>  m_lastUpdatedSourceTimestamp;

    bool                                            m_verbose = true;
    bool                                            m_enableNVAPIShaderExtension = false;

    int64_t                                         m_version = -1;

    std::mutex                                      m_mutex;    // for synchronizing work by variants
};
