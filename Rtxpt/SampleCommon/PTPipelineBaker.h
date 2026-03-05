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
#include <unordered_set>

#include "ShaderCompilerUtils.h"

namespace donut::vfs
{
    class RootFileSystem;
}

class PTPipelineVariant
{
    // contains all things needed to compile and build the library
    struct ShaderPermutation
    {
        std::filesystem::path                   ShaderSrcFileName;
        std::filesystem::path                   ShaderOutFileName;

        std::vector<donut::engine::ShaderMacro> CombinedAndSpecializedMacros;   // combined global + variant + per-material-shader-specialized (if any) macros

        std::string                             PermutationName;                // user readable name

        std::string                             CompiledHashHex;         // picosha2::k_digest_size
        std::string                             CompiledFileNameNoExt;
        std::string                             CompiledFullPath;

        std::string                             US_compileCmdLine;
        std::string                             US_compileError;
        ShaderPermutation *                     US_masterCopy = nullptr;

        nvrhi::ShaderLibraryHandle              ShaderLibrary;


        void                                    SetPath(const std::filesystem::path & path);
        void                                    FromMaterialPermutation(const std::string & shortUniqueDebugID, const std::vector<donut::engine::ShaderMacro> & macros, const struct MaterialShaderPermutation & msp);
        void                                    CompileIfNeeded();
        void                                    ResetShaderLibrary();
        void                                    LoadShaderLibraryIfNeeded(class PTPipelineBaker & baker);
    };

private:
    friend class PTPipelineBaker;
    PTPipelineVariant(const std::string & relativeSourcePath, const std::vector<donut::engine::ShaderMacro> & variantMacros, const std::shared_ptr<PTPipelineBaker> & baker, const std::string & shortUniqueDebugID, bool raygenOnly);
public:
    ~PTPipelineVariant();

public:

    // TODO: this will invalidate content; make sure to set m_localVerison to -1 and call ResetPipeline()
    //void                                    SetMacros(const std::vector<donut::engine::ShaderMacro> & variantMacros);

    const nvrhi::rt::ShaderTableHandle &    GetShaderTable() const { return m_shaderTable; }

private:
    void                                    UpdateStart(std::filesystem::file_time_type lastModifiedSourceCode);
    void                                    UpdateFinalize();
    int64_t                                 GetVersion() const { return m_localVersion; }

private:
    void                                    CompileIfNeeded_Enqueue(std::filesystem::file_time_type lastModifiedSourceCode);
    void                                    ResetPipeline();

private:

    int64_t                                 m_localVersion = -1;        // if it doesn't match Baker version, we're out of date
    nvrhi::rt::ShaderTableHandle            m_shaderTable;
    nvrhi::rt::PipelineHandle               m_pipeline;
    std::weak_ptr<class PTPipelineBaker>    m_baker;
    bool                                    m_rayGenOnly;
    bool                                    m_exportAnyHit = false;
    bool                                    m_exportMiss = false;
    std::shared_ptr<class PTPipelineBaker>  m_us_lockedBaker;


    std::vector<donut::engine::ShaderMacro> m_macros;                   // global macros (obtained from PTPipelineBaker::GetMacros)
    std::vector<donut::engine::ShaderMacro> m_combinedMacros;           // per-PTPipelineVariant macros 


    // Raygen and (optional) Miss
    ShaderPermutation                       m_raygen;
    
    // ClosestHit and AnyHit (AnyHit usually baked in and not separate)
    ShaderPermutation                       m_ubershaderMaterial;               // A generic, non-specialized-works-for-all-material (note: this will likely go away in the future)
    std::vector<ShaderPermutation>          m_specializedPerMaterial;           // See MaterialsBaker's m_shaderPermutationTable

    std::string                             m_shortUniqueDebugID;
};

struct HitGroupInfo
{
    std::shared_ptr<MaterialShaderPermutation> MaterialShaderPermutation;
    bool        HasAnyHitShader;

    std::string GetExportName() const;
    std::string GetShaderPermutationName() const;
    int         GetShaderPermutationIndex() const;
};

class PTPipelineBaker : public std::enable_shared_from_this<PTPipelineBaker>
{
public:
    PTPipelineBaker(nvrhi::IDevice* device, std::shared_ptr<class MaterialsBaker> & materialsBaker, nvrhi::BindingLayoutHandle bindingLayout, nvrhi::BindingLayoutHandle bindlessLayout);
    ~PTPipelineBaker();
    
    void                                Update(const std::shared_ptr<class ExtendedScene> & scene, unsigned int subInstanceCount, const std::function<void(std::vector<donut::engine::ShaderMacro> & macros)>& globalMacrosGetter, bool forceShaderReload);
    std::shared_ptr<PTPipelineVariant>  CreateVariant(const std::string & relativeSourcePath, std::vector<donut::engine::ShaderMacro> variantMacros, const std::string & shortUniqueDebugID, bool rayGenOnly = false );
    void                                ReleaseVariant(std::shared_ptr<PTPipelineVariant> & variant);
    
    const std::shared_ptr<class MaterialsBaker> & 
                                        GetMaterialsBaker() const           { return m_materialsBaker; }

private:
    friend class PTPipelineVariant;
    const std::vector<HitGroupInfo> &   GetPerSubInstanceHitGroup() const   { return m_perSubInstanceHitGroup; }
    const std::unordered_map<int, HitGroupInfo> &   
                                        GetUniqueHitGroups() const          { return m_uniqueHitGroups; }
    const std::vector<donut::engine::ShaderMacro> & 
                                        GetMacros() const                   { return m_macros; }
    int64_t                             GetVersion() const                  { return m_version; }

    nvrhi::BindingLayoutHandle          GetBindingLayout() const            { return m_bindingLayout; }
    nvrhi::BindingLayoutHandle          GetBindlessLayout() const           { return m_bindlessLayout; }
    nvrhi::DeviceHandle                 GetDevice() const                   { return m_device; }

    const std::filesystem::path &       GetShaderBinariesPath() const       { return m_compilerConfig.ShaderBinariesPath; }
    const std::filesystem::path &       GetShaderCompilerPath() const       { return m_compilerConfig.ShaderCompilerPath; }
    const std::filesystem::path &       GetShadersPath() const              { return m_compilerConfig.ShadersPath; }
    const std::filesystem::path &       GetShadersPathExternalIncludes1() const { return m_compilerConfig.ShadersPathExternalIncludes1; }
    const std::filesystem::path &       GetShadersPathExternalIncludes2() const { return m_compilerConfig.ShadersPathExternalIncludes2; }

    bool                                IsVerbose() const                       { return m_verbose; }
    bool                                IsNVAPIShaderExtensionEnabled() const   { return m_enableNVAPIShaderExtension; }

    const ShaderCompilerUtils::ShaderCompilerConfig& 
                                        GetCompilerConfig() const           { return m_compilerConfig; }

    std::shared_ptr<donut::vfs::RootFileSystem>     
                                        GetFS()                             { return m_shadersFS; }
    std::mutex &                        GetMutex()                          { return m_mutex; }

    void                                EnqueueShaderPermutation(PTPipelineVariant::ShaderPermutation* perm);

    bool                                RegisterShortUniqueDebugID(const std::string & id);
    void                                UnregisterShortUniqueDebugID(const std::string & id);


private:
    nvrhi::DeviceHandle                             m_device;
    std::shared_ptr<donut::vfs::RootFileSystem>     m_shadersFS;
    std::shared_ptr<class MaterialsBaker> &         m_materialsBaker;
    nvrhi::BindingLayoutHandle                      m_bindingLayout;
    nvrhi::BindingLayoutHandle                      m_bindlessLayout;

    ShaderCompilerUtils::ShaderCompilerConfig       m_compilerConfig;

    std::vector<donut::engine::ShaderMacro>         m_macros;

    std::vector<std::shared_ptr<PTPipelineVariant>> m_variants;

    std::vector<PTPipelineVariant::ShaderPermutation*> m_parallelCompileListAll;
    std::unordered_map<std::string, PTPipelineVariant::ShaderPermutation*> m_parallelCompileListUnique; // one big list of enqueued files to compile, tracked by output name to avoid duplicates as multiple PTPipelineVariant-s can request same outputs

    std::unordered_set<std::string>                 m_shortUniqueDebugIDs;

    // this allows sharing of hitGroups between 
    friend class Sample;
    std::vector<HitGroupInfo>                       m_perSubInstanceHitGroup;
    std::unordered_map<int, HitGroupInfo>           m_uniqueHitGroups;

    std::optional<std::filesystem::file_time_type>  m_lastUpdatedSourceTimestamp;

    bool                                            m_verbose = false;
    bool                                            m_enableNVAPIShaderExtension = false;

    donut::engine::ThreadPool                       m_threadPool;

    int64_t                                         m_version = -1;

    std::mutex                                      m_mutex;    // for synchronizing work by variants

    // Auto-reload via polling
    float                                           m_autoReloadPollIntervalSeconds = 0.5f;
    std::optional<std::filesystem::file_time_type>  m_cachedSourceTimestamp;
};
