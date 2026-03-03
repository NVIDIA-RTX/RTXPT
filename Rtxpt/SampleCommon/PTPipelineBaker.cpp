/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "PTPipelineBaker.h"
#include "ShaderCompilerUtils.h"

#include <donut/app/ApplicationBase.h>

#include "../Materials/MaterialsBaker.h"
#include "ExtendedScene.h"
#include "SampleCommon.h"
#include "../Shaders/PathTracer/PathTracerShared.h"

#define BAKER_ENABLE_MULTITHREADED_COMPILE_SHADER 1
#define BAKER_ENABLE_MULTITHREADED_COMPILE_PSO 1

// this will add pipeline variant's m_shortUniqueDebugID to raygen/miss and m_shortUniqueDebugID + material permutation name to all closesthit and anyhit shaders
// this is very useful for debugging
#define PIPELINE_BAKER_ENABLE_VERBOSE_FUNCTION_NAMING 1

// this is for testing
#define PIPELINE_BAKER_REVERT_TO_UBERSHADER 0

#define PIPELINE_BAKER_EMBED_PDBS 0 // otherwise export as .pdb-s

#define PIPELINE_BAKER_USE_OPTIMIZATIONS 1

using namespace donut;
using namespace donut::math;
using namespace donut::engine;

static const std::string c_PTShaderBinariesRoot = "ShaderDynamic/Bin";

void PTPipelineVariant::ShaderPermutation::SetPath(const std::filesystem::path & path)
{
    ShaderSrcFileName = path;
    assert(ShaderSrcFileName.extension().string() == ".hlsl");

    ShaderOutFileName = ShaderSrcFileName;
    ShaderOutFileName.replace_extension();
}

void PTPipelineVariant::ShaderPermutation::FromMaterialPermutation(const std::string & shortUniqueDebugID, const std::vector<donut::engine::ShaderMacro> & macros, const struct MaterialShaderPermutation & msp)
{
    SetPath( msp.ShaderFilePath );
    CombinedAndSpecializedMacros = macros;
    for (auto& macro : msp.Macros)
        CombinedAndSpecializedMacros.push_back(macro);
    #if PIPELINE_BAKER_ENABLE_VERBOSE_FUNCTION_NAMING
    std::string cleanNameSuffix = msp.UniqueMaterialName;
    #else
    std::string cleanNameSuffix = "unk";
    #endif

    PermutationName = shortUniqueDebugID + "_" + cleanNameSuffix;
    CombinedAndSpecializedMacros.push_back( {"RTXPT_MATERIAL_PERMUTATION_NAME", PermutationName } );  // used to rename ClosestHit (and, if any, AnyHit) to something readable in profilers and etc

    CombinedAndSpecializedMacros.push_back( {"RTXPT_SHADER_ID", std::to_string(msp.IndexInTable) } );  // used to rename ClosestHit (and, if any, AnyHit) to something readable in profilers and etc
}

void PTPipelineVariant::ShaderPermutation::CompileIfNeeded()
{
    if (US_compileCmdLine == "")
        return;

    donut::log::info("Compiling shader '%s' file '%s'...", PermutationName.c_str(), CompiledFullPath.c_str() );

    auto [resNum, resString, resErrorString] = SystemShell(US_compileCmdLine, false);

    US_compileError = "";

    if (resErrorString != "")
        US_compileError = StringFormat("ERROR compiling shader, command \n   %s\n result: \n   %s", US_compileCmdLine.c_str(), resErrorString.c_str());

    if (US_compileError != "")
        donut::log::warning(US_compileError.c_str());
}

void PTPipelineVariant::ShaderPermutation::ResetShaderLibrary()
{
    ShaderLibrary = nullptr;
}

void PTPipelineVariant::ShaderPermutation::LoadShaderLibraryIfNeeded(PTPipelineBaker & baker)
{
    if (US_compileError != "" || ShaderLibrary != nullptr)
        return;

    std::shared_ptr<donut::vfs::IBlob> data = baker.GetFS()->readFile(("/" + c_PTShaderBinariesRoot + "/" + CompiledFileNameNoExt + ".bin").c_str());

    if (data)
        ShaderLibrary = baker.GetDevice()->createShaderLibrary(data->data(), data->size());

    if (!ShaderLibrary)
        US_compileError = StringFormat("ERROR creating ShaderLibrary for file %s", CompiledFullPath.c_str());
}

PTPipelineVariant::PTPipelineVariant(const std::string & relativeSourcePath, const std::vector<donut::engine::ShaderMacro> & variantMacros, const std::shared_ptr<PTPipelineBaker>& baker, const std::string & shortUniqueDebugID, bool raygenOnly)
    : m_macros(variantMacros), m_baker(baker), m_rayGenOnly(raygenOnly)
{
    m_raygen.SetPath(relativeSourcePath);

    // short unique ID is only for debugging purposes - it will be baked into the shader as a part of the RTXPT_MATERIAL_PERMUTATION_NAME and also used to name ClosestHit and AnyHit
#if PIPELINE_BAKER_ENABLE_VERBOSE_FUNCTION_NAMING
    m_shortUniqueDebugID = StripNonAsciiAlnum(shortUniqueDebugID);
    assert( shortUniqueDebugID == m_shortUniqueDebugID ); // short unique debug ID must not contain any of the forbidden characters or bad things will happen, very bad
    if (m_shortUniqueDebugID.size()>6)
    {
        assert( false ); // too big
        m_shortUniqueDebugID = m_shortUniqueDebugID.substr(0, 6);
    }
    if (!baker->RegisterShortUniqueDebugID(m_shortUniqueDebugID))
    {
        assert( false ); // not unique!
    }
#else
    m_shortUniqueDebugID = "UNK";
#endif
}

PTPipelineVariant::~PTPipelineVariant()
{
    std::shared_ptr<PTPipelineBaker> & baker = m_us_lockedBaker; 
    if (baker)
        baker->UnregisterShortUniqueDebugID(m_shortUniqueDebugID);
}

void PTPipelineVariant::ResetPipeline()
{
    m_shaderTable = nullptr;
    m_pipeline = nullptr;
    m_localVersion = -1;
}

void PTPipelineVariant::CompileIfNeeded_Enqueue(std::filesystem::file_time_type lastModifiedSourceCode)
{
    std::shared_ptr<PTPipelineBaker> & baker = m_us_lockedBaker; assert( baker != nullptr );

    // no further specializations for these two
    m_raygen.CombinedAndSpecializedMacros = m_combinedMacros;
    m_raygen.PermutationName = m_shortUniqueDebugID + "_raygen";
    m_raygen.CombinedAndSpecializedMacros.push_back( {"RTXPT_PIPELINE_PERMUTATION_NAME", m_shortUniqueDebugID } );  // used to rename ClosestHit (and, if any, AnyHit) to something readable in profilers and etc
    
    m_ubershaderMaterial.FromMaterialPermutation( m_shortUniqueDebugID, m_combinedMacros, *baker->GetMaterialsBaker()->GetUbershader() );

    m_specializedPerMaterial.clear();
    const auto & sourceTable = baker->GetMaterialsBaker()->GetShaderPermutationTable();
    m_specializedPerMaterial.resize(sourceTable.size());
    for( int i = 0; i < sourceTable.size(); i++ )
        m_specializedPerMaterial[i].FromMaterialPermutation( m_shortUniqueDebugID, m_combinedMacros, *(sourceTable[i]) );

    std::vector<ShaderPermutation*>         currentList;

    currentList.clear();
    currentList.push_back(&m_raygen);
    currentList.push_back(&m_ubershaderMaterial);

    if (!m_rayGenOnly)
        for( int i = 0; i < m_specializedPerMaterial.size(); i++ )
            currentList.push_back( &(m_specializedPerMaterial[i]) );

    // start preparing the command - this is shared amongst all permutations
    const std::string commandBase = "\"" + baker->GetShaderCompilerPath().string() + "\"";
    bool resetPipelineNeeded = false;
    for( ShaderPermutation * permutation : currentList )
    {
        // must reset any past errors
        permutation->US_compileError = "";

        // source file
        auto srcFullPath = std::filesystem::absolute(baker->GetShadersPath() / permutation->ShaderSrcFileName);
        
        // see https://simoncoenen.com/blog/programming/graphics/DxcCompiling for switch reference
        std::string command;
        command += " \"" + srcFullPath.string() + "\"";
        command += " -Zi";              //  Enable debug information. Cannot be used together with -Zs  - DXC_ARG_DEBUG
#if PIPELINE_BAKER_EMBED_PDBS
        command += " -Qembed_debug";    //  Embed PDB in shader container (must be used with /Zi)
#endif
        command += " -Zsb";             //  Compute Shader Hash considering only output binary
#if PIPELINE_BAKER_USE_OPTIMIZATIONS
        command += " -O3";
#else
        command += " -Od";
#endif
        command += " -enable-16bit-types";
        command += " -WX";              //  Warnings are errors
        //command += " -Gfa";             //  Avoid flow control
        //command += " -Gfp";             //  Avoid flow control
        command += " -all_resources_bound";
#if RTXPT_D3D_AGILITY_SDK_VERSION >= 619
        command += " -T lib_6_9";
        //command += " -disable-payload-qualifiers";
        //command += " -Vd";
#else
        command += " -T lib_6_6";
        command += " -enable-payload-qualifiers";
#endif
        
        command += " -D ENABLE_DEBUG_PRINT"; // <- some issues with Linux and SPIRV? need to test this; also - test perf implications

        for (auto& macro : permutation->CombinedAndSpecializedMacros)
            command += " -D " + macro.name + "=" + macro.definition;

        command += " -I \"" + baker->GetShadersPathExternalIncludes1().string() + "\"";
        if (!baker->GetShadersPathExternalIncludes2().empty())
            command += " -I \"" + baker->GetShadersPathExternalIncludes2().string() + "\"";

        std::string targetMacro = " -D ";
        if (baker->GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
        {
            targetMacro += "TARGET_D3D12";
        }
        else if (baker->GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN)
        {
            targetMacro += "TARGET_VULKAN";
        }
        else assert(false);

        command += targetMacro;

        if (baker->GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN)
        {
            command += " -D SPIRV";
            command += " -spirv";
            command += " -fspv-target-env=vulkan1.2";
            command += " -fspv-extension=SPV_EXT_descriptor_indexing";
            command += " -fspv-extension=KHR";

            nvrhi::VulkanBindingOffsets cBindingOffsets;
            for (int i = 0; i < 7; i++)
            {
                // TODO: test with 'all' instead of the second %d - should work as well, no loop needed, see docs
                command += StringFormat(" -fvk-s-shift %d %d", cBindingOffsets.sampler, i);
                command += StringFormat(" -fvk-t-shift %d %d", cBindingOffsets.shaderResource, i);
                command += StringFormat(" -fvk-b-shift %d %d", cBindingOffsets.constantBuffer, i);
                command += StringFormat(" -fvk-u-shift %d %d", cBindingOffsets.unorderedAccess, i);
            }
        }

        std::string previousHashHex = permutation->CompiledHashHex;

        // adding all command params (which includes all macros) but NOT the full file path (just file name) - this is to avoid recompiling if just moving folders around
        std::string newHash = ShaderCompilerUtils::ComputeSha256Hex(command);
        if (newHash!=permutation->CompiledHashHex)
        {
            permutation->CompiledHashHex = newHash;
            resetPipelineNeeded = true;
            permutation->ResetShaderLibrary();
        }

        permutation->CompiledFileNameNoExt = permutation->ShaderOutFileName.string() + "_" + permutation->CompiledHashHex;
        permutation->CompiledFullPath = std::filesystem::absolute(baker->GetShaderBinariesPath() / permutation->CompiledFileNameNoExt).string() + ".bin";
        std::string compiledFullPathPdb = std::filesystem::absolute(baker->GetShaderBinariesPath() / permutation->CompiledFileNameNoExt).string() + ".pdb";

        // check if the file with the hash baked in exists and if it's newer than the last modified source (the latest modified file in the whole source directory)
        if (ShaderCompilerUtils::IsCompiledShaderUpToDate(permutation->CompiledFullPath, lastModifiedSourceCode))
        {
            if (baker->IsVerbose())
                donut::log::info("No need to compile shader variant of '%s', up-to-date file already exists...", srcFullPath.string().c_str());
            permutation->US_compileCmdLine = "";
        }
        else // we need to re-compile!
        {
            command = commandBase + command 
#if !PIPELINE_BAKER_EMBED_PDBS
                + " /Fd \"" + compiledFullPathPdb + "\"" 
#endif
                + " -Fo \"" + permutation->CompiledFullPath + "\"";

            if (baker->IsVerbose())
                donut::log::info("Enqueuing shader variant of '%s'...", srcFullPath.string().c_str());
            permutation->US_compileCmdLine = command;
            resetPipelineNeeded = true;
            permutation->ResetShaderLibrary();
        }
        baker->EnqueueShaderPermutation(permutation);
    }

    if (resetPipelineNeeded) // at least some need recompile - need to clear current PSO (pipeline)
        ResetPipeline();
}

//void PTPipelineVariant::CompileIfNeeded_Execute()
//{
//    std::shared_ptr<PTPipelineBaker>& baker = m_us_lockedBaker; assert(baker != nullptr);
//}

void PTPipelineVariant::UpdateStart(std::filesystem::file_time_type lastModifiedSourceCode)
{
    assert( m_us_lockedBaker == nullptr );

    std::shared_ptr<PTPipelineBaker> & baker = m_us_lockedBaker = m_baker.lock();
    if (baker == nullptr)
    {
        assert( false );
        return;
    }

    m_combinedMacros.clear(); m_combinedMacros.reserve( m_macros.size() + baker->GetMacros().size() );
    for (auto& macro : m_macros)
        m_combinedMacros.push_back(macro);
    for (auto& macro : baker->GetMacros())
        m_combinedMacros.push_back(macro);
    
    bool foundExportAnyHitDependency = false;
    for (auto& macro : m_combinedMacros)
    {
        if (macro.name == "USE_NVAPI_HIT_OBJECT_EXTENSION")
        {
            assert(!foundExportAnyHitDependency); // can and must have only 1
            if (macro.definition == "1")
                m_exportAnyHit = m_exportMiss = false;
            else
                m_exportAnyHit = m_exportMiss = true;
            foundExportAnyHitDependency = true;
        }
    }
    m_exportMiss = true; // it looks like not exporting a miss is a bug or considered a bug - need to verify
    assert(foundExportAnyHitDependency); // any changes in the way USE_NVAPI_HIT_OBJECT_EXTENSION is used?

    CompileIfNeeded_Enqueue(lastModifiedSourceCode);
}

void PTPipelineVariant::UpdateFinalize()
{
    std::shared_ptr<PTPipelineBaker> & baker = m_us_lockedBaker; assert( baker != nullptr );

    // if no changes, no need to re-create pipeline
    if (m_pipeline == nullptr )
    {
        std::string rayGenName  = ("RayGen_"+m_shortUniqueDebugID);
        std::string missName    = ("Miss_"+m_shortUniqueDebugID);

        nvrhi::rt::PipelineDesc pipelineDesc;
        pipelineDesc.globalBindingLayouts = { baker->GetBindingLayout(), baker->GetBindlessLayout() };
        pipelineDesc.shaders.push_back({ "", m_raygen.ShaderLibrary->getShader(rayGenName.c_str(), nvrhi::ShaderType::RayGeneration), nullptr });
        pipelineDesc.shaders.push_back({ "", m_raygen.ShaderLibrary->getShader(missName.c_str(), nvrhi::ShaderType::Miss), nullptr });
        pipelineDesc.allowOpacityMicromaps = true;
        
        if (!m_rayGenOnly)
        for (auto& [_, hitGroupInfo] : baker->GetUniqueHitGroups())
        {
            std::string shaderPermutationName = hitGroupInfo.GetShaderPermutationName();
            std::string closestHit = "ClosestHit_" + m_shortUniqueDebugID + "_" + shaderPermutationName;
            std::string anyHit = (m_exportAnyHit && hitGroupInfo.HasAnyHitShader)?("AnyHit_" + m_shortUniqueDebugID + "_" + shaderPermutationName):("");

#if PIPELINE_BAKER_REVERT_TO_UBERSHADER
            const ShaderPermutation & permutation = m_ubershaderMaterial;
#else
            const ShaderPermutation & permutation = m_specializedPerMaterial[hitGroupInfo.GetShaderPermutationIndex()];
#endif
            pipelineDesc.hitGroups.push_back(
                {
                    .exportName = hitGroupInfo.GetExportName(),
                    .closestHitShader = permutation.ShaderLibrary->getShader(closestHit.c_str(), nvrhi::ShaderType::ClosestHit),
                    .anyHitShader = (anyHit != "") ? (permutation.ShaderLibrary->getShader(anyHit.c_str(), nvrhi::ShaderType::AnyHit)) : (nullptr),
                    .intersectionShader = nullptr,
                    .bindingLayout = nullptr,
                    .isProceduralPrimitive = false
                }
            );
        }

        pipelineDesc.maxPayloadSize = PATH_TRACER_MAX_PAYLOAD_SIZE;
        pipelineDesc.maxRecursionDepth = 1; // 1 is enough if using inline visibility rays
        
        // NV HLSL extensions - DX12 only - we should probably expose some form of GetNvapiIsInitialized instead
        if (baker->IsNVAPIShaderExtensionEnabled())
            pipelineDesc.hlslExtensionsUAV = NV_SHADER_EXTN_SLOT_NUM;

        m_pipeline = baker->GetDevice()->createRayTracingPipeline(pipelineDesc);

        if (!m_pipeline)
            { assert( false ); return; }

        m_shaderTable = m_pipeline->createShaderTable();

        if (!m_shaderTable)
            { assert( false ); return; }

        m_shaderTable->setRayGenerationShader(rayGenName.c_str());

        if (!m_rayGenOnly)
        {
            auto& perSubInstanceHitGroup = baker->GetPerSubInstanceHitGroup();
            for (int i = 0; i < perSubInstanceHitGroup.size(); i++)
                m_shaderTable->addHitGroup(perSubInstanceHitGroup[i].GetExportName().c_str());
        }

        if (m_exportMiss)
            m_shaderTable->addMissShader(missName.c_str());

        m_localVersion = baker->GetVersion();

        assert(m_pipeline != nullptr && m_shaderTable != nullptr);
    }

    m_us_lockedBaker = nullptr;
}


PTPipelineBaker::PTPipelineBaker(nvrhi::IDevice* device, std::shared_ptr<MaterialsBaker>& materialsBaker, nvrhi::BindingLayoutHandle bindingLayout, nvrhi::BindingLayoutHandle bindlessLayout)
    : m_device(device)
    , m_materialsBaker(materialsBaker)
    , m_bindingLayout(bindingLayout)
    , m_bindlessLayout(bindlessLayout)
    , m_enableNVAPIShaderExtension(device->queryFeatureSupport(nvrhi::Feature::HlslExtensionUAV))
{
    if (!m_compilerConfig.Initialize(device, c_PTShaderBinariesRoot))
        donut::log::fatal("Failed to initialize shader compiler configuration");

    m_shadersFS = std::make_shared<vfs::RootFileSystem>();
    m_shadersFS->mount("/" + c_PTShaderBinariesRoot, m_compilerConfig.ShaderBinariesPath);
}

PTPipelineBaker::~PTPipelineBaker()
{
}

std::string HitGroupInfo::GetExportName() const { return "HitGroup_" + std::to_string(MaterialShaderPermutation->IndexInTable); }


#if PIPELINE_BAKER_ENABLE_VERBOSE_FUNCTION_NAMING
std::string HitGroupInfo::GetShaderPermutationName() const { return MaterialShaderPermutation->UniqueMaterialName; }
#else
std::string HitGroupInfo::GetShaderPermutationName() const { return "unk"; }
#endif

int HitGroupInfo::GetShaderPermutationIndex() const { return MaterialShaderPermutation->IndexInTable; }

HitGroupInfo ComputeSubInstanceHitGroupInfo(const MaterialsBaker & baker, const PTMaterial& material)
{
    HitGroupInfo info;

#if PIPELINE_BAKER_REVERT_TO_UBERSHADER
    info.MaterialShaderPermutation = baker.GetUbershader();
#else
    info.MaterialShaderPermutation = material.BakedShaderPermutation;
#endif

    info.HasAnyHitShader = material.HasAlphaTest();
    
    return info;
}

static bool macrosEqual(donut::engine::ShaderMacro& a, donut::engine::ShaderMacro& b)
{
    return a.name == b.name && a.definition == b.definition;
}

void PTPipelineBaker::EnqueueShaderPermutation(PTPipelineVariant::ShaderPermutation* perm)
{
    if( perm->US_compileCmdLine != "" )
    {   
        auto [it, inserted] = m_parallelCompileListUnique.try_emplace(perm->CompiledFileNameNoExt, perm);
        if (!inserted)
            perm->US_masterCopy = it->second;
    }
    m_parallelCompileListAll.push_back(perm);
}

void PTPipelineBaker::Update(const std::shared_ptr<class ExtendedScene>& scene, unsigned int subInstanceCount, const std::function<void(std::vector<donut::engine::ShaderMacro>& macros)>& globalMacrosGetter, bool forceShaderReload)
{
    // Auto-reload: poll for source file changes
    if (!forceShaderReload && !m_variants.empty())
    {
        static auto lastPollTime = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - lastPollTime).count();
        
        if (elapsed >= m_autoReloadPollIntervalSeconds)
        {
            lastPollTime = now;
            auto currentTimestamp = GetLatestModifiedTimeDirectoryRecursive(m_compilerConfig.ShadersPath);
            
            if (currentTimestamp.has_value())
            {
                if (!m_cachedSourceTimestamp.has_value())
                {
                    // First poll - just cache the timestamp
                    m_cachedSourceTimestamp = currentTimestamp;
                }
                else if (*currentTimestamp != *m_cachedSourceTimestamp)
                {
                    // Source files changed - trigger reload
                    m_cachedSourceTimestamp = currentTimestamp;
                    forceShaderReload = true;
                    log::info("RT shader source changes detected - triggering hot reload...");
                }
            }
        }
    }

    bool needsUpdate = m_uniqueHitGroups.size() == 0;

    std::vector<donut::engine::ShaderMacro> newMacros;
    globalMacrosGetter(newMacros);
    if (!std::equal(newMacros.begin(), newMacros.end(), m_macros.begin(), m_macros.end(), macrosEqual))
    {
        needsUpdate = true;
        m_macros = newMacros;
    }

    needsUpdate |= forceShaderReload;

    // no need to update these if already set up
    if (needsUpdate) // m_uniqueHitGroups.size() == 0)
    {
        assert(subInstanceCount > 0);
        // Note: these map 1-1 to m_subInstanceData, and are used to (see '->addHitGroup' below) build 1-1 mapped hit groups 
        m_perSubInstanceHitGroup.clear();
        m_perSubInstanceHitGroup.reserve(subInstanceCount);
        for (const auto& instance : scene->GetSceneGraph()->GetMeshInstances())
        {
            uint instanceID = (uint)m_perSubInstanceHitGroup.size();
            for (int gi = 0; gi < instance->GetMesh()->geometries.size(); gi++)
                m_perSubInstanceHitGroup.push_back(ComputeSubInstanceHitGroupInfo(*GetMaterialsBaker(), *PTMaterial::SafeCast(instance->GetMesh()->geometries[gi]->material)));
        }
        assert(m_perSubInstanceHitGroup.size() == subInstanceCount);

        // Prime the instances to make sure we only include the necessary CHS variants in the PSO. Many (sub)instances can map to same materials.
        m_uniqueHitGroups.clear();
        for (int i = 0; i < m_perSubInstanceHitGroup.size(); i++)
            m_uniqueHitGroups[m_perSubInstanceHitGroup[i].GetShaderPermutationIndex()] = m_perSubInstanceHitGroup[i];
        needsUpdate = true;
    }

    if (needsUpdate)
    {
        m_version++;
    }

    if (needsUpdate)
    {
        // we need the output folder
        EnsureDirectoryExists(m_compilerConfig.ShaderBinariesPath);

        std::optional<std::filesystem::file_time_type> a = GetLatestModifiedTimeDirectoryRecursive(m_compilerConfig.ShadersPath);
        // let's not track externals for perf reasons but here's the code in case it's needed
        //std::optional<std::filesystem::file_time_type> b = GetLatestModifiedTimeRecursive(m_compilerConfig.ShadersPathExternalIncludes1);
        //std::optional<std::filesystem::file_time_type> c = GetLatestModifiedTimeRecursive(m_compilerConfig.ShadersPathExternalIncludes2);
        m_lastUpdatedSourceTimestamp = a;
    }

    if (!m_lastUpdatedSourceTimestamp.has_value())
    {
        log::error("There is something wrong with the shader source path or logic - unable to dynamically compile shaders");
        return;
    }

    if (!needsUpdate)
        return;

    do // in case of compile errors allow user to modify and attempt recompile
    {
        assert( m_parallelCompileListAll.empty() && m_parallelCompileListUnique.empty() );

        std::vector<std::shared_ptr<PTPipelineVariant>> updateQueue;
        for (int i = 0; i < int(m_variants.size()); i++)
        {
            const std::shared_ptr<PTPipelineVariant>& variant = m_variants[i];
            if (variant.use_count() == 1)
                assert(false); // dangling Variant - forgotten a call to ReleaseVariant?
            if (variant->GetVersion() != m_version)
            {
                updateQueue.push_back(variant);
                variant->ResetPipeline();
                variant->UpdateStart(*m_lastUpdatedSourceTimestamp);
            }
        }

        std::atomic_int progressCounterCompleted;
        int progressTotal;

        ProgressBar progressCompilingShaders;
        progressCounterCompleted = 0;
        progressTotal = (int)m_parallelCompileListUnique.size();
        if (m_parallelCompileListUnique.size()>0)
            progressCompilingShaders.Start( StringFormat("Compiling shaders (%d)...", progressTotal).c_str() );


        for (auto it : m_parallelCompileListUnique)
        {
            PTPipelineVariant::ShaderPermutation* permutation = it.second;
            if (permutation->US_compileCmdLine == "")
            {
                assert(false); continue;
            } // not sure why this would happen

#if BAKER_ENABLE_MULTITHREADED_COMPILE_SHADER
            m_threadPool.AddTask([ this, permutation, &progressCompilingShaders, &progressCounterCompleted, progressTotal ]() {
#endif
                permutation->CompileIfNeeded();
                int completed = progressCounterCompleted.fetch_add(1) + 1;
                progressCompilingShaders.Set(0 + 100 * completed / progressTotal);

#if BAKER_ENABLE_MULTITHREADED_COMPILE_SHADER
            });
#endif
        }

        // wait for all to complete
#if BAKER_ENABLE_MULTITHREADED_COMPILE_SHADER
        m_threadPool.WaitForTasks();
#endif
 
        std::string firstError = "";

        // we've got to 
        for (PTPipelineVariant::ShaderPermutation* permutation : m_parallelCompileListAll)
        {
            // if error, break & skip
            if (permutation->US_compileError != "")
            {
                firstError = permutation->US_compileError;
                break;
            }

            permutation->LoadShaderLibraryIfNeeded(*this);
            // if error, break & skip
            if (permutation->ShaderLibrary == nullptr)
            {
                firstError = permutation->US_compileError;
                break;
            }
        }

        progressCompilingShaders.Set(100);
        m_parallelCompileListAll.clear();
        m_parallelCompileListUnique.clear();

        ProgressBar progressCompilingPSOs;
        progressCounterCompleted = 0;
        progressTotal = (int)updateQueue.size();
        progressCompilingPSOs.Start( StringFormat("Compiling PSOs (%d)...", progressTotal).c_str() );

        if (!updateQueue.empty() && firstError == "")
        {
            int updateQueueSize = (int)updateQueue.size();

            for (const std::shared_ptr<PTPipelineVariant>& variant : updateQueue)
            {
    #if BAKER_ENABLE_MULTITHREADED_COMPILE_PSO
                m_threadPool.AddTask([this, &variant, &progressCompilingPSOs, &progressCounterCompleted, progressTotal ](){
    #endif
                variant->UpdateFinalize();
                int completed = progressCounterCompleted.fetch_add(1)+1;
                progressCompilingPSOs.Set(0 + 99 * completed / progressTotal);
    #if BAKER_ENABLE_MULTITHREADED_COMPILE_PSO
                });
    #endif
            }
    #if BAKER_ENABLE_MULTITHREADED_COMPILE_PSO
            m_threadPool.WaitForTasks();
    #endif

            progressCompilingPSOs.Set(100);
        }


        if (firstError!="")
        {
#if _WIN32
            extern HWND HelpersGetActiveWindow();
            int result = MessageBoxA(HelpersGetActiveWindow(), firstError.c_str(),
                "Shader compile error", MB_RETRYCANCEL | MB_ICONWARNING | MB_SETFOREGROUND | MB_TASKMODAL);
            if (result == IDCANCEL)
#endif
                break;

            for (const std::shared_ptr<PTPipelineVariant>& variant : updateQueue)
                variant->m_us_lockedBaker = nullptr;
        }
        else
        {
            break;
        }
        
    } while (true);
}

void PTPipelineBaker::ReleaseVariant(std::shared_ptr<PTPipelineVariant>& variant)
{
    if (variant == nullptr)
        return;

    for (int i = int(m_variants.size()) - 1; i >= 0; i--)
    {
        if (m_variants[i] == variant)
        {
            m_variants.erase(m_variants.begin() + i);
            variant = nullptr;
            return;
        }
    }
    assert(false);
}

std::shared_ptr<PTPipelineVariant> PTPipelineBaker::CreateVariant(const std::string & relativeSourcePath, std::vector<donut::engine::ShaderMacro> variantMacros, const std::string & shortUniqueDebugID, bool rayGenOnly)
{
    std::shared_ptr<PTPipelineVariant> variant = std::shared_ptr<PTPipelineVariant>(new PTPipelineVariant(relativeSourcePath, variantMacros, this->shared_from_this(), shortUniqueDebugID, rayGenOnly));
    m_variants.push_back(variant);
    return variant;
}

bool PTPipelineBaker::RegisterShortUniqueDebugID(const std::string& id)
{
    auto [it, inserted] = m_shortUniqueDebugIDs.insert(id);

    return inserted;
}

void PTPipelineBaker::UnregisterShortUniqueDebugID(const std::string& id)
{
    int count = m_shortUniqueDebugIDs.erase(id);
    assert( count > 0 );
}
