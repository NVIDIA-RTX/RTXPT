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

#include <algorithm>

#include <donut/app/ApplicationBase.h>

#include "../Materials/MaterialsBaker.h"
#include "ExtendedScene.h"
#include "SampleCommon.h"
#include "DirectoryModifiedMonitor.h"
#include "../Shaders/PathTracer/PathTracerShared.h"

#define BAKER_ENABLE_MULTITHREADED_COMPILE_SHADER 1
#define BAKER_ENABLE_MULTITHREADED_COMPILE_PSO 1

// this will add pipeline variant's m_shortUniqueDebugID to raygen/miss and m_shortUniqueDebugID + material permutation name to all closesthit and anyhit shaders
// this is very useful for debugging
#define PIPELINE_BAKER_ENABLE_VERBOSE_FUNCTION_NAMING 1

// this is for testing
#define PIPELINE_BAKER_REVERT_TO_UBERSHADER 0

#define PIPELINE_BAKER_EMBED_PDBS 1 // otherwise export as .pdb-s

// Must stay >= -O1 for any shader that fetches bindlessly (see Shaders/Bindless.hlsli): a bindless resource stored
// in a struct that crosses a function boundary is only representable once the optimizer scalarises the struct, and
// at -Od DXC 1.9 fails with an internal compiler error rather than a diagnostic. SHaRC hits this via SharcParameters.
#define PIPELINE_BAKER_USE_OPTIMIZATIONS 1

using namespace donut;
using namespace donut::math;
using namespace donut::engine;

// defined further down, next to macrosEqual; used by PTPipelineVariant::UpdateStart above it
static void ApplyMacroOverrides(std::vector<donut::engine::ShaderMacro>& dst, const std::vector<donut::engine::ShaderMacro>& overrides);

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

    // Ensure the output directory exists (shader paths may include subdirectories)
    std::filesystem::path outputDir = std::filesystem::path(CompiledFullPath).parent_path();
    EnsureDirectoryExists(outputDir);

    auto [resNum, resString, resErrorString] = SystemShell(US_compileCmdLine, false);

    US_compileError = "";

    if (resErrorString != "")
        US_compileError = StringFormat("ERROR compiling shader, command \n   %s\n result: \n   %s", US_compileCmdLine.c_str(), resErrorString.c_str());

    if (US_compileError != "")
        donut::log::warning(US_compileError.c_str());
    else
    {   // Stamped with the source snapshot, not the compile end time, so a source saved mid-compile still counts as newer
        std::error_code ec;
        std::filesystem::last_write_time(CompiledFullPath, US_sourceSnapshot, ec);
    }
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

PTPipelineVariant::PTPipelineVariant(const std::string & relativeSourcePath, const std::vector<donut::engine::ShaderMacro> & variantMacros, const std::shared_ptr<PTPipelineBaker>& baker, const std::string & shortUniqueDebugID, bool raygenOnly, std::function<void(std::vector<donut::engine::ShaderMacro> & macros)> variantMacrosGetter)
    : m_macros(variantMacros), m_baker(baker), m_rayGenOnly(raygenOnly), m_dynamicMacrosGetter(std::move(variantMacrosGetter))
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

        // Tracks the C++ build configuration so shaders can gate debug-only work on it (see ShaderDebug.hlsl).
        // ComputePipelineBaker injects the same macro via ShaderCompilerUtils::BuildDxcCommand; shaders precompiled
        // offline by ShaderMake get neither and fall back to the 0 default declared in ShaderDebug.hlsl. The value is
        // part of the hashed command line, so switching configuration recompiles rather than reusing a stale blob.
#if defined(_DEBUG)
        command += " -D RTXPT_SHADER_DEBUG=1";
#else
        command += " -D RTXPT_SHADER_DEBUG=0";
#endif

        for (auto& macro : permutation->CombinedAndSpecializedMacros)
            command += " -D " + macro.name + "=" + macro.definition;

        for (const auto& includePath : baker->GetIncludePaths())
            command += " -I \"" + includePath.string() + "\"";

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
            command += " -fvk-bind-resource-heap 0 2";  // ResourceDescriptorHeap: the mutable heap layout is the third (set 2) - see Bindless.hlsli

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

        // hash of all command params, including macros and the absolute source and include paths (so moving the install recompiles everything)
        std::string newHash = ShaderCompilerUtils::ComputeSha256Hex(command);
        if (newHash!=permutation->CompiledHashHex)
        {
            permutation->CompiledHashHex = newHash;
            resetPipelineNeeded = true;
            permutation->ResetShaderLibrary();
        }

        permutation->CompiledFileNameNoExt = permutation->ShaderOutFileName.string() + "_" + permutation->CompiledHashHex.substr(0, ShaderCompilerUtils::k_CacheFileHashHexChars);
        permutation->CompiledFullPath = std::filesystem::absolute(baker->GetShaderBinariesPath() / permutation->CompiledFileNameNoExt).string() + ".bin";
#if !PIPELINE_BAKER_EMBED_PDBS
        std::string compiledFullPathPdb = std::filesystem::absolute(baker->GetShaderBinariesPath() / permutation->CompiledFileNameNoExt).string() + ".pdb";
#endif

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
            permutation->US_sourceSnapshot = lastModifiedSourceCode;
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

    // Globals first, then let the per-variant macros override them by name. Order matters: these all become
    // '-D NAME=VALUE' on the DXC command line, and a redefinition with a *different* value is a hard error under -WX
    // (see the '-WX' below), so a colliding name must replace rather than append. Replacing in place also keeps the
    // command line (and therefore the compiled-binary hash) stable when nothing actually changed.
    m_combinedMacros.clear(); m_combinedMacros.reserve( baker->GetMacros().size() + m_macros.size() + m_dynamicMacros.size() );
    for (auto& macro : baker->GetMacros())
        m_combinedMacros.push_back(macro);
    ApplyMacroOverrides(m_combinedMacros, m_macros);          // static per-variant macros beat globals
    ApplyMacroOverrides(m_combinedMacros, m_dynamicMacros);   // runtime per-variant macros beat both
    
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
        // Appended LAST on purpose. nvrhi matches binding set i to pipeline layout i positionally, so callers that
        // do not fetch via ResourceDescriptorHeap (IntroSample) can keep dispatching these pipelines with only two
        // binding sets and the trailing layout is simply never visited. Note nvrhi validates the set/layout count
        // for compute and graphics but NOT for ray tracing, which is what makes that safe here - if that ever
        // changes, every rt::State dispatch of these pipelines has to pass the third set.
        if (baker->GetBindlessHeapLayout() != nullptr)
            pipelineDesc.globalBindingLayouts.push_back(baker->GetBindlessHeapLayout());
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

        pipelineDesc.maxPayloadSize = PATH_TRACER_MAX_PAYLOAD_SIZE; // with SM 6.7+ with PAQs enabled, this is ignored by the driver
        pipelineDesc.maxRecursionDepth = 1; // 1 is enough if using inline visibility rays
        
        // NV HLSL extensions - DX12 only - we should probably expose some form of GetNvapiIsInitialized instead
        if (baker->IsNVAPIShaderExtensionEnabled())
            pipelineDesc.hlslExtensionsUAV = NV_SHADER_EXTN_SLOT_NUM;

        m_pipeline = baker->GetDevice()->createRayTracingPipeline(pipelineDesc);

        // Left at a stale version: PTPipelineBaker::Update reports it like a compile error (Retry, or exit)
        if (!m_pipeline)
            { m_us_lockedBaker = nullptr; return; }

        // CACHED shader table. An uncached one is re-baked on its first use in every command list - nvrhi keeps the
        // build state in CommandList::m_UncachedShaderTableStates, which close() clears - so with one hit group entry per
        // sub-instance this was rewriting 64 bytes x sub-instance count, once per variant per frame, on the calling
        // thread. It profiled at ~38% of main thread time, dwarfing the actual ray dispatch it was preparing.
        //
        // Caching moves the build state onto the table itself (ShaderTable::cacheState), so it survives command list
        // close and the table is baked only when ShaderTable::version changes. Two properties of this call site make that
        // safe rather than merely faster:
        //  - the entry set cannot change behind the cache's back: it only changes when the baker bumps its version, and
        //    that path runs ResetPipeline() on every stale variant, which nulls m_shaderTable and re-enters this block.
        //    So maxEntries computed here can never be exceeded later by an addHitGroup on a live table.
        //  - every hit group above is built with bindingLayout == nullptr, so maxLocalRootParameters == 0 and
        //    hasLocalResources() is false. isStateValid() therefore reduces to a version compare and descriptor heap
        //    churn cannot force a re-bake either.
        //
        // Costs a shader-binding-table-sized buffer per variant (64B per entry, no local root args) plus one
        // CopyBufferRegion per rebuild. Note cacheState lives on the table, not per command list, so this assumes the
        // table is never recorded into two command lists concurrently - true here, all of this is main-thread recording.
        //
        // Must match ShaderTable::getNumEntries() == 1 + missShaders + hitGroups + callableShaders; the three add*
        // calls immediately below are exactly what is counted here, and callables are never used.
        nvrhi::rt::ShaderTableDesc shaderTableDesc;
        shaderTableDesc.isCached   = true;
        shaderTableDesc.maxEntries = 1u                                                                     // ray generation
                                   + (m_exportMiss ? 1u : 0u)                                               // miss
                                   + (m_rayGenOnly ? 0u : (uint32_t)baker->GetPerSubInstanceHitGroup().size());
        shaderTableDesc.debugName  = "PTShaderTable_" + m_shortUniqueDebugID;

        m_shaderTable = m_pipeline->createShaderTable(shaderTableDesc);

        if (!m_shaderTable)
            { m_us_lockedBaker = nullptr; return; }

        m_shaderTable->setRayGenerationShader(rayGenName.c_str());

        if (!m_rayGenOnly)
        {
            auto& perSubInstanceHitGroup = baker->GetPerSubInstanceHitGroup();
            for (int i = 0; i < perSubInstanceHitGroup.size(); i++)
                m_shaderTable->addHitGroup(perSubInstanceHitGroup[i].GetExportName().c_str());
        }

        if (m_exportMiss)
            m_shaderTable->addMissShader(missName.c_str());

        // The one failure mode of the cached path: if the entries ever outgrow maxEntries, nvrhi reports an error and
        // returns from setRayTracingState WITHOUT committing the dispatch template, i.e. it dispatches rays against a
        // stale one. Silent and extremely confusing, so catch the sizing drift here instead.
        assert( m_shaderTable->getNumEntries() == shaderTableDesc.maxEntries && "cached shader table sized wrong - see maxEntries above" );

        m_localVersion = baker->GetVersion();

        assert(m_pipeline != nullptr && m_shaderTable != nullptr);
    }

    m_us_lockedBaker = nullptr;
}


PTPipelineBaker::PTPipelineBaker(nvrhi::IDevice* device, std::shared_ptr<MaterialsBaker>& materialsBaker, nvrhi::BindingLayoutHandle bindingLayout, nvrhi::BindingLayoutHandle bindlessLayout, std::function<void(std::vector<donut::engine::ShaderMacro> & macros)> globalMacrosGetter, nvrhi::BindingLayoutHandle bindlessHeapLayout, std::vector<donut::engine::ShaderMacro> hitGroupMacros)
    : m_device(device)
    , m_materialsBaker(materialsBaker)
    , m_bindingLayout(bindingLayout)
    , m_bindlessLayout(bindlessLayout)
    , m_bindlessHeapLayout(bindlessHeapLayout)
    , m_globalMacrosGetter(std::move(globalMacrosGetter))
    , m_hitGroupMacros(std::move(hitGroupMacros))
    , m_enableNVAPIShaderExtension(device->queryFeatureSupport(nvrhi::Feature::HlslExtensionUAV))
{
    if (!m_compilerConfig.Initialize(device, c_PTShaderBinariesRoot))
        donut::log::fatal("Failed to initialize shader compiler configuration");

    // Hot-reload source watching runs on DirectoryModifiedMonitor's worker; walking this tree inline from Update() was a
    // ~1.4 ms render-thread hitch twice a second. ComputePipelineBaker registers the same root, and the monitor folds
    // the duplicate into a single scan.
    m_sourceMonitor = DirectoryModifiedMonitor::Register( { m_compilerConfig.ShadersPath } );

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

// Apply 'overrides' on top of 'dst': a same-name macro replaces in place (keeping its position), a new one is appended.
static void ApplyMacroOverrides(std::vector<donut::engine::ShaderMacro>& dst, const std::vector<donut::engine::ShaderMacro>& overrides)
{
    for (const auto& ov : overrides)
    {
        auto it = std::find_if(dst.begin(), dst.end(), [&ov](const donut::engine::ShaderMacro& m) { return m.name == ov.name; });
        if (it != dst.end())
            it->definition = ov.definition;
        else
            dst.push_back(ov);
    }
}

bool PTPipelineVariant::RefreshDynamicMacros()
{
    if (!m_dynamicMacrosGetter)
        return false;

    m_dynamicMacrosScratch.clear();
    m_dynamicMacrosGetter(m_dynamicMacrosScratch);

    if (std::equal(m_dynamicMacrosScratch.begin(), m_dynamicMacrosScratch.end(), m_dynamicMacros.begin(), m_dynamicMacros.end(), macrosEqual))
        return false;

    m_dynamicMacros = m_dynamicMacrosScratch;
    return true;
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

void PTPipelineBaker::Update(const std::shared_ptr<class ExtendedScene>& scene, unsigned int subInstanceCount, bool forceShaderReload)
{
    assert(m_globalMacrosGetter); // supplied at construction
    // Auto-reload: check for source file changes. No timer and no directory walk here any more - the monitor's worker
    // does the walking and this is a mutex plus a fold over one cached value.
    if (!forceShaderReload && !m_variants.empty())
    {
        ZoneScopedN("PTPipelineHotReloadCheck");
        auto currentTimestamp = DirectoryModifiedMonitor::Query(m_sourceMonitor);

        if (currentTimestamp.has_value())
        {
            if (!m_cachedSourceTimestamp.has_value())
            {
                // First observation - just cache the timestamp
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

    bool needsUpdate = m_uniqueHitGroups.size() == 0;

    std::vector<donut::engine::ShaderMacro> newMacros;
    {
        // Unlike the poll above this is NOT throttled - it runs every frame. GetPTPipelineGlobalMacros pushes 27
        // ShaderMacros, each two std::strings, and most of the names are well past MSVC's 15-char SSO limit - so this
        // is ~50+ malloc/free pairs plus the vector growth, every frame, purely to discover nothing changed.
        ZoneScopedN("PTPipelineGlobalMacrosCompare");
        m_globalMacrosGetter(newMacros);
    }
    if (!std::equal(newMacros.begin(), newMacros.end(), m_macros.begin(), m_macros.end(), macrosEqual))
    {
        needsUpdate = true;
        m_macros = newMacros;
    }

    // Per-variant runtime macro overrides. Cheap no-op for variants without a getter; for those that have one this is a
    // compare against the cached set, so a recompile only happens when the produced macros actually change. Note '|=',
    // not '||': every variant must be refreshed so its cache stays current.
    for (auto& variant : m_variants)
        needsUpdate |= variant->RefreshDynamicMacros();

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
        {
            // Alpha testing is not part of the permutation, so alpha tested and opaque geometry can share a hit group:
            // it needs the any-hit if any of them does. Opaque geometry never invokes it (GeometryFlags::Opaque).
            const HitGroupInfo& info = m_perSubInstanceHitGroup[i];
            HitGroupInfo& unique = m_uniqueHitGroups.try_emplace(info.GetShaderPermutationIndex(), info).first->second;
            unique.HasAnyHitShader |= info.HasAnyHitShader;
        }
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

        // ScanNow, not Query: this value feeds ShaderCompilerUtils::IsCompiledShaderUpToDate, which does
        // 'binaryTime >= lastSourceModified' per shader, so a cached (up to one poll interval stale) timestamp could
        // silently skip a compile. This runs only when a reload is already happening, where a 1.4 ms walk is nothing
        // next to the compilation it is about to kick off.
        std::optional<std::filesystem::file_time_type> a = DirectoryModifiedMonitor::ScanNow(m_sourceMonitor);
        // let's not track externals for perf reasons but here's the code in case it's needed
        //for (const auto& includePath : m_compilerConfig.IncludePaths)
        //    std::optional<std::filesystem::file_time_type> b = GetLatestModifiedTimeRecursive(includePath);
        m_lastUpdatedSourceTimestamp = a;

        // Sync the watcher's baseline with what this batch compiles; otherwise an edit it hasn't polled yet (e.g. saved just
        // before Ctrl+R) triggers a second, redundant reload.
        if (a.has_value())
            m_cachedSourceTimestamp = a;
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

            // A pipeline or shader table that failed to create takes the same Retry / exit path as a compile error
            for (const std::shared_ptr<PTPipelineVariant>& variant : updateQueue)
                if (variant->GetVersion() != m_version)
                {
                    firstError = StringFormat("ERROR creating the ray tracing pipeline or shader table for '%s'", variant->m_shortUniqueDebugID.c_str());
                    break;
                }
        }


        if (firstError!="")
        {
            // Unlock for Retry; otherwise the next pass's UpdateStart finds them still locked. Reset too, so Retry
            // rebuilds the whole batch rather than keeping pipelines built from source that may have changed since
            for (const std::shared_ptr<PTPipelineVariant>& variant : updateQueue)
            {
                variant->m_us_lockedBaker = nullptr;
                variant->ResetPipeline();
            }

            firstError += m_compilerConfig.LongPathErrorHint();

            // Nobody to click Retry: log and take the Cancel path
            if (HelpersIsNonInteractive())
            {
                log::error("%s", firstError.c_str());
                HelpersExitProcess(1);
            }

#if _WIN32
            extern HWND HelpersGetActiveWindow();
            int result = MessageBoxA(HelpersGetActiveWindow(), firstError.c_str(),
                "Shader compile error", MB_RETRYCANCEL | MB_ICONWARNING | MB_SETFOREGROUND | MB_TASKMODAL);
            if (result == IDCANCEL)
#endif
                HelpersExitProcess(1); // the renderer can't work with broken shaders

            // Retry: rescan, as the source may have been fixed while the error box was up
            if (std::optional<std::filesystem::file_time_type> rescan = DirectoryModifiedMonitor::ScanNow(m_sourceMonitor); rescan.has_value())
                m_lastUpdatedSourceTimestamp = m_cachedSourceTimestamp = rescan;
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

std::shared_ptr<PTPipelineVariant> PTPipelineBaker::CreateVariant(const std::string & relativeSourcePath, std::vector<donut::engine::ShaderMacro> variantMacros, const std::string & shortUniqueDebugID, bool rayGenOnly, std::function<void(std::vector<donut::engine::ShaderMacro> & macros)> variantMacrosGetter)
{
    // Folded in here rather than at the hit-group step so they reach the raygen compile too - a
    // variant's raygen and its hit groups are one pipeline and have to be built against the same set.
    variantMacros.insert(variantMacros.end(), m_hitGroupMacros.begin(), m_hitGroupMacros.end());

    std::shared_ptr<PTPipelineVariant> variant = std::shared_ptr<PTPipelineVariant>(new PTPipelineVariant(relativeSourcePath, variantMacros, this->shared_from_this(), shortUniqueDebugID, rayGenOnly, std::move(variantMacrosGetter)));
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
