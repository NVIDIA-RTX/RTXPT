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

#include <donut/app/ApplicationBase.h>

#include "Materials/MaterialsBaker.h"
#include "ExtendedScene.h"
#include "SampleCommon.h"
#include "Shaders/PathTracer/PathTracerShared.h"
#include "Misc/picosha2.h"

#ifdef DONUT_WITH_TASKFLOW
#include <taskflow/taskflow.hpp>
#endif


using namespace donut;
using namespace donut::math;
using namespace donut::engine;

static const std::string c_PTShaderBinariesRoot = "ShaderDynamic/Bin";

static_assert( picosha2::k_digest_size == 32 );


PTPipelineVariant::PTPipelineVariant(const std::string & relativeSourcePath, const std::vector<donut::engine::ShaderMacro> & variantMacros, const std::shared_ptr<PTPipelineBaker>& baker)
    : m_macros(variantMacros), m_baker(baker), m_updateStarted(false)
{
    m_shaderFileName = relativeSourcePath;
    assert( m_shaderFileName.extension().string() == ".hlsl" );

    m_shaderFileOutName = m_shaderFileName;
    m_shaderFileOutName.replace_extension();
}

PTPipelineVariant::~PTPipelineVariant()
{
}

void PTPipelineVariant::ResetPipeline()
{
    m_shaderLibrary = nullptr;
    m_shaderTable = nullptr;
    m_pipeline = nullptr;
    m_localVersion = -1;
}

void PTPipelineVariant::CompileIfNeededPrepare(std::filesystem::file_time_type lastModifiedSourceCode)
{
    std::shared_ptr<PTPipelineBaker> & baker = m_us_lockedBaker; assert( baker != nullptr );

    std::string commandBase =  "\"" + baker->GetShaderCompilerPath().string() + "\"";

    auto srcFullPath = std::filesystem::absolute(baker->GetShadersPath() / m_shaderFileName);

    // source file
    commandBase += " \"" + srcFullPath.string() + "\"";

    // see https://simoncoenen.com/blog/programming/graphics/DxcCompiling for switch reference

    std::string command;

#if RTXPT_D3D_AGILITY_SDK_VERSION >= 717
        command += " -T lib_6_9";
        //command += " -Vd";
#else
        command += " -T lib_6_5";
#endif

    command += " -Zi";              //  Enable debug information. Cannot be used together with -Zs
    command += " -Qembed_debug";    //  Embed PDB in shader container (must be used with /Zi)
    command += " -Zsb";             //  Compute Shader Hash considering only output binary
    
    command += " -D ENABLE_DEBUG_PRINT"; // <- some issues with Linux? need to test this

    for (auto& macro : m_combinedMacros)
        command += " -D " + macro.name + "=" + macro.definition;
    
    command += " -I \"" + baker->GetShadersPathExternalIncludes1().string() + "\"";
    if (!baker->GetShadersPathExternalIncludes2().empty())
        command += " -I \"" + baker->GetShadersPathExternalIncludes2().string() + "\"";

    std::string targetMacro = " -D ";
    if( baker->GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12 )
    {
        targetMacro += "TARGET_D3D12";
    } 
    else if( baker->GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN )
    {
        targetMacro += "TARGET_VULKAN";
    } 
    else assert(false);

    command += targetMacro;

    command += " -O3";

    command += " -enable-16bit-types";

    if (baker->GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN)
    {
        command += " -D SPIRV";
        command += " -spirv";
        command += " -fspv-target-env=vulkan1.2";
        command += " -fspv-extension=SPV_EXT_descriptor_indexing";
        command += " -fspv-extension=KHR";

        nvrhi::VulkanBindingOffsets cBindingOffsets;
        for( int i = 0; i < 7; i++ )
        {
            // TODO: test with 'all' instead of the second %d - should work as well, no loop needed, see docs
            command += StringFormat(" -fvk-s-shift %d %d", cBindingOffsets.sampler, i);
            command += StringFormat(" -fvk-t-shift %d %d", cBindingOffsets.shaderResource, i);
            command += StringFormat(" -fvk-b-shift %d %d", cBindingOffsets.constantBuffer, i);
            command += StringFormat(" -fvk-u-shift %d %d", cBindingOffsets.unorderedAccess, i);
        }
    }

    std::string previousHashHex = m_compiledHashHex;

    // adding all command params (which includes all macros) but NOT the full file path (just file name) - this is to avoid recompiling if just moving folders around
    std::vector<unsigned char> hash(picosha2::k_digest_size);
    std::string stringToHash = command + m_shaderFileName.string(); // add file name because it's actually not included so far (it's in commandBase)
    picosha2::hash256(command.begin(), command.end(), hash.begin(), hash.end());
    m_compiledHashHex = picosha2::bytes_to_hex_string(hash.begin(), hash.end());

    if (previousHashHex != m_compiledHashHex)   // we'll have to rebuild PSOs, macros or something changed
        ResetPipeline();

    m_compiledFileNameNoExt = m_shaderFileOutName.string() + "_" + m_compiledHashHex;
    m_compiledFullPath = std::filesystem::absolute(baker->GetShaderBinariesPath() / m_compiledFileNameNoExt).string() + ".bin";

    auto lastModifiedTime = GetFileModifiedTime(m_compiledFullPath);
    if (lastModifiedTime.has_value() && (*lastModifiedTime) >= lastModifiedSourceCode)
    {
        if (baker->IsVerbose())
            donut::log::info("No need to compile shader variant of '%s', up-to-date file already exists...", srcFullPath.string().c_str());
        m_us_compileCmdLine = "";
        return;
    }

    command = commandBase + command + " -Fo \"" + m_compiledFullPath + "\"";

    if (baker->IsVerbose())
        donut::log::info("Compiling shader variant of '%s'...", srcFullPath.string().c_str());
    m_us_compileCmdLine = command;
    ResetPipeline();
}

void PTPipelineVariant::CompileIfNeededExecute()
{
    if (m_us_compileCmdLine == "")
        return;

    auto [resNum, resString, resErrorString] = SystemShell(m_us_compileCmdLine, false);
    // TODO: check for file existence too 
    if (resErrorString != "")
    {
        m_us_compileError = StringFormat( "ERROR compiling shader, command \n   %s\n result: \n   %s", m_us_compileCmdLine.c_str(), resErrorString.c_str() );
        donut::log::warning(m_us_compileError.c_str());
    }        

    m_us_compileCmdLine = "";
    return;
}

void PTPipelineVariant::UpdateStart(std::filesystem::file_time_type lastModifiedSourceCode)
{
    bool alreadyStarted = m_updateStarted.exchange(true);
    assert( !alreadyStarted );
    assert( m_us_compileCmdLine == "" );
    assert( m_us_lockedBaker == nullptr );
    m_us_compileError = "";

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
            m_exportAnyHit = (macro.definition == "1") ? (false) : (true);          // we only need anyHit if using fallback path
            foundExportAnyHitDependency = true;
        }
    }
    assert(foundExportAnyHitDependency); // any changes in the way USE_NVAPI_HIT_OBJECT_EXTENSION is used?

    CompileIfNeededPrepare(lastModifiedSourceCode);
}

void PTPipelineVariant::UpdateFinalize()
{
    bool alreadyStarted = m_updateStarted.exchange(true);
    assert(alreadyStarted);
    std::shared_ptr<PTPipelineBaker> & baker = m_us_lockedBaker; assert( baker != nullptr );

    m_us_compileError = "";
    CompileIfNeededExecute();

    if (m_shaderLibrary == nullptr && m_us_compileError == "" )
    {

        {
            std::lock_guard<std::mutex> lock(baker->GetMutex());    // TODO: revisit, this is very likely no longer needed due to simplified code below
            std::shared_ptr<donut::vfs::IBlob> data = baker->GetFS()->readFile(("/"+c_PTShaderBinariesRoot + "/" + m_compiledFileNameNoExt + ".bin").c_str());
            m_shaderLibrary = baker->GetDevice()->createShaderLibrary(data->data(), data->size());
        }

        if (!m_shaderLibrary)
            { assert( false ); return; }

        nvrhi::rt::PipelineDesc pipelineDesc;
        pipelineDesc.globalBindingLayouts = { baker->GetBindingLayout(), baker->GetBindlessLayout() };
        pipelineDesc.shaders.push_back({ "", m_shaderLibrary->getShader("RayGen", nvrhi::ShaderType::RayGeneration), nullptr });
        pipelineDesc.shaders.push_back({ "", m_shaderLibrary->getShader("Miss", nvrhi::ShaderType::Miss), nullptr });
        pipelineDesc.allowOpacityMicromaps = true;

        for (auto& [_, hitGroupInfo] : baker->GetUniqueHitGroups())
        {
            pipelineDesc.hitGroups.push_back(
                {
                    .exportName = hitGroupInfo.ExportName,
                    .closestHitShader = m_shaderLibrary->getShader(hitGroupInfo.ClosestHitShader.c_str(), nvrhi::ShaderType::ClosestHit),
                    .anyHitShader = (m_exportAnyHit && hitGroupInfo.AnyHitShader != "") ? (m_shaderLibrary->getShader(hitGroupInfo.AnyHitShader.c_str(), nvrhi::ShaderType::AnyHit)) : (nullptr),
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

        m_shaderTable->setRayGenerationShader("RayGen");
        auto & perSubInstanceHitGroup = baker->GetPerSubInstanceHitGroup();
        for (int i = 0; i < perSubInstanceHitGroup.size(); i++)
            m_shaderTable->addHitGroup(perSubInstanceHitGroup[i].ExportName.c_str());

        m_shaderTable->addMissShader("Miss");
        m_localVersion = baker->GetVersion();
    }
   
    assert( (m_shaderLibrary != nullptr && m_pipeline != nullptr && m_shaderTable != nullptr) || m_us_compileError != "" );

    m_us_lockedBaker = nullptr;
    m_us_compileCmdLine = "";

    alreadyStarted = m_updateStarted.exchange(false);
    assert(alreadyStarted);
}


PTPipelineBaker::PTPipelineBaker(nvrhi::IDevice* device, std::shared_ptr<MaterialsBaker>& materialsBaker, nvrhi::BindingLayoutHandle bindingLayout, nvrhi::BindingLayoutHandle bindlessLayout)
    : m_device(device)
    , m_materialsBaker(materialsBaker)
    , m_bindingLayout(bindingLayout)
    , m_bindlessLayout(bindlessLayout)
    , m_enableNVAPIShaderExtension(device->queryFeatureSupport(nvrhi::Feature::HlslExtensionUAV))
{
    std::string graphicsAPIName;
    if (m_device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
        graphicsAPIName = "d3d12";
    else if (m_device->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN)
        graphicsAPIName = "vk";
    std::string platformName = "x64"; // add "arm64" for ARM

    m_shaderCompilerPath = std::filesystem::absolute(donut::app::GetDirectoryWithExecutable() / "ShaderDynamic/Tools" / graphicsAPIName / platformName / "dxc.exe");

    if (!std::filesystem::exists(m_shaderCompilerPath))
        donut::log::fatal("Unable to find '%s' and it is necessary for runtime shader compilation", m_shaderCompilerPath.string().c_str());

    std::filesystem::path   shaderSourcePathDevelopment = donut::app::GetDirectoryWithExecutable() / "../Rtxpt/Shaders";
    std::filesystem::path   shaderSourcePathRuntime = donut::app::GetDirectoryWithExecutable() / "ShaderDynamic/Source/Rtxpt";

    if (!std::filesystem::exists(shaderSourcePathDevelopment))
    {
        donut::log::info("Shaders development folder '%s' not found, trying local '%s'...", shaderSourcePathDevelopment.string().c_str(), shaderSourcePathRuntime.string().c_str());
        if (!std::filesystem::exists(shaderSourcePathRuntime))
            donut::log::fatal("Unable to find shaders folder %s and it is necessary for the app to run", shaderSourcePathRuntime.string().c_str());
        else
        {
            m_shadersPath = shaderSourcePathRuntime;
            m_shadersPathExternalIncludes1 = donut::app::GetDirectoryWithExecutable() / "ShaderDynamic/Source/External";
            m_shadersPathExternalIncludes2 = std::filesystem::path();
        }
    }
    else
    {
        m_shadersPath = shaderSourcePathDevelopment;
        m_shadersPathExternalIncludes1 = donut::app::GetDirectoryWithExecutable() / "../External/Donut/Include";
        m_shadersPathExternalIncludes2 = donut::app::GetDirectoryWithExecutable() / "../External";
    }

    m_shaderBinariesPath = app::GetDirectoryWithExecutable() / c_PTShaderBinariesRoot / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
    EnsureDirectoryExists(m_shaderBinariesPath);

    m_shaderCompilerPath = std::filesystem::absolute(m_shaderCompilerPath);
    m_shadersPath = std::filesystem::absolute(m_shadersPath);
    m_shadersPathExternalIncludes1 = std::filesystem::absolute(m_shadersPathExternalIncludes1);
    if (!m_shadersPathExternalIncludes2.empty())
        m_shadersPathExternalIncludes2 = std::filesystem::absolute(m_shadersPathExternalIncludes2);
    m_shaderBinariesPath = std::filesystem::absolute(m_shaderBinariesPath);
    donut::log::info("Using shader compiler: '%s'", m_shaderCompilerPath.string().c_str());
    donut::log::info("Using shaders for dynamic compilation: '%s' (additional includes: '%s', '%s')", m_shadersPath.string().c_str(), m_shadersPathExternalIncludes1.string().c_str(), m_shadersPathExternalIncludes2.string().c_str());
    donut::log::info("Dynamic shader compilation binaries output: '%s'", m_shaderBinariesPath.string().c_str());

    m_shadersFS = std::make_shared<vfs::RootFileSystem>();
    m_shadersFS->mount("/" + c_PTShaderBinariesRoot, m_shaderBinariesPath);
}

PTPipelineBaker::~PTPipelineBaker()
{
}

// see OptimizationHints
HitGroupInfo ComputeSubInstanceHitGroupInfo(const PTMaterial& material)
{
    MaterialShadingProperties matProps = MaterialShadingProperties::Compute(material);

    HitGroupInfo info;

    info.ClosestHitShader = "ClosestHit";
    info.ClosestHitShader += std::to_string(matProps.NoTextures);
    info.ClosestHitShader += std::to_string(matProps.NoTransmission);
    info.ClosestHitShader += std::to_string(matProps.OnlyDeltaLobes);

    info.AnyHitShader = matProps.AlphaTest ? "AnyHit" : "";

    info.ExportName = "HitGroup";
    if (matProps.NoTextures)
        info.ExportName += "_NoTextures";
    if (matProps.NoTransmission)
        info.ExportName += "_NoTransmission";
    if (matProps.OnlyDeltaLobes)
        info.ExportName += "_OnlyDeltaLobes";
    if (matProps.AlphaTest)
        info.ExportName += "_HasAlphaTest";

    return info;
}

static bool macrosEqual(donut::engine::ShaderMacro& a, donut::engine::ShaderMacro& b)
{
    return a.name == b.name && a.definition == b.definition;
}

void PTPipelineBaker::Update(const std::shared_ptr<class ExtendedScene>& scene, unsigned int subInstanceCount, const std::function<void(std::vector<donut::engine::ShaderMacro>& macros)>& globalMacrosGetter, bool forceShaderReload)
{
    bool needsUpdate = m_uniqueHitGroups.size() == 0;

    std::vector<donut::engine::ShaderMacro> newMacros;
    globalMacrosGetter(newMacros);
    if (!std::equal(newMacros.begin(), newMacros.end(), m_macros.begin(), m_macros.end(), macrosEqual))
    {
        needsUpdate = true;
        m_macros = newMacros;
    }

    needsUpdate |= forceShaderReload;

    ProgressBar progressCompilingShaders;

    if (needsUpdate)
        progressCompilingShaders.Start("Compiling shaders");

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
                m_perSubInstanceHitGroup.push_back(ComputeSubInstanceHitGroupInfo(*PTMaterial::FromDonut(instance->GetMesh()->geometries[gi]->material)));
        }
        assert(m_perSubInstanceHitGroup.size() == subInstanceCount);

        // Prime the instances to make sure we only include the necessary CHS variants in the PSO. Many (sub)instances can map to same materials.
        m_uniqueHitGroups.clear();
        for (int i = 0; i < m_perSubInstanceHitGroup.size(); i++)
            m_uniqueHitGroups[m_perSubInstanceHitGroup[i].ExportName] = m_perSubInstanceHitGroup[i];
        needsUpdate = true;
    }

    if (needsUpdate)
    {
        m_version++;
    }

    if (needsUpdate)
    {
        std::optional<std::filesystem::file_time_type> a = GetLatestModifiedTimeDirectoryRecursive(m_shadersPath);
        // let's not track externals for perf reasons but here's the code in case it's needed
        //std::optional<std::filesystem::file_time_type> b = GetLatestModifiedTimeRecursive(m_shadersPathExternalIncludes1);
        //std::optional<std::filesystem::file_time_type> c = GetLatestModifiedTimeRecursive(m_shadersPathExternalIncludes2);
        m_lastUpdatedSourceTimestamp = a;
    }

    if (!m_lastUpdatedSourceTimestamp.has_value())
    {
        log::error("There is something wrong with the shader source path or logic - unable to dynamically compile shaders");
        return;
    }

    do
    {
        progressCompilingShaders.Set(10);
        std::atomic_int counterCompleted(0);
        std::vector<std::shared_ptr<PTPipelineVariant>> updateQueue;
        for (int i = 0; i < int(m_variants.size()); i++)
        {
            const std::shared_ptr<PTPipelineVariant>& variant = m_variants[i];
            if (variant.use_count() == 1)
                assert(false); // dangling Variant - forgotten a call to ReleaseVariant?
            if (variant->GetVersion() != m_version)
            {
                updateQueue.push_back(variant);
                if (!progressCompilingShaders.Active())
                    progressCompilingShaders.Start("Updating shaders");
                variant->ResetPipeline();
                variant->UpdateStart(*m_lastUpdatedSourceTimestamp);
            }
        }

        if (!updateQueue.empty())
        {
    #ifdef DONUT_WITH_TASKFLOW
    #define BAKER_ENABLE_MULTITHREADED_COMPILE 1
    #endif

            int updateQueueSize = (int)updateQueue.size();

    #if BAKER_ENABLE_MULTITHREADED_COMPILE
            tf::Executor executor;
    #endif
            for (const std::shared_ptr<PTPipelineVariant>& variant : updateQueue)
            {
    #if BAKER_ENABLE_MULTITHREADED_COMPILE
                executor.async([this, &variant, &counterCompleted, &progressCompilingShaders, updateQueueSize](){
    #endif
                variant->UpdateFinalize();
                int completed = counterCompleted.fetch_add(1)+1;
                progressCompilingShaders.Set(10 + 90 * completed / updateQueueSize);
    #if BAKER_ENABLE_MULTITHREADED_COMPILE
                });
    #endif
            }
    #if BAKER_ENABLE_MULTITHREADED_COMPILE
            executor.wait_for_all();
    #endif

            progressCompilingShaders.Set(100);
        }

        std::string firstError = "";
        for (const std::shared_ptr<PTPipelineVariant>& variant : updateQueue)
            if (variant->LastCompileError() != "")
            {
                firstError = variant->LastCompileError();
                break;
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

std::shared_ptr<PTPipelineVariant> PTPipelineBaker::CreateVariant(const std::string & relativeSourcePath, std::vector<donut::engine::ShaderMacro> variantMacros)
{
    std::shared_ptr<PTPipelineVariant> variant = std::shared_ptr<PTPipelineVariant>(new PTPipelineVariant(relativeSourcePath, variantMacros, this->shared_from_this()));
    m_variants.push_back(variant);
    return variant;
}