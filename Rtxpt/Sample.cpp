/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Sample.h"

#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/BindingCache.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/json.h>
#include <donut/core/math/math.h>
#include <donut/shaders/light_cb.h>
#include <nvrhi/utils.h>
#include <nvrhi/common/misc.h>
#include <cmath>

#include "PathTracer/StablePlanes.hlsli"
#include "AccelerationStructureUtil.h"

#include "Lighting/Distant/EnvMapImportanceSamplingBaker.h"
#include "Materials/MaterialsBaker.h"

#include "OpacityMicroMap/OmmBaker.h"

#include "LocalConfig.h"
#include "CommandLine.h"
#include "Korgi.h"

#include "GPUSort/GPUSort.h"

using namespace donut;
using namespace donut::math;
using namespace donut::app;
using namespace donut::vfs;
using namespace donut::engine;
using namespace donut::render;

#include <fstream>
#include <iostream>

#include <thread>

#ifdef _WIN32
// Use discrete GPU by default on laptops.
extern "C"
{
    // http://developer.download.nvidia.com/devzone/devcenter/gamegraphics/files/OptimusRenderingPolicies.pdf
    __declspec(dllexport) DWORD NvOptimusEnablement = 1;

    // https://gpuopen.com/learn/amdpowerxpressrequesthighperformance/
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

static const int c_swapchainCount = 3;

static const char* g_windowTitle = "RTX Path Tracing v1.5.1";

const char* g_assetsFolder = "Assets";

const float c_envMapRadianceScale = 1.0f / 4.0f; // used to make input 32bit float radiance fit into 16bit float range that baker supports; going lower than 1/4 causes issues with current BC6U compression algorithm when used

// Temp helper used to reduce FPS to specified target (i.e.) 30 - useful to avoid overheating the office :) but not intended for precise fps control
class FPSLimiter
{
private:
    std::chrono::high_resolution_clock::time_point   m_lastTimestamp = std::chrono::high_resolution_clock::now();
    double                                  m_prevError     = 0.0;

public:
    void                FramerateLimit( int fpsTarget )
    {
        std::chrono::high_resolution_clock::time_point   nowTimestamp = std::chrono::high_resolution_clock::now();
        double deltaTime = std::chrono::duration<double>(nowTimestamp - m_lastTimestamp).count();
        double targetDeltaTime = 1.0 / (double)fpsTarget;
        double diffFromTarget = targetDeltaTime - deltaTime + m_prevError;
        if (diffFromTarget > 0.0f)
        {
            size_t sleepInMs = std::min(1000, (int)(diffFromTarget * 1000));
            std::this_thread::sleep_for(std::chrono::milliseconds(sleepInMs));
        }

        auto prevTime = m_lastTimestamp;
        m_lastTimestamp = std::chrono::high_resolution_clock::now();
        double deltaError = targetDeltaTime - std::chrono::duration<double>( m_lastTimestamp - prevTime ).count();
        m_prevError = deltaError * 0.9 + m_prevError * 0.1;     // dampen the spring-like effect, but still remain accurate to any positive/negative creep induced by our sleep mechanism
        // clamp error handling to 1 frame length
        if( m_prevError > targetDeltaTime )
            m_prevError = targetDeltaTime;
        if( m_prevError < -targetDeltaTime )
            m_prevError = -targetDeltaTime;
        // shift last time by error to compensate
        m_lastTimestamp += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(m_prevError));
    }
};

template<typename ... Args>
std::string string_format(const std::string& format, Args ... args)
{
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
    if (size_s <= 0) { throw std::runtime_error("Error during formatting."); }
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

static FPSLimiter g_FPSLimiter;

std::filesystem::path GetLocalPath(std::string subfolder)
{
    static std::filesystem::path oneChoice;
    // if( oneChoice.empty() )
    {
        std::filesystem::path candidateA = app::GetDirectoryWithExecutable( ) / subfolder;
        std::filesystem::path candidateB = app::GetDirectoryWithExecutable( ).parent_path( ) / subfolder;
        if (std::filesystem::exists(candidateA))
            oneChoice = candidateA;
        else
            oneChoice = candidateB;
    }
    return oneChoice;
}

static donut::log::Callback g_DonutDefaultCallback = nullptr;

static void SampleLogCallback(donut::log::Severity severity, const char* message)
{
    // This lets us demote some of Streamline errors that aren't errors into warnings
    if (severity == donut::log::Severity::Error)
    {
        std::string msg(message);
        if (msg.find("Don't know the size") != std::string::npos)
            severity = donut::log::Severity::Warning;
        if (msg.find("dlss_gEntry.cpp") != std::string::npos)
        {
            if (    msg.find("Unable to find DRS context") != std::string::npos 
                ||  msg.find("NGX indicates DLSS-G is not available") != std::string::npos)
                severity = donut::log::Severity::Warning;
        }
        if( msg.find( "Missing NGX context" ) != std::string::npos
            || msg.find( "Unable to find NGX " ) != std::string::npos 
            || msg.find( "NvAPI_D3D_Sleep" ) != std::string::npos )
            severity = donut::log::Severity::Warning;
    }

    g_DonutDefaultCallback(severity, message);
}

Sample::Sample( donut::app::DeviceManager * deviceManager, CommandLineOptions& cmdLine, SampleUIData & ui )
    : app::ApplicationBase( deviceManager ), m_cmdLine(cmdLine), m_ui( ui )
{
    deviceManager->SetFrameTimeUpdateInterval(1.0f);

    std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable( ) / "shaders/framework" / app::GetShaderTypeName( GetDevice( )->getGraphicsAPI( ) );
    std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/RTXPT" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
    std::filesystem::path nrdShaderPath = app::GetDirectoryWithExecutable() / "shaders/nrd" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
    std::filesystem::path ommShaderPath = app::GetDirectoryWithExecutable( ) / "shaders/omm" / app::GetShaderTypeName( GetDevice( )->getGraphicsAPI( ) );

    m_RootFS = std::make_shared<vfs::RootFileSystem>( );
    m_RootFS->mount( "/shaders/donut", frameworkShaderPath );
    m_RootFS->mount( "/shaders/app", appShaderPath);
    m_RootFS->mount("/shaders/nrd", nrdShaderPath);
    m_RootFS->mount( "/shaders/omm", ommShaderPath);

    m_shaderFactory = std::make_shared<engine::ShaderFactory>( GetDevice( ), m_RootFS, "/shaders" );
    m_CommonPasses = std::make_shared<engine::CommonRenderPasses>( GetDevice( ), m_shaderFactory );
    m_bindingCache = std::make_unique<engine::BindingCache>( GetDevice( ) );

    m_camera.SetRotateSpeed(.003f);

#if DONUT_WITH_STREAMLINE
    m_ui.IsDLSSSuported = GetDeviceManager()->GetStreamline().IsDLSSAvailable();
    m_ui.IsDLSSGSupported = GetDeviceManager()->GetStreamline().IsDLSSGAvailable();
    m_ui.IsReflexSupported = GetDeviceManager()->GetStreamline().IsReflexAvailable();
    m_ui.IsDLSSRRSupported = GetDeviceManager()->GetStreamline().IsDLSSRRAvailable();
#endif

    korgi::Init();
}

Sample::~Sample()
{
    korgi::Shutdown();
}

void Sample::DebugDrawLine( float3 start, float3 stop, float4 col1, float4 col2 )
{
    if( int(m_cpuSideDebugLines.size())+2 >= MAX_DEBUG_LINES )
        return;
    DebugLineStruct dls = { float4(start, 1), col1 }, dle = { float4(stop, 1), col2 };
    m_cpuSideDebugLines.push_back(dls);
    m_cpuSideDebugLines.push_back(dle);
}

bool Sample::Init(const std::string& preferredScene)
{
    nvrhi::BindlessLayoutDesc bindlessLayoutDesc;
    bindlessLayoutDesc.visibility = nvrhi::ShaderType::All;
    bindlessLayoutDesc.firstSlot = 0;
    bindlessLayoutDesc.maxCapacity = 1024;
    bindlessLayoutDesc.registerSpaces = {
        nvrhi::BindingLayoutItem::RawBuffer_SRV(1),
        nvrhi::BindingLayoutItem::Texture_SRV(2)
    };
    m_bindlessLayout = GetDevice()->createBindlessLayout(bindlessLayoutDesc);

    nvrhi::BindingLayoutDesc globalBindingLayoutDesc;
    globalBindingLayoutDesc.visibility = nvrhi::ShaderType::All;
    globalBindingLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
        nvrhi::BindingLayoutItem::PushConstants(1, sizeof(SampleMiniConstants)),
        nvrhi::BindingLayoutItem::RayTracingAccelStruct(0),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(3),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(4),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(5),
        nvrhi::BindingLayoutItem::Texture_SRV(10),              // t_EnvironmentMap
        nvrhi::BindingLayoutItem::Texture_SRV(11),              // t_EnvironmentMapImportanceMap        <- TODO: remove this, no longer used
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(12),     // t_LightsCB
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(13),     // t_Lights
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(14),     // t_LightsEx
        nvrhi::BindingLayoutItem::TypedBuffer_SRV(15),          // t_LightProxyCounters
        nvrhi::BindingLayoutItem::TypedBuffer_SRV(16),          // t_LightProxyIndices
        nvrhi::BindingLayoutItem::Texture_SRV(17),              // t_LightNarrowSamplingBuffer
        nvrhi::BindingLayoutItem::Texture_SRV(18),              // t_EnvLookupMap
        nvrhi::BindingLayoutItem::Texture_UAV(10),              // u_LightFeedbackBuffer
#if USE_PRECOMPUTED_SOBOL_BUFFER
        nvrhi::BindingLayoutItem::TypedBuffer_SRV(42),
#endif
        nvrhi::BindingLayoutItem::Sampler(0),
        nvrhi::BindingLayoutItem::Sampler(1),
        nvrhi::BindingLayoutItem::Sampler(2),
        nvrhi::BindingLayoutItem::Texture_UAV(0),
        nvrhi::BindingLayoutItem::Texture_UAV(4),           // u_Throughput
        nvrhi::BindingLayoutItem::Texture_UAV(5),           // u_MotionVectors
        nvrhi::BindingLayoutItem::Texture_UAV(6),           // u_Depth
        // denoising slots go from 30-39
        //nvrhi::BindingLayoutItem::StructuredBuffer_UAV(30), // denoiser 'control buffer' (might be removed, might be reused)
        nvrhi::BindingLayoutItem::Texture_UAV(31),          // RWTexture2D<float>  u_DenoiserViewspaceZ
        nvrhi::BindingLayoutItem::Texture_UAV(32),          // RWTexture2D<float4> u_DenoiserMotionVectors
        nvrhi::BindingLayoutItem::Texture_UAV(33),          // RWTexture2D<float4> u_DenoiserNormalRoughness
        nvrhi::BindingLayoutItem::Texture_UAV(34),          // RWTexture2D<float4> u_DenoiserDiffRadianceHitDist
        nvrhi::BindingLayoutItem::Texture_UAV(35),          // RWTexture2D<float4> u_DenoiserSpecRadianceHitDist
        nvrhi::BindingLayoutItem::Texture_UAV(36),          // RWTexture2D<float4> u_DenoiserDisocclusionThresholdMix
        nvrhi::BindingLayoutItem::Texture_UAV(37),          // RWTexture2D<float4> u_CombinedHistoryClampRelax
        // debugging slots go from 50-59
        nvrhi::BindingLayoutItem::Texture_UAV(50),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(51),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(52),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(53),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(54),
        // ReSTIR GI
        nvrhi::BindingLayoutItem::Texture_UAV(60), // u_SecondarySurfacePositionNormal
        nvrhi::BindingLayoutItem::Texture_UAV(61),  // u_SecondarySurfaceRadiance

        nvrhi::BindingLayoutItem::Texture_UAV(70),          // u_RRDiffuseAlbedo
        nvrhi::BindingLayoutItem::Texture_UAV(71),          // u_RRSpecAlbedo   
        nvrhi::BindingLayoutItem::Texture_UAV(72),          // u_RRNormalsAndRoughness
        nvrhi::BindingLayoutItem::Texture_UAV(73),          // u_RRSpecMotionVectors

        nvrhi::BindingLayoutItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX)

#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
        // Stochastic texture filtering blue noise texture
        , nvrhi::BindingLayoutItem::Texture_SRV(63),              // t_STBN2DTexture
#endif
    };

    // NV HLSL extensions - DX12 only
    if (GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
    {
        globalBindingLayoutDesc.bindings.push_back(
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(NV_SHADER_EXTN_SLOT_NUM));
    }

    // stable planes buffers -- must be last because these items are appended to the BindingSetDesc after the main list
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_UAV(40));
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(42));
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_UAV(44));
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(45));

    m_bindingLayout = GetDevice()->createBindingLayout(globalBindingLayoutDesc);

    m_DescriptorTable = std::make_shared<engine::DescriptorTableManager>(GetDevice(), m_bindlessLayout);

    auto nativeFS = std::make_shared<vfs::NativeFileSystem>();
    m_TextureCache = std::make_shared<engine::TextureCache>(GetDevice(), nativeFS, m_DescriptorTable);

    memset( &m_feedbackData, 0, sizeof(DebugFeedbackStruct) * 1 );
    memset( &m_debugDeltaPathTree, 0, sizeof(DeltaTreeVizPathVertex) * cDeltaTreeVizMaxVertices );

    //Draw lines from the feedback buffer
    {
        std::vector<ShaderMacro> drawLinesMacro = { ShaderMacro("DRAW_LINES_SHADERS", "1") };
        m_linesVertexShader = m_shaderFactory->CreateShader("app/DebugLines.hlsl", "main_vs", &drawLinesMacro, nvrhi::ShaderType::Vertex);
        m_linesPixelShader = m_shaderFactory->CreateShader("app/DebugLines.hlsl", "main_ps", &drawLinesMacro, nvrhi::ShaderType::Pixel);

        nvrhi::VertexAttributeDesc attributes[] = {
            nvrhi::VertexAttributeDesc()
                .setName("POSITION")
                .setFormat(nvrhi::Format::RGBA32_FLOAT)
                .setOffset(0)
                .setElementStride(sizeof(DebugLineStruct)),
                nvrhi::VertexAttributeDesc()
                .setName("COLOR")
                .setFormat(nvrhi::Format::RGBA32_FLOAT)
                .setOffset(offsetof(DebugLineStruct, col))
                .setElementStride(sizeof(DebugLineStruct)),
        };
        m_linesInputLayout = GetDevice()->createInputLayout(attributes, uint32_t(std::size(attributes)), m_linesVertexShader);

        nvrhi::BindingLayoutDesc linesBindingLayoutDesc;
        linesBindingLayoutDesc.visibility = nvrhi::ShaderType::All;
        linesBindingLayoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::Texture_SRV(0)
        };

        m_linesBindingLayout = GetDevice()->createBindingLayout(linesBindingLayoutDesc);

        // debug stuff!
        {
            nvrhi::BufferDesc bufferDesc;
            bufferDesc.byteSize = sizeof(DebugFeedbackStruct) * 1;
            bufferDesc.isConstantBuffer = false;
            bufferDesc.isVolatile = false;
            bufferDesc.canHaveUAVs = true;
            bufferDesc.cpuAccess = nvrhi::CpuAccessMode::None;
            bufferDesc.maxVersions = engine::c_MaxRenderPassConstantBufferVersions;
            bufferDesc.structStride = sizeof(DebugFeedbackStruct);
            bufferDesc.keepInitialState = true;
            bufferDesc.initialState = nvrhi::ResourceStates::Common;
            bufferDesc.debugName = "Feedback_Buffer_Gpu";
            m_feedback_Buffer_Gpu = GetDevice()->createBuffer(bufferDesc);

            bufferDesc.canHaveUAVs = false;
            bufferDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
            bufferDesc.structStride = 0;
            bufferDesc.keepInitialState = false;
            bufferDesc.initialState = nvrhi::ResourceStates::Unknown;
            bufferDesc.debugName = "Feedback_Buffer_Cpu";
            m_feedback_Buffer_Cpu = GetDevice()->createBuffer(bufferDesc);

            bufferDesc.byteSize = sizeof(DebugLineStruct) * MAX_DEBUG_LINES;
            bufferDesc.isVertexBuffer = true;
            bufferDesc.isConstantBuffer = false;
            bufferDesc.isVolatile = false;
            bufferDesc.canHaveUAVs = true;
            bufferDesc.cpuAccess = nvrhi::CpuAccessMode::None;
            bufferDesc.structStride = sizeof(DebugLineStruct);
            bufferDesc.keepInitialState = true;
            bufferDesc.initialState = nvrhi::ResourceStates::Common;
            bufferDesc.debugName = "DebugLinesCapture";
            m_debugLineBufferCapture    = GetDevice()->createBuffer(bufferDesc);
            bufferDesc.debugName = "DebugLinesDisplay";
            m_debugLineBufferDisplay    = GetDevice()->createBuffer(bufferDesc);

            bufferDesc.byteSize = sizeof(DeltaTreeVizPathVertex) * cDeltaTreeVizMaxVertices;
            bufferDesc.isConstantBuffer = false;
            bufferDesc.isVolatile = false;
            bufferDesc.canHaveUAVs = true;
            bufferDesc.cpuAccess = nvrhi::CpuAccessMode::None;
            bufferDesc.maxVersions = engine::c_MaxRenderPassConstantBufferVersions;
            bufferDesc.structStride = sizeof(DeltaTreeVizPathVertex);
            bufferDesc.keepInitialState = true;
            bufferDesc.initialState = nvrhi::ResourceStates::Common;
            bufferDesc.debugName = "Feedback_PathDecomp_Gpu";
            m_debugDeltaPathTree_Gpu = GetDevice()->createBuffer(bufferDesc);

            bufferDesc.canHaveUAVs = false;
            bufferDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
            bufferDesc.structStride = 0;
            bufferDesc.keepInitialState = false;
            bufferDesc.initialState = nvrhi::ResourceStates::Unknown;
            bufferDesc.debugName = "Feedback_PathDecomp_Cpu";
            m_debugDeltaPathTree_Cpu = GetDevice()->createBuffer(bufferDesc);


            bufferDesc.byteSize = sizeof(PathPayload) * cDeltaTreeVizMaxStackSize;
            bufferDesc.isConstantBuffer = false;
            bufferDesc.isVolatile = false;
            bufferDesc.canHaveUAVs = true;
            bufferDesc.cpuAccess = nvrhi::CpuAccessMode::None;
            bufferDesc.maxVersions = engine::c_MaxRenderPassConstantBufferVersions;
            bufferDesc.structStride = sizeof(PathPayload);
            bufferDesc.keepInitialState = true;
            bufferDesc.initialState = nvrhi::ResourceStates::Common;
            bufferDesc.debugName = "DebugDeltaPathTreeSearchStack";
            m_debugDeltaPathTreeSearchStack = GetDevice()->createBuffer(bufferDesc);
        }
    }

    // Main constant buffer
    m_constantBuffer = GetDevice()->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(SampleConstants), "SampleConstants", engine::c_MaxRenderPassConstantBufferVersions*2));	// *2 because in some cases we update twice per frame

    // Command list!
    m_commandList = GetDevice()->createCommandList();

    m_ommBaker = std::make_shared<OmmBaker>(GetDevice(), m_DescriptorTable, m_TextureCache, m_shaderFactory, m_commandList, GetDevice()->queryFeatureSupport(nvrhi::Feature::RayTracingOpacityMicromap));

    // Setup precomputed Sobol' buffer.
#if USE_PRECOMPUTED_SOBOL_BUFFER
    {
        const uint precomputedSobolDimensions = SOBOL_MAX_DIMENSIONS; const uint precomputedSobolIndexCount = SOBOL_PRECOMPUTED_INDEX_COUNT;
        const uint precomputedSobolBufferCount = precomputedSobolIndexCount * precomputedSobolDimensions;

        // buffer that stores pre-generated samples which get updated once per frame
        nvrhi::BufferDesc buffDesc;
        buffDesc.byteSize = sizeof(uint) * precomputedSobolBufferCount;
        buffDesc.format = nvrhi::Format::R32_UINT;
        buffDesc.canHaveTypedViews = true;
        buffDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        buffDesc.keepInitialState = true;
        buffDesc.debugName = "PresampledEnvironmentSamples";
        buffDesc.canHaveUAVs = false;
        m_precomputedSobolBuffer = GetDevice()->createBuffer(buffDesc);

        uint * dataBuffer = new uint[precomputedSobolBufferCount];
        PrecomputeSobol(dataBuffer);

        nvrhi::DeviceHandle device = GetDevice();
        m_commandList->open();
        m_commandList->writeBuffer(m_precomputedSobolBuffer, dataBuffer, precomputedSobolBufferCount * sizeof(uint));
        m_commandList->close();
        GetDevice()->executeCommandList(m_commandList);
        GetDevice()->waitForIdle();

        delete[] dataBuffer;
    }
#endif

#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
    // Get blue noise texture to be used with stochastic texture filtering
    {
        const std::filesystem::path stbnTexturePath = GetLocalPath(g_assetsFolder) / "STBN/STBlueNoise_vec2_128x128x64.png";
        m_STBNTexture = m_TextureCache->LoadTextureFromFileDeferred(stbnTexturePath, false);
    }
#endif

    // Get all scenes in "assets" folder
    const std::string mediaExt = ".scene.json";
    for (const auto& file : std::filesystem::directory_iterator(GetLocalPath(g_assetsFolder)))
    {
        if (!file.is_regular_file()) continue;
        std::string fileName = file.path().filename().string();
        std::string longExt = (fileName.size()<=mediaExt.length())?(""):(fileName.substr(fileName.length()-mediaExt.length()));
        if ( longExt == mediaExt )
            m_sceneFilesAvailable.push_back( file.path().filename().string() );
    }

    std::string scene = FindPreferredScene(m_sceneFilesAvailable, preferredScene);

    // Select initial scene
    SetCurrentScene(scene);

    return true;
}

void Sample::SetCurrentScene( const std::string & sceneName, bool forceReload )
{
    if( m_currentSceneName == sceneName && !forceReload )
        return;
    m_currentSceneName = sceneName;
    m_ui.ResetAccumulation = true;
    SetAsynchronousLoadingEnabled( false );
    std::filesystem::path scenePath = GetLocalPath(g_assetsFolder) / sceneName;
    BeginLoadingScene( std::make_shared<vfs::NativeFileSystem>(), scenePath );
    if( m_scene == nullptr )
    {
        log::error( "Unable to load scene '%s'", sceneName.c_str() );
        return;
    }
    m_currentScenePath = scenePath;
}

void Sample::SceneUnloading( )
{
    m_ui.TogglableNodes = nullptr;
    ApplicationBase::SceneUnloading();
    m_bindingSet = nullptr;
    m_topLevelAS = nullptr;
    m_subInstanceBuffer = nullptr;
    m_bindingCache->Clear( );
    m_lights.clear();
    m_ui.SelectedMaterial = nullptr;
    m_ui.EnvironmentMapParams = EnvironmentMapRuntimeParameters();
    m_envMapBaker = nullptr;
    m_lightsBaker = nullptr;
    m_materialsBaker = nullptr;
    m_gpuSort = nullptr;
    m_uncompressedTextures.clear();
	m_rtxdiPass->Reset();
    m_ommBaker->SceneUnloading();
}

bool Sample::LoadScene(std::shared_ptr<vfs::IFileSystem> fs, const std::filesystem::path& sceneFileName)
{
    m_scene = std::shared_ptr<engine::ExtendedScene>( new engine::ExtendedScene(GetDevice(), *m_shaderFactory, fs, m_TextureCache, m_DescriptorTable, std::make_shared<ExtendedSceneTypeFactory>() ) );
    if (m_scene->Load(sceneFileName))
        return true;
    m_scene = nullptr;
    return false;
}

void Sample::UpdateCameraFromScene( const std::shared_ptr<donut::engine::PerspectiveCamera> & sceneCamera )
{
    dm::affine3 viewToWorld = sceneCamera->GetViewToWorldMatrix();
    dm::float3 cameraPos = viewToWorld.m_translation;
    m_camera.LookAt(cameraPos, cameraPos + viewToWorld.m_linear.row2, viewToWorld.m_linear.row1);
    m_cameraVerticalFOV = sceneCamera->verticalFov;
    m_cameraZNear = sceneCamera->zNear;

    std::shared_ptr<donut::engine::PerspectiveCameraEx> sceneCameraEx = std::dynamic_pointer_cast<PerspectiveCameraEx>(sceneCamera);
    if( sceneCameraEx != nullptr )
    {
        ToneMappingParameters defaults;

        m_ui.ToneMappingParams.autoExposure = sceneCameraEx->enableAutoExposure.value_or(defaults.autoExposure);
        m_ui.ToneMappingParams.exposureCompensation = sceneCameraEx->exposureCompensation.value_or(defaults.exposureCompensation);
        m_ui.ToneMappingParams.exposureValue = sceneCameraEx->exposureValue.value_or(defaults.exposureValue);
        m_ui.ToneMappingParams.exposureValueMin = sceneCameraEx->exposureValueMin.value_or(defaults.exposureValueMin);
        m_ui.ToneMappingParams.exposureValueMax = sceneCameraEx->exposureValueMax.value_or(defaults.exposureValueMax);
    }
}

void Sample::UpdateViews( nvrhi::IFramebuffer* framebuffer )
{
    // we currently use TAA for jitter even when it's not used itself
    if (m_temporalAntiAliasingPass)
        m_temporalAntiAliasingPass->SetJitter(m_ui.TemporalAntiAliasingJitter);

    nvrhi::Viewport windowViewport(float(m_renderSize.x), float(m_renderSize.y));
    m_view->SetViewport(windowViewport);
    float outputAspectRatio = m_displayAspectRatio; //windowViewport.width() / windowViewport.height();    // render and display outputs might not match in case of lower DLSS/etc resolution rounding!
    m_view->SetMatrices(m_camera.GetWorldToViewMatrix(), perspProjD3DStyleReverse(m_cameraVerticalFOV, outputAspectRatio, m_cameraZNear));
    m_view->SetPixelOffset(ComputeCameraJitter(m_sampleIndex));
    m_view->UpdateCache();
    if (GetFrameIndex() == 0)
    {
        m_viewPrevious->SetMatrices(m_view->GetViewMatrix(), m_view->GetProjectionMatrix());
        m_viewPrevious->SetPixelOffset(m_view->GetPixelOffset());
        m_viewPrevious->UpdateCache();
    }
}

void Sample::CollectUncompressedTextures()
{
    // Make a list of uncompressed textures
    m_uncompressedTextures.clear();
    auto listUncompressedTextureIfNeeded = [ & ](std::shared_ptr<LoadedTexture> texture, bool normalMap)//, TextureCompressionType compressionType)
    {
        if (texture == nullptr || texture->texture == nullptr)
            return;
        nvrhi::TextureDesc desc = texture->texture->getDesc();
        if (nvrhi::getFormatInfo(desc.format).blockSize != 1) // it's compressed, everything is fine!
            return;
        TextureCompressionType compressionType = normalMap ? (TextureCompressionType::Normalmap) : (
            (nvrhi::getFormatInfo(desc.format).isSRGB) ? (TextureCompressionType::GenericSRGB) : (TextureCompressionType::GenericLinear));

        auto it = m_uncompressedTextures.insert(std::make_pair(texture, compressionType));
        if (!it.second)
        {
            assert(it.first->second == compressionType); // not the same compression type? that's bad!
            return;
        }
    };
    for ( auto textureIT : m_materialsBaker->GetUsedTextures() )
        listUncompressedTextureIfNeeded(textureIT.second.Loaded, textureIT.second.NormalMap);
}

void Sample::SceneLoaded( )
{
    ApplicationBase::SceneLoaded( );

    m_sceneTime = 0.f;
    m_scene->FinishedLoading( GetFrameIndex( ) );

	// Find lights; do this before special cases to avoid duplicates
	for (auto light : m_scene->GetSceneGraph()->GetLights())
	{
		m_lights.push_back(light);
	}

    // seem like sensible defaults
    m_ui.ToneMappingParams.exposureCompensation = 2.0f;
    m_ui.ToneMappingParams.exposureValue = 0.0f;

    std::shared_ptr<EnvironmentLight> envLight = FindEnvironmentLight(m_lights);
    m_envMapLocalPath = (envLight==nullptr)?(""):(envLight->path);
    m_ui.EnvironmentMapParams = EnvironmentMapRuntimeParameters();


    if (m_ui.TogglableNodes == nullptr)
    {
        m_ui.TogglableNodes = std::make_shared<std::vector<TogglableNode>>();
        UpdateTogglableNodes(*m_ui.TogglableNodes, GetScene()->GetSceneGraph()->GetRootNode().get()); // UNSAFE - make sure not to keep m_ui.TogglableNodes longer than scenegraph!
    }

    // clean up invisible lights / markers because they slow things down
    for (int i = (int)m_lights.size() - 1; i >= 0; i--)
    {
        LightConstants lc;
        m_lights[i]->FillLightConstants(lc);
        if (length(lc.color * lc.intensity) <= 1e-7f)
            m_lights.erase( m_lights.begin() + i );
    }

    if( m_envMapLocalPath != "" )
    {
        // Make sure that there's an environment light object attached to the scene,
        // so that RTXDI will pick it up and sample.
        if (envLight == nullptr)
        {
            envLight = std::make_shared<EnvironmentLight>();
            m_scene->GetSceneGraph()->AttachLeafNode(m_scene->GetSceneGraph()->GetRootNode(), envLight);
            m_lights.push_back(envLight);
        }
    }

    // setup camera - just load the last from the scene if available
    auto cameras = m_scene->GetSceneGraph( )->GetCameras( );
    auto camScene = (cameras.empty( ))?(nullptr):(std::dynamic_pointer_cast<PerspectiveCamera>(cameras.back()));

    if( camScene == nullptr )
    {
        m_camera.LookAt(float3(0.f, 1.8f, 0.f), float3(1.f, 1.55f, 0.f), float3(0, 1, 0));
        m_cameraVerticalFOV = dm::radians(60.0f);
        m_cameraZNear = 0.001f;
    }
    else
    {
        UpdateCameraFromScene( camScene );
    }

    // mark raytracing acceleration structures as dirty
    m_ui.AccelerationStructRebuildRequested = true;
    m_subInstanceCount = 0;

    // if we don't re-set these, BLAS-es for animated stuff don't get updated
    for( const auto& anim : m_scene->GetSceneGraph( )->GetAnimations( ) )
        (void)anim->Apply( 0.0f );

    // PrintSceneGraph( m_scene->GetSceneGraph( )->GetRootNode( ) );

    m_ui.ShaderReloadRequested = true;  // we have to re-create shader hit table
    m_ui.EnableAnimations = false;
    m_ui.RealtimeMode = false;
    m_ui.UseReSTIRDI = false;
    m_ui.UseReSTIRGI = false;

    std::shared_ptr<SampleSettings> settings = m_scene->GetSampleSettingsNode();
    if (settings != nullptr)
    {
        m_ui.RealtimeMode = settings->realtimeMode.value_or(m_ui.RealtimeMode);
        m_ui.EnableAnimations = settings->enableAnimations.value_or(m_ui.EnableAnimations);
        if (settings->enableReSTIRDI.value_or(false))
            m_ui.UseReSTIRDI = true;
        if (settings->enableReSTIRGI.value_or(false))
            m_ui.UseReSTIRGI = true;
        if (settings->startingCamera.has_value())
            m_selectedCameraIndex = settings->startingCamera.value()+1; // slot 0 reserved for free flight camera
        if (settings->realtimeFireflyFilter.has_value())
        {
            m_ui.RealtimeFireflyFilterThreshold = settings->realtimeFireflyFilter.value();
            m_ui.RealtimeFireflyFilterEnabled = true;
        }
        m_ui.BounceCount = settings->maxBounces.value_or(m_ui.BounceCount);
        m_ui.RealtimeDiffuseBounceCount = settings->realtimeMaxDiffuseBounces.value_or(m_ui.RealtimeDiffuseBounceCount);
        m_ui.ReferenceDiffuseBounceCount = settings->referenceMaxDiffuseBounces.value_or(m_ui.ReferenceDiffuseBounceCount);
        m_ui.TexLODBias = settings->textureMIPBias.value_or(m_ui.TexLODBias);
    }

    LocalConfig::PostSceneLoad( *this, m_ui );

    if (m_materialsBaker!=nullptr) m_materialsBaker->SceneReloaded();
    if (m_envMapBaker!=nullptr) m_envMapBaker->SceneReloaded();
    if (m_lightsBaker!=nullptr) m_lightsBaker->SceneReloaded();
    if (m_ommBaker!=nullptr) m_ommBaker->SceneLoaded(m_scene);
}

bool Sample::KeyboardUpdate(int key, int scancode, int action, int mods)
{
    m_camera.KeyboardUpdate(key, scancode, action, mods);

    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    {
        m_ui.EnableAnimations = !m_ui.EnableAnimations;
        return true;
    }
    if( key == GLFW_KEY_F2 && action == GLFW_PRESS )
        m_ui.ShowUI = !m_ui.ShowUI;
    if( key == GLFW_KEY_R && action == GLFW_PRESS && mods == GLFW_MOD_CONTROL )
        m_ui.ShaderReloadRequested = true;

#if DONUT_WITH_STREAMLINE
    if (key == GLFW_KEY_F13 && action == GLFW_PRESS)
    {
        // As GLFW abstracts away from Windows messages
        // We instead set the F13 as the PC_Ping key in the constants and compare against that.
         GetDeviceManager()->GetStreamline().ReflexTriggerPcPing(GetFrameIndex());
    }
#endif

    return true;
}

bool Sample::MousePosUpdate(double xpos, double ypos)
{
    m_camera.MousePosUpdate(xpos, ypos);

    float2 upscalingScale = float2(1,1);
    if (m_renderTargets != nullptr)
        upscalingScale = float2(m_renderSize)/float2(m_displaySize);

    m_pickPosition = uint2( static_cast<uint>( xpos * upscalingScale.x ), static_cast<uint>( ypos * upscalingScale.y ) );
    m_ui.MousePos = uint2( static_cast<uint>( xpos * upscalingScale.x ), static_cast<uint>( ypos * upscalingScale.y ) );

    return true;
}

bool Sample::MouseButtonUpdate(int button, int action, int mods)
{
    m_camera.MouseButtonUpdate(button, action, mods);

    if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_2)
    {
        m_pick = true;
        m_ui.DebugPixel = m_pickPosition;
    }

#if DONUT_WITH_STREAMLINE
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
         GetDeviceManager()->GetStreamline().ReflexTriggerFlash(GetFrameIndex());
    }
#endif

    return true;
}

bool Sample::MouseScrollUpdate(double xoffset, double yoffset)
{
    m_camera.MouseScrollUpdate(xoffset, yoffset);
    return true;
}

void Sample::Animate(float fElapsedTimeSeconds)
{
    if (m_ui.FPSLimiter>0)    // essential for stable video recording
        fElapsedTimeSeconds = 1.0f / (float)m_ui.FPSLimiter;

    m_camera.SetMoveSpeed(m_ui.CameraMoveSpeed);

    if( m_ui.ShaderAndACRefreshDelayedRequest > 0 )
    {
        m_ui.ShaderAndACRefreshDelayedRequest -= fElapsedTimeSeconds;
        if (m_ui.ShaderAndACRefreshDelayedRequest <= 0 )
        {
            m_ui.ShaderAndACRefreshDelayedRequest = 0;
            m_ui.ShaderReloadRequested = true;
            m_ui.AccelerationStructRebuildRequested = true;
        }
    }

    if (m_toneMappingPass)
        m_toneMappingPass->AdvanceFrame(fElapsedTimeSeconds);

    const bool enableAnimations = m_ui.EnableAnimations && m_ui.RealtimeMode;
    const bool enableAnimationUpdate = enableAnimations || m_ui.ResetAccumulation;

    if (IsSceneLoaded() && enableAnimationUpdate)
    {
        if (enableAnimations)
            m_sceneTime += fElapsedTimeSeconds * 0.5f;
        float offset = 0;

        if (m_ui.LoopLongestAnimation)
        {
            float longestAnim = 0.0f;
            for (const auto& anim : m_scene->GetSceneGraph()->GetAnimations())
                longestAnim = std::max( longestAnim, anim->GetDuration() );
            if (longestAnim > 0)
            {
                if( longestAnim > 0.0f && m_sceneTime > longestAnim )
                    m_sceneTime -= int(m_sceneTime/longestAnim)*longestAnim;
                for (const auto& anim : m_scene->GetSceneGraph()->GetAnimations())
                    anim->Apply((float)m_sceneTime);
            }
        }
        else // loop each animation individually
        {
            for (const auto& anim : m_scene->GetSceneGraph()->GetAnimations())
                anim->Apply((float)fmod(m_sceneTime, anim->GetDuration()));
        }
    }
    else
    {
        m_sceneTime = 0.0f;
    }

    m_selectedCameraIndex = std::min( m_selectedCameraIndex, GetSceneCameraCount()-1 );
    if (m_selectedCameraIndex > 0)
    {
        std::shared_ptr<donut::engine::PerspectiveCamera> sceneCamera = std::dynamic_pointer_cast<PerspectiveCamera>(m_scene->GetSceneGraph()->GetCameras()[m_selectedCameraIndex-1]);
        if (sceneCamera != nullptr)
            UpdateCameraFromScene( sceneCamera );
    }

    m_camera.Animate(fElapsedTimeSeconds);

    dm::float3 camPos = m_camera.GetPosition();
    dm::float3 camDir = m_camera.GetDir();
    dm::float3 camUp = m_camera.GetUp();

    // if camera moves, reset accumulation
    if (m_lastCamDir.x != camDir.x || m_lastCamDir.y != camDir.y || m_lastCamDir.z != camDir.z || m_lastCamPos.x != camPos.x || m_lastCamPos.y != camPos.y || m_lastCamPos.z != camPos.z
        || m_lastCamUp.x != camUp.x || m_lastCamUp.y != camUp.y || m_lastCamUp.z != camUp.z )
    {
        m_lastCamPos = camPos;
        m_lastCamDir = camDir;
        m_lastCamUp = camUp;
        if( !m_ui.RealtimeMode )
            m_ui.ResetAccumulation = true;
    }

    double frameTime = GetDeviceManager()->GetAverageFrameTimeSeconds();
    if (frameTime > 0.0)
    {
#if DONUT_WITH_STREAMLINE
        if (m_ui.DLSSGMultiplier != 1)
            m_fpsInfo = string_format("%.3f ms/%d-frames* (%.1f FPS*) *DLSS-G", frameTime * 1e3, m_ui.DLSSGMultiplier, m_ui.DLSSGMultiplier / frameTime);
        else
#endif
            m_fpsInfo = string_format("%.3f ms/frame (%.1f FPS)", frameTime * 1e3, 1.0 / frameTime);
    }

    // Window title
    std::string extraInfo = ", " + m_fpsInfo + ", " + m_currentSceneName + ", " + GetResolutionInfo() + ", (L: " + std::to_string(m_scene->GetSceneGraph()->GetLights().size()) + ", MAT: " + std::to_string(m_scene->GetSceneGraph()->GetMaterials().size())
        + ", MESH: " + std::to_string(m_scene->GetSceneGraph()->GetMeshes().size()) + ", I: " + std::to_string(m_scene->GetSceneGraph()->GetMeshInstances().size()) + ", SI: " + std::to_string(m_scene->GetSceneGraph()->GetSkinnedMeshInstances().size())
        //+ ", AvgLum: " + std::to_string((m_renderTargets!=nullptr)?(m_renderTargets->AvgLuminanceLastCaptured):(0.0f))
#if ENABLE_DEBUG_VIZUALISATION
        + ", ENABLE_DEBUG_VIZUALISATION: 1"
#endif
        + ")";


    GetDeviceManager()->SetInformativeWindowTitle(g_windowTitle, false, extraInfo.c_str());
}

std::string Sample::GetResolutionInfo() const
{
    if (m_renderTargets == nullptr || m_renderTargets->OutputColor == nullptr)
        return "uninitialized";

    if (dm::all(m_renderSize == m_displaySize))
        return std::to_string(m_renderSize.x) + "x" + std::to_string(m_renderSize.y);
    else
        return std::to_string(m_renderSize.x) + "x" + std::to_string(m_renderSize.y) + "->" + std::to_string(m_displaySize.x) + "x" + std::to_string(m_displaySize.y);
}

float Sample::GetAvgTimePerFrame() const
{
    if (m_benchFrames == 0)
        return 0.0f;
    std::chrono::duration<double> elapsed = (m_benchLast - m_benchStart);
    return float(elapsed.count() / m_benchFrames);
}

void Sample::SaveCurrentCamera()
{
    float3 worldPos = m_camera.GetPosition();
    float3 worldDir = m_camera.GetDir();
    float3 worldUp  = m_camera.GetUp();
    dm::dquat rotation;
    dm::affine3 sceneWorldToView = dm::scaling(dm::float3(1.f, 1.f, -1.f)) * dm::inverse(m_camera.GetWorldToViewMatrix()); // see SceneCamera::GetViewToWorldMatrix
    dm::decomposeAffine<double>( daffine3(sceneWorldToView), nullptr, &rotation, nullptr );

    float4x4 projMatrix = m_view->GetProjectionMatrix();
    bool rowMajor = true;
    float tanHalfFOVY = 1.0f / (projMatrix.m_data[1 * 4 + 1]);
    float fovY = atanf(tanHalfFOVY) * 2.0f;

    bool autoExposure = m_ui.ToneMappingParams.autoExposure;
    float exposureCompensation = m_ui.ToneMappingParams.exposureCompensation;
    float exposureValue = m_ui.ToneMappingParams.exposureValue;

    std::ofstream file;
    file.open(app::GetDirectoryWithExecutable( ) / "campos.txt", std::ios_base::out | std::ios_base::trunc );
    if( file.is_open() )
    {
        file << worldPos.x << " " << worldPos.y << " " << worldPos.z << " " << std::endl;
        file << worldDir.x << " " << worldDir.y << " " << worldDir.z << " " << std::endl;
        file << worldUp.x  << " " << worldUp.y  << " " << worldUp.z  << " " << std::endl;

        file << std::endl;
        file << "below is the camera node that can be included into the *.scene.json;" << std::endl;
        file << "'Cameras' node goes into 'Graph' array" << std::endl;
        file << std::endl;
        file << "{"                                                             << std::endl;
        file << "    \"name\": \"Cameras\","                                    << std::endl;
        file << "        \"children\" : ["                                      << std::endl;
        file << "    {"                                                         << std::endl;
        file << "        \"name\": \"Default\","                                   << std::endl;
        file << "        \"type\" : \"PerspectiveCameraEx\","                     << std::endl;
        file << "        \"translation\" : [" << std::to_string(worldPos.x) << ", " << std::to_string(worldPos.y) << ", " << std::to_string(worldPos.z) << "]," << std::endl;
        file << "        \"rotation\" : [" << std::to_string(rotation.x) << ", " << std::to_string(rotation.y) << ", " << std::to_string(rotation.z) << ", " << std::to_string(rotation.w) << "]," << std::endl;
        file << "        \"verticalFov\" : " << std::to_string(fovY)            << "," << std::endl;
        file << "        \"zNear\" : " << std::to_string(m_cameraZNear)         << "," << std::endl;
        file << "        \"enableAutoExposure\" : " << (autoExposure?"true":"false") << "," << std::endl;
        file << "        \"exposureCompensation\" : " << std::to_string(exposureCompensation) << "," << std::endl;
        file << "        \"exposureValue\" : " << std::to_string(exposureValue) << std::endl;
        file << "    }"                                                         << std::endl;
        file << "        ]"                                                     << std::endl;
        file << "},"                                                            << std::endl;

        file.close();
    }
}

void Sample::LoadCurrentCamera()
{
    float3 worldPos;
    float3 worldDir;
    float3 worldUp;

    std::ifstream file;
    file.open(app::GetDirectoryWithExecutable( ) / "campos.txt", std::ios_base::in);
    if (file.is_open())
    {
        file >> worldPos.x >> std::ws >> worldPos.y >> std::ws >> worldPos.z; file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        file >> worldDir.x >> std::ws >> worldDir.y >> std::ws >> worldDir.z; file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        file >> worldUp.x  >> std::ws >> worldUp.y  >> std::ws >> worldUp.z; file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        file.close();
        m_camera.LookAt( worldPos, worldPos + worldDir, worldUp );
    }
}

struct HitGroupInfo
{
    std::string ExportName;
    std::string ClosestHitShader;
    std::string AnyHitShader;
};

MaterialShadingProperties MaterialShadingProperties::Compute(const MaterialPT & material)
{
    MaterialShadingProperties props;
    props.AlphaTest = material.EnableAlphaTesting;
    props.HasTransmission = material.EnableTransmission;
    props.NoTransmission = !props.HasTransmission;
    props.NoTextures = (!material.EnableBaseTexture || material.BaseTexture.Loaded == nullptr)
        && (!material.EnableEmissiveTexture || material.EmissiveTexture.Loaded == nullptr)
        && (!material.EnableNormalTexture || material.NormalTexture.Loaded == nullptr)
        && (!material.EnableOcclusionRoughnessMetallicTexture || material.OcclusionRoughnessMetallicTexture.Loaded == nullptr)
        && (!material.EnableTransmissionTexture || material.TransmissionTexture.Loaded == nullptr);
    static const float kMinGGXRoughness = 0.08f; // see BxDF.hlsli, kMinGGXAlpha constant: kMinGGXRoughness must match sqrt(kMinGGXAlpha)!
    props.OnlyDeltaLobes = ((props.HasTransmission && material.TransmissionFactor == 1.0) || (material.Metalness == 1)) && (material.Roughness < kMinGGXRoughness) && !(material.EnableOcclusionRoughnessMetallicTexture && material.OcclusionRoughnessMetallicTexture.Loaded != nullptr);
    props.ExcludeFromNEE = material.ExcludeFromNEE;
    return props;
}

// see OptimizationHints
HitGroupInfo ComputeSubInstanceHitGroupInfo(const MaterialPT & material)
{
    MaterialShadingProperties matProps = MaterialShadingProperties::Compute(material);

    HitGroupInfo info;

    info.ClosestHitShader = "ClosestHit";
    info.ClosestHitShader += std::to_string(matProps.NoTextures);
    info.ClosestHitShader += std::to_string(matProps.NoTransmission);
    info.ClosestHitShader += std::to_string(matProps.OnlyDeltaLobes);

    info.AnyHitShader = matProps.AlphaTest?"AnyHit":"";

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

bool Sample::CreatePTPipeline(engine::ShaderFactory& shaderFactory)
{
    bool SERSupported = GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12 && GetDevice()->queryFeatureSupport(nvrhi::Feature::ShaderExecutionReordering);

    assert( m_subInstanceCount > 0 );
    std::vector<HitGroupInfo> perSubInstanceHitGroup;
    perSubInstanceHitGroup.reserve(m_subInstanceCount);
    for (const auto& instance : m_scene->GetSceneGraph()->GetMeshInstances())
    {
        uint instanceID = (uint)perSubInstanceHitGroup.size();
        for (int gi = 0; gi < instance->GetMesh()->geometries.size(); gi++)
            perSubInstanceHitGroup.push_back(ComputeSubInstanceHitGroupInfo( *MaterialPT::FromDonut(instance->GetMesh()->geometries[gi]->material)) );
    }

    // Prime the instances to make sure we only include the nessesary CHS variants in the PSO.
    std::unordered_map<std::string, HitGroupInfo> uniqueHitGroups;
    for (int i = 0; i < perSubInstanceHitGroup.size(); i++)
        uniqueHitGroups[perSubInstanceHitGroup[i].ExportName] = perSubInstanceHitGroup[i];

    // We use separate variants for
    //  - PATH_TRACER_MODE : because it modifies path payload and has different code coverage; switching dynamically significantly reduces shader compiler's ability to optimize
    //  - USE_HIT_OBJECT_EXTENSION : because it requires use of extended API
    for (int variant = 0; variant < c_PathTracerVariants; variant++)
    {
        std::vector<engine::ShaderMacro> defines;

        // must match shaders.cfg - USE_HIT_OBJECT_EXTENSION path will possibly go away once part of API (it can be dynamic)
        if (variant == 0) { defines.push_back({ "PATH_TRACER_MODE", "PATH_TRACER_MODE_REFERENCE" });      defines.push_back({ "USE_HIT_OBJECT_EXTENSION", "0" }); }
        if (variant == 1) { defines.push_back({ "PATH_TRACER_MODE", "PATH_TRACER_MODE_BUILD_STABLE_PLANES" });    defines.push_back({ "USE_HIT_OBJECT_EXTENSION", "0" }); }
        if (variant == 2) { defines.push_back({ "PATH_TRACER_MODE", "PATH_TRACER_MODE_FILL_STABLE_PLANES" });    defines.push_back({ "USE_HIT_OBJECT_EXTENSION", "0" }); }
        if (variant == 3) { defines.push_back({ "PATH_TRACER_MODE", "PATH_TRACER_MODE_REFERENCE" });      defines.push_back({ "USE_HIT_OBJECT_EXTENSION", "1" }); }
        if (variant == 4) { defines.push_back({ "PATH_TRACER_MODE", "PATH_TRACER_MODE_BUILD_STABLE_PLANES" });    defines.push_back({ "USE_HIT_OBJECT_EXTENSION", "1" }); }
        if (variant == 5) { defines.push_back({ "PATH_TRACER_MODE", "PATH_TRACER_MODE_FILL_STABLE_PLANES" });    defines.push_back({ "USE_HIT_OBJECT_EXTENSION", "1" }); }

        m_PTShaderLibrary[variant] = shaderFactory.CreateShaderLibrary("app/Sample.hlsl", &defines);

        if (!m_PTShaderLibrary)
            return false;

        const bool exportAnyHit = variant < 3; // non-USE_HIT_OBJECT_EXTENSION codepaths require miss and hit; USE_HIT_OBJECT_EXTENSION codepaths can handle miss & anyhit inline!

        nvrhi::rt::PipelineDesc pipelineDesc;
        pipelineDesc.globalBindingLayouts = { m_bindingLayout, m_bindlessLayout };
        pipelineDesc.shaders.push_back({ "", m_PTShaderLibrary[variant]->getShader("RayGen", nvrhi::ShaderType::RayGeneration), nullptr });
        pipelineDesc.shaders.push_back({ "", m_PTShaderLibrary[variant]->getShader("Miss", nvrhi::ShaderType::Miss), nullptr });

        for (auto& [_, hitGroupInfo]: uniqueHitGroups)
        {
            pipelineDesc.hitGroups.push_back(
                {
                    .exportName = hitGroupInfo.ExportName,
                    .closestHitShader = m_PTShaderLibrary[variant]->getShader(hitGroupInfo.ClosestHitShader.c_str(), nvrhi::ShaderType::ClosestHit),
                    .anyHitShader = (exportAnyHit && hitGroupInfo.AnyHitShader!="")?(m_PTShaderLibrary[variant]->getShader(hitGroupInfo.AnyHitShader.c_str(), nvrhi::ShaderType::AnyHit)):(nullptr),
                    .intersectionShader = nullptr,
                    .bindingLayout = nullptr,
                    .isProceduralPrimitive = false
                }
            );
        }

        pipelineDesc.maxPayloadSize = PATH_TRACER_MAX_PAYLOAD_SIZE;
        pipelineDesc.maxRecursionDepth = 1; // 1 is enough if using inline visibility rays

        if (SERSupported)
            pipelineDesc.hlslExtensionsUAV = NV_SHADER_EXTN_SLOT_NUM;

        m_PTPipeline[variant] = GetDevice()->createRayTracingPipeline(pipelineDesc);

        if (!m_PTPipeline)
            return false;

        m_PTShaderTable[variant] = m_PTPipeline[variant]->createShaderTable();

        if (!m_PTShaderTable)
            return false;

        m_PTShaderTable[variant]->setRayGenerationShader("RayGen");
        for (int i = 0; i < perSubInstanceHitGroup.size(); i++)
            m_PTShaderTable[variant]->addHitGroup(perSubInstanceHitGroup[i].ExportName.c_str());

        m_PTShaderTable[variant]->addMissShader("Miss");
    }

    {
        std::vector<donut::engine::ShaderMacro> shaderMacros;
		// shaderMacros.push_back(donut::engine::ShaderMacro({ "USE_RTXDI", "0" }));
        m_exportVBufferCS = m_shaderFactory->CreateShader("app/ExportVisibilityBuffer.hlsl", "main", &shaderMacros, nvrhi::ShaderType::Compute);
        nvrhi::ComputePipelineDesc pipelineDesc;
		pipelineDesc.bindingLayouts = { m_bindingLayout, m_bindlessLayout };
		pipelineDesc.CS = m_exportVBufferCS;
        m_exportVBufferPSO = GetDevice()->createComputePipeline(pipelineDesc);
    }

    return true;
}

// sub-instance is a geometry within an instance
SubInstanceData ComputeSubInstanceData(const donut::engine::MeshInstance& meshInstance, uint meshInstanceIndex, const donut::engine::MeshGeometry& geometry, uint meshGeometryIndex, const MaterialPT & material)
{
    MaterialShadingProperties props = MaterialShadingProperties::Compute(material);

    SubInstanceData ret;

    bool alphaTest = props.AlphaTest;

    // we need alpha texture for alpha testing to work - disable otherwise
    if (alphaTest && (!material.EnableBaseTexture || material.BaseTexture.Loaded == nullptr))
        alphaTest = false;

    bool hasTransmission = props.HasTransmission;
    bool notMiss = true; // because miss defaults to 0 :)

    bool hasEmissive = material.IsEmissive();
    bool noTextures = props.NoTextures;
    bool hasNonDeltaLobes = !props.OnlyDeltaLobes;

    ret.FlagsAndSERSortKey = 0;
    ret.FlagsAndSERSortKey |= alphaTest ? 1 : 0;
    ret.FlagsAndSERSortKey <<= 1;
    ret.FlagsAndSERSortKey |= hasTransmission ? 1 : 0;
    ret.FlagsAndSERSortKey <<= 1;
    ret.FlagsAndSERSortKey |= hasEmissive ? 1 : 0;
    ret.FlagsAndSERSortKey <<= 1;
    ret.FlagsAndSERSortKey |= noTextures ? 1 : 0;
    ret.FlagsAndSERSortKey <<= 1;
    ret.FlagsAndSERSortKey |= hasNonDeltaLobes ? 1 : 0;

    ret.FlagsAndSERSortKey <<= 10;
    ret.FlagsAndSERSortKey |= meshInstanceIndex;

    ret.FlagsAndSERSortKey <<= 1;
    ret.FlagsAndSERSortKey |= notMiss ? 1 : 0;

#if 0
    ret.FlagsAndSERSortKey = 1 + material.materialID;
#elif 0
    const uint bitsForGeometryIndex = 6;
    ret.FlagsAndSERSortKey = 1 + (meshInstanceIndex << bitsForGeometryIndex) + meshGeometryIndex; // & ((1u << bitsForGeometryIndex) - 1u) );
#endif

    ret.FlagsAndSERSortKey &= 0xFFFF; // 16 bits for sort key above, clean anything else, the rest is used for flags

    float alphaCutoff = 0.0;

    const std::shared_ptr<MeshInfo>& mesh = meshInstance.GetMesh();
    if (alphaTest)
    {
        ret.FlagsAndSERSortKey |= SubInstanceData::Flags_AlphaTested;

        assert(mesh->buffers->hasAttribute(VertexAttribute::TexCoord1));
        assert(material.EnableBaseTexture && material.BaseTexture.Loaded != nullptr); // disable alpha testing if this happens to be possible
        ret.AlphaTextureIndex = material.BaseTexture.Loaded->bindlessDescriptor.Get();
        // ret.AlphaCutoff = material.alphaCutoff;
        alphaCutoff = material.AlphaCutoff;
    }
    uint globalGeometryIndex = mesh->geometries[0]->globalGeometryIndex + meshGeometryIndex;
    uint globalMaterialIndex = 0; // updated by MaterialsBaker with `std::dynamic_pointer_cast<MaterialPT>(mesh->geometries[0]->material)->GPUDataIndex;`
    ret.GlobalGeometryIndex_MaterialPTDataIndex = (globalGeometryIndex << 16) | globalMaterialIndex; assert(globalGeometryIndex <= 0xFFFF);
    ret.EmissiveLightMappingOffset = 0xFFFFFFFF;

    uint quantizedAlphaCutoff = (uint)(dm::clamp( alphaCutoff, 0.0f, 1.0f )*255.0f+0.5f); assert( quantizedAlphaCutoff < 256 );
    ret.FlagsAndSERSortKey |= (quantizedAlphaCutoff << SubInstanceData::Flags_AlphaOffsetOffset);

    if (material.ExcludeFromNEE)
    {
        ret.FlagsAndSERSortKey |= SubInstanceData::Flags_ExcludeFromNEE;
    }

    return ret;
}

void Sample::CreateBlases(nvrhi::ICommandList* commandList)
{
    for (const std::shared_ptr<MeshInfo>& mesh : m_scene->GetSceneGraph()->GetMeshes())
    {
        if (mesh->isSkinPrototype) //buffers->hasAttribute(engine::VertexAttribute::JointWeights))
            continue; // skip the skinning prototypes

        bvh::Config cfg = { .excludeTransmissive = m_ui.AS.ExcludeTransmissive };

        nvrhi::rt::AccelStructDesc blasDesc = bvh::GetMeshBlasDesc(cfg , *mesh, nullptr);
        assert((int)blasDesc.bottomLevelGeometries.size() < (1 << 12)); // we can only hold 13 bits for the geometry index in the HitInfo - see GeometryInstanceID in SceneTypes.hlsli

        nvrhi::rt::AccelStructHandle as = GetDevice()->createAccelStruct(blasDesc);

        nvrhi::utils::BuildBottomLevelAccelStruct(commandList, as, blasDesc);

        mesh->accelStruct = as;
    }
}

void Sample::UploadSubInstanceData(nvrhi::ICommandList* commandList)
{
    assert(m_subInstanceCount == m_subInstanceData.size());
    // upload data to GPU buffer
    commandList->writeBuffer(m_subInstanceBuffer, m_subInstanceData.data(), m_subInstanceData.size() * sizeof(SubInstanceData));
}

void Sample::UpdateSubInstanceContents()
{
    assert(m_subInstanceData.size() == m_subInstanceCount);
    uint subInstanceIndex = 0;
    for (const auto& instance : m_scene->GetSceneGraph()->GetMeshInstances())
    {
        uint instanceID = (uint)m_subInstanceData.size();
        for (int gi = 0; gi < instance->GetMesh()->geometries.size(); gi++)
            m_subInstanceData[subInstanceIndex++] = ComputeSubInstanceData(*instance, instanceID, *instance->GetMesh()->geometries[gi], gi, *MaterialPT::FromDonut(instance->GetMesh()->geometries[gi]->material) );
    }
    assert(subInstanceIndex == m_subInstanceCount);
}

void Sample::CreateTlas(nvrhi::ICommandList* commandList)
{

    nvrhi::rt::AccelStructDesc tlasDesc;
    tlasDesc.isTopLevel = true;
    tlasDesc.topLevelMaxInstances = m_scene->GetSceneGraph()->GetMeshInstances().size();
    tlasDesc.buildFlags = nvrhi::rt::AccelStructBuildFlags::PreferFastTrace;
    assert( tlasDesc.topLevelMaxInstances < (1 << 15) ); // we can only hold 16 bits for the identifier in the HitInfo - see GeometryInstanceID in SceneTypes.hlsli
    m_topLevelAS = GetDevice()->createAccelStruct(tlasDesc);


    // setup subInstances (entry is per geometry per instance) - some of it might require rebuild at runtime in more realistic scenarios
    {
        // figure out the required number
        m_subInstanceCount = 0;
        for (const auto& instance : m_scene->GetSceneGraph()->GetMeshInstances())
            m_subInstanceCount += (uint)instance->GetMesh()->geometries.size();
        // create GPU buffer
        nvrhi::BufferDesc bufferDesc;
        bufferDesc.byteSize = sizeof(SubInstanceData) * m_subInstanceCount;
        bufferDesc.debugName = "Instances";
        bufferDesc.structStride = sizeof(SubInstanceData);
        bufferDesc.canHaveRawViews = false;
        bufferDesc.canHaveUAVs = true;
        bufferDesc.isVertexBuffer = false;
        bufferDesc.initialState = nvrhi::ResourceStates::Common;
        bufferDesc.keepInitialState = true;
        m_subInstanceBuffer = GetDevice()->createBuffer(bufferDesc);
        // figure out the data
        m_subInstanceData.clear();
        m_subInstanceData.insert(m_subInstanceData.begin(), m_subInstanceCount, SubInstanceData{ .FlagsAndSERSortKey = 0, .GlobalGeometryIndex_MaterialPTDataIndex = 0, .AlphaTextureIndex = 0, .EmissiveLightMappingOffset = 0xFFFFFFFF } );
        UpdateSubInstanceContents();
        UploadSubInstanceData(commandList);
    }
}

void Sample::CreateAccelStructs(nvrhi::ICommandList* commandList)
{
    m_ommBaker->CreateOpacityMicromaps(m_scene);
    CreateBlases(commandList);
    CreateTlas(commandList);
}

void Sample::UpdateAccelStructs(nvrhi::ICommandList* commandList)
{
    // If the subInstanceData or BLAS build input data changes we trigger a full update here
    // could be made more efficient by only rebuilding the geometry in question,
    // or split the BLAS and subInstanceData updates
    if (m_ui.AccelerationStructRebuildRequested)
    {
        m_ui.AccelerationStructRebuildRequested = false;
        m_ui.ResetAccumulation = true;

        GetDevice()->waitForIdle();

        m_bindingSet = nullptr;
        m_topLevelAS = nullptr;

        for (const std::shared_ptr<MeshInfo>& _mesh : m_scene->GetSceneGraph()->GetMeshes())
        {
            assert(std::dynamic_pointer_cast<MeshInfoEx>(_mesh) != nullptr);
            const std::shared_ptr<MeshInfoEx>& mesh = std::static_pointer_cast<MeshInfoEx>(_mesh);
            mesh->accelStruct = nullptr;
            mesh->AccelStructOMM = nullptr;
            mesh->OpacityMicroMaps.clear();
            mesh->DebugData = nullptr;
            mesh->DebugDataDirty = true;
        }

        // raytracing acceleration structures
        commandList->open();
        CreateAccelStructs(commandList);
        commandList->close();
        GetDevice()->executeCommandList(commandList);
        GetDevice()->waitForIdle();
    }
}

void Sample::TransitionMeshBuffersToReadOnly(nvrhi::ICommandList* commandList)
{
    // Transition all the buffers to their necessary states before building the BLAS'es to allow BLAS batching
    for (const auto& skinnedInstance : m_scene->GetSceneGraph()->GetSkinnedMeshInstances())
        commandList->setBufferState(skinnedInstance->GetMesh()->buffers->vertexBuffer, nvrhi::ResourceStates::ShaderResource);
    commandList->commitBarriers();
}

void Sample::BuildTLAS(nvrhi::ICommandList* commandList, uint32_t frameIndex) const
{
    commandList->beginMarker("Skinned BLAS Updates");

    // Transition all the buffers to their necessary states before building the BLAS'es to allow BLAS batching
    for (const auto& skinnedInstance : m_scene->GetSceneGraph()->GetSkinnedMeshInstances())
    {
        if (skinnedInstance->GetLastUpdateFrameIndex() < frameIndex)
            continue;

        commandList->setAccelStructState(skinnedInstance->GetMesh()->accelStruct, nvrhi::ResourceStates::AccelStructWrite);
        commandList->setBufferState(skinnedInstance->GetMesh()->buffers->vertexBuffer, nvrhi::ResourceStates::AccelStructBuildInput);
    }
    commandList->commitBarriers();

    // Now build the BLAS'es
    for (const auto& skinnedInstance : m_scene->GetSceneGraph()->GetSkinnedMeshInstances())
    {
        if (skinnedInstance->GetLastUpdateFrameIndex() < frameIndex)
            continue;

        bvh::Config cfg = { .excludeTransmissive = m_ui.AS.ExcludeTransmissive };

        nvrhi::rt::AccelStructDesc blasDesc = bvh::GetMeshBlasDesc(cfg , *skinnedInstance->GetMesh(), nullptr);

        nvrhi::utils::BuildBottomLevelAccelStruct(commandList, skinnedInstance->GetMesh()->accelStruct, blasDesc);
    }
    commandList->endMarker();

    std::vector<nvrhi::rt::InstanceDesc> instances; // TODO: make this a member, avoid allocs :)

    uint subInstanceCount = 0;
    for (const auto& instance : m_scene->GetSceneGraph()->GetMeshInstances())
    {
        const bool ommDebugViewEnabled = m_ui.DebugView == DebugViewType::FirstHitOpacityMicroMapInWorld || m_ui.DebugView == DebugViewType::FirstHitOpacityMicroMapOverlay;
        assert( !ommDebugViewEnabled || ENABLE_DEBUG_OMM_VIZUALISATION );  // need to enable ENABLE_DEBUG_OMM_VIZUALISATION for this to work!
        // ommDebugViewEnabled must do two things: use a BLAS without OMMs and disable all alpha testing.
        // This may sound a bit counter intuitive, the goal is to intersect micro-triangles marked as transparent without them actually being treated as such.

        const std::shared_ptr<MeshInfoEx>& mesh = std::static_pointer_cast<MeshInfoEx>(instance->GetMesh());

        const bool forceOpaque          = ommDebugViewEnabled || m_ui.AS.ForceOpaque;
        const bool hasAttachementOMM    = m_ommBaker->IsEnabled() && mesh->AccelStructOMM.Get() != nullptr;
        const bool useOmmBLAS           = m_ommBaker->IsEnabled() && m_ommBaker->UIData().Enable && hasAttachementOMM && !forceOpaque;

        nvrhi::rt::InstanceDesc instanceDesc;
        instanceDesc.bottomLevelAS = useOmmBLAS ? mesh->AccelStructOMM.Get() : mesh->accelStruct.Get();
        instanceDesc.instanceMask = (m_ommBaker->IsEnabled() && m_ommBaker->UIData().OnlyOMMs && !hasAttachementOMM) ? 0 : 1;
        instanceDesc.instanceID = instance->GetGeometryInstanceIndex();
        instanceDesc.instanceContributionToHitGroupIndex = subInstanceCount;
        instanceDesc.flags = (m_ommBaker->IsEnabled() && m_ommBaker->UIData().Force2State) ? nvrhi::rt::InstanceFlags::ForceOMM2State : nvrhi::rt::InstanceFlags::None;
        if (forceOpaque)
            instanceDesc.flags = (nvrhi::rt::InstanceFlags)((uint32_t)instanceDesc.flags | (uint32_t)nvrhi::rt::InstanceFlags::ForceOpaque);

        assert( subInstanceCount == instance->GetGeometryInstanceIndex() );
        subInstanceCount += (uint)mesh->geometries.size();

        auto node = instance->GetNode();
        assert(node);
        dm::affineToColumnMajor(node->GetLocalToWorldTransformFloat(), instanceDesc.transform);

        instances.push_back(instanceDesc);
    }
    assert (m_subInstanceCount == subInstanceCount);

    // Compact acceleration structures that are tagged for compaction and have finished executing the original build
    commandList->compactBottomLevelAccelStructs();

    commandList->beginMarker("TLAS Update");
    commandList->buildTopLevelAccelStruct(m_topLevelAS, instances.data(), instances.size(), nvrhi::rt::AccelStructBuildFlags::AllowEmptyInstances);
    commandList->endMarker();
}


void Sample::BackBufferResizing()
{
    ApplicationBase::BackBufferResizing();
    //Todo: Needed for vulkan realtime path, remove
    if(GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN)
    {
      m_renderTargets = nullptr;
    }
    m_bindingCache->Clear();
    m_linesPipeline = nullptr; // the pipeline is based on the framebuffer so needs a reset
    for (int i=0; i < std::size(m_nrd); i++ )
        m_nrd[i] = nullptr;
    if (m_rtxdiPass)
        m_rtxdiPass->Reset();
}

void Sample::CreateRenderPasses( bool& exposureResetRequired, nvrhi::CommandListHandle initializeCommandList )
{
    const uint2 screenResolution = {m_renderTargets->OutputColor->getDesc().width, m_renderTargets->OutputColor->getDesc().height};

    m_shaderDebug = std::make_shared<ShaderDebug>(GetDevice(), initializeCommandList, m_shaderFactory, m_CommonPasses);

    m_rtxdiPass = std::make_unique<RtxdiPass>(GetDevice(), m_shaderFactory, m_CommonPasses, m_bindlessLayout);

    m_accumulationPass = std::make_unique<AccumulationPass>(GetDevice(), m_shaderFactory);
    m_accumulationPass->CreatePipeline();
    m_accumulationPass->CreateBindingSet(m_renderTargets->OutputColor, m_renderTargets->AccumulatedRadiance);

    // these get re-created every time intentionally, to pick up changes after at-runtime shader recompile
    m_toneMappingPass = std::make_unique<ToneMappingPass>(GetDevice(), m_shaderFactory, m_CommonPasses, m_renderTargets->LdrFramebuffer, *m_view, m_renderTargets->OutputColor);
    m_postProcess = std::make_shared<PostProcess>(GetDevice(), m_shaderFactory, m_CommonPasses, m_shaderDebug);

    for (int i = 0; i < std::size(m_nrd); i++)
    {
        if (m_nrd[i] == nullptr)
        {
            nrd::Denoiser denoiserMethod = m_ui.NRDMethod == NrdConfig::DenoiserMethod::REBLUR ?
                nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR : nrd::Denoiser::RELAX_DIFFUSE_SPECULAR;

            m_nrd[i] = std::make_unique<NrdIntegration>(GetDevice(), denoiserMethod);
            m_nrd[i]->Initialize(m_renderSize.x, m_renderSize.y, *m_shaderFactory);
        }
    }

    {
        TemporalAntiAliasingPass::CreateParameters taaParams;
        taaParams.sourceDepth = m_renderTargets->Depth;
        taaParams.motionVectors = m_renderTargets->ScreenMotionVectors;
        taaParams.unresolvedColor = m_renderTargets->OutputColor;
        taaParams.resolvedColor = m_renderTargets->ProcessedOutputColor;
        taaParams.feedback1 = m_renderTargets->TemporalFeedback1;
        taaParams.feedback2 = m_renderTargets->TemporalFeedback2;
        taaParams.historyClampRelax = m_renderTargets->CombinedHistoryClampRelax;
        taaParams.motionVectorStencilMask = 0; ///*uint32_t motionVectorStencilMask =*/ 0x01;
        taaParams.useCatmullRomFilter = true;

        m_temporalAntiAliasingPass = std::make_unique<TemporalAntiAliasingPass>(GetDevice(), m_shaderFactory, m_CommonPasses, *m_view, taaParams);
    }

    if (!CreatePTPipeline(*m_shaderFactory))
        { assert(false); }

    if (m_envMapBaker == nullptr || m_lightsBaker == nullptr)
    {
        m_envMapBaker = std::make_shared<EnvMapBaker>(GetDevice(), m_TextureCache, m_shaderFactory, m_CommonPasses);
        m_lightsBaker = std::make_shared<LightsBaker>(GetDevice(), m_TextureCache, m_shaderFactory, m_envMapBaker);
    }
    m_envMapBaker->CreateRenderPasses();
    m_lightsBaker->CreateRenderPasses(m_bindlessLayout, m_CommonPasses, m_shaderDebug, screenResolution);

#if 0 // enable if needed
    if (m_gpuSort == nullptr)
        m_gpuSort = std::make_shared<GPUSort>(GetDevice(), m_shaderFactory);
    m_gpuSort->CreateRenderPasses(m_CommonPasses, m_shaderDebug);
#endif
}

void Sample::PreUpdateLighting(nvrhi::CommandListHandle commandList, bool& needNewBindings)
{
    RAII_SCOPE(m_commandList->beginMarker("PreUpdateLighting"); , m_commandList->endMarker(); );

    auto preUpdateCube = m_envMapBaker->GetEnvMapCube();
    m_envMapBaker->PreUpdate(commandList, m_envMapLocalPath);

    if (preUpdateCube != m_envMapBaker->GetEnvMapCube())
        needNewBindings = true;
}

void Sample::UpdateLighting(nvrhi::CommandListHandle commandList)
{
    RAII_SCOPE( m_commandList->beginMarker("UpdateLighting");, m_commandList->endMarker(); );

    EMB_DirectionalLight dirLights[EnvMapBaker::c_MaxDirLights];
    uint dirLightCount = 0;
    {   // Find and pre-process directional analytic lights, and convert them to environment map local frame so they remain pointing in correct world direction!
        float3 rotationInRadians = radians(m_ui.EnvironmentMapParams.RotationXYZ);
        affine3 rotationTransform = donut::math::rotation(rotationInRadians);
        affine3 inverseTransform = inverse(rotationTransform);
        for (int i = 0; i < (int)m_lights.size(); i++)
        {
            std::shared_ptr<DirectionalLight> dirLight = std::dynamic_pointer_cast<DirectionalLight>(m_lights[i]);
            if( dirLight != nullptr )
            {
                LightConstants light;
                dirLight->FillLightConstants(light);

                const float minAngularSize = PI_f / (m_envMapBaker->GetTargetCubeResolution()/2.0f);
                assert( light.angularSizeOrInvRange >= minAngularSize );    // point lights smaller than this cannot be reliably baked into cubemap
                dirLights[dirLightCount].AngularSize = std::max( light.angularSizeOrInvRange, minAngularSize );
                dirLights[dirLightCount].ColorIntensity = float4(light.color, light.intensity);
                dirLights[dirLightCount].Direction = rotationTransform.transformVector(light.direction);
                dirLightCount++;
            }
        }
    }

    if (m_envMapBaker->Update(commandList, EnvMapBaker::BakeSettings { .EnvMapRadianceScale = c_envMapRadianceScale }, m_sceneTime, dirLights, dirLightCount) )
        m_ui.ResetAccumulation = true;

    {
        LightsBaker::BakeSettings settings;
        settings.ImportanceSamplingType = (uint)m_ui.NEEType;
        settings.CameraPosition = m_camera.GetPosition();
        settings.MouseCursorPos = m_ui.MousePos;
        settings.GlobalTemporalFeedbackEnabled  = m_ui.NEEAT_GlobalTemporalFeedbackEnabled;
        settings.GlobalTemporalFeedbackRatio    = m_ui.NEEAT_GlobalTemporalFeedbackRatio;
        settings.NarrowTemporalFeedbackEnabled  = m_ui.NEEAT_NarrowTemporalFeedbackEnabled;
        settings.NarrowTemporalFeedbackRatio    = m_ui.NEEAT_NarrowTemporalFeedbackRatio;
        settings.LightSampling_MIS_Boost        = m_ui.NEEAT_MIS_Boost;
        settings.DistantVsLocalImportanceScale  = m_ui.NEEAT_Distant_vs_Local_Importance;
        settings.ResetFeedback = m_ui.ResetAccumulation && !m_ui.RealtimeMode 
#if 1
            || m_ui.ResetRealtimeCaches
#endif
        ;
        settings.SampleIndex = m_sampleIndex;
        settings.PrevViewportSize = float2( (float)m_viewPrevious->GetViewExtent().width(), (float)m_viewPrevious->GetViewExtent().height() );
        settings.ViewportSize = float2( (float)m_view->GetViewExtent().width(), (float)m_view->GetViewExtent().height() );
        settings.EnvMapParams = m_envMapSceneParams;
        m_lightsBaker->Update(commandList, settings, m_sceneTime, m_scene, m_materialsBaker, m_ommBaker, m_subInstanceBuffer, m_subInstanceData);
    }
}

void Sample::PreUpdatePathTracing( bool resetAccum, nvrhi::CommandListHandle commandList )
{
    m_frameIndex++;

    resetAccum |= m_ui.ResetAccumulation;
    resetAccum |= m_ui.RealtimeMode;

    if( m_ui.AccumulationTarget != m_accumulationSampleTarget )
    {
        resetAccum = true;
        m_accumulationSampleTarget = m_ui.AccumulationTarget;
    }

    if (resetAccum)
    {
        m_accumulationSampleIndex = 0;
    }
#if ENABLE_DEBUG_VIZUALISATION
    if (resetAccum)
        m_shaderDebug->ClearDebugVizTexture(commandList);
#endif

    m_ui.AccumulationIndex = m_accumulationSampleIndex;

    // profile perf - only makes sense with high accumulation sample counts; only start counting after n-th after it stabilizes
    if( m_accumulationSampleIndex < 16 )
    {
        m_benchStart = std::chrono::high_resolution_clock::now( );
        m_benchLast = m_benchStart;
        m_benchFrames = 0;
    } else if( m_accumulationSampleIndex < m_accumulationSampleTarget )
    {
        m_benchFrames++;
        m_benchLast = std::chrono::high_resolution_clock::now( );
    }

    // 'min' in non-realtime path here is to keep looping the last sample for debugging purposes!
    if( !m_ui.RealtimeMode )
        m_sampleIndex = min(m_accumulationSampleIndex, m_accumulationSampleTarget - 1);
    else
        m_sampleIndex = (m_ui.RealtimeNoise)?( m_frameIndex % 1024 ):0;     // actual sample index
}

void Sample::PostUpdatePathTracing( )
{
    m_accumulationSampleIndex = std::min( m_accumulationSampleIndex+1, m_accumulationSampleTarget );

    if (m_ui.ActualUseRTXDIPasses())
        m_rtxdiPass->EndFrame();

    m_ui.ResetAccumulation = false;
    m_ui.ResetRealtimeCaches = false;
}

void Sample::UpdatePathTracerConstants( PathTracerConstants & constants, const PathTracerCameraData & cameraData )
{
#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
    auto GetStfMagnificationMethod = [](StfMagnificationMethod method)->int {
        switch (method)
        {
        case StfMagnificationMethod::Default:   return STF_MAGNIFICATION_METHOD_NONE;
        case StfMagnificationMethod::Quad2x2:   return STF_MAGNIFICATION_METHOD_2x2_QUAD;
        case StfMagnificationMethod::Fine2x2:   return STF_MAGNIFICATION_METHOD_2x2_FINE;
        case StfMagnificationMethod::FineTemporal2x2: return STF_MAGNIFICATION_METHOD_2x2_FINE_TEMPORAL;
        case StfMagnificationMethod::FineAlu3x3: return STF_MAGNIFICATION_METHOD_3x3_FINE_ALU;
        case StfMagnificationMethod::FineLut3x3: return STF_MAGNIFICATION_METHOD_3x3_FINE_LUT;
        case StfMagnificationMethod::Fine4x4:    return STF_MAGNIFICATION_METHOD_4x4_FINE;
        default:
            assert(!"Not Implemented");
            return 0;
        }
    };

    auto GetStfFilterMode = [](StfFilterMode mode)->int {
        switch (mode)
        {
        case StfFilterMode::Point:      return STF_FILTER_TYPE_POINT;
        case StfFilterMode::Linear:     return STF_FILTER_TYPE_LINEAR;
        case StfFilterMode::Cubic:      return STF_FILTER_TYPE_CUBIC;
        case StfFilterMode::Gaussian:   return STF_FILTER_TYPE_GAUSSIAN;
        default:
            assert(!"Not Implemented");
            return 0;
        }
    };
#endif // RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE

    constants.camera = cameraData;

    constants.bounceCount = m_ui.BounceCount;
    constants.diffuseBounceCount = (m_ui.RealtimeMode)?(m_ui.RealtimeDiffuseBounceCount):(m_ui.ReferenceDiffuseBounceCount);
    constants.perPixelJitterAAScale = (m_ui.RealtimeMode == false && m_ui.AccumulationAA)?(1):( (m_ui.RealtimeMode && m_ui.RealtimeAA == 3)?(m_ui.DLSSRRMicroJitter):(0.0f) );
    constants.texLODBias = m_ui.TexLODBias;
    constants.sampleBaseIndex = m_sampleIndex * m_ui.ActualSamplesPerPixel();

    //constants.subSampleCount = m_ui.ActualSamplesPerPixel();
    constants.invSubSampleCount = 1.0f / (float)m_ui.ActualSamplesPerPixel();

    constants.imageWidth = m_renderSize.x; assert( m_renderSize.x == m_renderTargets->OutputColor->getDesc().width );
    constants.imageHeight = m_renderSize.y; assert( m_renderSize.y == m_renderTargets->OutputColor->getDesc().height );

    // this is the dynamic luminance that when passed through current tonemapper with current exposure settings, produces the same 50% gray
    constants.preExposedGrayLuminance = m_ui.EnableToneMapping?(donut::math::luminance(m_toneMappingPass->GetPreExposedGray(0))):(1.0f);

    const float disabledFF = 0.0f;
    if (m_ui.RealtimeMode)
        constants.fireflyFilterThreshold = (m_ui.RealtimeFireflyFilterEnabled)?(m_ui.RealtimeFireflyFilterThreshold*sqrtf(constants.preExposedGrayLuminance)*1e3f):(disabledFF); // it does make sense to make the realtime variant dependent on avg luminance - just didn't have time to try it out yet
    else
        constants.fireflyFilterThreshold = (m_ui.ReferenceFireflyFilterEnabled)?(m_ui.ReferenceFireflyFilterThreshold*sqrtf(constants.preExposedGrayLuminance)*1e3f):(disabledFF); // making it exposure-adaptive breaks determinism with accumulation (because there's a feedback loop), so that's disabled
    constants.useReSTIRDI = m_ui.ActualUseReSTIRDI();
    constants.useReSTIRGI = m_ui.ActualUseReSTIRGI();
    constants.denoiserRadianceClampK = m_ui.DenoiserRadianceClampK;

    // no stable planes by default
    constants.denoisingEnabled = m_ui.ActualUseStandaloneDenoiser() || m_ui.RealtimeAA == 3;
    constants.suppressPrimaryNEE = m_ui.SuppressPrimaryNEE;

    constants.activeStablePlaneCount            = m_ui.StablePlanesActiveCount;
    constants.maxStablePlaneVertexDepth         = std::min( std::min( (uint)m_ui.StablePlanesMaxVertexDepth, cStablePlaneMaxVertexIndex ), (uint)m_ui.BounceCount );
    constants.allowPrimarySurfaceReplacement    = m_ui.AllowPrimarySurfaceReplacement;
    //constants.stablePlanesSkipIndirectNoiseP0   = m_ui.ActualSkipIndirectNoisePlane0();
    constants.stablePlanesSplitStopThreshold    = m_ui.StablePlanesSplitStopThreshold;
    constants._padding3                         = 0;
    constants.enableShaderExecutionReordering   = m_ui.ShaderExecutionReordering?1:0;
    constants.stablePlanesSuppressPrimaryIndirectSpecularK  = m_ui.StablePlanesSuppressPrimaryIndirectSpecular?m_ui.StablePlanesSuppressPrimaryIndirectSpecularK:0.0f;
    constants.stablePlanesAntiAliasingFallthrough = m_ui.StablePlanesAntiAliasingFallthrough;
    constants.enableRussianRoulette             = (m_ui.EnableRussianRoulette)?(1):(0);
    constants.frameIndex                        = GetFrameIndex();
    constants.genericTSLineStride               = GenericTSComputeLineStride(constants.imageWidth, constants.imageHeight);
    constants.genericTSPlaneStride              = GenericTSComputePlaneStride(constants.imageWidth, constants.imageHeight);

    constants.NEEEnabled                        = m_ui.UseNEE;
    constants.NEEType                           = m_ui.NEEType;
    constants.NEECandidateSamples               = m_ui.NEECandidateSamples;
    constants.NEEFullSamples                    = m_ui.NEEFullSamples;
    constants.NEEBoostSamplingOnDominantPlane   = m_ui.NEEBoostSamplingOnDominantPlane;

    constants.EnvironmentMapDiffuseSampleMIPLevel = m_ui.EnvironmentMapDiffuseSampleMIPLevel;

#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
    // stochastic texture filtering type and size.
    constants.STFUseBlueNoise                   = m_ui.STFUseBlueNoise;
    constants.STFMagnificationMethod            = GetStfMagnificationMethod(m_ui.STFMagnificationMethod);
    constants.STFFilterMode                     = GetStfFilterMode(m_ui.STFFilterMode);
    constants.STFGaussianSigma                  = m_ui.STFGaussianSigma;
#endif
}


void Sample::RtxdiSetupFrame(nvrhi::IFramebuffer* framebuffer, PathTracerCameraData cameraData, uint2 renderDims)
{
    const bool envMapPresent = m_ui.EnvironmentMapParams.Enabled;

    RtxdiBridgeParameters bridgeParameters;
	bridgeParameters.frameIndex = GetFrameIndex();
	bridgeParameters.frameDims = renderDims;
	bridgeParameters.cameraPosition = m_camera.GetPosition();
	bridgeParameters.userSettings = m_ui.RTXDI;
    bridgeParameters.usingLightSampling = m_ui.ActualUseReSTIRDI();
    bridgeParameters.usingReGIR = m_ui.ActualUseReSTIRDI();

    bridgeParameters.userSettings.restirDI.initialSamplingParams.environmentMapImportanceSampling = envMapPresent;

    if( m_ui.ResetRealtimeCaches )
        m_rtxdiPass->Reset();

	m_rtxdiPass->PrepareResources(m_commandList, *m_renderTargets, envMapPresent ? m_envMapBaker : nullptr, m_envMapSceneParams,
        m_scene, m_materialsBaker, m_ommBaker, m_subInstanceBuffer, bridgeParameters, m_bindingLayout, m_shaderDebug );
 }

bool Sample::ShouldRenderUnfocused()
{
    if (GetFrameIndex() == 0)
    {
        // Make sure we at least run one render frame to allow expensive resource creation to happen in background
        return true;
    }

    if (m_ui.RenderWhenOutOfFocus)
    {
        return true;
    }

    // Let Reference mode accumulate all frames before pausing
    return (!m_ui.RealtimeMode && (m_ui.AccumulationIndex < m_ui.AccumulationTarget));
}

// TODO: REFACTOR
void Sample::StreamlinePreRender()
{
#if DONUT_WITH_STREAMLINE
    // Setup Reflex
    {
        auto reflexConsts = donut::app::StreamlineInterface::ReflexOptions{};
        reflexConsts.mode = (donut::app::StreamlineInterface::ReflexMode) m_ui.ActualReflexMode();
        reflexConsts.frameLimitUs = m_ui.ReflexCappedFps == 0 ? 0 : int(1000000. / m_ui.ReflexCappedFps);
        reflexConsts.useMarkersToOptimize = true;
        reflexConsts.virtualKey = VK_F13;
        reflexConsts.idThread = 0; // std::hash<std::thread::id>()(std::this_thread::get_id())
        GetDeviceManager()->GetStreamline().SetReflexConsts(reflexConsts);

        // Need to update StreamlineIntegration with the ability to query reflex state
        donut::app::StreamlineInterface::ReflexState reflexState{};
        GetDeviceManager()->GetStreamline().GetReflexState(reflexState);
        if (m_ui.IsReflexSupported)
        {
            m_ui.IsReflexLowLatencyAvailable = reflexState.lowLatencyAvailable;
            m_ui.IsReflexFlashIndicatorDriverControlled = reflexState.flashIndicatorDriverControlled;

            auto report = reflexState.frameReport[63];
            if (reflexState.lowLatencyAvailable && report.gpuRenderEndTime != 0)
            {
                auto frameID = report.frameID;
                auto totalGameToRenderLatencyUs = report.gpuRenderEndTime - report.inputSampleTime;
                auto simDeltaUs = report.simEndTime - report.simStartTime;
                auto renderDeltaUs = report.renderSubmitEndTime - report.renderSubmitStartTime;
                auto presentDeltaUs = report.presentEndTime - report.presentStartTime;
                auto driverDeltaUs = report.driverEndTime - report.driverStartTime;
                auto osRenderQueueDeltaUs = report.osRenderQueueEndTime - report.osRenderQueueStartTime;
                auto gpuRenderDeltaUs = report.gpuRenderEndTime - report.gpuRenderStartTime;

                m_ui.ReflexStats = "frameID: " + std::to_string(frameID);
                m_ui.ReflexStats += "\ntotalGameToRenderLatencyUs: " + std::to_string(totalGameToRenderLatencyUs);
                m_ui.ReflexStats += "\nsimDeltaUs: " + std::to_string(simDeltaUs);
                m_ui.ReflexStats += "\nrenderDeltaUs: " + std::to_string(renderDeltaUs);
                m_ui.ReflexStats += "\npresentDeltaUs: " + std::to_string(presentDeltaUs);
                m_ui.ReflexStats += "\ndriverDeltaUs: " + std::to_string(driverDeltaUs);
                m_ui.ReflexStats += "\nosRenderQueueDeltaUs: " + std::to_string(osRenderQueueDeltaUs);
                m_ui.ReflexStats += "\ngpuRenderDeltaUs: " + std::to_string(gpuRenderDeltaUs);
            }
            else
            {
                m_ui.ReflexStats = "Latency Report Unavailable";
            }
        }
    }

    // DLSS-G Setup
    {
        // If DLSS-G has been turned off, then we tell tell SL to clean it up expressly
        if (m_ui.DLSSGOptions.mode == StreamlineInterface::DLSSGMode::eOn && m_ui.ActualDLSSGMode() == StreamlineInterface::DLSSGMode::eOff) {
            GetDeviceManager()->GetStreamline().CleanupDLSSG(true);
        }

        // This is where DLSS-G is toggled On and Off (using dlssgOptions.mode) and where we set DLSS-G parameters.
        auto dlssgOptions = StreamlineInterface::DLSSGOptions{};
        dlssgOptions.mode = m_ui.ActualDLSSGMode();
        dlssgOptions.numFramesToGenerate = m_ui.DLSSGNumFramesToGenerate;

        // This is where we query DLSS-G minimum swapchain size
        if (GetDeviceManager()->GetStreamline().IsDLSSGAvailable())
        {
            StreamlineInterface::DLSSGState state;
            GetDeviceManager()->GetStreamline().GetDLSSGState(state, dlssgOptions);
            m_ui.DLSSGMultiplier = state.numFramesActuallyPresented;
            m_ui.DLSSGMaxNumFramesToGenerate = state.numFramesToGenerateMax;

            GetDeviceManager()->GetStreamline().SetDLSSGOptions(dlssgOptions);
            m_ui.DLSSGOptions = dlssgOptions;
        }
    }

    // Ensure DLSS / DLSS-RR is available
    if (m_ui.RealtimeAA == 3 && !m_ui.IsDLSSRRSupported)
    {
        log::warning("Requested DLSS-RR mode not available. Switching to DLSS. ");
        m_ui.RealtimeAA = 2;
    }
    if ( m_ui.RealtimeAA == 2 && !m_ui.IsDLSSSuported )
    {
        log::warning("Requested DLSS mode not available. Switching to TAA. ");
        m_ui.RealtimeAA = 1;
    }

    // Setup DLSS
    const bool changeToDLSSMode = (m_ui.RealtimeAA >= 2 && m_ui.RealtimeAA <= 3) && m_ui.DLSSLastRealtimeAA != m_ui.RealtimeAA;
    {
        // Reset DLSS vars if we stop using it
        if (changeToDLSSMode || m_ui.DLSSMode == StreamlineInterface::DLSSMode::eOff)
        {
            m_ui.DLSSLastMode = SampleUIData::DLSSModeDefault;
            m_ui.DLSSMode = SampleUIData::DLSSModeDefault;
            m_ui.DLSSLastDisplaySize = { 0,0 };
        }

        m_ui.DLSSLastRealtimeAA = m_ui.RealtimeAA;

        // If we are using DLSS set its constants
        if ((m_ui.RealtimeAA == 2 || m_ui.RealtimeAA == 3) && m_ui.RealtimeMode)
        {
            StreamlineInterface::DLSSOptions dlssOptions = {};
            if (m_ui.IsDLSSSuported)
            {
                dlssOptions.mode = m_ui.DLSSMode;
                dlssOptions.outputWidth = m_displaySize.x;
                dlssOptions.outputHeight = m_displaySize.y;
                dlssOptions.sharpness = 0; //m_recommendedDLSSSettings.sharpness;    // <- is this no longer valid?
                dlssOptions.colorBuffersHDR = true;
                dlssOptions.useAutoExposure = false;
                dlssOptions.preset = StreamlineInterface::DLSSPreset::eDefault;
                // if (m_ui.RealtimeAA < 4) <- https://github.com/NVIDIAGameWorks/Streamline/blob/main/docs/ProgrammingGuideDLSS_RR.md#50-provide-dlss--dlss-rr-options seems to imply that these should be set even when DLSS-RR enabled
                    GetDeviceManager()->GetStreamline().SetDLSSOptions(dlssOptions);
            }
            else
            {
                assert( false ); // shouldn't happen, code above should have dropped us to "m_ui.RealtimeAA = 1" - check for recent code changes.
            }

            if (m_ui.RealtimeAA == 2 || m_ui.RealtimeAA == 3)
            {
                // Check if we need to update the rendertarget size.
                bool dlssResizeRequired = (m_ui.DLSSMode != m_ui.DLSSLastMode) || (m_displaySize.x != m_ui.DLSSLastDisplaySize.x) || (m_displaySize.y != m_ui.DLSSLastDisplaySize.y);
                if (dlssResizeRequired)
                {
                    // Only quality, target width and height matter here
                    GetDeviceManager()->GetStreamline().QueryDLSSOptimalSettings(dlssOptions, m_recommendedDLSSSettings);

                    if (dlssOptions.mode == SI::DLSSMode::eBalanced)
                    {
                        m_recommendedDLSSSettings.optimalRenderSize.x = dm::clamp((int)(dlssOptions.outputWidth * 3 / 5 + 0.5f), m_recommendedDLSSSettings.minRenderSize.x, m_recommendedDLSSSettings.maxRenderSize.x);
                        m_recommendedDLSSSettings.optimalRenderSize.y = dm::clamp((int)(dlssOptions.outputHeight * 3 / 5 + 0.5f), m_recommendedDLSSSettings.minRenderSize.y, m_recommendedDLSSSettings.maxRenderSize.y);
                    }

                    if (m_recommendedDLSSSettings.optimalRenderSize.x <= 0 || m_recommendedDLSSSettings.optimalRenderSize.y <= 0)
                    {
                        m_ui.RealtimeAA = 0;
                        m_ui.DLSSMode = SampleUIData::DLSSModeDefault;
                        m_renderSize = m_displaySize;
                    }
                    else
                    {
                        m_ui.DLSSLastMode = m_ui.DLSSMode;
                        m_ui.DLSSLastDisplaySize = m_displaySize;
                    }
                }

                m_renderSize = (uint2)m_recommendedDLSSSettings.optimalRenderSize;
            }

            if (m_ui.RealtimeAA == 3) // DLSS-RR
            {
                StreamlineInterface::DLSSRROptions dlssRROptions = {};
                dlssRROptions.mode              	= dlssOptions.mode;
                dlssRROptions.outputWidth       	= dlssOptions.outputWidth;
                dlssRROptions.outputHeight      	= dlssOptions.outputHeight;
                dlssRROptions.sharpness         	= dlssOptions.sharpness;
                dlssRROptions.preExposure       	= dlssOptions.preExposure;
                dlssRROptions.exposureScale     	= dlssOptions.exposureScale;
                dlssRROptions.colorBuffersHDR   	= dlssOptions.colorBuffersHDR;
                dlssRROptions.indicatorInvertAxisX 	= dlssOptions.indicatorInvertAxisX;
                dlssRROptions.indicatorInvertAxisY 	= dlssOptions.indicatorInvertAxisY;
                dlssRROptions.normalRoughnessMode 	= StreamlineInterface::DLSSRRNormalRoughnessMode::ePacked;
                dlssRROptions.alphaUpscalingEnabled = false;
                dlssRROptions.preset                = StreamlineInterface::DLSSRRPreset::eDefault;
                m_lastDLSSRROptions = dlssRROptions; // we need to fill them up with view info, but we can only have proper view after it was initialized with correct RT size
            }
        }
        else
        {
            if (m_ui.IsDLSSSuported)
            {
                StreamlineInterface::DLSSOptions dlssOptions = {};
                dlssOptions.mode = StreamlineInterface::DLSSMode::eOff;
                GetDeviceManager()->GetStreamline().SetDLSSOptions(dlssOptions);
            }

            m_renderSize = m_displaySize;
        }
    }
#else
    const bool changeToDLSSMode = false;
#endif // #if DONUT_WITH_STREAMLINE
}

void Sample::PreRenderScripts()
{
    if( m_ui.FPSLimiter > 0 )
        g_FPSLimiter.FramerateLimit( m_ui.FPSLimiter );

    korgi::Update();

    if (m_ui.ScreenshotMiniSequenceCounter == -1)
    {
        if (!m_ui.ScreenshotFileName.empty() && m_ui.ScreenshotResetAndDelay) 
        {
            if (m_ui.ScreenshotResetAndDelayCounter == -1) // we just started with delay, set it up
            {
                m_ui.ScreenshotResetAndDelayCounter = m_ui.ScreenshotResetAndDelayFrames+1;
                m_ui.ResetRealtimeCaches = true;
            }
            m_ui.ScreenshotResetAndDelayCounter--;
            assert(m_ui.ScreenshotResetAndDelayCounter >= 0);
        }
    }

}

void Sample::Render(nvrhi::IFramebuffer* framebuffer)
{
    const auto& fbinfo = framebuffer->getFramebufferInfo();
    m_displaySize = m_renderSize = uint2(fbinfo.width, fbinfo.height);
    float lodBias = 0.f;

    if (m_scene == nullptr)
    {
        assert( false ); // TODO: handle separately, just display pink color
        return;
    }

    PreRenderScripts();

    bool needNewPasses = false;
    bool needNewBindings = false;

    StreamlinePreRender();

    m_displayAspectRatio = m_displaySize.x/(float)m_displaySize.y;

    if (m_view == nullptr)
    {
        m_view = std::make_shared<PlanarView>();
        m_viewPrevious = std::make_shared<PlanarView>();
        m_viewPrevious->SetViewport(nvrhi::Viewport(float(m_renderSize.x), float(m_renderSize.y)));
        m_view->SetViewport(nvrhi::Viewport(float(m_renderSize.x), float(m_renderSize.y)));
    }

    if( m_renderTargets == nullptr || m_renderTargets->IsUpdateRequired( m_renderSize, m_displaySize ) )
    {
        m_renderTargets = nullptr;
        m_bindingCache->Clear( );
        m_renderTargets = std::make_unique<RenderTargets>( );
        m_renderTargets->Init(GetDevice( ), m_renderSize, m_displaySize, true, true, c_swapchainCount);
        for (int i = 0; i < std::size(m_nrd); i++)
            m_nrd[i] = nullptr;

        needNewPasses = true;
    }

    // Environment map settings
    if (m_ui.EnvironmentMapParams.Enabled)
    {
        float intensity = m_ui.EnvironmentMapParams.Intensity / c_envMapRadianceScale;
        m_envMapSceneParams.ColorMultiplier = m_ui.EnvironmentMapParams.TintColor * intensity;

        float3 rotationInRadians = donut::math::radians(m_ui.EnvironmentMapParams.RotationXYZ);
        affine3 rotationTransform = donut::math::rotation(rotationInRadians);
        affine3 inverseTransform = inverse(rotationTransform);
        affineToColumnMajor(rotationTransform, m_envMapSceneParams.Transform);
        affineToColumnMajor(inverseTransform, m_envMapSceneParams.InvTransform);
        m_envMapSceneParams.Enabled = 1;
    }
    else
    {
        m_envMapSceneParams.ColorMultiplier = {0,0,0};
        m_envMapSceneParams.Enabled = 0;
    }

    if (m_ui.ShaderReloadRequested)
    {
        m_ui.ShaderReloadRequested = false;
        m_shaderFactory->ClearCache();
        needNewPasses = true;
    }

    bool exposureResetRequired = false;

    if (m_ui.NRDModeChanged)
    {
        needNewPasses = true;
        for (int i = 0; i < std::size(m_nrd); i++)
            m_nrd[i] = nullptr;
    }

    // Acceleration structures need some material info, whilst other passes need acceleration structures, so first set up materials if needed
    if (needNewPasses)
    {
        if (m_materialsBaker == nullptr)
            m_materialsBaker = std::make_shared<MaterialsBaker>(GetDevice(), m_TextureCache, m_shaderFactory);
        m_materialsBaker->CreateRenderPassesAndLoadMaterials(m_bindlessLayout, m_CommonPasses, m_scene, m_currentScenePath, GetLocalPath(g_assetsFolder));
        CollectUncompressedTextures();
        m_ommBaker->CreateRenderPasses(m_bindlessLayout, m_CommonPasses, m_scene);
    }

    // Changes to material properties and settings can require a BLAS/TLAS or subInstanceBuffer rebuild (alpha tested/exclusion flags etc); otherwise this is a no-op.
    UpdateAccelStructs(m_commandList);

    // this will also create or update materials which can trigger the need to update acceleration structures
    if (needNewPasses)
    {
        GetDevice()->waitForIdle();    // some subsystems have resources that could still be in use and might be deleted - make sure that's safe
        m_commandList->open();
        CreateRenderPasses(exposureResetRequired, m_commandList);
        m_commandList->close();
        GetDevice()->executeCommandList(m_commandList);
    }

    m_commandList->open();

    PathTracerCameraData cameraData;
    {
        // Update camera data used by the path tracer & other systems
        UpdateViews(framebuffer);
        {   // TODO: pull all this to BridgeCamera - sizeX and sizeY are already inputs so we just need to pass projMatrix
            nvrhi::Viewport viewport = m_view->GetViewport();
            float2 jitter = m_view->GetPixelOffset();
            float4x4 projMatrix = m_view->GetProjectionMatrix();
            float2 viewSize = { viewport.maxX - viewport.minX, viewport.maxY - viewport.minY };
            float outputAspectRatio = m_displayAspectRatio; //windowViewport.width() / windowViewport.height();    // render and display outputs might not match in case of lower DLSS/etc resolution rounding!
            bool rowMajor = true;
            float tanHalfFOVY = 1.0f / ((rowMajor) ? (projMatrix.m_data[1 * 4 + 1]) : (projMatrix.m_data[1 + 1 * 4]));
            float fovY = atanf(tanHalfFOVY) * 2.0f;
            cameraData = BridgeCamera(uint(viewSize.x), uint(viewSize.y), outputAspectRatio, m_camera.GetPosition(), m_camera.GetDir(), m_camera.GetUp(), fovY, m_cameraZNear, 1e7f, m_ui.CameraFocalDistance, m_ui.CameraAperture, jitter);
        }

        if (needNewPasses || needNewBindings || m_bindingSet == nullptr)
            m_shaderDebug->CreateRenderPasses(framebuffer, m_renderTargets->Depth);

        if (m_ui.EnableShaderDebug)
        {
            dm::float4x4 viewProj = m_view->GetViewProjectionMatrix();
            m_shaderDebug->BeginFrame(m_commandList, viewProj);
        }

        if (m_scene != nullptr)
        {
            m_scene->Refresh(m_commandList, GetFrameIndex());
            m_ommBaker->BuildOpacityMicromaps(m_commandList, m_scene);
            BuildTLAS(m_commandList, GetFrameIndex());
            TransitionMeshBuffersToReadOnly(m_commandList);
            m_ommBaker->Update(m_commandList, m_scene);

            m_materialsBaker->Update(m_commandList, m_scene, m_subInstanceData);
            UploadSubInstanceData(m_commandList); // this is now partial subInstance data, but lights baker update requires it to find materials
        }


        // Update input lighting, environment map, etc.
        PreUpdateLighting(m_commandList, needNewBindings);

        // Early init for RTXDI
        if (needNewPasses || needNewBindings || m_bindingSet == nullptr)
            m_rtxdiPass->Reset();
        RtxdiSetupFrame(framebuffer, cameraData, m_renderSize);
    }

	if( needNewPasses || needNewBindings || m_bindingSet == nullptr )
    {
        RAII_SCOPE( m_commandList->close(); GetDevice()->executeCommandList(m_commandList);, m_commandList->open(););

        // WARNING: this must match the layout of the m_bindingLayout (or switch to CreateBindingSetAndLayout)
        // Fixed resources that do not change between binding sets
        nvrhi::BindingSetDesc bindingSetDescBase;
        bindingSetDescBase.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer),
            nvrhi::BindingSetItem::PushConstants(1, sizeof(SampleMiniConstants)),
            //nvrhi::BindingSetItem::ConstantBuffer(2, m_lightsBaker->GetLightingConstants()),
            nvrhi::BindingSetItem::RayTracingAccelStruct(0, m_topLevelAS),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_subInstanceBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_scene->GetInstanceBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(3, m_scene->GetGeometryBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(4, m_ommBaker->GetGeometryDebugBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(5, m_materialsBaker->GetMaterialDataBuffer()),
            nvrhi::BindingSetItem::Texture_SRV(10, m_envMapBaker->GetEnvMapCube()), //m_EnvironmentMap->IsEnvMapLoaded() ? m_EnvironmentMap->GetEnvironmentMap() : m_CommonPasses->m_BlackTexture),
            nvrhi::BindingSetItem::Texture_SRV(11, m_envMapBaker->GetImportanceSampling()->GetImportanceMap()), //m_EnvironmentMap->IsImportanceMapLoaded() ? m_EnvironmentMap->GetImportanceMap() : m_CommonPasses->m_BlackTexture),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(12, m_lightsBaker->GetControlBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(13, m_lightsBaker->GetLightBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(14, m_lightsBaker->GetLightExBuffer()),
            nvrhi::BindingSetItem::TypedBuffer_SRV(15, m_lightsBaker->GetLightProxyCounters()),    // t_tightProxyCounters
            nvrhi::BindingSetItem::TypedBuffer_SRV(16, m_lightsBaker->GetLightSamplingProxies()),  // t_LightProxyIndices
            nvrhi::BindingSetItem::Texture_SRV(17, m_lightsBaker->GetNarrowSamplingBuffer()),      // t_LightProxyKeys
            nvrhi::BindingSetItem::Texture_SRV(18, m_lightsBaker->GetEnvLightLookupMap()),         // t_EnvLookupMap
            //nvrhi::BindingSetItem::TypedBuffer_SRV(19, ),
            nvrhi::BindingSetItem::Texture_UAV(10, m_lightsBaker->GetFeedbackReservoirBuffer()),   // u_LightFeedbackBuffer

#if USE_PRECOMPUTED_SOBOL_BUFFER
            nvrhi::BindingSetItem::TypedBuffer_SRV(42, m_precomputedSobolBuffer),
#endif
            nvrhi::BindingSetItem::Sampler(0, m_CommonPasses->m_AnisotropicWrapSampler),
            nvrhi::BindingSetItem::Sampler(1, m_envMapBaker->GetEnvMapCubeSampler()),
            nvrhi::BindingSetItem::Sampler(2, m_envMapBaker->GetImportanceSampling()->GetImportanceMapSampler()),
            nvrhi::BindingSetItem::Texture_UAV(0, m_renderTargets->OutputColor),
            nvrhi::BindingSetItem::Texture_UAV(4, m_renderTargets->Throughput),
            nvrhi::BindingSetItem::Texture_UAV(5, m_renderTargets->ScreenMotionVectors),
            nvrhi::BindingSetItem::Texture_UAV(6, m_renderTargets->Depth),
            nvrhi::BindingSetItem::Texture_UAV(31, m_renderTargets->DenoiserViewspaceZ),
            nvrhi::BindingSetItem::Texture_UAV(32, m_renderTargets->DenoiserMotionVectors),
            nvrhi::BindingSetItem::Texture_UAV(33, m_renderTargets->DenoiserNormalRoughness),
            nvrhi::BindingSetItem::Texture_UAV(34, m_renderTargets->DenoiserDiffRadianceHitDist),
            nvrhi::BindingSetItem::Texture_UAV(35, m_renderTargets->DenoiserSpecRadianceHitDist),
            nvrhi::BindingSetItem::Texture_UAV(36, m_renderTargets->DenoiserDisocclusionThresholdMix),
            nvrhi::BindingSetItem::Texture_UAV(37, m_renderTargets->CombinedHistoryClampRelax),
            nvrhi::BindingSetItem::Texture_UAV(50, m_shaderDebug->GetDebugVizTexture()),    // todo: move to SHADER_DEBUG_VIZ_TEXTURE_UAV_INDEX
            nvrhi::BindingSetItem::StructuredBuffer_UAV(51, m_feedback_Buffer_Gpu),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(52, m_debugLineBufferCapture),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(53, m_debugDeltaPathTree_Gpu),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(54, m_debugDeltaPathTreeSearchStack),
            nvrhi::BindingSetItem::Texture_UAV(60, m_renderTargets->SecondarySurfacePositionNormal),
            nvrhi::BindingSetItem::Texture_UAV(61, m_renderTargets->SecondarySurfaceRadiance),
            nvrhi::BindingSetItem::Texture_UAV(70, m_renderTargets->RRDiffuseAlbedo),
            nvrhi::BindingSetItem::Texture_UAV(71, m_renderTargets->RRSpecAlbedo),
            nvrhi::BindingSetItem::Texture_UAV(72, m_renderTargets->RRNormalsAndRoughness),
            nvrhi::BindingSetItem::Texture_UAV(73, m_renderTargets->RRSpecMotionVectors),
            nvrhi::BindingSetItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX, m_shaderDebug->GetGPUWriteBuffer())

#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
            // Stochastic texture filtering blue noise texture
            , nvrhi::BindingSetItem::Texture_SRV(63, m_STBNTexture->texture),                                                 // t_STBN2DTexture
#endif
        };

        // NVAPI shader extension UAV is only applicable on DX12
        if (GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
        {
            bindingSetDescBase.bindings.push_back(
                nvrhi::BindingSetItem::TypedBuffer_UAV(NV_SHADER_EXTN_SLOT_NUM, nullptr));
        }

        // Main raytracing & etc binding set
		{
            nvrhi::BindingSetDesc bindingSetDesc;

            bindingSetDesc.bindings = bindingSetDescBase.bindings;

            bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_UAV(40, m_renderTargets->StablePlanesHeader));
            bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::StructuredBuffer_UAV(42, m_renderTargets->StablePlanesBuffer));
            bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_UAV(44, m_renderTargets->StableRadiance));
            bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::StructuredBuffer_UAV(45, m_renderTargets->SurfaceDataBuffer));

            m_bindingSet = GetDevice()->createBindingSet(bindingSetDesc, m_bindingLayout);
        }

        {
            nvrhi::BindingSetDesc lineBindingSetDesc;
            lineBindingSetDesc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer),
                nvrhi::BindingSetItem::Texture_SRV(0, m_renderTargets->Depth)
            };
            m_linesBindingSet = GetDevice()->createBindingSet(lineBindingSetDesc, m_linesBindingLayout);

            nvrhi::GraphicsPipelineDesc psoDesc;
            psoDesc.VS = m_linesVertexShader;
            psoDesc.PS = m_linesPixelShader;
            psoDesc.inputLayout = m_linesInputLayout;
            psoDesc.bindingLayouts = { m_linesBindingLayout };
            psoDesc.primType = nvrhi::PrimitiveType::LineList;
            psoDesc.renderState.depthStencilState.depthTestEnable = false;
            psoDesc.renderState.blendState.targets[0].enableBlend().setSrcBlend(nvrhi::BlendFactor::SrcAlpha)
                .setDestBlend(nvrhi::BlendFactor::InvSrcAlpha).setSrcBlendAlpha(nvrhi::BlendFactor::Zero).setDestBlendAlpha(nvrhi::BlendFactor::One);

            m_linesPipeline = GetDevice()->createGraphicsPipeline(psoDesc, framebuffer);

            // nvrhi::ComputePipelineDesc psoCSDesc;
            // psoCSDesc.bindingLayouts = { m_bindingLayout };
            // psoCSDesc.CS = m_linesAddExtraComputeShader;
            // m_linesAddExtraPipeline = GetDevice()->createComputePipeline(psoCSDesc);
        }
    }

    if (m_ui.EnableToneMapping)
        m_toneMappingPass->PreRender(m_ui.ToneMappingParams);

    PreUpdatePathTracing(needNewPasses, m_commandList);

    // I suppose we need to clear depth for right-click picking at least
    m_renderTargets->Clear( m_commandList );

    SampleConstants & constants = m_currentConstants; memset(&constants, 0, sizeof(constants));
    SampleMiniConstants miniConstants = { uint4(0, 0, 0, 0) }; // accessible but unused in path tracing at the moment
    if( m_scene == nullptr )
    {
        m_commandList->clearTextureFloat( m_renderTargets->OutputColor, nvrhi::AllSubresources, nvrhi::Color( 1, 1, 0, 0 ) );
        m_commandList->writeBuffer(m_constantBuffer, &constants, sizeof(constants));
    }
    else
    {
        UpdatePathTracerConstants(constants.ptConsts, cameraData);
        constants.MaterialCount = m_materialsBaker->GetMaterialDataCount(); // m_scene->GetSceneGraph()->GetMaterials().size();
        constants._padding1 = 0;
        constants._padding2 = 0;

        constants.envMapSceneParams = m_envMapSceneParams;
        constants.envMapImportanceSamplingParams = m_envMapBaker->GetImportanceSampling()->GetShaderParams();

        m_view->FillPlanarViewConstants(constants.view);
        m_viewPrevious->FillPlanarViewConstants(constants.previousView);

        constants.debug = {};
        constants.debug.pick = m_pick || m_ui.ContinuousDebugFeedback;
        constants.debug.pickX = (constants.debug.pick)?(m_ui.DebugPixel.x):(-1);
        constants.debug.pickY = (constants.debug.pick)?(m_ui.DebugPixel.y):(-1);
        constants.debug.debugLineScale = (m_ui.ShowDebugLines)?(m_ui.DebugLineScale):(0.0f);
        constants.debug.showWireframe = m_ui.ShowWireframe;
        constants.debug.debugViewType = (int)m_ui.DebugView;
        constants.debug.debugViewStablePlaneIndex = (m_ui.StablePlanesActiveCount==1)?(0):(m_ui.DebugViewStablePlaneIndex);
#if ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
        constants.debug.exploreDeltaTree = (m_ui.ShowDeltaTree && constants.debug.pick)?(1):(0);
#else
        constants.debug.exploreDeltaTree = false;
#endif
        constants.debug.imageWidth = constants.ptConsts.imageWidth;
        constants.debug.imageHeight = constants.ptConsts.imageHeight;
        constants.debug.mouseX = m_ui.MousePos.x;
        constants.debug.mouseY = m_ui.MousePos.y;
        constants.debug.cameraPosW = constants.ptConsts.camera.PosW;
        constants.debug._padding0 = 0;

        constants.denoisingHitParamConsts = { m_ui.ReblurSettings.hitDistanceParameters.A, m_ui.ReblurSettings.hitDistanceParameters.B,
                                              m_ui.ReblurSettings.hitDistanceParameters.C, m_ui.ReblurSettings.hitDistanceParameters.D };

        // This updates all lighting: distant (environment maps and directional analytic lights) and local (analytic lights and emissive triangle lights)
        // Must go before m_constantBuffer as when saving screenshots it closes and re-opens command list, flushing the volatile constant buffer!
        UpdateLighting(m_commandList);

        UploadSubInstanceData(m_commandList);

        m_commandList->writeBuffer(m_constantBuffer, &constants, sizeof(constants));

		if (m_ui.ActualUseRTXDIPasses())
            m_rtxdiPass->BeginFrame(m_commandList, *m_renderTargets, m_bindingLayout, m_bindingSet);

        PathTrace(framebuffer, constants);

        Denoise(framebuffer);

        PostProcessAA(framebuffer, needNewPasses || m_ui.ResetRealtimeCaches);
    }

    nvrhi::ITexture* finalColor = m_ui.RealtimeMode ? m_renderTargets->ProcessedOutputColor : m_renderTargets->AccumulatedRadiance;

    //Tone Mapping
    if (m_ui.EnableToneMapping)
    {
        donut::engine::PlanarView fullscreenView = *m_view;
        nvrhi::Viewport windowViewport(float(m_displaySize.x), float(m_displaySize.y));
        fullscreenView.SetViewport(windowViewport);
        fullscreenView.UpdateCache();

        if (m_toneMappingPass->Render(m_commandList, fullscreenView, finalColor))
        {
            // first run tonemapper can close command list - we have to re-upload volatile constants then
            m_commandList->writeBuffer(m_constantBuffer, &constants, sizeof(constants));
        }

        finalColor = m_renderTargets->LdrColor;
    }

    //m_postProcess->Render(m_commandList, finalColor);

    m_commandList->beginMarker("Blit");
    m_CommonPasses->BlitTexture(m_commandList, framebuffer, finalColor, m_bindingCache.get());
    m_commandList->endMarker();

    if (m_ui.ShowDebugLines == true)
    {
        m_commandList->beginMarker("Debug Lines");

        // // this copies over additional (CPU written) lines!
        // {
        //     nvrhi::ComputeState state;
        //     state.bindings = { m_bindingSet };
        //     state.pipeline = m_linesAddExtraPipeline;
        //     m_commandList->setComputeState(state);
        //     const dm::uint  threads = 256;
        //     const dm::uint2 dispatchSize = dm::uint2(1, 1);
        //     m_commandList->dispatch(dispatchSize.x, dispatchSize.y);
        // }

        // this draws the debug lines - should be the only actual rasterization around :)
        {
            nvrhi::GraphicsState state;
            state.bindings = { m_linesBindingSet };
            state.vertexBuffers = { {m_debugLineBufferDisplay, 0, 0} };
            state.pipeline = m_linesPipeline;
            state.framebuffer = framebuffer;
            state.viewport.addViewportAndScissorRect(fbinfo.getViewport());

            m_commandList->setGraphicsState(state);

            nvrhi::DrawArguments args;
            args.vertexCount = m_feedbackData.lineVertexCount;
            m_commandList->draw(args);
        }

        if (m_cpuSideDebugLines.size() > 0)
        {
            // using m_debugLineBufferCapture for direct drawing here
            m_commandList->writeBuffer( m_debugLineBufferCapture, m_cpuSideDebugLines.data(), sizeof(DebugLineStruct) * m_cpuSideDebugLines.size() );

            nvrhi::GraphicsState state;
            state.bindings = { m_linesBindingSet };
            state.vertexBuffers = { {m_debugLineBufferCapture, 0, 0} };
            state.pipeline = m_linesPipeline;
            state.framebuffer = framebuffer;
            state.viewport.addViewportAndScissorRect(fbinfo.getViewport());

            m_commandList->setGraphicsState(state);

            nvrhi::DrawArguments args;
            args.vertexCount = (uint32_t)m_cpuSideDebugLines.size();
            m_commandList->draw(args);
        }

        m_commandList->endMarker();
    }
    m_cpuSideDebugLines.clear();

    if( m_ui.EnableShaderDebug )
        m_shaderDebug->EndFrameAndOutput(m_commandList, framebuffer, m_renderTargets->Depth, fbinfo.getViewport() );

    if( m_ui.ContinuousDebugFeedback || m_pick )
    {
        m_commandList->copyBuffer(m_feedback_Buffer_Cpu, 0, m_feedback_Buffer_Gpu, 0, sizeof(DebugFeedbackStruct) * 1);
        m_commandList->copyBuffer(m_debugLineBufferDisplay, 0, m_debugLineBufferCapture, 0, sizeof(DebugLineStruct) * MAX_DEBUG_LINES );
        m_commandList->copyBuffer(m_debugDeltaPathTree_Cpu, 0, m_debugDeltaPathTree_Gpu, 0, sizeof(DeltaTreeVizPathVertex) * cDeltaTreeVizMaxVertices);
	}

    nvrhi::ITexture* framebufferTexture = framebuffer->getDesc().colorAttachments[0].texture;
    //m_commandList->copyTexture(m_renderTargets->PreUIColor, nvrhi::TextureSlice(), framebufferTexture, nvrhi::TextureSlice());


	m_commandList->close();
	GetDevice()->executeCommandList(m_commandList);

    // resolve right click picking and debug info
    if (m_ui.ContinuousDebugFeedback || m_pick)
    {
        GetDevice()->waitForIdle();
        void* pData = GetDevice()->mapBuffer(m_feedback_Buffer_Cpu, nvrhi::CpuAccessMode::Read);
        assert(pData);
        memcpy(&m_feedbackData, pData, sizeof(DebugFeedbackStruct)* 1);
        GetDevice()->unmapBuffer(m_feedback_Buffer_Cpu);

        pData = GetDevice()->mapBuffer(m_debugDeltaPathTree_Cpu, nvrhi::CpuAccessMode::Read);
        assert(pData);
        memcpy(&m_debugDeltaPathTree, pData, sizeof(DeltaTreeVizPathVertex) * cDeltaTreeVizMaxVertices);
        GetDevice()->unmapBuffer(m_debugDeltaPathTree_Cpu);

        if (m_pick)
            m_ui.SelectedMaterial = FindMaterial(int(m_feedbackData.pickedMaterialID));


        m_pick = false;
    }

    auto DumpScreenshot = [this](nvrhi::ITexture* framebufferTexture, const char* fileName, bool exitOnCompletion)
    {
        const bool success = SaveTextureToFile(GetDevice(), m_CommonPasses.get(), framebufferTexture, nvrhi::ResourceStates::Common, fileName);

        if (exitOnCompletion)
        {
            if (success)
            {
                donut::log::info("Image saved successfully %s. Exiting.", fileName);
                std::exit(0);
            }
            else
            {
                donut::log::fatal("Unable to save image %s. Exiting.", fileName);
                std::exit(1);
            }
        }
    };

    if (!m_ui.ScreenshotFileName.empty() && !(m_ui.ScreenshotResetAndDelay && m_ui.ScreenshotResetAndDelayCounter>0) )
    {
        std::filesystem::path screenshotFile = m_ui.ScreenshotFileName;
        if (m_ui.ScreenshotMiniSequence)
        {
            if ( m_ui.ScreenshotMiniSequenceCounter == -1 ) // start sequence if in sequence mode
                m_ui.ScreenshotMiniSequenceCounter = m_ui.ScreenshotMiniSequenceFrames;
            assert( m_ui.ScreenshotMiniSequenceFrames > 0 );
            m_ui.ScreenshotMiniSequenceCounter--;

            std::filesystem::path justName = screenshotFile.filename().stem();
            std::filesystem::path justExtension = screenshotFile.extension();
            screenshotFile.remove_filename();
            screenshotFile /= justName.string() + string_format("_%03d", m_ui.ScreenshotMiniSequenceFrames - m_ui.ScreenshotMiniSequenceCounter) + justExtension.string();
        }

        DumpScreenshot(framebufferTexture, screenshotFile.string().c_str(), false /*exitOnCompletion*/);

        if (!m_ui.ScreenshotMiniSequence || m_ui.ScreenshotMiniSequenceCounter==0)
        {
            m_ui.ScreenshotFileName = "";
            m_ui.ScreenshotResetAndDelayCounter = -1;
            m_ui.ScreenshotMiniSequenceCounter = -1;
        }
    }

	if (!m_cmdLine.screenshotFileName.empty() && (m_cmdLine.screenshotFrameIndex == GetFrameIndex()))
	{
        DumpScreenshot(framebufferTexture, m_cmdLine.screenshotFileName.c_str(), true /*exitOnCompletion*/);
	}

    if (m_ui.ExperimentalPhotoModeScreenshot)
    {
        DenoisedScreenshot( framebufferTexture );
        m_ui.ExperimentalPhotoModeScreenshot = false;
    }

    if (m_temporalAntiAliasingPass != nullptr)
        m_temporalAntiAliasingPass->AdvanceFrame();

	std::swap(m_view, m_viewPrevious);
	GetDeviceManager()->SetVsyncEnabled(m_ui.EnableVsync);

    PostUpdatePathTracing();
}

std::shared_ptr<donut::engine::Material> Sample::FindMaterial(int materialID) const
{
    // if slow switch to map
    for (const auto& material : m_scene->GetSceneGraph()->GetMaterials())
        if (material->materialID == materialID)
            return material;
    return nullptr;
}

void Sample::PathTrace(nvrhi::IFramebuffer* framebuffer, const SampleConstants & constants)
{
    //m_commandList->beginMarker("MainRendering"); <- removed (for now) since added hierarchy reduces readability

    bool useStablePlanes = m_ui.ActualUseStablePlanes();

    nvrhi::rt::State state;

    nvrhi::rt::DispatchRaysArguments args;
    nvrhi::Viewport viewport = m_view->GetViewport();
    uint32_t width = (uint32_t)(viewport.maxX - viewport.minX);
    uint32_t height = (uint32_t)(viewport.maxY - viewport.minY);
    args.width = width;
    args.height = height;

    uint version;
    uint versionBase = (m_ui.DXRHitObjectExtension)?(3):(0);    // HitObjectExtension-enabled permutations are offset by 3 - see CreatePTPipeline; this will possibly go away once part of API (it can be dynamic)

    // default miniConstants
    SampleMiniConstants miniConstants = { uint4(0, 0, 0, 0) };

    if (useStablePlanes)
    {
        m_commandList->beginMarker("PathTracePrePass");
        int version = versionBase+PATH_TRACER_MODE_BUILD_STABLE_PLANES;
        state.shaderTable = m_PTShaderTable[version];
        state.bindings = { m_bindingSet, m_DescriptorTable->GetDescriptorTable() };
        m_commandList->setRayTracingState(state);
        m_commandList->setPushConstants(&miniConstants, sizeof(miniConstants));
        m_commandList->dispatchRays(args);
        m_commandList->endMarker();

        m_commandList->setBufferState(m_renderTargets->StablePlanesBuffer, nvrhi::ResourceStates::UnorderedAccess);
        m_commandList->commitBarriers();

        m_commandList->beginMarker("VBufferExport");
		nvrhi::ComputeState state;
		state.bindings = { m_bindingSet, m_DescriptorTable->GetDescriptorTable() };
        state.pipeline = m_exportVBufferPSO;
        m_commandList->setComputeState(state);

		const dm::uint2 dispatchSize = { (width + NUM_COMPUTE_THREADS_PER_DIM - 1) / NUM_COMPUTE_THREADS_PER_DIM, (height + NUM_COMPUTE_THREADS_PER_DIM - 1) / NUM_COMPUTE_THREADS_PER_DIM };
        m_commandList->setPushConstants(&miniConstants, sizeof(miniConstants));
		m_commandList->dispatch(dispatchSize.x, dispatchSize.y);
		m_commandList->endMarker();
    }

    {
        RAII_SCOPE( m_commandList->beginMarker("UpdateLighting");, m_commandList->endMarker(); );
        m_lightsBaker->UpdateLate(m_commandList, m_scene, m_materialsBaker, m_ommBaker, m_subInstanceBuffer, m_renderTargets->Depth, m_renderTargets->ScreenMotionVectors);  // <- in the future this will provide motion vectors except in case of reference mode
    }

    version = (useStablePlanes ? PATH_TRACER_MODE_FILL_STABLE_PLANES : PATH_TRACER_MODE_REFERENCE) + versionBase;

    {
        m_commandList->beginMarker("PathTrace");

        for (uint subSampleIndex = 0; subSampleIndex < m_ui.ActualSamplesPerPixel(); subSampleIndex++)
        {
            //m_commandList->beginMarker("PTSubPass");
            state.shaderTable = m_PTShaderTable[version];
            state.bindings = { m_bindingSet, m_DescriptorTable->GetDescriptorTable() };
            m_commandList->setRayTracingState(state);

            // required to avoid race conditions in back to back dispatchRays
            m_commandList->setBufferState(m_renderTargets->StablePlanesBuffer, nvrhi::ResourceStates::UnorderedAccess);
            m_commandList->commitBarriers();

            // tell path tracer which subSampleIndex we're processing
            SampleMiniConstants miniConstants = { uint4(subSampleIndex, 0, 0, 0) };
            m_commandList->setPushConstants(&miniConstants, sizeof(miniConstants));
            m_commandList->dispatchRays(args);
            //m_commandList->endMarker();
        }

        m_commandList->endMarker();

        m_commandList->setBufferState(m_renderTargets->StablePlanesBuffer, nvrhi::ResourceStates::UnorderedAccess);
        m_commandList->commitBarriers();
    }

    // this is a performance optimization where final 2 passes from ReSTIR DI and ReSTIR GI are combined to avoid loading GBuffer twice
    static bool enableFusedDIGIFinal = true;
    bool useFusedDIGIFinal = m_ui.ActualUseReSTIRDI() && m_ui.ActualUseReSTIRGI() && enableFusedDIGIFinal;

    if (m_ui.ActualUseRTXDIPasses())
    {
        RAII_SCOPE( m_commandList->beginMarker("RTXDI");, m_commandList->endMarker(); );

        // this does all ReSTIR DI magic including applying the final sample into correct radiance buffer (depending on denoiser state)
        if (m_ui.ActualUseReSTIRDI())
            m_rtxdiPass->Execute(m_commandList, m_bindingSet, useFusedDIGIFinal);

        if (m_ui.ActualUseReSTIRGI())
            m_rtxdiPass->ExecuteGI(m_commandList, m_bindingSet, useFusedDIGIFinal);

        if (useFusedDIGIFinal)
            m_rtxdiPass->ExecuteFusedDIGIFinal(m_commandList, m_bindingSet);
    }

    if (useStablePlanes && (m_ui.DebugView >= DebugViewType::ImagePlaneRayLength && m_ui.DebugView <= DebugViewType::StablePlaneSpecHitDist || m_ui.DebugView == DebugViewType::StableRadiance) )
    {
        m_commandList->beginMarker("StablePlanesDebugViz");
        nvrhi::TextureDesc tdesc = m_renderTargets->OutputColor->getDesc();
        m_postProcess->Apply(m_commandList, PostProcess::ComputePassType::StablePlanesDebugViz, m_constantBuffer, miniConstants, m_bindingSet, m_bindingLayout, tdesc.width, tdesc.height);
        m_commandList->endMarker();

    }

    //m_commandList->endMarker(); // "MainRendering"
}

void Sample::Denoise(nvrhi::IFramebuffer* framebuffer)
{
    if( !m_ui.ActualUseStandaloneDenoiser() )
        return;

    //const auto& fbinfo = framebuffer->getFramebufferInfo();
    const char* passNames[] = { "Denoising plane 0", "Denoising plane 1", "Denoising plane 2", "Denoising plane 3" }; assert( std::size(m_nrd) <= std::size(passNames) );

    bool nrdUseRelax = m_ui.NRDMethod == NrdConfig::DenoiserMethod::RELAX;
    PostProcess::ComputePassType preparePassType = nrdUseRelax ? PostProcess::ComputePassType::RELAXDenoiserPrepareInputs : PostProcess::ComputePassType::REBLURDenoiserPrepareInputs;
    PostProcess::ComputePassType mergePassType = nrdUseRelax ? PostProcess::ComputePassType::RELAXDenoiserFinalMerge : PostProcess::ComputePassType::REBLURDenoiserFinalMerge;

    bool resetHistory = m_ui.ResetRealtimeCaches;

    int maxPassCount = std::min(m_ui.StablePlanesActiveCount, (int)std::size(m_nrd));
    for (int pass = maxPassCount-1; pass >= 0; pass--)
    {
        m_commandList->beginMarker(passNames[pass]);

        SampleMiniConstants miniConstants = { uint4((uint)pass, 0, 0, 0) };

        // Direct inputs to denoiser are reused between passes; there's redundant copies but it makes interfacing simpler
        nvrhi::TextureDesc tdesc = m_renderTargets->OutputColor->getDesc();
        m_commandList->beginMarker("PrepareInputs");
        m_postProcess->Apply(m_commandList, preparePassType, m_constantBuffer, miniConstants, m_bindingSet, m_bindingLayout, tdesc.width, tdesc.height);
        m_commandList->endMarker();

        const float timeDeltaBetweenFrames = m_cmdLine.noWindow ? 1.f/60.f : -1.f; // if we're rendering without a window we set a fix timeDeltaBetweenFrames to ensure that output is deterministic
        bool enableValidation = m_ui.DebugView == DebugViewType::StablePlaneDenoiserValidation;
        if (nrdUseRelax)
        {
            m_nrd[pass]->RunDenoiserPasses(m_commandList, *m_renderTargets, pass, *m_view, *m_viewPrevious, GetFrameIndex(), m_ui.NRDDisocclusionThreshold, m_ui.NRDDisocclusionThresholdAlternate, m_ui.NRDUseAlternateDisocclusionThresholdMix, timeDeltaBetweenFrames, enableValidation, resetHistory, &m_ui.RelaxSettings);
        }
        else
        {
            m_nrd[pass]->RunDenoiserPasses(m_commandList, *m_renderTargets, pass, *m_view, *m_viewPrevious, GetFrameIndex(), m_ui.NRDDisocclusionThreshold, m_ui.NRDDisocclusionThresholdAlternate, m_ui.NRDUseAlternateDisocclusionThresholdMix, timeDeltaBetweenFrames, enableValidation, resetHistory, &m_ui.ReblurSettings);
        }

        m_commandList->beginMarker("MergeOutputs");
        m_postProcess->Apply(m_commandList, mergePassType, pass, m_constantBuffer, miniConstants, m_renderTargets->OutputColor, *m_renderTargets, nullptr);
        m_commandList->endMarker();

        m_commandList->endMarker();
    }
}

void Sample::PostProcessAA(nvrhi::IFramebuffer* framebuffer, bool reset)
{
    if (m_ui.RealtimeMode)
    {
        if (m_ui.RealtimeAA == 0)
        {
            // TODO: Remove Redundant copy for non AA case
            m_commandList->copyTexture(m_renderTargets->ProcessedOutputColor, nvrhi::TextureSlice(), m_renderTargets->OutputColor, nvrhi::TextureSlice());
        }
        else if (m_ui.RealtimeAA == 1 && m_temporalAntiAliasingPass != nullptr )
        {
            bool previousViewValid = (GetFrameIndex() != 0);

            m_commandList->beginMarker("TAA");

            m_temporalAntiAliasingPass->TemporalResolve(m_commandList, m_ui.TemporalAntiAliasingParams, previousViewValid, *m_view, *m_view);

            m_commandList->endMarker();
        }

#if DONUT_WITH_STREAMLINE
        // SET STREAMLINE CONSTANTS
        {
            // This section of code updates the streamline constants every frame. Regardless of whether we are utilising the streamline plugins, as long as streamline is in use, we must set its constants.
            affine3 viewReprojection = m_view->GetChildView(ViewType::PLANAR, 0)->GetInverseViewMatrix() * m_viewPrevious->GetViewMatrix();
            float4x4 reprojectionMatrix = inverse(m_view->GetProjectionMatrix(false)) * affineToHomogeneous(viewReprojection) * m_viewPrevious->GetProjectionMatrix(false);
            float outputAspectRatio = m_displayAspectRatio; //windowViewport.width() / windowViewport.height();    // render and display outputs might not match in case of lower DLSS/etc resolution rounding!
            float4x4 projection = perspProjD3DStyleReverse(dm::radians(m_cameraVerticalFOV), outputAspectRatio, m_cameraZNear);

            //float2 jitterOffset = std::dynamic_pointer_cast<PlanarView, IView>(m_view)->GetPixelOffset();
            float2 jitterOffset = ComputeCameraJitter(m_sampleIndex);

            StreamlineInterface::Constants slConstants = {};
            slConstants.cameraAspectRatio = outputAspectRatio;
            slConstants.cameraFOV = dm::radians(m_cameraVerticalFOV);
            slConstants.cameraFar = m_cameraZFar;
            slConstants.cameraMotionIncluded = true;
            slConstants.cameraNear = m_cameraZNear;
            slConstants.cameraPinholeOffset = { 0.f, 0.f };
            slConstants.cameraPos = m_camera.GetPosition();
            slConstants.cameraFwd = m_camera.GetDir();
            slConstants.cameraUp = m_camera.GetUp();
            slConstants.cameraRight = normalize(cross(m_camera.GetDir(), m_camera.GetUp()));
            slConstants.cameraViewToClip = projection;
            slConstants.clipToCameraView = inverse(projection);
            slConstants.clipToPrevClip = reprojectionMatrix;
            slConstants.depthInverted = m_view->IsReverseDepth() ? true : false;
            slConstants.jitterOffset = jitterOffset;
            slConstants.mvecScale = { 1.0f / m_renderSize.x , 1.0f / m_renderSize.y }; // This are scale factors used to normalize mvec (to -1,1) and donut has mvec in pixel space
            slConstants.prevClipToClip = inverse(reprojectionMatrix);
            slConstants.reset = reset;
            slConstants.motionVectors3D = false;
            slConstants.motionVectorsInvalidValue = FLT_MIN;

            GetDeviceManager()->GetStreamline().SetConstants(slConstants);

            if (m_ui.RealtimeAA == 3) // DLSS-RR
            {
                // only used if hitT (hit distance) codepath is used
                m_lastDLSSRROptions.worldToCameraView = dm::affineToHomogeneous(m_view->GetViewMatrix());
                m_lastDLSSRROptions.cameraViewToWorld = dm::affineToHomogeneous(m_view->GetInverseViewMatrix());
                
                GetDeviceManager()->GetStreamline().SetDLSSRROptions(m_lastDLSSRROptions);
            }
        }

        m_commandList->commitBarriers();

        // TAG STREAMLINE RESOURCES
        GetDeviceManager()->GetStreamline().TagResourcesGeneral(m_commandList,
            m_view->GetChildView(ViewType::PLANAR, 0),
            m_renderTargets->ScreenMotionVectors,
            m_renderTargets->Depth,
            m_renderTargets->PreUIColor);

        // TAG STREAMLINE RESOURCES
        GetDeviceManager()->GetStreamline().TagResourcesDLSSNIS(m_commandList,
            m_view->GetChildView(ViewType::PLANAR, 0),
            m_renderTargets->ProcessedOutputColor,
            m_renderTargets->OutputColor);

        if (m_ui.RealtimeAA == 2)
        {
            GetDeviceManager()->GetStreamline().EvaluateDLSS(m_commandList);
            m_commandList->clearState();
        }
        if (m_ui.RealtimeAA == 3)
        {
            // Direct inputs to denoiser are reused between passes; there's redundant copies but it makes interfacing simpler
            SampleMiniConstants miniConstants = { uint4(0, 0, 0, 0) };
            nvrhi::TextureDesc tdesc = m_renderTargets->OutputColor->getDesc();
            m_commandList->beginMarker("DLSSRR_PrepareInputs");
            m_postProcess->Apply(m_commandList, PostProcess::ComputePassType::DLSSRRDenoiserPrepareInputs, m_constantBuffer, miniConstants, m_bindingSet, m_bindingLayout, tdesc.width, tdesc.height);
            m_commandList->endMarker();

            GetDeviceManager()->GetStreamline().TagResourcesDLSSRR(m_commandList,
                m_view->GetChildView(ViewType::PLANAR, 0),
                (int2)m_renderSize,
                (int2)m_displaySize,
                m_renderTargets->OutputColor, // nvrhi::ITexture* inputColor,
                m_renderTargets->RRDiffuseAlbedo,
                m_renderTargets->RRSpecAlbedo,
                m_renderTargets->RRNormalsAndRoughness,
                nullptr,
                nullptr,
                m_renderTargets->RRSpecMotionVectors,
                m_renderTargets->ProcessedOutputColor //nvrhi::ITexture* outputColor

            );

            m_commandList->commitBarriers();

            GetDeviceManager()->GetStreamline().EvaluateDLSSRR(m_commandList);
            m_commandList->clearState();
        }
#endif
    }
    else if (m_accumulationSampleIndex < m_accumulationSampleTarget)
    {
        // Reference mode - run the accumulation pass.
        // Don't run it when the sample count has reached the target, just keep the previous output.
        // Otherwise, the frames that are rendered past the target all have the same RNG sequence,
        // and the output starts to converge to that single sample.

        const float accumulationWeight = 1.f / float(m_accumulationSampleIndex + 1);

        m_accumulationPass->Render(m_commandList, *m_view, *m_view, accumulationWeight);
    }

}

bool Sample::CompressTextures()
{
    // if async needed, do something like std::thread([sytemCommand](){ system( sytemCommand.c_str() ); }).detach();

    std::string batchFileName = std::string(getenv("localappdata")) + "\\temp\\donut_compressor.bat";
    std::ofstream batchFile(batchFileName, std::ios_base::trunc);
    if (!batchFile.is_open())
    {
        log::message(log::Severity::Error, "Unable to write %s", batchFileName.c_str());
        return false;
    }

    std::string cmdLine;

    // prefix part
    //cmdLine += "echo off \n";
    cmdLine += "ECHO: \n";
    cmdLine += "WHERE nvtt_export \n";
    //cmdLine += "ECHO WHERE nvtt_export returns %ERRORLEVEL% \n";
    cmdLine += "IF %ERRORLEVEL% NEQ 0 (goto :error_tool)\n";
    cmdLine += "ECHO: \n";
    cmdLine += "ECHO nvtt_export exists in the Path, proceeding with compression (this might take a while!) \n";
    cmdLine += "ECHO: \n";

    uint i = 0; uint totalCount = (uint)m_uncompressedTextures.size();
    for (auto it : m_uncompressedTextures)
    {
        auto texture = it.first;
        std::string inPath = texture->path;
        std::string outPath = std::filesystem::path(inPath).replace_extension(".dds").string();

        cmdLine += "ECHO converting texture " + std::to_string(++i) + " " + " out of " + std::to_string( totalCount ) + "\n";

        cmdLine += "nvtt_export";
        cmdLine += " -f 23"; // this sets format BC7
        cmdLine += " ";

        if( it.second == TextureCompressionType::Normalmap )
        {
            // cmdLine += " --normal-filter 1";
            // cmdLine += " --normalize";
            cmdLine += " --no-mip-gamma-correct";
        }
        else if (it.second == TextureCompressionType::GenericLinear)
        {
            cmdLine += " --no-mip-gamma-correct";
        }
        else if (it.second == TextureCompressionType::GenericSRGB)
        {
            cmdLine += " --mip-gamma-correct";
        }
        // cmdLine += " -q 2";  // 2 is production quality, 1 is "normal" (default)

        cmdLine += " -o \"" + outPath;
        cmdLine += "\" \"" + inPath + "\"\n";
    }
    cmdLine += "ECHO:\n";
    cmdLine += "pause\n";
    cmdLine += "ECHO on\n";
    cmdLine += "exit /b 0\n";

    cmdLine += ":error_tool\n";
    cmdLine += "ECHO !! nvtt_export.exe not found !!\n";
    cmdLine += "ECHO nvtt_export.exe is part of the https://developer.nvidia.com/nvidia-texture-tools-exporter package - please install\n";
    cmdLine += "ECHO and add 'C:/Program Files/NVIDIA Corporation/NVIDIA Texture Tools' or equivalent to your PATH and retry!\n";
    cmdLine += "pause\n";
    cmdLine += "ECHO on\n";
    cmdLine += "exit /b 1\n";

    batchFile << cmdLine;
    batchFile.close();

    std::string startCmd = " \"\" " + batchFileName;
    std::system(startCmd.c_str());

    //remove(batchFileName.c_str());

    return true; // TODO: check error code
}

void Sample::DenoisedScreenshot(nvrhi::ITexture * framebufferTexture) const
{
    std::string noisyImagePath = (app::GetDirectoryWithExecutable( ) / "photo.bmp").string();

    auto execute = [&](const std::string & dn = "OptiX")
    {
	    const auto p1 = std::chrono::system_clock::now();
		const std::string timestamp = std::to_string(std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count());

		const std::string fileName = "photo-denoised_" + dn + "_" + timestamp + ".bmp";

        std::string denoisedImagePath = (app::GetDirectoryWithExecutable() / fileName).string();
        std::string denoiserPath = GetLocalPath("Support/denoiser_"+dn).string();
        if (denoiserPath == "")
        { assert(false); return; }
        denoiserPath += "/denoiser.exe";

        if (!SaveTextureToFile(GetDevice(), m_CommonPasses.get(), framebufferTexture, nvrhi::ResourceStates::Common, noisyImagePath.c_str()))
        { assert(false); return; }

        std::string startCmd = "\"\"" + denoiserPath + "\"" + " -hdr 0 -i \"" + noisyImagePath + "\"" " -o \"" + denoisedImagePath + "\"\"";
        std::system(startCmd.c_str());

        std::string viewCmd = "\"\"" + denoisedImagePath + "\"\"";
        std::system(viewCmd.c_str());
    };
    execute("OptiX");
    execute("OIDN");
}

donut::math::float2 Sample::ComputeCameraJitter(uint frameIndex)
{
    if (!m_ui.RealtimeMode || m_ui.RealtimeAA == 0 || m_temporalAntiAliasingPass == nullptr)
        return dm::float2(0,0);

    // we currently use TAA for jitter even when it's not used itself
    return m_temporalAntiAliasingPass->GetCurrentPixelOffset();
}


#ifdef WIN32
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
int main(int __argc, const char** __argv)
#endif
{
    // Override Donut callback
    g_DonutDefaultCallback = donut::log::GetCallback();
    donut::log::SetCallback(SampleLogCallback);

    nvrhi::GraphicsAPI api = app::GetGraphicsAPIFromCommandLine(__argc, __argv);
    app::DeviceManager* deviceManager = app::DeviceManager::Create(api);

    app::DeviceCreationParameters deviceParams;
    deviceParams.backBufferWidth = 0;   // initialized from CmdLine
    deviceParams.backBufferHeight = 0;  // initialized from CmdLine
    deviceParams.swapChainSampleCount = 1;
    deviceParams.swapChainBufferCount = c_swapchainCount;
    deviceParams.startFullscreen = false;
    deviceParams.vsyncEnabled = true;
    deviceParams.enableRayTracingExtensions = true;
#if DONUT_WITH_DX11 || DONUT_WITH_DX12
    deviceParams.featureLevel = D3D_FEATURE_LEVEL_12_1;
#endif
#ifdef _DEBUG
    deviceParams.enableDebugRuntime = true;
    deviceParams.enableWarningsAsErrors = true;
    deviceParams.enableNvrhiValidationLayer = true;
    deviceParams.enableGPUValidation = false; // <- this severely impact performance but is good to enable from time to time
#endif
    deviceParams.supportExplicitDisplayScaling = true;

#if DONUT_WITH_STREAMLINE
    deviceParams.checkStreamlineSignature = true; // <- Set to false if you're using a local build of streamline
    deviceParams.streamlineAppId = 231313132;
#if defined(_DEBUG)
    deviceParams.enableStreamlineLog = true;
#endif
#endif

#if DONUT_WITH_VULKAN
    deviceParams.requiredVulkanDeviceExtensions.push_back("VK_KHR_buffer_device_address");
    deviceParams.requiredVulkanDeviceExtensions.push_back("VK_KHR_format_feature_flags2");

    // Attachment 0 not written by fragment shader; undefined values will be written to attachment (OMM baker)
    deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x0000000023e43bb7);

    // vertex shader writes to output location 0.0 which is not consumed by fragment shader (OMM baker)
    deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x000000000609a13b);

    // vkCmdPipelineBarrier2(): pDependencyInfo.pBufferMemoryBarriers[0].dstAccessMask bit VK_ACCESS_SHADER_READ_BIT
    // is not supported by stage mask (Unhandled VkPipelineStageFlagBits)
    // Vulkan validation layer not supporting OMM?
    deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x00000000591f70f2);

    // vkCmdPipelineBarrier2(): pDependencyInfo->pBufferMemoryBarriers[0].dstAccessMask(VK_ACCESS_SHADER_READ_BIT) is not supported by stage mask(VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT)
    // Vulkan Validaiotn layer not supporting OMM bug
    deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x000000005e6e827d);
#endif

    deviceParams.enablePerMonitorDPI = true;

    std::string preferredScene = "kitchen.scene.json"; //"programmer-art.scene.json";
    LocalConfig::PreferredSceneOverride(preferredScene);

    CommandLineOptions cmdLine;
    if (!cmdLine.InitFromCommandLine(__argc, __argv))
    {
        return 1;
    }

    if (!cmdLine.scene.empty())
    {
        preferredScene = cmdLine.scene;
    }

    if (cmdLine.nonInteractive)
    {
        donut::log::EnableOutputToMessageBox(false);
    }

    if (cmdLine.debug)
    {
        deviceParams.enableDebugRuntime = true;
        deviceParams.enableNvrhiValidationLayer = true;
    }

    deviceParams.backBufferWidth = cmdLine.width;
    deviceParams.backBufferHeight = cmdLine.height;
    deviceParams.startFullscreen = cmdLine.fullscreen;

    if (cmdLine.noWindow)
    {
        if (!deviceManager->CreateInstance(deviceParams))
        {
            log::fatal("CreateDeviceAndSwapChain failed: Cannot initialize a graphics device with the requested parameters");
            return 3;
        }
    }
    else
    {
        if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_windowTitle))
        {
            log::fatal("Cannot initialize a graphics device with the requested parameters");
            return 3;
        }
    }

    if (!deviceManager->GetDevice()->queryFeatureSupport(nvrhi::Feature::RayTracingPipeline))
    {
        log::fatal("The graphics device does not support Ray Tracing Pipelines");
        return 4;
    }

    if (!deviceManager->GetDevice()->queryFeatureSupport(nvrhi::Feature::RayQuery))
    {
        log::fatal("The graphics device does not support Ray Queries");
        return 4;
    }

    bool SERSupported = deviceManager->GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12 && deviceManager->GetDevice()->queryFeatureSupport(nvrhi::Feature::ShaderExecutionReordering);

    {
        SampleUIData& uiData = g_sampleUIData;
        Sample example(deviceManager, cmdLine, uiData);
        SampleUI gui(deviceManager, example, uiData, SERSupported);

        if (example.Init(preferredScene))
        {
            if (!cmdLine.noWindow)
                gui.Init( example.GetShaderFactory( ) );

            LocalConfig::PostAppInit(example, uiData);

            deviceManager->AddRenderPassToBack(&example);
            if (!cmdLine.noWindow)
                deviceManager->AddRenderPassToBack(&gui);
            deviceManager->RunMessageLoop();
            if (!cmdLine.noWindow)
                deviceManager->RemoveRenderPass(&gui);
            deviceManager->RemoveRenderPass(&example);
        }
    }

    deviceManager->Shutdown();

    delete deviceManager;

    return 0;
}
