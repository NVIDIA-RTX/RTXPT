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
#include <donut/engine/View.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/json.h>
#include <donut/core/math/math.h>
#include <donut/shaders/light_cb.h>
#include <donut/shaders/view_cb.h>
#include <nvrhi/utils.h>
#include <nvrhi/common/misc.h>
#include <cmath>
#include <cstddef>

#if DONUT_WITH_DX12
#include <nvrhi/d3d12.h> // for the InstanceDesc layout asserts below
#endif

#include "SampleCommon/PTPipelineBaker.h"
#include "SampleCommon/ComputePipelineBaker.h"
#include "SampleCommon/DirectoryModifiedMonitor.h"

#include "SampleCommon/AccelerationStructureUtil.h"
#include "SampleCommon/BlasPool.h"
#include "SampleCommon/ParallelFor.h"

#include "Lighting/Distant/EnvMapImportanceSamplingBaker.h"
#include "Materials/MaterialsBaker.h"

#include "OpacityMicroMap/OmmBaker.h"

#include "SampleCommon/LocalConfig.h"
#include "SampleCommon/SampleBaseApp.h"
#include "SampleCommon/CaptureScriptManager.h"
#include "Misc/Korgi.h"

#include "GPUSort/GPUSort.h"

#include "Misc/ZoomTool.h"

#include "ProcessingPasses/DenoisingGuidesBaker.h"

#include "SampleGame/GameScene.h"

#include "Sharc/SharcIntegration.h"


using namespace donut;
using namespace donut::math;
using namespace donut::app;
using namespace donut::vfs;
using namespace donut::engine;
using namespace donut::render;

#include <fstream>
#include <iostream>
#include <random>

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

#if defined(RTXPT_D3D_AGILITY_SDK_VERSION)
// Required for Agility SDK on Windows 10. Setup 1.c. 2.a.
// https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/
extern "C"
{
    __declspec(dllexport) extern const UINT D3D12SDKVersion = RTXPT_D3D_AGILITY_SDK_VERSION;
    __declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\";
}
#endif

const char* g_versionString = "v1.9.0";

const float c_envMapRadianceScale = 1.0f / 4.0f; // used to make input 32bit float radiance fit into 16bit float range that baker supports; going lower than 1/4 causes issues with current BC6U compression algorithm when used

static FPSLimiter g_FPSLimiter;

Sample::Sample(donut::app::DeviceManager& deviceManager,
    const CommandLineOptions& cmdLine)
    : app::ApplicationBase(&deviceManager )
    , m_cmdLine(cmdLine)
    , m_ui(g_sampleUIData)
{
    m_progressLoading.Start("Initializing...");
    m_progressLoading.Set(50);

    // Explicit lifetime, deliberately not a file-scope global: a global would join its worker during static destruction,
    // which on Windows can run under the loader lock. The pipeline bakers register their shader roots with it.
    DirectoryModifiedMonitor::Init();

    m_camera.SetRotateSpeed(.003f);

#if DONUT_WITH_STREAMLINE
    m_ui.IsDLSSSuported = GetDeviceManager()->GetStreamline().IsDLSSAvailable();
    m_ui.IsDLSSFGSupported = GetDeviceManager()->GetStreamline().IsDLSSGAvailable();
    m_ui.IsReflexSupported = GetDeviceManager()->GetStreamline().IsReflexAvailable();
    m_ui.IsDLSSRRSupported = GetDeviceManager()->GetStreamline().IsDLSSRRAvailable();
#endif

    // Enumerate all environment maps in the media folder
    m_envMapMediaList.clear();
    m_envMapMediaFolder = GetLocalPath(c_AssetsFolder) / c_EnvMapSubFolder;
    for (const auto& file : std::filesystem::directory_iterator(m_envMapMediaFolder))
    {
        if (!file.is_regular_file()) continue;
        if (file.path().extension() == ".exr" || file.path().extension() == ".hdr" || file.path().extension() == ".dds")
            m_envMapMediaList.push_back(file.path());
    }

    m_captureScriptManager = std::make_unique<CaptureScriptManager>(*this, m_ui, m_cmdLine);

    deviceManager.m_callbacks.beforeRender = [this](donut::app::DeviceManager&, uint32_t)
    {
        auto now = std::chrono::high_resolution_clock::now();
        if (m_afterPresentTime.time_since_epoch().count() != 0)
        {
            m_timeInLastFrameGap = std::chrono::duration<double>(now - m_afterPresentTime).count();
            m_avgTimeInLastFrameGap = (m_avgTimeInLastFrameGap == 0.0)
                ? m_timeInLastFrameGap
                : m_avgTimeInLastFrameGap * 0.95 + m_timeInLastFrameGap * 0.05;
        }
        m_renderStartTime = now;
    };
    deviceManager.m_callbacks.afterRender = [this](donut::app::DeviceManager&, uint32_t)
    {
        m_timeInLastRender = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - m_renderStartTime).count();
        m_avgTimeInLastRender = (m_avgTimeInLastRender == 0.0)
            ? m_timeInLastRender
            : m_avgTimeInLastRender * 0.95 + m_timeInLastRender * 0.05;
    };
    deviceManager.m_callbacks.beforePresent = [this](donut::app::DeviceManager&, uint32_t)
    {
        m_presentStartTime = std::chrono::high_resolution_clock::now();
    };
    deviceManager.m_callbacks.afterPresent = [this](donut::app::DeviceManager&, uint32_t)
    {
        auto now = std::chrono::high_resolution_clock::now();
        m_timeInLastPresent = std::chrono::duration<double>(now - m_presentStartTime).count();
        m_avgTimeInLastPresent = (m_avgTimeInLastPresent == 0.0)
            ? m_timeInLastPresent
            : m_avgTimeInLastPresent * 0.95 + m_timeInLastPresent * 0.05;
        m_afterPresentTime = now;

        GpuProfiler::AfterPresent();
        FrameMark;
    };
}

Sample::~Sample()
{
    // Members are destroyed in reverse declaration order, and m_scene is declared well above m_blasPool -
    // so left to that order the pool would go first, while the scene still held pooled BLAS handles, and
    // each of those would then try to free its range back into a destroyed pool. Releasing in the order
    // the dependency actually runs is what keeps this correct no matter where a member sits in the class.
    ReleaseBlases();
    if (m_ommBaker != nullptr)
        m_ommBaker->SetBlasCompactor(nullptr);
    m_blasCompactor.reset();    // holds a BlasPool&, so it goes first
    m_blasPool.reset();

    // After the bakers are gone - their handles are just indices, so ordering only matters in that nothing should query
    // the monitor afterwards.
    DirectoryModifiedMonitor::Shutdown();

    GetDevice()->waitForIdle();     // the profiler's query heap and readback buffer may still be in use
    GpuProfiler::Shutdown();
}

void Sample::DebugDrawLine( float3 start, float3 stop, float4 col1, float4 col2 )
{
    if( int(m_cpuSideDebugLines.size())+2 >= MAX_DEBUG_LINES )
        return;
    DebugLineStruct dls = { float4(start, 1), col1 }, dle = { float4(stop, 1), col2 };
    m_cpuSideDebugLines.push_back(dls);
    m_cpuSideDebugLines.push_back(dle);
}

// donut::app::FindPreferredScene matches with a case-sensitive substring search (s.find(preferred)), which is what makes
// the --scene command line argument case-sensitive. This is the same logic but case-insensitive. Note the fallback
// behaviour is inherited deliberately: if nothing matches we still return the first available scene rather than failing,
// but we log it - a silent fallback to an unrelated scene is exactly what makes a mistyped --scene hard to diagnose.
static std::string FindPreferredSceneCaseInsensitive(const std::vector<std::string>& available, const std::string& preferred)
{
    if (available.empty())
        return "";

    // deliberately ASCII-only: scene filenames are ASCII and std::tolower is locale-dependent (and UB for negative chars)
    auto toLowerAscii = [](std::string s)
    {
        for (char& c : s)
            if (c >= 'A' && c <= 'Z')
                c += 'a' - 'A';
        return s;
    };

    const std::string preferredLower = toLowerAscii(preferred);
    // an exact name wins over a substring hit, e.g. 'programmer-art.scene.json' vs 'bistro-programmer-art.scene.json'
    for (const std::string& s : available)
        if (toLowerAscii(s) == preferredLower)
            return s;
    for (const std::string& s : available)
        if (toLowerAscii(s).find(preferredLower) != std::string::npos)
            return s;

    log::warning("Scene '%s' not found in the assets folder - falling back to '%s'", preferred.c_str(), available.front().c_str());
    return available.front();
}

void Sample::Init(const std::string& preferredScene,
    const std::shared_ptr<donut::engine::ShaderFactory>& shaderFactory)
{
    m_shaderFactory = shaderFactory;

    m_CommonPasses = std::make_shared<engine::CommonRenderPasses>(GetDevice(), m_shaderFactory);
    m_bindingCache = std::make_unique<engine::BindingCache>(GetDevice());
    
    m_sampleGame = std::make_unique<GameScene>(*this, m_cmdLine);
    m_progressLoading.Set(95);

    nvrhi::BindlessLayoutDesc bindlessLayoutDesc;
    bindlessLayoutDesc.visibility = nvrhi::ShaderType::All;
    bindlessLayoutDesc.firstSlot = 0;
    bindlessLayoutDesc.maxCapacity = 1024;
    bindlessLayoutDesc.registerSpaces = {
        nvrhi::BindingLayoutItem::RawBuffer_SRV(1),
        nvrhi::BindingLayoutItem::Texture_SRV(2)
    };
    auto device = GetDevice();
    m_bindlessLayout = device->createBindlessLayout(bindlessLayoutDesc);

    // Second bindless layout, for SM6.6 ResourceDescriptorHeap access (Rtxpt/Shaders/Bindless.hlsli). This is not
    // a second descriptor table binding - a mutable layout takes no root parameter at all, it only flips
    // D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED on the root signature.
    // registerSpaces MUST be left empty: the Vulkan backend errors out on a mutable layout that declares any,
    // and D3D12 silently ignores them, so getting this wrong is a Vulkan-only failure.
    nvrhi::BindlessLayoutDesc bindlessHeapLayoutDesc;
    bindlessHeapLayoutDesc.visibility = nvrhi::ShaderType::All;
    bindlessHeapLayoutDesc.layoutType = nvrhi::BindlessLayoutDesc::LayoutType::MutableSrvUavCbv;
    bindlessHeapLayoutDesc.maxCapacity = 1024;
    m_bindlessHeapLayout = device->createBindlessLayout(bindlessHeapLayoutDesc);

    nvrhi::BindingLayoutDesc globalBindingLayoutDesc;
    // No Pixel bit: SRVs would then need pixel shader resource states, illegal on a compute queue (async prefix)
    globalBindingLayoutDesc.visibility = nvrhi::ShaderType::Compute | nvrhi::ShaderType::AllRayTracing;
    globalBindingLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
        nvrhi::BindingLayoutItem::PushConstants(1, sizeof(SampleMiniConstants)),
        nvrhi::BindingLayoutItem::RayTracingAccelStruct(0),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(3),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(4),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(5),
        nvrhi::BindingLayoutItem::Texture_SRV(6),               // t_LdrColorScratch
        nvrhi::BindingLayoutItem::Texture_SRV(10),              // t_EnvironmentMap
        nvrhi::BindingLayoutItem::Texture_SRV(11),              // t_EnvironmentMapImportanceMap        <- TODO: remove this, no longer used
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(12),     // t_LightsCB
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(13),     // t_Lights
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(14),     // t_LightsEx
        nvrhi::BindingLayoutItem::Sampler(0),
        nvrhi::BindingLayoutItem::Sampler(1),
        nvrhi::BindingLayoutItem::Sampler(2),
        nvrhi::BindingLayoutItem::Sampler(3),               // s_LinearClampSampler (LUT sampling, e.g. BRDF LUT)
        nvrhi::BindingLayoutItem::Texture_UAV(0),           // u_OutputColor
        nvrhi::BindingLayoutItem::Texture_UAV(1),           // u_ProcessedOutputColor
        nvrhi::BindingLayoutItem::Texture_UAV(2),           // u_PostTonemapOutputColor
        nvrhi::BindingLayoutItem::Texture_UAV(4),           // u_Throughput
        nvrhi::BindingLayoutItem::Texture_UAV(5),           // u_MotionVectors
        nvrhi::BindingLayoutItem::Texture_UAV(6),           // u_Depth
        nvrhi::BindingLayoutItem::Texture_UAV(7),           // u_SpecularHitT
        nvrhi::BindingLayoutItem::Texture_UAV(8),           // u_ScratchFloat1
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
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(51),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(52),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(53),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(54),
        // ReSTIR GI
        nvrhi::BindingLayoutItem::Texture_UAV(60),          // u_SecondarySurfacePositionNormal
        nvrhi::BindingLayoutItem::Texture_UAV(61),          // u_SecondarySurfaceRadiance

        nvrhi::BindingLayoutItem::Texture_UAV(70),          // u_RRDiffuseAlbedo
        nvrhi::BindingLayoutItem::Texture_UAV(71),          // u_RRSpecAlbedo
        nvrhi::BindingLayoutItem::Texture_UAV(72),          // u_RRNormalsAndRoughness
        nvrhi::BindingLayoutItem::Texture_UAV(73),          // u_RRSpecMotionVectors
        nvrhi::BindingLayoutItem::Texture_UAV(74),          // u_RRTransparencyLayer
        nvrhi::BindingLayoutItem::Texture_UAV(75),          // u_DenoisingAvgLayerRadiance
        nvrhi::BindingLayoutItem::RawBuffer_UAV(76),        // u_SubSamplingPixelList

        nvrhi::BindingLayoutItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX),
        nvrhi::BindingLayoutItem::Texture_UAV(SHADER_DEBUG_VIZ_TEXTURE_UAV_INDEX)
    };

    if (m_sampleCapabilities.UseNeeAtLightSampling)
    {
        globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::TypedBuffer_SRV(15));      // t_LightProxyCounters
        globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::TypedBuffer_SRV(16));      // t_LightProxyIndices
        globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::TypedBuffer_SRV(17));      // t_LightLocalSamplingBuffer
        globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_SRV(18));          // t_EnvLookupMap
        globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::TypedBuffer_SRV(19));      // t_LightWeightsBoost
        globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_UAV(20));          // u_LightFeedbackTotalWeight
        globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_UAV(21));          // u_LightFeedbackCandidates
    }

    // NV HLSL extensions - DX12 only - we should probably expose some form of GetNvapiIsInitialized instead
    if (device->queryFeatureSupport(nvrhi::Feature::HlslExtensionUAV))
    {
        globalBindingLayoutDesc.bindings.push_back(
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(NV_SHADER_EXTN_SLOT_NUM));
    }

    // denoising layers buffers -- must be last because these items are appended to the BindingSetDesc after the main list
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_UAV(40));
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(42));
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_UAV(44));
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(45));

    // GBuffer
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_UAV(100));
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_UAV(101));
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_UAV(102));
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_UAV(103));
    
    // Environment IBL / reflection system textures (used by IntroSample for improved reflections)
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_SRV(80));   // t_PrefilteredEnvCube
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_SRV(81));   // t_EnvIrradianceCube
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_SRV(82));   // t_SSRBlurChain
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_SRV(83));   // t_BRDFLUT
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_SRV(84));   // t_DepthHierarchy
    // ReflectionConstants (b10) removed - intro sample now packs EnvGGXMaxMip / SSRMaxMip
    // into the existing SampleMiniConstants push-constant block (params.z / params.w).
    // SSRConstants removed - SSR now uses push constants instead of constant buffer
    
    // SSR result UAV (depth hierarchy UAVs u80-84 are in a dedicated binding set)
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_UAV(85));   // u_SSRResult

    // Previous frame depth (for temporal reprojection / disocclusion detection)
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_SRV(87));   // t_PrevDepth
    // Previous frame packed shading normal (ReSTIR DI temporal resampling)
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_SRV(95));   // t_PrevSpecNormal32
    // Intro RTXDI analytic light buffers (split into Base + Extended SoA, mirrors LightsBaker)
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(90));   // t_IntroRTXDILightsBase
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(91));   // t_IntroRTXDILightsEx
    // Direct lighting output (written by DirectLightingPass, read by deferred combine)
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_UAV(88));   // u_DirectLighting
    // RTGI output (written by RTGIRenderer, read by the DeferredLightingCombine pass)
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_UAV(89));   // u_RTGIOutput

    // Indirect path tracer output (written by IndirectPathTracer, read by deferred combine).
    // Moved from u93 -> u95 back when SHaRC claimed u90-u93. u93 is unused again now that the lock buffer is
    // gone, but IntroSample's shaders declare u95, so it stays there.
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_UAV(95));   // u_IndirectOutput

    // SHaRC buffers (hash u90, accumulation u91, resolved u92). Shared between IntroSample's native SHaRCRenderer
    // and AdvancedPathTracer's reference-style SharcIntegration. Bound with placeholders below and overridden by
    // the active sample's AddCustomBindings. The SHaRC constants live in the global SampleConstants
    // (g_Const.ptConsts.sharc, b0) — no dedicated constant buffer binding. There is no lock buffer: the SDK only
    // needs one when 64-bit atomics are unavailable, a configuration SharcBindings.hlsli now rejects outright.
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(90));  // u_SHaRCHashEntries / u_SharcHashEntriesBuffer
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(91));  // u_SHaRCAccumulation / u_SharcAccumulationBuffer
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(92));  // u_SHaRCResolved / u_SharcResolvedBuffer

    // Bindless validation metadata (t_BindlessValidationData, see Rtxpt/Shaders/Bindless.hlsli). Added
    // unconditionally: the layout is shared by every pipeline and by both build configurations, and the shader
    // side appears or disappears with RTXPT_SHADER_DEBUG, so making the slot conditional would desync them.
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(50));   // t_BindlessValidationData

    // Intro ReSTIR DI (direct RTXDI integration, not RtxdiPass)
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(94));   // u_LightReservoirs
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::TypedBuffer_SRV(94));      // t_NeighborOffsets
    globalBindingLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::VolatileConstantBuffer(12)); // g_IntroReSTIRConst

    m_bindingLayout = device->createBindingLayout(globalBindingLayoutDesc);

    m_DescriptorTable = std::make_shared<engine::DescriptorTableManager>(device, m_bindlessLayout);
    m_bindlessHeapDescriptorTable = std::make_shared<engine::DescriptorTableManager>(device, m_bindlessHeapLayout);

    {   // Created here rather than lazily so it is never null when the binding set is first built.
        nvrhi::BufferDesc validationBufferDesc;
        validationBufferDesc.byteSize = 1024 * sizeof(uint32_t);
        validationBufferDesc.structStride = sizeof(uint32_t);
        validationBufferDesc.debugName = "BindlessValidationData";
        validationBufferDesc.initialState = nvrhi::ResourceStates::NonPixelShaderResource;
        validationBufferDesc.keepInitialState = true;
        m_bindlessValidationBuffer = device->createBuffer(validationBufferDesc);
    }

    {   // Stands in for textures the async prefix binding set variants must not touch (see RecreateBindingSet)
        nvrhi::TextureDesc dummyDesc;
        dummyDesc.width = 1;
        dummyDesc.height = 1;
        dummyDesc.format = nvrhi::Format::RGBA16_FLOAT;
        dummyDesc.isUAV = true;
        dummyDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        dummyDesc.keepInitialState = true;
        dummyDesc.debugName = "PrefixDummyUAV";
        m_prefixDummyUAV = device->createTexture(dummyDesc);

        // Stands in for the ShaderDebug buffer in the tail variant; shader writes past its end are discarded
        nvrhi::BufferDesc dummyBufferDesc;
        dummyBufferDesc.byteSize = 256;
        dummyBufferDesc.canHaveUAVs = true;
        dummyBufferDesc.canHaveRawViews = true;
        dummyBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        dummyBufferDesc.keepInitialState = true;
        dummyBufferDesc.debugName = "DummyRawBufferUAV";
        m_dummyRawBufferUAV = device->createBuffer(dummyBufferDesc);
    }

    auto nativeFS = std::make_shared<vfs::NativeFileSystem>();
    m_TextureCache = std::make_shared<engine::TextureCache>(device, nativeFS, m_DescriptorTable);

    memset( &m_feedbackData, 0, sizeof(DebugFeedbackStruct) * 1 );
    memset( &m_debugDeltaPathTree, 0, sizeof(DeltaTreeVizPathVertex) * cDeltaTreeVizMaxVertices );

    //Draw lines from the feedback buffer - this is old and should all be replaced by ShaderDebug
    {
        std::vector<ShaderMacro> drawLinesMacro = { ShaderMacro("DRAW_LINES_SHADERS_OLD", "1") };
        m_linesVertexShader = m_shaderFactory->CreateShader("app/Misc/DebugLines.hlsl", "main_vs", &drawLinesMacro, nvrhi::ShaderType::Vertex);
        m_linesPixelShader = m_shaderFactory->CreateShader("app/Misc/DebugLines.hlsl", "main_ps", &drawLinesMacro, nvrhi::ShaderType::Pixel);

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
        m_linesInputLayout = device->createInputLayout(attributes, uint32_t(std::size(attributes)), m_linesVertexShader);

        nvrhi::BindingLayoutDesc linesBindingLayoutDesc;
        linesBindingLayoutDesc.visibility = nvrhi::ShaderType::All;
        linesBindingLayoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::Texture_SRV(0)
        };

        m_linesBindingLayout = device->createBindingLayout(linesBindingLayoutDesc);

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
            m_feedback_Buffer_Gpu = device->createBuffer(bufferDesc);

            bufferDesc.canHaveUAVs = false;
            bufferDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
            bufferDesc.structStride = 0;
            bufferDesc.keepInitialState = false;
            bufferDesc.initialState = nvrhi::ResourceStates::Unknown;
            bufferDesc.debugName = "Feedback_Buffer_Cpu";
            m_feedback_Buffer_Cpu = device->createBuffer(bufferDesc);

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
            m_debugLineBufferCapture    = device->createBuffer(bufferDesc);
            bufferDesc.debugName = "DebugLinesDisplay";
            m_debugLineBufferDisplay    = device->createBuffer(bufferDesc);

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
            m_debugDeltaPathTree_Gpu = device->createBuffer(bufferDesc);

            bufferDesc.canHaveUAVs = false;
            bufferDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
            bufferDesc.structStride = 0;
            bufferDesc.keepInitialState = false;
            bufferDesc.initialState = nvrhi::ResourceStates::Unknown;
            bufferDesc.debugName = "Feedback_PathDecomp_Cpu";
            m_debugDeltaPathTree_Cpu = device->createBuffer(bufferDesc);


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
            m_debugDeltaPathTreeSearchStack = device->createBuffer(bufferDesc);
        }
    }

    // Main constant buffer
    m_constantBuffer = device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(SampleConstants), "SampleConstants", engine::c_MaxRenderPassConstantBufferVersions*2));	// *2 because in some cases we update twice per frame

    // Every BLAS build suballocates its scratch from this command list's pool, and a pooled chunk is
    // only reusable by a request that fits in it whole - the 64KB default fragments it into thousands
    // of pieces on a large scene.
    m_graphicsCommandList = device->createCommandList(nvrhi::CommandListParameters().setScratchChunkSize(4 * 1024 * 1024));
    // Async prefix list (see ShouldRecordPrefixAsync); it refits skinned BLASes and builds the TLAS, hence the same scratch size
    m_prefixCommandList = device->createCommandList(nvrhi::CommandListParameters().setQueueType(nvrhi::CommandQueue::Compute).setEnableImmediateExecution(false).setScratchChunkSize(4 * 1024 * 1024));
    m_commandList = m_graphicsCommandList;
    m_asyncPrefixCapable = device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12 && device->queryFeatureSupport(nvrhi::Feature::EnhancedBarriers);
    GpuProfiler::Init(device);

    if(device->queryFeatureSupport(nvrhi::Feature::RayTracingOpacityMicromap))
        m_ommBaker = std::make_shared<OmmBaker>(device, m_DescriptorTable, m_TextureCache, m_shaderFactory);

    // Get all scenes in "assets" folder
    const std::string mediaExt = ".scene.json";
    for (const auto& file : std::filesystem::directory_iterator(GetLocalPath(c_AssetsFolder)))
    {
        if (!file.is_regular_file()) continue;
        std::string fileName = file.path().filename().string();
        std::string longExt = (fileName.size()<=mediaExt.length())?(""):(fileName.substr(fileName.length()-mediaExt.length()));
        if ( longExt == mediaExt )
            m_sceneFilesAvailable.push_back( file.path().filename().string() );
    }

    std::string scene = FindPreferredSceneCaseInsensitive(m_sceneFilesAvailable, preferredScene);

    // Select initial scene
    SetCurrentScene(scene);
}

void Sample::SetCurrentScene( const std::string & sceneName, bool forceReload )
{
    if( m_currentSceneName == sceneName && !forceReload )
        return;
    m_currentSceneName = sceneName;
    m_ui.ResetAccumulation = true;
    SetAsynchronousLoadingEnabled( false );
    std::filesystem::path scenePath = GetLocalPath(c_AssetsFolder) / sceneName;
    m_currentScenePath = scenePath;
    m_progressLoading.Stop();
    m_progressLoading.Start("Loading scene...");
    BeginLoadingScene( std::make_shared<vfs::NativeFileSystem>(), scenePath );
    if( m_scene == nullptr )
    {
        log::error( "Unable to load scene '%s'", sceneName.c_str() );
        m_currentScenePath = std::filesystem::path();
        m_progressLoading.Stop();
        return;
    }
}

void Sample::SceneUnloading( )
{
    m_ui.TogglableNodes = nullptr;
    ApplicationBase::SceneUnloading();
    InvalidateBindingSet();
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
    if (m_rtxdiPass != nullptr) 
	    m_rtxdiPass->Reset();
    if (m_ommBaker != nullptr)
        m_ommBaker->SceneUnloading();

    DestroyRTPipelines();
    m_ptPipelineBaker = nullptr;
    m_computePipelineBaker = nullptr;

    if (m_sampleGame!=nullptr) m_sampleGame->SceneUnloading();
}

bool Sample::LoadScene(std::shared_ptr<vfs::IFileSystem> fs, const std::filesystem::path& sceneFileName)
{
    m_scene = std::shared_ptr<ExtendedScene>( new ExtendedScene(GetDevice(), *m_shaderFactory, fs, m_TextureCache, m_DescriptorTable, std::make_shared<ExtendedSceneTypeFactory>() ) );
    m_progressLoading.Set(10);
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

    std::shared_ptr<PerspectiveCameraEx> sceneCameraEx = std::dynamic_pointer_cast<PerspectiveCameraEx>(sceneCamera);
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
    UpdateTemporalAntiAliasingPass();   // before ComputeCameraJitter, which uses its sequence when TAA is active

    nvrhi::Viewport windowViewport(float(m_renderSize.x), float(m_renderSize.y));
    m_view->SetViewport(windowViewport);
    float outputAspectRatio = m_displayAspectRatio; //windowViewport.width() / windowViewport.height();    // render and display outputs might not match in case of lower DLSS/etc resolution rounding!
    m_view->SetMatrices(m_camera.GetWorldToViewMatrix(), perspProjD3DStyleReverse(m_cameraVerticalFOV, outputAspectRatio, m_cameraZNear));
    m_view->SetPixelOffset(ComputeCameraJitter(m_frameIndex));
    m_view->UpdateCache();
    if ((m_frameIndex & 0xFFFFFFFF) == 0)
    {
        m_viewPrevious->SetMatrices(m_view->GetViewMatrix(), m_view->GetProjectionMatrix());
        m_viewPrevious->SetPixelOffset(m_view->GetPixelOffset());
        m_viewPrevious->UpdateCache();
    }
}

// TAA and its display-size history exist only while TAA is active, to save memory. CreateRenderPasses drops the pass,
// so it is recreated here against the current render targets and shaders.
void Sample::UpdateTemporalAntiAliasingPass()
{
    if (!m_ui.RealtimeMode || m_ui.RealtimeAA != 1)
    {
        m_temporalAntiAliasingPass = nullptr;
        m_renderTargets->ReleaseTemporalFeedback();
        return;
    }

    m_renderTargets->EnsureTemporalFeedback();

    if (m_temporalAntiAliasingPass == nullptr)
    {
        // history is either new, or the new pass would start on the older of its two textures
        m_temporalAntiAliasingFeedbackValid = false;

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
    m_temporalAntiAliasingPass->SetJitter(m_ui.TemporalAntiAliasingJitter);
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
    m_frameIndex = 0;

    m_progressLoading.Set(50);

    if (m_sampleGame != nullptr) m_sampleGame->SceneLoaded(m_scene, m_currentScenePath, GetLocalPath(c_AssetsFolder));

    m_progressLoading.Set(55);

    ApplicationBase::SceneLoaded( );

    // Apply initial material values for early OMM eligibility; fallback imports are finalized at first render.
    // Keep node animation timing below, after the initial scene refresh.
    for (const auto& animation : m_scene->GetSceneGraph()->GetAnimations())
        for (const auto& channel : animation->GetChannels())
        {
            if (channel->GetAttribute() != AnimationAttribute::LeafProperty)
                continue;
            const auto node = channel->GetTargetNode();
            if (!node || std::dynamic_pointer_cast<MeshInstance>(node->GetLeaf()))
                (void)channel->Apply(0.0f);
        }

    // OMM input identity must see the resolved material while CPU mesh data still exists.
    if (m_materialsBaker == nullptr)
        m_materialsBaker = std::make_shared<MaterialsBaker>(GetMaterialSpecializationShader(), GetDevice(), m_TextureCache, m_shaderFactory);
    else
        m_materialsBaker->SceneReloaded();
    m_materialsBaker->LoadMaterials(m_scene, m_currentScenePath, GetLocalPath(c_AssetsFolder));

    m_progressLoading.Set(60);

    m_sceneTime = 0.f;
    m_scene->FinishedLoading( GetFrameIndex( ) );

    m_progressLoading.Set(65);

	// Find lights; do this before special cases to avoid duplicates
	for (auto light : m_scene->GetSceneGraph()->GetLights())
	{
		m_lights.push_back(light);
	}

    // seem like sensible defaults
    m_ui.ToneMappingParams.exposureCompensation = 2.0f;
    m_ui.ToneMappingParams.exposureValue = 0.0f;

    m_ui.EnvironmentMapParams = EnvironmentMapRuntimeParameters();
    std::shared_ptr<EnvironmentLight> envLight = FindEnvironmentLight(m_lights);
    if (envLight)
    {
        m_envMapLocalPath = envLight->path;
        m_ui.EnvironmentMapParams.Intensity = envLight->intensity;
    }
    else
    {
        m_envMapLocalPath = "";
    }
    m_envMapOverride = c_EnvMapSceneDefault;

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

    std::shared_ptr<SampleSettings> settings = m_scene->GetSampleSettingsNode();
    if (settings != nullptr)
    {
        m_ui.RealtimeMode = settings->realtimeMode.value_or(m_ui.RealtimeMode);
        m_ui.EnableAnimations = settings->enableAnimations.value_or(m_ui.EnableAnimations);
        if (settings->startingCamera.has_value())
            m_selectedCameraIndex = settings->startingCamera.value()+1; // slot 0 reserved for free flight camera
        if (settings->realtimeFireflyFilter.has_value())
        {
            m_ui.RealtimeFireflyFilterThreshold = settings->realtimeFireflyFilter.value();
            m_ui.RealtimeFireflyFilterEnabled = true;
        }
        m_ui.BounceCount = settings->maxBounces.value_or(m_ui.BounceCount);
        m_ui.DiffuseBounceCount = settings->maxDiffuseBounces.value_or(m_ui.DiffuseBounceCount);
        m_ui.TexLODBias = settings->textureMIPBias.value_or(m_ui.TexLODBias);
    }

    if (m_cmdLine.stopAnimations)
        m_ui.EnableAnimations = false;

    m_progressLoading.Set(70);

    LocalConfig::PostSceneLoad( *this, m_ui );

    m_progressLoading.Set(90);

    if (m_envMapBaker!=nullptr) m_envMapBaker->SceneReloaded();
    if (m_lightsBaker!=nullptr) m_lightsBaker->SceneReloaded();
    if (m_ommBaker!=nullptr) m_ommBaker->SceneLoaded(*m_scene);

    m_progressLoading.Set(100);

    if (m_cmdLine.OverrideToRealtimeMode)
        m_ui.RealtimeMode = true;
    if (m_cmdLine.OverrideToReferenceMode)
        m_ui.RealtimeMode = false;
    
    if (m_cmdLine.OverrideAutoexposureOff)
    {
        m_ui.ToneMappingParams.autoExposure = false;
        m_ui.ToneMappingParams.exposureValue = 0.0f;
    }
    if (m_cmdLine.OverrideExposureOffset != FLT_MAX)
        m_ui.ToneMappingParams.exposureCompensation = m_cmdLine.OverrideExposureOffset;
    if (m_cmdLine.DisableFireflyFilters)
    {
        m_ui.RealtimeFireflyFilterEnabled = false;
        m_ui.ReferenceFireflyFilterEnabled = false;
    }
    if (m_cmdLine.DisablePostProcessFilters)
    {
        m_ui.EnableBloom = false;
    }
    if (m_cmdLine.OverrideRadianceCacheMode >= 0)
        m_ui.RadianceCacheMode = m_cmdLine.OverrideRadianceCacheMode;
    if (m_cmdLine.cameraPosDirUp != "")
        SetCurrentCameraPosDirUp(m_cmdLine.cameraPosDirUp);

    m_ui.MaterialVariantIndex = 0;

    ApplyPerformancePreset(m_ui, 2, (GetOMMBaker()!=nullptr)?(GetOMMBaker().get()):(nullptr) ); // 2 is the "Balanced"

    // after the preset, which sets it too
    if (m_cmdLine.OverrideSubSamplingMode >= 0)
        m_ui.RealtimeSubSamplingMode = dm::clamp(m_cmdLine.OverrideSubSamplingMode, 0, 3);

    m_asyncLoadingInProgress = true;
}

bool Sample::KeyboardUpdate(int key, int scancode, int action, int mods)
{
    if (m_zoomTool && m_zoomTool->KeyboardUpdate(key, scancode, action, mods))
        return true;

    if (mods != (GLFW_MOD_SHIFT | GLFW_MOD_CONTROL))
    {
        if (!(m_sampleGame && m_sampleGame->CameraActive()))
            m_camera.KeyboardUpdate(key, scancode, action, mods);

        if (m_sampleGame && m_sampleGame->KeyboardUpdate(key, scancode, action, mods))
            return true;
    }

    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS && mods != GLFW_MOD_CONTROL && mods != GLFW_MOD_ALT)
    {
        m_ui.EnableAnimations = !m_ui.EnableAnimations;
        return true;
    }
    if( key == GLFW_KEY_F2 && action == GLFW_PRESS )
        m_ui.ShowUI = !m_ui.ShowUI;
    if( key == GLFW_KEY_R && action == GLFW_PRESS && mods == GLFW_MOD_CONTROL )
        m_ui.ShaderReloadRequested = true;

    // Camera shift+control+c/v for copy-pasting position
    if (key == GLFW_KEY_C && action == GLFW_PRESS && mods == (GLFW_MOD_SHIFT | GLFW_MOD_CONTROL))
        ImGui::SetClipboardText(GetCurrentCameraPosDirUp().c_str());
    if (key == GLFW_KEY_V && action == GLFW_PRESS && mods == (GLFW_MOD_SHIFT | GLFW_MOD_CONTROL))
    {
        const char* cpbrdtxt = ImGui::GetClipboardText();
        SetCurrentCameraPosDirUp(cpbrdtxt ? cpbrdtxt : "");
    }

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
    if (!(m_sampleGame && m_sampleGame->CameraActive()))
        m_camera.MousePosUpdate(xpos, ypos);
    if (m_sampleGame)   m_sampleGame->MousePosUpdate(xpos /** upscalingScale.x*/, ypos /** upscalingScale.y*/);

    float2 upscalingScale = float2(1,1);
    if (m_renderTargets != nullptr)
        upscalingScale = float2(m_renderSize)/float2(m_displaySize);

    m_pickPosition = uint2( static_cast<uint>( xpos * upscalingScale.x ), static_cast<uint>( ypos * upscalingScale.y ) );
    m_ui.MousePos = uint2( static_cast<uint>( xpos * upscalingScale.x ), static_cast<uint>( ypos * upscalingScale.y ) );

    if (m_zoomTool)     m_zoomTool->MousePosUpdate( xpos, ypos );

    return true;
}

bool Sample::MouseButtonUpdate(int button, int action, int mods)
{
    if (m_zoomTool)     
        if (m_zoomTool->MouseButtonUpdate(button, action, mods))
            return true;

    if (!(m_sampleGame && m_sampleGame->CameraActive()))
        m_camera.MouseButtonUpdate(button, action, mods);
    if (m_sampleGame)   m_sampleGame->MouseButtonUpdate(button, action, mods);

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
    if (!(m_sampleGame && m_sampleGame->CameraActive()))
    {
        //m_camera.MouseScrollUpdate(xoffset, yoffset);
        m_ui.CameraMoveSpeed *= 1.0f + yoffset*0.1f;
    }
    return true;
}

void Sample::Animate(float fElapsedTimeSeconds)
{
    if (m_ui.ActualFPSLimiter()>0)    // essential for stable video recording
        fElapsedTimeSeconds = 1.0f / (float)m_ui.ActualFPSLimiter();

    m_captureScriptManager->PreAnim(fElapsedTimeSeconds);

    m_lastDeltaTime = fElapsedTimeSeconds;

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

    const bool enableAnimations = m_ui.EnableAnimations && m_ui.RealtimeMode;
    const bool enableAnimationUpdate = enableAnimations || m_ui.ResetAccumulation;

    if (m_sampleGame) m_sampleGame->Tick(fElapsedTimeSeconds, enableAnimations);

    if (m_toneMappingPass)
        m_toneMappingPass->AdvanceFrame(fElapsedTimeSeconds);

    if (IsSceneLoaded() && enableAnimationUpdate)
    {
        if (enableAnimations)
            m_sceneTime += fElapsedTimeSeconds;
        if (m_sampleGame && m_sampleGame->IsInitialized())
            m_sceneTime = m_sampleGame->GetGameTime();

        for (const auto& anim : m_scene->GetSceneGraph()->GetAnimations())
        {
            double cutLeft = 0.0; double cutRight = 0.0;
            // if (anim->GetName() == "Take 001") // special hack for mesh drone anim - TODO: fix properly in future
            // { cutLeft = 0.333; cutRight = 0.0; }
            anim->Apply((float)fmod(m_sceneTime+cutLeft, anim->GetDuration()-cutLeft-cutRight));
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

    if (m_sampleGame) m_sampleGame->TickCamera(fElapsedTimeSeconds, m_camera);

    if (m_ui.CameraAntiRRSleepJitter>0)
    {
        float off = 0.05f * ((m_frameIndex%2)?(-m_ui.CameraAntiRRSleepJitter):(m_ui.CameraAntiRRSleepJitter));

        float3 dir = m_camera.GetDir();
        float3 right = normalize(cross(dir, m_camera.GetUp()));
        affine3 rot = rotation(right, off);
        dir = rot.transformVector(dir);

        m_camera.LookTo( m_camera.GetPosition(), dir, m_camera.GetUp() );
    }

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

    m_captureScriptManager->PostAnim();


    double frameTime = GetDeviceManager()->GetAverageFrameTimeSeconds();
    if (frameTime > 0.0)
    {
#if DONUT_WITH_STREAMLINE
        if (m_ui.DLSSFGMultiplier != 1)
            m_fpsInfo = StringFormat("%.3f ms/%d-frames* (%.1f FPS*) *DLSS-G", frameTime * 1e3, m_ui.DLSSFGMultiplier, m_ui.DLSSFGMultiplier / frameTime);
        else
#endif
            m_fpsInfo = StringFormat("%.3f ms/frame (%.1f FPS)", frameTime * 1e3, 1.0 / frameTime);
    }

    // Window title
    std::string extraInfo = ", " + m_fpsInfo + ", " + m_currentSceneName + ", " + GetResolutionInfo() + ", (L: " + std::to_string(m_scene->GetSceneGraph()->GetLights().size()) + ", MAT: " + std::to_string(m_scene->GetSceneGraph()->GetMaterials().size())
        + ", MESH: " + std::to_string(m_scene->GetSceneGraph()->GetMeshes().size()) + ", I: " + std::to_string(m_scene->GetSceneGraph()->GetMeshInstances().size()) + ", SI: " + std::to_string(m_scene->GetSceneGraph()->GetSkinnedMeshInstances().size())
        //+ ", AvgLum: " + std::to_string((m_renderTargets!=nullptr)?(m_renderTargets->AvgLuminanceLastCaptured):(0.0f))
#if ENABLE_DEBUG_VIZUALISATIONS
        + ", ENABLE_DEBUG_VIZUALISATIONS: 1"
#endif
        + ")";

    std::string windowTitle = StringFormat( "%s %s", GetWindowTitle(), g_versionString );
    GetDeviceManager()->SetInformativeWindowTitle(windowTitle.c_str(), false, extraInfo.c_str());
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

std::string Sample::GetCurrentCameraPosDirUp() const
{
    auto toString = [](const float3 & val)    { return std::to_string(val.x) + "," + std::to_string(val.y) + "," + std::to_string(val.z); };
    return toString(m_camera.GetPosition()) + "," + toString(m_camera.GetDir()) + "," + toString(m_camera.GetUp());
}

bool Sample::SetCurrentCameraPosDirUp(const std::string & val)
{
	if (val=="")
        return false;

    // Extracts a 3-element array following a given JSON key, e.g. `"position": [-17.58, 1.215, 15]`.
    auto ParseJsonArray3 = []( const std::string & s, const char * key, float3 & out ) -> bool
    {
        size_t k = s.find( key );
        if (k == std::string::npos)
            return false;
        size_t lb = s.find( '[', k );
        if (lb == std::string::npos)
            return false;
        size_t rb = s.find( ']', lb );
        if (rb == std::string::npos)
            return false;
        std::string inner = s.substr( lb + 1, rb - lb - 1 );
        return ParseFloat3Consume( inner, out );
    };

    bool ok = true;
    float3 worldPos, worldDir, worldUp;

    // If the string looks like the JSON-style "position"/"direction" format, parse it that way;
    // otherwise fall back to the plain "px, py, pz, dx, dy, dz, ux, uy, uz" format.
    if (val.find( "\"position\"" ) != std::string::npos && val.find( "\"direction\"" ) != std::string::npos)
    {
        worldUp = float3( 0, 1, 0 ); // default up if not explicitly provided
        ok &= ParseJsonArray3( val, "\"position\"", worldPos );
        ok &= ParseJsonArray3( val, "\"direction\"", worldDir );
        ParseJsonArray3( val, "\"up\"", worldUp ); // optional
    }
    else
    {
        std::string temp = val;
        ok &= ParseFloat3Consume( temp, worldPos );
        ok &= ParseFloat3Consume( temp, worldDir );
        ok &= ParseFloat3Consume( temp, worldUp );
    }

    if (ok)
        m_camera.LookAt( worldPos, worldPos + worldDir, worldUp );
    return ok;
}

std::string Sample::GetSceneCameraName(uint index) const
{
    // Guarded on size rather than trusting the caller's bound: GetSceneCameraCount
    // counts free flight as well, so it is one larger than GetCameras().
    if (m_scene != nullptr)
    {
        const auto& cameras = m_scene->GetSceneGraph()->GetCameras();
        if (index < cameras.size() && !cameras[index]->GetName().empty())
            return cameras[index]->GetName();
    }
    return "Scene cam " + std::to_string(index);
}

void Sample::SaveCurrentCamera() const
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

void Sample::GetPTPipelineGlobalMacros(std::vector<donut::engine::ShaderMacro> & macros)
{
    macros.clear();

    assert(!m_ui.NVAPIHitObjectExtension || !m_ui.DXHitObjectExtension);

    macros.push_back({ "ENABLE_DEBUG_SURFACE_VIZ",  (m_ui.DebugView != DebugViewType::Disabled)?("1"):("0") });
    macros.push_back({ "ENABLE_DEBUG_LINES_VIZ",    (m_ui.ShowDebugLines)?("1"):("0") });

    macros.push_back({ "USE_NVAPI_HIT_OBJECT_EXTENSION", (m_ui.NVAPIHitObjectExtension)?("1"):("0") });
    macros.push_back({ "USE_NVAPI_REORDER_THREADS", (m_ui.NVAPIHitObjectExtension && m_ui.NVAPIReorderThreads)?("1"):("0") });

    macros.push_back({ "USE_DX_HIT_OBJECT_EXTENSION", (m_ui.DXHitObjectExtension) ? ("1") : ("0") });
    macros.push_back({ "USE_DX_MAYBE_REORDER_THREADS", (m_ui.DXHitObjectExtension && m_ui.DXMaybeReorderThreads) ? ("1") : ("0") });

    macros.push_back({ "PT_ENABLE_RUSSIAN_ROULETTE", (m_ui.EnableRussianRoulette) ? ("1") : ("0") });

    macros.push_back({ "PT_NEE_ENABLED", (m_ui.UseNEE)?("1"):("0") });

    macros.push_back({ "PT_USE_RESTIR_DI", (m_ui.ActualUseReSTIRDI()) ? ("1") : ("0") });   // these will match constants.useReSTIRDI but constants are used in other passes too
    macros.push_back({ "PT_USE_RESTIR_GI", (m_ui.ActualUseReSTIRGI()) ? ("1") : ("0") });   // these will match constants.useReSTIRGI but constants are used in other passes too

    macros.push_back({ "RTXPT_USE_APPROXIMATE_MIS", (m_ui.ActualUseApproximateMIS()) ? ("1") : ("0") });

    // It helps performance a lot when these are baked in
    macros.push_back({ "RTXPT_NEE_FULL_SAMPLE_COUNT", std::to_string(m_ui.NEEFullSamples) });
    uint localCandidateSamples = NEEAT::ComputeCandidateSampleLocalCount(m_ui.ActualNEEAT_LocalToGlobalSampleRatio(), m_ui.NEECandidateSamples);
    uint globalCandidateSamples = NEEAT::ComputeCandidateSampleGlobalCount(m_ui.ActualNEEAT_LocalToGlobalSampleRatio(), m_ui.NEECandidateSamples);
    macros.push_back({ "RTXPT_NEE_LOCAL_CANDIDATE_SAMPLE_COUNT", std::to_string(localCandidateSamples) });
    macros.push_back({ "RTXPT_NEE_GLOBAL_CANDIDATE_SAMPLE_COUNT", std::to_string(globalCandidateSamples) });
    macros.push_back({ "RTXPT_NEE_TOTAL_CANDIDATE_SAMPLE_COUNT", std::to_string(m_ui.NEECandidateSamples) });

    macros.push_back({ "RTXPT_DISABLE_SER_TERMINATION_HINT", (m_ui.DbgDisableSERTerminationHint)?("1"):("0") });
    macros.push_back({ "RTXPT_DISCARD_NON_NEE_LIGHTING", (m_ui.DbgDiscardNonNEELighting) ? ("1") : ("0") });
    macros.push_back({ "RTXPT_DISCARD_NEE_LIGHTING", (m_ui.DbgDiscardNEELighting) ? ("1") : ("0") });
    
    macros.push_back({ "RTXPT_FIREFLY_FILTER", (m_ui.ActualFireflyFilterEnabled()) ? ("1") : ("0") });

    macros.push_back({ "RTXPT_ACTIVE_DENOISING_LAYER_COUNT", std::to_string(m_ui.ActualDenoisingLayersActiveCount()) });

    macros.push_back({ "RTXPT_NESTED_DIELECTRICS_QUALITY", std::to_string(m_ui.NestedDielectricsQuality) });

    macros.push_back({ "RTXPT_LP_TYPES_USE_16BIT_PRECISION", (m_ui.UseFp16Types) ? ("1") : ("0") });

    macros.push_back({ "RTXPT_ENABLE_LOW_DISCREPANCY_SAMPLER_FOR_BSDF", (m_ui.EnableLDSamplerForBSDF) ? ("1") : ("0") });

    m_lightsBaker->SetGlobalShaderMacros(macros);
    if (m_ommBaker != nullptr)
        m_ommBaker->SetGlobalShaderMacros(macros);

    // Let the derived sample contribute its own global macros.
    GetCustomPTPipelineGlobalMacros(macros);
}

extern HitGroupInfo ComputeSubInstanceHitGroupInfo(const PTMaterial& material);

bool Sample::CreatePTPipeline(engine::ShaderFactory& shaderFactory)
{
    {
        std::vector<donut::engine::ShaderMacro> shaderMacros;
		// shaderMacros.push_back(donut::engine::ShaderMacro({ "USE_RTXDI", "0" }));
        m_exportVBufferCS = m_shaderFactory->CreateShader("app/ProcessingPasses/ExportVisibilityBuffer.hlsl", "main", &shaderMacros, nvrhi::ShaderType::Compute);
        nvrhi::ComputePipelineDesc pipelineDesc;
		pipelineDesc.bindingLayouts = { m_bindingLayout, m_bindlessLayout };
		pipelineDesc.CS = m_exportVBufferCS;
        m_exportVBufferPSO = GetDevice()->createComputePipeline(pipelineDesc);
    }

    {
        // Offline-compiled (see shaders.cfg) so a solution build catches errors in it, and recreated
        // here so it follows the shader hot-reload path like every other pass.
        EnsureGPUTLASResources();
        m_tlasCullCS = shaderFactory.CreateShader("app/Shaders/TLASCull.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);
        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.bindingLayouts = { m_tlasCullBindingLayout };
        pipelineDesc.CS = m_tlasCullCS;
        m_tlasCullPSO = GetDevice()->createComputePipeline(pipelineDesc);
    }

    return true;
}

void Sample::CreateBlases(nvrhi::ICommandList* commandList)
{
    const bool poolAvailable = m_blasCompactor->IsAvailable();

    std::vector<nvrhi::rt::AccelStructDesc> pooledDescs;
    std::vector<MeshInfo*>                  pooledMeshes;

    for (const std::shared_ptr<MeshInfo>& mesh : m_scene->GetSceneGraph()->GetMeshes())
    {
        if (mesh->isSkinPrototype) //buffers->hasAttribute(engine::VertexAttribute::JointWeights))
            continue; // skip the skinning prototypes

        bvh::Config cfg = { .excludeTransmissive = m_ui.AS.ExcludeTransmissive };

        nvrhi::rt::AccelStructDesc blasDesc = bvh::GetMeshBlasDesc(cfg , *mesh, nullptr, false);
        // 12 bits, not 13, and the reason is RTXDI - not the long-gone HitInfo: the light-prep task packs geometryIndex
        // into the low 12 bits of instanceAndGeometryIndex (RTXDI/ShaderParameters.h, PrepareLightsPass.cpp:259).
        assert((int)blasDesc.bottomLevelGeometries.size() < (1 << 12)); // RTXDI instanceAndGeometryIndex: low 12 bits are geometryIndex

        // A skinned mesh is refit in place every frame, so it carries AllowUpdate rather than
        // AllowCompaction and UpdateSkinnedBLASs hands it straight back to NVRHI - it cannot be pooled.
        // This fork and the build-flag choice in GetMeshBlasDesc share bvh::IsSkinnedMesh so they cannot
        // drift apart.
        if (poolAvailable && !bvh::IsSkinnedMesh(*mesh))
        {
            pooledMeshes.push_back(mesh.get());
            pooledDescs.push_back(std::move(blasDesc));
            continue;
        }

        nvrhi::rt::AccelStructHandle as = GetDevice()->createAccelStruct(blasDesc);

        nvrhi::utils::BuildBottomLevelAccelStruct(commandList, as, blasDesc);

        mesh->accelStruct = as;
    }

    if (pooledDescs.empty())
        return;

    const std::vector<nvrhi::rt::AccelStructHandle> pooledResults = m_blasCompactor->BuildAll(commandList, pooledDescs);
    for (size_t i = 0; i < pooledResults.size(); i++)
    {
        if (pooledResults[i] != nullptr)
        {
            // Pooled implies not skinned. This is the one place the two decisions can come apart, so it
            // is the one worth checking: the NVRHI entry points that would misread a pooled handle are
            // only ever reached from the skinned instance list.
            assert(!bvh::IsSkinnedMesh(*pooledMeshes[i]));
            pooledMeshes[i]->accelStruct = pooledResults[i];
            continue;
        }

        // Pooling failed for this mesh; an ordinary acceleration structure still renders correctly.
        nvrhi::rt::AccelStructHandle as = GetDevice()->createAccelStruct(pooledDescs[i]);
        nvrhi::utils::BuildBottomLevelAccelStruct(commandList, as, pooledDescs[i]);
        pooledMeshes[i]->accelStruct = as;
    }

    const bvh::BlasPool::Stats stats = m_blasPool->GetStats();
    donut::log::info("[BlasPool] %u blocks, %.1f MB reserved, %.1f MB used, over %u build batches",
        stats.BlockCount, stats.ReservedBytes / (1024.0 * 1024.0), stats.UsedBytes / (1024.0 * 1024.0),
        m_blasCompactor->GetLastBatchCount());
}

// Drops every BLAS the scene holds, then the blocks behind the pooled ones. The two have to happen in
// that order and in one place, because the pool's blocks are only safe to drop once nothing is still
// suballocated out of them - which is exactly what ReleaseAll asserts.
void Sample::ReleaseBlases()
{
    if (m_scene != nullptr)
    {
        for (const std::shared_ptr<MeshInfo>& _mesh : m_scene->GetSceneGraph()->GetMeshes())
        {
            assert(std::dynamic_pointer_cast<MeshInfoEx>(_mesh) != nullptr);
            const std::shared_ptr<MeshInfoEx>& mesh = std::static_pointer_cast<MeshInfoEx>(_mesh);
            mesh->accelStruct = nullptr;
            mesh->AccelStructOMM = nullptr;
        }
    }

    // Every pooled handle was just released, so the blocks backing them can go too. ReleaseAll asserts
    // nothing is still allocated - the tripwire for a pooled BLAS outliving its pool.
    if (m_blasPool != nullptr)
        m_blasPool->ReleaseAll();
}

void Sample::UploadSubInstanceData(nvrhi::ICommandList* commandList)
{
    ZoneScoped;

    assert(m_subInstanceCount == m_subInstanceData.size());
    // upload data to GPU buffer
    commandList->writeBuffer(m_subInstanceBuffer, m_subInstanceData.data(), m_subInstanceData.size() * sizeof(SubInstanceData));
}

void Sample::CreateTlas(nvrhi::ICommandList* commandList)
{
    nvrhi::rt::AccelStructDesc tlasDesc;
    tlasDesc.isTopLevel = true;
    tlasDesc.topLevelMaxInstances = m_scene->GetSceneGraph()->GetMeshInstances().size();
    tlasDesc.buildFlags = nvrhi::rt::AccelStructBuildFlags::PreferFastTrace;
    // NOTE: the old assert here (topLevelMaxInstances < (1<<15), "16 bits for the identifier in the HitInfo") was stale -
    // both HitInfo and GeometryInstanceID are gone from the codebase, and the live limits are asserted where the counts
    // are finalised (see m_subInstanceCount in CreateAccelStructs). Instance count itself has no 15-bit constraint.
    m_topLevelAS = GetDevice()->createAccelStruct(tlasDesc);

    // setup subInstances (entry is per geometry per instance) - some of it might require rebuild at runtime in more realistic scenarios
    {
        // figure out the required number
        m_subInstanceCount = 0;
        for (const auto& instance : m_scene->GetSceneGraph()->GetMeshInstances())
            m_subInstanceCount += (uint)instance->GetMesh()->geometries.size();

        // DXR packs both of these into 24-bit fields (see nvrhi::rt::InstanceDesc: 'instanceID : 24' and
        // 'instanceContributionToHitGroupIndex : 24'), and BuildTLAS assigns sub-instance values to both. Overflow is a
        // silent truncation, not an error, so assert here where the count is finalised.
        assert( m_subInstanceCount < (1u << 24) ); // DXR instanceID / instanceContributionToHitGroupIndex are 24-bit

        // RTXDI's light-prep task packs the instance index into 19 bits (bits 12..30; bit 31 is TASK_PRIMITIVE_LIGHT_BIT)
        // - see instanceAndGeometryIndex in RTXDI/ShaderParameters.h and PrepareLightsPass.cpp.
        assert( m_scene->GetSceneGraph()->GetMeshInstances().size() < (1u << 19) ); // RTXDI instanceAndGeometryIndex: 19 bits of instance index
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
        // assign() rather than clear()+insert(): one call, and the canonical "discard and fill with N copies" form.
        // Note Default() is not redundant init - it sets AnalyticProxyLightIndex, which only ProcessEmissiveGeometry
        // writes, and that does not run when NEE-AT light sampling is disabled.
        m_subInstanceData.assign( m_subInstanceCount, SubInstanceData::Default() );
    }
}

void Sample::CreateAccelStructs(nvrhi::ICommandList* commandList)
{
    // The OMM queue builds its BLASes over the following frames and needs the same pool, so both have
    // to exist before it is queued rather than being created lazily inside CreateBlases.
    if (m_blasPool == nullptr)
        m_blasPool = std::make_unique<bvh::BlasPool>(GetDevice());
    if (m_blasCompactor == nullptr)
        m_blasCompactor = std::make_unique<bvh::BlasCompactor>(GetDevice(), *m_blasPool);

    if (m_ommBaker)
    {
        m_ommBaker->SetBlasCompactor(m_blasCompactor.get());
        m_ommBaker->CreateOpacityMicromaps(*m_scene);
    }
    CreateBlases(commandList);
    CreateTlas(commandList);
    m_tlasInstanceDataDirty = true; // BLAS handles just changed - the GPU cull pass's address table is stale
}

void Sample::RecreateAccelStructs(nvrhi::ICommandList* commandList)
{
    // If the subInstanceData or BLAS build input data changes we trigger a full update here
    // could be made more efficient by only rebuilding the geometry in question,
    // or split the BLAS and subInstanceData updates
    if (m_ui.AccelerationStructRebuildRequested)
    {
        m_ui.AccelerationStructRebuildRequested = false;
        m_ui.ResetAccumulation = true;

        GetDevice()->waitForIdle();

        InvalidateBindingSet();
        m_topLevelAS = nullptr;
        m_bindingCache->Clear();    // CreateTlas below replaces m_subInstanceBuffer, which cached binding sets reference

        for (const std::shared_ptr<MeshInfo>& _mesh : m_scene->GetSceneGraph()->GetMeshes())
        {
            assert(std::dynamic_pointer_cast<MeshInfoEx>(_mesh) != nullptr);
            const std::shared_ptr<MeshInfoEx>& mesh = std::static_pointer_cast<MeshInfoEx>(_mesh);
            mesh->OpacityMicroMaps.clear();
            mesh->DebugData = nullptr;
            mesh->MarkDebugDataDirty();
        }

        ReleaseBlases();

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
    ZoneScoped;

    // Transition all the buffers to their necessary states before building the BLAS'es to allow BLAS batching
    for (const auto& skinnedInstance : m_scene->GetSceneGraph()->GetSkinnedMeshInstances())
        commandList->setBufferState(skinnedInstance->GetMesh()->buffers->vertexBuffer, nvrhi::ResourceStates::NonPixelShaderResource);
    commandList->commitBarriers();
}

void Sample::UpdateSkinnedBLASs(nvrhi::ICommandList* commandList, uint32_t frameIndex) const
{
    ZoneScoped;

    GpuProfiler::Begin(commandList, "Skinned BLAS Updates");

    // Transition all the buffers to their necessary states before building the BLAS'es to allow BLAS batching
    for (const auto& skinnedInstance : m_scene->GetSceneGraph()->GetSkinnedMeshInstances())
    {
        if (skinnedInstance->GetLastUpdateFrameIndex() < frameIndex)
            continue;

        commandList->setAccelStructState(skinnedInstance->GetMesh()->accelStruct, nvrhi::ResourceStates::AccelStructWrite);
        commandList->setBufferState(skinnedInstance->GetMesh()->buffers->vertexBuffer, nvrhi::ResourceStates::AccelStructBuildInput);
    }
    commandList->commitBarriers();

    // Now update the BLAS'es
    for (const auto& skinnedInstance : m_scene->GetSceneGraph()->GetSkinnedMeshInstances())
    {
        if (skinnedInstance->GetLastUpdateFrameIndex() < frameIndex)
            continue;

        bvh::Config cfg = { .excludeTransmissive = m_ui.AS.ExcludeTransmissive };

        nvrhi::rt::AccelStructDesc blasDesc = bvh::GetMeshBlasDesc(cfg, *skinnedInstance->GetMesh(), nullptr, true);

        nvrhi::utils::BuildBottomLevelAccelStruct(commandList, skinnedInstance->GetMesh()->accelStruct, blasDesc);
    }

    // The TLAS build reads the BLASes through instance descs in a buffer, so nvrhi can't order these refits before it;
    // committed together with TransitionMeshBuffersToReadOnly's barriers
    for (const auto& skinnedInstance : m_scene->GetSceneGraph()->GetSkinnedMeshInstances())
        if (skinnedInstance->GetLastUpdateFrameIndex() >= frameIndex)
            commandList->setAccelStructState(skinnedInstance->GetMesh()->accelStruct, nvrhi::ResourceStates::AccelStructBuildBlas);
    GpuProfiler::End(commandList);
}

void Sample::BuildTLAS(nvrhi::ICommandList* commandList) const
{
    ZoneScoped;

    // Reuse a persistent scratch buffer instead of building a fresh vector every frame. PostRender() clear()s it, which
    // keeps the capacity, so the resize() below only allocates when the instance count grows past anything seen so far.
    std::vector<nvrhi::rt::InstanceDesc> & instances = m_tlasInstanceDescs;
    // assert( instances.size() == 0 ); // should have been clean here <- no longer the case, saving perf on not cleaning & re-constructing
    const auto & meshInstances = m_scene->GetSceneGraph()->GetMeshInstances();
    // Cached node pointers, index-matched with meshInstances. Avoids MeshInstance::GetNode()'s weak_ptr lock (an atomic
    // CAS loop plus a decrement) per instance per frame.
    const std::vector<donut::engine::SceneGraphNode*> & instanceNodes = m_sceneRenderCache.GetMeshInstanceNodes();
    const size_t count = meshInstances.size();
    assert( instanceNodes.size() == count );

    // resize() rather than reserve()+push_back(): chunks write disjoint indices with no synchronisation.
    instances.resize( count );

    // Hoisted out of the loop - all of these are loop-invariant. This matters more now than it did serially: left
    // inside, every worker would re-chase m_ommBaker->UIData() per instance, hammering the same shared cache lines.
    const bool ommEnabled           = m_ommBaker != nullptr;
    const bool ommDebugViewEnabled  = ommEnabled && m_ommBaker->UIData().DebugView != OpacityMicroMapDebugView::Disabled;
    const bool ommUIEnable          = ommEnabled && m_ommBaker->UIData().Enable;
    const bool ommOnlyOMMs          = ommEnabled && m_ommBaker->UIData().OnlyOMMs;
    const bool ommForce2State       = ommEnabled && m_ommBaker->UIData().Force2State;
    // ommDebugViewEnabled must do two things: use a BLAS without OMMs and disable all alpha testing.
    // This may sound a bit counter intuitive, the goal is to intersect micro-triangles marked as transparent without them actually being treated as such.
    const bool forceOpaque          = ommDebugViewEnabled || m_ui.AS.ForceOpaque;

    auto buildRange = [&]( size_t begin, size_t end )
    {
        for( size_t instanceIndex = begin; instanceIndex < end; instanceIndex++ )
        {
            const auto & instance = meshInstances[instanceIndex];

            // GetMesh() returns a 'const shared_ptr&' so .get() is free, whereas static_pointer_cast would construct a
            // new shared_ptr - an atomic incref+decref per instance - just to do an unchecked downcast.
            MeshInfoEx* mesh = static_cast<MeshInfoEx*>(instance->GetMesh().get());

            const bool hasAttachementOMM    = ommEnabled && mesh->AccelStructOMM.Get() != nullptr;
            const bool useOmmBLAS           = ommUIEnable && hasAttachementOMM && !forceOpaque;

            nvrhi::rt::InstanceDesc instanceDesc;
            // Device address, not the BLAS pointer.
            const nvrhi::rt::IAccelStruct* blas = useOmmBLAS ? mesh->AccelStructOMM.Get() : mesh->accelStruct.Get();
            instanceDesc.blasDeviceAddress = (blas != nullptr) ? blas->getDeviceAddress() : 0;
            instanceDesc.instanceMask = (ommOnlyOMMs && !hasAttachementOMM) ? 0 : 1;
            instanceDesc.instanceID = instance->GetGeometryInstanceIndex();
            // Was a running sum ('subInstanceCount') which serialised the loop. The old per-iteration assert stated that
            // the sum always equalled GetGeometryInstanceIndex() anyway, so use that directly - the invariant it was
            // asserting is re-checked serially in Debug below.
            instanceDesc.instanceContributionToHitGroupIndex = instance->GetGeometryInstanceIndex();
            instanceDesc.flags = (ommForce2State) ? nvrhi::rt::InstanceFlags::ForceOMM2State : nvrhi::rt::InstanceFlags::None;
            if (forceOpaque)
                instanceDesc.flags = (nvrhi::rt::InstanceFlags)((uint32_t)instanceDesc.flags | (uint32_t)nvrhi::rt::InstanceFlags::ForceOpaque);

            donut::engine::SceneGraphNode * node = instanceNodes[instanceIndex];
            assert(node);
            dm::affineToColumnMajor(node->GetLocalToWorldTransformFloat(), instanceDesc.transform);

            instances[instanceIndex] = instanceDesc;
        }
    };

    ParallelForChunked( m_frameWorkerPool, count, buildRange );

#ifdef _DEBUG
    {   // the parallel loop above uses GetGeometryInstanceIndex() directly instead of a running sum; verify that the
        // indices really are contiguous and total m_subInstanceCount, which is what the old inline asserts checked
        uint subInstanceCountCheck = 0;
        for( size_t i = 0; i < count; i++ )
        {
            assert( subInstanceCountCheck == (uint)meshInstances[i]->GetGeometryInstanceIndex() );
            subInstanceCountCheck += (uint)static_cast<MeshInfoEx*>(meshInstances[i]->GetMesh().get())->geometries.size();
        }
        assert( m_subInstanceCount == subInstanceCountCheck );
    }
#endif

    {
        RTXPT_GPU_SCOPE(commandList, "TLAS Update");
        ZoneScopedN("buildTopLevelAccelStruct");
        if (!instances.empty())
            commandList->writeBuffer(m_tlasInstanceBuffer, instances.data(), instances.size() * sizeof(nvrhi::rt::InstanceDesc));
        commandList->setBufferState(m_tlasInstanceBuffer, nvrhi::ResourceStates::AccelStructBuildInput);
        commandList->setAccelStructState(m_topLevelAS, nvrhi::ResourceStates::AccelStructWrite);    // same batch, else the build adds its own
        commandList->commitBarriers();
        commandList->buildTopLevelAccelStructFromBuffer(m_topLevelAS, m_tlasInstanceBuffer, 0, instances.size(), nvrhi::rt::AccelStructBuildFlags::AllowEmptyInstances);
    }
}

// Static per-mesh bounds and per-instance BLAS/OMM-BLAS device addresses for the GPU cull pass.
// Rebuilt when the pool generation moves or a BLAS handle changes, never per frame. Pooled BLASes do
// relocate - they are compacted into pool blocks as they are built, and OMM BLASes land over the
// frames after a rebuild - so the generation counter, bumped inside BlasPool on every allocation and
// free, keeps this table from pointing at stale addresses.
// nvrhi::rt::InstanceDesc is written straight into the buffer D3D12 reads as an array of
// D3D12_RAYTRACING_INSTANCE_DESC. Shaders/TLASCull.hlsl hand-packs the
// same layout on the GPU side.
#if DONUT_WITH_DX12
static_assert(sizeof(nvrhi::rt::InstanceDesc) == sizeof(D3D12_RAYTRACING_INSTANCE_DESC));
static_assert(offsetof(nvrhi::rt::InstanceDesc, transform) == offsetof(D3D12_RAYTRACING_INSTANCE_DESC, Transform));
static_assert(offsetof(nvrhi::rt::InstanceDesc, bottomLevelAS) == offsetof(D3D12_RAYTRACING_INSTANCE_DESC, AccelerationStructure));
#endif

void Sample::EnsureTLASInstanceBuffer(size_t instanceCount)
{
    const size_t byteSize = sizeof(nvrhi::rt::InstanceDesc) * std::max<size_t>(1, instanceCount);
    if (m_tlasInstanceBuffer != nullptr && m_tlasInstanceBuffer->getDesc().byteSize == byteSize)
        return;

    // Filled either by the cull dispatch or by BuildTLAS on the CPU, and consumed by
    // buildTopLevelAccelStructFromBuffer; 64 bytes/instance == sizeof(nvrhi::rt::InstanceDesc).
    nvrhi::BufferDesc tlasInstanceDesc;
    tlasInstanceDesc.byteSize = byteSize;
    tlasInstanceDesc.structStride = sizeof(nvrhi::rt::InstanceDesc);
    tlasInstanceDesc.debugName = "TLASInstanceDescs";
    tlasInstanceDesc.canHaveUAVs = true;
    tlasInstanceDesc.isAccelStructBuildInput = true;
    tlasInstanceDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    tlasInstanceDesc.keepInitialState = true;
    m_tlasInstanceBuffer = GetDevice()->createBuffer(tlasInstanceDesc);
}

void Sample::UploadTLASInstanceData(nvrhi::ICommandList* commandList)
{
    ZoneScoped;

    const auto & meshes = m_scene->GetSceneGraph()->GetMeshes();
    const auto & instances = m_scene->GetSceneGraph()->GetMeshInstances();

    std::vector<TLASObjectBounds> meshBounds(meshes.size());
    for (const auto & mesh : meshes)
    {
        assert((size_t)mesh->globalMeshIndex < meshBounds.size());
        // Designated initializers, not positional - TLASObjectBounds interleaves padding between Min and
        // Max to satisfy HLSL's vector packing rules, so positional init would fill the wrong fields.
        meshBounds[mesh->globalMeshIndex] = TLASObjectBounds{ .Min = mesh->objectSpaceBounds.m_mins, .Max = mesh->objectSpaceBounds.m_maxs };
    }

    std::vector<TLASInstanceData> instanceData(instances.size());
    for (size_t i = 0; i < instances.size(); i++)
    {
        // GetMesh() returns a 'const shared_ptr&', so .get() is free where static_pointer_cast would
        // construct a new shared_ptr just to do an unchecked downcast.
        MeshInfoEx * mesh = static_cast<MeshInfoEx*>(instances[i]->GetMesh().get());
        instanceData[i].BlasAddress    = (mesh->accelStruct.Get()   != nullptr) ? mesh->accelStruct->getDeviceAddress()   : 0;
        instanceData[i].OmmBlasAddress = (mesh->AccelStructOMM.Get()!= nullptr) ? mesh->AccelStructOMM->getDeviceAddress(): 0;
        instanceData[i].MeshIndex      = (uint32_t)mesh->globalMeshIndex;
        instanceData[i].Flags          = 0;
    }
    // Marked from the skinned list rather than a dynamic_cast per instance.
    for (const auto & skinnedInstance : m_scene->GetSceneGraph()->GetSkinnedMeshInstances())
    {
        const int index = skinnedInstance->GetInstanceIndex();
        assert(index >= 0 && (size_t)index < instanceData.size());
        instanceData[index].Flags |= kTLASInstanceData_IsSkinned;
    }

    auto createStructuredBuffer = [this](size_t elementCount, size_t elementStride, const char* debugName)
    {
        nvrhi::BufferDesc bufferDesc;
        bufferDesc.byteSize = elementStride * std::max<size_t>(1, elementCount);
        bufferDesc.structStride = (uint32_t)elementStride;
        bufferDesc.debugName = debugName;
        bufferDesc.initialState = nvrhi::ResourceStates::Common;
        bufferDesc.keepInitialState = true;
        return GetDevice()->createBuffer(bufferDesc);
    };

    m_tlasInstanceCount = (uint32_t)instances.size();

    // Sized per buffer rather than off one shared "did anything change" flag: mesh count and instance
    // count move independently, so keying the bounds buffer off the instance count would leave it
    // undersized whenever a scene edit changed only the mesh list.
    auto needsResize = [](const nvrhi::BufferHandle & buffer, size_t requiredBytes)
    {
        return buffer == nullptr || buffer->getDesc().byteSize != requiredBytes;
    };

    const size_t meshBoundsBytes   = sizeof(TLASObjectBounds) * std::max<size_t>(1, meshBounds.size());
    const size_t instanceDataBytes = sizeof(TLASInstanceData) * std::max<size_t>(1, instanceData.size());

    if (needsResize(m_tlasMeshBoundsBuffer, meshBoundsBytes))
        m_tlasMeshBoundsBuffer = createStructuredBuffer(meshBounds.size(), sizeof(TLASObjectBounds), "TLASMeshBounds");
    if (needsResize(m_tlasInstanceDataBuffer, instanceDataBytes))
        m_tlasInstanceDataBuffer = createStructuredBuffer(instanceData.size(), sizeof(TLASInstanceData), "TLASInstanceData");

    EnsureTLASInstanceBuffer(instances.size());

    if (!meshBounds.empty())
        commandList->writeBuffer(m_tlasMeshBoundsBuffer, meshBounds.data(), meshBounds.size() * sizeof(TLASObjectBounds));
    if (!instanceData.empty())
        commandList->writeBuffer(m_tlasInstanceDataBuffer, instanceData.data(), instanceData.size() * sizeof(TLASInstanceData));

    m_tlasInstanceDataDirty = false;
}

// Creates the cull pass's layout, constants and stats buffers once, and rebuilds the binding set
// whenever a bound buffer handle changed. Returns false while anything it needs is still missing.
bool Sample::EnsureGPUTLASResources()
{
    if (!m_tlasCullBindingLayout)
    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0),  // t_InstanceData
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1),  // t_InstanceAccel
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2),  // t_MeshBounds
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0),  // u_TLASInstances
        };
        m_tlasCullBindingLayout = GetDevice()->createBindingLayout(layoutDesc);
    }

    if (!m_tlasCullCB)
    {
        nvrhi::BufferDesc cbDesc;
        cbDesc.byteSize = sizeof(TLASCullConstants);
        cbDesc.debugName = "TLASCullConstants";
        cbDesc.isConstantBuffer = true;
        cbDesc.isVolatile = true;
        cbDesc.maxVersions = 16;
        m_tlasCullCB = GetDevice()->createBuffer(cbDesc);
    }

    if (!m_tlasInstanceBuffer || !m_tlasInstanceDataBuffer || !m_tlasMeshBoundsBuffer
        || !m_scene || !m_scene->GetInstanceBuffer())
        return false;

    const bool handlesChanged = !m_tlasCullBindingSet
        || m_tlasCullBoundInstanceData  != m_scene->GetInstanceBuffer()
        || m_tlasCullBoundInstanceAccel != m_tlasInstanceDataBuffer.Get()
        || m_tlasCullBoundMeshBounds    != m_tlasMeshBoundsBuffer.Get()
        || m_tlasCullBoundOutput        != m_tlasInstanceBuffer.Get();

    if (handlesChanged)
    {
        nvrhi::BindingSetDesc setDesc;
        setDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_tlasCullCB),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_scene->GetInstanceBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_tlasInstanceDataBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_tlasMeshBoundsBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_tlasInstanceBuffer),
        };
        m_tlasCullBindingSet = GetDevice()->createBindingSet(setDesc, m_tlasCullBindingLayout);
        m_tlasCullBoundInstanceData  = m_scene->GetInstanceBuffer();
        m_tlasCullBoundInstanceAccel = m_tlasInstanceDataBuffer.Get();
        m_tlasCullBoundMeshBounds    = m_tlasMeshBoundsBuffer.Get();
        m_tlasCullBoundOutput        = m_tlasInstanceBuffer.Get();
    }

    return m_tlasCullBindingSet != nullptr && m_tlasCullPSO != nullptr;
}

bool Sample::BuildTLASGPU(nvrhi::ICommandList* commandList)
{
    ZoneScoped;

    if (m_tlasInstanceCount == 0 || !EnsureGPUTLASResources())
        return false;

    RTXPT_GPU_SCOPE(commandList, "TLAS Cull+Build");

    // Culling makes the traced scene depend on where the camera is looking, which an accumulating
    // reference image must never see - rays are not bounded by the view frustum, so a dropped instance
    // silently removes its shadows, reflections and GI contribution.
    const bool cullEnable = m_ui.AS.GPUTLASCull && m_ui.RealtimeMode;

    TLASCullConstants cullConstants = {};
    if (cullEnable)
    {
        // Side planes only - near/far are dropped deliberately, for the same reason.
        dm::frustum viewFrustum = m_view->GetViewFrustum().grow(m_ui.AS.GPUTLASCullFrustumMargin);
        cullConstants.FrustumPlanes[0] = float4(viewFrustum.leftPlane().normal,   viewFrustum.leftPlane().distance);
        cullConstants.FrustumPlanes[1] = float4(viewFrustum.rightPlane().normal,  viewFrustum.rightPlane().distance);
        cullConstants.FrustumPlanes[2] = float4(viewFrustum.topPlane().normal,    viewFrustum.topPlane().distance);
        cullConstants.FrustumPlanes[3] = float4(viewFrustum.bottomPlane().normal, viewFrustum.bottomPlane().distance);
        cullConstants.CameraPosAndCutoff = float4(m_camera.GetPosition(), m_ui.AS.GPUTLASCullDistanceCutoff);
    }

    // Reproduces BuildTLAS's per-instance OMM / force-opaque / instance-mask selection as per-frame
    // flags; the static half of that lookup already lives in m_tlasInstanceDataBuffer.
    const bool ommDebugViewEnabled = m_ommBaker && m_ommBaker->UIData().DebugView != OpacityMicroMapDebugView::Disabled;
    const bool forceOpaque = ommDebugViewEnabled || m_ui.AS.ForceOpaque;

    cullConstants.Params.x = m_tlasInstanceCount;
    cullConstants.Params.y = cullEnable ? 1u : 0u;
    cullConstants.Params.z = ((m_ommBaker && m_ommBaker->UIData().OnlyOMMs)    ? 1u : 0u)
                           | ((m_ommBaker && m_ommBaker->UIData().Force2State) ? 2u : 0u)
                           | ((m_ommBaker && m_ommBaker->UIData().Enable)      ? 4u : 0u);
    cullConstants.Params.w = forceOpaque ? 1u : 0u;

    commandList->writeBuffer(m_tlasCullCB, &cullConstants, sizeof(cullConstants));

    nvrhi::ComputeState state;
    state.pipeline = m_tlasCullPSO;
    state.bindings = { m_tlasCullBindingSet };
    commandList->setComputeState(state);
    commandList->dispatch(dm::div_ceil(m_tlasInstanceCount, 256u));

    // UAV write -> accel-struct build read: the transition orders the cull's writes on its own. One batch with the TLAS's
    // own transition; a UAV barrier here would merge into a UAV|BuildInput state and make the build add a second batch.
    commandList->setBufferState(m_tlasInstanceBuffer, nvrhi::ResourceStates::AccelStructBuildInput);
    commandList->setAccelStructState(m_topLevelAS, nvrhi::ResourceStates::AccelStructWrite);
    commandList->commitBarriers();

    commandList->buildTopLevelAccelStructFromBuffer(m_topLevelAS, m_tlasInstanceBuffer, 0, m_tlasInstanceCount, nvrhi::rt::AccelStructBuildFlags::AllowEmptyInstances);

    return true;
}

void Sample::UpdateTLAS(nvrhi::ICommandList* commandList)
{
    ZoneScoped;
    EnsureTLASInstanceBuffer(m_scene->GetSceneGraph()->GetMeshInstances().size());
    const uint32_t ommBuildsLeft = m_ommBaker ? m_ommBaker->UIData().BuildsLeftInQueue : 0;
    const uint64_t ommGeneration = m_ommBaker ? m_ommBaker->GetMemoryGeneration() : 0;
    if (ommBuildsLeft != m_tlasInstanceDataOmmBuildsLeft || ommGeneration != m_tlasInstanceDataOmmGeneration)
    {
        m_tlasInstanceDataOmmBuildsLeft = ommBuildsLeft;
        m_tlasInstanceDataOmmGeneration = ommGeneration;
        m_tlasInstanceDataDirty = true;
    }
    if (m_ui.AS.GPUTLASBuild)
    {
        // BLAS addresses move as meshes are pooled, and again as OMM BLASes land over the frames
        // after a rebuild. The pool bumps its generation on every allocation and every free, so
        // comparing it catches both. The OMM queue also invalidates the table for builds
        // that fall back to ordinary NVRHI allocations outside the pool.
        const uint64_t blasGeneration = (m_blasPool != nullptr) ? m_blasPool->GetGeneration() : 0;
        if (m_tlasInstanceDataDirty || blasGeneration != m_tlasInstanceDataBlasGeneration)
        {
            UploadTLASInstanceData(commandList);
            m_tlasInstanceDataBlasGeneration = blasGeneration;
        }

        if (BuildTLASGPU(commandList))
            return;
    }

    BuildTLAS(commandList);
}


void Sample::BackBufferResizing()
{
    ApplicationBase::BackBufferResizing();
    
    GetDevice()->waitForIdle();
    GetDevice()->runGarbageCollection();
    m_bindingCache->Clear();
    m_renderTargets = nullptr;
    m_linesPipeline = nullptr; // the pipeline is based on the framebuffer so needs a reset
    for (int i=0; i < std::size(m_nrd); i++ )
        m_nrd[i] = nullptr;
    if (m_rtxdiPass)
        m_rtxdiPass->Reset();

// NOTE: we're not yet sure if this is necessary to avoid crash with going in/out of fullscreen and FG
#if DONUT_WITH_STREAMLINE
    if (m_ui.DLSSFGOptions.mode == StreamlineInterface::DLSSGMode::eOn || m_ui.ActualDLSSFGMode() == StreamlineInterface::DLSSGMode::eOn) 
    {
        GetDeviceManager()->GetStreamline().CleanupDLSS(false);
        GetDeviceManager()->GetStreamline().CleanupDLSSG(false);

        if (GetDeviceManager()->GetStreamline().IsDLSSGAvailable())
        {
            auto dlssgOptions = StreamlineInterface::DLSSGOptions{};
            StreamlineInterface::DLSSGState state;
            GetDeviceManager()->GetStreamline().GetDLSSGState(state, dlssgOptions);
            m_ui.DLSSFGMultiplier = state.numFramesActuallyPresented;
            m_ui.DLSSFGMaxNumFramesToGenerate = state.numFramesToGenerateMax;

            GetDeviceManager()->GetStreamline().SetDLSSGOptions(dlssgOptions);
            m_ui.DLSSFGOptions = dlssgOptions;
        }
    }
#endif
}

void Sample::CreateRenderPasses( bool& exposureResetRequired, nvrhi::CommandListHandle initializeCommandList )
{
    m_bindingCache->Clear();

    const uint2 screenResolution = {m_renderTargets->OutputColor->getDesc().width, m_renderTargets->OutputColor->getDesc().height};

    m_shaderDebug = std::make_shared<ShaderDebug>(GetDevice(), initializeCommandList, m_shaderFactory, m_CommonPasses);

    if (m_ui.ActualUseRTXDIPasses())
        m_rtxdiPass = std::make_unique<RtxdiPass>(GetDevice(), m_shaderFactory, m_CommonPasses, m_bindlessLayout);
    else
        m_rtxdiPass = nullptr;

    m_accumulationPass = std::make_unique<AccumulationPass>(GetDevice(), m_shaderFactory);
    m_accumulationPass->CreatePipeline();
    m_accumulationPass->CreateBindingSet(m_renderTargets->OutputColor, m_renderTargets->AccumulatedRadiance, m_renderTargets->ProcessedOutputColor);

    // these get re-created every time intentionally, to pick up changes after at-runtime shader recompile
    std::unique_ptr<ToneMappingPass> prevToneMappingPass = std::move(m_toneMappingPass);
    m_toneMappingPass = std::make_unique<ToneMappingPass>(GetDevice(), m_shaderFactory, m_CommonPasses, m_renderTargets->LdrFramebuffer, *m_view, m_renderTargets->OutputColor);
    if (prevToneMappingPass != nullptr)
        m_toneMappingPass->InheritAutoExposure(*prevToneMappingPass);
    m_bloomPass = std::make_unique<BloomPass>(GetDevice(), m_shaderFactory, m_CommonPasses, m_renderTargets->ProcessedOutputFramebuffer, *m_view);
    m_postProcess = std::make_shared<PostProcess>(GetDevice(), m_shaderFactory, m_CommonPasses, m_shaderDebug);

    m_temporalAntiAliasingPass = nullptr;   // recreated on demand by UpdateTemporalAntiAliasingPass

    if (!CreatePTPipeline(*m_shaderFactory))
        { assert(false); }

    if (m_envMapBaker == nullptr)
        m_envMapBaker = std::make_shared<EnvMapBaker>(GetDevice(), m_TextureCache, m_sampleCapabilities.EnvMapRasterPrecompute);
    if (m_lightsBaker == nullptr)
        m_lightsBaker = std::make_shared<LightsBaker>(GetDevice());
    m_envMapBaker->CreateRenderPasses(m_shaderDebug, m_shaderFactory, m_computePipelineBaker);
    m_envMapBaker->GenerateBRDFLUT(initializeCommandList.Get(), *m_bindingCache);  // One-time BRDF LUT generation
    m_lightsBaker->CreateRenderPasses(m_computePipelineBaker, m_bindlessLayout, m_CommonPasses, m_shaderDebug, screenResolution, m_envMapBaker->GetImportanceSampling()->GetImportanceMapResolution(), m_ui.NEEAT_QualityPreset);

#if 0 // enable if needed
    if (m_gpuSort == nullptr)
        m_gpuSort = std::make_shared<GPUSort>(GetDevice(), m_shaderFactory);
    m_gpuSort->CreateRenderPasses(m_CommonPasses, m_shaderDebug);
#endif

    CreateCustomRenderPasses();
}

// Default no-op; derived samples override to create their own compute / render passes.
void Sample::CreateCustomRenderPasses()
{
}

void Sample::SetEnvMapOverrideSource(const std::string& envMapOverride) 
{ 
    if (m_envMapOverride != envMapOverride && m_envMapBaker != nullptr)
        m_envMapBaker->SetTargetCubeResolution(0);  // reset resolution just to avoid getting crazy with procedural sky as it's very slow
    m_envMapOverride = envMapOverride; 
}


std::string Sample::GetEnvMapActualPath() const
{
    if (m_envMapOverride != "" && m_envMapOverride != c_EnvMapSceneDefault)
        return (IsProceduralSky(m_envMapOverride.c_str()))?(m_envMapOverride):(std::string(c_EnvMapSubFolder) + "/" + m_envMapOverride);
    return m_envMapLocalPath;
}

void Sample::PreUpdateLighting(nvrhi::CommandListHandle commandList, bool& needNewBindings)
{
    RTXPT_GPU_SCOPE(m_commandList, "PreUpdateLighting");

    auto preUpdateCube = m_envMapBaker->GetEnvMapCube();

    const std::string envMapActualPath = GetEnvMapActualPath();

    // Cached binding sets would otherwise keep the replaced cube set / source env map alive until the next unrelated Clear.
    if (m_envMapBaker->PreUpdate(commandList, m_CommonPasses, envMapActualPath))
        m_bindingCache->Clear();

    if (preUpdateCube != m_envMapBaker->GetEnvMapCube())
        needNewBindings = true;
}

void Sample::UpdateLighting(nvrhi::CommandListHandle commandList)
{
    ZoneScoped;

    RTXPT_GPU_SCOPE(m_commandList, "UpdateLighting");

    EMB_DirectionalLight dirLights[EnvMapBaker::c_MaxDirLights];
    uint dirLightCount = 0;

    if (m_bakeDirectionalLightsIntoEnvMap)
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

    if (m_envMapBaker->Update(commandList, *m_bindingCache, m_CommonPasses, EnvMapBaker::BakeSettings{ .EnvMapRadianceScale = c_envMapRadianceScale }, m_sceneTime, dirLights, dirLightCount, !m_ui.RealtimeMode || !m_ui.EnableAnimations))
    {
        m_ui.ResetAccumulation = true;
    }

    {
        LightsBaker::BakeSettings settings;
        settings.ImportanceSamplingType = m_sampleCapabilities.UseNeeAtLightSampling ? (uint)m_ui.NEEType : 0;
        settings.CameraPosition = m_camera.GetPosition();
        settings.CameraDirection = m_camera.GetDir();
        settings.ViewProjMatrix = m_view->GetViewProjectionMatrix();
        settings.MouseCursorPos = m_ui.MousePos;
        settings.GlobalTemporalFeedbackWeight   = m_ui.NEEAT_GlobalTemporalFeedbackWeight;
        settings.LocalToGlobalSampleRatio       = m_ui.ActualNEEAT_LocalToGlobalSampleRatio();
        settings.UseApproximateMIS              = m_ui.ActualUseApproximateMIS();
        settings.DistantVsLocalImportanceScale  = m_ui.NEEAT_Distant_vs_Local_Importance;
        settings.ResetFeedback = m_ui.ResetAccumulation && !m_ui.RealtimeMode 
#if 1
            || m_ui.ResetRealtimeCaches
#endif
        ;
        settings.PrevViewportSize = float2( (float)m_viewPrevious->GetViewExtent().width(), (float)m_viewPrevious->GetViewExtent().height() );
        settings.ViewportSize = float2( (float)m_view->GetViewExtent().width(), (float)m_view->GetViewExtent().height() );
        settings.EnvMapParams = NEEAT::LightsBakerEnvMapParams{ .Transform = m_envMapSceneParams.Transform, .InvTransform = m_envMapSceneParams.InvTransform, .ColorMultiplier = m_envMapSceneParams.ColorMultiplier, .Enabled = m_envMapSceneParams.Enabled };
        settings.FrameIndex = m_frameIndex;

        m_lightsBaker->UpdateBegin(commandList, *m_bindingCache, settings, m_sceneTime, m_scene, m_materialsBaker, m_ommBaker, m_subInstanceBuffer, m_subInstanceData, m_envMapBaker->GetImportanceSampling()->GetRadianceAndImportanceMap(), m_sceneRenderCache);
    }
}

void Sample::PreUpdatePathTracing( bool resetAccum, nvrhi::CommandListHandle commandList )
{
    ZoneScoped;

    resetAccum |= m_ui.ResetAccumulation;
    resetAccum |= m_ui.RealtimeMode;

    if (resetAccum)
    {
        m_accumulationSampleIndex = (m_ui.AccumulationPreWarmRealtimeCaches)?(-32):(0);
    }
#if ENABLE_DEBUG_VIZUALISATIONS
    if (resetAccum)
        m_shaderDebug->ClearDebugVizTexture(commandList);
#endif

    // profile perf - only makes sense with high accumulation sample counts; only start counting after n-th after it stabilizes
    if( m_accumulationSampleIndex < 16 )
    {
        m_benchStart = std::chrono::high_resolution_clock::now( );
        m_benchLast = m_benchStart;
        m_benchFrames = 0;
    } else if( m_accumulationSampleIndex < m_ui.AccumulationTarget )
    {
        m_benchFrames++;
        m_benchLast = std::chrono::high_resolution_clock::now( );
    }
    m_accumulationCompleted = false;
    // 'min' in non-realtime path here is to keep looping the last sample for debugging purposes!
    if( !m_ui.RealtimeMode )
    {
        m_sampleIndex = (m_accumulationSampleIndex<0)?(m_accumulationSampleIndex+4096):(min(m_accumulationSampleIndex, m_ui.AccumulationTarget - 1));
        m_accumulationCompleted |= m_accumulationSampleIndex == m_ui.AccumulationTarget - 1;
    }
    else
        m_sampleIndex = (!m_ui.DbgFreezeRealtimeNoiseSeed)?( m_frameIndex % 8192 ):0;     // actual sample index
}

void Sample::PostUpdatePathTracing( )
{
    ZoneScoped;

    m_accumulationSampleIndex = std::min( m_accumulationSampleIndex+1, m_ui.AccumulationTarget );

    if (m_ui.ActualUseRTXDIPasses())
        m_rtxdiPass->EndFrame();

    m_ui.ResetAccumulation = false;
    m_ui.ResetRealtimeCaches = false;
    m_frameIndex++;
}

float Sample::FireflyFilterThreshold(float uiThreshold, float preExposedGrayLuminance)
{
    // Linear in preExposedGrayLuminance (a fixed multiple of on-screen middle gray), so the clamp is invariant to light
    // units and exposure; scaled to match the previous sqrt law (uiThreshold * sqrt(gray) * 1e3) at gray = kGrayRef.
    constexpr float kGrayRef = 0.18f;
    return uiThreshold * preExposedGrayLuminance * (1e3f / sqrtf(kGrayRef));
}

void Sample::UpdatePathTracerConstants( PathTracerConstants & constants, const PathTracerCameraData & cameraData )
{
    ZoneScoped;

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
    // Reset on the same frame as m_viewPrevious (UpdateViews), so prevCamera.PosW stays the origin of previousView.
    constants.prevCamera = ((m_frameIndex & 0xFFFFFFFF) == 0) ? cameraData : m_lastCameraData;
    m_lastCameraData = cameraData;

    constants.bounceCount = m_ui.BounceCount;
    constants.diffuseBounceCount = m_ui.DiffuseBounceCount;
    constants.perPixelJitterAAScale = (m_ui.RealtimeMode == false && m_ui.AccumulationAA)?(1):( (m_ui.RealtimeMode && m_ui.RealtimeAA == 3)?(m_ui.DLSSRRMicroJitter):(0.0f) );

    // needed to allow super-resolution to work best
    float dlssBias = -donut::math::log2f(sqrtf((m_displaySize.x * m_displaySize.y) / float(m_renderSize.x * m_renderSize.y)));

    constants.texLODBias = m_ui.TexLODBias + dlssBias;
    constants.sampleBaseIndex = m_sampleIndex * m_ui.ActualSamplesPerPixel();

    //constants.subSampleCount = m_ui.ActualSamplesPerPixel();
    constants.invSubSampleCount = 1.0f / (float)m_ui.ActualSamplesPerPixel();

    constants.imageWidth = m_renderSize.x; assert( m_renderSize.x == m_renderTargets->OutputColor->getDesc().width );
    constants.imageHeight = m_renderSize.y; assert( m_renderSize.y == m_renderTargets->OutputColor->getDesc().height );

    // this is the dynamic luminance that when passed through current tonemapper with current exposure settings, produces the same 50% gray
    constants.preExposedGrayLuminance = m_ui.EnableToneMapping?(donut::math::luminance(m_toneMappingPass->GetPreExposedGray(0))):(1.0f);

    const float disabledFF = 0.0f;
    if (m_ui.RealtimeMode)
        constants.fireflyFilterThreshold = (m_ui.RealtimeFireflyFilterEnabled)?(FireflyFilterThreshold(m_ui.RealtimeFireflyFilterThreshold, constants.preExposedGrayLuminance)):(disabledFF);
    else
        constants.fireflyFilterThreshold = (m_ui.ReferenceFireflyFilterEnabled)?(FireflyFilterThreshold(m_ui.ReferenceFireflyFilterThreshold, constants.preExposedGrayLuminance)):(disabledFF); // making it exposure-adaptive breaks determinism with accumulation (because there's a feedback loop), so that's disabled
    constants.useReSTIRDI = m_ui.ActualUseReSTIRDI();
    constants.useReSTIRGI = m_ui.ActualUseReSTIRGI();
    constants.denoiserRadianceClampK = m_ui.DenoiserRadianceClampK;
    constants.DLSSRRBrightnessClampK = (m_ui.DLSSRRBrightnessClampK>0)?(m_ui.DLSSRRBrightnessClampK * constants.preExposedGrayLuminance):(FLT_MAX);   // <= 0: no clamp (0 would scale everything to black)

    // no denoising layers by default
    constants.denoisingEnabled = m_ui.ActualUseStandaloneDenoiser() || m_ui.RealtimeAA == 3;

    constants._activeDenoisingLayerCount           = m_ui.ActualDenoisingLayersActiveCount();
    constants.maxDenoisingLayerVertexDepth         = std::min( std::min( (uint)m_ui.DenoisingLayersMaxVertexDepth, cDenoisingLayerMaxVertexIndex ), (uint)m_ui.BounceCount );
    constants.allowPrimarySurfaceReplacement    = m_ui.AllowPrimarySurfaceReplacement;
    constants.denoisingLayersSplitStopThreshold    = m_ui.DenoisingLayersSplitStopThreshold;
    constants._padding3                         = 0;
    constants.denoisingLayersSuppressPrimaryIndirectSpecularK  = m_ui.DenoisingLayersSuppressPrimaryIndirectSpecular?m_ui.DenoisingLayersSuppressPrimaryIndirectSpecularK:0.0f;
    constants.denoisingLayersAntiAliasingFallthrough = m_ui.DenoisingLayersAntiAliasingFallthrough;
    constants.frameIndex                        = m_frameIndex & 0xFFFFFFFF; //GetFrameIndex();
    constants.genericTSLineStride               = GenericTSComputeLineStride(constants.imageWidth, constants.imageHeight);
    constants.genericTSPlaneStride              = GenericTSComputePlaneStride(constants.imageWidth, constants.imageHeight);

    constants.NEEEnabled                        = m_ui.UseNEE;
    constants.NEEType                           = m_ui.NEEType;
    constants.NEECandidateSamples               = m_ui.NEECandidateSamples;
    constants.NEEFullSamples                    = m_ui.NEEFullSamples;

    constants.EnvironmentMapDiffuseSampleMIPLevel = m_ui.EnvironmentMapDiffuseSampleMIPLevel;

#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
    // stochastic texture filtering type and size.
    // constants.STFUseBlueNoise                   = m_ui.STFUseBlueNoise;
    constants.STFMagnificationMethod            = GetStfMagnificationMethod(m_ui.STFMagnificationMethod);
    constants.STFFilterMode                     = GetStfFilterMode(m_ui.STFFilterMode);
    constants.STFGaussianSigma                  = m_ui.STFGaussianSigma;
#endif
}


void Sample::RtxdiSetupFrame(nvrhi::IFramebuffer* framebuffer, PathTracerCameraData cameraData, uint2 renderDims)
{
    const bool envMapPresent = m_ui.EnvironmentMapParams.Enabled;

    RtxdiBridgeParameters bridgeParameters;
	bridgeParameters.frameIndex = m_frameIndex & 0xFFFFFFFF;
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
    if (m_frameIndex < 16 || m_ui.ResetAccumulation || m_ui.ResetRealtimeCaches || m_captureScriptManager->IsDoingWork() )
    {
        // Make sure we at least run one render frame to allow expensive resource creation to happen in background, and to allow at least somewhat decent convergence so when user alt-tabs they get a nice image
        return true;
    }

    if (m_ui.RenderWhenOutOfFocus)
    {
        return true;
    }

    // Let Reference mode accumulate all frames before pausing
    return (!m_ui.RealtimeMode && (m_accumulationSampleIndex < m_ui.AccumulationTarget));
}

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
        if (m_ui.DLSSFGOptions.mode == StreamlineInterface::DLSSGMode::eOn && m_ui.ActualDLSSFGMode() == StreamlineInterface::DLSSGMode::eOff) {
            GetDeviceManager()->GetStreamline().CleanupDLSSG(true);
        }

        // This is where DLSS-G is toggled On and Off (using dlssgOptions.mode) and where we set DLSS-G parameters.
        auto dlssgOptions = StreamlineInterface::DLSSGOptions{};
        dlssgOptions.mode = m_ui.ActualDLSSFGMode();
        dlssgOptions.numFramesToGenerate = m_ui.DLSSFGNumFramesToGenerate;

        // This is where we query DLSS-G minimum swapchain size
        if (GetDeviceManager()->GetStreamline().IsDLSSGAvailable())
        {
            StreamlineInterface::DLSSGState state;
            GetDeviceManager()->GetStreamline().GetDLSSGState(state, dlssgOptions);
            m_ui.DLSSFGMultiplier = state.numFramesActuallyPresented;
            m_ui.DLSSFGMaxNumFramesToGenerate = state.numFramesToGenerateMax;
            // Signalled once DLSS-G has consumed the last presented frame's tagged inputs, see the async prefix in Render
            const bool fgActive = dlssgOptions.mode != StreamlineInterface::DLSSGMode::eOff;
            m_dlssgInputsFence = fgActive ? state.inputsProcessingCompletionFence : nullptr;
            m_dlssgInputsFenceValue = fgActive ? state.lastPresentInputsProcessingCompletionFenceValue : 0;

            GetDeviceManager()->GetStreamline().SetDLSSGOptions(dlssgOptions);
            m_ui.DLSSFGOptions = dlssgOptions;
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

        // every frame, as the reset above (which also runs on the very first DLSS frame) would undo it
        if (m_cmdLine.OverrideDLSSMode > 0 && m_cmdLine.OverrideDLSSMode < (int)SI::DLSSMode::eCount)
            m_ui.DLSSMode = (SI::DLSSMode)m_cmdLine.OverrideDLSSMode;

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
                dlssOptions.useAutoExposure = true;     // Optional: provide proper "kBufferTypeExposure" for 0-lag for better precision handling
                dlssOptions.preset = StreamlineInterface::DLSSPreset::eDefault;
                // if (m_ui.RealtimeAA < 4) <- docs https://github.com/NVIDIAGameWorks/Streamline/blob/main/docs/ProgrammingGuideDLSS_RR.md#50-provide-dlss--dlss-rr-options seem to imply that these should be set even when DLSS-RR enabled
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

                    // this is an example on how to override defaults - overriding default 2/3 to higher res 3/4
                    if (dlssOptions.mode == SI::DLSSMode::eMaxQuality)
                    {
                        m_recommendedDLSSSettings.optimalRenderSize.x = dm::clamp((int)(dlssOptions.outputWidth * 3 / 4 + 0.5f), m_recommendedDLSSSettings.minRenderSize.x, m_recommendedDLSSSettings.maxRenderSize.x);
                        m_recommendedDLSSSettings.optimalRenderSize.y = dm::clamp((int)(dlssOptions.outputHeight * 3 / 4 + 0.5f), m_recommendedDLSSSettings.minRenderSize.y, m_recommendedDLSSSettings.maxRenderSize.y);
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
                dlssRROptions.preset                = m_ui.DLSRRPreset;
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

void Sample::SetSceneTime( double sceneTime ) 
{ 
    if (m_sampleGame->IsInitialized())
        m_sampleGame->SetGameTime(sceneTime);
    m_sceneTime = sceneTime; 
}

double Sample::GetSceneTime()
{
    if (m_sampleGame->IsInitialized())
        return m_sampleGame->GetGameTime();
    return m_sceneTime;
}


void Sample::PreRender()
{
    // Limit FPS
    if (m_ui.ActualFPSLimiter() > 0)
        g_FPSLimiter.FramerateLimit(m_ui.ActualFPSLimiter());

    korgi::Update();

    m_captureScriptManager->PreRender();
}

SimpleViewConstants FromPlanarViewConstants(PlanarViewConstants & view)
{
    SimpleViewConstants ret;
    ret.matWorldToView          = view.matWorldToView;
    ret.matViewToClip           = view.matViewToClip;
    ret.matWorldToClipNoOffset  = view.matWorldToClipNoOffset;
    ret.matClipToWorldNoOffset  = view.matClipToWorldNoOffset;
    ret.matWorldToClip          = view.matWorldToClip;
    ret.clipToWindowBias        = view.clipToWindowBias;
    ret.clipToWindowScale       = view.clipToWindowScale;
    ret.viewportOrigin          = view.viewportOrigin;
    ret.viewportSize            = view.viewportSize;
    ret.viewportSizeInv         = view.viewportSizeInv;
    ret.pixelOffset             = view.pixelOffset;
    return ret;
}

void Sample::PostProcessPreToneMapping(nvrhi::ICommandList* commandList, const donut::engine::ICompositeView& compositeView)
{ 
    ZoneScoped;

    // a.k.a. HDR post-process (e.g. bloom goes here)
    donut::engine::PlanarView fullscreenView = *m_view;
    nvrhi::Viewport windowViewport(float(m_displaySize.x), float(m_displaySize.y));
    fullscreenView.SetViewport(windowViewport);
    fullscreenView.UpdateCache();

    if (m_ui.EnableBloom && m_ui.BloomIntensity > 0.f && m_ui.BloomRadius > 0.f)
    {
        m_bloomPass->Render(m_commandList, m_renderTargets->ProcessedOutputFramebuffer, fullscreenView, m_renderTargets->ProcessedOutputColor, m_ui.BloomRadius, m_ui.BloomIntensity);
    }

    if (m_ui.PostProcessTestPassHDR)
    {
        GpuProfiler::Begin(m_commandList, "TestRaygenPP_HDR");

        m_commandList->setTextureState(m_renderTargets->ProcessedOutputColor, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);

        nvrhi::rt::DispatchRaysArguments args;
        args.width = m_displaySize.x;
        args.height = m_displaySize.y;

        nvrhi::rt::State state;
        state.shaderTable = m_ptPipelineTestRaygenPPHDR->GetShaderTable();
        state.bindings = { m_bindingSet, m_DescriptorTable->GetDescriptorTable(), m_bindlessHeapDescriptorTable->GetDescriptorTable() };
        m_commandList->setRayTracingState(state);

        SampleMiniConstants miniConstants = { uint4(0, 0, 0, 0) };
        m_commandList->setPushConstants(&miniConstants, sizeof(miniConstants));
        m_commandList->dispatchRays(args);

        m_commandList->setTextureState(m_renderTargets->ProcessedOutputColor, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);

        GpuProfiler::End(m_commandList);
    }
}

void Sample::PostProcessPostToneMapping(nvrhi::ICommandList* commandList, const donut::engine::ICompositeView& compositeView)
{ 
    ZoneScoped;

    // a.k.a. LDR post-process (e.g. colour filters go here)
    if (m_ui.PostProcessEdgeDetection)
    {
        GpuProfiler::Begin(m_commandList, "PPEdgeDetection");

        m_commandList->copyTexture(m_renderTargets->LdrColorScratch, nvrhi::TextureSlice(), m_renderTargets->LdrColor, nvrhi::TextureSlice());

        nvrhi::rt::DispatchRaysArguments args;
        args.width  = m_displaySize.x;
        args.height = m_displaySize.y;

        nvrhi::rt::State state;
        state.shaderTable = m_ptPipelineEdgeDetection->GetShaderTable();
        state.bindings = { m_bindingSet, m_DescriptorTable->GetDescriptorTable(), m_bindlessHeapDescriptorTable->GetDescriptorTable() };
        m_commandList->setRayTracingState(state);

        SampleMiniConstants miniConstants = { uint4( *reinterpret_cast<uint*>(&m_ui.PostProcessEdgeDetectionThreshold), 0, 0, 0) };
        m_commandList->setPushConstants(&miniConstants, sizeof(miniConstants));
        m_commandList->dispatchRays(args);

        m_commandList->setTextureState(m_renderTargets->LdrColor, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);

        GpuProfiler::End(m_commandList);
    }
}

// Steady-state DLSS-RR frames record everything up to the DLSS-RR inputs on m_prefixCommandList; anything else stays on
// the single graphics list
bool Sample::ShouldRecordPrefixAsync(bool needNewPasses)
{
#if DONUT_WITH_STREAMLINE
    // Cached PT shader tables are baked on first use through a pixel-stage barrier, illegal on compute, so the previous
    // frame must have run the same passes with the same pipelines
    SharcIntegration* sharc = GetSharc();
    const bool sharcTraining = sharc != nullptr && m_ui.ActualRadianceCacheMode() == 1 && sharc->EnableUpdate();
    const int64_t prefixPassesKey = (m_ui.RealtimeMode && m_ui.RealtimeAA == 3 && m_ptPipelineBaker != nullptr) ? (m_ptPipelineBaker->GetVersion() * 2 + (sharcTraining ? 1 : 0)) : -1;
    const bool prefixPassesBaked = prefixPassesKey == m_lastPrefixPassesKey;
    m_lastPrefixPassesKey = prefixPassesKey;

    if (!m_ui.EnableAsyncPrefix || !IsAsyncPrefixAvailable() || !m_ui.RealtimeMode || m_ui.RealtimeAA != 3 || m_scene == nullptr)
        return false;
    if (needNewPasses || m_bindingSet == nullptr || m_bindingSetPrefixEarly == nullptr || !prefixPassesBaked)
        return false;
    // RTXDI and the debug views write u_OutputColor in the early prefix, where it is a dummy (see RecreateBindingSet)
    if (m_ui.ActualUseRTXDIPasses() || m_ui.DebugView != DebugViewType::Disabled)
        return false;
    // Anything that rebuilds, or flushes the list it is given to the graphics queue
    if (m_scene->GetSceneGraph()->HasPendingStructureChanges() || m_materialsBaker->HasDeferredTexturesLoad()
        || (m_ommBaker != nullptr && m_ommBaker->NeedsGraphicsQueue()) || m_envMapBaker->NeedsGraphicsQueue(GetEnvMapActualPath()))
        return false;
    // The graphics tail reads prefix outputs for these
    if (m_pick || m_ui.ContinuousDebugFeedback || m_ui.ShowDebugLines || m_ui.PostProcessTestPassHDR || m_ui.PostProcessEdgeDetection)
        return false;
    return true;
#else
    (void)needNewPasses;
    return false;
#endif
}

// ShaderDebug ends here, at the end of the prefix, in every frame: async prefix frames hand its resources to the next
// frame's prefix at this point. Read back, copy what the overlay draws, and bind the global set without ShaderDebug.
void Sample::CloseShaderDebug()
{
    if (m_ui.EnableShaderDebug)
    {
        RTXPT_GPU_SCOPE(m_commandList, "ShaderDebugEndFrame");
        m_shaderDebug->EndFrame(m_commandList);
        m_shaderDebug->CopyForOutput(m_commandList);
    }
    m_shaderDebug->SetClosed(true);
    if (m_bindingSetTail != nullptr)
        m_bindingSet = m_bindingSetTail;
}

void Sample::Render(nvrhi::IFramebuffer* framebuffer)
{
    ZoneScoped;

    const auto& fbinfo = framebuffer->getFramebufferInfo();
    m_displaySize = m_renderSize = uint2(fbinfo.width, fbinfo.height);
    float lodBias = 0.f;

    if (m_scene == nullptr)
    {
        assert( false ); // TODO: handle separately, just display pink color
        return;
    }
    m_progressLoading.Stop();
    m_asyncLoadingInProgress = false; // can be changed back to 'true' during the frame below, for example by OMMs

    PreRender();
    RAII_SCOPE( ;, PostRender(); );     // ensure PostRender is called in case of any early out!

    StreamlinePreRender();

 
    m_displayAspectRatio = m_displaySize.x/(float)m_displaySize.y;

    if (m_view == nullptr)
    {
        m_view = std::make_shared<PlanarView>();
        m_viewPrevious = std::make_shared<PlanarView>();
        m_viewPrevious->SetViewport(nvrhi::Viewport(float(m_renderSize.x), float(m_renderSize.y)));
        m_view->SetViewport(nvrhi::Viewport(float(m_renderSize.x), float(m_renderSize.y)));
    }

    bool needNewPasses = false;
    if( m_renderTargets == nullptr || m_renderTargets->IsUpdateRequired( m_renderSize, m_displaySize ) )
    {
        GetDevice()->waitForIdle();
        GetDevice()->runGarbageCollection();
        for (int i = 0; i < std::size(m_nrd); i++)
            m_nrd[i] = nullptr;
        m_renderTargets = nullptr;
        m_bindingCache->Clear( );
        m_renderTargets = std::make_unique<RenderTargets>( );
        m_renderTargets->Init(GetDevice( ), m_renderSize, m_displaySize, c_SwapchainCount,
            RenderTargets::OptionalBuffers{
                .NeedsGBuffer              = m_sampleCapabilities.NeedsGBuffer,
                .NeedsDirectLightingBuffer = m_sampleCapabilities.NeedsDirectLightingBuffer,
                .EnableMotionVectors       = true,
                .UseReverseProjection      = true,
            });

        needNewPasses = true;
        OnRenderTargetsRecreated();
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

    // Re-lays out the NEE-AT buffers, and recompiles its passes and the path tracer in the same frame so they agree
    if (m_lightsBaker != nullptr && m_lightsBaker->GetQualityPreset() != m_ui.NEEAT_QualityPreset)
        needNewPasses = true;

    if (m_ui.NRDModeChanged) // if changing between ReLAX and ReBLUR
    {
        needNewPasses = true;
        for (int i = 0; i < std::size(m_nrd); i++)
            m_nrd[i] = nullptr;
    }
    if (!m_ui.ActualUseStandaloneDenoiser()) // clean up the memory if not used
    {
        for (int i = 0; i < std::size(m_nrd); i++)
            m_nrd[i] = nullptr;
    }

    // Acceleration structures need some material info, whilst other passes need acceleration structures, so first set up materials if needed
    if (needNewPasses)
    {
        m_progressInitializingRenderer.Start("Initializing renderer...");

        if (m_materialsBaker == nullptr)
            m_materialsBaker = std::make_shared<MaterialsBaker>(GetMaterialSpecializationShader(), GetDevice(), m_TextureCache, m_shaderFactory);

        if (m_ptPipelineBaker == nullptr)
        {
            m_ptPipelineBaker = std::make_shared<PTPipelineBaker>(GetDevice(), m_materialsBaker, m_bindingLayout, m_bindlessLayout,
                [this](std::vector<donut::engine::ShaderMacro> & macros){ this->GetPTPipelineGlobalMacros(macros); }, m_bindlessHeapLayout);
            
            m_computePipelineBaker = std::make_shared<ComputePipelineBaker>(GetDevice());
            
            CreateRTPipelines();
        }

        m_materialsBaker->CreateRenderPassesAndLoadMaterials(m_bindlessLayout, m_CommonPasses, m_scene, m_currentScenePath, GetLocalPath(c_AssetsFolder));
        m_progressInitializingRenderer.Set(5);
        CollectUncompressedTextures();
        if(m_ommBaker) m_ommBaker->CreateRenderPasses(m_bindlessLayout, m_CommonPasses);
        m_progressInitializingRenderer.Set(20);

        if (m_zoomTool == nullptr)
            m_zoomTool = std::make_unique<ZoomTool>(GetDevice(), m_shaderFactory);
    }

    // Changes to material properties and settings can require a BLAS/TLAS or subInstanceBuffer rebuild (alpha tested/exclusion flags etc); otherwise this is a no-op.
    RecreateAccelStructs(m_commandList);

    if (m_ui.ActualUseRTXDIPasses() && m_rtxdiPass == nullptr )
        needNewPasses = true; // this will initialize rtxdi passes
    if (!m_ui.ActualUseRTXDIPasses())
        m_rtxdiPass = nullptr;

    // this will also create or update materials which can trigger the need to update acceleration structures
    if (needNewPasses)
    {
        m_progressInitializingRenderer.Set(40);
        GetDevice()->waitForIdle();    // some subsystems have resources that could still be in use and might be deleted - make sure that's safe
        m_commandList->open();
        CreateRenderPasses(exposureResetRequired, m_commandList);
        m_commandList->close();
        GetDevice()->executeCommandList(m_commandList);
        m_progressInitializingRenderer.Set(70);
    }

    // this is the point where main ray tracing pipelines will actually get compiled
    m_ptPipelineBaker->Update(m_scene, (unsigned int)m_subInstanceData.size(), needNewPasses);
    
    // Update compute shaders (compile if needed)
    if (m_computePipelineBaker)
        m_computePipelineBaker->Update(needNewPasses);
    
    m_progressInitializingRenderer.Set(90);

    // Everything up to the DLSS-RR inputs goes on m_prefixCommandList, so DLSS-RR and the rest of this frame can overlap
    // the next frame's prefix; anything unusual keeps the whole frame on the graphics list
    const bool asyncPrefix = ShouldRecordPrefixAsync(needNewPasses);
    if (asyncPrefix != m_lastFrameAsyncPrefix)
        log::info("AsyncPrefix %s", asyncPrefix ? "on" : "off");
    // DLSS-G reads the previous frame's tagged inputs after its Present, on Streamline's own queues; a non-presenting queue
    // must wait on this fence before rewriting them (Streamline DLSS-G guide, section 15.1)
    auto computeQueueWaitForDLSSGInputs = [&]()
    {
#if DONUT_WITH_STREAMLINE
        if (m_dlssgInputsFence != nullptr)
        {
            ID3D12CommandQueue* computeQueue = GetDevice()->getNativeQueue(nvrhi::ObjectTypes::D3D12_CommandQueue, nvrhi::CommandQueue::Compute);
            computeQueue->Wait(static_cast<ID3D12Fence*>(m_dlssgInputsFence), m_dlssgInputsFenceValue);
        }
#endif
    };
    if (asyncPrefix)
    {
        if (!m_lastFrameAsyncPrefix)
        {   // That frame ran on the graphics queue, unordered with this compute work, and may have tagged the live Depth/MVs
            // for DLSS-G, which this frame's early prefix rewrites
            GetDevice()->queueWaitForCommandList(nvrhi::CommandQueue::Compute, nvrhi::CommandQueue::Graphics, m_lastGraphicsSubmit);
            computeQueueWaitForDLSSGInputs();
        }
        m_commandList = m_prefixCommandList;
    }

    m_commandList->open();
    GpuProfiler::BeginFrameZone(m_commandList);    // Tracy-only; its "CPU frame" annotation matches Tracy's frame bar and MainRender's value

    bool needNewBindings = false;
    PathTracerCameraData cameraData;
    {
        ZoneScopedN("EarlyUpdate");

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

        // Against LdrFramebuffer, which it draws into - not the swapchain, whose format differs on Vulkan (BGRA).
        if (needNewPasses || needNewBindings || m_bindingSet == nullptr)
            m_shaderDebug->CreateRenderPasses(m_renderTargets->LdrFramebuffer->GetFramebuffer(*m_view), m_renderTargets->Depth);

        if (m_ui.EnableShaderDebug)
        {
            dm::float4x4 viewProj = m_view->GetViewProjectionMatrix();
            m_shaderDebug->BeginFrame(m_commandList, viewProj);
        }

        // NOTE: this refreshes geometry buffers and updates stuff needed by m_ommBaker and m_materialsBaker and others below!
        {
            ZoneScopedN("SceneRefresh");
            m_scene->Refresh(m_commandList, GetFrameIndex());
        }

        // Open the per-frame scene cache HERE rather than in PreRender(): Scene::Refresh() above is where pending scene
        // graph structure changes are applied, so anything cached before it would be stale on exactly the frames where
        // the graph changed. Closed unconditionally in PostRender().
        m_sceneRenderCache.Open( m_scene, &m_frameWorkerPool );

        // Before the AS work: the sub-instance upload's barrier (out of Common) waits on all earlier GPU work, the TLAS build included
        m_materialsBaker->Update(m_commandList, m_scene, m_subInstanceData, m_sceneRenderCache);
        UploadSubInstanceData(m_commandList); // this is now partial subInstance data, but lights baker update requires it to find materials and create emissive triangle lights

        if(m_ommBaker) m_ommBaker->BuildOpacityMicromaps(*m_commandList, *m_scene);
        UpdateSkinnedBLASs(m_commandList, GetFrameIndex());


        TransitionMeshBuffersToReadOnly(m_commandList);    // before the TLAS build, so its barrier waits on the BLAS refits only
        UpdateTLAS(m_commandList);
        if (m_ommBaker) 
        {
            m_asyncLoadingInProgress |= m_ommBaker->Update(*m_commandList, *m_scene);
            m_asyncLoadingInProgress |= m_ommBaker->UIData().BuildsLeftInQueue > 0;
        }

        // Update input lighting, environment map, etc.
        PreUpdateLighting(m_commandList, needNewBindings);

        // Derived samples create their resolution-dependent resources and bindless descriptors here, so they
        // are in place before the descriptor table is snapshotted just below.
        UpdateCustomResources();

        // After every descriptor-creating step above, before the binding set rebuild below.
        UpdateBindlessValidationData(m_commandList, needNewBindings);

        // Early init for RTXDI
        if (m_rtxdiPass != nullptr) 
        {
            if (needNewPasses || needNewBindings || m_bindingSet == nullptr)
                m_rtxdiPass->Reset();
            RtxdiSetupFrame(framebuffer, cameraData, m_renderSize);
        }
    }

	if( needNewPasses || needNewBindings || m_bindingSet == nullptr )
    {
        m_progressInitializingRenderer.Set(95);
        RAII_SCOPE( m_commandList->close(); GetDevice()->executeCommandList(m_commandList, m_commandList->getDesc().queueType);, m_commandList->open(););

        RecreateBindingSet();

        m_progressInitializingRenderer.Set(100);

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
        }
        m_progressInitializingRenderer.Stop();
    }

    // The async prefix binds a variant without what the previous frame's graphics tail may still be using
    if (asyncPrefix)
        m_bindingSet = m_bindingSetPrefixEarly;

    m_toneMappingPass->PreRender(m_ui.ToneMappingParams);

    PreUpdatePathTracing(needNewPasses, m_commandList);


    SampleConstants & constants = m_currentConstants; memset(&constants, 0, sizeof(constants));
    SampleMiniConstants miniConstants = { uint4(0, 0, 0, 0) }; // accessible but unused in path tracing at the moment
    if( m_scene == nullptr )
    {
        m_commandList->clearTextureFloat( m_renderTargets->OutputColor, nvrhi::AllSubresources, nvrhi::Color( 1, 1, 0, 0 ) );
        m_commandList->writeBuffer(m_constantBuffer, &constants, sizeof(constants));
        CloseShaderDebug();
    }
    else
    {
        ZoneScopedN("MainRender");
        ZoneValue(GpuProfiler::FrameNumber());

        // I suppose we need to clear depth for right-click picking at least
        m_renderTargets->Clear( m_commandList );

        UpdatePathTracerConstants(constants.ptConsts, cameraData);
        constants.MaterialCount = m_materialsBaker->GetMaterialDataCount(); // m_scene->GetSceneGraph()->GetMaterials().size();

        constants.envMapSceneParams = m_envMapSceneParams;
        constants.envMapImportanceSamplingParams = m_envMapBaker->GetImportanceSampling()->GetShaderParams();

        PlanarViewConstants view;           m_view->FillPlanarViewConstants(view);
        PlanarViewConstants previousView;   m_viewPrevious->FillPlanarViewConstants(previousView);
        constants.view          = FromPlanarViewConstants(view);
        constants.previousView  = FromPlanarViewConstants(previousView);

        constants.debug = {};
        constants.debug.pick = m_pick || m_ui.ContinuousDebugFeedback;
        constants.debug.pickX = (constants.debug.pick)?(m_ui.DebugPixel.x):(-1);
        constants.debug.pickY = (constants.debug.pick)?(m_ui.DebugPixel.y):(-1);
        constants.debug.debugLineScale = (m_ui.ShowDebugLines)?(m_ui.DebugLineScale):(0.0f);
        constants.debug.showWireframe = m_ui.ShowWireframe;
        constants.debug.debugViewType = (int)m_ui.DebugView;
        constants.debug.debugViewDenoisingLayerIndex = (m_ui.ActualDenoisingLayersActiveCount()==1)?(0):(m_ui.DebugViewDenoisingLayerIndex);
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

        // NRD v4.17 removed "D" from ReblurHitDistanceParameters (it was unused internally); .w is unused padding here
        constants.denoisingHitParamConsts = { m_ui.ReblurSettings.hitDistanceParameters.A, m_ui.ReblurSettings.hitDistanceParameters.B,
                                              m_ui.ReblurSettings.hitDistanceParameters.C };

        // This updates all lighting: distant (environment maps and directional analytic lights) and local (analytic lights and emissive triangle lights)
        // Must go before m_constantBuffer as when saving screenshots it closes and re-opens command list, flushing the volatile constant buffer!
        UpdateLighting(m_commandList);

#if RTXPT_SUBINSTANCE_LIGHT_PATCH_PASS
        // Only the two light fields changed since the upload above, and that upload reset both to invalid
        // (MaterialsBaker -> SubInstanceData::Default), so a sparse patch is enough to complete the data.
        m_lightsBaker->ApplySubInstanceLightPatches(m_commandList, *m_bindingCache, m_subInstanceBuffer);
#else
        UploadSubInstanceData(m_commandList); // this is now full subInstance data
#endif
        constants.lightControl.AnalyticLightOffset = m_lightsBaker->GetAnalyticLightOffset();
        constants.lightControl.AnalyticLightCount = m_lightsBaker->GetAnalyticLightCount();
        constants.lightControl.LocalRegionFirstLightIndex = 0;
        constants.lightControl.LocalRegionNumLights = 0;
        constants.lightControl.InfiniteRegionFirstLightIndex = 0;
        constants.lightControl.InfiniteRegionNumLights = 0;

        // Fill sample-specific constants (e.g. ptConsts.sharc) just before the upload.
        UpdateCustomConstants(constants, m_commandList);

        m_commandList->writeBuffer(m_constantBuffer, &constants, sizeof(constants));

        SampleRenderCode(framebuffer, m_commandList, constants, needNewPasses);

        if (asyncPrefix)
        {   // The DLSS-RR inputs below are still being read by the previous frame's DLSS-RR
            m_commandList->close();
            GetDevice()->executeCommandList(m_commandList, nvrhi::CommandQueue::Compute);
            GetDevice()->queueWaitForCommandList(nvrhi::CommandQueue::Compute, nvrhi::CommandQueue::Graphics, m_lastGraphicsSubmit);
            computeQueueWaitForDLSSGInputs();   // before rewriting RRDepth/RRMotionVectors
            m_commandList->open();
            m_commandList->writeBuffer(m_constantBuffer, &constants, sizeof(constants));   // volatile, lost at close
            m_bindingSet = m_bindingSetPrefixLate;
        }

#if DONUT_WITH_STREAMLINE
        if (m_ui.RealtimeMode && m_ui.RealtimeAA == 3) // DLSS-RR
        {
            // Produce the DLSS-RR inputs before tagging transitions them to Streamline's handoff state.
            // This pass reuses the global compute bindings, so rebinding them after tagging would not
            // necessarily restore OutputColor and ScreenMotionVectors to UAV state.
            // IntroPathTracer prepares its guides separately; its base implementation is a no-op.
            PrepareDLSSRRGuides(m_commandList);
            if (UseRRDepthMVCopies())
            {
                m_renderTargets->EnsureRRDepthAndMotionVectors();
                // One barrier batch for both copies (copyTexture commits one per call)
                m_commandList->setTextureState(m_renderTargets->Depth, nvrhi::AllSubresources, nvrhi::ResourceStates::CopySource);
                m_commandList->setTextureState(m_renderTargets->ScreenMotionVectors, nvrhi::AllSubresources, nvrhi::ResourceStates::CopySource);
                m_commandList->setTextureState(m_renderTargets->RRDepth, nvrhi::AllSubresources, nvrhi::ResourceStates::CopyDest);
                m_commandList->setTextureState(m_renderTargets->RRMotionVectors, nvrhi::AllSubresources, nvrhi::ResourceStates::CopyDest);
                m_commandList->commitBarriers();
                m_commandList->copyTexture(m_renderTargets->RRDepth, nvrhi::TextureSlice(), m_renderTargets->Depth, nvrhi::TextureSlice());
                m_commandList->copyTexture(m_renderTargets->RRMotionVectors, nvrhi::TextureSlice(), m_renderTargets->ScreenMotionVectors, nvrhi::TextureSlice());
            }
        }
#endif

        CloseShaderDebug();

        if (asyncPrefix)
        {
            GpuProfiler::End(m_commandList);    // a GPU zone can't cross queues: each queue gets its own "Frame" zone
            m_commandList->close();
            const uint64_t prefixSubmit = GetDevice()->executeCommandList(m_commandList, nvrhi::CommandQueue::Compute);
            GetDevice()->queueWaitForCommandList(nvrhi::CommandQueue::Graphics, nvrhi::CommandQueue::Compute, prefixSubmit);
            m_commandList = m_graphicsCommandList;
            m_commandList->open();
            GpuProfiler::BeginFrameZone(m_commandList);
            m_commandList->writeBuffer(m_constantBuffer, &constants, sizeof(constants));   // volatile, lost at close
        }

        PostProcessAA(framebuffer, needNewPasses || m_ui.ResetRealtimeCaches);
    }

    donut::engine::PlanarView fullscreenView = *m_view;
    {
        ZoneScopedN("PostProcessAndTonemap");

        nvrhi::Viewport windowViewport(float(m_displaySize.x), float(m_displaySize.y));
        {
            ZoneScopedN("UpdateViewports");

            fullscreenView.SetViewport(windowViewport);
            fullscreenView.UpdateCache();
        }

        PostProcessPreToneMapping(m_commandList, fullscreenView);   // writing to m_renderTargets->ProcessedOutputColor

        //Tone Mapping; it will read from m_renderTargets->ProcessedOutputColor and write into m_renderTargets->LdrColor; in case tonemapping is disabled, it's just a passthrough

        if (m_toneMappingPass->Render(m_commandList, GetBindingCache(), fullscreenView, m_renderTargets->ProcessedOutputColor, m_ui.EnableToneMapping))
        {
            ZoneScopedN("Re-update main constant buffer");;

            // first run tonemapper can close & re-open command list - when that happens, we have to re-upload volatile constants
            m_commandList->writeBuffer(m_constantBuffer, &constants, sizeof(constants));
        }

        PostProcessPostToneMapping(m_commandList, fullscreenView);  // writing to m_renderTargets->LdrColor

        // DLSS-G's HUD-less color (tagged in PostProcessAA): the frame before the debug overlay, zoom tool and UI go on top
        if (m_ui.ActualDLSSFGMode() != StreamlineInterface::DLSSGMode::eOff)
            m_commandList->copyTexture(m_renderTargets->PreUIColor, nvrhi::TextureSlice(), m_renderTargets->LdrColor, nvrhi::TextureSlice());
    }

    {
        ZoneScopedN("DebugAndBlit");

        // Drawn from the copies made by CloseShaderDebug; async prefix frames depth-test against DLSS-RR's Depth copy, as the
        // next frame's prefix may be rewriting Depth by now
        if (m_ui.EnableShaderDebug)
            m_shaderDebug->OutputFromCopies(m_commandList, m_renderTargets->LdrFramebuffer->GetFramebuffer(fullscreenView),
                asyncPrefix ? m_renderTargets->RRDepth : m_renderTargets->Depth, fbinfo.getViewport());

        m_zoomTool->Render(m_commandList, *m_bindingCache, m_renderTargets->LdrColor);

        GpuProfiler::Begin(m_commandList, "Blit");
        m_CommonPasses->BlitTexture(m_commandList, framebuffer, m_renderTargets->LdrColor, m_bindingCache.get());
        GpuProfiler::End(m_commandList);

        if (m_ui.ShowDebugLines == true)
        {
            GpuProfiler::Begin(m_commandList, "Debug Lines");

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

            GpuProfiler::End(m_commandList);
        }
        m_cpuSideDebugLines.clear();

        if( m_ui.ContinuousDebugFeedback || m_pick )
        {
            m_commandList->copyBuffer(m_feedback_Buffer_Cpu, 0, m_feedback_Buffer_Gpu, 0, sizeof(DebugFeedbackStruct) * 1);
            m_commandList->copyBuffer(m_debugLineBufferDisplay, 0, m_debugLineBufferCapture, 0, sizeof(DebugLineStruct) * MAX_DEBUG_LINES );
            m_commandList->copyBuffer(m_debugDeltaPathTree_Cpu, 0, m_debugDeltaPathTree_Gpu, 0, sizeof(DeltaTreeVizPathVertex) * cDeltaTreeVizMaxVertices);
	    }

      }

      nvrhi::ITexture* framebufferTexture = framebuffer->getDesc().colorAttachments[0].texture;

      {
          ZoneScopedN("ExecuteCommandList");

          assert(m_commandList.Get() == m_graphicsCommandList.Get());
          GpuProfiler::End(m_commandList);
          m_commandList->close();
          m_lastGraphicsSubmit = GetDevice()->executeCommandList(m_commandList);
          m_lastFrameAsyncPrefix = asyncPrefix;
          m_bindingSet = m_bindingSetMain;
          m_shaderDebug->SetClosed(false);
      }


    // resolve right click picking and debug info
    if (m_ui.ContinuousDebugFeedback || m_pick)
    {
        ZoneScopedN("DebugPickReadback");

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

    auto DumpScreenshot = [this, framebufferTexture](const char* fileName)
    {
        // Present, the swapchain's kept state - Common is the same on D3D12 but means UNDEFINED (discard) on Vulkan.
        return SaveTextureToFile(GetDevice(), m_CommonPasses.get(), framebufferTexture, nvrhi::ResourceStates::Present, fileName);
    };

    m_captureScriptManager->PostRender(DumpScreenshot);

    if (m_ui.ExperimentalPhotoModeScreenshot)
    {
        DenoisedScreenshot( framebufferTexture );
        m_ui.ExperimentalPhotoModeScreenshot = false;
    }

    if (m_temporalAntiAliasingPass != nullptr)
        m_temporalAntiAliasingPass->AdvanceFrame();

	std::swap(m_view, m_viewPrevious);
	GetDeviceManager()->SetVsyncEnabled(m_ui.ActualEnableVsync());

    PostUpdatePathTracing();
}

void Sample::PostRender()
{
    // Post-Render() cleanup
    m_sceneRenderCache.Close();
    
    // This clear is actually costly itself for very high instance counts - not here, but at the next resize where they get initialized to 0; let's not clear but beware of dangling pointers
    // m_tlasInstanceDescs.clear();
}

// Perhaps in the future all of this bindless stuff needs to be moved to Bindless.h/Bindless.cpp
// Translates a descriptor into the packed metadata word the shader checks. MUST match the RTXPT_BINDLESS_VIEW_*
// and RTXPT_BINDLESS_META_* defines in Rtxpt/Shaders/Bindless.hlsli.
static uint32_t EncodeBindlessValidationEntry(const nvrhi::BindingSetItem& item)
{
    constexpr uint32_t c_allocatedBit   = 1u << 0;
    constexpr uint32_t c_uavBit         = 1u << 1;
    constexpr uint32_t c_viewShift      = 8;
    constexpr uint32_t c_strideShift    = 16;
    constexpr uint32_t c_strideMask     = 0xFFFFu;

    uint32_t viewType = 0;      // RTXPT_BINDLESS_VIEW_NONE
    bool isUAV = false;

    switch (item.type)
    {
    case nvrhi::ResourceType::None:                         return 0;   // free slot - reads back as 'not allocated'
    case nvrhi::ResourceType::Texture_SRV:                   viewType = 1; break;
    case nvrhi::ResourceType::Texture_UAV:                   viewType = 1; isUAV = true; break;
    case nvrhi::ResourceType::TypedBuffer_SRV:               viewType = 2; break;
    case nvrhi::ResourceType::TypedBuffer_UAV:               viewType = 2; isUAV = true; break;
    case nvrhi::ResourceType::StructuredBuffer_SRV:          viewType = 3; break;
    case nvrhi::ResourceType::StructuredBuffer_UAV:          viewType = 3; isUAV = true; break;
    case nvrhi::ResourceType::RawBuffer_SRV:                 viewType = 4; break;
    case nvrhi::ResourceType::RawBuffer_UAV:                 viewType = 4; isUAV = true; break;
    case nvrhi::ResourceType::SamplerFeedbackTexture_UAV:    isUAV = true; break;
    default: break;     // constant buffers, samplers, accel structs - allocated, but not reachable via GetBindless*
    }

    // Structured-buffer element stride, straight off the buffer desc - this is what lets the shader tell two
    // same-typed buffers apart, since their view type and UAV bit are identical. Zero for everything else.
    uint32_t stride = 0;
    if (viewType == 3 /* RTXPT_BINDLESS_VIEW_STRUCTURED_BUFFER */ && item.resourceHandle != nullptr)
        stride = std::min(static_cast<nvrhi::IBuffer*>(item.resourceHandle)->getDesc().structStride, c_strideMask);

    return c_allocatedBit | (isUAV ? c_uavBit : 0u) | (viewType << c_viewShift) | (stride << c_strideShift);
}

void Sample::UpdateBindlessValidationData(nvrhi::ICommandList* commandList, bool& needNewBindings)
{
    nvrhi::IDescriptorTable* table = m_bindlessHeapDescriptorTable->GetDescriptorTable();

    // Indexed by heap index, exactly as the shader receives it from DescriptorHandle::GetIndexInHeap(): entry
    // [firstDescriptorInHeap + i] describes table slot i. Everything below that base, and everything past the end
    // of the buffer, reads back as zero - i.e. 'not allocated' - which is precisely the desired answer for an
    // index that does not belong to this table.
    const uint32_t firstInHeap = table->getFirstDescriptorIndexInHeap();
    const uint32_t capacity = m_bindlessHeapDescriptorTable->GetUsage().capacity;
    const size_t required = size_t(firstInHeap) + capacity;

    if (m_bindlessValidationBuffer == nullptr || m_bindlessValidationBuffer->getDesc().byteSize < required * sizeof(uint32_t))
    {
        // Grow only, and with headroom: firstDescriptorInHeap can move DOWN as well as up, because the heap
        // allocator reuses freed ranges, so sizing exactly to 'required' would churn the binding set.
        nvrhi::BufferDesc validationBufferDesc;
        validationBufferDesc.byteSize = std::max<size_t>(required * 2, 1024) * sizeof(uint32_t);
        validationBufferDesc.structStride = sizeof(uint32_t);
        validationBufferDesc.debugName = "BindlessValidationData";
        validationBufferDesc.initialState = nvrhi::ResourceStates::NonPixelShaderResource;
        validationBufferDesc.keepInitialState = true;
        m_bindlessValidationBuffer = GetDevice()->createBuffer(validationBufferDesc);
        needNewBindings = true;     // the binding set holds a raw reference to the old buffer
    }

    // Debug-only: GetDescriptor takes the manager's mutex per call, and in Release the shader side compiles out
    // entirely so the contents are never read.
#if defined(_DEBUG)
    m_bindlessValidationData.assign(required, 0u);
    for (uint32_t i = 0; i < capacity; i++)
        m_bindlessValidationData[size_t(firstInHeap) + i] = EncodeBindlessValidationEntry(m_bindlessHeapDescriptorTable->GetDescriptor(i));

    if (!m_bindlessValidationData.empty())
        commandList->writeBuffer(m_bindlessValidationBuffer, m_bindlessValidationData.data(), m_bindlessValidationData.size() * sizeof(uint32_t));
#else
    (void)commandList;
#endif
}

void Sample::RecreateBindingSet()
{
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
        nvrhi::BindingSetItem::StructuredBuffer_SRV(4, (m_ommBaker)?(m_ommBaker->GetGeometryDebugBuffer()):(m_materialsBaker->GetMaterialDataBuffer().Get()) ),   // YUCK
        nvrhi::BindingSetItem::StructuredBuffer_SRV(5, m_materialsBaker->GetMaterialDataBuffer()),
        nvrhi::BindingSetItem::Texture_SRV(6,  m_renderTargets->LdrColorScratch, nvrhi::Format::SRGBA8_UNORM),
        nvrhi::BindingSetItem::Texture_SRV(10, m_envMapBaker->GetEnvMapCube()), //m_EnvironmentMap->IsEnvMapLoaded() ? m_EnvironmentMap->GetEnvironmentMap() : m_CommonPasses->m_BlackTexture),
        nvrhi::BindingSetItem::Texture_SRV(11, m_envMapBaker->GetImportanceSampling()->GetImportanceMapOnly()),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(12, m_lightsBaker->GetControlBuffer()),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(13, m_lightsBaker->GetLightBuffer()),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(14, m_lightsBaker->GetLightExBuffer()),
        nvrhi::BindingSetItem::Sampler(0, m_CommonPasses->m_AnisotropicWrapSampler),
        nvrhi::BindingSetItem::Sampler(1, m_envMapBaker->GetEnvMapCubeSampler()),
        nvrhi::BindingSetItem::Sampler(2, m_envMapBaker->GetImportanceSampling()->GetImportanceMapSampler()),
        nvrhi::BindingSetItem::Sampler(3, m_CommonPasses->m_LinearClampSampler),
        nvrhi::BindingSetItem::Texture_UAV(0, m_renderTargets->OutputColor),
        nvrhi::BindingSetItem::Texture_UAV(1, m_renderTargets->ProcessedOutputColor),
        nvrhi::BindingSetItem::Texture_UAV(2, m_renderTargets->LdrColor, nvrhi::Format::RGBA8_UNORM),
        nvrhi::BindingSetItem::Texture_UAV(4, m_renderTargets->Throughput),
        nvrhi::BindingSetItem::Texture_UAV(5, m_renderTargets->ScreenMotionVectors),
        nvrhi::BindingSetItem::Texture_UAV(6, m_renderTargets->Depth),
        nvrhi::BindingSetItem::Texture_UAV(7, m_renderTargets->SpecularHitT), 
        nvrhi::BindingSetItem::Texture_UAV(8, m_renderTargets->ScratchFloat1), 
        nvrhi::BindingSetItem::Texture_UAV(31, m_renderTargets->DenoiserViewspaceZ),
        nvrhi::BindingSetItem::Texture_UAV(32, m_renderTargets->DenoiserMotionVectors),
        nvrhi::BindingSetItem::Texture_UAV(33, m_renderTargets->DenoiserNormalRoughness),
        nvrhi::BindingSetItem::Texture_UAV(34, m_renderTargets->DenoiserDiffRadianceHitDist),
        nvrhi::BindingSetItem::Texture_UAV(35, m_renderTargets->DenoiserSpecRadianceHitDist),
        nvrhi::BindingSetItem::Texture_UAV(36, m_renderTargets->DenoiserDisocclusionThresholdMix),
        nvrhi::BindingSetItem::Texture_UAV(37, m_renderTargets->CombinedHistoryClampRelax),
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
        nvrhi::BindingSetItem::Texture_UAV(74, (m_renderTargets->RRTransparencyLayer!=nullptr)?m_renderTargets->RRTransparencyLayer:m_renderTargets->RRSpecMotionVectors),
        nvrhi::BindingSetItem::Texture_UAV(75, m_renderTargets->DenoiserAvgLayerRadianceHalfRes),
        nvrhi::BindingSetItem::RawBuffer_UAV(76, m_renderTargets->SubSamplingPixelList),

        // Intro Sample's GBuffer, overridden by AddCustomBindings
        nvrhi::BindingSetItem::Texture_UAV(100, m_renderTargets->OutputColor),
        nvrhi::BindingSetItem::Texture_UAV(101, m_renderTargets->OutputColor),
        nvrhi::BindingSetItem::Texture_UAV(102, m_renderTargets->OutputColor),
        nvrhi::BindingSetItem::Texture_UAV(103, m_renderTargets->OutputColor),

        nvrhi::BindingSetItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX, m_shaderDebug->GetGPUWriteBuffer()),
        nvrhi::BindingSetItem::Texture_UAV(SHADER_DEBUG_VIZ_TEXTURE_UAV_INDEX, m_shaderDebug->GetDebugVizTexture())
    };

    if (m_sampleCapabilities.UseNeeAtLightSampling)
    {
        bindingSetDescBase.bindings.push_back(nvrhi::BindingSetItem::TypedBuffer_SRV(15, m_lightsBaker->GetLightProxyCounters()));
        bindingSetDescBase.bindings.push_back(nvrhi::BindingSetItem::TypedBuffer_SRV(16, m_lightsBaker->GetLightSamplingProxies()));
        bindingSetDescBase.bindings.push_back(nvrhi::BindingSetItem::TypedBuffer_SRV(17, m_lightsBaker->GetLocalSamplingBuffer()));
        bindingSetDescBase.bindings.push_back(nvrhi::BindingSetItem::Texture_SRV(18, m_lightsBaker->GetEnvLightLookupMap()));
        bindingSetDescBase.bindings.push_back(nvrhi::BindingSetItem::TypedBuffer_SRV(19, m_lightsBaker->GetLightWeightsBoost()));
        bindingSetDescBase.bindings.push_back(nvrhi::BindingSetItem::Texture_UAV(20, m_lightsBaker->GetFeedbackTotalWeight()));
        bindingSetDescBase.bindings.push_back(nvrhi::BindingSetItem::Texture_UAV(21, m_lightsBaker->GetFeedbackCandidates()));
    }

    // NV HLSL extensions - DX12 only - we should probably expose some form of GetNvapiIsInitialized instead
    if (GetDevice()->queryFeatureSupport(nvrhi::Feature::HlslExtensionUAV))
    {
        bindingSetDescBase.bindings.push_back(
            nvrhi::BindingSetItem::TypedBuffer_UAV(NV_SHADER_EXTN_SLOT_NUM, nullptr));
    }

    // Main raytracing & etc binding set
    {
        nvrhi::BindingSetDesc bindingSetDesc;

        bindingSetDesc.bindings = bindingSetDescBase.bindings;

        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_UAV(40, m_renderTargets->DenoisingLayersHeader));
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::StructuredBuffer_UAV(42, m_renderTargets->DenoisingLayersBuffer));
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_UAV(44, m_renderTargets->StableRadiance));
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::StructuredBuffer_UAV(45, m_renderTargets->SurfaceDataBuffer));

        // Environment IBL / reflection system bindings (t80-t83)
        // Derived classes can override AddCustomBindings to provide actual textures
        // Default to black texture fallbacks (NVRHI doesn't allow null textures)
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_SRV(80, m_CommonPasses->m_BlackCubeMapArray));  // t_PrefilteredEnvCube
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_SRV(81, m_CommonPasses->m_BlackCubeMapArray));  // t_EnvIrradianceCube
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_SRV(82, m_CommonPasses->m_BlackTexture));  // t_SSRBlurChain
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_SRV(83, (m_envMapBaker->GetBRDFLUT()!=nullptr)?m_envMapBaker->GetBRDFLUT():m_CommonPasses->m_BlackTexture ));  // t_BRDFLUT
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_SRV(84, m_CommonPasses->m_BlackTexture));  // t_DepthHierarchy placeholder

        
        // SSR result UAV (depth hierarchy UAVs u80-84 are in a dedicated binding set in IntroSample)
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_UAV(85, m_renderTargets->Depth));   // u_SSRResult placeholder

        // Previous frame depth (default to black = zero depth; overridden by AddCustomBindings)
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_SRV(87, m_CommonPasses->m_BlackTexture));  // t_PrevDepth placeholder
        // Previous frame packed shading normal (overridden by IntroPathTracer::AddCustomBindings)
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_SRV(95, m_CommonPasses->m_BlackTexture));  // t_PrevSpecNormal32 placeholder
        // Intro RTXDI analytic lights (overridden by IntroSample::AddCustomBindings)
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::StructuredBuffer_SRV(90, m_lightsBaker->GetLightBuffer()));    // t_IntroRTXDILightsBase placeholder
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::StructuredBuffer_SRV(91, m_lightsBaker->GetLightExBuffer()));  // t_IntroRTXDILightsEx placeholder
        // Direct lighting output (overridden by AddCustomBindings)
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_UAV(88, m_renderTargets->OutputColor));  // u_DirectLighting placeholder
        // RTGI output (overridden by AddCustomBindings)
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_UAV(89, m_renderTargets->OutputColor));  // u_RTGIOutput placeholder

        // Indirect path tracer output (overridden by AddCustomBindings). Lives at u95 - see the layout above.
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_UAV(95, m_renderTargets->OutputColor));  // u_IndirectOutput placeholder

        // SHaRC buffers (u90-u92) — placeholders overridden by the active sample's AddCustomBindings. Shared between
        // IntroSample's native SHaRCRenderer and AdvancedPathTracer's SharcIntegration. SHaRC constants live in
        // g_Const.ptConsts.sharc (b0), so there is no dedicated CB binding.
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::StructuredBuffer_UAV(90, m_renderTargets->DenoisingLayersBuffer));  // u_SHaRCHashEntries / u_SharcHashEntriesBuffer placeholder
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::StructuredBuffer_UAV(91, m_renderTargets->DenoisingLayersBuffer));  // u_SHaRCAccumulation / u_SharcAccumulationBuffer placeholder
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::StructuredBuffer_UAV(92, m_renderTargets->DenoisingLayersBuffer));  // u_SHaRCResolved / u_SharcResolvedBuffer placeholder
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::StructuredBuffer_SRV(50, m_bindlessValidationBuffer));              // t_BindlessValidationData

        // Intro ReSTIR DI placeholders (overridden by IntroPathTracer::AddCustomBindings)
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::StructuredBuffer_UAV(94, m_renderTargets->DenoisingLayersBuffer));  // u_LightReservoirs
        // Intro t_NeighborOffsets (RG8_SNORM); IntroPathTracer::AddCustomBindings overrides with m_neighborOffsetsBuffer
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::TypedBuffer_SRV(94, m_lightsBaker->GetLightProxyCounters()));
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::ConstantBuffer(12, m_constantBuffer));                         // g_IntroReSTIRConst

        // Allow derived classes to customize bindings (e.g., add reflection textures, GTAO output)
        AddCustomBindings(bindingSetDesc);

        m_bindingSetMain = m_bindingSet = GetDevice()->createBindingSet(bindingSetDesc, m_bindingLayout);

        // Variants, same layout, with some resources replaced by dummies
        auto createVariant = [&](std::initializer_list<nvrhi::IResource*> replaced)
        {
            nvrhi::BindingSetDesc desc = bindingSetDesc;
            for (nvrhi::BindingSetItem& item : desc.bindings)
                for (nvrhi::IResource* resource : replaced)
                    if (item.resourceHandle == resource)
                    {
                        assert(item.type == nvrhi::ResourceType::Texture_UAV || item.type == nvrhi::ResourceType::Texture_SRV || item.type == nvrhi::ResourceType::RawBuffer_UAV);
                        if (item.type == nvrhi::ResourceType::RawBuffer_UAV)
                            item.resourceHandle = m_dummyRawBufferUAV.Get();
                        else
                            item.resourceHandle = (item.type == nvrhi::ResourceType::Texture_UAV) ? m_prefixDummyUAV.Get() : m_CommonPasses->m_BlackTexture.Get();
                        item.format = nvrhi::Format::UNKNOWN;
                        break;
                    }
            return GetDevice()->createBindingSet(desc, m_bindingLayout);
        };

        // The rest of the frame after ShaderDebug::EndFrame (see CloseShaderDebug) can't reach ShaderDebug through this
        m_bindingSetTail = createVariant({ m_shaderDebug->GetGPUWriteBuffer().Get(), m_shaderDebug->GetDebugVizTexture().Get() });

        // Async prefix variants: both drop textures resting in RenderTarget (illegal in compute queue barriers), 'early'
        // also drops what the previous frame's DLSS-RR may still read. No prefix shader accesses the replaced slots.
        m_bindingSetPrefixEarly = m_bindingSetPrefixLate = nullptr;
        if (IsAsyncPrefixAvailable())
        {
            const RenderTargets& rt = *m_renderTargets;
            m_bindingSetPrefixLate  = createVariant({ rt.ProcessedOutputColor.Get(), rt.LdrColor.Get(), rt.LdrColorScratch.Get() });
            m_bindingSetPrefixEarly = createVariant({ rt.ProcessedOutputColor.Get(), rt.LdrColor.Get(), rt.LdrColorScratch.Get(),
                rt.OutputColor.Get(), rt.RRDiffuseAlbedo.Get(), rt.RRSpecAlbedo.Get(), rt.RRNormalsAndRoughness.Get(), rt.RRSpecMotionVectors.Get() });
        }
    }
}

std::shared_ptr<donut::engine::Material> Sample::FindMaterial(int materialID) const
{
    // if slow switch to map
    for (const auto& material : m_scene->GetSceneGraph()->GetMaterials())
        if (material->materialID == materialID)
            return material;
    return nullptr;
}

void Sample::CreateRTPipelines()
{
    // Shared post-process raygen pipelines (consumed by PostProcessPreToneMapping
    // and PostProcessPostToneMapping). Created here so any sample inherits them.
    m_ptPipelineTestRaygenPPHDR = m_ptPipelineBaker->CreateVariant("TestRaygenPP.hlsl", { ShaderMacro("PP_TEST_HDR", "1") }, "TESTRG", true);
    m_ptPipelineEdgeDetection   = m_ptPipelineBaker->CreateVariant("TestRaygenPP.hlsl", { ShaderMacro("PP_EDGE_DETECTION", "1") }, "EDGY", true);
}

void Sample::DestroyRTPipelines()
{
    m_ptPipelineBaker->ReleaseVariant(m_ptPipelineTestRaygenPPHDR);
    m_ptPipelineBaker->ReleaseVariant(m_ptPipelineEdgeDetection);
}

void Sample::Denoise(nvrhi::IFramebuffer* framebuffer)
{
    ZoneScopedN("Denoise");

    if( !m_ui.ActualUseStandaloneDenoiser() )
        return;

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

    //const auto& fbinfo = framebuffer->getFramebufferInfo();
    const char* passNames[] = { "Denoising plane 0", "Denoising plane 1", "Denoising plane 2", "Denoising plane 3" }; assert( std::size(m_nrd) <= std::size(passNames) );

    bool nrdUseRelax = m_ui.NRDMethod == NrdConfig::DenoiserMethod::RELAX;
    PostProcess::ComputePassType preparePassType = nrdUseRelax ? PostProcess::ComputePassType::RELAXDenoiserPrepareInputs : PostProcess::ComputePassType::REBLURDenoiserPrepareInputs;
    PostProcess::ComputePassType mergePassType = nrdUseRelax ? PostProcess::ComputePassType::RELAXDenoiserFinalMerge : PostProcess::ComputePassType::REBLURDenoiserFinalMerge;

    bool resetHistory = m_ui.ResetRealtimeCaches;

    int maxPassCount = std::min(m_ui.ActualDenoisingLayersActiveCount(), (int)std::size(m_nrd));
    bool initWithStableRadiance = true;
    for (int pass = maxPassCount-1; pass >= 0; pass--)
    {
        GpuProfiler::Begin(m_commandList, passNames[pass]);

        SampleMiniConstants miniConstants = { uint4((uint)pass, initWithStableRadiance?1:0, 0, 0) };
        initWithStableRadiance = false;

        // Direct inputs to denoiser are reused between passes; there's redundant copies but it makes interfacing simpler
        nvrhi::TextureDesc tdesc = m_renderTargets->OutputColor->getDesc();
        GpuProfiler::Begin(m_commandList, "PrepareInputs");
        m_postProcess->Apply(m_commandList, preparePassType, m_constantBuffer, miniConstants, m_bindingSet, m_bindingLayout, tdesc.width, tdesc.height);
        GpuProfiler::End(m_commandList);

        const float timeDeltaBetweenFrames = -1.f; // < 0: NRD tracks the frame time itself
        bool enableValidation = m_ui.DebugView == DebugViewType::DenoisingLayer_DenoiserValidation;
        if (nrdUseRelax)
        {
            m_nrd[pass]->RunDenoiserPasses(m_commandList, *m_renderTargets, pass, *m_view, *m_viewPrevious, GetFrameIndex(), m_ui.NRDDisocclusionThreshold, m_ui.NRDDisocclusionThresholdAlternate, m_ui.NRDUseAlternateDisocclusionThresholdMix, timeDeltaBetweenFrames, enableValidation, resetHistory, &m_ui.RelaxSettings);
        }
        else
        {
            m_nrd[pass]->RunDenoiserPasses(m_commandList, *m_renderTargets, pass, *m_view, *m_viewPrevious, GetFrameIndex(), m_ui.NRDDisocclusionThreshold, m_ui.NRDDisocclusionThresholdAlternate, m_ui.NRDUseAlternateDisocclusionThresholdMix, timeDeltaBetweenFrames, enableValidation, resetHistory, &m_ui.ReblurSettings);
        }

        GpuProfiler::Begin(m_commandList, "MergeOutputs");
        m_postProcess->Apply(m_commandList, mergePassType, pass, m_constantBuffer, miniConstants, m_renderTargets->OutputColor, *m_renderTargets, nullptr);
        GpuProfiler::End(m_commandList);

        GpuProfiler::End(m_commandList);
    }
}

void Sample::PostProcessAA(nvrhi::IFramebuffer* framebuffer, bool reset)
{
    ZoneScopedN("PostProcessAA");

    if (m_ui.RealtimeMode)
    {
        if (m_ui.RealtimeAA == 0)
        {
            // TODO: Remove Redundant copy for non AA case
            m_commandList->copyTexture(m_renderTargets->ProcessedOutputColor, nvrhi::TextureSlice(), m_renderTargets->OutputColor, nvrhi::TextureSlice());
        }
        else if (m_ui.RealtimeAA == 1 && m_temporalAntiAliasingPass != nullptr )
        {
            bool previousViewValid = (GetFrameIndex() != 0) && m_temporalAntiAliasingFeedbackValid;

            GpuProfiler::Begin(m_commandList, "TAA");

            m_temporalAntiAliasingPass->TemporalResolve(m_commandList, m_ui.TemporalAntiAliasingParams, previousViewValid, *m_view, *m_view);
            m_temporalAntiAliasingFeedbackValid = true;

            GpuProfiler::End(m_commandList);
        }

#if DONUT_WITH_STREAMLINE
        // SET STREAMLINE CONSTANTS
        {
            // This section of code updates the streamline constants every frame. Regardless of whether we are utilising the streamline plugins, as long as streamline is in use, we must set its constants.
            affine3 viewReprojection = m_view->GetChildView(ViewType::PLANAR, 0)->GetInverseViewMatrix() * m_viewPrevious->GetViewMatrix();
            float4x4 reprojectionMatrix = inverse(m_view->GetProjectionMatrix(false)) * affineToHomogeneous(viewReprojection) * m_viewPrevious->GetProjectionMatrix(false);
            float outputAspectRatio = m_displayAspectRatio; //windowViewport.width() / windowViewport.height();    // render and display outputs might not match in case of lower DLSS/etc resolution rounding!
            float4x4 projection = m_view->GetProjectionMatrix(false);   // built from m_cameraVerticalFOV and outputAspectRatio in UpdateViews

            float2 jitterOffset = m_view->GetPixelOffset();     // what UpdateViews applied this frame

            StreamlineInterface::Constants slConstants = {};
            slConstants.cameraAspectRatio = outputAspectRatio;
            slConstants.cameraFOV = m_cameraVerticalFOV;
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

        // DLSS-RR inputs (PrepareDLSSRRGuides) are produced in Render, at the end of the (possibly async) prefix

        m_commandList->commitBarriers();

        // TAG STREAMLINE RESOURCES
        const bool useRRCopies = UseRRDepthMVCopies();
        GetDeviceManager()->GetStreamline().TagResourcesGeneral(m_commandList,
            m_view->GetChildView(ViewType::PLANAR, 0),
            useRRCopies ? m_renderTargets->RRMotionVectors : m_renderTargets->ScreenMotionVectors,
            useRRCopies ? m_renderTargets->RRDepth : m_renderTargets->Depth,
            m_renderTargets->PreUIColor);

        // TAG STREAMLINE RESOURCES
        GetDeviceManager()->GetStreamline().TagResourcesDLSSNIS(m_commandList,
            m_view->GetChildView(ViewType::PLANAR, 0),
            m_renderTargets->ProcessedOutputColor,
            m_renderTargets->OutputColor);

        // Commit any pending resource transitions before evaluating Streamline features.
        m_commandList->commitBarriers();

        if (m_ui.RealtimeAA == 2)
        {
            RTXPT_GPU_SCOPE(m_commandList, "DLSS");

            GetDeviceManager()->GetStreamline().EvaluateDLSS(m_commandList);
            m_commandList->clearState();
        }
        if (m_ui.RealtimeAA == 3)
        {
            RTXPT_GPU_SCOPE(m_commandList, "DLSS-RR");

            static bool useSpecHitT = false; // it's buggy
            //if (ImGui::IsKeyPressed(ImGuiKey::ImGuiKey_G))
            //    gmgkmgk = !gmgkmgk;

#if 0
            GetDeviceManager()->GetStreamline().TagResourcesDLSSRR(m_commandList,
                m_view->GetChildView(ViewType::PLANAR, 0),
                (int2)m_renderSize,
                (int2)m_displaySize,
                m_renderTargets->ProcessedOutputColor,  //nvrhi::ITexture* outputColor
                m_renderTargets->OutputColor,           // nvrhi::ITexture* inputColor
                m_renderTargets->RRDiffuseAlbedo,
                m_renderTargets->RRSpecAlbedo,
                m_renderTargets->RRNormalsAndRoughness,
                nullptr,
                (useSpecHitT) ? (m_renderTargets->SpecularHitT) : (nullptr),
                (useSpecHitT) ? (nullptr) : (m_renderTargets->RRSpecMotionVectors),
                m_renderTargets->RRTransparencyLayer
#else
            GetDeviceManager()->GetStreamline().TagResourcesDLSSRR(m_commandList,
                m_view->GetChildView(ViewType::PLANAR, 0),
                (int2)m_renderSize,
                (int2)m_displaySize,
                m_renderTargets->OutputColor,           // nvrhi::ITexture* inputColor
                m_renderTargets->RRDiffuseAlbedo,
                m_renderTargets->RRSpecAlbedo,
                m_renderTargets->RRNormalsAndRoughness,
                nullptr,
                (useSpecHitT)?(m_renderTargets->SpecularHitT):(nullptr),
                (useSpecHitT)?(nullptr):(m_renderTargets->RRSpecMotionVectors),
                m_renderTargets->ProcessedOutputColor   //nvrhi::ITexture* outputColor
            );
#endif

            m_commandList->commitBarriers();

            GetDeviceManager()->GetStreamline().EvaluateDLSSRR(m_commandList);
            m_commandList->clearState();
        }
        else if ( !m_ui.ActualUseStandaloneDenoiser() )
        {
            // If all denoisers disabled, this is a pass-through that just merges and outputs noisy data
            SampleMiniConstants miniConstants = { uint4(0, 0, 0, 0) };
            nvrhi::TextureDesc tdesc = m_renderTargets->OutputColor->getDesc();
            GpuProfiler::Begin(m_commandList, "NoDenoiserFinalMerge");
            m_postProcess->Apply(m_commandList, PostProcess::ComputePassType::NoDenoiserFinalMerge, m_constantBuffer, miniConstants, m_bindingSet, m_bindingLayout, tdesc.width, tdesc.height);
            GpuProfiler::End(m_commandList);
        }
#endif
    }
    else
    {
        // Reference mode - run the accumulation pass.
        // If the sample count has reached the target, just keep copying the accumulated output.
        // If m_accumulationSampleIndex is negative - that's warm-up, just display.
        const float accumulationWeight = (m_accumulationSampleIndex < m_ui.AccumulationTarget)?(1.f / float(max(0, m_accumulationSampleIndex) + 1)):(0.0f);

        m_accumulationPass->Render(m_commandList, *m_view, *m_view, accumulationWeight);
    }

}

void Sample::DenoisedScreenshot(nvrhi::ITexture * framebufferTexture) const
{
    ZoneScoped;

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

        if (!SaveTextureToFile(GetDevice(), m_CommonPasses.get(), framebufferTexture, nvrhi::ResourceStates::Present, noisyImagePath.c_str()))
        { assert(false); return; }

        std::string startCmd = "\"" + denoiserPath + "\"" + " -hdr 0 -i \"" + noisyImagePath + "\"" " -o \"" + denoisedImagePath + "\"";
        auto [resNum, resString, errorString] =  SystemShell(startCmd.c_str());
        if (resString!="")
            donut::log::info("result: %s", resString.c_str());
        if (errorString != "")
            donut::log::info("error: %s", errorString.c_str());

        std::string viewCmd = "\"" + denoisedImagePath + "\"";
        SystemShell(viewCmd.c_str(), true);
    };
    execute("OptiX");
    execute("OIDN");
}

static float VanDerCorput(uint32_t base, uint32_t index)
{
    float result = 0.0f;
    for (float denominator = float(base); index > 0; index /= base, denominator *= base)
        result += float(index % base) / denominator;
    return result;
}

// Sample count of the standard D3D MSAA pattern used for TemporalAntiAliasingJitter::MSAA: 16 or 8 (donut's TAA uses 8)
#define RTXPT_MSAA_JITTER_SAMPLE_COUNT 16

// Same sequences as donut's TemporalAntiAliasingPass, but indexed by frame instead of stepped by the pass, so the pass
// need not exist. When TAA is active its own sequence is used.
donut::math::float2 Sample::ComputeCameraJitter(uint64_t frameIndex)
{
    if (!m_ui.RealtimeMode || m_ui.RealtimeAA == 0 || m_ui.DbgFreezeRealtimeNoiseSeed)
        return dm::float2(0,0);

    if (m_ui.RealtimeAA == 1 && m_temporalAntiAliasingPass != nullptr)
        return m_temporalAntiAliasingPass->GetCurrentPixelOffset();

    switch (m_ui.TemporalAntiAliasingJitter)
    {
        default:
        case TemporalAntiAliasingJitter::MSAA:
        {
            // standard D3D MSAA sample patterns (same as Vulkan's standard sample locations), in 1/16 pixel units
#if RTXPT_MSAA_JITTER_SAMPLE_COUNT == 16
            static const int2 offsets[] = {
                int2(1, 1), int2(-1, -3), int2(-3, 2), int2(4, -1), int2(-5, -2), int2(2, 5), int2(5, 3), int2(3, -5),
                int2(-2, 6), int2(0, -7), int2(-4, -6), int2(-6, 4), int2(-8, 0), int2(7, -4), int2(6, 7), int2(-7, -8)
            };
#elif RTXPT_MSAA_JITTER_SAMPLE_COUNT == 8
            static const int2 offsets[] = {
                int2(1, -3), int2(-1, 3), int2(5, 1), int2(-3, -5), int2(-5, 5), int2(-7, -1), int2(3, 7), int2(7, -7)
            };
#else
#error RTXPT_MSAA_JITTER_SAMPLE_COUNT must be 8 or 16
#endif
            return float2(offsets[frameIndex % RTXPT_MSAA_JITTER_SAMPLE_COUNT]) / 16.0f;
        }
        case TemporalAntiAliasingJitter::Halton:
        {
            // phases: 8 * (output / render pixel area) per DLSS guide 3.7.1.1, and at least 32 per DLSS-RR guide 3.6
            const float areaRatio = float(m_displaySize.x * m_displaySize.y) / float(m_renderSize.x * m_renderSize.y);
            const uint32_t phaseCount = std::max(32u, uint32_t(std::ceil(8.0f * areaRatio)));
            uint32_t index = uint32_t(frameIndex % phaseCount) + 1;
            return float2{ VanDerCorput(2, index), VanDerCorput(3, index) } - 0.5f;
        }
        case TemporalAntiAliasingJitter::R2:
        {
            // http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/ - in double, so it holds
            // up over long runs where donut's float accumulation drifts
            const double g = 1.32471795724474602596;
            return float2{ float(std::fmod(double(frameIndex) / g, 1.0)), float(std::fmod(double(frameIndex) / (g * g), 1.0)) } - 0.5f;
        }
        case TemporalAntiAliasingJitter::WhiteNoise:
        {
            std::mt19937 rng{ uint32_t(frameIndex) };
            std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
            return float2{ dist(rng), dist(rng) };  // braces: left-to-right evaluation
        }
    }
}
