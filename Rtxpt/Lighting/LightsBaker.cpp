/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "LightsBaker.h"

#include <donut/engine/ShaderFactory.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/TextureCache.h>

#include <donut/app/UserInterfaceUtils.h>

#include <nvrhi/utils.h>

#include <donut/app/imgui_renderer.h>

#include "../SampleCommon/SampleCommon.h"
#include "../SampleCommon/ComputePipelineBaker.h"
#include "../SampleCommon/ExtendedScene.h"
#include "../SceneRenderCache.h"
#include "../SampleCommon/PackingUtils.h"

#include "../Shaders/Libraries/LightsBaker/LightsBaker.hlsl"

#include "../Misc/ShaderDebug.h"

#include "../GPUSort/GPUSort.h"
#include "../Shaders/PathTracer/Utils/NoiseAndSequences.hlsli"

#include "Distant/EnvMapBaker.h"
#include "Distant/EnvMapImportanceSamplingBaker.h"

#include "../Materials/MaterialsBaker.h"
#include "../OpacityMicroMap/OmmBaker.h"

namespace
{
    // Tile size and local proxy count for each NEEAT_QUALITY_PRESET block in NEEATConfig.h, which C++ only sees with the default
    struct LocalSamplingPreset { uint TileSize, LocalProxyCount; };
    constexpr LocalSamplingPreset c_localSamplingPresets[] = { { 8, 128 }, { 8, 256 }, { 6, 512 } };
    static_assert( c_localSamplingPresets[NEEAT_QUALITY_PRESET].TileSize == RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE && c_localSamplingPresets[NEEAT_QUALITY_PRESET].LocalProxyCount == RTXPT_LIGHTING_LOCAL_PROXY_COUNT );
}

using namespace donut;
using namespace donut::math;
using namespace donut::engine;

LightsBaker::LightsBaker(nvrhi::IDevice* device)
    : m_device(device)
{
    SceneReloaded();

#if 0 // Switch to this when nvrhi::Feature::WaveLaneCountMinMax lands
    nvrhi::WaveLaneCountMinMaxFeatureInfo waveLaneCountMinMaxFeatureInfo;
    if (m_device->queryFeatureSupport(nvrhi::Feature::WaveLaneCountMinMax, (void*)&waveLaneCountMinMaxFeatureInfo, sizeof(waveLaneCountMinMaxFeatureInfo)))
    {
        m_deviceHas32ThreadWaves = (waveLaneCountMinMaxFeatureInfo.minWaveLaneCount == 32) && (waveLaneCountMinMaxFeatureInfo.maxWaveLaneCount == 32);
    }
#elif DONUT_WITH_DX12
    // Native DX12 version to query lane counts
    if (m_device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
    {
        ID3D12Device* d3dDevice = (ID3D12Device*)m_device->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1;
        if (d3dDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1)) == S_OK)
        {
            m_deviceHas32ThreadWaves = (options1.WaveLaneCountMin == 32) && (options1.WaveLaneCountMax == 32);
        }
    }
#endif
}

void LightsBaker::SceneReloaded() 
{ 
    m_NEE_AT_FeedbackBufferFilled = false;
    m_framesFromLastReadbackCopy = -1; 
    memset( &m_lastReadback, 0, sizeof(m_lastReadback) ); 

    // clear history
#ifdef HASH_LOOKUP_BASED_HISTORIC_LIGHT_SOURCE_MATCHING
    for( auto & m : m_historyRemapAnalyticLightIndices ) m.clear();
    for( auto & m : m_historyRemapEmissiveLightBlockOffsets ) m.clear();
#endif
    m_historicTotalLightCount = 0;
    m_historicLightClusterBase = 0;
    m_lastFrameIndex = -1;
    memset(&m_currentCtrlBuff, 0, sizeof(m_currentCtrlBuff));
    memset(&m_currentSettings, 0, sizeof(m_currentSettings));
}

LightsBaker::~LightsBaker()
{
    ReleasePasses();
}

void LightsBaker::ReleasePasses()
{
    // ComputePipelineBaker asserts on variants that were dropped without being released
    std::shared_ptr<ComputePipelineBaker> baker = m_computeBaker.lock();
    for( std::shared_ptr<ComputeShaderVariant> * pass : m_passes )
    {
        if( baker != nullptr && *pass != nullptr )
            baker->ReleaseVariant( *pass );
        *pass = nullptr;
    }
    m_passes.clear();
}

void LightsBaker::CreateRenderPasses(const std::shared_ptr<ComputePipelineBaker> & computeBaker, nvrhi::IBindingLayout* bindlessLayout, std::shared_ptr<engine::CommonRenderPasses> commonPasses, std::shared_ptr<ShaderDebug> shaderDebug, const uint2 renderResolution, const uint envMapProcessedResolution, const int qualityPreset)
{
    m_commonPasses = commonPasses;
    m_shaderDebug = shaderDebug;
    m_renderResolution = renderResolution;

    m_qualityPreset = dm::clamp( qualityPreset, 0, (int)std::size(c_localSamplingPresets)-1 );
    m_localSamplingTileSize = c_localSamplingPresets[m_qualityPreset].TileSize;

    // Layouts and passes are created once per baker and quality preset: a hot reload recompiles them in place, while
    // re-creating them on every call (resize, Ctrl+R, ...) would throw away the last good pipelines. A preset change does
    // re-create them, so the old preset's pipelines can't outlive a failed compile of the new one (see AllPassesReady).
    assert( computeBaker != nullptr );
    if( computeBaker != nullptr && (m_computeBaker.lock() != computeBaker || m_bindlessLayout.Get() != bindlessLayout || m_passesQualityPreset != m_qualityPreset) )
    {
        ReleasePasses();    // from the baker they were created on
        m_computeBaker = computeBaker;
        m_bindlessLayout = bindlessLayout;
        m_passesQualityPreset = m_qualityPreset;

        std::vector<donut::engine::ShaderMacro> shaderMacros;
        //shaderMacros.push_back(donut::engine::ShaderMacro({              "BLEND_DEBUG_BUFFER", "1" }));
        shaderMacros.push_back({ "NEEAT_QUALITY_PRESET", std::to_string(m_qualityPreset) });   // same value the path tracer gets through SetGlobalShaderMacros

        const char * shaderFile = "Libraries/LightsBaker/LightsBaker.hlsl";  // relative to Rtxpt/Shaders

        {
            nvrhi::BindingLayoutDesc layoutDesc;
            layoutDesc.visibility = nvrhi::ShaderType::Compute;
            layoutDesc.bindings = {
                nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0),      // u_controlBuffer
                nvrhi::BindingLayoutItem::StructuredBuffer_UAV(1),      // u_lightsBuffer
                nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2),      // u_lightsExBuffer
                nvrhi::BindingLayoutItem::RawBuffer_UAV(3),             // u_scratchBuffer
                nvrhi::BindingLayoutItem::TypedBuffer_UAV(4),           // u_scratchList
                nvrhi::BindingLayoutItem::TypedBuffer_UAV(5),           // u_lightWeights
                nvrhi::BindingLayoutItem::TypedBuffer_UAV(6),           // u_historyRemapCurrentToPast
                nvrhi::BindingLayoutItem::TypedBuffer_UAV(7),           // u_historyRemapPastToCurrent
                nvrhi::BindingLayoutItem::TypedBuffer_UAV(8),           // u_perLightProxyCounters
                nvrhi::BindingLayoutItem::TypedBuffer_UAV(9),           // u_lightSamplingProxies
                nvrhi::BindingLayoutItem::Texture_UAV(10),              // u_envLightLookupMap
                //nvrhi::BindingLayoutItem::TypedBuffer_UAV(11),
                nvrhi::BindingLayoutItem::Texture_UAV(11),              // u_feedbackTotalWeight
                nvrhi::BindingLayoutItem::Texture_UAV(12),              // u_feedbackCandidates
                nvrhi::BindingLayoutItem::Texture_UAV(13),              // u_feedbackTotalWeightScratch
                nvrhi::BindingLayoutItem::Texture_UAV(14),              // u_feedbackCandidatesScratch
                nvrhi::BindingLayoutItem::Texture_UAV(15),              // u_feedbackTotalWeightBlended
                nvrhi::BindingLayoutItem::Texture_UAV(16),              // u_feedbackCandidatesBlended
                nvrhi::BindingLayoutItem::Texture_UAV(17),              // u_historyDepth
                nvrhi::BindingLayoutItem::TypedBuffer_UAV(18),          // u_localSamplingBuffer
                nvrhi::BindingLayoutItem::TypedBuffer_UAV(19),          // u_lightWeightsBoost
                nvrhi::BindingLayoutItem::Texture_SRV(10),              // t_depthBuffer
                nvrhi::BindingLayoutItem::Texture_SRV(11),              // t_motionVectors
                nvrhi::BindingLayoutItem::Texture_SRV(12),              // t_envmapImportanceMap
                nvrhi::BindingLayoutItem::Sampler(0),                   // point sampler
                nvrhi::BindingLayoutItem::Sampler(1),                   // linear sampler
                nvrhi::BindingLayoutItem::Sampler(2),                   // s_MaterialSampler
                nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1),      // StructuredBuffer<SubInstanceData> t_SubInstanceData
                nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2),      // StructuredBuffer<InstanceData> t_InstanceData
                nvrhi::BindingLayoutItem::StructuredBuffer_SRV(3),      // StructuredBuffer<GeometryData> t_GeometryData
                //nvrhi::BindingLayoutItem::StructuredBuffer_SRV(4),      // geometry debug buffer not needed here?
                nvrhi::BindingLayoutItem::StructuredBuffer_SRV(5),      // StructuredBuffer<PTMaterialData> t_PTMaterialData
                nvrhi::BindingLayoutItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX),
                nvrhi::BindingLayoutItem::Texture_UAV(SHADER_DEBUG_VIZ_TEXTURE_UAV_INDEX),
            };
            m_commonBindingLayout = m_device->createBindingLayout(layoutDesc);
        }

        {   // See the member declaration for why this is not part of m_commonBindingLayout.
            nvrhi::BindingLayoutDesc layoutDesc;
            layoutDesc.visibility = nvrhi::ShaderType::Compute;
            layoutDesc.bindings = {
                nvrhi::BindingLayoutItem::StructuredBuffer_SRV(4),      // t_SubInstanceLightPatches
                nvrhi::BindingLayoutItem::StructuredBuffer_UAV(20),     // u_SubInstanceDataRW
            };
            m_subInstanceLightPatchBindingLayout = m_device->createBindingLayout(layoutDesc);
        }

        auto initPass = [&]( std::shared_ptr<ComputeShaderVariant> & pass, const char * entryPoint, const nvrhi::BindingLayoutVector & layouts )
        {
            pass = computeBaker->CreateVariant( shaderFile, entryPoint, shaderMacros, layouts, entryPoint );
            m_passes.push_back( &pass );
        };

        // These need to know about the scene
        initPass( m_bakeEmissiveTriangles,              "BakeEmissiveTriangles",            { m_commonBindingLayout, bindlessLayout } );
        initPass( m_patchSubInstanceLightData,          "PatchSubInstanceLightData",        { m_subInstanceLightPatchBindingLayout } );

        // these don't need to know anything about the scene
        const nvrhi::BindingLayoutVector commonLayouts = { m_commonBindingLayout };

        initPass( m_resetPastToCurrentHistory,          "ResetPastToCurrentHistory",        commonLayouts );

        initPass( m_envLightsBackupPast,                "EnvLightsBackupPast",              commonLayouts );
        initPass( m_envLightsSubdivideBase,             "EnvLightsSubdivideBase",           commonLayouts );
        initPass( m_envLightsSubdivideBoost,            "EnvLightsSubdivideBoost",          commonLayouts );
        initPass( m_envLightsFillLookupMap,             "EnvLightsFillLookupMap",           commonLayouts );
        initPass( m_envLightsMapPastToCurrent,          "EnvLightsMapPastToCurrent",        commonLayouts );

        initPass( m_clearFeedbackHistory,               "ClearFeedbackHistory",             commonLayouts );

        initPass( m_processFeedbackHistoryP0,           "ProcessFeedbackHistoryP0",         commonLayouts );
        initPass( m_processFeedbackHistoryP1a,          "ProcessFeedbackHistoryP1a",        commonLayouts );
        initPass( m_processFeedbackHistoryP1b,          "ProcessFeedbackHistoryP1b",        commonLayouts );
        initPass( m_processFeedbackHistoryP2,           "ProcessFeedbackHistoryP2",         commonLayouts );
        initPass( m_processFeedbackHistoryP3,           "ProcessFeedbackHistoryP3",         commonLayouts );
        initPass( m_processFeedbackHistoryDebugViz,     "ProcessFeedbackHistoryDebugViz",   commonLayouts );

        initPass( m_processFeedbackHistoryPreFilterA,   "ProcessFeedbackHistoryPreFilterA", commonLayouts );
        initPass( m_processFeedbackHistoryPreFilterB,   "ProcessFeedbackHistoryPreFilterB", commonLayouts );

        initPass( m_resetLightProxyCounters,            "ResetLightProxyCounters",          commonLayouts );
        initPass( m_computeWeights,                     "ComputeWeights",                   commonLayouts );
        initPass( m_processClusters,                    "ProcessClusters",                  commonLayouts );
        initPass( m_computeProxyCounts,                 "ComputeProxyCounts",               commonLayouts );
        initPass( m_computeProxyBaselineOffsets,        "ComputeProxyBaselineOffsets",      commonLayouts );
        initPass( m_createProxyJobs,                    "CreateProxyJobs",                  commonLayouts );
        initPass( m_executeProxyJobs,                   "ExecuteProxyJobs",                 commonLayouts );
        initPass( m_debugDrawLights,                    "DebugDrawLights",                  commonLayouts );
    }

    nvrhi::SamplerDesc samplerDesc;
    samplerDesc.setBorderColor(nvrhi::Color(0.f));
    samplerDesc.setAllFilters(true);
    samplerDesc.setMipFilter(true);
    samplerDesc.setAllAddressModes(nvrhi::SamplerAddressMode::Wrap);
    m_linearSampler = m_device->createSampler(samplerDesc);

    samplerDesc.setAllFilters(false);
    m_pointSampler = m_device->createSampler(samplerDesc);

    // destroy resources before creating to avoid lifetimes of old and new overlapping (even with itself, due to assignment operator) - avoids fragmentation and peaks
    m_controlBuffer = m_lightsBuffer = m_lightsExBuffer = m_historyRemapCurrentToPastBuffer = m_historyRemapPastToCurrentBuffer = m_scratchBuffer = m_lightWeights = m_lightWeightsBoost = m_perLightProxyCounters = m_scratchList = m_lightSamplingProxies = nullptr;
    //m_lightingConstants = nullptr;
    m_device->waitForIdle();    // make sure readback buffer is no longer used by the GPU
    m_controlBufferReadback = nullptr;

    if (m_envLightLookupMap != nullptr && m_envLightLookupMap->getDesc().width != envMapProcessedResolution)
        m_envLightLookupMap = nullptr;

    assert(renderResolution.x > 0 && renderResolution.y > 0);
    // NOTE: this compares the *allocated* texture dimensions against what the requested resolution implies. The feedback
    // textures are allocated taller than the logical resolution (prefilter ping-pong region), so the expected height must
    // be scaled to match - otherwise this triggers a full recreate + waitForIdle every single frame.
    if (m_NEE_AT_FeedbackTotalWeight == nullptr || m_NEE_AT_FeedbackTotalWeight->getDesc().width != renderResolution.x || m_NEE_AT_FeedbackTotalWeight->getDesc().height != renderResolution.y * LLB_PREPROCESS_PINGPONG_SLICES)
    {
        if (m_NEE_AT_FeedbackTotalWeight)
            m_device->waitForIdle();    // make sure none of the buffers are used by the GPU

        // destroy before creating to avoid lifetimes of old and new overlapping (even with itself, due to assignment operator) - avoids fragmentation and peaks
        m_NEE_AT_FeedbackTotalWeight = nullptr;
        m_NEE_AT_FeedbackCandidates = nullptr;
        m_NEE_AT_FeedbackTotalWeightScratch = nullptr;
        m_NEE_AT_FeedbackCandidatesScratch = nullptr;
        m_NEE_AT_FeedbackTotalWeightBlended = nullptr;
        m_NEE_AT_FeedbackCandidatesBlended = nullptr;

        m_NEE_AT_LocalSamplingBuffer = nullptr;
        m_NEE_AT_HistoryDepth = nullptr;
    }

    m_device->waitForIdle();    // make sure everything is deallocated and garbage collected

    {
        nvrhi::BufferDesc bufferDesc;
        bufferDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        bufferDesc.keepInitialState = true;
        bufferDesc.canHaveUAVs = false;

        // Main control buffer holding all constants and build metadata
        bufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        bufferDesc.keepInitialState = true;
        bufferDesc.canHaveUAVs = true;
        bufferDesc.byteSize = sizeof(NEEAT::ControlData) * 1;
        bufferDesc.structStride = sizeof(NEEAT::ControlData);
        bufferDesc.debugName = "NEEAT::ControlData";
        m_controlBuffer = m_device->createBuffer(bufferDesc);
        m_controlBufferCleared = false;

        // Lights buffer
        bufferDesc.byteSize = sizeof(PolymorphicLightInfo) * NEEAT_MAX_LIGHTS;
        bufferDesc.structStride = sizeof(PolymorphicLightInfo);
        bufferDesc.debugName = "LightsBuffer";
        m_lightsBuffer = m_device->createBuffer(bufferDesc);
        
        bufferDesc.byteSize = sizeof(PolymorphicLightInfoEx) * NEEAT_MAX_LIGHTS;
        bufferDesc.structStride = sizeof(PolymorphicLightInfoEx);
        bufferDesc.debugName = "LightsExBuffer";
        m_lightsExBuffer = m_device->createBuffer(bufferDesc);

        // Emissive triangle processing tasks buffer
        bufferDesc.structStride = 0;
        bufferDesc.byteSize = LLB_SCRATCH_BUFFER_SIZE;
        bufferDesc.canHaveRawViews = true;
        bufferDesc.debugName = "LightsScratchBuffer";
        m_scratchBuffer = m_device->createBuffer(bufferDesc);
        // CPU side scratch storage for emissive light processing
        m_scratchTaskBuffer = std::make_shared<std::vector<NEEAT::EmissiveTrianglesProcTask>>();
        m_scratchTaskBuffer->reserve(LLB_MAX_PROC_TASKS);

        // Subsequent buffers are non-structured
        bufferDesc.structStride = 0;
        bufferDesc.canHaveTypedViews = true;
        bufferDesc.canHaveRawViews = false;
        
        bufferDesc.byteSize = 2 * sizeof(float) * RTXPT_LIGHTING_WEIGHTS_COUNT_HALF;
        bufferDesc.format = nvrhi::Format::R32_FLOAT;
        bufferDesc.debugName = "LightsWeights";
        m_lightWeights = m_device->createBuffer(bufferDesc);

        bufferDesc.byteSize = sizeof(float) * RTXPT_LIGHTING_WEIGHTS_COUNT_HALF;
        bufferDesc.debugName = "LightsWeightsBoost";
        m_lightWeightsBoost = m_device->createBuffer(bufferDesc);

        bufferDesc.byteSize = sizeof(float) * (NEEAT_MAX_LIGHTS+1);
        bufferDesc.format = nvrhi::Format::R32_UINT;
        bufferDesc.debugName = "HistoryRemapCurrentToPast";
        m_historyRemapCurrentToPastBuffer = m_device->createBuffer(bufferDesc);
        bufferDesc.debugName = "HistoryRemapPastToCurrent";
        m_historyRemapPastToCurrentBuffer = m_device->createBuffer(bufferDesc);
        bufferDesc.debugName = "PerLightProxyCounters";
        m_perLightProxyCounters = m_device->createBuffer(bufferDesc);
        bufferDesc.debugName = "ScratchList";
#if NEEAT_ENABLE_ENVMAP_PROCESSING
        assert( bufferDesc.byteSize / sizeof(uint) >= (RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT*2) );    // we need at least 2 times RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT for temporary envmap quads stuff
#endif
        m_scratchList = m_device->createBuffer(bufferDesc);
        bufferDesc.byteSize = sizeof(uint) * RTXPT_LIGHTING_MAX_SAMPLING_PROXIES;
        bufferDesc.debugName = "LightSamplingProxies";
        m_lightSamplingProxies = m_device->createBuffer(bufferDesc);

        // For debugging/UI
        bufferDesc.byteSize = sizeof(NEEAT::ControlData);   // must be reset - otherwise it inherits the (huge) size of the last buffer set up above
        bufferDesc.canHaveUAVs = false;
        bufferDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
        bufferDesc.structStride = 0;
        bufferDesc.keepInitialState = false;
        bufferDesc.canHaveTypedViews = false;
        bufferDesc.initialState = nvrhi::ResourceStates::Unknown;
        bufferDesc.debugName = "LightingControlDataReadback";
        m_controlBufferReadback = m_device->createBuffer(bufferDesc);
        m_framesFromLastReadbackCopy = -1;
    }

    if (m_envLightLookupMap == nullptr)
    {
        nvrhi::TextureDesc texDesc;
        texDesc.width = envMapProcessedResolution;  texDesc.height = envMapProcessedResolution; texDesc.mipLevels = 1;
        texDesc.format = nvrhi::Format::R32_UINT; texDesc.isRenderTarget = true; texDesc.isUAV = true;
        texDesc.setInitialState(nvrhi::ResourceStates::UnorderedAccess); texDesc.keepInitialState = true;
        texDesc.debugName = "EnvLightLookupMap";

        m_envLightLookupMap = m_device->createTexture(texDesc);
    }

    assert(renderResolution.x > 0 && renderResolution.y > 0);
    if (m_NEE_AT_FeedbackTotalWeight == nullptr)
    {
        // feedback reservoirs
        nvrhi::TextureDesc desc;
        desc.width = renderResolution.x;
        desc.height = renderResolution.y;
        desc.isVirtual = false;
        desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        desc.isRenderTarget = false;
        desc.useClearValue = false;
        desc.clearValue = nvrhi::Color(0.f);
        desc.sampleCount = 1;
        desc.dimension = nvrhi::TextureDimension::Texture2D;
        desc.keepInitialState = true;
        desc.isTypeless = false;
        desc.isUAV = true;
        desc.mipLevels = 1;
        desc.format = nvrhi::Format::R32_FLOAT;
        // The two main reservoir surfaces are allocated taller than the logical resolution so the prefilter can ping-pong
        // between disjoint regions; everything else (including miniDesc below) must keep deriving from the logical 'desc'.
        nvrhi::TextureDesc pingPongDesc = desc;
        pingPongDesc.height = renderResolution.y * LLB_PREPROCESS_PINGPONG_SLICES;

        desc.debugName = "NEE_AT_HistoryDepth";
        m_NEE_AT_HistoryDepth = m_device->createTexture(desc);
        pingPongDesc.debugName = "NEE_AT_FeedbackTotalWeight";
        m_NEE_AT_FeedbackTotalWeight = m_device->createTexture(pingPongDesc);
        desc.debugName = "NEE_AT_FeedbackTotalWeightScratch";
        m_NEE_AT_FeedbackTotalWeightScratch = m_device->createTexture(desc);
        nvrhi::TextureDesc miniDesc = desc; miniDesc.width = div_ceil(desc.width, RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE); miniDesc.height = div_ceil(desc.height, RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE);
        desc.debugName = "NEE_AT_EarlyFeedbackTotalWeightScratch";
        m_NEE_AT_FeedbackTotalWeightBlended = m_device->createTexture(miniDesc);
        m_NEE_AT_FeedbackBufferFilled = false;
        desc.format = nvrhi::Format::R32_UINT;
        pingPongDesc.format = nvrhi::Format::R32_UINT;
        pingPongDesc.debugName = "NEE_AT_FeedbackCandidates";
        m_NEE_AT_FeedbackCandidates = m_device->createTexture(pingPongDesc);
        desc.debugName = "NEE_AT_FeedbackCandidatesScratch";
        m_NEE_AT_FeedbackCandidatesScratch = m_device->createTexture(desc);
        miniDesc = desc; miniDesc.width = div_ceil(desc.width, RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE); miniDesc.height = div_ceil(desc.height, RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE);
        desc.debugName = "NEE_AT_EarlyFeedbackCandidatesScratch";
        m_NEE_AT_FeedbackCandidatesBlended = m_device->createTexture(miniDesc);
    }

    {   // Laid out by the quality preset as well as the resolution
        const LocalSamplingPreset & preset = c_localSamplingPresets[m_qualityPreset];
        const int width     = dm::div_ceil(renderResolution.x, preset.TileSize) + 1;   // add border to accommodate for jitter offset for the local sampling buffers
        const int height    = dm::div_ceil(renderResolution.y, preset.TileSize) + 1;
        const int depth     = (int)preset.LocalProxyCount;
        if (m_NEE_AT_LocalSamplingBuffer == nullptr || width != m_localSamplingBufferWidth || height != m_localSamplingBufferHeight || depth != m_localSamplingBufferDepth)
        {
            m_NEE_AT_LocalSamplingBuffer = nullptr;
            m_localSamplingBufferWidth  = width;
            m_localSamplingBufferHeight = height;
            m_localSamplingBufferDepth  = depth;
            nvrhi::BufferDesc bufferDesc;
            bufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            bufferDesc.keepInitialState = true;
            bufferDesc.byteSize = sizeof(uint) * m_localSamplingBufferWidth * m_localSamplingBufferHeight * m_localSamplingBufferDepth;
            bufferDesc.canHaveUAVs = true;
            bufferDesc.canHaveTypedViews = true;
            bufferDesc.canHaveRawViews = false;
            bufferDesc.format = nvrhi::Format::R32_UINT;
            bufferDesc.debugName = "NEE_AT_LocalSamplingBuffer";
            m_NEE_AT_LocalSamplingBuffer = m_device->createBuffer(bufferDesc);
        }
    }

    m_allocatedVRAM = 0;
    m_allocatedVRAM += m_controlBuffer->getDesc().byteSize;
    m_allocatedVRAM += m_lightsBuffer->getDesc().byteSize;
    m_allocatedVRAM += m_lightsExBuffer->getDesc().byteSize;
    m_allocatedVRAM += m_scratchBuffer->getDesc().byteSize;
    m_allocatedVRAM += m_scratchList->getDesc().byteSize;
    m_allocatedVRAM += m_historyRemapCurrentToPastBuffer->getDesc().byteSize;
    m_allocatedVRAM += m_historyRemapPastToCurrentBuffer->getDesc().byteSize;
    m_allocatedVRAM += m_lightWeights->getDesc().byteSize;
    m_allocatedVRAM += m_lightWeightsBoost->getDesc().byteSize;
    m_allocatedVRAM += m_perLightProxyCounters->getDesc().byteSize;
    m_allocatedVRAM += m_lightSamplingProxies->getDesc().byteSize;
    m_allocatedVRAM += m_NEE_AT_LocalSamplingBuffer->getDesc().byteSize;
    m_allocatedVRAM += GetEstimatedTextureSize(m_NEE_AT_FeedbackTotalWeight->getDesc());
    m_allocatedVRAM += GetEstimatedTextureSize(m_NEE_AT_FeedbackCandidates->getDesc());
    m_allocatedVRAM += GetEstimatedTextureSize(m_NEE_AT_FeedbackTotalWeightScratch->getDesc());
    m_allocatedVRAM += GetEstimatedTextureSize(m_NEE_AT_FeedbackCandidatesScratch->getDesc());
    m_allocatedVRAM += GetEstimatedTextureSize(m_NEE_AT_FeedbackTotalWeightBlended->getDesc());
    m_allocatedVRAM += GetEstimatedTextureSize(m_NEE_AT_FeedbackCandidatesBlended->getDesc());
    m_allocatedVRAM += GetEstimatedTextureSize(m_envLightLookupMap->getDesc());
    m_allocatedVRAM += GetEstimatedTextureSize(m_NEE_AT_HistoryDepth->getDesc());

    SceneReloaded();
}

bool LightsBaker::CollectEnvmapLightPlaceholders(const BakeSettings & settings, NEEAT::ControlData & ctrlBuff, std::vector<PolymorphicLightInfo> & outLightBuffer, std::vector<PolymorphicLightInfoEx> & outLightExBuffer, std::vector<uint> & outLightHistoryRemapCurrentToPastBuffer, std::vector<uint> & outLightHistoryRemapPastToCurrent)
{
    ZoneScoped;

#if NEEAT_ENABLE_ENVMAP_PROCESSING
    ctrlBuff.EnvmapQuadNodeCount += RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT;
    ctrlBuff.TotalLightCount += RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT;
    assert( ctrlBuff.TotalLightCount < NEEAT_MAX_LIGHTS );

    assert( outLightBuffer.size() == 0 );
    assert( outLightExBuffer.size() == 0 );

    // insert placeholder light info
    PolymorphicLightInfo dummy; memset(&dummy, 0, sizeof(dummy));
    PolymorphicLightInfoEx dummyEx = PolymorphicLightInfoEx::empty();
    dummy.ColorTypeAndFlags = (uint32_t)PolymorphicLightType::kEnvironmentQuad << kPolymorphicLightTypeShift;   // no need to fill this, it will be completely overwritten
    outLightBuffer.insert( outLightBuffer.end(), RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT, dummy );
    outLightExBuffer.insert( outLightExBuffer.end(), RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT, dummyEx );

    outLightHistoryRemapCurrentToPastBuffer.insert(outLightHistoryRemapCurrentToPastBuffer.end(), RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT, NEEAT_INVALID_LIGHT_INDEX);
    outLightHistoryRemapPastToCurrent.insert(outLightHistoryRemapPastToCurrent.end(), ctrlBuff.EnvmapQuadNodeCount, NEEAT_INVALID_LIGHT_INDEX);
    return ctrlBuff.TotalLightCount < NEEAT_MAX_LIGHTS;
#else
    return true;
#endif
}

// #ifdef _DEBUG
// #pragma optimize("gt", on)
// #endif
bool LightsBaker::CollectAnalyticLightsCPU(const BakeSettings & settings, const std::shared_ptr<ExtendedScene> & scene, NEEAT::ControlData & ctrlBuff, std::vector<PolymorphicLightInfo> & outLightBuffer, std::vector<PolymorphicLightInfoEx> & outLightExBuffer, std::vector<uint> & outLightHistoryRemapCurrentToPastBuffer, std::vector<uint> & outLightHistoryRemapPastToCurrent)
{
    ZoneScoped;

    bool allGood = true;
    const auto & allLights = scene->GetSceneGraph()->GetLights();

    for ( auto light : allLights )
    {
        if (outLightBuffer.size() >= NEEAT_MAX_LIGHTS)
        {
            assert(false); // no more room for lights!
            allGood = false;
            break;
        }

        std::shared_ptr<LightExtension> lightEx = std::dynamic_pointer_cast<LightExtension>(light);
        if (lightEx == nullptr)
            continue; // that's ok, might be directional or environment

        switch (( light->GetLightType() ))
        {
        case LightType_Spot:
        case LightType_Point:
        {    
            LightSamplerLink& pastLink = lightEx->LightLink;
            if (pastLink.IndexOrBase != -1 && pastLink.LastUpdateTag != m_lastFrameIndex) // if not used specifically during last frame, 
                pastLink.IndexOrBase = -1;
            pastLink.LastUpdateTag = settings.FrameIndex;

            PolymorphicLightInfoFull lightPackedFull = PackingUtils::ConvertLight(*light);
            outLightBuffer.push_back( lightPackedFull.Base );
            outLightExBuffer.push_back( lightPackedFull.Extended );
            
            // we do this to see if we had this light in the previous frame and mark the index it had; we could add more than a memory pointer to hash, as memory locations can be reused for new lights
            size_t lightHash = reinterpret_cast<size_t>(light.get());
            
#ifdef HASH_LOOKUP_BASED_HISTORIC_LIGHT_SOURCE_MATCHING
            uint historicIndex = NEEAT_INVALID_LIGHT_INDEX;
            // Read last frame's map; absence means the light was not present last frame.
            auto entry = HistoryAnalyticPrev().find(lightHash);
            if( entry != HistoryAnalyticPrev().end() )
                historicIndex = entry->second;
            // Always record into this frame's map (see FlipHistoryRemapBuffers) - lights should be unique.
            HistoryAnalyticCurr().insert_or_assign( lightHash, ctrlBuff.TotalLightCount );
            assert(historicIndex == pastLink.IndexOrBase);
#else
            uint historicIndex = pastLink.IndexOrBase;
#endif
            outLightHistoryRemapCurrentToPastBuffer.push_back(historicIndex);
            pastLink.IndexOrBase = ctrlBuff.TotalLightCount;

            
            // ComputeBounds( ctrlBuff, outLightBuffer.back().Center );
            ctrlBuff.AnalyticLightCount++;
            ctrlBuff.TotalLightCount++;
        } break;
        default: break;
        }
    }

    // use current-to-past to create past-to-current: 1st init past-to-current values to invalid; then fill them up for those we can find historic match
    uint startingLight = (uint)outLightHistoryRemapPastToCurrent.size(); 
#if NEEAT_ENABLE_ENVMAP_PROCESSING
    assert( startingLight == RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT ); // we know we should have envmap placeholders set up before so do sanity check
#endif
    outLightHistoryRemapPastToCurrent.insert(outLightHistoryRemapPastToCurrent.end(), ctrlBuff.AnalyticLightCount, NEEAT_INVALID_LIGHT_INDEX);
    for( uint lightIndex = startingLight; lightIndex < outLightHistoryRemapCurrentToPastBuffer.size(); lightIndex++ )
    {
        uint historicIndex = outLightHistoryRemapCurrentToPastBuffer[lightIndex];
        if( historicIndex != NEEAT_INVALID_LIGHT_INDEX )
        {
            // Past index can exceed this frame's analytic range if lights were removed since the last bake
            assert( historicIndex < NEEAT_MAX_LIGHTS );
            if( historicIndex >= outLightHistoryRemapPastToCurrent.size() )
                outLightHistoryRemapPastToCurrent.resize( historicIndex + 1, NEEAT_INVALID_LIGHT_INDEX );
            outLightHistoryRemapPastToCurrent[historicIndex] = lightIndex;
        }
    }

    assert(outLightBuffer.size() == outLightHistoryRemapCurrentToPastBuffer.size());
    return allGood;
};

// #ifdef _DEBUG
// #pragma optimize("gt", on)
// #endif
bool LightsBaker::ProcessEmissiveGeometry( const BakeSettings & settings, const std::shared_ptr<ExtendedScene> & scene, std::vector<SubInstanceData> & subInstanceData, NEEAT::ControlData & ctrlBuff, std::vector<struct NEEAT::EmissiveTrianglesProcTask> & tasks, SceneRenderCache & sceneRenderCache )
{
    ZoneScoped;

    bool allGood = true;

    tasks.clear();
    assert( m_subInstanceLightPatches.size() == 0 );

    assert( ctrlBuff.TotalLightCount == (ctrlBuff.AnalyticLightCount+ctrlBuff.EnvmapQuadNodeCount) );


    const auto& instances = scene->GetSceneGraph()->GetMeshInstances();
    // Flat views built once (in parallel) by SceneRenderCache. The two casts they replace - the MeshInstance* ->
    // MeshInstanceExtension* cross-cast and PTMaterial::SafeCast - were together ~57% of this function's cost when done
    // per instance/sub-instance here. This loop stays serial: ctrlBuff.TotalLightCount is a running light-index
    // allocator feeding an order-dependent task list, so it cannot be chunked without a count/prefix-sum/emit rewrite.
    const auto & instanceInfos    = sceneRenderCache.GetInstanceInfos();
    const auto & subInstanceInfos = sceneRenderCache.GetSubInstanceInfos();
    assert( instanceInfos.size() == instances.size() );
    assert( subInstanceInfos.size() == subInstanceData.size() );

    // Only visit sub-instances that can actually do work here - emissive geometry is very sparse, and this loop cannot be
    // parallelised (ctrlBuff.TotalLightCount is a running light-index allocator feeding an order-dependent task list), so
    // shrinking what it walks is the only lever. The list is built during the cache's parallel fill and is strictly
    // ascending, i.e. the same order a full scan would visit - which keeps light-index allocation deterministic.
    const auto & lightRelevant = sceneRenderCache.GetLightRelevantSubInstances();

#ifdef _DEBUG
    {   // Drift guard: if this loop's gates and the cache's predicate ever diverge, geometry is silently skipped and
        // lights just go missing - no crash. Re-derive over EVERY sub-instance and check nothing was wrongly excluded.
        size_t expected = 0;
        for( size_t i = 0; i < instanceInfos.size(); i++ )
        {
            const SceneRenderCache::InstanceInfo & info = instanceInfos[i];
            for( uint32_t g = 0; g < info.GeometryCount; g++ )
            {
                const PTMaterial * mat = subInstanceInfos[ info.FirstGeometryInstanceIndex + g ].MaterialPT;
                if( mat == nullptr )
                    continue;
                const bool qualifies = mat->EnableAsAnalyticLightProxy || ( mat->IsEmissiveSurfaceLightSampling() && !mat->SkipRender );
                if( qualifies )
                    expected++;
            }
        }
        assert( expected == lightRelevant.size() && "SceneRenderCache light-relevant predicate has drifted from ProcessEmissiveGeometry's gates" );
    }
#endif

    for( const SceneRenderCache::LightRelevantSubInstance & entry : lightRelevant )
    {
        const size_t instanceArrayIndex = entry.InstanceArrayIndex;
        const size_t geometryIndex      = entry.GeometryIndex;
        const uint   subInstanceIndex   = entry.SubInstanceIndex;

        const auto& instance = instances[instanceArrayIndex];
        const SceneRenderCache::InstanceInfo & instanceInfo = instanceInfos[instanceArrayIndex];

        // cached cross-cast (MeshInstanceExtension is a separate base, so this cannot be a static_cast)
        MeshInstanceExtension* instanceEx = instanceInfo.InstanceEx;
        if( instanceEx == nullptr )
        {
            assert( false && "Something is wrong with types" );
            continue;
        }
        std::vector<LightSamplerLink> * perGeometryLightSamplerLinks = &(instanceEx->PerGeometryLightSamplerLinks);

        if( (*perGeometryLightSamplerLinks).size() != instanceInfo.GeometryCount )
        {
            // MeshInstanceEx constructor should have initialized this
            assert(false);
            continue;
        }

        {
            assert( subInstanceIndex == instanceInfo.FirstGeometryInstanceIndex + geometryIndex );
            const SceneRenderCache::SubInstanceInfo & subInfo = subInstanceInfos[subInstanceIndex];
            const donut::engine::MeshGeometry * geometry = subInfo.Geometry;

            PTMaterial & materialPT = *subInfo.MaterialPT;   // cached by SceneRenderCache

            // this has nothing to do with emissive materials, it's instead a mechanism used to evaluate specific analytic light after hitting this mesh
            if (materialPT.EnableAsAnalyticLightProxy)
            {
                static_assert( NEEAT_INVALID_LIGHT_INDEX == 0xFFFFFFFF ); // check that it matches SubInstanceData Default
                uint analyticProxyLightIndex = NEEAT_INVALID_LIGHT_INDEX;
                // this is the first way to set proxy lights
                // Not sceneRenderCache.GetMeshInstanceNodes(): with the default GPU TLAS build nothing else fills it, and
                // a whole-scene fill every frame costs far more than locking the node for the few proxy sub-instances.
                SceneGraphNode * parent = instance->GetNode()->GetParent();
                if (parent != nullptr && parent->GetLeaf() != nullptr && parent->GetLeaf()->GetContentFlags() == SceneContentFlags::Lights )
                {
                    auto light = std::dynamic_pointer_cast<Light>(parent->GetLeaf());
                    auto lightEx = std::dynamic_pointer_cast<LightExtension>(parent->GetLeaf());
                    if (light != nullptr && (light->GetLightType() == LightType_Spot || light->GetLightType() == LightType_Point) )
                    {
#ifdef HASH_LOOKUP_BASED_HISTORIC_LIGHT_SOURCE_MATCHING
                        // NOTE: Curr(), not Prev() - unlike the other two sites this is NOT a history lookup. It reads
                        // the index CollectAnalyticLightsCPU wrote earlier in this same frame (see the call order in
                        // UpdateBegin), which is why the non-hash path below tests LastUpdateTag == settings.FrameIndex
                        // and takes the current index rather than the previous one.
                        size_t lightHash = reinterpret_cast<size_t>(light.get());
                        auto entry = HistoryAnalyticCurr().find(lightHash);
                        if (entry != HistoryAnalyticCurr().end())
                            analyticProxyLightIndex = entry->second;
                        assert(lightEx->LightLink.LastUpdateTag == settings.FrameIndex && lightEx->LightLink.IndexOrBase == analyticProxyLightIndex);
#else
                        if (lightEx->LightLink.LastUpdateTag == settings.FrameIndex)
                            analyticProxyLightIndex = lightEx->LightLink.IndexOrBase;
#endif
                    }
                }

                // this is the second way to set proxy lights - via light marking the nodes
                if (analyticProxyLightIndex == NEEAT_INVALID_LIGHT_INDEX )
                {
                    auto proxyLight = instanceEx->ProxiedAnalyticLight.lock();
                    if (proxyLight != nullptr && proxyLight->LightLink.LastUpdateTag == settings.FrameIndex)
                        analyticProxyLightIndex = proxyLight->LightLink.IndexOrBase;
                }

                // now set the convertedLightIndex into subInstanceData - if NEEAT_INVALID_LIGHT_INDEX that's fine, nothing happens
#if RTXPT_SUBINSTANCE_LIGHT_PATCH_PASS
                assert( subInstanceIndex < SUBINSTANCE_LIGHT_PATCH_INDEX_MASK );    // top value is reserved, see SUBINSTANCE_LIGHT_PATCH_SKIP
                m_subInstanceLightPatches.push_back( { subInstanceIndex, analyticProxyLightIndex } );
#else
                subInstanceData[subInstanceIndex].AnalyticProxyLightIndex = analyticProxyLightIndex;
#endif
            }

            // Not emissive (or skipped) - nothing below applies, so bail before the light-index allocation work.
            // Note this deliberately does NOT reset pastLink.IndexOrBase: the staleness guard below does that on the
            // frame the geometry becomes emissive again, since LastUpdateTag is left behind here too.
            if (!materialPT.IsEmissiveSurfaceLightSampling() || materialPT.SkipRender)
                continue;

            LightSamplerLink& pastLink = (*perGeometryLightSamplerLinks)[geometryIndex];
            if (pastLink.IndexOrBase != -1 && pastLink.LastUpdateTag != m_lastFrameIndex) // if not used specifically during last frame, 
                pastLink.IndexOrBase = -1;
            pastLink.LastUpdateTag = settings.FrameIndex;

            // Both budgets are checked before anything is committed, so a geometry gets all of its lights and tasks or none.
            // Checking the task budget inside the chunking loop instead left it half-baked and aborted every later sub-instance.
            const uint   triangleCount = geometry->numIndices / 3;
            const uint   clusterCount  = (triangleCount > LLB_CLUSTER_SKIP_TRIANGLE_COUNT) ? (uint)div_ceil( (int)triangleCount, LLB_MAX_CLUSTER_SIZE ) : 0;
            const size_t taskCount     = (size_t)div_ceil( (int)triangleCount, LLB_MAX_TRIANGLES_PER_TASK );
            const bool   taskOverflow  = tasks.size() + taskCount > LLB_MAX_PROC_TASKS;
            assert( !taskOverflow && "Emissive triangle task buffer too small" );
            bool overflow = (ctrlBuff.TotalLightCount + ctrlBuff.LightClusterCount + triangleCount + clusterCount >= NEEAT_MAX_LIGHTS) || taskOverflow;    // clusters join TotalLightCount after the loop
            allGood &= !overflow;

            // Only 'overflow' can be true here - the non-emissive / SkipRender cases already returned via the early
            // continue above. Kept as a separate branch because it must still release the historic link: an overflowing
            // geometry emits no lights this frame, so any recorded offset from a previous frame is no longer valid.
            if (overflow)
            {
#ifdef HASH_LOOKUP_BASED_HISTORIC_LIGHT_SOURCE_MATCHING
                // nothing to erase - simply not writing this hash into the current map drops it next frame
#endif
                pastLink.IndexOrBase = -1;
                continue;
            }

#ifdef HASH_LOOKUP_BASED_HISTORIC_LIGHT_SOURCE_MATCHING
            size_t instanceHash = 0;
            nvrhi::hash_combine(instanceHash, instance.get());
            nvrhi::hash_combine(instanceHash, geometryIndex);

            uint historicBufferOffset = NEEAT_INVALID_LIGHT_INDEX;
            // Read last frame's map; absence means "not emissive last frame", so no valid history.
            auto entry = HistoryEmissivePrev().find(instanceHash);
            if (entry != HistoryEmissivePrev().end())
                historicBufferOffset = entry->second;
            // Always record into this frame's map - it starts empty, so anything not written here simply will not exist
            // next frame. That is what makes the structure self-cleaning.
            HistoryEmissiveCurr().insert_or_assign(instanceHash, ctrlBuff.TotalLightCount);
            assert( historicBufferOffset == pastLink.IndexOrBase );
#else
            uint historicBufferOffset = pastLink.IndexOrBase;
#endif
            pastLink.IndexOrBase = ctrlBuff.TotalLightCount;

#if RTXPT_SUBINSTANCE_LIGHT_PATCH_PASS
            assert( subInstanceIndex < SUBINSTANCE_LIGHT_PATCH_INDEX_MASK );    // top value is reserved, see SUBINSTANCE_LIGHT_PATCH_SKIP
            m_subInstanceLightPatches.push_back( { subInstanceIndex | SUBINSTANCE_LIGHT_PATCH_EMISSIVE_FLAG, ctrlBuff.TotalLightCount } );
#else
            subInstanceData[subInstanceIndex].EmissiveLightMappingOffset = ctrlBuff.TotalLightCount;
#endif

            assert(geometryIndex < 0xfff);

            int triangleFrom = 0;
            int remainingTriangles = geometry->numIndices / 3;
            assert( geometry->numIndices % 3 == 0 );

            // Clusters go into their own block after all triangles; offsets are relative to it until the post-pass below
            const uint clusterBlockOffset = ctrlBuff.LightClusterCount;
            uint historicClusterOffset = NEEAT_INVALID_LIGHT_INDEX;
            if( clusterCount > 0 && historicBufferOffset != NEEAT_INVALID_LIGHT_INDEX && pastLink.ClusterIndexOrBase != -1 )
                historicClusterOffset = m_historicLightClusterBase + (uint)pastLink.ClusterIndexOrBase;
            pastLink.ClusterIndexOrBase = (clusterCount > 0) ? (int)clusterBlockOffset : -1;

            while( remainingTriangles > 0 )
            {
                NEEAT::EmissiveTrianglesProcTask task;

                task.InstanceIndex      = instance->GetInstanceIndex();
                task.GeometryIndex      = (uint)geometryIndex;
                task.TriangleIndexFrom  = triangleFrom;
                int taskTriangleCount   = std::min( remainingTriangles, LLB_MAX_TRIANGLES_PER_TASK );
                task.TriangleIndexTo    = task.TriangleIndexFrom + taskTriangleCount;
                triangleFrom += taskTriangleCount;
                task.DestinationBufferOffset = ctrlBuff.TotalLightCount;
                task.HistoricBufferOffset = historicBufferOffset;
                task.ClusterBufferOffset = (clusterCount > 0) ? (clusterBlockOffset + task.TriangleIndexFrom / LLB_MAX_CLUSTER_SIZE) : NEEAT_INVALID_LIGHT_INDEX;
                task.HistoricClusterBufferOffset = (historicClusterOffset != NEEAT_INVALID_LIGHT_INDEX) ? (historicClusterOffset + task.TriangleIndexFrom / LLB_MAX_CLUSTER_SIZE) : NEEAT_INVALID_LIGHT_INDEX;

                tasks.push_back(task);

                remainingTriangles -= taskTriangleCount;
                ctrlBuff.TotalLightCount += taskTriangleCount;
                ctrlBuff.TriangleLightCount += taskTriangleCount;

                if( historicBufferOffset != NEEAT_INVALID_LIGHT_INDEX )
                    historicBufferOffset += taskTriangleCount;
            }

            ctrlBuff.LightClusterCount += clusterCount;
        }
    }

    // The cluster block starts right after the last triangle - make the tasks' cluster offsets absolute
    const uint clusterBlockBase = ctrlBuff.TotalLightCount;
    assert( clusterBlockBase == ctrlBuff.LightClusterOffset() );
    for( NEEAT::EmissiveTrianglesProcTask & task : tasks )
        if( task.ClusterBufferOffset != NEEAT_INVALID_LIGHT_INDEX )
            task.ClusterBufferOffset += clusterBlockBase;
    ctrlBuff.TotalLightCount += ctrlBuff.LightClusterCount;
    m_historicLightClusterBase = clusterBlockBase;

    // NOTE: the old "subInstanceData.size() == subInstanceIndex" assert is gone - it relied on visiting every
    // sub-instance, which we deliberately no longer do. The equivalent coverage now lives in the drift guard above and
    // in SceneRenderCache::ValidateLightRelevantSubInstances().
    return allGood;
}

bool LightsBaker::TotalLightCountOverflow() const
{
    return !m_noOverflow;
}

void LightsBaker::ApplySubInstanceLightPatches(nvrhi::ICommandList* commandList, donut::engine::BindingCache & bindingCache, nvrhi::BufferHandle subInstanceDataBuffer)
{
    ZoneScoped;

    const uint32_t patchCount = (uint32_t)m_subInstanceLightPatches.size();
    if( patchCount == 0 || subInstanceDataBuffer == nullptr || m_subInstanceLightPatchBindingLayout == nullptr )
        return;

    RTXPT_GPU_SCOPE(commandList, "PatchSubInstanceLightData");

    // Pad to a whole number of thread groups so the pass needs no count: every thread reads a real record, and the
    // padding ones opt out. Costs at most SUBINSTANCE_LIGHT_PATCH_THREADS-1 records in the upload.
    const uint32_t paddedCount = div_ceil((int)patchCount, SUBINSTANCE_LIGHT_PATCH_THREADS) * SUBINSTANCE_LIGHT_PATCH_THREADS;
    m_subInstanceLightPatches.resize( paddedCount, SubInstanceLightPatch{ SUBINSTANCE_LIGHT_PATCH_SKIP, 0 } );

    if( paddedCount > m_subInstanceLightPatchCapacity )
    {
        // Grow-only: the count tracks emissive/proxy geometry, which is stable frame to frame in practice.
        uint32_t capacity = std::max( 1024u, m_subInstanceLightPatchCapacity );
        while( capacity < paddedCount )
            capacity *= 2;

        nvrhi::BufferDesc desc;
        desc.byteSize           = sizeof(SubInstanceLightPatch) * (size_t)capacity;
        desc.structStride       = sizeof(SubInstanceLightPatch);
        desc.debugName          = "SubInstanceLightPatches";
        desc.initialState       = nvrhi::ResourceStates::NonPixelShaderResource;
        desc.keepInitialState   = true;
        m_subInstanceLightPatchBuffer = m_device->createBuffer(desc);
        m_subInstanceLightPatchCapacity = capacity;
    }

    // Both bound handles can change under us - the list buffer on growth, the target on a scene or accel struct
    // rebuild - and the cache keys on the layout too, so a hot-reloaded layout gets its own entry.
    nvrhi::BindingSetDesc setDesc;
    setDesc.bindings = {
        nvrhi::BindingSetItem::StructuredBuffer_SRV(4, m_subInstanceLightPatchBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(20, subInstanceDataBuffer),
    };
    nvrhi::BindingSetHandle bindingSet = bindingCache.GetOrCreateBindingSet(setDesc, m_subInstanceLightPatchBindingLayout);
    if( bindingSet == nullptr )
        { assert(false); return; }

    commandList->writeBuffer( m_subInstanceLightPatchBuffer, m_subInstanceLightPatches.data(), sizeof(SubInstanceLightPatch) * (size_t)paddedCount );

    m_patchSubInstanceLightData->Dispatch( commandList, paddedCount / SUBINSTANCE_LIGHT_PATCH_THREADS, 1, 1, { bindingSet });

    m_subInstanceLightPatches.clear();
}

void LightsBaker::FillBindings(nvrhi::BindingSetDesc& outBindingSetDesc, const std::shared_ptr<ExtendedScene>& scene, std::shared_ptr<class MaterialsBaker> materialsBaker, std::shared_ptr<OmmBaker> ommBaker, nvrhi::BufferHandle subInstanceDataBuffer,
nvrhi::TextureHandle depthBuffer, nvrhi::TextureHandle motionVectors, nvrhi::TextureHandle envMapProcessed)
{
    if( depthBuffer == nullptr )
        depthBuffer = ((nvrhi::TextureHandle)m_commonPasses->m_BlackTexture.Get());
    if (motionVectors == nullptr)
        motionVectors = ((nvrhi::TextureHandle)m_commonPasses->m_BlackTexture.Get());
    nvrhi::TextureHandle envMapRadianceAndImportanceMap = envMapProcessed;
    if (envMapRadianceAndImportanceMap == nullptr || !m_currentSettings.EnvMapParams.Enabled )
        envMapRadianceAndImportanceMap = ((nvrhi::TextureHandle)m_commonPasses->m_BlackTexture.Get());

    outBindingSetDesc.bindings = {
            //nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer),
            //nvrhi::BindingSetItem::PushConstants(1, sizeof(SampleMiniConstants)),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_controlBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(1, m_lightsBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(2, m_lightsExBuffer),
            nvrhi::BindingSetItem::RawBuffer_UAV(3, m_scratchBuffer),
            nvrhi::BindingSetItem::TypedBuffer_UAV(4, m_scratchList),
            nvrhi::BindingSetItem::TypedBuffer_UAV(5, m_lightWeights),
            nvrhi::BindingSetItem::TypedBuffer_UAV(6, m_historyRemapCurrentToPastBuffer),
            nvrhi::BindingSetItem::TypedBuffer_UAV(7, m_historyRemapPastToCurrentBuffer),
            nvrhi::BindingSetItem::TypedBuffer_UAV(8, m_perLightProxyCounters),
            nvrhi::BindingSetItem::TypedBuffer_UAV(9, m_lightSamplingProxies),
            nvrhi::BindingSetItem::Texture_UAV(10, m_envLightLookupMap),
            //nvrhi::BindingSetItem::TypedBuffer_UAV(11, ),
            nvrhi::BindingSetItem::Texture_UAV(11, m_NEE_AT_FeedbackTotalWeight ),
            nvrhi::BindingSetItem::Texture_UAV(12, m_NEE_AT_FeedbackCandidates ),
            nvrhi::BindingSetItem::Texture_UAV(13, m_NEE_AT_FeedbackTotalWeightScratch ),
            nvrhi::BindingSetItem::Texture_UAV(14, m_NEE_AT_FeedbackCandidatesScratch ),
            nvrhi::BindingSetItem::Texture_UAV(15, m_NEE_AT_FeedbackTotalWeightBlended ),
            nvrhi::BindingSetItem::Texture_UAV(16, m_NEE_AT_FeedbackCandidatesBlended ),
            nvrhi::BindingSetItem::Texture_UAV(17, m_NEE_AT_HistoryDepth ),
            nvrhi::BindingSetItem::TypedBuffer_UAV(18, m_NEE_AT_LocalSamplingBuffer),
            nvrhi::BindingSetItem::TypedBuffer_UAV(19, m_lightWeightsBoost),
            nvrhi::BindingSetItem::Texture_SRV(10, depthBuffer), //((nvrhi::TextureHandle)m_NEE_AT_FeedbackBuffer.Get())),
            nvrhi::BindingSetItem::Texture_SRV(11, motionVectors),
            nvrhi::BindingSetItem::Texture_SRV(12, envMapRadianceAndImportanceMap),
            nvrhi::BindingSetItem::Sampler(0, m_pointSampler),
            nvrhi::BindingSetItem::Sampler(1, m_linearSampler),
            nvrhi::BindingSetItem::Sampler(2, m_commonPasses->m_AnisotropicWrapSampler),    // s_MaterialSampler
            nvrhi::BindingSetItem::StructuredBuffer_SRV(1, subInstanceDataBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(2, scene->GetInstanceBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(3, scene->GetGeometryBuffer()),
            //nvrhi::BindingSetItem::StructuredBuffer_SRV(4, ommBaker->GetGeometryDebugBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(5, materialsBaker->GetMaterialDataBuffer()),
            nvrhi::BindingSetItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX, m_shaderDebug->GetGPUWriteBuffer()),
            nvrhi::BindingSetItem::Texture_UAV(SHADER_DEBUG_VIZ_TEXTURE_UAV_INDEX, m_shaderDebug->GetDebugVizTexture()),
    };
}

void LightsBaker::UpdateFrustumConsts(NEEAT::LightsBakerConstants & outConsts, const LightsBaker::BakeSettings & settings)
{
    float4 frustPlanes[6];

    auto vp = [&settings](int row, int col) { return settings.ViewProjMatrix.col(col)[row]; };
    // Left clipping plane
    frustPlanes[0] = float4( vp(0, 3) + vp(0, 0), vp(1, 3) + vp(1, 0), vp(2, 3) + vp(2, 0), -(vp(3, 3) + vp(3, 0)));
    // Right clipping plane
    frustPlanes[1] = float4( vp(0, 3) - vp(0, 0), vp(1, 3) - vp(1, 0), vp(2, 3) - vp(2, 0), -(vp(3, 3) - vp(3, 0)));
    // Top clipping plane
    frustPlanes[2] = float4( vp(0, 3) - vp(0, 1), vp(1, 3) - vp(1, 1), vp(2, 3) - vp(2, 1), -(vp(3, 3) - vp(3, 1)));
    // Bottom clipping plane
    frustPlanes[3] = float4( vp(0, 3) + vp(0, 1), vp(1, 3) + vp(1, 1), vp(2, 3) + vp(2, 1), -(vp(3, 3) + vp(3, 1)));
    // Near clipping plane
    frustPlanes[4] = float4( vp(0, 3) - vp(0, 2), vp(1, 3) - vp(1, 2), vp(2, 3) - vp(2, 2), -(vp(3, 3) - vp(3, 2)));

    auto normalizePlane = [ ](const float4& plane)
    {
        float lengthSq = dot(plane.xyz(), plane.xyz());
        float scale = (lengthSq > 0.f ? (1.0f / sqrtf(lengthSq)) : 0);
        return plane * scale;
    };

    // Normalize the plane equations
    for (int i = 0; i < 5; i++)
        frustPlanes[i] = normalizePlane(frustPlanes[i]);

    // compute far plane with inverted near plane pushed away by DISTANT_LIGHT_DISTANCE
    frustPlanes[5] = dm::float4(-frustPlanes[4].xyz(), -frustPlanes[4].w - DISTANT_LIGHT_DISTANCE);

    // backup for debugging and sanity check and write to const buffer
    for (int i = 0; i < 6; i++)
    {
        float dist = dm::dot(frustPlanes[i].xyz(), settings.CameraPosition + settings.CameraDirection * float(DISTANT_LIGHT_DISTANCE * 0.001f) ) - frustPlanes[i].w;
        assert( dist > 0 );
        if (m_dbgFreezeFrustumUpdates)
            frustPlanes[i] = m_dbgFrozenFrustum[i];
        else
            m_dbgFrozenFrustum[i] = frustPlanes[i];
        outConsts.FrustumPlanes[i] = frustPlanes[i];
    }
    outConsts.DebugDrawFrustum  = m_dbgFreezeFrustumUpdates;

    auto getCorner = [&](int index) 
    {
        bool bone = (index & 1) != 0;
        bool btwo = (index & 2) != 0;
        const float4 & a = (bone == btwo) ? frustPlanes[1] : frustPlanes[0];
        const float4 & b = (index & 2) ? frustPlanes[3] : frustPlanes[2];
        const float4 & c = (index & 4) ? frustPlanes[5] : frustPlanes[4];

        float3x3 m = float3x3(a.xyz(), b.xyz(), c.xyz());
        float3 d = float3(a.w, b.w, c.w);
        return inverse(m) * d;
    };
    for (int i = 0; i < 8; i++ )
        outConsts.FrustumCorners[i] = float4(getCorner(i), 0);
}

void LightsBaker::UpdateLocalJitter()
{
    m_prevLocalJitter = m_localJitter;
    if (!m_dbgDebugDisableJitter)
    {
        // Advance R2 jitter sequence
        // http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/

        if ( (m_updateCounter % 1024) == 0 )
            m_localJitterF = { 0, 0 }; // not sure how long can the sequence remain high quality, so perhaps best to reset after a period

        static const float g = 1.32471795724474602596f;
        static const float a1 = 1.0f / g;
        static const float a2 = 1.0f / (g * g);
        m_localJitterF[0] = fmodf(m_localJitterF[0] + a1, 1.0f);
        m_localJitterF[1] = fmodf(m_localJitterF[1] + a2, 1.0f);

        m_localJitter = dm::clamp(uint2(m_localJitterF * (float)m_localSamplingTileSize), uint2(0, 0), uint2(m_localSamplingTileSize - 1, m_localSamplingTileSize - 1));
    }
}


#ifdef HASH_LOOKUP_BASED_HISTORIC_LIGHT_SOURCE_MATCHING
void LightsBaker::FlipHistoryRemapBuffers()
{
    // Swap roles, then empty the new 'current'. Everything written this frame becomes next frame's history; anything not
    // written is simply gone - which is the whole point, and why no per-entry erase or staleness tag is needed.
    // clear() retains bucket capacity, so this does not churn allocations, and it is O(live entries) - the same order we
    // insert each frame anyway, and deterministic rather than an occasional sweep spike.
    m_historyRemapWriteIndex = 1 - m_historyRemapWriteIndex;
    HistoryEmissiveCurr().clear();
    HistoryAnalyticCurr().clear();
}
#endif

bool LightsBaker::AllPassesReady() const
{
    // Null only until their first compile succeeds: at scene load (new baker), or after a preset change re-created them
    for( const std::shared_ptr<ComputeShaderVariant> * pass : m_passes )
        if( (*pass)->GetPipeline() == nullptr )
            return false;
    return !m_passes.empty();
}

void LightsBaker::UpdateBegin(nvrhi::ICommandList* commandList, donut::engine::BindingCache & bindingCache, const BakeSettings& _settings, double sceneTime, const std::shared_ptr<ExtendedScene>& scene, std::shared_ptr<class MaterialsBaker> materialsBaker, 
    std::shared_ptr<class OmmBaker> ommBaker, nvrhi::BufferHandle subInstanceDataBuffer, std::vector<SubInstanceData>& subInstanceData, nvrhi::TextureHandle envMapProcessed, SceneRenderCache & sceneRenderCache)
{
    ZoneScoped;

    RTXPT_GPU_SCOPE(commandList, "LightingUpdateBegin");
    // RAII_SCOPE( commandList->setEnableAutomaticBarriers(false);, commandList->setEnableAutomaticBarriers(true); );

    assert( _settings.FrameIndex != -1 && "forgot to set?" );

    // The passes are one chain over shared buffers, so while any is missing (first compile failed) skip the whole bake.
    // Nothing is touched, and the stale FrameIndex forces ResetFeedback on the first frame that does run.
    m_bakeSkipped = !AllPassesReady();
    if( m_bakeSkipped )
    {
        // Zeroed, the path tracer sees no lights (LightSampler::IsEmpty) rather than a never-written buffer (garbage on Vulkan);
        // the host copy is zeroed too, so GetAnalyticLightOffset/Count agree with it.
        if( !m_controlBufferCleared )
        {
            memset(&m_currentCtrlBuff, 0, sizeof(m_currentCtrlBuff));
            commandList->writeBuffer(m_controlBuffer, &m_currentCtrlBuff, sizeof(m_currentCtrlBuff));
            m_controlBufferCleared = true;
        }
        return;
    }

    if(_settings.ImportanceSamplingType == 2)
    {
        assert(m_verifyBeginHasMatchingEnd == 0);
        m_verifyBeginHasMatchingEnd++; 
    }

    m_ping = !m_ping;

    m_lastFrameIndex = m_currentSettings.FrameIndex;
    m_currentSettings = _settings;

    if (m_currentSettings.FrameIndex-1 != m_lastFrameIndex) // not sure if we want this
        m_currentSettings.ResetFeedback = true;

    if (m_currentSettings.ResetFeedback)
    {
        m_updateCounter = 0;
        m_localJitterF = { 0,0 };
        m_localJitter = m_prevLocalJitter = { 0,0 };
        m_NEE_AT_FeedbackBufferFilled = false;
    }

    UpdateLocalJitter();

    m_updateCounter++;

    bool lastFrameLocalSamplesAvailable = m_currentCtrlBuff.LastFrameTemporalFeedbackAvailable; // if last frame had temporal feedback, it will have had built local (tile) sampling

    m_currentSettings.GlobalTemporalFeedbackWeight  = dm::clamp(m_currentSettings.GlobalTemporalFeedbackWeight, 0.0f, 0.95f);
    m_currentSettings.LocalToGlobalSampleRatio      = dm::clamp(m_currentSettings.LocalToGlobalSampleRatio, 0.0f, 1.0f);
    if (m_currentSettings.ImportanceSamplingType != 2)  // no feedback needed if not using NEE_AT
    {
        m_currentSettings.GlobalTemporalFeedbackWeight = 0.0f;
        m_currentSettings.LocalToGlobalSampleRatio = 0.0f;
        lastFrameLocalSamplesAvailable = false;
        m_NEE_AT_FeedbackBufferFilled = false;
    }

    // Constants
    NEEAT::ControlData ctrlBuff; memset(&ctrlBuff, 0, sizeof(ctrlBuff)); 
    NEEAT::LightsBakerConstants & bakerConsts = ctrlBuff.BakerConstants;

    UpdateFrustumConsts(bakerConsts, m_currentSettings);

    bakerConsts.UpdateCounter = m_updateCounter;
    bakerConsts.EnableMotionReprojection = true && lastFrameLocalSamplesAvailable;
    bakerConsts.DepthDisocclusionThreshold   = m_depthDisocclusionThreshold;
    ctrlBuff.LocalSamplingTileJitter     = m_localJitter;
    ctrlBuff.LocalSamplingTileJitterPrev = m_prevLocalJitter;

    assert( _settings.ViewportSize.x > 0 && _settings.ViewportSize.y > 0 && _settings.PrevViewportSize.x > 0 && _settings.PrevViewportSize.y > 0 );
    bakerConsts.PrevOverCurrentViewportSize = m_currentSettings.PrevViewportSize / m_currentSettings.ViewportSize;

    bool lastFrameFeedbackAvailable = m_NEE_AT_FeedbackBufferFilled && !m_dbgDebugDisableLastFrameFeedback;
    const bool temporalFeedbackRequired = m_currentSettings.ImportanceSamplingType == 2;

    {
        if( m_currentSettings.EnvMapParams.Enabled )
        {
            assert( envMapProcessed != nullptr );
            bakerConsts.EnvMapParams = m_currentSettings.EnvMapParams;
            const float baseScale = 0.0002f;
            bakerConsts.DistantVsLocalRelativeImportance = m_currentSettings.DistantVsLocalImportanceScale * baseScale;

            bakerConsts.EnvMapImportanceMapMIPCount = envMapProcessed->getDesc().mipLevels;
            bakerConsts.EnvMapImportanceMapResolution = envMapProcessed->getDesc().width; assert( envMapProcessed->getDesc().height == bakerConsts.EnvMapImportanceMapResolution );
            assert(m_envLightLookupMap != nullptr && m_envLightLookupMap->getDesc().width == bakerConsts.EnvMapImportanceMapResolution);
        }
        else
        {
            bakerConsts.DistantVsLocalRelativeImportance = 0.0f;
            bakerConsts.EnvMapParams = NEEAT::LightsBakerEnvMapParams{ .Transform = float3x4::identity(), .InvTransform = float3x4::identity(), .ColorMultiplier = float3(1,1,1), .Enabled = 0.0f };
            bakerConsts.EnvMapImportanceMapMIPCount = 0;
            bakerConsts.EnvMapImportanceMapResolution = 0;
        }
    }

    bakerConsts.FeedbackResolution          = m_renderResolution;   // note: intentionally not derived from the feedback texture dimensions - those can be allocated larger than the logical resolution
    assert( m_NEE_AT_FeedbackCandidates->getDesc().width == m_renderResolution.x && m_NEE_AT_FeedbackCandidates->getDesc().height >= m_renderResolution.y );
    bakerConsts.BlendedFeedbackResolution   = uint2(m_NEE_AT_FeedbackCandidatesBlended->getDesc().width, m_NEE_AT_FeedbackCandidatesBlended->getDesc().height);
    uint numTotalP0ThreadCount              = div_ceil(bakerConsts.FeedbackResolution.x, LLB_NUM_COMPUTE_THREADS_2D) * div_ceil(bakerConsts.FeedbackResolution.y, LLB_NUM_COMPUTE_THREADS_2D) * LLB_NUM_COMPUTE_THREADS_2D * LLB_NUM_COMPUTE_THREADS_2D;
    ctrlBuff.TotalMaxFeedbackCount          = (lastFrameFeedbackAvailable)?(numTotalP0ThreadCount):(0);
    ctrlBuff.LocalSamplingResolution        = uint2(m_localSamplingBufferWidth, m_localSamplingBufferHeight);
    ctrlBuff.GlobalFeedbackUseWeight        = (lastFrameFeedbackAvailable) ? (m_currentSettings.GlobalTemporalFeedbackWeight): (0.0f);
    ctrlBuff.LocalToGlobalSampleRatio       = (lastFrameFeedbackAvailable) ? (m_currentSettings.LocalToGlobalSampleRatio) : (0.0f);
    bakerConsts.ReservoirHistoryDropoff     = m_advSetting_reservoirHistoryDropoff;
    bakerConsts.EnableLightClusters         = m_advSetting_LightClustersEnabled;
    bakerConsts.LightClusterLODExponent     = m_advSetting_LightClusterLODExponent;
    bakerConsts.LightClusterLODMultiplier   = m_advSetting_LightClusterLODMultiplier;
    bakerConsts.LightClusterBrightnessExponent = m_advSetting_LightClusterBrightnessExponent;
    ctrlBuff.ScreenSpaceVsWorldSpaceThreshold = m_advSetting_ScreenSpaceVsWorldSpaceThreshold;

    ctrlBuff.ImportanceSamplingType = m_currentSettings.ImportanceSamplingType;

    ctrlBuff.TileBufferHeight = ctrlBuff.LocalSamplingResolution.y;

    bakerConsts.DebugDrawType = (int)m_dbgDebugDrawType;
    bakerConsts.DebugDrawTileLights = m_dbgDebugDrawTileLightConnections;
    bakerConsts.MouseCursorPos = m_currentSettings.MouseCursorPos;
    bakerConsts.ImportanceBoostIntensityDelta = m_importanceBoost_IntensityDelta?m_importanceBoost_IntensityDeltaMul:0.0f;
    bakerConsts.ImportanceBoostFrustumMul = m_importanceBoost_Frustum?m_importanceBoost_FrustumMul:0.0f;
    bakerConsts.ImportanceBoostFrustumFadeDistance = m_importanceBoost_FrustumFadeDistance;
    ctrlBuff.LastFrameTemporalFeedbackAvailable = lastFrameFeedbackAvailable;
    bakerConsts.SceneCameraPos = m_currentSettings.CameraPosition;
    bakerConsts.SceneAverageContentsDistance = m_currentSettings.AverageContentsDistance;
    ctrlBuff.LastFrameLocalSamplesAvailable = lastFrameLocalSamplesAvailable && lastFrameFeedbackAvailable;
    ctrlBuff.LastFrameTemporalFeedbackAvailable = lastFrameFeedbackAvailable;
    ctrlBuff.TemporalFeedbackRequired = temporalFeedbackRequired;
    ctrlBuff.DbgFreezeFeedback = m_dbgFreezeFeedback;

    // clear buffers
    m_scratchLightBuffer.clear(); m_scratchLightExBuffer.clear();
    m_scratchLightHistoryRemapCurrentToPastBuffer.clear();
    m_scratchLightHistoryRemapPastToCurrentBuffer.clear();
    // collect all environment lights (create placeholders to be filled on the GPU later)
    m_noOverflow = true;
    m_noOverflow &= CollectEnvmapLightPlaceholders( m_currentSettings, ctrlBuff, m_scratchLightBuffer, m_scratchLightExBuffer, m_scratchLightHistoryRemapCurrentToPastBuffer, m_scratchLightHistoryRemapPastToCurrentBuffer );
    // collect all analytic lights
    m_noOverflow &= CollectAnalyticLightsCPU( m_currentSettings, scene, ctrlBuff, m_scratchLightBuffer, m_scratchLightExBuffer, m_scratchLightHistoryRemapCurrentToPastBuffer, m_scratchLightHistoryRemapPastToCurrentBuffer );
    // collect all emissive triangles and other geometry specific work - this builds batch jobs on the CPU that are executed on the GPU later, but at the end of this step we know the exact number of added emissive triangles (even though some might be black)
    m_noOverflow &= ProcessEmissiveGeometry(m_currentSettings, scene, subInstanceData, ctrlBuff, *m_scratchTaskBuffer, sceneRenderCache);
    bakerConsts.TriangleLightTaskCount = (int)(*m_scratchTaskBuffer).size();
#if NEEAT_ENABLE_ENVMAP_PROCESSING
    assert( ctrlBuff.EnvmapQuadNodeCount == RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT );
#else
    assert( ctrlBuff.EnvmapQuadNodeCount == 0 );
#endif
    assert( ctrlBuff.TotalLightCount == ctrlBuff.EnvmapQuadNodeCount + ctrlBuff.AnalyticLightCount + ctrlBuff.TriangleLightCount + ctrlBuff.LightClusterCount ); assert(ctrlBuff.TotalLightCount <= NEEAT_MAX_LIGHTS);
    ctrlBuff.HistoricTotalLightCount = m_historicTotalLightCount;
    m_historicTotalLightCount = ctrlBuff.TotalLightCount;

    bakerConsts.CurrentWeightsBufferOffset  = (m_ping) ? 0 : RTXPT_LIGHTING_WEIGHTS_COUNT_HALF;
    bakerConsts.HistoricWeightsBufferOffset = (m_ping) ? RTXPT_LIGHTING_WEIGHTS_COUNT_HALF : 0;

    // Constant & control buffers must go first
    {
        ZoneScopedN("ControlDataSetup");
        RTXPT_GPU_SCOPE(commandList, "ControlDataSetup");

        // upload control data
        // control buffer (used for build but also later for sampling)
        commandList->writeBuffer(m_controlBuffer, &ctrlBuff, sizeof(ctrlBuff));
        m_controlBufferCleared = false;
        m_currentCtrlBuff = ctrlBuff;
        commandList->setBufferState(m_controlBuffer, nvrhi::ResourceStates::UnorderedAccess);
    }

    // Bindings
    nvrhi::BindingSetDesc bindingSetDesc;
    FillBindings(bindingSetDesc, scene, materialsBaker, ommBaker, subInstanceDataBuffer, nullptr, nullptr, envMapProcessed);
    nvrhi::BindingSetHandle bindingSet = bindingCache.GetOrCreateBindingSet(bindingSetDesc, m_commonBindingLayout);

    nvrhi::BindingSetVector bindings = { bindingSet };
    nvrhi::BindingSetVector bindingsEx = { bindingSet, scene->GetDescriptorTable() };

    // Up to P0 the passes run in groups of independent dispatches, with one barrier (all of the set's UAVs) between groups.
    // Automatic barriers are off for the span: BakeEmissiveTriangles' root signature would re-apply the set around it.
    // Resources already in UAV get a UAV->UAV barrier only because nvrhi's per-resource enableUavBarriers defaults to true
    auto groupBarrier = [&]() { commandList->setResourceStatesForBindingSet(bindingSet); commandList->commitBarriers(); };
    commandList->setEnableAutomaticBarriers(false);

    const bool preFilter = lastFrameFeedbackAvailable && m_importanceBoost_PreFilter;
    const uint preFilterGroupsX = div_ceil(bakerConsts.FeedbackResolution.x, LLB_PREPROCESS_BLOCK_SIZE_INNER);
    const uint preFilterGroupsY = div_ceil(bakerConsts.FeedbackResolution.y, LLB_PREPROCESS_BLOCK_SIZE_INNER);

    // Group 1: reads only last frame's data
    groupBarrier();

#if NEEAT_ENABLE_ENVMAP_PROCESSING
    {
        RTXPT_GPU_SCOPE(commandList, "EnvLightsSubdivideBase");    // a single group, so it goes first
        m_envLightsSubdivideBase->Dispatch(commandList, 1, 1, 1, { bindingSet }); //the main output goes to scratchBuffer, with RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT offset and is consumed by EnvLightsBake
    }
#endif

    if (preFilter)
    {
        // First half of the pre-filter; pass B is in group 2, so the barrier between them is the group barrier
        RTXPT_GPU_SCOPE(commandList, "ProcessFeedbackHistoryPreFilterA");
        m_processFeedbackHistoryPreFilterA->Dispatch(commandList, preFilterGroupsX, preFilterGroupsY, 1, bindings);  // writes the scratch rows pass B reads
    }

    {
        // we can do this early although we might have to move it to a later location if doing multiple global updates per frame (unlikely?)
        ZoneScopedN("ResetLightProxyCounters");
        RTXPT_GPU_SCOPE(commandList, "ResetLightProxyCounters");

        const dm::uint  items = ctrlBuff.TotalLightCount+1; // WE HAVE TO HAVE +1 here to account for resetting the last entry used to count valid feedback count u_perLightProxyCounters[g_controlInfo.TotalLightCount]
        const dm::uint  itemsPerGroup = LLB_NUM_COMPUTE_THREADS;
        m_resetLightProxyCounters->Dispatch(commandList, div_ceil(items, itemsPerGroup), 1, 1, { bindingSet });
    }

    if (std::max(ctrlBuff.HistoricTotalLightCount, ctrlBuff.TotalLightCount) > 0)
    {
        // this is mostly for correctness/determinism - it will clean everything so any gaps in mapping to previous frame don't result in incorrect mapping
        ZoneScopedN("ResetPastToCurrentHistory");
        RTXPT_GPU_SCOPE(commandList, "ResetPastToCurrentHistory");
        m_resetPastToCurrentHistory->Dispatch(commandList, div_ceil(std::max(ctrlBuff.HistoricTotalLightCount, ctrlBuff.TotalLightCount), LLB_NUM_COMPUTE_THREADS), 1, 1, { bindingSet });
    }

#if NEEAT_ENABLE_ENVMAP_PROCESSING
    {
        ZoneScopedN("EnvLightsBackupPast");
        RTXPT_GPU_SCOPE(commandList, "EnvLightsBackupPast");
        m_envLightsBackupPast->Dispatch(commandList, div_ceil(RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT, LLB_NUM_COMPUTE_THREADS), 1, 1, { bindingSet });
    }
#endif

    // empty emissive and analytic lights get copied over first - they've been fully processed on the CPU
    {
        ZoneScopedN("EnvmapAndAnalyticLightBuffers");
        RTXPT_GPU_SCOPE(commandList, "EnvmapAndAnalyticLightBuffers");

        // One barrier for all five (writeBuffer doesn't transition with automatic barriers off); it also makes group 1's
        // reads of the old lights and its past to current reset finish before these overwrite them
        commandList->setBufferState(m_lightsBuffer, nvrhi::ResourceStates::CopyDest);
        commandList->setBufferState(m_lightsExBuffer, nvrhi::ResourceStates::CopyDest);
        commandList->setBufferState(m_historyRemapCurrentToPastBuffer, nvrhi::ResourceStates::CopyDest);
        commandList->setBufferState(m_historyRemapPastToCurrentBuffer, nvrhi::ResourceStates::CopyDest);
        commandList->setBufferState(m_scratchBuffer, nvrhi::ResourceStates::CopyDest);
        commandList->commitBarriers();
        assert( (int)m_scratchLightBuffer.size() == (ctrlBuff.EnvmapQuadNodeCount+ctrlBuff.AnalyticLightCount) );
        assert( (int)m_scratchLightExBuffer.size() == (ctrlBuff.EnvmapQuadNodeCount+ctrlBuff.AnalyticLightCount) );
        commandList->writeBuffer(m_lightsBuffer, m_scratchLightBuffer.data(), sizeof(PolymorphicLightInfo) * m_scratchLightBuffer.size());
        commandList->writeBuffer(m_lightsExBuffer, m_scratchLightExBuffer.data(), sizeof(PolymorphicLightInfoEx)* m_scratchLightExBuffer.size());
        commandList->writeBuffer(m_historyRemapCurrentToPastBuffer, m_scratchLightHistoryRemapCurrentToPastBuffer.data(), sizeof(uint) * m_scratchLightHistoryRemapCurrentToPastBuffer.size());
        commandList->writeBuffer(m_historyRemapPastToCurrentBuffer, m_scratchLightHistoryRemapPastToCurrentBuffer.data(), sizeof(uint) * m_scratchLightHistoryRemapPastToCurrentBuffer.size());
        if (bakerConsts.TriangleLightTaskCount > 0)    // a zero-size upload is invalid on Vulkan (vkCmdUpdateBuffer)
            commandList->writeBuffer(m_scratchBuffer, m_scratchTaskBuffer->data(), sizeof(NEEAT::EmissiveTrianglesProcTask)* bakerConsts.TriangleLightTaskCount);
    }

    // Group 2: SubdivideBoost needs SubdivideBase's seed nodes, BakeEmissiveTriangles the uploaded tasks, PreFilterB pass A's rows
    groupBarrier();

#if NEEAT_ENABLE_ENVMAP_PROCESSING
    {
        RTXPT_GPU_SCOPE(commandList, "EnvLightsSubdivideBoost");
        m_envLightsSubdivideBoost->Dispatch(commandList, RTXPT_NEEAT_ENVMAP_QT_UNBOOSTED_NODE_COUNT, 1, 1, { bindingSet }); //the main output goes to scratchBuffer, with RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT offset and is consumed by EnvLightsBake
    }
#endif

    if (bakerConsts.TriangleLightTaskCount > 0)
    {
        RTXPT_GPU_SCOPE(commandList, "BakeEmissiveTriangles");
        m_bakeEmissiveTriangles->Dispatch(commandList, bakerConsts.TriangleLightTaskCount, 1, 1, bindingsEx);   // one group per task
    }

    if (preFilter)
    {
        RTXPT_GPU_SCOPE(commandList, "ProcessFeedbackHistoryPreFilterB");
        m_processFeedbackHistoryPreFilterB->Dispatch(commandList, preFilterGroupsX, preFilterGroupsY, 1, bindings);
    }

    // Group 3: both need this frame's lights, and FillLookupMap must not overwrite the lookup map before SubdivideBoost has read it
    groupBarrier();

    // note: this has to come after all lights have been baked and remap current to past & past to current buffers are valid
    if (ctrlBuff.TotalLightCount > 0)
    {
        RTXPT_GPU_SCOPE(commandList, "ComputeWeights");

        const dm::uint  items = ctrlBuff.TotalLightCount;
        const dm::uint  itemsPerGroup = LLB_LOCAL_BLOCK_SIZE * LLB_NUM_COMPUTE_THREADS;
        m_computeWeights->Dispatch(commandList, div_ceil(items, itemsPerGroup), 1, 1, { bindingSet });
    }

#if NEEAT_ENABLE_ENVMAP_PROCESSING
    {
        RTXPT_GPU_SCOPE(commandList, "EnvLightFillLookupMap");
        m_envLightsFillLookupMap->Dispatch(commandList, RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT, 1, 1, bindings);
    }
#endif

    // Group 4: ProcessClusters needs the weights and their sum, MapPastToCurrent the new lookup map
    groupBarrier();

    // needs the lights' weights from ComputeWeights; P0 and ComputeProxyCounts below need the clusters' activity and weights.
    // When disabled, clusters keep BakeEmissiveTriangles' init state (inactive, zero weight, no flags), which is all consumers need.
    if (ctrlBuff.LightClusterCount > 0 && m_advSetting_LightClustersEnabled)
    {
        RTXPT_GPU_SCOPE(commandList, "ProcessClusters");
        m_processClusters->Dispatch(commandList, div_ceil(ctrlBuff.LightClusterCount, LLB_NUM_COMPUTE_THREADS), 1, 1, { bindingSet });
    }

#if NEEAT_ENABLE_ENVMAP_PROCESSING
    {
        RTXPT_GPU_SCOPE(commandList, "EnvLightsMapPastToCurrent");
        m_envLightsMapPastToCurrent->Dispatch(commandList, div_ceil(RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT, LLB_NUM_COMPUTE_THREADS), 1, 1, bindings);
    }
#endif

    groupBarrier();
    commandList->setEnableAutomaticBarriers(true);

    if (lastFrameFeedbackAvailable)
    {
        RTXPT_GPU_SCOPE(commandList, "ProcessFeedbackHistoryP0");

        m_processFeedbackHistoryP0->Dispatch(commandList, div_ceil(bakerConsts.FeedbackResolution.x, LLB_NUM_COMPUTE_THREADS_2D), div_ceil(bakerConsts.FeedbackResolution.y, LLB_NUM_COMPUTE_THREADS_2D), 1, bindings);

        commandList->setTextureState(m_NEE_AT_FeedbackTotalWeight, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setTextureState(m_NEE_AT_FeedbackCandidates, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setBufferState(m_controlBuffer, nvrhi::ResourceStates::UnorderedAccess);           // we've InterlockedAdd into u_controlBuffer (actually, we haven't, except in the validation verison, but leaving in for when enabling validation)
        commandList->setBufferState(m_perLightProxyCounters, nvrhi::ResourceStates::UnorderedAccess);   // we've InterlockedAdd into m_perLightProxyCounters
    }

    if (ctrlBuff.TotalLightCount > 0)
    {
        RTXPT_GPU_SCOPE(commandList, "ComputeProxyCounts");

        const dm::uint  items = ctrlBuff.TotalLightCount;
        const dm::uint  itemsPerGroup = LLB_NUM_COMPUTE_THREADS;
        m_computeProxyCounts->Dispatch(commandList, div_ceil(items, itemsPerGroup), 1, 1, { bindingSet });

        commandList->setBufferState(m_perLightProxyCounters, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setBufferState(m_controlBuffer, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setBufferState(m_scratchList, nvrhi::ResourceStates::UnorderedAccess);
    }

    if (ctrlBuff.TotalLightCount > 0)
    {
        nvrhi::ComputeState state;
        state.bindings = { bindingSet };

        {
            RTXPT_GPU_SCOPE(commandList, "ComputeProxyBaselineOffsets");

            m_computeProxyBaselineOffsets->Dispatch(commandList, 1, 1, 1, { bindingSet });

            commandList->setBufferState(m_lightSamplingProxies, nvrhi::ResourceStates::UnorderedAccess);
        }

        {
            RTXPT_GPU_SCOPE(commandList, "CreateProxyJobs");

            const dm::uint  items = ctrlBuff.TotalLightCount;
            const dm::uint  itemsPerGroup = LLB_NUM_COMPUTE_THREADS;
            m_createProxyJobs->Dispatch(commandList, div_ceil(items, itemsPerGroup), 1, 1, { bindingSet });

            commandList->setBufferState(m_controlBuffer, nvrhi::ResourceStates::UnorderedAccess);   // because we've written into u_controlBuffer[0].ProxyBuildTaskCount
            commandList->setBufferState(m_scratchBuffer, nvrhi::ResourceStates::UnorderedAccess);   // because this is where jobs are stored
        }

        {
            RTXPT_GPU_SCOPE(commandList, "ExecuteProxyJobs");

            // The task count is only known on the GPU, so dispatch its bound - sum of ceil(proxies/32) over the lights, which is at most
            // lights + proxies/32 (LLB_MAX_PROXY_PROC_TASKS with the actual light count); the shader skips the rest
            const dm::uint  items = ctrlBuff.TotalLightCount + div_ceil(RTXPT_LIGHTING_MAX_SAMPLING_PROXIES, LLB_MAX_PROXIES_PER_TASK);
            const dm::uint  itemsPerGroup = LLB_NUM_COMPUTE_THREADS;
            const dm::uint  dispatchCountX = div_ceil(items, itemsPerGroup); assert(dispatchCountX <= 65535); // more than this triggers EXECUTION WARNING #1296: OVERSIZED_DISPATCH
            m_executeProxyJobs->Dispatch(commandList, dispatchCountX, 1, 1, { bindingSet });

            commandList->setBufferState(m_lightSamplingProxies, nvrhi::ResourceStates::UnorderedAccess);    // because we've filled it up
        }
    }
    
    if( m_dbgDebugDrawLights && ctrlBuff.TotalLightCount > 0 )
    {
        RTXPT_GPU_SCOPE(commandList, "DebugDrawLights");

        commandList->setBufferState(m_controlBuffer, nvrhi::ResourceStates::UnorderedAccess);

        const dm::uint  items = NEEAT_MAX_LIGHTS; // this one is updated on GPU so it's not correct here so let's just brute force to max, compute shader will skip...
        const dm::uint  itemsPerGroup = LLB_NUM_COMPUTE_THREADS;
        m_debugDrawLights->Dispatch(commandList, div_ceil(items, itemsPerGroup), 1, 1, { bindingSet });
    }
    const int readbackHalfPeriod = 5; // it's actually nicer to display a bit slower changing value
    // for debugging only
    {
        ZoneScopedN("Copy back debug buffers");

        if (m_framesFromLastReadbackCopy == 0)
            commandList->copyBuffer(m_controlBufferReadback, 0, m_controlBuffer, 0, sizeof(NEEAT::ControlData) * 1); // copy, do nothing else
        else
        {
    #if LLB_ENABLE_VALIDATION   // instant feedback but significant perf hit
            m_device->waitForIdle(); 
    #else
            if (m_framesFromLastReadbackCopy > readbackHalfPeriod) // 5 is always safe, we won't have that many frames overlapping
    #endif
            {
                // we must do that to avoid invalid mix of barriers 
                commandList->commitBarriers();

                // Copy from readback buffer to struct that's displayed in UI
                void* pData = m_device->mapBuffer(m_controlBufferReadback, nvrhi::CpuAccessMode::Read);
                assert(pData);
                memcpy(&m_lastReadback, pData, sizeof(NEEAT::ControlData) * 1);
                m_device->unmapBuffer(m_controlBufferReadback);

                // Reset counter
                m_framesFromLastReadbackCopy = -readbackHalfPeriod;
            }
        }
        commandList->commitBarriers();  // committing now avoids "D3D12 ERROR: ID3D12CommandList::ResourceBarrier: D3D12_RESOURCE_STATES has an invalid combination of state bits." later in DispatchRays; this needs to be debugged
        m_framesFromLastReadbackCopy++;
    }

#ifdef HASH_LOOKUP_BASED_HISTORIC_LIGHT_SOURCE_MATCHING
    // All hash-map reads/writes for this frame happen inside UpdateBegin's call tree (CollectAnalyticLightsCPU and
    // ProcessEmissiveGeometry); UpdateEnd touches neither. So this is the correct point to roll them over.
    FlipHistoryRemapBuffers();
#endif
}

#define UAV_BARRIER_m_NEE_AT_LocalSamplingBuffer() { commandList->setBufferState(m_NEE_AT_LocalSamplingBuffer, nvrhi::ResourceStates::UnorderedAccess); }

void LightsBaker::UpdateEnd(nvrhi::ICommandList * commandList, donut::engine::BindingCache & bindingCache, const std::shared_ptr<ExtendedScene> & scene, std::shared_ptr<class MaterialsBaker> materialsBaker, std::shared_ptr<class OmmBaker> ommBaker, nvrhi::BufferHandle subInstanceDataBuffer, nvrhi::TextureHandle depthBuffer, nvrhi::TextureHandle motionVectors)
{
    RTXPT_GPU_SCOPE(commandList, "LightingUpdateEnd");

    if (m_bakeSkipped)
        return;

    if (m_currentSettings.ImportanceSamplingType != 2)  // no feedback or local sampling code needed if not using NEE_AT - just make sure you're not actually using local sampling!
        return;

    m_verifyBeginHasMatchingEnd--; assert( m_verifyBeginHasMatchingEnd == 0 );
    bool lastFrameFeedbackAvailable = m_NEE_AT_FeedbackBufferFilled;

    nvrhi::BindingSetDesc bindingSetDesc;
    FillBindings(bindingSetDesc, scene, materialsBaker, ommBaker, subInstanceDataBuffer, depthBuffer, motionVectors, nullptr);
    nvrhi::BindingSetHandle bindingSet = bindingCache.GetOrCreateBindingSet(bindingSetDesc, m_commonBindingLayout);
    nvrhi::BindingSetVector bindings = { bindingSet };

    const dm::uint  itemsPerGroup = LLB_NUM_COMPUTE_THREADS_2D;

    {
        RTXPT_GPU_SCOPE(commandList, "ProcessFeedbackHistoryP1a");

        commandList->setTextureState(m_NEE_AT_FeedbackTotalWeight, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setTextureState(m_NEE_AT_FeedbackCandidates, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setTextureState(m_NEE_AT_FeedbackTotalWeightScratch, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setTextureState(m_NEE_AT_FeedbackCandidatesScratch, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);

        m_processFeedbackHistoryP1a->Dispatch(commandList, div_ceil(m_currentCtrlBuff.BakerConstants.BlendedFeedbackResolution.x, itemsPerGroup), div_ceil(m_currentCtrlBuff.BakerConstants.BlendedFeedbackResolution.y, itemsPerGroup), 1, bindings);

        commandList->setTextureState(m_NEE_AT_FeedbackTotalWeightScratch, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setTextureState(m_NEE_AT_FeedbackCandidatesScratch, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setTextureState(m_NEE_AT_FeedbackTotalWeightBlended, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setTextureState(m_NEE_AT_FeedbackCandidatesBlended, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    }

    {
        RTXPT_GPU_SCOPE(commandList, "ProcessFeedbackHistoryP1b");

        commandList->setTextureState(m_NEE_AT_FeedbackTotalWeight, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setTextureState(m_NEE_AT_FeedbackCandidates, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);

        m_processFeedbackHistoryP1b->Dispatch(commandList, div_ceil(m_currentCtrlBuff.BakerConstants.FeedbackResolution.x, itemsPerGroup), div_ceil(m_currentCtrlBuff.BakerConstants.FeedbackResolution.y, itemsPerGroup), 1, bindings);

        commandList->setTextureState(m_NEE_AT_FeedbackTotalWeightScratch, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setTextureState(m_NEE_AT_FeedbackCandidatesScratch, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        // ClearFeedbackHistory below overwrites these, which P1a/P1b read, with no other barrier in between
        commandList->setTextureState(m_NEE_AT_FeedbackTotalWeight, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setTextureState(m_NEE_AT_FeedbackCandidates, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setTextureState(m_NEE_AT_HistoryDepth, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    }

    {
        RTXPT_GPU_SCOPE(commandList, "ProcessFeedbackHistoryP2");
        m_processFeedbackHistoryP2->Dispatch(commandList, div_ceil(m_currentCtrlBuff.LocalSamplingResolution.x, itemsPerGroup), div_ceil(m_currentCtrlBuff.LocalSamplingResolution.y, itemsPerGroup), 1, bindings);
    }

    // Overlaps P2 (no barrier between them): P2 only reads the scratch and blended feedback and writes the local sampling buffer
    const bool temporalFeedbackRequired = m_currentSettings.ImportanceSamplingType == 2;
    if (m_currentCtrlBuff.TemporalFeedbackRequired)
    {
        RTXPT_GPU_SCOPE(commandList, "ClearFeedbackHistory");

        //commandList->setTextureState(m_NEE_AT_FeedbackTotalWeightScratch, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        //commandList->setTextureState(m_NEE_AT_FeedbackCandidatesScratch, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);

        m_clearFeedbackHistory->Dispatch( commandList, div_ceil(m_currentCtrlBuff.BakerConstants.FeedbackResolution.x, itemsPerGroup), div_ceil(m_currentCtrlBuff.BakerConstants.FeedbackResolution.y, itemsPerGroup), 1, bindings);

        m_NEE_AT_FeedbackBufferFilled = true;  // the assumption is that the path tracing happens after and actually fills the data; it's fine if it doesn't, the clear ^ resets it to empty anyway
    }
    UAV_BARRIER_m_NEE_AT_LocalSamplingBuffer();

    {
        RTXPT_GPU_SCOPE(commandList, "ProcessFeedbackHistoryP3");
        m_processFeedbackHistoryP3->Dispatch(commandList, m_currentCtrlBuff.LocalSamplingResolution.x, m_currentCtrlBuff.LocalSamplingResolution.y, 1, bindings);
        UAV_BARRIER_m_NEE_AT_LocalSamplingBuffer();
    }

    if (m_currentCtrlBuff.BakerConstants.DebugDrawTileLights || m_dbgDebugDrawType == NEEAT::DebugViewType::TileHeatmap || m_dbgDebugDrawType == NEEAT::DebugViewType::ValidateCorrectness || m_dbgFreezeFrustumUpdates)
    {
        UAV_BARRIER_m_NEE_AT_LocalSamplingBuffer();
        commandList->commitBarriers();

        RTXPT_GPU_SCOPE(commandList, "ProcessFeedbackHistoryDebugViz");
        m_processFeedbackHistoryDebugViz->Dispatch(commandList, div_ceil(m_currentCtrlBuff.LocalSamplingResolution.x, itemsPerGroup), div_ceil(m_currentCtrlBuff.LocalSamplingResolution.y, itemsPerGroup), 1, bindings);

        UAV_BARRIER_m_NEE_AT_LocalSamplingBuffer();
        commandList->commitBarriers();
    }

    // ClearFeedbackHistory's writes, for the path tracer
    commandList->setTextureState(m_NEE_AT_FeedbackTotalWeight, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    commandList->setTextureState(m_NEE_AT_FeedbackCandidates, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);

    // this is useful to avoid "leaking" any barrier issues to subsequent passes which makes it difficult to debug
    commandList->commitBarriers();
}

bool LightsBaker::InfoGUI(float indent)
{
    RAII_SCOPE(ImGui::PushID("LightsBakerInfoGUI");, ImGui::PopID(); );

    if (TotalLightCountOverflow())
    {
        ImGui::TextColored({ 1,0.5f,0.5f,1 }, "!!WARNING - scene light count overflow!!");
        ImGui::TextColored({ 1,0.5f,0.5f,1 }, "increase NEEAT_MAX_LIGHTS (%d)", NEEAT_MAX_LIGHTS);
    }

    const char* modes[] = { "Uniform", "Power+", "NEE-AT" };
    ImGui::Text("Current mode:  %s", modes[dm::clamp(m_lastReadback.ImportanceSamplingType, 0u, 2u)]);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("As set in Path Tracer Next Event Estimation options\n(in the future, mode will be set here)");

    ImGui::Text("Scene lights by type: ");
    ImGui::Text("   envmap quads:  %d", m_lastReadback.EnvmapQuadNodeCount);
    auto percent = [](uint part, uint total) { return 100.0f * part / (float)std::max(1u, total); };
    ImGui::Text("   emissive tris: %d (%d in active clusters, %.1f%%)", m_lastReadback.TriangleLightCount, m_lastReadback.ClusteredTriangleCount, percent(m_lastReadback.ClusteredTriangleCount, m_lastReadback.TriangleLightCount));
    ImGui::Text("   clusters:      %d (%d active, %.1f%%)", m_lastReadback.LightClusterCount, m_lastReadback.ActiveLightClusterCount, percent(m_lastReadback.ActiveLightClusterCount, m_lastReadback.LightClusterCount));
    ImGui::Text("   analytic:      %d", m_lastReadback.AnalyticLightCount);
    ImGui::Text("   TOTAL:         %d", m_lastReadback.TotalLightCount);
    ImGui::Text("(used: %.2f%% of max %d)", (m_lastReadback.TotalLightCount / (float)NEEAT_MAX_LIGHTS * 100.0f), NEEAT_MAX_LIGHTS );
    ImGui::Text("(proxies: %d, weightsum: %.5f)", m_lastReadback.SamplingProxyCount, m_lastReadback.WeightsSum());
#if LLB_ENABLE_VALIDATION
    ImGui::Text("Validation:");
    float feedbackPerc = m_lastReadback.ValidFeedbackCount / float(m_currentCtrlBuff.BakerConstants.FeedbackResolution.x * m_currentCtrlBuff.BakerConstants.FeedbackResolution.y);
    ImGui::Text(" feedback num: %d (%.3f)", m_lastReadback.ValidFeedbackCount, feedbackPerc);
#endif
    float allocRam = (float)(double(m_allocatedVRAM)/1024.0/1024.0);
    ImGui::Text("Memory use: %.1f MiB", allocRam);

    return false;
}

bool LightsBaker::DebugGUI(float indent)
{
    RAII_SCOPE(ImGui::PushID("LightsBakerDebugGUI"); , ImGui::PopID(); );

    bool resetAccumulation = false;
    #define IMAGE_QUALITY_OPTION(code) do{if (code) resetAccumulation = true;} while(false)

    ImGui::Checkbox("Debug draw all lights", &m_dbgDebugDrawLights);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Wireframe colour indicates type: red - environment map; green - emissive triangles; blue - analytic.\nBright green boxes are active light clusters, drawn instead of their triangles.");

    ImGui::Checkbox("Debug draw NEE-AT tile light connections", &m_dbgDebugDrawTileLightConnections);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Shows lights sampled by a specific tile local sampling pdf");

    ImGui::Checkbox("Freeze NEE-AT feedback", &m_dbgFreezeFeedback);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Feedback from the path tracer will remain frozen while this option is enabled.\nSome parts like pre-filter get disabled as well.");

    const char* debugOptions = "Disabled\0Disocclusion\0FeedbackHistoryInfo\0MissingFeedbackScreenSpaceCoherent\0MissingFeedbackWorldSpaceCoherent\0FeedbackRawScreenSpaceCoherent\0FeedbackRawWorldSpaceCoherent\0LowResBlendedFeedback\0FeedbackAfterClear\0TileHeatmap\0ValidateCorrectness\0\0";
    ImGui::Combo("NEE-AT debug view", (int*)&m_dbgDebugDrawType, debugOptions);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show various NEE-AT buffers");

    ImGui::Checkbox("Debug disable local tile jitter", &m_dbgDebugDisableJitter);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Mapping from pixels to tiles will be jittered to avoid denoising artifacts.\nIt also helps with spatial sharing.\nDisable for debugging.");

    ImGui::Checkbox("Debug disable last frame feedback", &m_dbgDebugDisableLastFrameFeedback);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Simply disables last frame's feedback for debugging/validation.\nQuality should revert to slightly worse than power based sampling.");
    
    ImGui::Checkbox("Debug freeze frustum updates", &m_dbgFreezeFrustumUpdates);

#if 1
    ImGui::Separator();
    if (ImGui::CollapsingHeader("Advanced settings", 0/*ImGuiTreeNodeFlags_DefaultOpen*/))
    {
        ImGui::SliderFloat("ScreenSpaceVsWorldSpaceThreshold", &m_advSetting_ScreenSpaceVsWorldSpaceThreshold, 0.02f, 2.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Used to determine, for each sampling location, whether it's more optimal to use screen tiles or world voxels for caching.");

        ImGui::SliderFloat("ReservoirHistoryDropoff", &m_advSetting_reservoirHistoryDropoff, 0.0f, 0.1f, "%.3f", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("The amount of history sharing from past and from neighbours. Some is useful, \ntoo much will add lag and allow strong lights to dwarf out others.");

        ImGui::SliderFloat("DepthDisocclusionThreshold", &m_depthDisocclusionThreshold, 0.999f, 20.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("During motion reprojection, drop samples if really far from target");

        ImGui::Checkbox("Sample environment proxy lights", &m_advSetting_SampleBakedEnvironment);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("If enabled, environment map texture will not be sampled directly by NEE\nbut will be baked into sampling proxies like emissive triangles.\nBiased, faster but more blurry shadows in some cases.");

        ImGui::Checkbox("Feedback from BSDF rays", &m_advSetting_BSDFFeedbackEnabled);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("BSDF rays hitting emissive surfaces or the sky also provide NEE-AT feedback.\nHelps primarily to speed up convergence / reduce lag but can increase converged noise.\nChanging it recompiles the path tracer.");

        ImGui::Checkbox("Enable light clusters", &m_advSetting_LightClustersEnabled);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("When disabled, all light clusters stay inactive and emissive triangles are used directly\n(same as an infinite LOD multiplier).");
        const char * clusterLODTooltip = "A light cluster represents its emissive triangles (is active) when\nLODMultiplier * pow(clusterRadius / distanceToCamera, LODExponent) * pow(max(0.1, brightness), BrightnessExponent) < 1\n"
                                         "where brightness is the cluster's average light weight over the scene's average light weight (0 exponent disables it)";
        ImGui::SliderFloat("Light cluster LOD exponent", &m_advSetting_LightClusterLODExponent, 0.5f, 4.0f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", clusterLODTooltip);
        ImGui::SliderFloat("Light cluster LOD multiplier", &m_advSetting_LightClusterLODMultiplier, 1.0f, 100000.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", clusterLODTooltip);
        ImGui::SliderFloat("Light cluster brightness exponent", &m_advSetting_LightClusterBrightnessExponent, 0.0f, 1.0f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", clusterLODTooltip);

        {
            ImGui::Text("Importance boosts:");
            RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent););
            ImGui::Checkbox("...by light intensity change", &m_importanceBoost_IntensityDelta);
            {
                RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent););
                RAII_SCOPE(ImGui::PushID("Delta");, ImGui::PopID(); );
                ImGui::InputFloat("multiplier", &m_importanceBoost_IntensityDeltaMul);
                m_importanceBoost_IntensityDeltaMul = dm::clamp(m_importanceBoost_IntensityDeltaMul, 0.0f, 1000.0f);
            }
            ImGui::Checkbox("...by light frustum proximity", &m_importanceBoost_Frustum);
            {
                RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent););
                RAII_SCOPE(ImGui::PushID("FrustProx"); , ImGui::PopID(); );
                ImGui::InputFloat("multiplier", &m_importanceBoost_FrustumMul);
                m_importanceBoost_FrustumMul = dm::clamp(m_importanceBoost_FrustumMul, 0.0f, 1000.0f);
                ImGui::InputFloat("fade distance", &m_importanceBoost_FrustumFadeDistance);
                m_importanceBoost_FrustumFadeDistance = dm::clamp(m_importanceBoost_FrustumFadeDistance, 0.0f, 1000.0f);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("How fast the boost fades outside of the frustum\nThe bigger the value, the slower it fades");
                // ImGui::InputFloat("angle expand", &m_importanceBoost_FrustumAngleExpand);
                // m_importanceBoost_FrustumAngleExpand = dm::clamp(m_importanceBoost_FrustumAngleExpand, 0.0f, 1000.0f);
                // if (ImGui::IsItemHovered()) ImGui::SetTooltip("Expand frustum by specified number of degrees");
            }
            ImGui::Checkbox("...by pre-filter merge", &m_importanceBoost_PreFilter);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Will allow stronger feedback in 3x3 kernel to 'overwhelm' neighbors\nEXPERIMENTAL - SUPER-SLOW");
        }
    }
#endif

    return resetAccumulation;
}

void LightsBaker::SetGlobalShaderMacros(std::vector<donut::engine::ShaderMacro> & macros)
{
    macros.push_back({ "NEE_AT_SAMPLE_BAKED_ENVIRONMENT", (m_advSetting_SampleBakedEnvironment) ? ("1") : ("0") });
    macros.push_back({ "RTXPT_LIGHTING_ENABLE_BSDF_FEEDBACK", (m_advSetting_BSDFFeedbackEnabled) ? ("1") : ("0") });
    macros.push_back({ "NEEAT_QUALITY_PRESET", std::to_string(m_qualityPreset) });  // must match the LightsBaker passes, see CreateRenderPasses
}
