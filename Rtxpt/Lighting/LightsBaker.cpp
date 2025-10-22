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

#include "../SampleCommon.h"
#include "../ExtendedScene.h"

#include "LightsBaker.hlsl"

#include "../ShaderDebug.h"

#include "../GPUSort/GPUSort.h"
#include "../Shaders/PathTracer/Utils/NoiseAndSequences.hlsli"

#include "Distant/EnvMapBaker.h"
#include "Distant/EnvMapImportanceSamplingBaker.h"

#include "../Materials/MaterialsBaker.h"
#include "../OpacityMicroMap/OmmBaker.h"

using namespace donut;
using namespace donut::math;
using namespace donut::engine;

LightsBaker::LightsBaker(nvrhi::IDevice* device, std::shared_ptr<donut::engine::TextureCache> textureCache, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory, std::shared_ptr<EnvMapBaker> envMapBaker)
    : m_device(device)
    , m_textureCache(textureCache)
    , m_bindingCache(device)
    , m_shaderFactory(shaderFactory)
    , m_envMapBaker(envMapBaker)
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
    m_historyRemapAnalyticLightIndices.clear();
    m_historyRemapEmissiveLightBlockOffsets.clear();
    m_historicTotalLightCount = 0;
    memset(&m_currentCtrlBuff, 0, sizeof(m_currentCtrlBuff));
    memset(&m_currentSettings, 0, sizeof(m_currentSettings));
    memset(&m_currentConsts, 0, sizeof(m_currentConsts));
}

LightsBaker::~LightsBaker()
{
}

void LightsBaker::CreateRenderPasses(nvrhi::IBindingLayout* bindlessLayout, std::shared_ptr<engine::CommonRenderPasses> commonPasses, std::shared_ptr<ShaderDebug> shaderDebug, const uint2 renderResolution)
{
    m_bindlessLayout = bindlessLayout;
    m_commonPasses = commonPasses;
    m_shaderDebug = shaderDebug;

    std::vector<donut::engine::ShaderMacro> shaderMacros;
    //shaderMacros.push_back(donut::engine::ShaderMacro({              "BLEND_DEBUG_BUFFER", "1" }));

    const char * shaderFile = "app/Lighting/LightsBaker.hlsl";
        
    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(20),     // m_constantBuffer
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
#if RTXPT_LIGHTING_LOCAL_SAMPLING_BUFFER_IS_3D_TEXTURE
            nvrhi::BindingLayoutItem::Texture_UAV(18),              // u_localSamplingBuffer
#else
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(18),          // u_localSamplingBuffer
#endif
            nvrhi::BindingLayoutItem::Texture_SRV(10),              // t_depthBuffer
            nvrhi::BindingLayoutItem::Texture_SRV(11),              // t_motionVectors
            nvrhi::BindingLayoutItem::Texture_SRV(12),              // t_envmapImportanceMap
            nvrhi::BindingLayoutItem::TypedBuffer_SRV(13),          // t_lightWeightsHistoric
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

    nvrhi::ComputePipelineDesc pipelineDesc;

    // These need to know about the scene
    pipelineDesc.bindingLayouts = { m_commonBindingLayout, m_bindlessLayout };
    m_bakeEmissiveTriangles     .Init(m_device, *m_shaderFactory, shaderFile, "BakeEmissiveTriangles",      shaderMacros, pipelineDesc.bindingLayouts);
    
    // these don't need to know anything about the scene
    pipelineDesc.bindingLayouts = { m_commonBindingLayout };

    m_resetPastToCurrentHistory.Init(m_device, *m_shaderFactory, shaderFile, "ResetPastToCurrentHistory",   shaderMacros, pipelineDesc.bindingLayouts);

    m_envLightsBackupPast       .Init(m_device, *m_shaderFactory, shaderFile, "EnvLightsBackupPast"      ,   shaderMacros, pipelineDesc.bindingLayouts);
    m_envLightsSubdivideBase    .Init(m_device, *m_shaderFactory, shaderFile, "EnvLightsSubdivideBase"   ,   shaderMacros, pipelineDesc.bindingLayouts);
    m_envLightsSubdivideBoost   .Init(m_device, *m_shaderFactory, shaderFile, "EnvLightsSubdivideBoost"  ,   shaderMacros, pipelineDesc.bindingLayouts);
    m_envLightsFillLookupMap    .Init(m_device, *m_shaderFactory, shaderFile, "EnvLightsFillLookupMap"   ,   shaderMacros, pipelineDesc.bindingLayouts);
    m_envLightsMapPastToCurrent .Init(m_device, *m_shaderFactory, shaderFile, "EnvLightsMapPastToCurrent",   shaderMacros, pipelineDesc.bindingLayouts);

    m_clearFeedbackHistory      .Init(m_device, *m_shaderFactory, shaderFile, "ClearFeedbackHistory",        shaderMacros, pipelineDesc.bindingLayouts);
    m_clearAntiLagFeedback      .Init(m_device, *m_shaderFactory, shaderFile, "ClearAntiLagFeedback",        shaderMacros, pipelineDesc.bindingLayouts);

    m_processFeedbackHistoryP0      .Init(m_device, *m_shaderFactory, shaderFile, "ProcessFeedbackHistoryP0"        , shaderMacros, pipelineDesc.bindingLayouts);
    m_processFeedbackHistoryP1a     .Init(m_device, *m_shaderFactory, shaderFile, "ProcessFeedbackHistoryP1a"       , shaderMacros, pipelineDesc.bindingLayouts);
    m_processFeedbackHistoryP1b     .Init(m_device, *m_shaderFactory, shaderFile, "ProcessFeedbackHistoryP1b"       , shaderMacros, pipelineDesc.bindingLayouts);
    m_processFeedbackHistoryP2      .Init(m_device, *m_shaderFactory, shaderFile, "ProcessFeedbackHistoryP2"        , shaderMacros, pipelineDesc.bindingLayouts);
    m_processFeedbackHistoryP3      .Init(m_device, *m_shaderFactory, shaderFile, "ProcessFeedbackHistoryP3"        , shaderMacros, pipelineDesc.bindingLayouts);
    m_processFeedbackHistoryDebugViz.Init(m_device, *m_shaderFactory, shaderFile, "ProcessFeedbackHistoryDebugViz"  , shaderMacros, pipelineDesc.bindingLayouts);
    m_updateControlBufferMultipass  .Init(m_device, *m_shaderFactory, shaderFile, "UpdateControlBufferMultipass"    , shaderMacros, pipelineDesc.bindingLayouts);

    m_resetLightProxyCounters       .Init(m_device, *m_shaderFactory, shaderFile, "ResetLightProxyCounters"         , shaderMacros, pipelineDesc.bindingLayouts);
    m_computeWeights                .Init(m_device, *m_shaderFactory, shaderFile, "ComputeWeights"                  , shaderMacros, pipelineDesc.bindingLayouts);
    m_computeProxyCounts            .Init(m_device, *m_shaderFactory, shaderFile, "ComputeProxyCounts"              , shaderMacros, pipelineDesc.bindingLayouts);
    m_computeProxyBaselineOffsets   .Init(m_device, *m_shaderFactory, shaderFile, "ComputeProxyBaselineOffsets"     , shaderMacros, pipelineDesc.bindingLayouts);
    m_createProxyJobs               .Init(m_device, *m_shaderFactory, shaderFile, "CreateProxyJobs"                 , shaderMacros, pipelineDesc.bindingLayouts);
    m_executeProxyJobs              .Init(m_device, *m_shaderFactory, shaderFile, "ExecuteProxyJobs"                , shaderMacros, pipelineDesc.bindingLayouts);
    m_debugDrawLights               .Init(m_device, *m_shaderFactory, shaderFile, "DebugDrawLights"                 , shaderMacros, pipelineDesc.bindingLayouts);

    nvrhi::SamplerDesc samplerDesc;
    samplerDesc.setBorderColor(nvrhi::Color(0.f));
    samplerDesc.setAllFilters(true);
    samplerDesc.setMipFilter(true);
    samplerDesc.setAllAddressModes(nvrhi::SamplerAddressMode::Wrap);
    m_linearSampler = m_device->createSampler(samplerDesc);

    samplerDesc.setAllFilters(false);
    m_pointSampler = m_device->createSampler(samplerDesc);

    // destroy resources before creating to avoid lifetimes of old and new overlapping (even with itself, due to assignment operator) - avoids fragmentation and peaks
    m_constantBuffer = m_controlBuffer = m_lightsBuffer = m_lightsExBuffer = m_historyRemapCurrentToPastBuffer = m_historyRemapPastToCurrentBuffer = m_scratchBuffer = m_lightWeightsPing = m_lightWeightsPong = m_perLightProxyCounters = m_scratchList = m_lightSamplingProxies = nullptr;
    //m_lightingConstants = nullptr;
    m_device->waitForIdle();    // make sure readback buffer is no longer used by the GPU
    m_controlBufferReadback = nullptr;

    // // Main constant buffer
    // m_constantBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
    //     sizeof(LightsBakerConstants), "LightsBakerConstants", engine::c_MaxRenderPassConstantBufferVersions * 32));	// *32 we could be updating few times per frame

    {
        nvrhi::BufferDesc bufferDesc;
        bufferDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        bufferDesc.keepInitialState = true;
        bufferDesc.canHaveUAVs = false;

        // Main constant buffer
        bufferDesc.byteSize = sizeof(LightsBakerConstants) * 1;
        bufferDesc.structStride = sizeof(LightsBakerConstants);
        bufferDesc.debugName = "LightsBakerConstants";
        m_constantBuffer = m_device->createBuffer(bufferDesc);

        bufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        bufferDesc.keepInitialState = true;
        bufferDesc.canHaveUAVs = true;

        bufferDesc.byteSize = sizeof(LightingControlData) * 1;
        bufferDesc.structStride = sizeof(LightingControlData);
        bufferDesc.debugName = "LightingControlData";
        m_controlBuffer = m_device->createBuffer(bufferDesc);

        // bufferDesc.isConstantBuffer = true;
        // bufferDesc.canHaveUAVs = false;
        // bufferDesc.debugName = "LightingConstants";
        // bufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
        // m_lightingConstants = m_device->createBuffer(bufferDesc);
        // bufferDesc.isConstantBuffer = false;
        // bufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        // bufferDesc.canHaveUAVs = true;

        // Lights buffer
        bufferDesc.byteSize = sizeof(PolymorphicLightInfo) * RTXPT_LIGHTING_MAX_LIGHTS;
        bufferDesc.structStride = sizeof(PolymorphicLightInfo);
        bufferDesc.debugName = "LightsBuffer";
        m_lightsBuffer = m_device->createBuffer(bufferDesc);
        
        bufferDesc.byteSize = sizeof(PolymorphicLightInfoEx) * RTXPT_LIGHTING_MAX_LIGHTS;
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
        m_scratchTaskBuffer = std::make_shared<std::vector<struct EmissiveTrianglesProcTask>>();
        m_scratchTaskBuffer->reserve(LLB_MAX_PROC_TASKS);

        // Subsequent buffers are non-structured
        bufferDesc.structStride = 0;
        bufferDesc.canHaveTypedViews = true;
        bufferDesc.canHaveRawViews = false;
        
        bufferDesc.byteSize = sizeof(float) * (RTXPT_LIGHTING_MAX_LIGHTS+1);    // +1 is purely because perLightProxyCounters needs one more to store invalid feedback
        bufferDesc.format = nvrhi::Format::R32_FLOAT;
        bufferDesc.debugName = "LightsWeightsPing";
        m_lightWeightsPing = m_device->createBuffer(bufferDesc);
        bufferDesc.debugName = "LightsWeightsPong";
        m_lightWeightsPong = m_device->createBuffer(bufferDesc);

        bufferDesc.format = nvrhi::Format::R32_UINT;
        bufferDesc.debugName = "HistoryRemapCurrentToPast";
        m_historyRemapCurrentToPastBuffer = m_device->createBuffer(bufferDesc);
        bufferDesc.debugName = "HistoryRemapPastToCurrent";
        m_historyRemapPastToCurrentBuffer = m_device->createBuffer(bufferDesc);
        bufferDesc.debugName = "PerLightProxyCounters";
        m_perLightProxyCounters = m_device->createBuffer(bufferDesc);
        bufferDesc.debugName = "ScratchList";
        assert( bufferDesc.byteSize / sizeof(uint) >= (RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT*2) );    // we need at least 2 times RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT for temporary envmap quads stuff
        m_scratchList = m_device->createBuffer(bufferDesc);
        bufferDesc.byteSize = sizeof(uint) * RTXPT_LIGHTING_MAX_SAMPLING_PROXIES;
        bufferDesc.debugName = "LightSamplingProxies";
        m_lightSamplingProxies = m_device->createBuffer(bufferDesc);

        // For debugging/UI
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

    assert(renderResolution.x > 0 && renderResolution.y > 0);
    if (m_NEE_AT_FeedbackTotalWeight == nullptr || m_NEE_AT_FeedbackTotalWeight->getDesc().width != renderResolution.x || m_NEE_AT_FeedbackTotalWeight->getDesc().height != renderResolution.y)
    {
        if (m_NEE_AT_FeedbackTotalWeight )
            m_device->waitForIdle();    // make sure none of the buffers are used by the GPU

        // destroy before creating to avoid lifetimes of old and new overlapping (even with itself, due to assignment operator) - avoids fragmentation and peaks
        m_NEE_AT_FeedbackTotalWeight = nullptr;
        m_NEE_AT_FeedbackCandidates  = nullptr;
        m_NEE_AT_FeedbackTotalWeightScratch = nullptr;
        m_NEE_AT_FeedbackCandidatesScratch  = nullptr;
        m_NEE_AT_FeedbackTotalWeightBlended = nullptr;
        m_NEE_AT_FeedbackCandidatesBlended  = nullptr;
        
        m_NEE_AT_LocalSamplingBuffer = nullptr;
        m_NEE_AT_HistoryDepth = nullptr;

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
        desc.debugName = "NEE_AT_HistoryDepth";
        m_NEE_AT_HistoryDepth = m_device->createTexture(desc);
        desc.debugName = "NEE_AT_FeedbackTotalWeight";
        m_NEE_AT_FeedbackTotalWeight = m_device->createTexture(desc);
        desc.debugName = "NEE_AT_FeedbackTotalWeightScratch";
        m_NEE_AT_FeedbackTotalWeightScratch = m_device->createTexture(desc);
        nvrhi::TextureDesc miniDesc = desc; miniDesc.width = div_ceil(desc.width, RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE); miniDesc.height = div_ceil(desc.height, RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE);
        desc.debugName = "NEE_AT_EarlyFeedbackTotalWeightScratch";
        m_NEE_AT_FeedbackTotalWeightBlended = m_device->createTexture(miniDesc);
        m_NEE_AT_FeedbackBufferFilled = false;
        static_assert(RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH == 1 || RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH == 2 || RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH == 4); // TODO: upgrade to allow 1
        if (RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH == 1)
            desc.format = nvrhi::Format::R32_UINT;
        else if (RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH == 2)
            desc.format = nvrhi::Format::RG32_UINT; 
        else if (RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH == 4)
            desc.format = nvrhi::Format::RGBA32_UINT; 
        else assert(false);
        desc.debugName = "NEE_AT_FeedbackCandidates";
        m_NEE_AT_FeedbackCandidates = m_device->createTexture(desc);
        desc.debugName = "NEE_AT_FeedbackCandidatesScratch";
        m_NEE_AT_FeedbackCandidatesScratch = m_device->createTexture(desc);
        miniDesc = desc; miniDesc.width = div_ceil(desc.width, RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE); miniDesc.height = div_ceil(desc.height, RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE);
        desc.debugName = "NEE_AT_EarlyFeedbackCandidatesScratch";
        m_NEE_AT_FeedbackCandidatesBlended = m_device->createTexture(miniDesc);

        {
            m_localSamplingBufferWidth  = dm::div_ceil(renderResolution.x, RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE);
            m_localSamplingBufferHeight = dm::div_ceil(renderResolution.y, RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE);
            m_localSamplingBufferWidth  += 1;   // add border to accommodate for jitter offset for the local sampling buffers
            m_localSamplingBufferHeight += 1;   // add border to accommodate for jitter offset for the local sampling buffers
            // m_localSamplingBufferDepth          = RTXPT_LIGHTING_LOCAL_PROXY_COUNT
#if RTXPT_LIGHTING_LOCAL_SAMPLING_BUFFER_IS_3D_TEXTURE
            desc.dimension = nvrhi::TextureDimension::Texture3D;
            desc.width = m_localSamplingBufferWidth;
            desc.height = m_localSamplingBufferHeight;
            desc.depth = m_localSamplingBufferDepth;
            desc.format = nvrhi::Format::R32_UINT;
            desc.debugName = "NEE_AT_LocalSamplingBuffer";
            m_NEE_AT_LocalSamplingBuffer = m_device->createTexture(desc);
#else
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
#endif
        }


        assert(RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE>=RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE && ((RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE-RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE)%2==0));
    }

    if (m_envLightLookupMap == nullptr || m_envLightLookupMap->getDesc().width != m_envMapBaker->GetImportanceSampling()->GetImportanceMapResolution() || m_envLightLookupMap->getDesc().mipLevels != m_envMapBaker->GetImportanceSampling()->GetImportanceMapMIPLevels() )
    {
        int lookupRes = m_envMapBaker->GetImportanceSampling()->GetImportanceMapResolution();
        nvrhi::TextureDesc texDesc;
        texDesc.format = nvrhi::Format::R32_UINT;
        texDesc.width = lookupRes;
        texDesc.height = lookupRes;
        texDesc.mipLevels = 1;
        texDesc.isRenderTarget = true;
        texDesc.isUAV = true;
        texDesc.debugName = "EnvLightLookupMap";
        texDesc.setInitialState(nvrhi::ResourceStates::UnorderedAccess);
        texDesc.keepInitialState = true;
        m_envLightLookupMap = m_device->createTexture(texDesc);
    }

    SceneReloaded();
}

// TODO: combine these

static inline uint floatToUInt(float _V, float _Scale)
{
    return (uint)floor(_V * _Scale + 0.5f);
}

static inline uint FLOAT3_to_R8G8B8_UNORM(float unpackedInputX, float unpackedInputY, float unpackedInputZ)
{
    return (floatToUInt(saturate(unpackedInputX), 0xFF) & 0xFF) |
        ((floatToUInt(saturate(unpackedInputY), 0xFF) & 0xFF) << 8) |
        ((floatToUInt(saturate(unpackedInputZ), 0xFF) & 0xFF) << 16);
}

static void packLightColor(const float3& color, PolymorphicLightInfo& lightInfo)
{
    float maxRadiance = std::max(color.x, std::max(color.y, color.z));

    if (maxRadiance <= 0.f)
        return;

    float logRadiance = (::log2f(maxRadiance) - kPolymorphicLightMinLog2Radiance) / (kPolymorphicLightMaxLog2Radiance - kPolymorphicLightMinLog2Radiance);
    logRadiance = saturate(logRadiance);
    uint32_t packedRadiance = std::min(uint32_t(ceilf(logRadiance * 65534.f)) + 1, 0xffffu);
    float unpackedRadiance = ::exp2f((float(packedRadiance - 1) / 65534.f) * (kPolymorphicLightMaxLog2Radiance - kPolymorphicLightMinLog2Radiance) + kPolymorphicLightMinLog2Radiance);

    lightInfo.ColorTypeAndFlags |= FLOAT3_to_R8G8B8_UNORM(color.x / unpackedRadiance, color.y / unpackedRadiance, color.z / unpackedRadiance);
    lightInfo.LogRadiance |= packedRadiance;
    assert((lightInfo.LogRadiance & 0xFFFF0000)==0); 
}

// TODO: move this to Utils.hlsli and include from here and it should all work
static float2 OctWrap(float2 v)
{
    return float2((1.0f - abs(v.y)) * ((v.x >= 0.0) ? 1.0f : -1.0f),
        (1.0f - abs(v.x)) * ((v.y >= 0.0) ? 1.0f : -1.0f));
}

static float2 Encode_Oct(float3 n3)
{
    n3 /= (abs(n3.x) + abs(n3.y) + abs(n3.z));
    float2 n = n3.xy();
    n = n3.z >= 0.0 ? n : OctWrap(n);
    n = n * 0.5f + 0.5f;
    return n;
}

static uint NDirToOctUnorm32(float3 n)
{
    float2 p = Encode_Oct(n);
    p = saturate(p * 0.5f + 0.5f);
    return uint(p.x * 0xfffe) | (uint(p.y * 0xfffe) << 16);
}

// Modified from original, based on the method from the DX fallback layer sample
static uint16_t fp32ToFp16(float v)
{
    // Multiplying by 2^-112 causes exponents below -14 to denormalize
    static const union FU {
        uint ui;
        float f;
    } multiple = { 0x07800000 }; // 2**-112

    FU BiasedFloat;
    BiasedFloat.f = v * multiple.f;
    const uint u = BiasedFloat.ui;

    const uint sign = u & 0x80000000;
    uint body = u & 0x0fffffff;

    return (uint16_t)(sign >> 16 | body >> 13) & 0xFFFF;
}

static PolymorphicLightInfoFull ConvertLight( donut::engine::Light & light )
{
    PolymorphicLightInfo polymorphic; memset(&polymorphic, 0, sizeof(polymorphic));
    PolymorphicLightInfoEx polymorphicEx; memset(&polymorphicEx, 0, sizeof(polymorphicEx));

    switch (light.GetLightType())
    {
	    case LightType_Spot: 
        {
		    auto& spot = static_cast<const SpotLight&>(light);

            if (spot.radius == 0.f)
            {
                assert(false); // not tested with radius == 0
			    float3 flux = spot.color * spot.intensity;

			    polymorphic.ColorTypeAndFlags = (uint32_t)PolymorphicLightType::kPoint << kPolymorphicLightTypeShift | ((spot.outerAngle < 0)?(kPolymorphicLightShapingUseMinFalloff):(0));
                
			    packLightColor(flux, polymorphic);
			    polymorphic.Center = float3(spot.GetPosition());
                polymorphic.Direction1 = NDirToOctUnorm32(float3(normalize(spot.GetDirection())));
                polymorphic.Direction2 = fp32ToFp16(dm::radians(abs(spot.outerAngle)));
			    polymorphic.Direction2 |= fp32ToFp16(dm::radians(spot.innerAngle)) << 16;
            }
            else
            {
                float projectedArea = dm::PI_f * (spot.radius*spot.radius);
                float3 radiance = spot.color * spot.intensity / projectedArea;
                float softness = saturate(1.f - spot.innerAngle / abs(spot.outerAngle));

                polymorphic.ColorTypeAndFlags = (uint32_t)PolymorphicLightType::kSphere << kPolymorphicLightTypeShift | ((spot.outerAngle < 0)?(kPolymorphicLightShapingUseMinFalloff):(0));
                polymorphic.ColorTypeAndFlags |= kPolymorphicLightShapingEnableBit;
                packLightColor(radiance, polymorphic);
                polymorphic.Center = float3(spot.GetPosition());
                polymorphic.Scalars = fp32ToFp16(spot.radius);
                if (abs(spot.outerAngle) > 0)
                {
                    polymorphic.ColorTypeAndFlags |= kPolymorphicLightShapingEnableBit;
                    polymorphicEx.PrimaryAxis = NDirToOctUnorm32(float3(normalize(spot.GetDirection())));
                    polymorphicEx.CosConeAngleAndSoftness = fp32ToFp16(cosf(dm::radians(abs(spot.outerAngle))));
                    polymorphicEx.CosConeAngleAndSoftness |= fp32ToFp16(softness) << 16;
                }
                packLightColor(radiance, polymorphic);
            }

            // example for the IES profile - few things need connecting
            /* case LightType_Spot: {
            *    // Spot Light with ies profile
                 auto& spot = static_cast<const SpotLightWithProfile&>(light);
                 float projectedArea = dm::PI_f * square(spot.radius);
                 float3 radiance = spot.color * spot.intensity / projectedArea;
                 float softness = saturate(1.f - spot.innerAngle / abs(spot.outerAngle));

                 polymorphic.colorTypeAndFlags = (uint32_t)PolymorphicLightType::kSphere << kPolymorphicLightTypeShift;
                 polymorphic.colorTypeAndFlags |= kPolymorphicLightShapingEnableBit;
                 packLightColor(radiance, polymorphic);
                 polymorphic.center = float3(spot.GetPosition());
                 polymorphic.scalars = fp32ToFp16(spot.radius);
                 polymorphic.primaryAxis = packNormalizedVector(float3(normalize(spot.GetDirection())));
                 polymorphic.cosConeAngleAndSoftness = fp32ToFp16(cosf(dm::radians(abs(spot.outerAngle))));
                 polymorphic.cosConeAngleAndSoftness |= fp32ToFp16(softness) << 16;

                 if (spot.profileTextureIndex >= 0)
                 {
                     polymorphic.iesProfileIndex = spot.profileTextureIndex; <- note, shader side needs fixing too
                     polymorphic.colorTypeAndFlags |= kPolymorphicLightIesProfileEnableBit;
                 }

                 return true;
             }*/

	    } break;
        case LightType_Point: 
        {
            auto& point = static_cast<const donut::engine::PointLight&>(light);
     
            if (point.radius == 0.f)
            {
                float3 flux = point.color * point.intensity;

                polymorphic.ColorTypeAndFlags = (uint32_t)PolymorphicLightType::kPoint << kPolymorphicLightTypeShift;
                packLightColor(flux, polymorphic);
                polymorphic.Center = float3(point.GetPosition());
                // Set the default values so we can use the same path for spot lights 
                polymorphic.Direction2 = fp32ToFp16(dm::PI_f) | fp32ToFp16(0.0f) << 16;
            }
            else
            {
                float projectedArea = dm::PI_f * (point.radius*point.radius);
                float3 radiance = point.color * point.intensity / projectedArea;

                polymorphic.ColorTypeAndFlags = (uint32_t)PolymorphicLightType::kSphere << kPolymorphicLightTypeShift;
                packLightColor(radiance, polymorphic);
                polymorphic.Center = float3(point.GetPosition());
                polymorphic.Scalars = fp32ToFp16(point.radius);
            }
        } break;
    }

    return PolymorphicLightInfoFull::make(polymorphic, polymorphicEx);
}

// inline void ComputeBounds(LightingControlData& ctrlBuff, const float3 pos)
// {
//     ctrlBuff.SceneWorldMax = donut::math::max(ctrlBuff.SceneWorldMax, float4(pos, 0));
//     ctrlBuff.SceneWorldMin = donut::math::min(ctrlBuff.SceneWorldMin, float4(pos, 0));
// }

void LightsBaker::CollectEnvmapLightPlaceholders(const BakeSettings & settings, LightingControlData & ctrlBuff, std::vector<PolymorphicLightInfo> & outLightBuffer, std::vector<PolymorphicLightInfoEx> & outLightExBuffer, std::vector<uint> & outLightHistoryRemapCurrentToPastBuffer, std::vector<uint> & outLightHistoryRemapPastToCurrent)
{
    ctrlBuff.EnvmapQuadNodeCount += RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT;
    ctrlBuff.TotalLightCount += RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT;

    // insert placeholder light info
    PolymorphicLightInfo dummy; memset(&dummy, 0, sizeof(dummy));
    PolymorphicLightInfoEx dummyEx; memset(&dummyEx, 0, sizeof(dummyEx));
    dummy.ColorTypeAndFlags = (uint32_t)PolymorphicLightType::kEnvironmentQuad << kPolymorphicLightTypeShift;   // no need to fill this, it will be completely overwritten
    outLightBuffer.insert( outLightBuffer.end(), RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT, dummy );
    outLightExBuffer.insert( outLightExBuffer.end(), RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT, dummyEx );

    outLightHistoryRemapCurrentToPastBuffer.insert(outLightHistoryRemapCurrentToPastBuffer.end(), RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT, RTXPT_INVALID_LIGHT_INDEX);
    outLightHistoryRemapPastToCurrent.insert(outLightHistoryRemapPastToCurrent.end(), ctrlBuff.EnvmapQuadNodeCount, RTXPT_INVALID_LIGHT_INDEX);
}

void LightsBaker::CollectAnalyticLightsCPU(const BakeSettings & settings, const std::shared_ptr<ExtendedScene> & scene, LightingControlData & ctrlBuff, std::vector<PolymorphicLightInfo> & outLightBuffer, std::vector<PolymorphicLightInfoEx> & outLightExBuffer, std::unordered_map<size_t, uint32_t> & inoutHistoryRemapAnalyticLightIndices, std::vector<uint> & outLightHistoryRemapCurrentToPastBuffer, std::vector<uint> & outLightHistoryRemapPastToCurrent)
{
    const auto & allLights = scene->GetSceneGraph()->GetLights();

    for ( auto light : allLights )
    {
        switch (( light->GetLightType() ))
        {
        case LightType_Spot:
        case LightType_Point:
        {    
            PolymorphicLightInfoFull lightPackedFull = ConvertLight(*light);
            outLightBuffer.push_back( lightPackedFull.Base );
            outLightExBuffer.push_back( lightPackedFull.Extended );
            
            // we do this to see if we had this light in the previous frame and mark the index it had; we could add more than a memory pointer to hash, as memory locations can be reused for new lights
            size_t lightHash = reinterpret_cast<size_t>(light.get());
            outLightExBuffer.back().UniqueID = Hash32Combine( uint(lightHash>>32), uint(lightHash&0xFFFFFFFFULL) ); // this is only used for debug view coloring and validation
            
            uint historicIndex = RTXPT_INVALID_LIGHT_INDEX;
            auto entry = inoutHistoryRemapAnalyticLightIndices.find(lightHash);
            if( entry != inoutHistoryRemapAnalyticLightIndices.end() )
            {
                historicIndex = entry->second;
                entry->second = ctrlBuff.TotalLightCount; // update with the new index for next search; lights should be unique
            }
            else
                inoutHistoryRemapAnalyticLightIndices.insert( std::make_pair(lightHash, ctrlBuff.TotalLightCount) );
            outLightHistoryRemapCurrentToPastBuffer.push_back(historicIndex);
            
            // ComputeBounds( ctrlBuff, outLightBuffer.back().Center );
            ctrlBuff.AnalyticLightCount++;
            ctrlBuff.TotalLightCount++;
        } break;
        default: break;
        }
        if( outLightBuffer.size() >= RTXPT_LIGHTING_MAX_LIGHTS )
        {
            assert( false ); // no more room for lights!
            break;
        }
    }

    // use current-to-past to create past-to-current: 1st init past-to-current values to invalid; then fill them up for those we can find historic match
    uint startingLight = (uint)outLightHistoryRemapPastToCurrent.size(); assert( startingLight == RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT ); // we know we should have envmap placeholders set up before so do sanity check
    outLightHistoryRemapPastToCurrent.insert(outLightHistoryRemapPastToCurrent.end(), ctrlBuff.AnalyticLightCount, RTXPT_INVALID_LIGHT_INDEX);
    for( uint lightIndex = startingLight; lightIndex < outLightHistoryRemapCurrentToPastBuffer.size(); lightIndex++ )
    {
        uint historicIndex = outLightHistoryRemapCurrentToPastBuffer[lightIndex];
        if( historicIndex != RTXPT_INVALID_LIGHT_INDEX )
            outLightHistoryRemapPastToCurrent[historicIndex] = lightIndex;
    }

    assert(outLightBuffer.size() == outLightHistoryRemapCurrentToPastBuffer.size());
};

uint LightsBaker::ProcessGeometry( const BakeSettings & settings, const std::shared_ptr<ExtendedScene> & scene, std::vector<SubInstanceData> & subInstanceData, LightingControlData & ctrlBuff, std::vector<struct EmissiveTrianglesProcTask> & tasks, std::unordered_map<size_t, uint32_t> & inoutHistoryRemapAnalyticLightIndices )
{
    tasks.clear();

    uint totalTriangleLightCount = 0;

    uint lightBufferAllocCounter = ctrlBuff.TotalLightCount; // we've already pre-filled emissive and analytic
    assert( ctrlBuff.TotalLightCount == (ctrlBuff.AnalyticLightCount+ctrlBuff.EnvmapQuadNodeCount) );

    uint subInstanceIndex = 0;

    const auto& instances = scene->GetSceneGraph()->GetMeshInstances();
    for (const auto& instance : instances)
    {
        const auto& mesh = instance->GetMesh();

        // auto boundingBox = instance->GetNode()->GetGlobalBoundingBox();
        // ComputeBounds( ctrlBuff, boundingBox.m_mins );
        // ComputeBounds( ctrlBuff, boundingBox.m_maxs );

        //assert(instance->GetGeometryInstanceIndex() < geometryInstanceToLight.size());
        uint32_t firstGeometryInstanceIndex = instance->GetGeometryInstanceIndex();

        for (size_t geometryIndex = 0; geometryIndex < mesh->geometries.size(); ++geometryIndex, subInstanceIndex++)
        {
            const auto& geometry = mesh->geometries[geometryIndex];

            size_t instanceHash = 0;
            nvrhi::hash_combine(instanceHash, instance.get());
            nvrhi::hash_combine(instanceHash, geometryIndex);

            PTMaterial & materialPT = *PTMaterial::FromDonut(geometry->material);

            // this has nothing to do with emissive materials, it's instead used to match 
            if (materialPT.EnableAsAnalyticLightProxy)
            {
                SceneGraphNode * parent = instance->GetNode()->GetParent();
                if (parent != nullptr && parent->GetLeaf() != nullptr && parent->GetLeaf()->GetContentFlags() == SceneContentFlags::Lights )
                {
                    auto light = std::dynamic_pointer_cast<Light>(parent->GetLeaf());
                    if (light != nullptr && (light->GetLightType() == LightType_Spot || light->GetLightType() == LightType_Point) )
                    {
                        size_t lightHash = reinterpret_cast<size_t>(light.get());
                        uint convertedLightIndex = RTXPT_INVALID_LIGHT_INDEX;
                        auto entry = inoutHistoryRemapAnalyticLightIndices.find(lightHash);
                        if (entry != inoutHistoryRemapAnalyticLightIndices.end())
                            convertedLightIndex = entry->second;
                        
                        // now set the convertedLightIndex into subInstanceData
                        //fill it into what used to be sort key!!!
                        subInstanceData[subInstanceIndex].AnalyticProxyLightIndex = convertedLightIndex;
                    }
                }
            }
            else
                subInstanceData[subInstanceIndex].AnalyticProxyLightIndex = RTXPT_INVALID_LIGHT_INDEX;

            if (!materialPT.IsEmissive() || materialPT.SkipRender)
            {
                // remove the info about this instance, just in case it was emissive and now it's not
                m_historyRemapEmissiveLightBlockOffsets.erase(instanceHash);
                continue;
            }

            uint historicBufferOffset = RTXPT_INVALID_LIGHT_INDEX;
            auto entry = m_historyRemapEmissiveLightBlockOffsets.find(instanceHash);
            if (entry != m_historyRemapEmissiveLightBlockOffsets.end())
            {
                historicBufferOffset = entry->second;
                entry->second = lightBufferAllocCounter; // update with the new index for next search; lights should be unique
                subInstanceData[subInstanceIndex].EmissiveLightMappingOffset = 0xFFFFFFFF;
            }
            else
                m_historyRemapEmissiveLightBlockOffsets.insert(std::make_pair(instanceHash, lightBufferAllocCounter));

            subInstanceData[subInstanceIndex].EmissiveLightMappingOffset = lightBufferAllocCounter;

            // geometryInstanceToLight[firstGeometryInstanceIndex + geometryIndex] = lightBufferOffset;

            // // find the previous offset of this instance in the light buffer
            // auto pOffset = m_InstanceLightBufferOffsets.find(instanceHash);

            assert(geometryIndex < 0xfff);

            int triangleFrom = 0;
            int remainingTriangles = geometry->numIndices / 3;
            assert( geometry->numIndices % 3 == 0 );

            while( remainingTriangles > 0 )
            {
                EmissiveTrianglesProcTask task;

                task.InstanceIndex      = instance->GetInstanceIndex();
                task.GeometryIndex      = (uint)geometryIndex;
                task.TriangleIndexFrom  = triangleFrom;
                int taskTriangleCount   = std::min( remainingTriangles, LLB_MAX_TRIANGLES_PER_TASK );
                task.TriangleIndexTo    = task.TriangleIndexFrom + taskTriangleCount;
                triangleFrom += taskTriangleCount;
                task.DestinationBufferOffset = lightBufferAllocCounter;
                task.HistoricBufferOffset = historicBufferOffset;
                tasks.push_back(task);

                if( tasks.size() >= LLB_MAX_PROC_TASKS )
                {
                    assert( false && "Emissive triangle task buffer too small" );
                    return totalTriangleLightCount;
                }

                remainingTriangles -= taskTriangleCount;
                totalTriangleLightCount += taskTriangleCount;

                lightBufferAllocCounter += taskTriangleCount;
                if( historicBufferOffset != RTXPT_INVALID_LIGHT_INDEX )
                    historicBufferOffset += taskTriangleCount;
                assert( lightBufferAllocCounter < RTXPT_LIGHTING_MAX_LIGHTS );
            }
        }
    }

    assert( subInstanceData.size() == subInstanceIndex );

    return totalTriangleLightCount;
}

void LightsBaker::FillBindings(nvrhi::BindingSetDesc& outBindingSetDesc, const std::shared_ptr<ExtendedScene>& scene, std::shared_ptr<class MaterialsBaker> materialsBaker, std::shared_ptr<OmmBaker> ommBaker, nvrhi::BufferHandle subInstanceDataBuffer,
nvrhi::TextureHandle depthBuffer, nvrhi::TextureHandle motionVectors)
{
    if( depthBuffer == nullptr )
        depthBuffer = ((nvrhi::TextureHandle)m_commonPasses->m_BlackTexture.Get());
    if (motionVectors == nullptr)
        motionVectors = ((nvrhi::TextureHandle)m_commonPasses->m_BlackTexture.Get());
    nvrhi::TextureHandle envMapRadianceAndImportanceMap = m_envMapBaker->GetImportanceSampling()->GetRadianceAndImportanceMap();
    if (envMapRadianceAndImportanceMap == nullptr || !m_currentSettings.EnvMapParams.Enabled )
        envMapRadianceAndImportanceMap = ((nvrhi::TextureHandle)m_commonPasses->m_BlackTexture.Get());

    outBindingSetDesc.bindings = {
            //nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer),
            //nvrhi::BindingSetItem::PushConstants(1, sizeof(SampleMiniConstants)),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(20, m_constantBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_controlBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(1, m_lightsBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(2, m_lightsExBuffer),
            nvrhi::BindingSetItem::RawBuffer_UAV(3, m_scratchBuffer),
            nvrhi::BindingSetItem::TypedBuffer_UAV(4, m_scratchList),
            nvrhi::BindingSetItem::TypedBuffer_UAV(5, (m_ping)?(m_lightWeightsPing):(m_lightWeightsPong)),
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
#if RTXPT_LIGHTING_LOCAL_SAMPLING_BUFFER_IS_3D_TEXTURE
            nvrhi::BindingSetItem::Texture_UAV(18, m_NEE_AT_LocalSamplingBuffer ),
#else
            nvrhi::BindingSetItem::TypedBuffer_UAV(18, m_NEE_AT_LocalSamplingBuffer),
#endif
            nvrhi::BindingSetItem::Texture_SRV(10, depthBuffer), //((nvrhi::TextureHandle)m_NEE_AT_FeedbackBuffer.Get())),
            nvrhi::BindingSetItem::Texture_SRV(11, motionVectors),
            nvrhi::BindingSetItem::Texture_SRV(12, envMapRadianceAndImportanceMap),
            nvrhi::BindingSetItem::TypedBuffer_SRV(13, (!m_ping)?(m_lightWeightsPing):(m_lightWeightsPong)),
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

void LightsBaker::UpdateFrustumConsts(LightsBakerConstants & outConsts, const LightsBaker::BakeSettings & settings)
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

    //planes[LEFT_PLANE]  = float4(-m[0].w - m[0].x, -m[1].w - m[1].x, -m[2].w - m[2].x, m[3].w + m[3].x);
    //planes[RIGHT_PLANE] = float4(-m[0].w + m[0].x, -m[1].w + m[1].x, -m[2].w + m[2].x, m[3].w - m[3].x);
    //planes[TOP_PLANE]   = float4(-m[0].w + m[0].y, -m[1].w + m[1].y, -m[2].w + m[2].y, m[3].w - m[3].y);
    //planes[BOTTOM_PLANE]= float4(-m[0].w - m[0].y, -m[1].w - m[1].y, -m[2].w - m[2].y, m[3].w + m[3].y);
    //planes[NEAR_PLANE]  = float4(-m[0].z, -m[1].z, -m[2].z, m[3].z);
    //planes[FAR_PLANE]   = float4(-m[0].w + m[0].z, -m[1].w + m[1].z, -m[2].w + m[2].z, m[3].w - m[3].z);

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

        m_localJitter = dm::clamp(uint2(m_localJitterF * (float)RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE), uint2(0, 0), uint2(RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE - 1, RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE - 1));
    }
}

void LightsBaker::UpdateFrame(nvrhi::ICommandList* commandList, const BakeSettings& _settings, double sceneTime, const std::shared_ptr<ExtendedScene>& scene, std::shared_ptr<class MaterialsBaker> materialsBaker, 
    std::shared_ptr<class OmmBaker> ommBaker, nvrhi::BufferHandle subInstanceDataBuffer, std::vector<SubInstanceData>& subInstanceData)
{
    RAII_SCOPE( commandList->beginMarker("LightBaker");, commandList->endMarker(); );
    // RAII_SCOPE( commandList->setEnableAutomaticBarriers(false);, commandList->setEnableAutomaticBarriers(true); );

    m_ping = !m_ping;
    m_updateFrameCalledBeforePreRender = true;

    m_currentSettings = _settings;

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

    m_currentSettings.GlobalTemporalFeedbackRatio   = dm::clamp(m_currentSettings.GlobalTemporalFeedbackRatio, 0.0f, 0.95f);
    m_currentSettings.LocalTemporalFeedbackRatio    = dm::clamp(m_currentSettings.LocalTemporalFeedbackRatio, 0.0f, 0.95f);
    if (m_currentSettings.ImportanceSamplingType != 2)  // no feedback needed if not using NEE_AT
    {
        m_currentSettings.EnableAntiLag = false;
        m_currentSettings.GlobalTemporalFeedbackEnabled = m_currentSettings.LocalTemporalFeedbackEnabled = false;
        lastFrameLocalSamplesAvailable = false;
    }
    if (!m_currentSettings.GlobalTemporalFeedbackEnabled)
        m_currentSettings.GlobalTemporalFeedbackRatio = 0.0f;
    if (!m_currentSettings.LocalTemporalFeedbackEnabled)
    {
        m_currentSettings.LocalTemporalFeedbackRatio = 0.0f;
        m_currentSettings.EnableAntiLag = false;                // anti-lag only helps with local sampler, so no point doing it if disabled
    }

    // Constants
    LightingControlData ctrlBuff; memset(&ctrlBuff, 0, sizeof(ctrlBuff)); 
    LightsBakerConstants consts; memset(&consts, 0, sizeof(consts));

    UpdateFrustumConsts(consts, m_currentSettings);

    consts.UpdateCounter = m_updateCounter;
    consts.EnableMotionReprojection      = true;
    consts.DepthDisocclusionThreshold   = m_depthDisocclusionThreshold;
    consts.LocalSamplingTileJitter       = m_localJitter;
    consts.LocalSamplingTileJitterPrev   = m_prevLocalJitter;
    ctrlBuff.LocalSamplingTileJitter     = m_localJitter;
    ctrlBuff.LocalSamplingTileJitterPrev = m_prevLocalJitter;

    assert( _settings.ViewportSize.x > 0 && _settings.ViewportSize.y > 0 && _settings.PrevViewportSize.x > 0 && _settings.PrevViewportSize.y > 0 );
    consts.PrevOverCurrentViewportSize = m_currentSettings.PrevViewportSize / m_currentSettings.ViewportSize;

    bool lastFrameFeedbackAvailable = m_NEE_AT_FeedbackBufferFilled && (m_currentSettings.GlobalTemporalFeedbackEnabled || m_currentSettings.LocalTemporalFeedbackEnabled);
    const bool temporalFeedbackRequired = m_currentSettings.ImportanceSamplingType == 2;

    {
        if( m_currentSettings.EnvMapParams.Enabled )
        {
            assert(m_envMapBaker->GetImportanceSampling()->GetImportanceMapOnly() != nullptr);   //< if enabled, must have importance map
            consts.EnvMapParams = m_currentSettings.EnvMapParams;
            const float baseScale = 0.0002f;
            consts.DistantVsLocalRelativeImportance = m_currentSettings.DistantVsLocalImportanceScale * baseScale;
        }
        else
        {
            consts.DistantVsLocalRelativeImportance = 0.0f;
            consts.EnvMapParams = EnvMapSceneParams{ .Transform = float3x4::identity(), .InvTransform = float3x4::identity(), .ColorMultiplier = float3(1,1,1), .Enabled = 0.0f };
        }
        consts.EnvMapImportanceMapMIPCount      = m_envMapBaker->GetImportanceSampling()->GetImportanceMapMIPLevels();
        consts.EnvMapImportanceMapResolution    = m_envMapBaker->GetImportanceSampling()->GetImportanceMapResolution();
    }

    consts.FeedbackResolution           = uint2(m_NEE_AT_FeedbackCandidates->getDesc().width, m_NEE_AT_FeedbackCandidates->getDesc().height);
    consts.BlendedFeedbackResolution    = uint2(m_NEE_AT_FeedbackCandidatesBlended->getDesc().width, m_NEE_AT_FeedbackCandidatesBlended->getDesc().height);
    uint numTotalP0ThreadCount          = div_ceil(consts.FeedbackResolution.x, LLB_NUM_COMPUTE_THREADS_2D) * div_ceil(consts.FeedbackResolution.y, LLB_NUM_COMPUTE_THREADS_2D) * LLB_NUM_COMPUTE_THREADS_2D * LLB_NUM_COMPUTE_THREADS_2D;
    if (RTXPT_LIGHTING_COUNT_ONLY_ONE_GLOBAL_FEEDBACK==0) numTotalP0ThreadCount *= RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH;
    consts.TotalMaxFeedbackCount        = (lastFrameFeedbackAvailable)?(numTotalP0ThreadCount):(0);
    consts.LocalSamplingResolution      = uint2(m_localSamplingBufferWidth, m_localSamplingBufferHeight);
    consts.GlobalFeedbackUseRatio       = (lastFrameFeedbackAvailable) ? (m_currentSettings.GlobalTemporalFeedbackRatio): (0.0f);
    consts.LocalFeedbackUseRatio        = (lastFrameFeedbackAvailable) ? (m_currentSettings.LocalTemporalFeedbackRatio) : (0.0f);
    consts.ReservoirHistoryDropoff      = m_advSetting_reservoirHistoryDropoff;
    ctrlBuff.LocalSamplingResolution    = consts.LocalSamplingResolution;
    ctrlBuff.TotalMaxFeedbackCount      = consts.TotalMaxFeedbackCount;
    ctrlBuff.GlobalFeedbackUseRatio     = consts.GlobalFeedbackUseRatio;
    ctrlBuff.LocalFeedbackUseRatio      = consts.LocalFeedbackUseRatio;
    ctrlBuff.LightSampling_MIS_Boost    = m_currentSettings.LightSampling_MIS_Boost;
    ctrlBuff.DirectVsIndirectThreshold  = m_advSetting_DirectVsIndirectThreshold;

    ctrlBuff.ImportanceSamplingType = m_currentSettings.ImportanceSamplingType;

    ctrlBuff.TileBufferHeight = consts.LocalSamplingResolution.y;

    consts.DebugDrawType = (int)m_dbgDebugDrawType;
    //consts.DebugDrawDirect = m_dbgDebugDrawDirect?1:0;
    consts.DebugDrawTileLights = m_dbgDebugDrawTileLightConnections;
    consts.MouseCursorPos = m_currentSettings.MouseCursorPos;
    consts.ImportanceBoostIntensityDelta = m_importanceBoost_IntensityDelta?m_importanceBoost_IntensityDeltaMul:0.0f;
    consts.ImportanceBoostFrustumMul = m_importanceBoost_Frustum?m_importanceBoost_FrustumMul:0.0f;
    consts.ImportanceBoostFrustumFadeRangeInt = m_importanceBoost_FrustumFadeRangeInt;
    consts.ImportanceBoostFrustumFadeRangeExt = m_importanceBoost_FrustumFadeRangeExt;
    consts.LastFrameTemporalFeedbackAvailable = lastFrameFeedbackAvailable;
    consts.SceneCameraPos = m_currentSettings.CameraPosition;
    consts.SceneAverageContentsDistance = m_currentSettings.AverageContentsDistance;
    consts.LastFrameLocalSamplesAvailable = lastFrameLocalSamplesAvailable && lastFrameFeedbackAvailable;
    consts.AntiLagEnabled = m_currentSettings.EnableAntiLag;
    ctrlBuff.LastFrameTemporalFeedbackAvailable = lastFrameFeedbackAvailable;
    ctrlBuff.LastFrameLocalSamplesAvailable = consts.LastFrameLocalSamplesAvailable;
    ctrlBuff.TemporalFeedbackRequired = temporalFeedbackRequired && !m_dbgFreezeUpdates;

    // clear buffers
    m_scratchLightBuffer.clear(); m_scratchLightExBuffer.clear();
    m_scratchLightHistoryRemapCurrentToPastBuffer.clear();
    m_scratchLightHistoryRemapPastToCurrentBuffer.clear();
    // collect all environment lights (create placeholders to be filled on the GPU later)
    CollectEnvmapLightPlaceholders( m_currentSettings, ctrlBuff, m_scratchLightBuffer, m_scratchLightExBuffer, m_scratchLightHistoryRemapCurrentToPastBuffer, m_scratchLightHistoryRemapPastToCurrentBuffer );
    // collect all analytic lights
    CollectAnalyticLightsCPU( m_currentSettings, scene, ctrlBuff, m_scratchLightBuffer, m_scratchLightExBuffer, m_historyRemapAnalyticLightIndices, m_scratchLightHistoryRemapCurrentToPastBuffer, m_scratchLightHistoryRemapPastToCurrentBuffer );
    // collect all emissive triangles and other geometry specific work - this builds batch jobs on the CPU that are executed on the GPU later, but at the end of this step we know the exact number of added emissive triangles (even though some might be black)
    uint emissiveTriangleLightCount = ProcessGeometry(m_currentSettings, scene, subInstanceData, ctrlBuff, *m_scratchTaskBuffer, m_historyRemapAnalyticLightIndices);
    consts.TriangleLightTaskCount = (int)(*m_scratchTaskBuffer).size();
    assert( ctrlBuff.EnvmapQuadNodeCount == RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT );
    ctrlBuff.TriangleLightCount = emissiveTriangleLightCount;
    ctrlBuff.TotalLightCount = ctrlBuff.EnvmapQuadNodeCount + ctrlBuff.AnalyticLightCount + ctrlBuff.TriangleLightCount; assert(ctrlBuff.TotalLightCount <= RTXPT_LIGHTING_MAX_LIGHTS);
    consts.TotalLightCount = ctrlBuff.TotalLightCount;
    ctrlBuff.HistoricTotalLightCount = m_historicTotalLightCount;
    m_historicTotalLightCount = ctrlBuff.TotalLightCount;

    // Constant & control buffers must go first
    {
        RAII_SCOPE(commandList->beginMarker("ControlDataSetup");, commandList->endMarker(); );

        // build constants
        commandList->writeBuffer(m_constantBuffer, &consts, sizeof(consts));
        // control buffer (used for build but also later for sampling)
        commandList->writeBuffer(m_controlBuffer, &ctrlBuff, sizeof(ctrlBuff));
        m_currentCtrlBuff = ctrlBuff;
        m_currentConsts = consts;
        commandList->setBufferState(m_constantBuffer, nvrhi::ResourceStates::ShaderResource);
        commandList->setBufferState(m_controlBuffer, nvrhi::ResourceStates::UnorderedAccess);
    }

    // Bindings
    nvrhi::BindingSetDesc bindingSetDesc;
    FillBindings(bindingSetDesc, scene, materialsBaker, ommBaker, subInstanceDataBuffer, nullptr, nullptr);
    nvrhi::BindingSetHandle bindingSet = m_bindingCache.GetOrCreateBindingSet(bindingSetDesc, m_commonBindingLayout);

    nvrhi::BindingSetVector bindings = { bindingSet };
    nvrhi::BindingSetVector bindingsEx = { bindingSet, scene->GetDescriptorTable() };

    {
        // we can do this early although we might have to move it to a later location if doing multiple global updates per frame (unlikely?)
        RAII_SCOPE(commandList->beginMarker("ResetLightProxyCounters"); , commandList->endMarker(); );

        const dm::uint  items = ctrlBuff.TotalLightCount;
        const dm::uint  itemsPerGroup = LLB_NUM_COMPUTE_THREADS;
        m_resetLightProxyCounters.Execute(commandList, div_ceil(items, itemsPerGroup), 1, 1, bindingSet);
    }

    {
        // this is mostly for correctness/determinism - it will clean everything so any gaps in mapping to previous frame don't result in incorrect mapping
        RAII_SCOPE(commandList->beginMarker("ResetPastToCurrentHistory"); , commandList->endMarker(); );
        commandList->setBufferState(m_historyRemapPastToCurrentBuffer, nvrhi::ResourceStates::UnorderedAccess);
        m_resetPastToCurrentHistory.Execute(commandList, div_ceil(std::max(ctrlBuff.HistoricTotalLightCount, ctrlBuff.TotalLightCount), LLB_NUM_COMPUTE_THREADS), 1, 1, bindingSet);
    }

    {
        RAII_SCOPE(commandList->beginMarker("EnvLightsBackupPast"); , commandList->endMarker(); );

        commandList->setBufferState(m_lightsBuffer, nvrhi::ResourceStates::UnorderedAccess); // very likely unnecessary in practice, but the old lightsBuffer is read in this pass
        m_envLightsBackupPast.Execute(commandList, div_ceil(RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT, LLB_NUM_COMPUTE_THREADS), 1, 1, bindingSet);
    }

    // empty emissive and analytic lights get copied over first - they've been fully processed on the CPU
    {
        RAII_SCOPE(commandList->beginMarker("EnvmapAndAnalyticLightBuffers");, commandList->endMarker(); );

        commandList->commitBarriers();
        assert( (int)m_scratchLightBuffer.size() == (ctrlBuff.EnvmapQuadNodeCount+ctrlBuff.AnalyticLightCount) );
        assert( (int)m_scratchLightExBuffer.size() == (ctrlBuff.EnvmapQuadNodeCount+ctrlBuff.AnalyticLightCount) );
        // TODO: setting all barriers before to copy_dest will potentially reduce gaps between copies
        commandList->writeBuffer(m_lightsBuffer, m_scratchLightBuffer.data(), sizeof(PolymorphicLightInfo) * m_scratchLightBuffer.size());
        commandList->writeBuffer(m_lightsExBuffer, m_scratchLightExBuffer.data(), sizeof(PolymorphicLightInfoEx)* m_scratchLightExBuffer.size());
        commandList->writeBuffer(m_historyRemapCurrentToPastBuffer, m_scratchLightHistoryRemapCurrentToPastBuffer.data(), sizeof(uint) * m_scratchLightHistoryRemapCurrentToPastBuffer.size());
        commandList->writeBuffer(m_historyRemapPastToCurrentBuffer, m_scratchLightHistoryRemapPastToCurrentBuffer.data(), sizeof(uint) * m_scratchLightHistoryRemapPastToCurrentBuffer.size());
        commandList->writeBuffer(m_scratchBuffer, m_scratchTaskBuffer->data(), sizeof(EmissiveTrianglesProcTask)* consts.TriangleLightTaskCount);
    }

    // todo: make sure only those needed are set
    commandList->setBufferState(m_perLightProxyCounters, nvrhi::ResourceStates::UnorderedAccess); // we've written into proxy counters - barrier needs to be added to the queue 
    commandList->setTextureState(m_NEE_AT_FeedbackTotalWeight, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    commandList->setTextureState(m_NEE_AT_FeedbackCandidates, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    //commandList->setTextureState(m_NEE_AT_FeedbackTotalWeightScratch, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    //commandList->setTextureState(m_NEE_AT_FeedbackCandidatesScratch, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    commandList->setBufferState(m_lightsBuffer, nvrhi::ResourceStates::UnorderedAccess);
    commandList->setBufferState(m_lightsExBuffer, nvrhi::ResourceStates::UnorderedAccess);
    commandList->setBufferState(m_historyRemapCurrentToPastBuffer, nvrhi::ResourceStates::UnorderedAccess);
    commandList->setBufferState(m_historyRemapPastToCurrentBuffer, nvrhi::ResourceStates::UnorderedAccess);

    {
        RAII_SCOPE(commandList->beginMarker("EnvLightsSubdivideBase");, commandList->endMarker(); );
        m_envLightsSubdivideBase.Execute(commandList, 1, 1, 1, bindingSet); //the main output goes to scratchBuffer, with RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT offset and is consumed by EnvLightsBake
    }
    
    {
        RAII_SCOPE(commandList->beginMarker("EnvLightsSubdivideBoost"); , commandList->endMarker(); );
        commandList->setBufferState(m_scratchList, nvrhi::ResourceStates::UnorderedAccess);
        m_envLightsSubdivideBoost.Execute(commandList, RTXPT_NEEAT_ENVMAP_QT_UNBOOSTED_NODE_COUNT, 1, 1, bindingSet); //the main output goes to scratchBuffer, with RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT offset and is consumed by EnvLightsBake
    }

    // We can probably overlap this with EnvLightsSubdivide but I measure no perf benefit
    {
        RAII_SCOPE(commandList->beginMarker("BakeEmissiveTriangles"); , commandList->endMarker(); );
        
        if (consts.TriangleLightTaskCount > 0)
            m_bakeEmissiveTriangles.Execute(commandList, div_ceil(consts.TriangleLightTaskCount, 8), 1, 1, bindingsEx);

        commandList->setBufferState(m_lightsBuffer, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setBufferState(m_historyRemapCurrentToPastBuffer, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setBufferState(m_historyRemapPastToCurrentBuffer, nvrhi::ResourceStates::UnorderedAccess);
    }

    {
        RAII_SCOPE(commandList->beginMarker("EnvLightFillLookupMap"); , commandList->endMarker(); );
        
        commandList->setBufferState(m_lightsBuffer, nvrhi::ResourceStates::UnorderedAccess);

        m_envLightsFillLookupMap.Execute(commandList, RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT, 1, 1, bindings );
        
        commandList->setTextureState(m_envLightLookupMap, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    }

    {
        RAII_SCOPE(commandList->beginMarker("EnvLightsMapPastToCurrent"); , commandList->endMarker(); );

        commandList->setBufferState(m_scratchList, nvrhi::ResourceStates::UnorderedAccess);

        m_envLightsMapPastToCurrent.Execute(commandList, div_ceil(RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT, LLB_NUM_COMPUTE_THREADS), 1, 1, bindings );

        commandList->setBufferState(m_scratchList, nvrhi::ResourceStates::UnorderedAccess);
    }

    // note: this has to come after all lights have been baked and remap current to past & past to current buffers are valid
    if (lastFrameFeedbackAvailable)
    {
        {
            RAII_SCOPE(commandList->beginMarker("ProcessFeedbackHistoryP0"); , commandList->endMarker(); );

            m_processFeedbackHistoryP0.Execute(commandList, div_ceil(consts.FeedbackResolution.x, LLB_NUM_COMPUTE_THREADS_2D), div_ceil(consts.FeedbackResolution.y, LLB_NUM_COMPUTE_THREADS_2D), 1, bindings );

            commandList->setTextureState(m_NEE_AT_FeedbackTotalWeight, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
            commandList->setTextureState(m_NEE_AT_FeedbackCandidates, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
            commandList->setBufferState(m_controlBuffer, nvrhi::ResourceStates::UnorderedAccess);           // we've InterlockedAdd into u_controlBuffer (actually, we haven't, except in the validation verison, but leaving in for when enabling validation)
            commandList->setBufferState(m_perLightProxyCounters, nvrhi::ResourceStates::UnorderedAccess);   // we've InterlockedAdd into m_perLightProxyCounters
        }
    }

    if (m_currentSettings.EnableAntiLag)
    {
        RAII_SCOPE(commandList->beginMarker("ClearAntiLagFeedback");, commandList->endMarker(); );

        m_clearAntiLagFeedback.Execute(commandList, div_ceil(m_currentConsts.FeedbackResolution.x, 8), div_ceil(m_currentConsts.FeedbackResolution.y, 8), 1, bindings);

        commandList->setTextureState(m_NEE_AT_FeedbackTotalWeightScratch, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setTextureState(m_NEE_AT_FeedbackCandidatesScratch, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    }

    {
        RAII_SCOPE(commandList->beginMarker("ComputeWeights"); , commandList->endMarker(); );

        const dm::uint  items = ctrlBuff.TotalLightCount;
        const dm::uint  itemsPerGroup = LLB_LOCAL_BLOCK_SIZE * LLB_NUM_COMPUTE_THREADS;

        commandList->setBufferState(m_historyRemapCurrentToPastBuffer, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setBufferState(m_historyRemapPastToCurrentBuffer, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setBufferState((m_ping) ? (m_lightWeightsPing) : (m_lightWeightsPong), nvrhi::ResourceStates::UnorderedAccess);

        m_computeWeights.Execute(commandList, div_ceil(items, itemsPerGroup), 1, 1, bindingSet);

        commandList->setBufferState((m_ping)?(m_lightWeightsPing):(m_lightWeightsPong), nvrhi::ResourceStates::UnorderedAccess);
        commandList->setBufferState(m_controlBuffer, nvrhi::ResourceStates::UnorderedAccess);
    }

    {
        RAII_SCOPE(commandList->beginMarker("ComputeProxyCounts"); , commandList->endMarker(); );

        const dm::uint  items = ctrlBuff.TotalLightCount;
        const dm::uint  itemsPerGroup = LLB_NUM_COMPUTE_THREADS;
        m_computeProxyCounts.Execute(commandList, div_ceil(items, itemsPerGroup), 1, 1, bindingSet);

        commandList->setBufferState(m_perLightProxyCounters, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setBufferState(m_controlBuffer, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setBufferState(m_scratchList, nvrhi::ResourceStates::UnorderedAccess);
    }

    {
        nvrhi::ComputeState state;
        state.bindings = { bindingSet };

        {
            RAII_SCOPE(commandList->beginMarker("ComputeProxyBaselineOffsets");, commandList->endMarker(); );

            m_computeProxyBaselineOffsets.Execute(commandList, 1, 1, 1, bindingSet);

            commandList->setBufferState(m_lightSamplingProxies, nvrhi::ResourceStates::UnorderedAccess);
        }

        {
            RAII_SCOPE(commandList->beginMarker("CreateProxyJobs"); , commandList->endMarker(); );

            const dm::uint  items = ctrlBuff.TotalLightCount;
            const dm::uint  itemsPerGroup = LLB_NUM_COMPUTE_THREADS;
            m_createProxyJobs.Execute(commandList, div_ceil(items, itemsPerGroup), 1, 1, bindingSet);

            commandList->setBufferState(m_controlBuffer, nvrhi::ResourceStates::UnorderedAccess);   // because we've written into u_controlBuffer[0].ProxyBuildTaskCount
            commandList->setBufferState(m_scratchBuffer, nvrhi::ResourceStates::UnorderedAccess);   // because this is where jobs are stored
        }
    }
    
    {
        RAII_SCOPE(commandList->beginMarker("ExecuteProxyJobs"); , commandList->endMarker(); );

        const dm::uint  items = LLB_MAX_PROXY_PROC_TASKS; // this one is updated on GPU so it's not correct here so let's just brute force to max, compute shader will skip...
        const dm::uint  itemsPerGroup = LLB_NUM_COMPUTE_THREADS;
        const dm::uint  dispatchCountX = div_ceil(items, itemsPerGroup); assert(dispatchCountX<=65535); // more than this triggers EXECUTION WARNING #1296: OVERSIZED_DISPATCH
        m_executeProxyJobs.Execute(commandList, dispatchCountX, 1, 1, bindingSet);

        commandList->setBufferState(m_lightSamplingProxies, nvrhi::ResourceStates::UnorderedAccess);    // because we've filled it up
    }

    if( m_dbgDebugDrawLights )
    {
        RAII_SCOPE(commandList->beginMarker("DebugDrawLights"); , commandList->endMarker(); );

        commandList->setBufferState(m_controlBuffer, nvrhi::ResourceStates::UnorderedAccess);

        const dm::uint  items = RTXPT_LIGHTING_MAX_SAMPLING_PROXIES; // this one is updated on GPU so it's not correct here so let's just brute force to max, compute shader will skip...
        const dm::uint  itemsPerGroup = LLB_NUM_COMPUTE_THREADS;
        m_debugDrawLights.Execute(commandList, div_ceil(items, itemsPerGroup), 1, 1, bindingSet);
    }

    // for debugging only
    if (m_framesFromLastReadbackCopy == -1)
        commandList->copyBuffer(m_controlBufferReadback, 0, m_controlBuffer, 0, sizeof(LightingControlData) * 1); // first time copy, do nothing else
    else
    {
#if LLB_ENABLE_VALIDATION   // instant feedback but significant perf hit
        m_device->waitForIdle(); 
#else
        if (m_framesFromLastReadbackCopy > 5) // 5 is always safe, we won't have that many frames overlapping
#endif
        {
            // Copy from readback buffer to struct that's displayed in UI
            void* pData = m_device->mapBuffer(m_controlBufferReadback, nvrhi::CpuAccessMode::Read);
            assert(pData);
            memcpy(&m_lastReadback, pData, sizeof(LightingControlData) * 1);
            m_device->unmapBuffer(m_controlBufferReadback);

            // Copy from GPU buffer to CPU readback buffer
            commandList->copyBuffer(m_controlBufferReadback, 0, m_controlBuffer, 0, sizeof(LightingControlData) * 1);

            // Reset counter
            m_framesFromLastReadbackCopy = 0;
        }
    }
    commandList->commitBarriers();  // committing now avoids "D3D12 ERROR: ID3D12CommandList::ResourceBarrier: D3D12_RESOURCE_STATES has an invalid combination of state bits." later in DispatchRays; this needs to be debugged
    m_framesFromLastReadbackCopy++;

    // commandList->copyBuffer(m_lightingConstants, 0, m_controlBuffer, 0, sizeof(LightingControlData) * 1);
    // commandList->setBufferState(m_lightingConstants, nvrhi::ResourceStates::ConstantBuffer);
    // commandList->commitBarriers();
}

// #pragma optimize("", off)
#if RTXPT_LIGHTING_LOCAL_SAMPLING_BUFFER_IS_3D_TEXTURE
#define UAV_BARRIER_m_NEE_AT_LocalSamplingBuffer() { commandList->setTextureState(m_NEE_AT_LocalSamplingBuffer, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess); }
#else
#define UAV_BARRIER_m_NEE_AT_LocalSamplingBuffer() { commandList->setBufferState(m_NEE_AT_LocalSamplingBuffer, nvrhi::ResourceStates::UnorderedAccess); }
#endif

void LightsBaker::UpdatePreRender(nvrhi::ICommandList * commandList, const std::shared_ptr<ExtendedScene> & scene, std::shared_ptr<class MaterialsBaker> materialsBaker, std::shared_ptr<class OmmBaker> ommBaker, nvrhi::BufferHandle subInstanceDataBuffer, nvrhi::TextureHandle depthBuffer, nvrhi::TextureHandle motionVectors)
{
	bool updateControlBuffer = false;
    if (m_updateFrameCalledBeforePreRender)
    {
        m_updateFrameCalledBeforePreRender = false;
        bool lastFrameFeedbackAvailable = m_NEE_AT_FeedbackBufferFilled && (m_currentSettings.GlobalTemporalFeedbackEnabled || m_currentSettings.LocalTemporalFeedbackEnabled);
        m_currentConsts.LocalFeedbackUseRatio = (lastFrameFeedbackAvailable || m_currentSettings.EnableAntiLag) ? (m_currentSettings.LocalTemporalFeedbackRatio) : (0.0f);
        m_currentCtrlBuff.LocalFeedbackUseRatio = m_currentConsts.LocalFeedbackUseRatio;
        updateControlBuffer = true; // not always necessary - TODO: clean up
    }
    else
    {
        // this is the second+ pass when multi-sampling enabled - we need to update things again
        UpdateLocalJitter();
        
        m_updateCounter++;
        
        bool lastFrameLocalSamplesAvailable = m_currentCtrlBuff.LastFrameTemporalFeedbackAvailable; // if last frame had temporal feedback, it will have had built local (tile) sampling
        m_currentConsts.AntiLagEnabled = false; // there already was a full pass before so there's no point looking at anti-lag early data
        m_currentConsts.UpdateCounter = m_updateCounter;
        m_currentConsts.EnableMotionReprojection = false;
        m_currentCtrlBuff.LocalSamplingTileJitter = m_localJitter;
        m_currentCtrlBuff.LocalSamplingTileJitterPrev = m_prevLocalJitter;
        bool lastFrameFeedbackAvailable = m_NEE_AT_FeedbackBufferFilled && (m_currentSettings.GlobalTemporalFeedbackEnabled || m_currentSettings.LocalTemporalFeedbackEnabled);
        m_currentConsts.LocalFeedbackUseRatio = (lastFrameFeedbackAvailable) ? (m_currentSettings.LocalTemporalFeedbackRatio) : (0.0f);
        m_currentCtrlBuff.LocalFeedbackUseRatio = m_currentConsts.LocalFeedbackUseRatio;
        m_currentConsts.LastFrameTemporalFeedbackAvailable = lastFrameFeedbackAvailable;
        m_currentCtrlBuff.LastFrameTemporalFeedbackAvailable = lastFrameFeedbackAvailable;
        m_currentCtrlBuff.LastFrameLocalSamplesAvailable = lastFrameLocalSamplesAvailable && lastFrameFeedbackAvailable;
        updateControlBuffer = true;
    }

    nvrhi::BindingSetDesc bindingSetDesc;
    FillBindings(bindingSetDesc, scene, materialsBaker, ommBaker, subInstanceDataBuffer, depthBuffer, motionVectors);
    nvrhi::BindingSetHandle bindingSet = m_bindingCache.GetOrCreateBindingSet(bindingSetDesc, m_commonBindingLayout);
    nvrhi::BindingSetVector bindings = { bindingSet };

	if (updateControlBuffer) // do the partial control buffer update using CS - in practice, we could just do a partial copy into the buffer
    {
        RAII_SCOPE(commandList->beginMarker("ReUploadConstAndControlBuffers");, commandList->endMarker(); );

        // build constants
        commandList->writeBuffer(m_constantBuffer, &m_currentConsts, sizeof(m_currentConsts));
        commandList->setBufferState(m_constantBuffer, nvrhi::ResourceStates::ShaderResource);

        // control buffer
    	m_updateControlBufferMultipass.Execute(commandList, 1, 1, 1, bindings);
    	commandList->setBufferState(m_controlBuffer, nvrhi::ResourceStates::UnorderedAccess);
    }

    const dm::uint  itemsPerGroup = LLB_NUM_COMPUTE_THREADS_2D;

    // note: temporal feedback must come after ComputeWeights as ComputeWeights initializes counters to 0
    if (m_currentCtrlBuff.LastFrameTemporalFeedbackAvailable || m_currentConsts.AntiLagEnabled)
    {
        {
            RAII_SCOPE(commandList->beginMarker("ProcessFeedbackHistoryP1a"); , commandList->endMarker(); );

            commandList->setTextureState(m_NEE_AT_FeedbackTotalWeight, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
            commandList->setTextureState(m_NEE_AT_FeedbackCandidates, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
            commandList->setTextureState(m_NEE_AT_FeedbackTotalWeightScratch, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
            commandList->setTextureState(m_NEE_AT_FeedbackCandidatesScratch, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);

            m_processFeedbackHistoryP1a.Execute(commandList, div_ceil(m_currentConsts.BlendedFeedbackResolution.x, itemsPerGroup), div_ceil(m_currentConsts.BlendedFeedbackResolution.y, itemsPerGroup), 1, bindings);

            commandList->setTextureState(m_NEE_AT_FeedbackTotalWeightScratch, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
            commandList->setTextureState(m_NEE_AT_FeedbackCandidatesScratch, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
            commandList->setTextureState(m_NEE_AT_FeedbackTotalWeightBlended, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
            commandList->setTextureState(m_NEE_AT_FeedbackCandidatesBlended, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        }

        {
            RAII_SCOPE(commandList->beginMarker("ProcessFeedbackHistoryP1b");, commandList->endMarker(); );

            commandList->setTextureState(m_NEE_AT_FeedbackTotalWeight, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
            commandList->setTextureState(m_NEE_AT_FeedbackCandidates, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);

            m_processFeedbackHistoryP1b.Execute(commandList, div_ceil(m_currentConsts.FeedbackResolution.x, itemsPerGroup), div_ceil(m_currentConsts.FeedbackResolution.y, itemsPerGroup), 1, bindings);

            commandList->setTextureState(m_NEE_AT_FeedbackTotalWeightScratch, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
            commandList->setTextureState(m_NEE_AT_FeedbackCandidatesScratch, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        }

        {
            RAII_SCOPE(commandList->beginMarker("ProcessFeedbackHistoryP2"); , commandList->endMarker(); );
            m_processFeedbackHistoryP2.Execute(commandList, div_ceil(m_currentConsts.LocalSamplingResolution.x, itemsPerGroup), div_ceil(m_currentConsts.LocalSamplingResolution.y, itemsPerGroup), 1, bindings);
            UAV_BARRIER_m_NEE_AT_LocalSamplingBuffer();
        }

        RAII_SCOPE(commandList->beginMarker("ProcessFeedbackHistoryP3"); , commandList->endMarker(); );
        m_processFeedbackHistoryP3.Execute(commandList, m_currentConsts.LocalSamplingResolution.x, m_currentConsts.LocalSamplingResolution.y, 1, bindings);
        UAV_BARRIER_m_NEE_AT_LocalSamplingBuffer();

        if (m_currentConsts.DebugDrawTileLights || m_dbgDebugDrawType == LightingDebugViewType::TileHeatmap || m_dbgDebugDrawType == LightingDebugViewType::ValidateCorrectness || m_dbgFreezeFrustumUpdates)
        {
            UAV_BARRIER_m_NEE_AT_LocalSamplingBuffer();
            commandList->commitBarriers();

            RAII_SCOPE(commandList->beginMarker("ProcessFeedbackHistoryDebugViz"); , commandList->endMarker(); );
            m_processFeedbackHistoryDebugViz.Execute(commandList, div_ceil(m_currentConsts.LocalSamplingResolution.x, itemsPerGroup), div_ceil(m_currentConsts.LocalSamplingResolution.y, itemsPerGroup), 1, bindings);

            UAV_BARRIER_m_NEE_AT_LocalSamplingBuffer();
            commandList->commitBarriers();
        }
    }

    const bool temporalFeedbackRequired = m_currentSettings.ImportanceSamplingType == 2;
    if (m_currentCtrlBuff.TemporalFeedbackRequired)
    {
        RAII_SCOPE(commandList->beginMarker("ClearFeedbackHistory"); , commandList->endMarker(); );

        //commandList->setTextureState(m_NEE_AT_FeedbackTotalWeightScratch, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        //commandList->setTextureState(m_NEE_AT_FeedbackCandidatesScratch, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);

        m_clearFeedbackHistory.Execute( commandList, div_ceil(m_currentConsts.FeedbackResolution.x, itemsPerGroup), div_ceil(m_currentConsts.FeedbackResolution.y, itemsPerGroup), 1, bindings );

        commandList->setTextureState(m_NEE_AT_FeedbackTotalWeight, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setTextureState(m_NEE_AT_FeedbackCandidates, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);

        m_NEE_AT_FeedbackBufferFilled = true;  // the assumption is that the path tracing happens after and actually fills the data; it's fine if it doesn't, the clear ^ resets it to empty anyway
    }

    // this is useful to avoid "leaking" any barrier issues to subsequent passes which makes it difficult to debug
    commandList->commitBarriers();
}

bool LightsBaker::InfoGUI(float indent)
{
    RAII_SCOPE(ImGui::PushID("LightsBakerInfoGUI");, ImGui::PopID(); );

    const char* modes[] = { "Uniform", "Power+", "NEE-AT" };
    ImGui::Text("Current mode:  %s", modes[dm::clamp(m_lastReadback.ImportanceSamplingType, 0u, 2u)]);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("As set in Path Tracer Next Event Estimation options\n(in the future, mode will be set here)");

    ImGui::Text("Scene lights by type: ");
    ImGui::Text("   envmap quads:  %d", m_lastReadback.EnvmapQuadNodeCount);
    ImGui::Text("   emissive tris: %d", m_lastReadback.TriangleLightCount);
    ImGui::Text("   analytic:      %d", m_lastReadback.AnalyticLightCount);
    ImGui::Text("   TOTAL:         %d", m_lastReadback.TotalLightCount);
    ImGui::Text("(proxies: %d, weightsum: %.5f)", m_lastReadback.SamplingProxyCount, m_lastReadback.WeightsSum());
#if LLB_ENABLE_VALIDATION
    ImGui::Text("Validation:");
    float feedbackPerc = m_lastReadback.ValidFeedbackCount / float(m_currentConsts.FeedbackResolution.x * m_currentConsts.FeedbackResolution.y);
    if (RTXPT_LIGHTING_COUNT_ONLY_ONE_GLOBAL_FEEDBACK==0) feedbackPerc *= RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH;
    ImGui::Text(" feedback num: %d (%.3f)", m_lastReadback.ValidFeedbackCount, feedbackPerc);
#endif

    return false;
}

bool LightsBaker::DebugGUI(float indent)
{
    RAII_SCOPE(ImGui::PushID("LightsBakerDebugGUI"); , ImGui::PopID(); );
    
    bool resetAccumulation = false;
    #define IMAGE_QUALITY_OPTION(code) do{if (code) resetAccumulation = true;} while(false)

    ImGui::Checkbox("Debug draw all lights", &m_dbgDebugDrawLights);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Wireframe colour indicates type: red - environment map; green - emissive triangles; blue - analytic.");

    ImGui::Checkbox("Debug draw NEE-AT tile light connections", &m_dbgDebugDrawTileLightConnections);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Shows lights sampled by a specific tile local sampling pdf");

    ImGui::Checkbox("Freeze NEE-AT feedback updates", &m_dbgFreezeUpdates);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Feedback from the path tracer will remain frozen while this option is enabled.");

    const char* debugOptions = "Disabled\0MissingFeedbackDirect\0MissingFeedbackIndirect\0FeedbackRawDirect\0FeedbackRawIndirect\0FeedbackAfterClear\0LowResBlendedFeedback\0TileHeatmap\0Disocclusion\0ValidateCorrectness\0\0";
    ImGui::Combo("NEE-AT debug view", (int*)&m_dbgDebugDrawType, debugOptions);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show various NEE-AT buffers");

    // if( m_dbgDebugDrawType != LightingDebugViewType::Disabled || m_dbgDebugDrawTileLightConnections )
    // {
    //     RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent););
    //     ImGui::Checkbox("NEE-AT: debug show direct part", &m_dbgDebugDrawDirect);
    //     if (ImGui::IsItemHovered()) ImGui::SetTooltip("If set, debug view shows direct lighting buffers; otherwise it shows indirect lighting buffers");
    // }

    ImGui::Checkbox("Disable local tile jitter", &m_dbgDebugDisableJitter);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Mapping from pixels to tiles will be jittered to avoid denoising artifacts.\nIt also helps with spatial sharing.\nDisable for debugging.");
    
    ImGui::Checkbox("Debug freeze frustum updates", &m_dbgFreezeFrustumUpdates);

#if 1
    ImGui::Separator();
    if (ImGui::CollapsingHeader("Advanced settings", 0/*ImGuiTreeNodeFlags_DefaultOpen*/))
    {
        ImGui::SliderFloat("DirectVsIndirectThreshold", &m_advSetting_DirectVsIndirectThreshold, 0.02f, 2.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Used to determine whether to use direct vs indirect light caching strategy for current surface.");

        ImGui::SliderFloat("ReservoirHistoryDropoff", &m_advSetting_reservoirHistoryDropoff, 0.0f, 0.1f, "%.2f", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("The amount of history sharing from past and from neighbours. Some is useful, \ntoo much will add lag and allow strong lights to dwarf out others.");

        ImGui::SliderFloat("DepthDisocclusionThreshold", &m_depthDisocclusionThreshold, 0.999f, 20.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("During motion reprojection, drop samples if really far from target");

        ImGui::Checkbox("Sample environment proxy lights", &m_advSetting_SampleBakedEnvironment);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("If enabled, environment map texture will not be sampled directly by NEE\nbut will be baked into sampling proxies like emissive triangles.\nBiased, faster but more blurry shadows in some cases.");

        {
            ImGui::Text("Importance boosts:");
            RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent););
            ImGui::Checkbox("...by light intensity change", &m_importanceBoost_IntensityDelta);
            {
                RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent););
                RAII_SCOPE(ImGui::PushID("Delta");, ImGui::PopID(); );
                ImGui::InputFloat("multiplier", &m_importanceBoost_IntensityDeltaMul);
                m_importanceBoost_IntensityDeltaMul = dm::clamp(m_importanceBoost_IntensityDeltaMul, 0.0f, 100.0f);
            }
            ImGui::Checkbox("...by light frustum proximity", &m_importanceBoost_Frustum);
            {
                RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent););
                RAII_SCOPE(ImGui::PushID("FrustProx"); , ImGui::PopID(); );
                ImGui::InputFloat("multiplier", &m_importanceBoost_FrustumMul);
                m_importanceBoost_FrustumMul = dm::clamp(m_importanceBoost_FrustumMul, 0.0f, 100.0f);
                ImGui::InputFloat("internal fade range scale", &m_importanceBoost_FrustumFadeRangeInt);
                m_importanceBoost_FrustumFadeRangeInt = dm::clamp(m_importanceBoost_FrustumFadeRangeInt, 0.01f, 10000.0f);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("How fast the boost fades within the frustum, based on distance from viewer\nThe bigger the value, the slower it fades");
                ImGui::InputFloat("external fade range scale", &m_importanceBoost_FrustumFadeRangeExt);
                m_importanceBoost_FrustumFadeRangeExt = dm::clamp(m_importanceBoost_FrustumFadeRangeExt, 0.01f, 10000.0f);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("How fast the boost fades outside of the frustum, based on smallest distance from frustum planes\nThe bigger the value, the slower it fades");
            }
        }
    }
#endif

    return resetAccumulation;
}

void LightsBaker::SetGlobalShaderMacros(std::vector<donut::engine::ShaderMacro> & macros)
{
    macros.push_back({ "NEE_AT_SAMPLE_BAKED_ENVIRONMENT", (m_advSetting_SampleBakedEnvironment) ? ("1") : ("0") });
}
