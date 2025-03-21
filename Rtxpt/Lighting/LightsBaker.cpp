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
#include "../PathTracer/Utils/NoiseAndSequences.hlsli"

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

    m_scratchLightHistoryReset.insert(m_scratchLightHistoryReset.begin(), RTXPT_LIGHTING_MAX_LIGHTS, RTXPT_INVALID_LIGHT_INDEX);  // TODO: this is a leftover and should be replaced by proper reset in the shader

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

void LightsBaker::CreateRenderPasses(nvrhi::IBindingLayout* bindlessLayout, std::shared_ptr<engine::CommonRenderPasses> commonPasses, std::shared_ptr<ShaderDebug> shaderDebug, const uint2 screenResolution)
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
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
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
            nvrhi::BindingLayoutItem::Texture_UAV(12),              // u_feedbackReservoirBuffer
            nvrhi::BindingLayoutItem::Texture_UAV(13),              // u_processedFeedbackBuffer
            nvrhi::BindingLayoutItem::Texture_UAV(14),              // u_reprojectedFeedbackBuffer
            nvrhi::BindingLayoutItem::Texture_UAV(15),              // u_reprojectedLRFeedbackBuffer
            nvrhi::BindingLayoutItem::Texture_UAV(16),              // u_narrowSamplingBuffer
#if RTXPT_LIGHTING_NEEAT_ENABLE_RESERVOIR_HISTORY
            nvrhi::BindingLayoutItem::Texture_UAV(17),              // u_feedbackReservoirBufferScratch
#endif
            nvrhi::BindingLayoutItem::Texture_SRV(10),              // t_depthBuffer
            nvrhi::BindingLayoutItem::Texture_SRV(11),              // t_motionVectors
            nvrhi::BindingLayoutItem::Texture_SRV(12),              // t_envmapImportanceMap
            nvrhi::BindingLayoutItem::Sampler(0),                   // point sampler
            nvrhi::BindingLayoutItem::Sampler(1),                   // linear sampler
            nvrhi::BindingLayoutItem::Sampler(2),                   // s_MaterialSampler
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1),      // StructuredBuffer<SubInstanceData> t_SubInstanceData
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2),      // StructuredBuffer<InstanceData> t_InstanceData          
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(3),      // StructuredBuffer<GeometryData> t_GeometryData          
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(4),      // geometry debug buffer not needed here?
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(5),      // StructuredBuffer<MaterialPTData> t_MaterialPTData
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

    m_envLightsBackupPast      .Init(m_device, *m_shaderFactory, shaderFile, "EnvLightsBackupPast"      ,   shaderMacros, pipelineDesc.bindingLayouts);
    m_envLightsBake            .Init(m_device, *m_shaderFactory, shaderFile, "EnvLightsBake"            ,   shaderMacros, pipelineDesc.bindingLayouts);
    m_envLightsFillLookupMap   .Init(m_device, *m_shaderFactory, shaderFile, "EnvLightsFillLookupMap"   ,   shaderMacros, pipelineDesc.bindingLayouts);
    m_envLightsMapPastToCurrent.Init(m_device, *m_shaderFactory, shaderFile, "EnvLightsMapPastToCurrent",   shaderMacros, pipelineDesc.bindingLayouts);

    m_clearFeedbackHistory     .Init(m_device, *m_shaderFactory, shaderFile, "ClearFeedbackHistory",        shaderMacros, pipelineDesc.bindingLayouts);

    m_processFeedbackHistoryP0      .Init(m_device, *m_shaderFactory, shaderFile, "ProcessFeedbackHistoryP0"        , shaderMacros, pipelineDesc.bindingLayouts);
    m_processFeedbackHistoryP1      .Init(m_device, *m_shaderFactory, shaderFile, "ProcessFeedbackHistoryP1"        , shaderMacros, pipelineDesc.bindingLayouts);
    m_processFeedbackHistoryP2      .Init(m_device, *m_shaderFactory, shaderFile, "ProcessFeedbackHistoryP2"        , shaderMacros, pipelineDesc.bindingLayouts);
    m_processFeedbackHistoryP3a     .Init(m_device, *m_shaderFactory, shaderFile, "ProcessFeedbackHistoryP3a"       , shaderMacros, pipelineDesc.bindingLayouts);
    m_processFeedbackHistoryP3b     .Init(m_device, *m_shaderFactory, shaderFile, "ProcessFeedbackHistoryP3b"       , shaderMacros, pipelineDesc.bindingLayouts);
    m_processFeedbackHistoryDebugViz.Init(m_device, *m_shaderFactory, shaderFile, "ProcessFeedbackHistoryDebugViz"  , shaderMacros, pipelineDesc.bindingLayouts);

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
    m_constantBuffer = m_controlBuffer = m_lightsBuffer = m_lightsExBuffer = m_historyRemapCurrentToPastBuffer = m_historyRemapPastToCurrentBuffer = m_scratchBuffer = m_lightWeights = m_perLightProxyCounters = m_scratchList = m_lightSamplingProxies = nullptr;
    //m_lightingConstants = nullptr;
    m_device->waitForIdle();    // make sure readback buffer is no longer used by the GPU
    m_controlBufferReadback = nullptr;

    // Main constant buffer
    m_constantBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(LightsBakerConstants), "LightsBakerConstants", engine::c_MaxRenderPassConstantBufferVersions * 5));	// *5 we could be updating few times per frame

    {
        nvrhi::BufferDesc bufferDesc;
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

        bufferDesc.byteSize = sizeof(float) * RTXPT_LIGHTING_MAX_LIGHTS;
        bufferDesc.format = nvrhi::Format::R32_FLOAT;
        bufferDesc.debugName = "LightsWeights";
        m_lightWeights = m_device->createBuffer(bufferDesc);

        bufferDesc.format = nvrhi::Format::R32_UINT;
        bufferDesc.debugName = "HistoryRemapCurrentToPast";
        m_historyRemapCurrentToPastBuffer = m_device->createBuffer(bufferDesc);
        bufferDesc.debugName = "HistoryRemapPastToCurrent";
        m_historyRemapPastToCurrentBuffer = m_device->createBuffer(bufferDesc);
        bufferDesc.debugName = "PerLightProxyCounters";
        m_perLightProxyCounters = m_device->createBuffer(bufferDesc);
        bufferDesc.debugName = "ScratchList";
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

    assert(screenResolution.x > 0 && screenResolution.y > 0);
    if (m_NEE_AT_FeedbackBuffer == nullptr || m_NEE_AT_FeedbackBuffer->getDesc().width != screenResolution.x || m_NEE_AT_FeedbackBuffer->getDesc().height != screenResolution.y*2)
    {
        // destroy before creating to avoid lifetimes of old and new overlapping (even with itself, due to assignment operator) - avoids fragmentation and peaks
        m_NEE_AT_ProcessedFeedbackBuffer = nullptr;
        m_NEE_AT_ReprojectedFeedbackBuffer = nullptr;
        m_NEE_AT_ReprojectedLRFeedbackBuffer = nullptr;
        m_NEE_AT_SamplingBuffer = nullptr;
        if (m_NEE_AT_FeedbackBuffer != nullptr)
        {
            m_device->waitForIdle();    // make sure buffer is no longer used by the GPU
            m_NEE_AT_FeedbackBuffer = nullptr;
#if RTXPT_LIGHTING_NEEAT_ENABLE_RESERVOIR_HISTORY
            m_NEE_AT_FeedbackBufferScratch = nullptr;
#endif
        }

        nvrhi::TextureDesc desc;
        desc.width = screenResolution.x;
        desc.height = screenResolution.y*2;
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
        desc.format = nvrhi::Format::RG32_UINT;
        desc.debugName = "NEE_AT_FeedbackReservoirBuffer";
        m_NEE_AT_FeedbackBuffer = m_device->createTexture(desc);
#if RTXPT_LIGHTING_NEEAT_ENABLE_RESERVOIR_HISTORY
        desc.debugName = "NEE_AT_FeedbackReservoirBufferScratch";
        m_NEE_AT_FeedbackBufferScratch = m_device->createTexture(desc);
#endif
        m_NEE_AT_FeedbackBufferFilled = false;

        desc.format = nvrhi::Format::RG32_UINT;
        desc.debugName = "NEE_AT_ReprojectedFeedbackBuffer";
        m_NEE_AT_ReprojectedFeedbackBuffer = m_device->createTexture(desc);
        desc.format = nvrhi::Format::R32_UINT;
        desc.debugName = "NEE_AT_ProcessedFeedbackBuffer";
        m_NEE_AT_ProcessedFeedbackBuffer = m_device->createTexture(desc);
        desc.debugName = "NEE_AT_ReprojectedLRFeedbackBuffer";
        desc.width = dm::div_ceil(screenResolution.x, RTXPT_LIGHTING_LR_SAMPLING_BUFFER_SCALE);
#if RTXPT_LIGHTING_NEEAT_ENABLE_INDIRECT_LOCAL_LAYER
        desc.height = dm::div_ceil(screenResolution.y, RTXPT_LIGHTING_LR_SAMPLING_BUFFER_SCALE)*2;
#else
        desc.height = dm::div_ceil(screenResolution.y, RTXPT_LIGHTING_LR_SAMPLING_BUFFER_SCALE);
#endif
        m_NEE_AT_ReprojectedLRFeedbackBuffer = m_device->createTexture(desc);

        desc.dimension = nvrhi::TextureDimension::Texture3D;
        desc.format = nvrhi::Format::R32_UINT;
        desc.debugName = "NEE_AT_SamplingBuffer";
        desc.width  = dm::div_ceil(screenResolution.x, RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE);
#if RTXPT_LIGHTING_NEEAT_ENABLE_INDIRECT_LOCAL_LAYER
        desc.height = dm::div_ceil(screenResolution.y, RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE)*2;
#else
        desc.height = dm::div_ceil(screenResolution.y, RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE);
#endif
        // add border to accommodate for jitter offset
        desc.width += 1;
        desc.height += 1;

        desc.depth = RTXPT_LIGHTING_NARROW_PROXY_COUNT;
        assert(desc.depth == RTXPT_LIGHTING_NARROW_PROXY_COUNT);
        static_assert(RTXPT_LIGHTING_NARROW_PROXY_COUNT <= 256);
        m_NEE_AT_SamplingBuffer = m_device->createTexture(desc);

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
			    float3 flux = spot.color * spot.intensity;

			    polymorphic.ColorTypeAndFlags = (uint32_t)PolymorphicLightType::kPoint << kPolymorphicLightTypeShift;
			    packLightColor(flux, polymorphic);
			    polymorphic.Center = float3(spot.GetPosition());
                polymorphic.Direction1 = NDirToOctUnorm32(float3(normalize(spot.GetDirection())));
                polymorphic.Direction2 = fp32ToFp16(dm::radians(spot.outerAngle));
			    polymorphic.Direction2 |= fp32ToFp16(dm::radians(spot.innerAngle)) << 16;
            }
            else
            {
                float projectedArea = dm::PI_f * (spot.radius*spot.radius);
                float3 radiance = spot.color * spot.intensity / projectedArea;
                float softness = saturate(1.f - spot.innerAngle / spot.outerAngle);

                polymorphic.ColorTypeAndFlags = (uint32_t)PolymorphicLightType::kSphere << kPolymorphicLightTypeShift;
                polymorphic.ColorTypeAndFlags |= kPolymorphicLightShapingEnableBit;
                packLightColor(radiance, polymorphic);
                polymorphic.Center = float3(spot.GetPosition());
                polymorphic.Scalars = fp32ToFp16(spot.radius);
                polymorphicEx.PrimaryAxis = NDirToOctUnorm32(float3(normalize(spot.GetDirection())));
                polymorphicEx.CosConeAngleAndSoftness = fp32ToFp16(cosf(dm::radians(spot.outerAngle)));
                polymorphicEx.CosConeAngleAndSoftness |= fp32ToFp16(softness) << 16;
            }

            // example for the IES profile - few things need connecting
            /* case LightType_Spot: {
            *    // Spot Light with ies profile
                 auto& spot = static_cast<const SpotLightWithProfile&>(light);
                 float projectedArea = dm::PI_f * square(spot.radius);
                 float3 radiance = spot.color * spot.intensity / projectedArea;
                 float softness = saturate(1.f - spot.innerAngle / spot.outerAngle);

                 polymorphic.colorTypeAndFlags = (uint32_t)PolymorphicLightType::kSphere << kPolymorphicLightTypeShift;
                 polymorphic.colorTypeAndFlags |= kPolymorphicLightShapingEnableBit;
                 packLightColor(radiance, polymorphic);
                 polymorphic.center = float3(spot.GetPosition());
                 polymorphic.scalars = fp32ToFp16(spot.radius);
                 polymorphic.primaryAxis = packNormalizedVector(float3(normalize(spot.GetDirection())));
                 polymorphic.cosConeAngleAndSoftness = fp32ToFp16(cosf(dm::radians(spot.outerAngle)));
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
    ctrlBuff.EnvmapQuadNodeCount += RTXPT_LIGHTING_ENVMAP_QT_TOTAL_NODE_COUNT;
    ctrlBuff.TotalLightCount += RTXPT_LIGHTING_ENVMAP_QT_TOTAL_NODE_COUNT;

    // insert placeholder light info
    PolymorphicLightInfo dummy; memset(&dummy, 0, sizeof(dummy));
    PolymorphicLightInfoEx dummyEx; memset(&dummyEx, 0, sizeof(dummyEx));
    dummy.ColorTypeAndFlags = (uint32_t)PolymorphicLightType::kEnvironmentQuad << kPolymorphicLightTypeShift;   // no need to fill this, it will be completely overwritten
    outLightBuffer.insert( outLightBuffer.end(), RTXPT_LIGHTING_ENVMAP_QT_TOTAL_NODE_COUNT, dummy );
    outLightExBuffer.insert( outLightExBuffer.end(), RTXPT_LIGHTING_ENVMAP_QT_TOTAL_NODE_COUNT, dummyEx );

    outLightHistoryRemapCurrentToPastBuffer.insert(outLightHistoryRemapCurrentToPastBuffer.end(), RTXPT_LIGHTING_ENVMAP_QT_TOTAL_NODE_COUNT, RTXPT_INVALID_LIGHT_INDEX);
    outLightHistoryRemapPastToCurrent.insert(outLightHistoryRemapPastToCurrent.end(), ctrlBuff.EnvmapQuadNodeCount, RTXPT_INVALID_LIGHT_INDEX);
}

void LightsBaker::CollectAnalyticLightsCPU(const BakeSettings & settings, const std::shared_ptr<donut::engine::ExtendedScene> & scene, LightingControlData & ctrlBuff, std::vector<PolymorphicLightInfo> & outLightBuffer, std::vector<PolymorphicLightInfoEx> & outLightExBuffer, std::unordered_map<size_t, uint32_t> & inoutHistoryRemapAnalyticLightIndices, std::vector<uint> & outLightHistoryRemapCurrentToPastBuffer, std::vector<uint> & outLightHistoryRemapPastToCurrent)
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
    uint startingLight = (uint)outLightHistoryRemapPastToCurrent.size(); assert( startingLight == RTXPT_LIGHTING_ENVMAP_QT_TOTAL_NODE_COUNT ); // we know we should have envmap placeholders set up before so do sanity check
    outLightHistoryRemapPastToCurrent.insert(outLightHistoryRemapPastToCurrent.end(), ctrlBuff.AnalyticLightCount, RTXPT_INVALID_LIGHT_INDEX);
    for( uint lightIndex = startingLight; lightIndex < outLightHistoryRemapCurrentToPastBuffer.size(); lightIndex++ )
    {
        uint historicIndex = outLightHistoryRemapCurrentToPastBuffer[lightIndex];
        if( historicIndex != RTXPT_INVALID_LIGHT_INDEX )
            outLightHistoryRemapPastToCurrent[historicIndex] = lightIndex;
    }

    assert(outLightBuffer.size() == outLightHistoryRemapCurrentToPastBuffer.size());
};

uint LightsBaker::CreateEmissiveTriangleProcTasks( const BakeSettings & settings, const std::shared_ptr<donut::engine::ExtendedScene> & scene, std::vector<SubInstanceData> & subInstanceData, LightingControlData & ctrlBuff, std::vector<struct EmissiveTrianglesProcTask> & tasks )
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

            MaterialPT & materialPT = *MaterialPT::FromDonut(geometry->material);
            if (!materialPT.IsEmissive())
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

void LightsBaker::FillBindings(nvrhi::BindingSetDesc& outBindingSetDesc, const std::shared_ptr<donut::engine::ExtendedScene>& scene, std::shared_ptr<class MaterialsBaker> materialsBaker, std::shared_ptr<OmmBaker> ommBaker, nvrhi::BufferHandle subInstanceDataBuffer,
nvrhi::TextureHandle depthBuffer, nvrhi::TextureHandle motionVectors)
{
    if( depthBuffer == nullptr )
        depthBuffer = ((nvrhi::TextureHandle)m_commonPasses->m_BlackTexture.Get());
    if (motionVectors == nullptr)
        motionVectors = ((nvrhi::TextureHandle)m_commonPasses->m_BlackTexture.Get());
    nvrhi::TextureHandle envMapImportanceMap = m_envMapBaker->GetImportanceSampling()->GetImportanceMap();
    if (envMapImportanceMap == nullptr || !m_currentSettings.EnvMapParams.Enabled )
        envMapImportanceMap = ((nvrhi::TextureHandle)m_commonPasses->m_BlackTexture.Get());

    outBindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer),
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
            nvrhi::BindingSetItem::Texture_UAV(12, m_NEE_AT_FeedbackBuffer),
            nvrhi::BindingSetItem::Texture_UAV(13, m_NEE_AT_ProcessedFeedbackBuffer),
            nvrhi::BindingSetItem::Texture_UAV(14, m_NEE_AT_ReprojectedFeedbackBuffer),
            nvrhi::BindingSetItem::Texture_UAV(15, m_NEE_AT_ReprojectedLRFeedbackBuffer),
            nvrhi::BindingSetItem::Texture_UAV(16, m_NEE_AT_SamplingBuffer),
#if RTXPT_LIGHTING_NEEAT_ENABLE_RESERVOIR_HISTORY
            nvrhi::BindingSetItem::Texture_UAV(17, m_NEE_AT_FeedbackBufferScratch),
#endif
            nvrhi::BindingSetItem::Texture_SRV(10, depthBuffer), //((nvrhi::TextureHandle)m_NEE_AT_FeedbackBuffer.Get())),
            nvrhi::BindingSetItem::Texture_SRV(11, motionVectors),
            nvrhi::BindingSetItem::Texture_SRV(12, envMapImportanceMap),
            nvrhi::BindingSetItem::Sampler(0, m_pointSampler),
            nvrhi::BindingSetItem::Sampler(1, m_linearSampler),
            nvrhi::BindingSetItem::Sampler(2, m_commonPasses->m_AnisotropicWrapSampler),    // s_MaterialSampler
            nvrhi::BindingSetItem::StructuredBuffer_SRV(1, subInstanceDataBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(2, scene->GetInstanceBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(3, scene->GetGeometryBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(4, ommBaker->GetGeometryDebugBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(5, materialsBaker->GetMaterialDataBuffer()),
            nvrhi::BindingSetItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX, m_shaderDebug->GetGPUWriteBuffer()),
            nvrhi::BindingSetItem::Texture_UAV(SHADER_DEBUG_VIZ_TEXTURE_UAV_INDEX, m_shaderDebug->GetDebugVizTexture()),
    };
}

void LightsBaker::Update(nvrhi::ICommandList* commandList, const BakeSettings& _settings, double sceneTime, const std::shared_ptr<donut::engine::ExtendedScene>& scene, std::shared_ptr<class MaterialsBaker> materialsBaker, 
    std::shared_ptr<class OmmBaker> ommBaker, nvrhi::BufferHandle subInstanceDataBuffer, std::vector<SubInstanceData>& subInstanceData)
{
    RAII_SCOPE( commandList->beginMarker("LightBaker");, commandList->endMarker(); );
    // RAII_SCOPE( commandList->setEnableAutomaticBarriers(false);, commandList->setEnableAutomaticBarriers(true); );

    uint2 prevLocalJitter = m_localJitter;
    if (_settings.ResetFeedback)
    {
        m_updateCounter = 0;
        m_localJitter = prevLocalJitter = {0,0};
    }
    m_updateCounter++;

    if (!m_dbgDebugDisableJitter)
    {
        // Advance R2 jitter sequence
        // http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/

        if (m_updateCounter == (1<<16) )
            m_localJitterF = {0, 0}; // not sure how long can the sequence remain high quality, so perhaps best to reset after a period

        static const float g = 1.32471795724474602596f;
        static const float a1 = 1.0f / g;
        static const float a2 = 1.0f / (g * g);
        m_localJitterF[0] = fmodf(m_localJitterF[0] + a1, 1.0f);
        m_localJitterF[1] = fmodf(m_localJitterF[1] + a2, 1.0f);

        m_localJitter = dm::clamp( uint2(m_localJitterF * (float)RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE), uint2(0, 0), uint2(RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE-1, RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE-1) );
    }

    bool lastFrameLocalSamplesAvailable = m_currentCtrlBuff.LastFrameTemporalFeedbackAvailable; // if last frame had temporal feedback, it will have had built local (tile) sampling

    m_currentSettings = _settings;
    m_currentSettings.GlobalTemporalFeedbackRatio   = dm::clamp(m_currentSettings.GlobalTemporalFeedbackRatio, 0.0f, 0.95f);
    m_currentSettings.NarrowTemporalFeedbackRatio    = dm::clamp(m_currentSettings.NarrowTemporalFeedbackRatio, 0.0f, 0.95f);
    if (m_currentSettings.ImportanceSamplingType != 2)  // no feedback needed if not using NEE_AT
        m_currentSettings.GlobalTemporalFeedbackEnabled = m_currentSettings.NarrowTemporalFeedbackEnabled = false;
    if (!m_currentSettings.GlobalTemporalFeedbackEnabled)
        m_currentSettings.GlobalTemporalFeedbackRatio = 0.0f;
    if (!m_currentSettings.NarrowTemporalFeedbackEnabled)
        m_currentSettings.NarrowTemporalFeedbackRatio = 0.0f;

    // Constants
    LightingControlData ctrlBuff; memset(&ctrlBuff, 0, sizeof(ctrlBuff)); 
    LightsBakerConstants consts; memset(&consts, 0, sizeof(consts));

    ctrlBuff.LocalSamplingTileJitter = m_localJitter;
    ctrlBuff.LocalSamplingTileJitterPrev = prevLocalJitter;

    assert( _settings.ViewportSize.x > 0 && _settings.ViewportSize.y > 0 && _settings.PrevViewportSize.x > 0 && _settings.PrevViewportSize.y > 0 );
    consts.PrevOverCurrentViewportSize = m_currentSettings.PrevViewportSize / m_currentSettings.ViewportSize;

    bool lastFrameFeedbackAvailable = m_NEE_AT_FeedbackBufferFilled && !m_currentSettings.ResetFeedback && (m_currentSettings.GlobalTemporalFeedbackEnabled || m_currentSettings.NarrowTemporalFeedbackEnabled);
    const bool temporalFeedbackRequired = m_currentSettings.ImportanceSamplingType == 2;

    {
        if( m_currentSettings.EnvMapParams.Enabled )
        {
            assert(m_envMapBaker->GetImportanceSampling()->GetImportanceMap() != nullptr);   //< if enabled, must have importance map
            consts.EnvMapParams = m_currentSettings.EnvMapParams;
            const float defaultScale = 0.001f;
            consts.DistantVsLocalRelativeImportance = m_currentSettings.DistantVsLocalImportanceScale * defaultScale;
        }
        else
        {
            consts.DistantVsLocalRelativeImportance = 0.0f;
            consts.EnvMapParams = EnvMapSceneParams{ .Transform = float3x4::identity(), .InvTransform = float3x4::identity(), .ColorMultiplier = float3(1,1,1), .Enabled = 0.0f };
        }
        consts.EnvMapImportanceMapMIPCount      = m_envMapBaker->GetImportanceSampling()->GetImportanceMapMIPLevels();
        consts.EnvMapImportanceMapResolution    = m_envMapBaker->GetImportanceSampling()->GetImportanceMapResolution();
    }

    consts.FeedbackResolution = uint2(m_NEE_AT_FeedbackBuffer->getDesc().width, m_NEE_AT_FeedbackBuffer->getDesc().height / 2);
#if RTXPT_LIGHTING_NEEAT_ENABLE_INDIRECT_LOCAL_LAYER
    consts.LRFeedbackResolution = uint2(m_NEE_AT_ReprojectedLRFeedbackBuffer->getDesc().width, m_NEE_AT_ReprojectedLRFeedbackBuffer->getDesc().height/2);
    consts.NarrowSamplingResolution = uint2(m_NEE_AT_SamplingBuffer->getDesc().width, m_NEE_AT_SamplingBuffer->getDesc().height/2);
#else
    consts.LRFeedbackResolution = uint2(m_NEE_AT_ReprojectedLRFeedbackBuffer->getDesc().width, m_NEE_AT_ReprojectedLRFeedbackBuffer->getDesc().height);
    consts.NarrowSamplingResolution = uint2(m_NEE_AT_SamplingBuffer->getDesc().width, m_NEE_AT_SamplingBuffer->getDesc().height);
#endif
    consts.GlobalFeedbackUseRatio   = (lastFrameFeedbackAvailable) ? (m_currentSettings.GlobalTemporalFeedbackRatio): (0.0f);
    consts.NarrowFeedbackUseRatio    = (lastFrameFeedbackAvailable) ? (m_currentSettings.NarrowTemporalFeedbackRatio) : (0.0f);
    consts.SampleIndex = m_currentSettings.SampleIndex;
    ctrlBuff.GlobalFeedbackUseRatio = consts.GlobalFeedbackUseRatio;
    ctrlBuff.NarrowFeedbackUseRatio  = consts.NarrowFeedbackUseRatio;
    ctrlBuff.LightSampling_MIS_Boost    = m_currentSettings.LightSampling_MIS_Boost;
    ctrlBuff.DirectVsIndirectThreshold = m_advSetting_DirectVsIndirectThreshold;

    //ctrlBuff.SceneWorldMax = float4( -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX ); ctrlBuff.SceneWorldMin = float4( FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX );
    //ComputeBounds( ctrlBuff, settings.CameraPosition );
    ctrlBuff.SceneCameraPos = float4( m_currentSettings.CameraPosition, 0 );

    ctrlBuff.ImportanceSamplingType = m_currentSettings.ImportanceSamplingType;

    ctrlBuff.FeedbackBufferHeight = consts.FeedbackResolution.y;
    ctrlBuff.TileBufferHeight = consts.NarrowSamplingResolution.y;

    consts.DebugDrawType = (int)m_dbgDebugDrawType;
    consts.DebugDrawDirect = m_dbgDebugDrawDirect?1:0;
    consts.DebugDrawTileLights = m_dbgDebugDrawTileLightConnections;
    consts.MouseCursorPos = m_currentSettings.MouseCursorPos;
    ctrlBuff.LastFrameTemporalFeedbackAvailable = lastFrameFeedbackAvailable;
    ctrlBuff.LastFrameLocalSamplesAvailable = lastFrameLocalSamplesAvailable && lastFrameFeedbackAvailable;
    ctrlBuff.TemporalFeedbackRequired = temporalFeedbackRequired && !m_dbgFreezeUpdates;

    // clear buffers
    m_scratchLightBuffer.clear(); m_scratchLightExBuffer.clear();
    m_scratchLightHistoryRemapCurrentToPastBuffer.clear();
    m_scratchLightHistoryRemapPastToCurrentBuffer.clear();
    // collect all environment lights (create placeholders to be filled on the GPU later)
    CollectEnvmapLightPlaceholders( m_currentSettings, ctrlBuff, m_scratchLightBuffer, m_scratchLightExBuffer, m_scratchLightHistoryRemapCurrentToPastBuffer, m_scratchLightHistoryRemapPastToCurrentBuffer );
    // collect all analytic lights
    CollectAnalyticLightsCPU( m_currentSettings, scene, ctrlBuff, m_scratchLightBuffer, m_scratchLightExBuffer, m_historyRemapAnalyticLightIndices, m_scratchLightHistoryRemapCurrentToPastBuffer, m_scratchLightHistoryRemapPastToCurrentBuffer );
    // collect all emissive triangles - this builds batch jobs on the CPU that are executed on the GPU later, but at the end of this step we know the exact number of added emissive triangles (even though some might be black)
    uint emissiveTriangleLightCount = CreateEmissiveTriangleProcTasks(m_currentSettings, scene, subInstanceData, ctrlBuff, *m_scratchTaskBuffer);
    consts.TriangleLightTaskCount = (int)(*m_scratchTaskBuffer).size();
    assert( ctrlBuff.EnvmapQuadNodeCount == RTXPT_LIGHTING_ENVMAP_QT_TOTAL_NODE_COUNT );
    ctrlBuff.TriangleLightCount = emissiveTriangleLightCount;
    ctrlBuff.TotalLightCount = ctrlBuff.EnvmapQuadNodeCount + ctrlBuff.AnalyticLightCount + ctrlBuff.TriangleLightCount; assert(ctrlBuff.TotalLightCount <= RTXPT_LIGHTING_MAX_LIGHTS);
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
        commandList->setBufferState(m_constantBuffer, nvrhi::ResourceStates::ConstantBuffer);
        commandList->setBufferState(m_controlBuffer, nvrhi::ResourceStates::UnorderedAccess);
    }

    // Bindings
    nvrhi::BindingSetDesc bindingSetDesc;
    FillBindings(bindingSetDesc, scene, materialsBaker, ommBaker, subInstanceDataBuffer, nullptr, nullptr);
    nvrhi::BindingSetHandle bindingSet = m_bindingCache.GetOrCreateBindingSet(bindingSetDesc, m_commonBindingLayout);

    nvrhi::BindingSetVector bindings = { bindingSet };
    nvrhi::BindingSetVector bindingsEx = { bindingSet, scene->GetDescriptorTable() };

    // we can do this early
    {
        RAII_SCOPE(commandList->beginMarker("ResetLightProxyCounters"); , commandList->endMarker(); );

        const dm::uint  items = RTXPT_LIGHTING_MAX_LIGHTS; // this one is updated on GPU so it's not correct here so let's just brute force to max, compute shader will skip...
        const dm::uint  itemsPerGroup = LLB_NUM_COMPUTE_THREADS;
        m_resetLightProxyCounters.Execute(commandList, div_ceil(items, itemsPerGroup), 1, 1, bindingSet);
    }

    {
        RAII_SCOPE(commandList->beginMarker("ResetPastToCurrentHistory"); , commandList->endMarker(); );
         // TODO: this is a leftover and should be replaced by proper reset in the shader: last pass can clean after itself
        commandList->writeBuffer(m_historyRemapPastToCurrentBuffer, m_scratchLightHistoryReset.data(), sizeof(uint) * std::max(ctrlBuff.HistoricTotalLightCount, ctrlBuff.TotalLightCount));
    }

    {
        RAII_SCOPE(commandList->beginMarker("EnvLightsBackupPast"); , commandList->endMarker(); );

        commandList->setBufferState(m_lightsBuffer, nvrhi::ResourceStates::UnorderedAccess); // very likely unnecessary in practice, but the old lightsBuffer is read in this pass
        m_envLightsBackupPast.Execute(commandList, div_ceil(RTXPT_LIGHTING_ENVMAP_QT_TOTAL_NODE_COUNT, LLB_NUM_COMPUTE_THREADS), 1, 1, bindingSet);
    }

    // empty emissive and analytic lights get copied over first - they've been fully processed on the CPU
    {
        RAII_SCOPE(commandList->beginMarker("EnvmapAndAnalyticLightBuffers");, commandList->endMarker(); );

        commandList->commitBarriers();
        assert( (int)m_scratchLightBuffer.size() == (ctrlBuff.EnvmapQuadNodeCount+ctrlBuff.AnalyticLightCount) );
        assert( (int)m_scratchLightExBuffer.size() == (ctrlBuff.EnvmapQuadNodeCount+ctrlBuff.AnalyticLightCount) );
        commandList->writeBuffer(m_lightsBuffer, m_scratchLightBuffer.data(), sizeof(PolymorphicLightInfo) * m_scratchLightBuffer.size());
        commandList->writeBuffer(m_lightsExBuffer, m_scratchLightExBuffer.data(), sizeof(PolymorphicLightInfoEx)* m_scratchLightExBuffer.size());
        commandList->writeBuffer(m_historyRemapCurrentToPastBuffer, m_scratchLightHistoryRemapCurrentToPastBuffer.data(), sizeof(uint) * m_scratchLightHistoryRemapCurrentToPastBuffer.size());
        commandList->writeBuffer(m_historyRemapPastToCurrentBuffer, m_scratchLightHistoryRemapPastToCurrentBuffer.data(), sizeof(uint) * m_scratchLightHistoryRemapPastToCurrentBuffer.size());
        commandList->writeBuffer(m_scratchBuffer, m_scratchTaskBuffer->data(), sizeof(EmissiveTrianglesProcTask)* consts.TriangleLightTaskCount);
    }

    // needed for ProcessFeedbackHistoryP0
    commandList->setBufferState(m_perLightProxyCounters, nvrhi::ResourceStates::UnorderedAccess); // we've written into proxy counters - barrier needs to be added to the queue 
    commandList->setTextureState(m_NEE_AT_FeedbackBuffer, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);  // note: setComputeState below will commit barriers so ordering is important

    {
        RAII_SCOPE(commandList->beginMarker("EnvLightsBake");, commandList->endMarker(); );

        commandList->setBufferState(m_lightsBuffer, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setBufferState(m_lightsExBuffer, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setBufferState(m_historyRemapCurrentToPastBuffer, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setBufferState(m_historyRemapPastToCurrentBuffer, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setBufferState(m_scratchBuffer, nvrhi::ResourceStates::UnorderedAccess); // not needed for this pass, first needed in BakeEmissiveTriangles
        
        m_envLightsBake.Execute(commandList, 1, 1, 1, bindingSet);
    }

    // We can overlap this with EnvLightsBake
    {
        RAII_SCOPE(commandList->beginMarker("BakeEmissiveTriangles"); , commandList->endMarker(); );

        if (consts.TriangleLightTaskCount > 0)
            m_bakeEmissiveTriangles.Execute(commandList, div_ceil(consts.TriangleLightTaskCount, LLB_NUM_COMPUTE_THREADS), 1, 1, bindingsEx);

        commandList->setBufferState(m_lightsBuffer, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setBufferState(m_historyRemapCurrentToPastBuffer, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setBufferState(m_historyRemapPastToCurrentBuffer, nvrhi::ResourceStates::UnorderedAccess);
    }

    {
        RAII_SCOPE(commandList->beginMarker("EnvLightFillLookupMap"); , commandList->endMarker(); );
        
        commandList->setBufferState(m_lightsBuffer, nvrhi::ResourceStates::UnorderedAccess);

        m_envLightsFillLookupMap.Execute(commandList, RTXPT_LIGHTING_ENVMAP_QT_TOTAL_NODE_COUNT, 1, 1, bindings );
        
        commandList->setTextureState(m_envLightLookupMap, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    }

    {
        RAII_SCOPE(commandList->beginMarker("EnvLightsMapPastToCurrent"); , commandList->endMarker(); );

        commandList->setBufferState(m_scratchList, nvrhi::ResourceStates::UnorderedAccess);

        m_envLightsMapPastToCurrent.Execute(commandList, div_ceil(RTXPT_LIGHTING_ENVMAP_QT_TOTAL_NODE_COUNT, LLB_NUM_COMPUTE_THREADS), 1, 1, bindings );

        commandList->setBufferState(m_scratchList, nvrhi::ResourceStates::UnorderedAccess);
    }

    // note: this has to come after all lights have been baked and remap current to past & past to current buffers are valid
    if (lastFrameFeedbackAvailable)
    {
        RAII_SCOPE(commandList->beginMarker("ProcessFeedbackHistoryP0"); , commandList->endMarker(); );

        const dm::uint  itemsPerGroup = 8;

        m_processFeedbackHistoryP0.Execute(commandList, div_ceil(consts.FeedbackResolution.x, itemsPerGroup * 3), div_ceil(consts.FeedbackResolution.y * 2, itemsPerGroup * 3), 1, bindings );

        commandList->setTextureState(m_NEE_AT_ProcessedFeedbackBuffer, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setBufferState(m_controlBuffer, nvrhi::ResourceStates::UnorderedAccess);   // we've InterlockedAdd into u_controlBuffer for TotalLightCount, barrier needed
    }

    {
        RAII_SCOPE(commandList->beginMarker("ComputeWeights"); , commandList->endMarker(); );

        const dm::uint  items = RTXPT_LIGHTING_MAX_LIGHTS; // this one is updated on GPU so it's not correct here so let's just brute force to max, compute shader will skip...
        const dm::uint  itemsPerGroup = LLB_LOCAL_BLOCK_SIZE * LLB_NUM_COMPUTE_THREADS;
        m_computeWeights.Execute(commandList, div_ceil(items, itemsPerGroup), 1, 1, bindingSet);

        commandList->setBufferState(m_lightWeights, nvrhi::ResourceStates::UnorderedAccess);
        commandList->setBufferState(m_controlBuffer, nvrhi::ResourceStates::UnorderedAccess);
    }

    {
        RAII_SCOPE(commandList->beginMarker("ComputeProxyCounts"); , commandList->endMarker(); );

        const dm::uint  items = RTXPT_LIGHTING_MAX_LIGHTS; // this one is updated on GPU so it's not correct here so let's just brute force to max, compute shader will skip...
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

            const dm::uint  items = RTXPT_LIGHTING_MAX_LIGHTS; // this one is updated on GPU so it's not correct here so let's just brute force to max, compute shader will skip...
            const dm::uint  itemsPerGroup = LLB_NUM_COMPUTE_THREADS;
            m_createProxyJobs.Execute(commandList, div_ceil(items, itemsPerGroup), 1, 1, bindingSet);

            commandList->setBufferState(m_controlBuffer, nvrhi::ResourceStates::UnorderedAccess);   // because we've written into u_controlBuffer[0].ProxyBuildTaskCount
            commandList->setBufferState(m_scratchBuffer, nvrhi::ResourceStates::UnorderedAccess);   // because this is where jobs are stored
        }
    }

    {
        RAII_SCOPE(commandList->beginMarker("ExecuteProxyJobs"); , commandList->endMarker(); );

        const dm::uint  items = LLB_MAX_PROXY_PROC_TASKS; // this one is updated on GPU so it's not correct here so let's just brute force to max, compute shader will skip...
        const dm::uint  itemsPerGroup = LLB_MAX_PROXIES_PER_TASK;
        m_executeProxyJobs.Execute(commandList, div_ceil(items, itemsPerGroup), 1, 1, bindingSet);

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
        if (m_framesFromLastReadbackCopy > 5) // 5 is always safe, we won't have that many frames overlapping
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

void LightsBaker::UpdateLate(nvrhi::ICommandList * commandList, const std::shared_ptr<donut::engine::ExtendedScene> & scene, std::shared_ptr<class MaterialsBaker> materialsBaker, std::shared_ptr<class OmmBaker> ommBaker, nvrhi::BufferHandle subInstanceDataBuffer, nvrhi::TextureHandle depthBuffer, nvrhi::TextureHandle motionVectors)
{
    nvrhi::BindingSetDesc bindingSetDesc;
    FillBindings(bindingSetDesc, scene, materialsBaker, ommBaker, subInstanceDataBuffer, depthBuffer, motionVectors);
    nvrhi::BindingSetHandle bindingSet = m_bindingCache.GetOrCreateBindingSet(bindingSetDesc, m_commonBindingLayout);
    nvrhi::BindingSetVector bindings = { bindingSet };

#if RTXPT_LIGHTING_NEEAT_ENABLE_INDIRECT_LOCAL_LAYER
    uint totalFeedbackY = m_currentConsts.FeedbackResolution.y*2;
    uint narrowSamplingY = m_currentConsts.NarrowSamplingResolution.y*2;
#else
    uint totalFeedbackY = m_currentConsts.FeedbackResolution.y;
    uint narrowSamplingY = m_currentConsts.NarrowSamplingResolution.y;
#endif

    const dm::uint  itemsPerGroup = 8;

    // note: temporal feedback must come after ComputeWeights as ComputeWeights initializes counters to 0
    if (m_currentCtrlBuff.LastFrameTemporalFeedbackAvailable)
    {
        {
            RAII_SCOPE(commandList->beginMarker("ProcessFeedbackHistoryP1");, commandList->endMarker(); );

            m_processFeedbackHistoryP1.Execute(commandList, div_ceil(m_currentConsts.FeedbackResolution.x, itemsPerGroup), div_ceil(totalFeedbackY, itemsPerGroup), 1, bindings);

            commandList->setTextureState(m_NEE_AT_ReprojectedFeedbackBuffer, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
            commandList->setTextureState(m_NEE_AT_ReprojectedLRFeedbackBuffer, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        }

        {
            RAII_SCOPE(commandList->beginMarker("ProcessFeedbackHistoryP2"); , commandList->endMarker(); );
            m_processFeedbackHistoryP2.Execute(commandList, div_ceil(m_currentConsts.NarrowSamplingResolution.x, itemsPerGroup), div_ceil(narrowSamplingY, itemsPerGroup), 1, bindings);
            commandList->setTextureState(m_NEE_AT_SamplingBuffer, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        }


        static bool optimizeFor32ThreadWaves = true;
        //ImGui::Checkbox("Optimize for 32-thread waves", &optimizeFor32ThreadWaves);
        if (m_deviceHas32ThreadWaves && optimizeFor32ThreadWaves)
        {
            // Optimised code for wave (warp) size of 32
            // Thread group size is [32 (wave size), 4 (num waves to put into a group), 1]
            // We dispatch one wave per tile
            // Thread coords: x : thread in wave
            //                y : tile x
            //                z : tile y
            const uint numWavesInGroup = 4;
            RAII_SCOPE(commandList->beginMarker("ProcessFeedbackHistoryP3b");, commandList->endMarker(); );
            m_processFeedbackHistoryP3b.Execute(commandList, 1, div_ceil(m_currentConsts.NarrowSamplingResolution.x, numWavesInGroup), narrowSamplingY, bindings);
        }
        else
        {
            // Thread group size is [itemsPerGroup, itemsPerGroup, 1]
            // We dispatch one thread per tile
            // Thread coords: x : tile x
            //                y : tile y
            //                z : 0
            RAII_SCOPE(commandList->beginMarker("ProcessFeedbackHistoryP3a");, commandList->endMarker(); );
            m_processFeedbackHistoryP3a.Execute(commandList, div_ceil(m_currentConsts.NarrowSamplingResolution.x, itemsPerGroup), div_ceil(narrowSamplingY, itemsPerGroup), 1, bindings);
        }
        commandList->setTextureState(m_NEE_AT_SamplingBuffer, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);

        if (m_currentConsts.DebugDrawTileLights || m_dbgDebugDrawType == LightingDebugViewType::TileHeatmap )
        {
            commandList->setTextureState(m_NEE_AT_SamplingBuffer, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
            commandList->commitBarriers();

            RAII_SCOPE(commandList->beginMarker("ProcessFeedbackHistoryDebugViz"); , commandList->endMarker(); );
            m_processFeedbackHistoryDebugViz.Execute(commandList, div_ceil(m_currentConsts.NarrowSamplingResolution.x, itemsPerGroup), div_ceil(narrowSamplingY, itemsPerGroup), 1, bindings);

            commandList->setTextureState(m_NEE_AT_SamplingBuffer, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
            commandList->commitBarriers();
        }
    }

    const bool temporalFeedbackRequired = m_currentSettings.ImportanceSamplingType == 2;
    if (m_currentCtrlBuff.TemporalFeedbackRequired)
    {
        RAII_SCOPE(commandList->beginMarker("ClearFeedbackHistory"); , commandList->endMarker(); );

        commandList->setTextureState(m_NEE_AT_FeedbackBuffer, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);

        m_clearFeedbackHistory.Execute( commandList, div_ceil(m_currentConsts.FeedbackResolution.x, itemsPerGroup), div_ceil(m_currentConsts.FeedbackResolution.y*2, itemsPerGroup), 1, bindings );

        commandList->setTextureState(m_NEE_AT_FeedbackBuffer, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);

        m_NEE_AT_FeedbackBufferFilled = true;  // the assumption is that the path tracing happens after and actually fills the data; it's fine if it doesn't, the clear ^ resets it to empty anyway
    }
}

bool LightsBaker::InfoGUI(float indent)
{
    RAII_SCOPE(ImGui::PushID("LightsBakerInfoGUI");, ImGui::PopID(); );

    const char* modes[] = { "Uniform", "Power", "NEE-AT" };
    ImGui::Text("Current use mode:  %s", modes[dm::clamp(m_lastReadback.ImportanceSamplingType, 0u, 2u)]);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("As set in Path Tracer Next Event Estimation options\n(in the future, mode will be set here)");

    ImGui::Text("Scene lights by type: ");
    ImGui::Text("   envmap quads:  %d", m_lastReadback.EnvmapQuadNodeCount);
    ImGui::Text("   emissive tris: %d", m_lastReadback.TriangleLightCount);
    ImGui::Text("   analytic:      %d", m_lastReadback.AnalyticLightCount);
    ImGui::Text("   TOTAL:         %d", m_lastReadback.TotalLightCount);
    ImGui::Text("(proxies: %d, weightsum: %.5f)", m_lastReadback.SamplingProxyCount, m_lastReadback.WeightsSum());

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

    const char* debugOptions = "Disabled\0MissingFeedback\0FeedbackRaw\0FeedbackProcessed\0FeedbackHistoric\0FeedbackLowRes\0FeedbackReadyForNew\0TileHeatmap\0\0";
    ImGui::Combo("NEE-AT debug view", (int*)&m_dbgDebugDrawType, debugOptions);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show various NEE-AT buffers");

    if( m_dbgDebugDrawType != LightingDebugViewType::Disabled || m_dbgDebugDrawTileLightConnections )
    {
        RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent););
        ImGui::Checkbox("NEE-AT: debug show direct part", &m_dbgDebugDrawDirect);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("If set, debug view shows direct lighting buffers; otherwise it shows indirect lighting buffers");
    }

    ImGui::Checkbox("Disable local tile jitter", &m_dbgDebugDisableJitter);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Mapping from pixels to tiles will be jittered to avoid denoising artifacts.\nIt also helps with spatial sharing.\nDisable for debugging.");
    

#if 1
    ImGui::Separator();
    if (ImGui::CollapsingHeader("Advanced settings", 0/*ImGuiTreeNodeFlags_DefaultOpen*/))
    {
        ImGui::SliderFloat("DirectVsIndirectThreshold", &m_advSetting_DirectVsIndirectThreshold, 0.01f, 1.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Used to determine whether to use direct vs indirect light caching strategy for current surface.");
    }
#endif

    return resetAccumulation;
}

