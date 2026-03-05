/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "GTAORenderer.h"
#include "SampleCommon/ComputePipelineBaker.h"
#include <donut/core/math/math.h>
#include "Shaders/SampleConstantBuffer.h"
#include "SampleCommon/SampleCommon.h"

namespace
{
    // Must match GTAOConstants in GTAOPasses.hlsl
    struct GTAOConstants
    {
        float4x4      MatWorldToView;
        float4x4      MatClipToWorldNoOffset;
        float       ProjScale;          // matViewToClip[0][0]
        float       Radius;
        float       FalloffEnd;
        float       TemporalAlpha;
        uint32_t    HalfWidth;
        uint32_t    HalfHeight;
        uint32_t    FullWidth;
        uint32_t    FullHeight;
        uint32_t    FrameIndex;
        float       ViewportSizeInvX;
        float       ViewportSizeInvY;
        uint32_t    Padding;
    };

    constexpr uint32_t kThreadGroupSize = 8;
    constexpr float    kDefaultRadius = 2.0f;
    constexpr float    kDefaultFalloffEnd = 2.0f;
    constexpr float    kDefaultTemporalAlpha = 0.9f;
}

static uint32_t DivCeil(uint32_t x, uint32_t y) { return (x + y - 1) / y; }

GTAORenderer::GTAORenderer(nvrhi::IDevice* device)
    : m_device(device)
{
}

GTAORenderer::~GTAORenderer()
{
    assert(!m_computeAOVariant); // Make sure destroy pipelines was called ahead of destruction
}

void GTAORenderer::CreatePipelines(nvrhi::BindingLayoutHandle globalBindingLayout,
                                   nvrhi::BindingLayoutHandle globalBindlessLayout,
                                   ComputePipelineBaker& computeBaker)
{
    // Point-clamp sampler for depth/normal reads
    {
        nvrhi::SamplerDesc desc;
        desc.setAllFilters(false);
        desc.addressU = nvrhi::SamplerAddressMode::Clamp;
        desc.addressV = nvrhi::SamplerAddressMode::Clamp;
        desc.addressW = nvrhi::SamplerAddressMode::Clamp;
        m_pointClampSampler = m_device->createSampler(desc);
    }

    // Constant buffer
    {
        nvrhi::BufferDesc desc;
        desc.byteSize = sizeof(GTAOConstants);
        desc.structStride = 0;
        desc.debugName = "GTAOConstants";
        desc.isConstantBuffer = true;
        desc.initialState = nvrhi::ResourceStates::ConstantBuffer;
        desc.keepInitialState = true;
        m_constantBuffer = m_device->createBuffer(desc);
    }

    // === ComputeAO binding layout (space1) ===
    // b0: GTAOConstants
    // t0: normals (full-res)
    // u0: raw AO output (half-res)
    // s0: point clamp sampler
    // Depth is read from t_DepthHierarchy in the global binding set (space0)
    // Push constants come from the global binding layout (space0)
    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.registerSpace = 1;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::ConstantBuffer(0),
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::Texture_UAV(0),
            nvrhi::BindingLayoutItem::Sampler(0),
        };
        m_computeAOBindingLayout = m_device->createBindingLayout(layoutDesc);

        m_computeAOVariant = computeBaker.CreateVariant(
            "IntroSample/GTAOPasses.hlsl", "GTAOComputeCS",
            { {"__GTAO_COMPUTE_CS__", "1"} },
            { globalBindingLayout, m_computeAOBindingLayout },
            "GTAOCompute");
    }

    // === SpatialFilter binding layout ===
    // b0: GTAOConstants, b1: push constants
    // t0: raw AO (half-res), t1: depth hierarchy (mip 0 = full-res depth)
    // u0: filtered AO output (half-res)
    // s0: point clamp sampler
    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::ConstantBuffer(0),
            nvrhi::BindingLayoutItem::PushConstants(1, sizeof(SampleMiniConstants)),
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::Texture_SRV(1),
            nvrhi::BindingLayoutItem::Texture_UAV(0),
            nvrhi::BindingLayoutItem::Sampler(0),
        };
        m_spatialFilterBindingLayout = m_device->createBindingLayout(layoutDesc);

        m_spatialFilterVariant = computeBaker.CreateVariant(
            "IntroSample/GTAOPasses.hlsl", "GTAOSpatialFilterCS",
            { {"__GTAO_SPATIAL_FILTER_CS__", "1"} },
            { m_spatialFilterBindingLayout },
            "GTAOSpatialFilter");
    }

    // === TemporalAccumulate binding layout ===
    // b0: GTAOConstants, b1: push constants
    // t0: filtered AO (half-res), t1: history AO (half-res), t2: motion vectors (full-res),
    // t3: depth hierarchy (mip 0 = full-res depth), t4: previous frame depth (full-res)
    // u0: output AO (full-res), u1: history write (half-res)
    // s0: point clamp sampler
    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::ConstantBuffer(0),
            nvrhi::BindingLayoutItem::PushConstants(1, sizeof(SampleMiniConstants)),
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::Texture_SRV(1),
            nvrhi::BindingLayoutItem::Texture_SRV(2),
            nvrhi::BindingLayoutItem::Texture_SRV(3),
            nvrhi::BindingLayoutItem::Texture_SRV(4),
            nvrhi::BindingLayoutItem::Texture_UAV(0),
            nvrhi::BindingLayoutItem::Texture_UAV(1),
            nvrhi::BindingLayoutItem::Sampler(0),
        };
        m_temporalAccumBindingLayout = m_device->createBindingLayout(layoutDesc);

        m_temporalAccumVariant = computeBaker.CreateVariant(
            "IntroSample/GTAOPasses.hlsl", "GTAOTemporalCS",
            { {"__GTAO_TEMPORAL_CS__", "1"} },
            { m_temporalAccumBindingLayout },
            "GTAOTemporal");
    }
}

void GTAORenderer::DestroyPipelines(ComputePipelineBaker& computeBaker)
{
    if (m_computeAOVariant)
        computeBaker.ReleaseVariant(m_computeAOVariant);
    if (m_spatialFilterVariant)
        computeBaker.ReleaseVariant(m_spatialFilterVariant);
    if (m_temporalAccumVariant)
        computeBaker.ReleaseVariant(m_temporalAccumVariant);

    m_computeAOVariant = nullptr;
    m_spatialFilterVariant = nullptr;
    m_temporalAccumVariant = nullptr;

    m_computeAOBindingSet = nullptr;
    m_spatialFilterBindingSet = nullptr;
    m_temporalAccumBindingSet = nullptr;
}

void GTAORenderer::OnRenderTargetsRecreated(uint32_t renderWidth, uint32_t renderHeight)
{
    m_fullWidth = renderWidth;
    m_fullHeight = renderHeight;
    m_halfWidth = (renderWidth + 1) / 2;
    m_halfHeight = (renderHeight + 1) / 2;

    auto createTexture = [&](const char* name, uint32_t w, uint32_t h) -> nvrhi::TextureHandle
    {
        nvrhi::TextureDesc desc;
        desc.width = w;
        desc.height = h;
        desc.format = nvrhi::Format::R8_UNORM;
        desc.debugName = name;
        desc.isUAV = true;
        desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        desc.keepInitialState = true;
        return m_device->createTexture(desc);
    };

    m_GTAORaw       = createTexture("GTAO_Raw",       m_halfWidth, m_halfHeight);
    m_GTAOFiltered  = createTexture("GTAO_Filtered",  m_halfWidth, m_halfHeight);
    m_GTAOHistory[0]= createTexture("GTAO_History0",  m_halfWidth, m_halfHeight);
    m_GTAOHistory[1]= createTexture("GTAO_History1",  m_halfWidth, m_halfHeight);
    m_historyReadIdx = 0;
    m_GTAOOutput    = createTexture("GTAO_Output",    m_fullWidth, m_fullHeight);

    // Invalidate binding sets so they get recreated with the new textures
    m_computeAOBindingSet = nullptr;
    m_spatialFilterBindingSet = nullptr;
    m_temporalAccumBindingSet = nullptr;
}

void GTAORenderer::Render(nvrhi::CommandListHandle commandList,
                          const SampleConstants& constants,
                          uint32_t frameIndex,
                          nvrhi::TextureHandle depthHierarchy,
                          nvrhi::TextureHandle fullResNormals,
                          nvrhi::TextureHandle fullResMotionVectors,
                          nvrhi::TextureHandle prevDepth,
                          nvrhi::BindingSetHandle globalBindingSet)
{
    if (!m_GTAOOutput || !m_computeAOVariant)
        return;

    ScopedPerfMarker perfMarker("GTAO", commandList);

    m_cachedDepthHierarchy = depthHierarchy;
    m_cachedNormals = fullResNormals;
    m_cachedMotionVectors = fullResMotionVectors;
    m_cachedPrevDepth = prevDepth;
    m_cachedGlobalBindingSet = globalBindingSet;
    m_cachedFrameIndex = frameIndex;

    GTAOConstants gtaoConst = {};
    gtaoConst.MatWorldToView = constants.view.matWorldToView;
    gtaoConst.MatClipToWorldNoOffset = constants.view.matClipToWorldNoOffset;
    gtaoConst.ProjScale = constants.view.matViewToClip.m_data[0]; // P[0][0]
    gtaoConst.Radius = kDefaultRadius;
    gtaoConst.FalloffEnd = kDefaultFalloffEnd;
    gtaoConst.TemporalAlpha = kDefaultTemporalAlpha;
    gtaoConst.HalfWidth = m_halfWidth;
    gtaoConst.HalfHeight = m_halfHeight;
    gtaoConst.FullWidth = m_fullWidth;
    gtaoConst.FullHeight = m_fullHeight;
    gtaoConst.FrameIndex = frameIndex;
    gtaoConst.ViewportSizeInvX = 1.0f / float(m_fullWidth);
    gtaoConst.ViewportSizeInvY = 1.0f / float(m_fullHeight);
    gtaoConst.Padding = 0;
    commandList->writeBuffer(m_constantBuffer, &gtaoConst, sizeof(gtaoConst));

    ComputeAO(commandList);
    SpatialFilter(commandList);
    TemporalAccumulate(commandList);
}

void GTAORenderer::ComputeAO(nvrhi::CommandListHandle commandList)
{
    auto pipeline = m_computeAOVariant ? m_computeAOVariant->GetPipeline() : nullptr;
    if (!pipeline)
        return;

    ScopedPerfMarker perfMarker("Trace", commandList);

    if (!m_computeAOBindingSet)
    {
        nvrhi::BindingSetDesc desc;
        desc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer),
            nvrhi::BindingSetItem::Texture_SRV(0, m_cachedNormals),
            nvrhi::BindingSetItem::Texture_UAV(0, m_GTAORaw),
            nvrhi::BindingSetItem::Sampler(0, m_pointClampSampler),
        };
        m_computeAOBindingSet = m_device->createBindingSet(desc, m_computeAOBindingLayout);
    }

    nvrhi::ComputeState state;
    state.pipeline = pipeline;
    state.bindings = { m_cachedGlobalBindingSet, m_computeAOBindingSet };
    commandList->setComputeState(state);

    SampleMiniConstants mini = {};
    commandList->setPushConstants(&mini, sizeof(mini));

    commandList->dispatch(DivCeil(m_halfWidth, kThreadGroupSize),
                          DivCeil(m_halfHeight, kThreadGroupSize), 1);
}

void GTAORenderer::SpatialFilter(nvrhi::CommandListHandle commandList)
{
    auto pipeline = m_spatialFilterVariant ? m_spatialFilterVariant->GetPipeline() : nullptr;
    if (!pipeline)
        return;

    ScopedPerfMarker perfMarker("Spatial Filter", commandList);

    if (!m_spatialFilterBindingSet)
    {
        nvrhi::BindingSetDesc desc;
        desc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer),
            nvrhi::BindingSetItem::PushConstants(1, sizeof(SampleMiniConstants)),
            nvrhi::BindingSetItem::Texture_SRV(0, m_GTAORaw),
            nvrhi::BindingSetItem::Texture_SRV(1, m_cachedDepthHierarchy),
            nvrhi::BindingSetItem::Texture_UAV(0, m_GTAOFiltered),
            nvrhi::BindingSetItem::Sampler(0, m_pointClampSampler),
        };
        m_spatialFilterBindingSet = m_device->createBindingSet(desc, m_spatialFilterBindingLayout);
    }

    nvrhi::ComputeState state;
    state.pipeline = pipeline;
    state.bindings = { m_spatialFilterBindingSet };
    commandList->setComputeState(state);

    SampleMiniConstants mini = {};
    commandList->setPushConstants(&mini, sizeof(mini));

    commandList->dispatch(DivCeil(m_halfWidth, kThreadGroupSize),
                          DivCeil(m_halfHeight, kThreadGroupSize), 1);
}

void GTAORenderer::TemporalAccumulate(nvrhi::CommandListHandle commandList)
{
    auto pipeline = m_temporalAccumVariant ? m_temporalAccumVariant->GetPipeline() : nullptr;
    if (!pipeline)
        return;

    ScopedPerfMarker perfMarker("Temporal Accumulate", commandList);

    // Ping-pong: read from current history, write to the other
    uint32_t readIdx = m_historyReadIdx;
    uint32_t writeIdx = 1 - readIdx;

    // Rebuild binding set each frame due to ping-pong swap
    nvrhi::BindingSetDesc desc;
    desc.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer),
        nvrhi::BindingSetItem::PushConstants(1, sizeof(SampleMiniConstants)),
        nvrhi::BindingSetItem::Texture_SRV(0, m_GTAOFiltered),
        nvrhi::BindingSetItem::Texture_SRV(1, m_GTAOHistory[readIdx]),
        nvrhi::BindingSetItem::Texture_SRV(2, m_cachedMotionVectors),
        nvrhi::BindingSetItem::Texture_SRV(3, m_cachedDepthHierarchy),
        nvrhi::BindingSetItem::Texture_SRV(4, m_cachedPrevDepth),
        nvrhi::BindingSetItem::Texture_UAV(0, m_GTAOOutput),
        nvrhi::BindingSetItem::Texture_UAV(1, m_GTAOHistory[writeIdx]),
        nvrhi::BindingSetItem::Sampler(0, m_pointClampSampler),
    };
    m_temporalAccumBindingSet = m_device->createBindingSet(desc, m_temporalAccumBindingLayout);

    nvrhi::ComputeState state;
    state.pipeline = pipeline;
    state.bindings = { m_temporalAccumBindingSet };
    commandList->setComputeState(state);

    SampleMiniConstants mini = {};
    commandList->setPushConstants(&mini, sizeof(mini));

    commandList->dispatch(DivCeil(m_fullWidth, kThreadGroupSize),
                          DivCeil(m_fullHeight, kThreadGroupSize), 1);

    // Swap history buffers for next frame
    m_historyReadIdx = writeIdx;
}
