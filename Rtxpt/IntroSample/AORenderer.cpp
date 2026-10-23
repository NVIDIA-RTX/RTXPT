/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "AORenderer.h"
#include "SampleCommon/ComputePipelineBaker.h"
#include <donut/core/math/math.h>
#include "Shaders/SampleConstantBuffer.h"
#include "SampleCommon/SampleCommon.h"

namespace
{
    // Must match AOConstants in IntroSampleCommon.hlsli
    struct AOConstants
    {
        float4x4    MatWorldToView;
        float4x4    MatClipToWorldNoOffset;

        float2      ViewportInvSize;
        float       ProjScale;          // matViewToClip[0][0]
        float       Radius;

        float       FalloffEnd;
        float       TemporalAlpha;
        uint        HalfWidth;
        uint        HalfHeight;

        uint        FullWidth;
        uint        FullHeight;
        uint        FrameIndex;
        float       ZNear;

        float2      SubPixelOffset;
        uint        Denoise;
        uint        Pad;
    };

    constexpr float    kDefaultFalloffEnd = 2.0f;
    constexpr float    kDefaultTemporalAlpha = 0.9f;
}

static uint32_t DivCeil(uint32_t x, uint32_t y) { return (x + y - 1) / y; }

AORenderer::AORenderer(nvrhi::IDevice* device)
    : m_device(device)
{
}

AORenderer::~AORenderer()
{
}

void AORenderer::CreatePipelines(nvrhi::BindingLayoutHandle globalBindingLayout,
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
        desc.byteSize = sizeof(AOConstants);
        desc.structStride = 0;
        desc.debugName = "AOConstants";
        desc.isConstantBuffer = true;
        desc.initialState = nvrhi::ResourceStates::ConstantBuffer;
        desc.keepInitialState = true;
        m_constantBuffer = m_device->createBuffer(desc);
    }

    // === ComputeAO binding layout (space1) ===
    // b0: AOConstants
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
            "IntroSample/GTAOPass.hlsl", "main", {},
            { globalBindingLayout, m_computeAOBindingLayout },
            "GTAOCompute");

        m_traceAOVariant = computeBaker.CreateVariant(
            "IntroSample/RTAOTrace.hlsl", "main", {},
            { globalBindingLayout, m_computeAOBindingLayout },
            "RTAOTrace");
    }

    // === SpatialFilter binding layout ===
    // b0: AOConstants, b1: push constants
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
            "IntroSample/AOSpatialPass.hlsl", "main", {},
            { m_spatialFilterBindingLayout },
            "AOSpatialFilter");
    }

    // === TemporalAccumulate binding layout ===
    // b0: AOConstants, b1: push constants
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
            "IntroSample/AOTemporalUpscale.hlsl", "main",{},
            { m_temporalAccumBindingLayout },
            "AOTemporal");
    }
}

void AORenderer::DestroyPipelines(ComputePipelineBaker& computeBaker)
{
    computeBaker.ReleaseVariant(m_computeAOVariant);
    computeBaker.ReleaseVariant(m_traceAOVariant);
    computeBaker.ReleaseVariant(m_spatialFilterVariant);
    computeBaker.ReleaseVariant(m_temporalAccumVariant);

    // CreatePipelines() rebuilds the binding layouts, so the cached binding sets (which were created
    // against the now-destroyed layouts) must be invalidated too. Otherwise, after a scene reload the
    // AO passes would dispatch freshly-built pipelines with stale binding sets, corrupting GTAO/RTAO.
    // (Mirrors DepthHierarchyRenderer::DestroyPipelines.)
    m_computeAOBindingSet = nullptr;
    m_spatialFilterBindingSet = nullptr;
    m_temporalAccumBindingSet[0] = nullptr;
    m_temporalAccumBindingSet[1] = nullptr;
}

void AORenderer::OnRenderTargetsRecreated(uint32_t renderWidth, uint32_t renderHeight)
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

    // Invalidate binding sets so they get recreated with the new textures
    m_computeAOBindingSet = nullptr;
    m_spatialFilterBindingSet = nullptr;
    m_temporalAccumBindingSet[0] = nullptr;
    m_temporalAccumBindingSet[1] = nullptr;
}

void AORenderer::SetupConstants(nvrhi::CommandListHandle commandList,
                                const SampleConstants& constants,
                                uint32_t frameIndex,
                                float radius,
                                const nvrhi::TextureHandle& depthHierarchy,
                                const nvrhi::TextureHandle& fullResNormals,
                                const nvrhi::TextureHandle& fullResMotionVectors,
                                const nvrhi::TextureHandle& prevDepth,
                                const nvrhi::TextureHandle& gbufferMaterialInfo,
                                nvrhi::BindingSetHandle globalBindingSet,
                                bool denoise)
{
    m_cachedDepthHierarchy = depthHierarchy;
    m_cachedNormals = fullResNormals;
    m_cachedMotionVectors = fullResMotionVectors;
    m_cachedPrevDepth = prevDepth;
    m_cachedMaterialInfo = gbufferMaterialInfo;
    m_cachedGlobalBindingSet = globalBindingSet;

    AOConstants aoConst = {};
    aoConst.MatWorldToView = constants.view.matWorldToView;
    aoConst.MatClipToWorldNoOffset = constants.view.matClipToWorldNoOffset;
    aoConst.ProjScale = constants.view.matViewToClip.m_data[0]; // P[0][0]
    aoConst.Radius = radius;
    aoConst.FalloffEnd = kDefaultFalloffEnd;
    aoConst.TemporalAlpha = kDefaultTemporalAlpha;
    aoConst.HalfWidth = m_halfWidth;
    aoConst.HalfHeight = m_halfHeight;
    aoConst.FullWidth = m_fullWidth;
    aoConst.FullHeight = m_fullHeight;
    aoConst.FrameIndex = frameIndex;
    aoConst.ViewportInvSize = float2(1.0f / float(m_fullWidth), 1.0f / float(m_fullHeight));
    aoConst.ZNear = constants.view.matViewToClip.m_data[14]; // P[3][2] = zNear for reverse-Z infinite projection
    aoConst.SubPixelOffset = constants.view.pixelOffset;
    aoConst.Denoise = denoise;
    commandList->writeBuffer(m_constantBuffer, &aoConst, sizeof(aoConst));
}

void AORenderer::RenderGTAO(nvrhi::CommandListHandle commandList,
                              const SampleConstants& constants,
                              uint32_t frameIndex,
                              float radius,
                              nvrhi::TextureHandle depthHierarchy,
                              nvrhi::TextureHandle fullResNormals,
                              nvrhi::TextureHandle fullResMotionVectors,
                              nvrhi::TextureHandle prevDepth,
                              nvrhi::TextureHandle gbufferMaterialInfo,
                              nvrhi::BindingSetHandle globalBindingSet)
{
    if (!m_computeAOVariant)
        return;

    ScopedPerfMarker perfMarker("GTAO", commandList);

    SetupConstants(commandList, constants, frameIndex, radius, depthHierarchy, fullResNormals,
                   fullResMotionVectors, prevDepth, gbufferMaterialInfo, globalBindingSet, true);

    ComputeGTAO(commandList);
    SpatialFilter(commandList);
    TemporalAccumulate(commandList);
}

void AORenderer::RenderRTAO(nvrhi::CommandListHandle commandList,
                              const SampleConstants& constants,
                              uint32_t frameIndex,
                              float radius,
                              nvrhi::TextureHandle depthHierarchy,
                              nvrhi::TextureHandle fullResNormals,
                              nvrhi::TextureHandle fullResMotionVectors,
                              nvrhi::TextureHandle prevDepth,
                              nvrhi::TextureHandle gbufferMaterialInfo,
                              nvrhi::BindingSetHandle globalBindingSet,
                              bool denoise)
{
    if (!m_traceAOVariant)
        return;

    ScopedPerfMarker perfMarker("RTAO", commandList);

    SetupConstants(commandList, constants, frameIndex, radius, depthHierarchy, fullResNormals,
                   fullResMotionVectors, prevDepth, gbufferMaterialInfo, globalBindingSet, denoise);

    TraceAO(commandList);
    SpatialFilter(commandList);
    TemporalAccumulate(commandList);
}

void AORenderer::ComputeGTAO(nvrhi::CommandListHandle commandList)
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

    commandList->dispatch(DivCeil(m_halfWidth, 8),
                          DivCeil(m_halfHeight, 8), 1);
}

void AORenderer::TraceAO(nvrhi::CommandListHandle commandList)
{
    auto pipeline = m_traceAOVariant ? m_traceAOVariant->GetPipeline() : nullptr;
    if (!pipeline)
        return;

    ScopedPerfMarker perfMarker("Trace RT", commandList);

    // Reuse the same binding set as GTAO (same layout: constants, normals, raw AO, sampler)
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

    commandList->dispatch(DivCeil(m_halfWidth, 8),
                          DivCeil(m_halfHeight, 8), 1);
}

void AORenderer::SpatialFilter(nvrhi::CommandListHandle commandList)
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

    commandList->dispatch(DivCeil(m_halfWidth, 8),
                          DivCeil(m_halfHeight, 8), 1);
}

void AORenderer::TemporalAccumulate(nvrhi::CommandListHandle commandList)
{
    auto pipeline = m_temporalAccumVariant ? m_temporalAccumVariant->GetPipeline() : nullptr;
    if (!pipeline)
        return;

    if (!m_temporalAccumBindingSet[0])
    {
        nvrhi::BindingSetDesc desc;

        // Build Two Temporal accumulation sets for ping-pong
        desc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer),
            nvrhi::BindingSetItem::PushConstants(1, sizeof(SampleMiniConstants)),
            nvrhi::BindingSetItem::Texture_SRV(0, m_GTAOFiltered),
            nvrhi::BindingSetItem::Texture_SRV(1, m_GTAOHistory[0]),
            nvrhi::BindingSetItem::Texture_SRV(2, m_cachedMotionVectors),
            nvrhi::BindingSetItem::Texture_SRV(3, m_cachedDepthHierarchy),
            nvrhi::BindingSetItem::Texture_SRV(4, m_cachedPrevDepth),
            nvrhi::BindingSetItem::Texture_UAV(0, m_cachedMaterialInfo),
            nvrhi::BindingSetItem::Texture_UAV(1, m_GTAOHistory[1]),
            nvrhi::BindingSetItem::Sampler(0, m_pointClampSampler),
        };
        m_temporalAccumBindingSet[0] = m_device->createBindingSet(desc, m_temporalAccumBindingLayout);

        // Swap read and write buffers
        desc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer),
            nvrhi::BindingSetItem::PushConstants(1, sizeof(SampleMiniConstants)),
            nvrhi::BindingSetItem::Texture_SRV(0, m_GTAOFiltered),
            nvrhi::BindingSetItem::Texture_SRV(1, m_GTAOHistory[1]),
            nvrhi::BindingSetItem::Texture_SRV(2, m_cachedMotionVectors),
            nvrhi::BindingSetItem::Texture_SRV(3, m_cachedDepthHierarchy),
            nvrhi::BindingSetItem::Texture_SRV(4, m_cachedPrevDepth),
            nvrhi::BindingSetItem::Texture_UAV(0, m_cachedMaterialInfo),
            nvrhi::BindingSetItem::Texture_UAV(1, m_GTAOHistory[0]),
            nvrhi::BindingSetItem::Sampler(0, m_pointClampSampler),
        };
        m_temporalAccumBindingSet[1] = m_device->createBindingSet(desc, m_temporalAccumBindingLayout);
    }

    ScopedPerfMarker perfMarker("Temporal Accumulate", commandList);

    nvrhi::ComputeState state;
    state.pipeline = pipeline;
    state.bindings = { m_temporalAccumBindingSet[m_bindingSetIdx]};
    commandList->setComputeState(state);

    SampleMiniConstants mini = {};
    commandList->setPushConstants(&mini, sizeof(mini));

    commandList->dispatch(DivCeil(m_fullWidth, 8),
                          DivCeil(m_fullHeight, 8), 1);

    // Ping-pong: read from current history, write to the other
    m_bindingSetIdx = 1 - m_bindingSetIdx;
}
