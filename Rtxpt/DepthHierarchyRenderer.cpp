/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "DepthHierarchyRenderer.h"
#include "SampleCommon/ComputePipelineBaker.h"
#include "SampleCommon/RenderTargets.h"
#include "SampleCommon/SampleCommon.h"
#include <donut/core/math/math.h>
#include "Shaders/SampleConstantBuffer.h"

static uint32_t DivCeil(uint32_t x, uint32_t y) { return (x + y - 1) / y; }

DepthHierarchyRenderer::DepthHierarchyRenderer(nvrhi::IDevice* device)
    : m_device(device)
{
}

DepthHierarchyRenderer::~DepthHierarchyRenderer()
{
    assert(!m_variant); // Make sure destroy pipelines was called ahead of destruction
}

void DepthHierarchyRenderer::CreatePipelines(ComputePipelineBaker& computeBaker)
{
    // Max-reduction sampler for implicitly reducing level one in hardware
    {
        nvrhi::SamplerDesc samplerDesc;
        samplerDesc.minFilter = true;
        samplerDesc.magFilter = true;
        samplerDesc.mipFilter = true;
        samplerDesc.reductionType = nvrhi::SamplerReductionType::Maximum;
        m_maxSampler = m_device->createSampler(samplerDesc);
    }

    // Dedicated binding layout for depth hierarchy pass
    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings.reserve(3 + RenderTargets::SSRMaxMipLevels);

        layoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::PushConstants(1, sizeof(SampleMiniConstants)));
        layoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Sampler(0));
        // Source depth SRV (mip 0 is copied here, then sampled with max sampler for the first reduction)
        layoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_SRV(0));

        // Each remaining mip bound as a UAV
        for (uint32_t i = 1; i < RenderTargets::SSRMaxMipLevels; ++i)
        {
            layoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_UAV(i - 1));
        }

        m_bindingLayout = m_device->createBindingLayout(layoutDesc);
    }

    m_variant = computeBaker.CreateVariant(
        "IntroSample/SSRPasses.hlsl",
        "DepthHierarchyCS",
        {},
        { m_bindingLayout },
        "DepthHierarchy");
}

void DepthHierarchyRenderer::DestroyPipelines(ComputePipelineBaker& computeBaker)
{
    if (m_variant)
        computeBaker.ReleaseVariant(m_variant);
    m_variant = nullptr;
    m_bindingSet = nullptr;
}

void DepthHierarchyRenderer::OnRenderTargetsRecreated(uint32_t width, uint32_t height)
{
    uint32_t numMipForRes = RenderTargets::GetNumMipLevels(width, height);
    m_numMips = std::min(numMipForRes, RenderTargets::SSRMaxMipLevels);

    nvrhi::TextureDesc depthHierDesc;
    depthHierDesc.width = width;
    depthHierDesc.height = height;
    depthHierDesc.mipLevels = m_numMips;
    depthHierDesc.format = nvrhi::Format::R32_FLOAT;
    depthHierDesc.dimension = nvrhi::TextureDimension::Texture2D;
    depthHierDesc.debugName = "DepthHierarchy";
    depthHierDesc.isUAV = true;
    depthHierDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    depthHierDesc.keepInitialState = true;
    m_depthHierarchy = m_device->createTexture(depthHierDesc);

    // Invalidate binding set so it gets lazily recreated in Generate()
    m_bindingSet = nullptr;
}

void DepthHierarchyRenderer::Generate(nvrhi::CommandListHandle commandList,
                                       nvrhi::TextureHandle depthTexture,
                                       uint32_t width, uint32_t height)
{
    ScopedPerfMarker perfMarker("DepthHierarchy", commandList);

    auto pipeline = m_variant ? m_variant->GetPipeline() : nullptr;
    if (!pipeline || !m_depthHierarchy)
        return;

    // Lazy creation of binding set (depends on the texture existing)
    if (!m_bindingSet)
    {
        assert(m_bindingLayout);

        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings.reserve(3 + RenderTargets::SSRMaxMipLevels);
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Sampler(0, m_maxSampler));
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::PushConstants(1, sizeof(SampleMiniConstants)));
        bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_SRV(0, m_depthHierarchy));

        // Bind each mip (starting at mip 1) as a UAV
        for (uint32_t i = 1; i < RenderTargets::SSRMaxMipLevels; ++i)
        {
            uint32_t subresource = std::min(m_numMips - 1, i); // Pad last few mips if not allocated
            bindingSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_UAV(
                i - 1,
                m_depthHierarchy,
                nvrhi::Format::UNKNOWN,
                nvrhi::TextureSubresourceSet(subresource, 1, 0, 1)));
        }

        m_bindingSet = m_device->createBindingSet(bindingSetDesc, m_bindingLayout);
    }

    // Copy full-res depth into mip 0 of the hierarchy texture
    nvrhi::TextureSlice mip0Slice;
    commandList->copyTexture(m_depthHierarchy, mip0Slice, depthTexture, mip0Slice);

    nvrhi::ComputeState state;
    state.pipeline = pipeline;
    state.bindings = { m_bindingSet };
    commandList->setComputeState(state);

    SampleMiniConstants miniConstants = { uint4(width, height, 0, 0) };
    commandList->setPushConstants(&miniConstants, sizeof(miniConstants));

    // One group per 64x64 tile
    uint32_t dispatchX = DivCeil(width, 64);
    uint32_t dispatchY = DivCeil(height, 64);
    commandList->dispatch(dispatchX, dispatchY, 1);
}
