/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "ReflectionRenderer.h"
#include "SampleCommon/ComputePipelineBaker.h"
#include "SampleCommon/PTPipelineBaker.h"
#include "SampleCommon/RenderTargets.h"
#include "SampleCommon/SampleCommon.h"
#include "Shaders/SampleConstantBuffer.h"
#include <donut/core/math/math.h>

namespace
{
    static uint32_t DivCeil(uint32_t x, uint32_t y) { return (x + y - 1) / y; }
}

ReflectionRenderer::ReflectionRenderer(nvrhi::IDevice* device)
    : m_device(device)
{
}

ReflectionRenderer::~ReflectionRenderer()
{
}

uint32_t ReflectionRenderer::GetNumMipLevels(uint32_t width, uint32_t height)
{
    uint32_t maxDim = std::max(width, height);
    return 1 + static_cast<uint32_t>(std::log2(static_cast<float>(maxDim)));
}

void ReflectionRenderer::CreatePipelines(ComputePipelineBaker& computeBaker)
{
    // Samplers
    {
        nvrhi::SamplerDesc desc;
        desc.minFilter = true;
        desc.magFilter = true;
        desc.mipFilter = true;
        desc.addressU = nvrhi::SamplerAddressMode::Clamp;
        desc.addressV = nvrhi::SamplerAddressMode::Clamp;
        desc.addressW = nvrhi::SamplerAddressMode::Clamp;
        m_linearClampSampler = m_device->createSampler(desc);

        desc.minFilter = false;
        desc.magFilter = false;
        desc.mipFilter = false;
        m_nearestClampSampler = m_device->createSampler(desc);
    }

    // === SSR ray-march pass ===
    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::PushConstants(0, sizeof(SSRPushConstants)),
            nvrhi::BindingLayoutItem::Texture_SRV(0),   // t_SSR_DepthHierarchy
            nvrhi::BindingLayoutItem::Texture_SRV(1),   // t_SSR_Depth
            nvrhi::BindingLayoutItem::Texture_SRV(2),   // t_SSR_Normal
            nvrhi::BindingLayoutItem::Texture_SRV(3),   // t_SSR_RoughnessMetal
            nvrhi::BindingLayoutItem::Texture_SRV(4),   // t_SSR_BaseColor
            nvrhi::BindingLayoutItem::Texture_SRV(5),   // t_SSR_PrefilteredEnvCube
            nvrhi::BindingLayoutItem::Texture_SRV(6),   // t_SSR_EnvIrradianceCube
            nvrhi::BindingLayoutItem::Texture_SRV(7),   // t_SSR_BRDFLUT
            nvrhi::BindingLayoutItem::Texture_UAV(0),   // u_SSR_Result
            nvrhi::BindingLayoutItem::Sampler(0),       // s_SSR_LinearClamp
            nvrhi::BindingLayoutItem::Sampler(1),       // s_SSR_NearestClamp
        };
        m_ssrBindingLayout = m_device->createBindingLayout(layoutDesc);

        m_ssrVariant = computeBaker.CreateVariant(
            "IntroSample/SSRPasses.hlsl",
            "SSRCS",
            {},
            { m_ssrBindingLayout },
            "SSR");
    }

    // === SSR blur mip-chain pass ===
    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::PushConstants(0, sizeof(SSRBlurConstants)),
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::Texture_UAV(0),
            nvrhi::BindingLayoutItem::Sampler(0),
        };
        m_ssrBlurBindingLayout = m_device->createBindingLayout(layoutDesc);

        m_ssrBlurVariant = computeBaker.CreateVariant(
            "IntroSample/SSRPasses.hlsl",
            "SSRBlurCS",
            {},
            { m_ssrBlurBindingLayout },
            "SSRBlur");
    }
}

void ReflectionRenderer::DestroyPipelines(ComputePipelineBaker& computeBaker)
{
    if (m_ssrVariant)
        computeBaker.ReleaseVariant(m_ssrVariant);
    if (m_ssrBlurVariant)
        computeBaker.ReleaseVariant(m_ssrBlurVariant);

    m_ssrVariant = nullptr;
    m_ssrBlurVariant = nullptr;

    m_ssrBindingSet = nullptr;
    m_ssrBlurMipBindingSets.clear();
}

void ReflectionRenderer::CreateRTPipeline(PTPipelineBaker& pipelineBaker)
{
    m_rtReflectionsPipeline = pipelineBaker.CreateVariant("IntroSample/RTReflections.hlsl", {}, "RTRefl");
}

void ReflectionRenderer::DestroyRTPipeline(PTPipelineBaker& pipelineBaker)
{
    pipelineBaker.ReleaseVariant(m_rtReflectionsPipeline);
}

bool ReflectionRenderer::IsRTReady() const
{
    return m_rtReflectionsPipeline && m_rtReflectionsPipeline->GetShaderTable();
}

void ReflectionRenderer::OnRenderTargetsRecreated(uint32_t renderWidth, uint32_t renderHeight)
{
    // SSR result (screen-sized, with alpha for confidence)
    {
        nvrhi::TextureDesc desc;
        desc.width = renderWidth;
        desc.height = renderHeight;
        desc.format = nvrhi::Format::RGBA16_FLOAT;
        desc.dimension = nvrhi::TextureDimension::Texture2D;
        desc.debugName = "SSRResult";
        desc.isUAV = true;
        desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        desc.keepInitialState = true;
        m_ssrResult = m_device->createTexture(desc);

        // SSR blur mip chain
        desc.mipLevels = GetNumMipLevels(renderWidth, renderHeight);
        desc.debugName = "SSRBlurMipChain";
        m_ssrBlurMipChain = m_device->createTexture(desc);
    }

    m_ssrBindingSet = nullptr;
    m_ssrBlurMipBindingSets.clear();
}

void ReflectionRenderer::RenderSSR(nvrhi::CommandListHandle commandList,
                                   const SampleConstants& constants,
                                   uint32_t renderWidth, uint32_t renderHeight,
                                   const RenderTargets& renderTargets,
                                   nvrhi::TextureHandle depthHierarchy,
                                   uint32_t depthHierarchyNumMips,
                                   nvrhi::TextureHandle prefilteredEnvCube,
                                   nvrhi::TextureHandle envIrradianceCube,
                                   nvrhi::TextureHandle brdfLUT,
                                   uint32_t envGGXMaxMip)
{
    ScopedPerfMarker perfMarker("SSR", commandList);

    auto pipeline = m_ssrVariant ? m_ssrVariant->GetPipeline() : nullptr;
    if (!pipeline || !m_ssrBindingLayout)
        return;

    // Lazy creation of dedicated binding set
    if (!m_ssrBindingSet && depthHierarchy && m_ssrResult)
    {
        nvrhi::BindingSetDesc desc;
        desc.bindings = {
            nvrhi::BindingSetItem::PushConstants(0, sizeof(SSRPushConstants)),
            nvrhi::BindingSetItem::Texture_SRV(0, depthHierarchy),
            nvrhi::BindingSetItem::Texture_SRV(1, renderTargets.Depth),
            nvrhi::BindingSetItem::Texture_SRV(2, renderTargets.SpecNormal),
            nvrhi::BindingSetItem::Texture_SRV(3, renderTargets.RoughnessMetal),
            nvrhi::BindingSetItem::Texture_SRV(4, renderTargets.BaseColor),
            nvrhi::BindingSetItem::Texture_SRV(5, prefilteredEnvCube),
            nvrhi::BindingSetItem::Texture_SRV(6, envIrradianceCube),
            nvrhi::BindingSetItem::Texture_SRV(7, brdfLUT),
            nvrhi::BindingSetItem::Texture_UAV(0, m_ssrResult),
            nvrhi::BindingSetItem::Sampler(0, m_linearClampSampler),
            nvrhi::BindingSetItem::Sampler(1, m_nearestClampSampler),
        };
        m_ssrBindingSet = m_device->createBindingSet(desc, m_ssrBindingLayout);
    }

    if (!m_ssrBindingSet)
        return;

    // Build push constants from SampleConstants view/projection matrices
    const dm::float4x4& viewMatrix = constants.view.matWorldToView;
    const dm::float4x4& projMatrix = constants.view.matViewToClip;

    float maxRayDistance = 100.0f;
    uint32_t maxSteps = 512;
    float thickness = 0.1f;
    float jitter = 0;

    SSRPushConstants pushConst;
    pushConst.frameIndex = constants.ptConsts.frameIndex;
    pushConst.viewRow0_P00 = dm::float4(viewMatrix.m_data[0], viewMatrix.m_data[1], viewMatrix.m_data[2], projMatrix.m_data[0]);
    pushConst.viewRow1_P11 = dm::float4(viewMatrix.m_data[4], viewMatrix.m_data[5], viewMatrix.m_data[6], projMatrix.m_data[5]);
    pushConst.viewRow2_P22 = dm::float4(viewMatrix.m_data[8], viewMatrix.m_data[9], viewMatrix.m_data[10], projMatrix.m_data[10]);
    pushConst.projParams = dm::float4(
        1.0f / projMatrix.m_data[0],
        1.0f / projMatrix.m_data[5],
        projMatrix.m_data[14],
        0.0f
    );
    pushConst.screenAndParams = dm::float4(
        static_cast<float>(renderWidth),
        static_cast<float>(renderHeight),
        static_cast<float>(envGGXMaxMip),  // repurposed (was unused "farZ" slot)
        maxRayDistance
    );
    pushConst.ssrParams = dm::float4(
        *reinterpret_cast<float*>(&maxSteps),
        thickness,
        jitter,
        *reinterpret_cast<const float*>(&depthHierarchyNumMips)
    );

    nvrhi::ComputeState state;
    state.pipeline = pipeline;
    state.bindings = { m_ssrBindingSet };
    commandList->setComputeState(state);
    commandList->setPushConstants(&pushConst, sizeof(pushConst));

    commandList->dispatch(DivCeil(renderWidth, 8), DivCeil(renderHeight, 8), 1);

    BlurSSR(commandList, renderWidth, renderHeight);
}

void ReflectionRenderer::RenderRTReflections(nvrhi::CommandListHandle commandList,
                                             uint32_t renderWidth, uint32_t renderHeight,
                                             nvrhi::BindingSetHandle globalBindingSet,
                                             nvrhi::BindingSetHandle descriptorTable,
                                             uint32_t envGGXMaxMip)
{
    ScopedPerfMarker perfMarker("RTReflections", commandList);

    if (!IsRTReady() || !m_ssrResult)
        return;

    nvrhi::rt::State state;
    state.shaderTable = m_rtReflectionsPipeline->GetShaderTable();
    state.bindings = { globalBindingSet, descriptorTable };
    commandList->setRayTracingState(state);

    SampleMiniConstants miniConstants = {};
    miniConstants.params.z = envGGXMaxMip;
    commandList->setPushConstants(&miniConstants, sizeof(miniConstants));

    nvrhi::rt::DispatchRaysArguments args;
    args.width = renderWidth;
    args.height = renderHeight;
    args.depth = 1;
    commandList->dispatchRays(args);

    // No blur / denoise: RT reflections write fully-shaded specular directly into u_SSRResult, which the
    // deferred combine reads directly (like the indirect path tracer). Denoising is left to DLSS-RR.
}

void ReflectionRenderer::BlurSSR(nvrhi::CommandListHandle commandList,
                                 uint32_t renderWidth, uint32_t renderHeight)
{
    ScopedPerfMarker perfMarker("SSRBlur", commandList);

    auto pipeline = m_ssrBlurVariant ? m_ssrBlurVariant->GetPipeline() : nullptr;
    if (!pipeline)
        return;

    // Copy mip 0 from SSR result to blur chain
    commandList->copyTexture(m_ssrBlurMipChain, nvrhi::TextureSlice(0, 0),
                             m_ssrResult, nvrhi::TextureSlice(0, 0));

    uint32_t srcWidth = renderWidth;
    uint32_t srcHeight = renderHeight;

    uint32_t mipsForRes = GetNumMipLevels(renderWidth, renderHeight);
    uint32_t numMips = std::min(mipsForRes, MaxMipLevels);

    if (m_ssrBlurMipBindingSets.size() < numMips)
        m_ssrBlurMipBindingSets.resize(numMips);

    for (uint32_t mip = 1; mip < numMips; mip++)
    {
        uint32_t dstWidth = std::max(1u, srcWidth / 2);
        uint32_t dstHeight = std::max(1u, srcHeight / 2);

        SSRBlurConstants blurConst = {};
        blurConst.SrcSize.x = srcWidth;
        blurConst.SrcSize.y = srcHeight;
        blurConst.DstSize.x = dstWidth;
        blurConst.DstSize.y = dstHeight;
        blurConst.SrcMipLevel = mip - 1;
        blurConst.DstMipLevel = mip;

        nvrhi::BindingSetHandle& bindingSet = m_ssrBlurMipBindingSets[mip];
        if (!bindingSet)
        {
            nvrhi::BindingSetDesc desc;
            desc.bindings = {
                nvrhi::BindingSetItem::PushConstants(0, sizeof(SSRBlurConstants)),
                nvrhi::BindingSetItem::Texture_SRV(0, m_ssrBlurMipChain,
                    nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(mip - 1, 1, 0, 1)),
                nvrhi::BindingSetItem::Texture_UAV(0, m_ssrBlurMipChain,
                    nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(mip, 1, 0, 1)),
                nvrhi::BindingSetItem::Sampler(0, m_linearClampSampler),
            };
            bindingSet = m_device->createBindingSet(desc, m_ssrBlurBindingLayout);
        }

        nvrhi::ComputeState state;
        state.pipeline = pipeline;
        state.bindings = { bindingSet };
        commandList->setComputeState(state);
        commandList->setPushConstants(&blurConst, sizeof(blurConst));

        commandList->dispatch(DivCeil(dstWidth, 8), DivCeil(dstHeight, 8), 1);

        srcWidth = dstWidth;
        srcHeight = dstHeight;
    }
}
