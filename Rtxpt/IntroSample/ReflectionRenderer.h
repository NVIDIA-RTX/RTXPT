/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <nvrhi/nvrhi.h>
#include <memory>
#include <cstdint>
#include <vector>

class ComputePipelineBaker;
class ComputeShaderVariant;
class PTPipelineBaker;
class PTPipelineVariant;
class RenderTargets;
struct SampleConstants;

// Reflection renderer supporting both screen-space (SSR) and ray-traced modes.
// Encapsulates hierarchical Hi-Z ray-marched SSR, RT reflection dispatch, and
// roughness-aware blur mip-chain generation. Owns its render targets (SSRResult
// and SSRBlurMipChain) and SSR shader pipelines. The RT pipeline is created
// externally via PTPipelineBaker and passed in for dispatch.
class ReflectionRenderer
{
public:
    ReflectionRenderer(nvrhi::IDevice* device);
    ~ReflectionRenderer();

    void CreatePipelines(ComputePipelineBaker& computeBaker);
    void DestroyPipelines(ComputePipelineBaker& computeBaker);

    void CreateRTPipeline(PTPipelineBaker& pipelineBaker);
    void DestroyRTPipeline(PTPipelineBaker& pipelineBaker);

    void OnRenderTargetsRecreated(uint32_t renderWidth, uint32_t renderHeight);

    // Screen-space reflections (compute): Hi-Z ray march + blur.
    // envGGXMaxMip is the max sampleable mip index of the GGX-prefiltered environment cube,
    // forwarded to the SSR trace pass through SSRPushConstants::screenAndParams.z.
    void RenderSSR(nvrhi::CommandListHandle commandList,
                   const SampleConstants& constants,
                   uint32_t renderWidth, uint32_t renderHeight,
                   const RenderTargets& renderTargets,
                   nvrhi::TextureHandle depthHierarchy,
                   uint32_t depthHierarchyNumMips,
                   nvrhi::TextureHandle prefilteredEnvCube,
                   nvrhi::TextureHandle envIrradianceCube,
                   nvrhi::TextureHandle brdfLUT,
                   uint32_t envGGXMaxMip);

    // Ray-traced reflections: per-pixel reflection rays + blur.
    // envGGXMaxMip is forwarded to the shader through SampleMiniConstants::params.z.
    void RenderRTReflections(nvrhi::CommandListHandle commandList,
                             uint32_t renderWidth, uint32_t renderHeight,
                             nvrhi::BindingSetHandle globalBindingSet,
                             nvrhi::BindingSetHandle descriptorTable,
                             uint32_t envGGXMaxMip);

    bool IsRTReady() const;

    void InvalidateBindingSet() { m_ssrBindingSet = nullptr; }

    nvrhi::TextureHandle GetSSRResult() const { return m_ssrResult; }
    nvrhi::TextureHandle GetSSRBlurMipChain() const { return m_ssrBlurMipChain; }

    static constexpr uint32_t MaxMipLevels = 13;

private:
    void BlurSSR(nvrhi::CommandListHandle commandList,
                 uint32_t renderWidth, uint32_t renderHeight);

    static uint32_t GetNumMipLevels(uint32_t width, uint32_t height);

    nvrhi::DeviceHandle m_device;

    // Render targets (owned)
    nvrhi::TextureHandle m_ssrResult;        // Screen-sized RGBA16F (rgb=reflection, a=confidence)
    nvrhi::TextureHandle m_ssrBlurMipChain;  // Screen-sized RGBA16F with mip levels for roughness blur

    // SSR ray-march pass
    std::shared_ptr<ComputeShaderVariant> m_ssrVariant;
    nvrhi::BindingLayoutHandle m_ssrBindingLayout;
    nvrhi::BindingSetHandle    m_ssrBindingSet;

    // SSR blur mip-chain pass
    std::shared_ptr<ComputeShaderVariant> m_ssrBlurVariant;
    nvrhi::BindingLayoutHandle m_ssrBlurBindingLayout;
    // One binding set per destination mip (index 0 unused; src/dst pair is mip-1 -> mip).
    // Built lazily in BlurSSR; invalidated on resize and on DestroyPipelines.
    std::vector<nvrhi::BindingSetHandle> m_ssrBlurMipBindingSets;

    // Samplers
    nvrhi::SamplerHandle m_linearClampSampler;
    nvrhi::SamplerHandle m_nearestClampSampler;

    // RT reflections pipeline (alternative to SSR)
    std::shared_ptr<PTPipelineVariant> m_rtReflectionsPipeline;
};
