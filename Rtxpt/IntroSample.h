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
#include "GTAORenderer.h"
#include "DepthHierarchyRenderer.h"
#include <SampleCommon/SampleBaseApp.h>
#include <SampleCommon/PTPipelineBaker.h>
#include <donut/engine/View.h>
#include <donut/engine/BindingCache.h>
#include <donut/engine/CommonRenderPasses.h>

// Simplified renderer for introductory samples with improved reflections.
// Features:
// - Ray-traced local cubemap (updated over 3 frames)
// - GGX-filtered specular + SH-based diffuse irradiance
// - Hierarchical screen-space reflections
// - Split-sum IBL with BRDF LUT
// - Ground truth ambient occlusion (GTAO)
class IntroPathTracer : public Sample
{
public:

    IntroPathTracer(donut::app::DeviceManager& deviceManager, const CommandLineOptions& cmdLine);

    virtual void SampleRenderCode(nvrhi::IFramebuffer* framebuffer, nvrhi::CommandListHandle commandList, const SampleConstants& constants) override;
        
    void UpdateLocalCubemap(nvrhi::CommandListHandle commandList, const SampleConstants& constants);
    
    void ProcessLocalCubemap(nvrhi::CommandListHandle commandList);
    
    void RenderSSR(nvrhi::CommandListHandle commandList, const SampleConstants& constants);
    
    void BlurSSR(nvrhi::CommandListHandle commandList);

    void RasterDeferredLighting(nvrhi::CommandListHandle commandList);

    void CopyDepthForNextFrame(nvrhi::CommandListHandle commandList);

    void PopulateGBuffer(nvrhi::CommandListHandle commandList);

    void ReferencePathTracer(nvrhi::CommandListHandle commandList);

    virtual void CreateRTPipelines() override;

    void CreateReflectionPipelines();

    virtual void DestroyRTPipelines() override;

    virtual std::string GetMaterialSpecializationShader() const override;
    
    virtual void SceneLoaded() override;
    
    // Override to add reflection textures to the main binding set
    virtual void AddCustomBindings(nvrhi::BindingSetDesc& bindingSetDesc) override;

    virtual bool NeedsIntroPathTracerBuffers() override { return true; }
    virtual bool NeedsRasterPrecompute() override { return true; }

    virtual void OnRenderTargetsRecreated() override;

private:
    std::unique_ptr<GTAORenderer>               m_gtaoRenderer;
    std::unique_ptr<DepthHierarchyRenderer>     m_depthHierarchyRenderer;
    nvrhi::TextureHandle                        m_prevDepth;    // Full-res R32_FLOAT: previous frame depth
    bool                                        m_reflectionTexturesBound = false;
    std::shared_ptr<class PTPipelineVariant>    m_GBufferPipeline;
    std::shared_ptr<class PTPipelineVariant>    m_realTimePathTracer;

    // Deferred lighting - managed by ComputePipelineBaker for hot reload
    std::shared_ptr<ComputeShaderVariant>       m_deferredLightingVariant;
    
    // === Reflection System ===
    bool                                        m_reflectionSystemReady = false;
    uint32_t                                    m_frameIndex = 0;
    
    // Samplers
    nvrhi::SamplerHandle                        m_pointSampler;
    nvrhi::SamplerHandle                        m_linearSampler;
    nvrhi::SamplerHandle                        m_ssrSampler;
    nvrhi::SamplerHandle                        m_nearestSampler;
        
    // Allocates cubemap processing textures if needed
    void AllocateLocalCubemapTextures();
    
    // Reflection constants
    nvrhi::BufferHandle                         m_reflectionConstantBuffer;
    
    // Local cubemap pass
    nvrhi::TextureHandle                        m_ggxFilteredCubemap;   // GGX-filtered mip chain (for specular)
    nvrhi::TextureHandle                        m_IrradianceCube;       // Low-res irradiance cubemap (for diffuse)
    std::shared_ptr<class PTPipelineVariant>    m_localCubemapPipeline;
    
    // SSR pass - managed by ComputePipelineBaker for hot reload
    // Uses dedicated binding layout with custom push constants (no constant buffer)
    std::shared_ptr<ComputeShaderVariant>       m_ssrVariant;
    nvrhi::BindingLayoutHandle                  m_ssrBindingLayout;
    nvrhi::BindingSetHandle                     m_ssrBindingSet;
    
    // SSR blur pass - managed by ComputePipelineBaker for hot reload
    std::shared_ptr<ComputeShaderVariant>       m_ssrBlurVariant;
    nvrhi::BindingLayoutHandle                  m_ssrBlurBindingLayout;
    nvrhi::BufferHandle                         m_ssrBlurConstantBuffer;
};
