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
#include <SampleCommon/SampleBaseApp.h>
#include <SampleCommon/PTPipelineBaker.h>
#include <SampleCommon/ComputePipelineBaker.h>
#include <donut/engine/View.h>
#include <donut/engine/BindingCache.h>
#include <donut/engine/CommonRenderPasses.h>

// Include shader constant structures
#include "Lighting/Distant/EnvMapBaker.h"

#include "IntroSample.h"
#include "DepthHierarchyRenderer.h"

namespace
{
    // Constants for reflection system
    struct ReflectionConstants
    {
        float LocalCubeWeight;
        float LocalCubeMaxMip;
        float SSRMaxMip;
        float Padding;
    };

    // SSR Push Constants (96 bytes) - self-contained, no constant buffer needed
    // Must match SSRPushConstants in SSRPasses.hlsl
    struct SSRPushConstants
    {
        dm::float4 viewRow0_P00;      // View row 0 + proj[0][0]
        dm::float4 viewRow1_P11;      // View row 1 + proj[1][1]
        dm::float4 viewRow2_P22;      // View row 2 + proj[2][2]
        dm::float4 projParams;        // rcpP00, rcpP11, P32, nearZ
        dm::float4 screenAndParams;   // width, height, farZ, maxRayDistance
        dm::float4 ssrParams;         // maxSteps (asuint), thickness, jitter, maxMipLevel (asuint)
    };

    struct SSRBlurConstants
    {
        dm::uint2   SrcSize;
        dm::uint2   DstSize;
        uint32_t    SrcMipLevel;
        uint32_t    DstMipLevel;
        uint32_t    Padding0;
        uint32_t    Padding1;
    };

} // Anonymous namespace

IntroPathTracer::IntroPathTracer(donut::app::DeviceManager& deviceManager,
    const CommandLineOptions& cmdLine)
    : Sample(deviceManager, cmdLine)
{
    m_ui.RealtimeAA = 0; // No TAA or DLSS by default
    m_ui.EnableBloom = false;
    m_ui.RenderWhenOutOfFocus = true;

    m_depthHierarchyRenderer = std::make_unique<DepthHierarchyRenderer>(GetDevice());
    m_gtaoRenderer = std::make_unique<GTAORenderer>(GetDevice());
}

void IntroPathTracer::SampleRenderCode(nvrhi::IFramebuffer* framebuffer, nvrhi::CommandListHandle commandList, const SampleConstants& constants)
{
    ScopedPerfMarker perfMarker("Intro-Sample", commandList);
        
    nvrhi::rt::State state;

    nvrhi::rt::DispatchRaysArguments args;
    nvrhi::Viewport viewport = GetView().GetViewport();
    uint32_t width = (uint32_t)(viewport.maxX - viewport.minX);
    uint32_t height = (uint32_t)(viewport.maxY - viewport.minY);
    args.width = width;
    args.height = height;

    if (m_ui.RealtimeMode)
    {
        PopulateGBuffer(commandList);

        // Depth hierarchy (needed by GTAO and SSR, must run before both)
        if (m_depthHierarchyRenderer)
        {
            ScopedPerfMarker perfMarker("Depth hierarchy", commandList);
            m_depthHierarchyRenderer->Generate(commandList, m_renderTargets->Depth, width, height);
        }

        // Screen-space ambient occlusion
        if (m_gtaoRenderer)
        {
            m_gtaoRenderer->Render(commandList, constants, m_frameIndex,
                m_depthHierarchyRenderer->m_depthHierarchy, m_renderTargets->SpecNormal,
                m_renderTargets->ScreenMotionVectors,
                m_prevDepth, m_bindingSet);
        }

        // Reflections
        if (m_reflectionSystemReady && m_localCubemapPipeline)
        {
            ScopedPerfMarker perfMarker("SSR", commandList);

            UpdateLocalCubemap(commandList, constants);
            RenderSSR(commandList, constants);
            BlurSSR(commandList);
        }

        RasterDeferredLighting(commandList);

        CopyDepthForNextFrame(commandList);

        m_frameIndex++;
    }
    else
    {
        ReferencePathTracer(commandList);
    }

    commandList->setBufferState(m_renderTargets->StablePlanesBuffer, nvrhi::ResourceStates::UnorderedAccess);
}
    
void IntroPathTracer::UpdateLocalCubemap(nvrhi::CommandListHandle commandList, const SampleConstants& constants)
{
    ScopedPerfMarker perfMarker("UpdateLocalCubemap", commandList);
    
    if (!m_localCubemapPipeline || !m_localCubemapPipeline->GetShaderTable())
        return;
        
    // Calculate which faces to update this frame (2 faces per frame, cycling through 6)
    uint32_t baseFace = (m_frameIndex * 2) % 6;
    
    // Set up RT state
    nvrhi::rt::State state;
    state.shaderTable = m_localCubemapPipeline->GetShaderTable();
    state.bindings = { m_bindingSet, m_DescriptorTable->GetDescriptorTable() };
    commandList->setRayTracingState(state);
    
    // Pass face indices via push constants (g_MiniConst.x = face0, g_MiniConst.y = face1)
    SampleMiniConstants miniConstants = { uint4(baseFace, 0, 0, 0) };
    commandList->setPushConstants(&miniConstants, sizeof(miniConstants));
    
    // Dispatch rays: (cubemapSize, cubemapSize, 2 faces)
    nvrhi::rt::DispatchRaysArguments args;
    args.width = RenderTargets::LocalCubemapSize;
    args.height = RenderTargets::LocalCubemapSize;
    args.depth = 2;  // 2 faces per dispatch
    commandList->dispatchRays(args);

    // Process cubemap
    ProcessLocalCubemap(commandList);
}
    
void IntroPathTracer::AllocateLocalCubemapTextures()
{
    if (m_ggxFilteredCubemap)
        return; // Already allocated
    
    uint cubemapSize = RenderTargets::LocalCubemapSize;
    uint mipLevels = uint(std::log2((float)cubemapSize) + 0.5f);
    
    // GGX filtered cubemap with mip chain
    {
        nvrhi::TextureDesc desc;
        desc.width = cubemapSize;
        desc.height = cubemapSize;
        desc.depth = 1;
        desc.arraySize = 6;
        desc.mipLevels = mipLevels;
        desc.format = nvrhi::Format::RGBA16_FLOAT;
        desc.dimension = nvrhi::TextureDimension::TextureCube;
        desc.debugName = "LocalCube_GGXFiltered";
        desc.isUAV = true;
        desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        desc.keepInitialState = true;
        
        m_ggxFilteredCubemap = GetDevice()->createTexture(desc);
    }
    
    // Diffuse irradiance cubemap
    {
        nvrhi::TextureDesc desc;
        desc.width = EnvMapBaker::c_IrradianceCubeSize;
        desc.height = EnvMapBaker::c_IrradianceCubeSize;
        desc.depth = 1;
        desc.arraySize = 6;
        desc.mipLevels = 1;
        desc.format = nvrhi::Format::RGBA16_FLOAT;
        desc.dimension = nvrhi::TextureDimension::TextureCube;
        desc.debugName = "LocalCube_DiffuseIrradiance";
        desc.isUAV = true;
        desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        desc.keepInitialState = true;
        
        m_IrradianceCube = GetDevice()->createTexture(desc);
    }

    // Invalidate SSR binding set so it gets recreated with the real cubemap textures
    m_ssrBindingSet = nullptr;
}

void IntroPathTracer::ProcessLocalCubemap(nvrhi::CommandListHandle commandList)
{
    ScopedPerfMarker perfMarker("ProcessLocalCubemap", commandList);
            
    EnvMapBaker::CubemapProcessingOptions opts;
    opts.generateMips = true;
    opts.ggxPrefilter = true;
    opts.projectToSH = true;
    
    EnvMapBaker::CubemapProcessingResults dstTextures;
    dstTextures.filteredCubemap = m_ggxFilteredCubemap;
    dstTextures.diffuseIrradianceCube = m_IrradianceCube;

    GetEnvMapBaker()->ProcessCubemap(
        commandList.Get(), GetBindingCache(),
        m_renderTargets->LocalCubemap,
        opts, dstTextures);
}
    
void IntroPathTracer::RenderSSR(nvrhi::CommandListHandle commandList, const SampleConstants& constants)
{
    ScopedPerfMarker perfMarker("SSR", commandList);
        
    // Get pipeline from hot-reloadable variant
    auto pipeline = m_ssrVariant ? m_ssrVariant->GetPipeline() : nullptr;
    if (!pipeline || !m_ssrBindingLayout)
        return; // Pipeline not yet compiled
        
    // Lazy creation of dedicated binding set (depends on render targets)
    auto depthHierarchy = m_depthHierarchyRenderer->m_depthHierarchy;
    if (!m_ssrBindingSet && depthHierarchy && m_renderTargets->SSRResult)
    {
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::PushConstants(0, sizeof(SSRPushConstants)),
            nvrhi::BindingSetItem::Texture_SRV(0, depthHierarchy),  // t_SSR_DepthHierarchy
            nvrhi::BindingSetItem::Texture_SRV(1, m_renderTargets->Depth),           // t_SSR_Depth
            nvrhi::BindingSetItem::Texture_SRV(2, m_renderTargets->SpecNormal),      // t_SSR_Normal
            nvrhi::BindingSetItem::Texture_SRV(3, m_renderTargets->RoughnessMetal),  // t_SSR_RoughnessMetal
            nvrhi::BindingSetItem::Texture_SRV(4, m_renderTargets->BaseColor),       // t_SSR_BaseColor
            nvrhi::BindingSetItem::Texture_SRV(5, m_ggxFilteredCubemap ? m_ggxFilteredCubemap : GetCommonPasses()->m_BlackCubeMapArray),  // t_SSR_CubemapGGX
            nvrhi::BindingSetItem::Texture_SRV(6, m_IrradianceCube ? m_IrradianceCube : GetCommonPasses()->m_BlackCubeMapArray),          // t_SSR_IrradianceCube
            nvrhi::BindingSetItem::Texture_SRV(7, GetEnvMapBaker()->GetBRDFLUT()),   // t_SSR_BRDFLUT
            nvrhi::BindingSetItem::Texture_UAV(0, m_renderTargets->SSRResult),       // u_SSR_Result
            nvrhi::BindingSetItem::Sampler(0, m_ssrSampler),                         // s_SSR_LinearClamp
            nvrhi::BindingSetItem::Sampler(1, m_nearestSampler),                     // s_SSR_NearestClamp
        };
        m_ssrBindingSet = GetDevice()->createBindingSet(bindingSetDesc, m_ssrBindingLayout);
    }
    
    if (!m_ssrBindingSet)
        return;

    nvrhi::Viewport viewport = GetView().GetViewport();
    uint32_t width = (uint32_t)(viewport.maxX - viewport.minX);
    uint32_t height = (uint32_t)(viewport.maxY - viewport.minY);
        
    // Extract view and projection matrices
    const auto& view = GetView();
    dm::float4x4 viewMatrix = dm::affineToHomogeneous(view.GetViewMatrix());
    dm::float4x4 projMatrix = view.GetProjectionMatrix(false);
    
    // SSR parameters
    float maxRayDistance = 100.0f;
    uint32_t maxSteps = 512;
    float thickness = 0.1f;
    float jitter = 0; // TODO: Use DLSS/TAA jitter
    uint32_t ssrNumMips = m_depthHierarchyRenderer ? m_depthHierarchyRenderer->GetNumMips() : 1;
        
    // Pack push constants (96 bytes)
    // See SSRPushConstants in SSRPasses.hlsl for layout
    SSRPushConstants pushConst;
    
    // View matrix rows (xyz) + projection diagonal (w)
    pushConst.viewRow0_P00 = dm::float4(viewMatrix.m_data[0], viewMatrix.m_data[1], viewMatrix.m_data[2], projMatrix.m_data[0]);
    pushConst.viewRow1_P11 = dm::float4(viewMatrix.m_data[4], viewMatrix.m_data[5], viewMatrix.m_data[6], projMatrix.m_data[5]);
    pushConst.viewRow2_P22 = dm::float4(viewMatrix.m_data[8], viewMatrix.m_data[9], viewMatrix.m_data[10], projMatrix.m_data[10]);
    
    // Projection params: rcpP00, rcpP11, P32, nearZ
    pushConst.projParams = dm::float4(
        1.0f / projMatrix.m_data[0],   // rcpP00
        1.0f / projMatrix.m_data[5],   // rcpP11
        projMatrix.m_data[14],          // P32 (assuming row-major: proj[3][2])
        0.0f
    );
    
    // Screen size and params: width, height, padding, maxRayDistance
    pushConst.screenAndParams = dm::float4(
        static_cast<float>(width),
        static_cast<float>(height),
        0,
        maxRayDistance
    );
    
    // SSR params: maxSteps (as float bits), thickness, jitter, maxMipLevel (as float bits)
    pushConst.ssrParams = dm::float4(
        *reinterpret_cast<float*>(&maxSteps),
        thickness,
        jitter,
        *reinterpret_cast<const float*>(&ssrNumMips)
    );
        
    // Set up compute state using dedicated SSR binding set
    nvrhi::ComputeState state;
    state.pipeline = pipeline;
    state.bindings = { m_ssrBindingSet };
    commandList->setComputeState(state);
    
    commandList->setPushConstants(&pushConst, sizeof(pushConst));
        
    uint32_t dispatchX = div_ceil(width, 8);
    uint32_t dispatchY = div_ceil(height, 8);
    commandList->dispatch(dispatchX, dispatchY, 1);
}
    
void IntroPathTracer::BlurSSR(nvrhi::CommandListHandle commandList)
{
    ScopedPerfMarker perfMarker("SSRBlur", commandList);
        
    // Get pipeline from hot-reloadable variant
    auto pipeline = m_ssrBlurVariant ? m_ssrBlurVariant->GetPipeline() : nullptr;
    if (!pipeline)
        return; // Pipeline not yet compiled or compilation failed

    nvrhi::Viewport viewport = GetView().GetViewport();
    uint32_t width = (uint32_t)(viewport.maxX - viewport.minX);
    uint32_t height = (uint32_t)(viewport.maxY - viewport.minY);
        
    // Copy mip 0 from SSR result to blur chain
    commandList->copyTexture(m_renderTargets->SSRBlurMipChain, nvrhi::TextureSlice(0, 0),
        m_renderTargets->SSRResult, nvrhi::TextureSlice(0, 0));
        
    uint32_t srcWidth = width;
    uint32_t srcHeight = height;
        
    // Generate blur mip chain
    uint32_t mipsForRes = RenderTargets::GetNumMipLevels(width, height);
    uint32_t numMips = std::min(mipsForRes, RenderTargets::SSRMaxMipLevels);
    for (uint32_t mip = 1; mip < numMips; mip++)
    {
        uint32_t dstWidth = std::max(1u, srcWidth / 2);
        uint32_t dstHeight = std::max(1u, srcHeight / 2);
            
        // Update constants
        SSRBlurConstants blurConst = {};
        blurConst.SrcSize.x = srcWidth;
        blurConst.SrcSize.y = srcHeight;
        blurConst.DstSize.x = dstWidth;
        blurConst.DstSize.y = dstHeight;
        blurConst.SrcMipLevel = mip - 1;
        blurConst.DstMipLevel = mip;
            
        commandList->writeBuffer(m_ssrBlurConstantBuffer, &blurConst, sizeof(blurConst));
            
        // Create binding set for this mip level
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_ssrBlurConstantBuffer),
            nvrhi::BindingSetItem::PushConstants(1, sizeof(SampleMiniConstants)),  // g_MiniConst (from global bindings include)
            nvrhi::BindingSetItem::Texture_SRV(0, m_renderTargets->SSRBlurMipChain,
                nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(mip - 1, 1, 0, 1)),
            nvrhi::BindingSetItem::Texture_UAV(0, m_renderTargets->SSRBlurMipChain,
                nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(mip, 1, 0, 1)),
            nvrhi::BindingSetItem::Sampler(0, m_linearSampler)
        };
        nvrhi::BindingSetHandle bindingSet = GetDevice()->createBindingSet(bindingSetDesc, m_ssrBlurBindingLayout);
            
        nvrhi::ComputeState state;
        state.pipeline = pipeline;
        state.bindings = { bindingSet };
        commandList->setComputeState(state);
        
        // Push constants required because shader includes global bindings that declare g_MiniConst
        SampleMiniConstants miniConstants = { uint4(0, 0, 0, 0) };
        commandList->setPushConstants(&miniConstants, sizeof(miniConstants));
            
        uint32_t dispatchX = div_ceil(dstWidth, 8);
        uint32_t dispatchY = div_ceil(dstHeight, 8);
        commandList->dispatch(dispatchX, dispatchY, 1);
                    
        srcWidth = dstWidth;
        srcHeight = dstHeight;
    }
}

void IntroPathTracer::RasterDeferredLighting(nvrhi::CommandListHandle commandList)
{
    ScopedPerfMarker perfMarker("Deferred Lighting", commandList);
            
    if (m_reflectionSystemReady && m_ggxFilteredCubemap)
    {            
        // Update reflection constants
        ReflectionConstants reflConst = {};
        reflConst.LocalCubeMaxMip = 7.0f;  // Based on 256 cubemap (log2(256) = 8, minus 1)
        uint32_t numMips = m_depthHierarchyRenderer ? m_depthHierarchyRenderer->GetNumMips() : 1;
        reflConst.SSRMaxMip = float(numMips - 1);
        commandList->writeBuffer(m_reflectionConstantBuffer, &reflConst, sizeof(reflConst));
    }

    // Get pipeline from hot-reloadable variant
    auto pipeline = m_deferredLightingVariant ? m_deferredLightingVariant->GetPipeline() : nullptr;
    if (!pipeline)
        return; // Pipeline not yet compiled or compilation failed

    nvrhi::ComputeState state;
    // Both regular and improved deferred lighting use the main binding set
    // The reflection textures (t80-t83, b3) are added via AddCustomBindings
    state.bindings = { m_bindingSet, m_DescriptorTable->GetDescriptorTable() };
    state.pipeline = pipeline;
    commandList->setComputeState(state);

    nvrhi::Viewport viewport = GetView().GetViewport();
    uint32_t width = (uint32_t)(viewport.maxX - viewport.minX);
    uint32_t height = (uint32_t)(viewport.maxY - viewport.minY);
    const dm::uint2 dispatchSize = { (width + NUM_COMPUTE_THREADS_PER_DIM - 1) / NUM_COMPUTE_THREADS_PER_DIM, (height + NUM_COMPUTE_THREADS_PER_DIM - 1) / NUM_COMPUTE_THREADS_PER_DIM };

    // default miniConstants
    SampleMiniConstants miniConstants = { uint4(0, 0, 0, 0) };
    commandList->setPushConstants(&miniConstants, sizeof(miniConstants));
    commandList->dispatch(dispatchSize.x, dispatchSize.y);
}

void IntroPathTracer::CopyDepthForNextFrame(nvrhi::CommandListHandle commandList)
{
    if (!m_prevDepth || !m_renderTargets->Depth)
        return;

    ScopedPerfMarker perfMarker("Depth Copy", commandList);
    commandList->copyTexture(m_prevDepth, nvrhi::TextureSlice(), m_renderTargets->Depth, nvrhi::TextureSlice());
}

void IntroPathTracer::PopulateGBuffer(nvrhi::CommandListHandle commandList)
{
    ScopedPerfMarker perfMarker("Populate GBuffer", commandList);

    nvrhi::rt::State state;

    nvrhi::rt::DispatchRaysArguments args;
    nvrhi::Viewport viewport = GetView().GetViewport();
    uint32_t width = (uint32_t)(viewport.maxX - viewport.minX);
    uint32_t height = (uint32_t)(viewport.maxY - viewport.minY);
    args.width = width;
    args.height = height;

    // default miniConstants
    SampleMiniConstants miniConstants = { uint4(0, 0, 0, 0) };

    state.shaderTable = m_GBufferPipeline->GetShaderTable();
    state.bindings = { m_bindingSet, m_DescriptorTable->GetDescriptorTable() };
    commandList->setRayTracingState(state);
    commandList->setPushConstants(&miniConstants, sizeof(miniConstants));
    commandList->dispatchRays(args);
}

void IntroPathTracer::ReferencePathTracer(nvrhi::CommandListHandle commandList)
{
    ScopedPerfMarker perfMarker("Reference PT", commandList);

    nvrhi::rt::State state;

    nvrhi::rt::DispatchRaysArguments args;
    nvrhi::Viewport viewport = GetView().GetViewport();
    uint32_t width = (uint32_t)(viewport.maxX - viewport.minX);
    uint32_t height = (uint32_t)(viewport.maxY - viewport.minY);
    args.width = width;
    args.height = height;

    // default miniConstants
    SampleMiniConstants miniConstants = { uint4(0, 0, 0, 0) };

    state.shaderTable = m_realTimePathTracer->GetShaderTable();
    state.bindings = { m_bindingSet, m_DescriptorTable->GetDescriptorTable() };
    commandList->setRayTracingState(state);
    commandList->setPushConstants(&miniConstants, sizeof(miniConstants));
    commandList->dispatchRays(args);
}

void IntroPathTracer::CreateRTPipelines()
{
    auto pipelineBaker = GetRTPipelineBaker();

    std::vector<donut::engine::ShaderMacro> shaderMacros = { donut::engine::ShaderMacro("PATH_TRACER_MODE", "PATH_TRACER_MODE_REFERENCE") };
    // these don't actually compile any shaders - this happens later in m_ptPipelineBaker->Update
    m_GBufferPipeline = pipelineBaker->CreateVariant("IntroSample/PopulateGBuffer.hlsl", shaderMacros, "GBuf");
    m_realTimePathTracer = pipelineBaker->CreateVariant("IntroSample/IntroPathTracer.hlsl", shaderMacros, "Intro");
    m_localCubemapPipeline = pipelineBaker->CreateVariant("IntroSample/LocalCubemapRT.hlsl", shaderMacros, "Cube");

    // Deferred Lighting
    auto computeBaker = GetComputePipelineBaker();
    m_deferredLightingVariant = computeBaker->CreateVariant(
        "IntroSample/RasterDeferredLighting.hlsl",  // source path relative to ShadersPath (Rtxpt/Shaders)
        "main",                                   // entry point
        {},                                       // macros (empty)
        { m_bindingLayout, m_bindlessLayout },    // binding layouts
        "DeferredLighting");                      // debug name

    // Depth hierarchy
    m_depthHierarchyRenderer->CreatePipelines(*computeBaker);

    // GTAO
    m_gtaoRenderer->CreatePipelines(m_bindingLayout, m_bindlessLayout, *computeBaker);

    // Create samplers
    nvrhi::SamplerDesc samplerDesc;
    samplerDesc.setAllFilters(false);  // Point sampling
    m_pointSampler = GetDevice()->createSampler(samplerDesc);

    samplerDesc.setAllFilters(true);   // Linear sampling
    m_linearSampler = GetDevice()->createSampler(samplerDesc);

    InvalidateBindingSet(); // For refreshing the samplers
    CreateReflectionPipelines();
}

void IntroPathTracer::CreateReflectionPipelines()
{
    m_reflectionSystemReady = false;
        
    // === Reflection constant buffer ===
    nvrhi::BufferDesc bufferDesc;
    bufferDesc.byteSize = sizeof(ReflectionConstants);
    bufferDesc.structStride = 0;
    bufferDesc.debugName = "ReflectionConstants";
    bufferDesc.isConstantBuffer = true;
    bufferDesc.initialState = nvrhi::ResourceStates::ConstantBuffer;
    bufferDesc.keepInitialState = true;
    m_reflectionConstantBuffer = GetDevice()->createBuffer(bufferDesc);
        
    // === SSR Pass ===
    // Uses dedicated binding layout with custom push constants (96 bytes)
    {
        // Create linear clamp sampler for depth hierarchy sampling
        nvrhi::SamplerDesc samplerDesc;
        samplerDesc.minFilter = true;
        samplerDesc.magFilter = true;
        samplerDesc.mipFilter = true;
        samplerDesc.addressU = nvrhi::SamplerAddressMode::Clamp;
        samplerDesc.addressV = nvrhi::SamplerAddressMode::Clamp;
        samplerDesc.addressW = nvrhi::SamplerAddressMode::Clamp;
        m_ssrSampler = GetDevice()->createSampler(samplerDesc);
        samplerDesc.minFilter = false;
        samplerDesc.magFilter = false;
        samplerDesc.mipFilter = false;
        m_nearestSampler = GetDevice()->createSampler(samplerDesc);

        // Dedicated binding layout - no constant buffers, all via push constants
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::PushConstants(0, sizeof(SSRPushConstants)),
            nvrhi::BindingLayoutItem::Texture_SRV(0),   // t_SSR_DepthHierarchy
            nvrhi::BindingLayoutItem::Texture_SRV(1),   // t_SSR_Depth
            nvrhi::BindingLayoutItem::Texture_SRV(2),   // t_SSR_Normal
            nvrhi::BindingLayoutItem::Texture_SRV(3),   // t_SSR_RoughnessMetal
            nvrhi::BindingLayoutItem::Texture_SRV(4),   // t_SSR_BaseColor
            nvrhi::BindingLayoutItem::Texture_SRV(5),   // t_SSR_CubemapGGX
            nvrhi::BindingLayoutItem::Texture_SRV(6),   // t_SSR_IrradianceCube
            nvrhi::BindingLayoutItem::Texture_SRV(7),   // t_SSR_BRDFLUT
            nvrhi::BindingLayoutItem::Texture_UAV(0),   // u_SSR_Result
            nvrhi::BindingLayoutItem::Sampler(0),       // s_SSR_LinearClamp
            nvrhi::BindingLayoutItem::Sampler(1),       // s_SSR_NearestClamp
        };
        m_ssrBindingLayout = GetDevice()->createBindingLayout(layoutDesc);
        
        m_ssrVariant = GetComputePipelineBaker()->CreateVariant(
            "IntroSample/SSRPasses.hlsl",                          // source path relative to ShadersPath
            "SSRCS",                                   // entry point
            {},                                        // macros (empty)
            { m_ssrBindingLayout },                    // dedicated binding layout
            "SSR");                                    // debug name
    }
        
    // === SSR Blur ===
    bufferDesc.byteSize = sizeof(SSRBlurConstants);
    bufferDesc.structStride = 0;
    bufferDesc.debugName = "SSRBlurConstants";
    bufferDesc.isConstantBuffer = true;
    bufferDesc.initialState = nvrhi::ResourceStates::ConstantBuffer;
    bufferDesc.keepInitialState = true;
    m_ssrBlurConstantBuffer = GetDevice()->createBuffer(bufferDesc);
            
    nvrhi::BindingLayoutDesc layoutDesc;
    layoutDesc.visibility = nvrhi::ShaderType::Compute;
    layoutDesc.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(0),
        nvrhi::BindingLayoutItem::PushConstants(1, sizeof(SampleMiniConstants)),  // g_MiniConst (from global bindings include)
        nvrhi::BindingLayoutItem::Texture_SRV(0),
        nvrhi::BindingLayoutItem::Texture_UAV(0),
        nvrhi::BindingLayoutItem::Sampler(0)
    };
    m_ssrBlurBindingLayout = GetDevice()->createBindingLayout(layoutDesc);
            
    m_ssrBlurVariant = GetComputePipelineBaker()->CreateVariant(
        "IntroSample/SSRPasses.hlsl",           // source path relative to ShadersPath
        "SSRBlurCS",                // entry point
        {},                         // macros (empty)
        { m_ssrBlurBindingLayout }, // binding layouts
        "SSRBlur");                 // debug name
        
    m_reflectionSystemReady = true;
}
    
void IntroPathTracer::DestroyRTPipelines()
{
    auto computeBaker = GetComputePipelineBaker();
    if (computeBaker)
    {
        if (m_depthHierarchyRenderer)
        {
            m_depthHierarchyRenderer->DestroyPipelines(*computeBaker);
            m_depthHierarchyRenderer.reset();
        }

        if (m_gtaoRenderer)
        {
            m_gtaoRenderer->DestroyPipelines(*computeBaker);
            m_gtaoRenderer.reset();
        }

        // Release compute shader variants from the baker
        if (m_deferredLightingVariant)
            computeBaker->ReleaseVariant(m_deferredLightingVariant);
        if (m_ssrVariant)
            computeBaker->ReleaseVariant(m_ssrVariant);
        if (m_ssrBlurVariant)
            computeBaker->ReleaseVariant(m_ssrBlurVariant);
    }

    m_deferredLightingVariant = nullptr;
    m_ssrVariant = nullptr;
    m_ssrBlurVariant = nullptr;
    
    m_realTimePathTracer = nullptr;
    m_GBufferPipeline = nullptr;
    m_localCubemapPipeline = nullptr;
}

std::string IntroPathTracer::GetMaterialSpecializationShader() const
{
    return "IntroPathTracer.hlsl";
}
    
void IntroPathTracer::SceneLoaded()
{
    Sample::SceneLoaded();

    // Ensure textures are allocated
    AllocateLocalCubemapTextures();
}

void IntroPathTracer::OnRenderTargetsRecreated()
{
    m_ssrBindingSet = nullptr;

    uint2 renderSize = GetRenderSize();

    // Recreate previous depth buffer to match new resolution
    nvrhi::TextureDesc desc;
    desc.width = renderSize.x;
    desc.height = renderSize.y;
    desc.format = nvrhi::Format::R32_FLOAT;
    desc.debugName = "PrevDepth";
    desc.isUAV = false;
    desc.initialState = nvrhi::ResourceStates::ShaderResource;
    desc.keepInitialState = true;
    m_prevDepth = GetDevice()->createTexture(desc);

    m_depthHierarchyRenderer->OnRenderTargetsRecreated(renderSize.x, renderSize.y);
    m_gtaoRenderer->OnRenderTargetsRecreated(renderSize.x, renderSize.y);

    InvalidateBindingSet();
}
    
// Override to add reflection textures to the main binding set
void IntroPathTracer::AddCustomBindings(nvrhi::BindingSetDesc& bindingSetDesc)
{
    // Replace the null placeholder bindings with actual reflection textures
    // Note: Depth hierarchy UAVs (u80-84) and constant buffer (b11) are in a dedicated
    // binding set, not the global set, to avoid duplicate barriers with t_DepthHierarchy SRV
        
    if (!m_reflectionSystemReady || !m_ggxFilteredCubemap)
        return; // Keep null placeholders if reflection system not ready
        
    for (auto& binding : bindingSetDesc.bindings)
    {
        if (binding.type == nvrhi::ResourceType::Texture_SRV)
        {
            switch (binding.slot)
            {
                case 80: binding.resourceHandle = m_ggxFilteredCubemap; break;
                case 81: binding.resourceHandle = m_IrradianceCube; break;
                case 82: binding.resourceHandle = m_renderTargets->SSRBlurMipChain; break;
                case 84: binding.resourceHandle = m_depthHierarchyRenderer->m_depthHierarchy; break;
                case 86:
                    if (m_gtaoRenderer && m_gtaoRenderer->GetOutputTexture())
                        binding.resourceHandle = m_gtaoRenderer->GetOutputTexture();
                    break;
                case 87:
                    if (m_prevDepth)
                        binding.resourceHandle = m_prevDepth;
                    break;
            }
        }
        else if (binding.type == nvrhi::ResourceType::Texture_UAV)
        {
            // Only SSR result is in global set (depth hierarchy UAVs are in dedicated set)
            if (binding.slot == 85)
            {
                binding.resourceHandle = m_renderTargets->SSRResult;
            }
        }
        else if (binding.type == nvrhi::ResourceType::ConstantBuffer)
        {
            switch (binding.slot)
            {
                case 10: binding.resourceHandle = m_reflectionConstantBuffer; break;
            }
        }
    }
    
    m_reflectionTexturesBound = true;
}

class IntroSample : public SampleBaseApp
{
    std::unique_ptr<Sample> CreateMainRenderPass(donut::app::DeviceManager& deviceManager, const CommandLineOptions& cmdLineOptions) override
    {
        // For now, create the basic Sample. Later we'll use IntroRenderer
        return std::make_unique<IntroPathTracer>(deviceManager, cmdLineOptions);
    }
};

#ifdef _WIN32
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
int main(int __argc, const char** __argv)
#endif
{
    IntroSample example;

    // Run the sample app
    const auto status = example.Init(__argc, __argv);
    if (status == SampleBaseApp::InitReturnCodes::Success)
    {
        example.RunMainLoop();

        example.End();
    }
    
    return static_cast<int>(status);
}
