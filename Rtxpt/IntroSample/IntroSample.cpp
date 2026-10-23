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
#include <nvrhi/utils.h>

// Include shader constant structures
#include "Lighting/Distant/EnvMapBaker.h"
#include "Materials/MaterialsBaker.h"
#include "RTXDI/ShaderParameters.h"
#include "SampleCommon/PackingUtils.h"
#include "SampleCommon/ExtendedScene.h"
#include <donut/engine/Scene.h>
#include <donut/engine/SceneGraph.h>
#include <rtxdi/RtxdiUtils.h>

#include "IntroSample.h"
#include "IntroSampleUI.h"
#include "DepthHierarchyRenderer.h"

using namespace donut;
using namespace donut::math;
using namespace donut::engine;

namespace
{
    // Reflection-related parameters that vary at runtime are passed to the deferred-lighting,
    // RTGI, RT-reflections and SSR passes via push constants instead of a dedicated cbuffer.
    // For passes that bind the global SampleMiniConstants block, two channels are reserved:
    //   params.z = prefiltered environment cube max sampleable mip index
    //   params.w = SSR blur chain max mip index
    // The SSR trace pass uses its own SSRPushConstants and packs the same value into
    // SSRPushConstants::screenAndParams.z (formerly an unused "farZ" slot).
    // RTGI is the one exception: its params.z instead carries the UI-configurable, fixed mip its own
    // environment lookups sample at (RTGIRenderer::Render's envMip), clamped to kEnvGGXMaxMip here
    // before it's passed down - not the max mip index itself.

    // Size of the GGX-prefiltered environment cube used for specular IBL. Kept compact and
    // independent of the (potentially large) environment map resolution.
    constexpr uint32_t kEnvGGXCubeSize = 256u;
    // log2(kEnvGGXCubeSize) - 1: skips the smallest 1x1 mip.
    constexpr uint32_t kEnvGGXMaxMip = 7u;

    constexpr uint32_t kIntroReSTIRNeighborOffsetCount = 8192u;

    // C++ mirror of the HLSL `IntroReSTIRDIConstants` defined in
    // Shaders/IntroSample/IntroReSTIRDIConstants.hlsli - keep the field order in sync.
    struct IntroReSTIRDIConstants
    {
        RTXDI_RuntimeParameters runtimeParams;
        RTXDI_Parameters restirDI;
        uint32_t numAnalyticLocalLightSamples  = 8;
        uint32_t numTriangleLocalLightSamples  = 8;
        uint32_t numInfiniteLightSamples       = 1;
        uint32_t numEnvironmentSamples         = 0;
        uint32_t lightRegionCapacity           = 0;
        uint32_t lightCount                    = 0;
        uint32_t temporalHistoryValid          = 0;
        uint32_t padding                       = 0;
    };

    // C++ mirror of the HLSL `IntroPrepareLightsConstants` defined in
    // Shaders/IntroSample/IntroPrepareLights.hlsl - keep the field order in sync.
    struct IntroPrepareLightsConstants
    {
        uint32_t numTasks             = 0;
        uint32_t triangleDestBaseIndex = 0;
        uint32_t padding[2]           = {};
    };
} // Anonymous namespace

IntroPathTracer::IntroPathTracer(donut::app::DeviceManager& deviceManager,
    const CommandLineOptions& cmdLine)
    : Sample(deviceManager, cmdLine)
{
    m_sampleCapabilities = SampleCapabilities{
        .NeedsGBuffer              = true,
        .NeedsDirectLightingBuffer = true,
        .EnvMapRasterPrecompute    = true,
        .UseNeeAtLightSampling     = false,
    };

    m_ui.EnableBloom = false;
    m_ui.RenderWhenOutOfFocus = true;

    // The Intro path samples scene directional lights analytically (see PrepareLights), so disable
    // baking them into the environment cubemap.
    m_bakeDirectionalLightsIntoEnvMap = false;

    // Launch on the default step of the progression. kIntroDefaultStep (IntroSampleUI.h) is the
    // single source of truth; ApplyIntroStepPreset() sets every RT toggle + direct-lighting mode
    // from the matching kIntroStepPresets entry. The recommended AA is re-applied in CreateUI(),
    // after the base UI/perf preset has run (which would otherwise clobber RealtimeAA).
    m_introUIData.IntroStepIndex = kIntroDefaultStep;
    ApplyIntroStepPreset(m_ui, kIntroDefaultStep);
    m_ui.UseReSTIRDI = false; // Advance Sample's integration
    m_ui.UseReSTIRGI = false; // Advance Sample's integration

    m_depthHierarchyRenderer = std::make_unique<DepthHierarchyRenderer>(GetDevice());
    m_aoRenderer = std::make_unique<AORenderer>(GetDevice());
    m_reflectionRenderer = std::make_unique<ReflectionRenderer>(GetDevice());
    m_rtgiRenderer = std::make_unique<RTGIRenderer>(GetDevice());
    m_indirectPathTracer = std::make_unique<IndirectPathTracer>(GetDevice());
    m_sharcRenderer = std::make_unique<SHaRCRenderer>(GetDevice());

    m_introReSTIRConstBuffer = GetDevice()->createBuffer(
        nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(IntroReSTIRDIConstants), "IntroReSTIRDIConstants", 16));

    CreateIntroNeighborOffsetsBuffer();
}

void IntroPathTracer::SampleRenderCode(nvrhi::IFramebuffer* framebuffer, nvrhi::CommandListHandle commandList, const SampleConstants& constants, bool needNewPasses)
{
    ScopedPerfMarker perfMarker("Intro-Sample", commandList);
    
    m_simplifiedRTPipelines->Update(m_scene, (unsigned int)m_subInstanceData.size(), needNewPasses);

    // Prepare the Intro RTXDI light buffers (local analytic + directional + emissive triangles) and
    // the light-region indices up front, before any pass that reads them.
    PrepareLights(commandList, m_currentConstants.lightControl);

    if (needNewPasses)
        m_reSTIRHistoryValid = false;
    if (m_ui.RealtimeMode && m_ui.IntroDirectLighting == IntroDirectLightingMode::ReSTIRDI)
        EnsureIntroReSTIRDIResources();
    RefreshBindingSetIfNeeded();

    // Upload after preparation: a growing light set may have changed the region capacity,
    // and a new ReSTIR context must supply valid reservoir indices on its very first frame.
    // b12 is volatile and bound globally, so write it even in non-ReSTIR modes.
    FillIntroReSTIRDIConstants(commandList);
    commandList->writeBuffer(m_constantBuffer, &m_currentConstants, sizeof(m_currentConstants));

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

        // Derive the prefiltered specular + diffuse-irradiance cubes from the environment map.
        // Reprocesses only when the environment contents change (see ProcessEnvIBL).
        ProcessEnvIBL(commandList);

        // Gather render settings and make sure we only turn on compatible components.
        const bool useMISIndirect = m_ui.UseMISIndirect && m_indirectPathTracer
            && (m_indirectPathTracer->IsReady() || m_indirectPathTracer->IsSharcReady());
        const bool useRTReflections = !useMISIndirect && m_ui.UseRTReflections && m_reflectionRenderer->IsRTReady();
        const bool useSSR = !useRTReflections && !useMISIndirect;
        const bool useRTGI = !useMISIndirect && m_ui.UseRTGI && m_rtgiRenderer && m_rtgiRenderer->IsReady();
        // AO only modulates the diffuse environment fallback and the SSR environment fallback.
        // Direct-light sampling does not determine whether either of those consumers is active.
        const bool useAO = m_aoRenderer && (!(useRTGI || useMISIndirect) || useSSR);
        const bool useSHaRC = (useRTGI || useMISIndirect) && m_ui.UseSHaRC && m_sharcRenderer && m_sharcRenderer->IsUpdateReady();

        // Depth hierarchy (needed by GTAO and SSR, must run before both)
        if (m_depthHierarchyRenderer && (useAO || useSSR))
        {
            ScopedPerfMarker perfMarker("Depth hierarchy", commandList);
            m_depthHierarchyRenderer->Generate(commandList, m_renderTargets->Depth, width, height);
        }

        // Ambient occlusion (GTAO or RTAO based on UI toggle)
        if (useAO)
        {
            if (m_ui.UseRTAO)
                m_aoRenderer->RenderRTAO(commandList, constants, m_frameIndex, m_ui.AORadius,
                    m_depthHierarchyRenderer->m_depthHierarchy, m_renderTargets->SpecNormal,
                    m_renderTargets->ScreenMotionVectors,
                    m_prevDepth, m_renderTargets->MaterialInfo, m_bindingSet, false); // GTAO always runs its internal denoiser because the half resolution scaling doesn't play nice with DLSS-RR
            else
                m_aoRenderer->RenderGTAO(commandList, constants, m_frameIndex, m_ui.AORadius,
                    m_depthHierarchyRenderer->m_depthHierarchy, m_renderTargets->SpecNormal,
                    m_renderTargets->ScreenMotionVectors,
                    m_prevDepth, m_renderTargets->MaterialInfo, m_bindingSet);
        }

        // SHaRC Update + Resolve (only when SHaRC is enabled)
        uint32_t sharcCapacity = 0; // Used to retrieve entries from the cache in later passes
        if (useSHaRC)
        {
            sharcCapacity = m_sharcRenderer->GetCapacity();
        }

        // Direct lighting runs before SHaRC in case we want to dump the NEE samples into the cache.
        DirectLightingPass(commandList, sharcCapacity);

        // SHaRC Update + Resolve (only when SHaRC is enabled)
        if (useSHaRC)
        {
            m_sharcRenderer->DispatchUpdate(commandList, width, height,
                m_bindingSet, m_DescriptorTable->GetDescriptorTable());
            m_sharcRenderer->Resolve(commandList, m_bindingSet, m_DescriptorTable->GetDescriptorTable());
        }

        // MIS Indirect: unified specular+diffuse in a single pass (replaces both reflections and RTGI)
        if (useMISIndirect)
        {
            // Only suppress paths that the active primary direct-light sampler actually covers.
            // Steps 6/7 sample analytic lights only; Step 8 can also sample triangles and the sky.
            const auto& lightControl = m_currentConstants.lightControl;
            const bool usesReSTIR = m_ui.IntroDirectLighting == IntroDirectLightingMode::ReSTIRDI;
            const bool primaryEmissiveNEE = usesReSTIR && m_introUIData.IntroReSTIRNumTriangleLocalSamples > 0
                && lightControl.LocalRegionNumLights > lightControl.LocalRegionNumAnalyticLights;
            const bool primaryEnvironmentNEE = usesReSTIR && m_introUIData.IntroReSTIRSampleEnvironmentMap
                && m_introUIData.IntroReSTIRNumEnvironmentSamples > 0 && lightControl.EnvironmentLightPresent != 0;
            m_indirectPathTracer->Render(commandList, width, height,
                m_bindingSet, m_DescriptorTable->GetDescriptorTable(),
                useSHaRC, sharcCapacity, primaryEmissiveNEE, primaryEnvironmentNEE);
        }
        else
        {
            // Indirect diffuse
            if (useRTGI)
            {
                // ReSTIR DI samples emissive triangles explicitly at the primary surface, so RTGI
                // must not also count emission at its first hit (that "primary -> emitter" path is
                // already covered by direct NEE). Only let RTGI add hit emission when ReSTIR DI is off.
                const bool rtgiIncludeHitEmissive = m_ui.IntroDirectLighting != IntroDirectLightingMode::ReSTIRDI;
                // Clamp the UI-configurable env sample mip to the prefiltered cube's actual max mip.
                const uint32_t rtgiEnvMip = dm::min(m_ui.RTGIEnvMip, kEnvGGXMaxMip);
                m_rtgiRenderer->Render(commandList, width, height,
                    m_bindingSet, m_DescriptorTable->GetDescriptorTable(),
                    useSHaRC, sharcCapacity, rtgiEnvMip, rtgiIncludeHitEmissive);
            }

            // Reflections
            {
                ScopedPerfMarker perfMarker("Reflections", commandList);

                if (useRTReflections)
                {
                    m_reflectionRenderer->RenderRTReflections(commandList, width, height,
                        m_bindingSet, m_DescriptorTable->GetDescriptorTable(),
                        kEnvGGXMaxMip);
                }
                else // SSR path
                {
                    m_reflectionRenderer->RenderSSR(commandList, constants, width, height,
                        *m_renderTargets,
                        m_depthHierarchyRenderer->m_depthHierarchy,
                        m_depthHierarchyRenderer->GetNumMips(),
                        m_envGGXFilteredCube ? m_envGGXFilteredCube : GetCommonPasses()->m_BlackCubeMapArray,
                        m_envIrradianceCube ? m_envIrradianceCube : GetCommonPasses()->m_BlackCubeMapArray,
                        GetEnvMapBaker()->GetBRDFLUT(),
                        kEnvGGXMaxMip);
                }
            }
        }

        DeferredLightingCombine(commandList, useRTReflections);

        if (m_ui.RealtimeAA == 3) // DLSS-RR
            PrepareGuideBuffers(commandList);

        CopyGBufferHistory(commandList);
    }
    else
    {
        ReferencePathTracer(commandList);
    }

    commandList->setBufferState(m_renderTargets->DenoisingLayersBuffer, nvrhi::ResourceStates::UnorderedAccess);
}
    
void IntroPathTracer::AllocateEnvIBLTextures()
{
    if (m_envGGXFilteredCube)
        return; // Already allocated
    
    uint cubemapSize = kEnvGGXCubeSize;
    uint mipLevels = uint(std::log2((float)cubemapSize) + 0.5f);
    
    // GGX-prefiltered environment cube with mip chain (specular)
    {
        nvrhi::TextureDesc desc;
        desc.width = cubemapSize;
        desc.height = cubemapSize;
        desc.depth = 1;
        desc.arraySize = 6;
        desc.mipLevels = mipLevels;
        desc.format = nvrhi::Format::RGBA16_FLOAT;
        desc.dimension = nvrhi::TextureDimension::TextureCube;
        desc.debugName = "EnvGGXFiltered";
        desc.isUAV = true;
        desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        desc.keepInitialState = true;
        
        m_envGGXFilteredCube = GetDevice()->createTexture(desc);
    }
    
    // Diffuse irradiance cube
    {
        nvrhi::TextureDesc desc;
        desc.width = EnvMapBaker::c_IrradianceCubeSize;
        desc.height = EnvMapBaker::c_IrradianceCubeSize;
        desc.depth = 1;
        desc.arraySize = 6;
        desc.mipLevels = 1;
        desc.format = nvrhi::Format::RGBA16_FLOAT;
        desc.dimension = nvrhi::TextureDimension::TextureCube;
        desc.debugName = "EnvIrradiance";
        desc.isUAV = true;
        desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        desc.keepInitialState = true;
        
        m_envIrradianceCube = GetDevice()->createTexture(desc);
    }

    // Invalidate SSR binding set so it gets recreated with the real IBL textures
    if (m_reflectionRenderer)
        m_reflectionRenderer->InvalidateBindingSet();
}

void IntroPathTracer::ProcessEnvIBL(nvrhi::CommandListHandle commandList)
{
    AllocateEnvIBLTextures();

    const std::shared_ptr<EnvMapBaker>& envMapBaker = GetEnvMapBaker();
    nvrhi::TextureHandle envCube = envMapBaker ? envMapBaker->GetEnvMapCube() : nullptr;
    if (!envCube)
        return; // Environment not baked yet

    ScopedPerfMarker perfMarker("ProcessEnvIBL", commandList);

    // The environment cube already carries a solid-angle mip chain from EnvMapBaker::Update,
    // so only GGX prefiltering (specular) and SH convolution (diffuse) are needed here.
    EnvMapBaker::CubemapProcessingOptions opts;
    opts.generateMips = false;
    opts.ggxPrefilter = true;
    opts.projectToSH = true;

    EnvMapBaker::CubemapProcessingResults dstTextures;
    dstTextures.filteredCubemap = m_envGGXFilteredCube;
    dstTextures.diffuseIrradianceCube = m_envIrradianceCube;

    envMapBaker->ProcessCubemap(
        commandList.Get(), GetBindingCache(),
        envCube,
        m_envMapSceneParams,
        opts, dstTextures);
}

void IntroPathTracer::EnsureIntroRTXDIResources(uint32_t maxAnalyticLights)
{
    const uint32_t requestedCapacity = std::max(1u, maxAnalyticLights);
    if (m_RTXDILightsBase && m_RTXDILightsEx && requestedCapacity <= m_RTXDILightCapacity)
        return;

    m_RTXDILightCapacity = requestedCapacity;

    // Two parallel SoA buffers (Base + Extended) mirror the LightsBaker layout.
    // Each is sized for the two ping-pong regions (capacity * 2 entries).
    nvrhi::BufferDesc baseDesc;
    baseDesc.byteSize = sizeof(PolymorphicLightInfo) * m_RTXDILightCapacity * 2;
    baseDesc.structStride = sizeof(PolymorphicLightInfo);
    baseDesc.canHaveUAVs = true;    // emissive-triangle bake writes triangles here as a UAV
    baseDesc.initialState = nvrhi::ResourceStates::ShaderResource;
    baseDesc.keepInitialState = true;
    baseDesc.debugName = "IntroRTXDILightsBase";
    m_RTXDILightsBase = GetDevice()->createBuffer(baseDesc);

    nvrhi::BufferDesc exDesc;
    exDesc.byteSize = sizeof(PolymorphicLightInfoEx) * m_RTXDILightCapacity * 2;
    exDesc.structStride = sizeof(PolymorphicLightInfoEx);
    exDesc.canHaveUAVs = true;      // emissive-triangle bake writes triangles here as a UAV
    exDesc.initialState = nvrhi::ResourceStates::ShaderResource;
    exDesc.keepInitialState = true;
    exDesc.debugName = "IntroRTXDILightsEx";
    m_RTXDILightsEx = GetDevice()->createBuffer(exDesc);

    if (!m_materialsBaker || !m_materialsBaker->GetMaterialDataBuffer())
    {
        InvalidateBindingSet();
        return;
    }

    RecreateBindingSet();
}

uint32_t IntroPathTracer::BuildIntroEmissiveTriangleTasks(std::vector<PrepareLightsTask>& outTasks,
    std::vector<EmissiveGeometryKey>* outKeys) const
{
    outTasks.clear();
    if (outKeys)
        outKeys->clear();
    if (!m_scene || !m_scene->GetSceneGraph())
        return 0;

    // Mirrors PrepareLightsPass::Process emissive loop: one task per emissive geometry.
    // No per-triangle mapping table: the host checks the ordered geometry keys and resets
    // temporal reuse when the layout changes. Otherwise indices translate by the region capacity.
    uint32_t lightBufferOffset = 0;
    const auto& instances = m_scene->GetSceneGraph()->GetMeshInstances();
    for (const auto& instance : instances)
    {
        const auto& mesh = instance->GetMesh();
        if (!mesh)
            continue;

        for (size_t geometryIndex = 0; geometryIndex < mesh->geometries.size(); ++geometryIndex)
        {
            const auto& geometry = mesh->geometries[geometryIndex];
            PTMaterial* materialPT = PTMaterial::SafeCast(geometry->material);
            if (!materialPT || !materialPT->IsEmissiveSurfaceLightSampling() || materialPT->SkipRender)
                continue;

            assert(geometryIndex < 0xfff);

            PrepareLightsTask task = {};
            task.instanceAndGeometryIndex = (uint32_t(instance->GetInstanceIndex()) << 12) | uint32_t(geometryIndex & 0xfff);
            task.triangleCount = geometry->numIndices / 3;
            task.lightBufferOffset = lightBufferOffset;
            task.previousLightBufferOffset = -1;

            lightBufferOffset += task.triangleCount;
            outTasks.push_back(task);
            if (outKeys)
                outKeys->push_back({ instance.get(), geometry.get(), task.triangleCount });
        }
    }

    return lightBufferOffset; // total triangle count T
}

void IntroPathTracer::EnsureIntroPrepareLightsResources(uint32_t taskCount)
{
    if (!m_introPrepareLightsBindingLayout)
    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0),  // u_LightsBase
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(1),  // u_LightsEx
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0),  // t_TaskBuffer
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1),  // t_SubInstanceData
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2),  // t_InstanceData
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(3),  // t_GeometryData
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(5),  // t_PTMaterialData
            nvrhi::BindingLayoutItem::Sampler(0),               // s_MaterialSampler
        };
        m_introPrepareLightsBindingLayout = GetDevice()->createBindingLayout(layoutDesc);
    }

    if (!m_introPrepareLightsCB)
    {
        nvrhi::BufferDesc cbDesc;
        cbDesc.byteSize = sizeof(IntroPrepareLightsConstants);
        cbDesc.debugName = "IntroPrepareLightsConstants";
        cbDesc.isConstantBuffer = true;
        cbDesc.isVolatile = true;
        cbDesc.maxVersions = 16;
        m_introPrepareLightsCB = GetDevice()->createBuffer(cbDesc);
    }

    const uint32_t requestedTasks = std::max(1u, taskCount);
    if (!m_introPrepareLightsTaskBuffer || requestedTasks > m_introPrepareLightsTaskCapacity)
    {
        m_introPrepareLightsTaskCapacity = requestedTasks;
        nvrhi::BufferDesc taskDesc;
        taskDesc.byteSize = sizeof(PrepareLightsTask) * m_introPrepareLightsTaskCapacity;
        taskDesc.structStride = sizeof(PrepareLightsTask);
        taskDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        taskDesc.keepInitialState = true;
        taskDesc.debugName = "IntroPrepareLightsTasks";
        m_introPrepareLightsTaskBuffer = GetDevice()->createBuffer(taskDesc);
    }

    const bool dependenciesReady = m_introPrepareLightsBindingLayout
        && m_RTXDILightsBase && m_RTXDILightsEx
        && m_subInstanceBuffer
        && m_scene && m_scene->GetInstanceBuffer() && m_scene->GetGeometryBuffer()
        && m_materialsBaker && m_materialsBaker->GetMaterialDataBuffer();

    if (!dependenciesReady)
        return;

    // Rebuild the binding set whenever any referenced resource handle changed (light/task buffers
    // grow, or scene/material buffers are recreated on a scene or material rebake), or on first creation.
    const bool handlesChanged = !m_introPrepareLightsBindingSet
        || m_introPrepareLightsBoundBase        != m_RTXDILightsBase.Get()
        || m_introPrepareLightsBoundEx          != m_RTXDILightsEx.Get()
        || m_introPrepareLightsBoundTasks       != m_introPrepareLightsTaskBuffer.Get()
        || m_introPrepareLightsBoundSubInstance != m_subInstanceBuffer.Get()
        || m_introPrepareLightsBoundInstance    != m_scene->GetInstanceBuffer()
        || m_introPrepareLightsBoundGeometry    != m_scene->GetGeometryBuffer()
        || m_introPrepareLightsBoundMaterial    != m_materialsBaker->GetMaterialDataBuffer().Get();

    if (handlesChanged)
    {
        nvrhi::BindingSetDesc setDesc;
        setDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_introPrepareLightsCB),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_RTXDILightsBase),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(1, m_RTXDILightsEx),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_introPrepareLightsTaskBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_subInstanceBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_scene->GetInstanceBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(3, m_scene->GetGeometryBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(5, m_materialsBaker->GetMaterialDataBuffer()),
            nvrhi::BindingSetItem::Sampler(0, m_CommonPasses->m_AnisotropicWrapSampler),
        };
        m_introPrepareLightsBindingSet = GetDevice()->createBindingSet(setDesc, m_introPrepareLightsBindingLayout);
        m_introPrepareLightsBoundBase        = m_RTXDILightsBase.Get();
        m_introPrepareLightsBoundEx          = m_RTXDILightsEx.Get();
        m_introPrepareLightsBoundTasks       = m_introPrepareLightsTaskBuffer.Get();
        m_introPrepareLightsBoundSubInstance = m_subInstanceBuffer.Get();
        m_introPrepareLightsBoundInstance    = m_scene->GetInstanceBuffer();
        m_introPrepareLightsBoundGeometry    = m_scene->GetGeometryBuffer();
        m_introPrepareLightsBoundMaterial    = m_materialsBaker->GetMaterialDataBuffer().Get();
    }
}

void IntroPathTracer::PrepareLights(nvrhi::ICommandList* commandList, LightControlData& lightControl)
{
    ScopedPerfMarker perfMarker("IntroRTXDIPrepareLights", commandList);

    // Two parallel groups: positional analytic lights ("local" region) and directional
    // lights ("infinite" region). Each group is packed into two parallel SoA buffers
    // (Base + Extended), mirroring the LightsBaker layout for a fair perf comparison.
    std::vector<PolymorphicLightInfo>   localBase;
    std::vector<PolymorphicLightInfoEx> localEx;
    std::vector<PolymorphicLightInfo>   infiniteBase;
    std::vector<PolymorphicLightInfoEx> infiniteEx;
    LightLayout layout;
    localBase.reserve(m_lights.size());
    localEx.reserve(m_lights.size());
    infiniteBase.reserve(m_lights.size());
    infiniteEx.reserve(m_lights.size());
    for (const std::shared_ptr<donut::engine::Light>& light : m_lights)
    {
        if (!light)
            continue;
        if (PackingUtils::IsAnalyticLight(*light))
        {
            PolymorphicLightInfoFull packed = PackingUtils::ConvertLight(*light);
            localBase.push_back(packed.Base);
            localEx.push_back(packed.Extended);
            layout.analyticLights.push_back(light.get());
        }
        else if (PackingUtils::IsInfiniteAnalyticLight(*light))
        {
            PolymorphicLightInfoFull packed = PackingUtils::ConvertLight(*light);
            infiniteBase.push_back(packed.Base);
            infiniteEx.push_back(packed.Extended);
            layout.infiniteLights.push_back(light.get());
        }
    }

    // Fold in the procedural sky's sun (if any) as an extra directional light in the infinite group.
    AddProceduralSunLight(infiniteBase, infiniteEx);
    if (infiniteBase.size() > layout.infiniteLights.size())
        layout.infiniteLights.push_back(nullptr); // Stable identity for the synthetic sun.

    // Emissive triangles are baked on the GPU into the local region, right after the analytic
    // local lights (see IntroPrepareLights.hlsl). Build the per-geometry task list up front so the
    // buffer can be sized for the combined A + T + I before any writes.
    std::vector<PrepareLightsTask> triangleTasks;
    const uint32_t triangleCount = BuildIntroEmissiveTriangleTasks(triangleTasks, &layout.emissiveGeometries);  // T

    const uint32_t analyticLocalCount = uint32_t(localBase.size());     // A
    const uint32_t infiniteCount = uint32_t(infiniteBase.size());       // I

    // ReSTIR DI can importance-sample the environment map as an extra RIS stream. That needs a single
    // synthetic kEnvironment light appended after the triangles [I | A | T | E]; it is resolved through
    // the normal polymorphic path but sampled only by the custom env stream (never by the local/infinite
    // streams, which stay within their own regions).
    const bool envSamplingOn = m_introUIData.IntroReSTIRSampleEnvironmentMap
        && m_ui.EnvironmentMapParams.Enabled
        && GetEnvMapBaker() != nullptr
        && GetEnvMapBaker()->GetImportanceSampling() != nullptr;
    const uint32_t envLightCount = envSamplingOn ? 1u : 0u;

    const uint32_t totalLightCount = analyticLocalCount + triangleCount + infiniteCount + envLightCount;
    EnsureIntroRTXDIResources(totalLightCount);
    layout.capacity = m_RTXDILightCapacity;
    layout.environmentLight = envSamplingOn;

    const uint32_t currentRegion = m_RTXDIFrameIndex & 1u;
    const uint32_t base = currentRegion * m_RTXDILightCapacity;

    // Infinite-first layout per ping-pong half: [infinite I | local analytic A | triangles T].
    // The CPU-written regions (infinite, then analytic) are contiguous at the front; the GPU-baked
    // triangles trail at the end. Analytic locals stay adjacent to triangles so ReSTIR DI can treat
    // them as one merged local region, while Streaming RIS reads only the analytic sub-range.
    const uint32_t infiniteOffset = base;
    const uint32_t analyticOffset = base + infiniteCount;
    const uint32_t triangleOffset = base + infiniteCount + analyticLocalCount;

    // Infinite lights: [base, base + I)
    if (!infiniteBase.empty() && m_RTXDILightsBase && m_RTXDILightsEx)
    {
        commandList->writeBuffer(
            m_RTXDILightsBase,
            infiniteBase.data(),
            sizeof(PolymorphicLightInfo) * infiniteBase.size(),
            sizeof(PolymorphicLightInfo) * infiniteOffset);
        commandList->writeBuffer(
            m_RTXDILightsEx,
            infiniteEx.data(),
            sizeof(PolymorphicLightInfoEx) * infiniteEx.size(),
            sizeof(PolymorphicLightInfoEx) * infiniteOffset);
    }
    // Local analytic lights: [base + I, base + I + A)
    if (!localBase.empty() && m_RTXDILightsBase && m_RTXDILightsEx)
    {
        commandList->writeBuffer(
            m_RTXDILightsBase,
            localBase.data(),
            sizeof(PolymorphicLightInfo) * localBase.size(),
            sizeof(PolymorphicLightInfo) * analyticOffset);
        commandList->writeBuffer(
            m_RTXDILightsEx,
            localEx.data(),
            sizeof(PolymorphicLightInfoEx) * localEx.size(),
            sizeof(PolymorphicLightInfoEx) * analyticOffset);
    }

    // Environment light: single slot at the very end [base + I + A + T]. Written on the CPU (a bare type
    // marker); it sits past every sampled region so the local/infinite/triangle streams never touch it.
    if (envSamplingOn && m_RTXDILightsBase && m_RTXDILightsEx)
    {
        const uint32_t envIndex = triangleOffset + triangleCount;
        PolymorphicLightInfoFull envLight = PackingUtils::MakeEnvironmentLight();
        commandList->writeBuffer(m_RTXDILightsBase, &envLight.Base, sizeof(PolymorphicLightInfo), sizeof(PolymorphicLightInfo) * envIndex);
        commandList->writeBuffer(m_RTXDILightsEx, &envLight.Extended, sizeof(PolymorphicLightInfoEx), sizeof(PolymorphicLightInfoEx) * envIndex);
    }

    // Triangles: [base + I + A, base + I + A + T) - GPU bake reads scene geometry through bindless buffers.
    bool trianglesReady = triangleCount == 0;
    if (triangleCount > 0 && m_RTXDILightsBase && m_RTXDILightsEx)
    {
        EnsureIntroPrepareLightsResources(uint32_t(triangleTasks.size()));

        if (m_introPrepareLightsVariant && m_introPrepareLightsVariant->GetPipeline() && m_introPrepareLightsBindingSet)
        {
            commandList->writeBuffer(
                m_introPrepareLightsTaskBuffer,
                triangleTasks.data(),
                sizeof(PrepareLightsTask) * triangleTasks.size());

            IntroPrepareLightsConstants bakeConstants;
            bakeConstants.numTasks = uint32_t(triangleTasks.size());
            bakeConstants.triangleDestBaseIndex = triangleOffset;
            commandList->writeBuffer(m_introPrepareLightsCB, &bakeConstants, sizeof(bakeConstants));

            commandList->setBufferState(m_RTXDILightsBase, nvrhi::ResourceStates::UnorderedAccess);
            commandList->setBufferState(m_RTXDILightsEx, nvrhi::ResourceStates::UnorderedAccess);
            commandList->commitBarriers();

            nvrhi::ComputeState state;
            state.pipeline = m_introPrepareLightsVariant->GetPipeline();
            state.bindings = { m_introPrepareLightsBindingSet, m_scene->GetDescriptorTable() };
            commandList->setComputeState(state);
            commandList->dispatch(div_ceil(triangleCount, 256u));
            trianglesReady = true;
        }
    }

    // Publish the freshly written light buffers as SRVs (t90/t91). Every direct-lighting mode
    // (Unshadowed, UniformSampling, StreamingRIS, ReSTIR DI), the RTGI hit lighting, and eventually
    // the reference PT read from these, so they must be in ShaderResource state before any pass.
    if (totalLightCount > 0 && m_RTXDILightsBase && m_RTXDILightsEx)
    {
        commandList->setBufferState(m_RTXDILightsBase, nvrhi::ResourceStates::ShaderResource);
        commandList->setBufferState(m_RTXDILightsEx, nvrhi::ResourceStates::ShaderResource);
        commandList->commitBarriers();
    }

    // Region advertisement (layout [infinite | local analytic | triangles]):
    //  - Infinite region covers [base, base + I).
    //  - Local region starts at the analytic block; ReSTIR DI (Step 8) samples the full A + T,
    //    while Streaming RIS (Step 7) reads LocalRegionNumAnalyticLights (= A) only.
    lightControl.InfiniteRegionFirstLightIndex = infiniteOffset;
    lightControl.InfiniteRegionNumLights = infiniteCount;
    lightControl.LocalRegionFirstLightIndex = analyticOffset;
    lightControl.LocalRegionNumLights = analyticLocalCount + triangleCount;
    lightControl.LocalRegionNumAnalyticLights = analyticLocalCount;
    lightControl.EnvironmentLightIndex = envSamplingOn ? (triangleOffset + triangleCount) : 0u;
    lightControl.EnvironmentLightPresent = envSamplingOn ? 1u : 0u;

    m_lightLayoutMatchesPrevious = m_previousLightsReady && trianglesReady && layout == m_previousLightLayout;
    m_previousLightLayout = std::move(layout);
    m_previousLightsReady = trianglesReady;

    m_RTXDIFrameIndex++;

    // Guarantee the shared binding set that exposes these light buffers (t90/t91) to every pass is
    // valid before any of them run. EnsureIntroRTXDIResources above only rebuilds the set when the
    // buffer is (re)allocated, so this catches the case where the set was invalidated elsewhere.
    // This keeps the light buffer usable by all direct-lighting modes, RTGI, and the reference path
    // tracer without depending on ReSTIRDIDirectLightingPass (the only other per-frame place that
    // rebuilds the set) actually executing this frame.
    RefreshBindingSetIfNeeded();
}

void IntroPathTracer::AddProceduralSunLight(std::vector<PolymorphicLightInfo>& infiniteBase, std::vector<PolymorphicLightInfoEx>& infiniteEx)
{
    // The procedural sky's sun is not a scene-graph Light: it is computed inside the sky shader and
    // baked into the environment cubemap. So it never appears in m_lights and the light loops in
    // PrepareLights do not pick it up. Synthesize a directional analytic light from the sky's sun
    // parameters so the Intro direct-lighting/RTGI passes can shade and shadow it like any other
    // infinite light.
    // NOTE: this is an approximation - the sun's radiance is derived from the sky's FinalRadianceMultiplier
    // and the environment tint/intensity, but ignores the atmospheric transmittance toward the sun (a GPU
    // texture lookup done during the bake), so the magnitude is roughly ~1/transmittance too bright. It
    // also double-counts slightly with the sun already smeared into the IBL irradiance cube.
    if (!(m_ui.EnvironmentMapParams.Enabled && GetEnvMapBaker() != nullptr && GetEnvMapBaker()->IsProcedural() && GetEnvMapBaker()->GetProceduralSky() != nullptr))
        return;

    const ProceduralSkyConstants& sky = GetEnvMapBaker()->GetProceduralSky()->GetLastConstants();

    float3 finalRadiance = float3(sky.FinalRadianceMultiplier);
    if (sky.SunAngularDiameter <= 0.f || dm::length(finalRadiance) <= 1e-7f)
        return;

    // SunDir is the to-sun direction, but it lives in the procedural sky's Z-up frame (the sky shader
    // puts the observer at (0,0,r) so "up" is +Z). The env cube - and the engine world - are Y-up, and
    // the bake bridges the two with a Y<->Z swap (see EnvMapBaker.hlsl 'toLocal', which maps a cube-frame
    // direction (x,y,z) to sky-local (x,z,y)). Undo that swap first to bring SunDir into cube/world axes
    // (otherwise the sun's elevation ends up on the horizontal world-Z axis, so it appears to come from
    // the side instead of overhead).
    float3 sunDirCube = float3(sky.SunDir.x, sky.SunDir.z, sky.SunDir.y);

    // Then bring it from cube frame into world using the inverse of the env-map rotation (scene-graph
    // directionals are baked with the forward rotation, see Sample::UpdateLighting), and flip it to the
    // light travel direction.
    float3 rotationInRadians = dm::radians(m_ui.EnvironmentMapParams.RotationXYZ);
    dm::affine3 inverseTransform = dm::inverse(dm::rotation(rotationInRadians));
    float3 sunToLightWorld = dm::normalize(inverseTransform.transformVector(sunDirCube));
    float3 travelDirWorld = -sunToLightWorld;

    // Effective sun irradiance seen by the renderer = sky radiance * env tint * env intensity
    // (the c_envMapRadianceScale used during bake cancels against the sampling ColorMultiplier).
    float3 irradiance = finalRadiance * float3(m_ui.EnvironmentMapParams.TintColor) * m_ui.EnvironmentMapParams.Intensity;

    PolymorphicLightInfoFull sunLight = PackingUtils::MakeDirectionalLight(travelDirWorld, irradiance, sky.SunAngularDiameter);
    infiniteBase.push_back(sunLight.Base);
    infiniteEx.push_back(sunLight.Extended);
}

void IntroPathTracer::CreateIntroNeighborOffsetsBuffer()
{
    if (m_neighborOffsetsBuffer)
        return;

    nvrhi::BufferDesc neighborDesc;
    neighborDesc.byteSize = kIntroReSTIRNeighborOffsetCount * 2;
    neighborDesc.format = nvrhi::Format::RG8_SNORM;
    neighborDesc.canHaveTypedViews = true;
    neighborDesc.debugName = "IntroNeighborOffsets";
    neighborDesc.initialState = nvrhi::ResourceStates::ShaderResource;
    neighborDesc.keepInitialState = true;
    m_neighborOffsetsBuffer = GetDevice()->createBuffer(neighborDesc);
}

void IntroPathTracer::EnsureIntroReSTIRDIResources()
{
    const uint2 renderSize = GetRenderSize();
    const uint32_t width = renderSize.x;
    const uint32_t height = renderSize.y;
    if (width == 0 || height == 0)
        return;

    if (m_restirDIContext && m_lightReservoirBuffer)
    {
        const auto& staticParams = m_restirDIContext->GetStaticParameters();
        if (staticParams.RenderWidth == width && staticParams.RenderHeight == height)
            return;
    }

    rtxdi::ReSTIRDIStaticParameters staticParams;
    staticParams.RenderWidth = width;
    staticParams.RenderHeight = height;
    staticParams.NeighborOffsetCount = kIntroReSTIRNeighborOffsetCount;

    m_restirDIContext = std::make_unique<rtxdi::ReSTIRDIContext>(staticParams);
    // Set construction defaults; FillIntroReSTIRDIConstants applies the current UI settings
    // before uploading constants, including on the first frame.
    m_restirDIContext->SetResamplingMode(rtxdi::ReSTIRDI_ResamplingMode::TemporalAndSpatial);

    RTXDI_DIInitialSamplingParameters initialParams = m_restirDIContext->GetInitialSamplingParameters();
    initialParams.numLocalLightSamples = 8;
    initialParams.numInfiniteLightSamples = 1;
    initialParams.numEnvironmentSamples = 0;
    initialParams.numBrdfSamples = 0;
    initialParams.enableInitialVisibility = false;
    initialParams.environmentMapImportanceSampling = false;
    initialParams.localLightSamplingMode = static_cast<ReSTIRDI_LocalLightSamplingMode>(ReSTIRDI_LocalLightSamplingMode_UNIFORM);
    m_restirDIContext->SetInitialSamplingParameters(initialParams);

    RTXDI_ShadingParameters shadingParams = m_restirDIContext->GetShadingParameters();
    shadingParams.enableFinalVisibility = true;
    m_restirDIContext->SetShadingParameters(shadingParams);

    // Initialize temporal parameters. The live UI settings override these before dispatch;
    // the launch preset enables ray-traced bias correction and permutation sampling.
    RTXDI_DITemporalResamplingParameters temporalParams = m_restirDIContext->GetTemporalResamplingParameters();
    temporalParams.maxHistoryLength = 20;
    temporalParams.biasCorrectionMode = ReSTIRDI_TemporalBiasCorrectionMode::Basic;
    temporalParams.depthThreshold = 0.1f;
    temporalParams.normalThreshold = 0.5f;
    temporalParams.enableVisibilityShortcut = 0;
    temporalParams.enablePermutationSampling = 0;
    m_restirDIContext->SetTemporalResamplingParameters(temporalParams);

    // Spatial params: RayTraced bias correction. The target function is unshadowed, so the
    // normalization denominator (piSum) counts neighbor domains that could not actually have
    // produced the selected sample because it is occluded there. Under Basic/Pairwise that
    // overcount shrinks the reservoir weight and darkens contact/shadow-boundary regions -
    // a visibility (occlusion) bias that is present even in a static scene and is strongly
    // amplified by the spatial<->temporal reuse feedback loop. RayTraced mode traces a
    // shadow ray (RAB_GetConservativeVisibility) per contributing neighbor and zeroes its
    // ps when occluded, removing the bias. Modest neighbor count and radius.
    // numDisocclusionBoostSamples kicks in for low-M reservoirs (fresh disocclusions) and
    // pulls more neighbors to compensate for the missing history. Material similarity test
    // disabled - the Intro sample is simple enough that depth+normal gating is sufficient.
    RTXDI_DISpatialResamplingParameters spatialParams = m_restirDIContext->GetSpatialResamplingParameters();
    spatialParams.numSamples = 2;
    spatialParams.numDisocclusionBoostSamples = 8;
    spatialParams.targetHistoryLength = 12;
    spatialParams.biasCorrectionMode = ReSTIRDI_SpatialBiasCorrectionMode::Raytraced;
    spatialParams.samplingRadius = 32.0f;
    spatialParams.depthThreshold = 0.1f;
    spatialParams.normalThreshold = 0.5f;
    spatialParams.enableMaterialSimilarityTest = 0;
    spatialParams.discountNaiveSamples = 0;
    m_restirDIContext->SetSpatialResamplingParameters(spatialParams);

    const RTXDI_ReservoirBufferParameters reservoirParams = m_restirDIContext->GetReservoirBufferParameters();

    nvrhi::BufferDesc reservoirDesc;
    reservoirDesc.byteSize = sizeof(RTXDI_PackedDIReservoir) * reservoirParams.reservoirArrayPitch * rtxdi::c_NumReSTIRDIReservoirBuffers;
    reservoirDesc.structStride = sizeof(RTXDI_PackedDIReservoir);
    reservoirDesc.debugName = "IntroLightReservoirBuffer";
    reservoirDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    reservoirDesc.keepInitialState = true;
    reservoirDesc.canHaveUAVs = true;
    m_lightReservoirBuffer = GetDevice()->createBuffer(reservoirDesc);
    m_reSTIRHistoryValid = false;

    InvalidateBindingSet();
}

void IntroPathTracer::RefreshBindingSetIfNeeded()
{
    if (m_bindingSet != nullptr)
        return;

    if (!m_materialsBaker || !m_materialsBaker->GetMaterialDataBuffer())
        return;

    RecreateBindingSet();
}

void IntroPathTracer::InitializeIntroReSTIRNeighborOffsets(nvrhi::CommandListHandle commandList)
{
    if (m_neighborOffsetsInitialized || !m_neighborOffsetsBuffer)
        return;

    std::vector<uint8_t> offsets(kIntroReSTIRNeighborOffsetCount * 2);
    rtxdi::FillNeighborOffsetBuffer(offsets.data(), kIntroReSTIRNeighborOffsetCount);
    commandList->writeBuffer(m_neighborOffsetsBuffer, offsets.data(), offsets.size());

    m_neighborOffsetsInitialized = true;
}

void IntroPathTracer::FillIntroReSTIRDIConstants(nvrhi::CommandListHandle commandList)
{
    if (!m_introReSTIRConstBuffer || !commandList)
        return;

    // The buffer is volatile and the global binding set always references it (slot b12),
    // even when the active mode is not ReSTIR DI. NVRHI requires volatile constant
    // buffers to be written between command-list open() and any use, so we must always
    // emit a writeBuffer here. When the ReSTIR DI context doesn't exist yet, send a
    // zero-initialized block - shaders in other modes won't read it anyway.
    IntroReSTIRDIConstants constants{};
    bool historyValid = m_reSTIRHistoryValid && m_lightLayoutMatchesPrevious;
    if (m_restirDIContext)
    {
        // Honor the UI toggles every frame. The context recomputes the reservoir
        // buffer ping-pong indices internally when the mode changes, so the
        // Initial / Temporal / Spatial / Final dispatches below always see consistent
        // shadingInputBufferIndex / temporalResamplingInputBufferIndex /
        // spatialResamplingInputBufferIndex triples.
        rtxdi::ReSTIRDI_ResamplingMode desiredMode = rtxdi::ReSTIRDI_ResamplingMode::None;
        if (m_introUIData.IntroReSTIRTemporalReuse && m_introUIData.IntroReSTIRSpatialReuse)
            desiredMode = rtxdi::ReSTIRDI_ResamplingMode::TemporalAndSpatial;
        else if (m_introUIData.IntroReSTIRTemporalReuse)
            desiredMode = rtxdi::ReSTIRDI_ResamplingMode::Temporal;
        else if (m_introUIData.IntroReSTIRSpatialReuse)
            desiredMode = rtxdi::ReSTIRDI_ResamplingMode::Spatial;

        if (m_restirDIContext->GetResamplingMode() != desiredMode)
        {
            m_restirDIContext->SetResamplingMode(desiredMode);
            historyValid = false;
        }

        // Push live UI tuning into the context so shaders observe the latest values.
        // Most other fields were locked in at construction (see EnsureIntroReSTIRDIResources).
        RTXDI_DITemporalResamplingParameters temporal = m_restirDIContext->GetTemporalResamplingParameters();
        temporal.maxHistoryLength = static_cast<uint32_t>(std::max(1, m_introUIData.IntroReSTIRMaxHistoryLength));
        temporal.depthThreshold = m_introUIData.IntroReSTIRDepthThreshold;
        temporal.normalThreshold = m_introUIData.IntroReSTIRNormalThreshold;
        temporal.enablePermutationSampling = m_introUIData.IntroReSTIREnablePermutationSampling ? 1u : 0u;
        temporal.biasCorrectionMode = m_introUIData.IntroReSTIRTemporalBiasRaytraced
            ? ReSTIRDI_TemporalBiasCorrectionMode::Raytraced
            : ReSTIRDI_TemporalBiasCorrectionMode::Basic;
        m_restirDIContext->SetTemporalResamplingParameters(temporal);

        RTXDI_DISpatialResamplingParameters spatial = m_restirDIContext->GetSpatialResamplingParameters();
        spatial.samplingRadius = m_introUIData.IntroReSTIRSpatialSamplingRadius;
        spatial.numSamples = static_cast<uint32_t>(std::max(1, m_introUIData.IntroReSTIRSpatialNumSamples));
        spatial.depthThreshold = m_introUIData.IntroReSTIRDepthThreshold;
        spatial.normalThreshold = m_introUIData.IntroReSTIRNormalThreshold;
        spatial.discountNaiveSamples = m_introUIData.IntroReSTIRDiscountNaiveSamples ? 1u : 0u;
        m_restirDIContext->SetSpatialResamplingParameters(spatial);

        // Boiling filter runs inside the Temporal pass (see IntroReSTIRDI_Temporal.hlsl).
        RTXDI_BoilingFilterParameters boiling = m_restirDIContext->GetBoilingFilterParameters();
        boiling.enableBoilingFilter = m_introUIData.IntroReSTIREnableBoilingFilter ? 1u : 0u;
        boiling.boilingFilterStrength = m_introUIData.IntroReSTIRBoilingFilterStrength;
        m_restirDIContext->SetBoilingFilterParameters(boiling);

        m_restirDIContext->SetFrameIndex(m_frameIndex);

        constants.runtimeParams = m_restirDIContext->GetRuntimeParams();
        constants.restirDI.reservoirBufferParams = m_restirDIContext->GetReservoirBufferParameters();
        constants.restirDI.bufferIndices = m_restirDIContext->GetBufferIndices();
        // No temporal pass runs, so shading must read the initial-sampling buffer.
        if (m_restirDIContext->GetResamplingMode() == rtxdi::ReSTIRDI_ResamplingMode::None)
        {
            constants.restirDI.bufferIndices.shadingInputBufferIndex = constants.restirDI.bufferIndices.initialSamplingOutputBufferIndex;
        }
        constants.restirDI.initialSamplingParams = m_restirDIContext->GetInitialSamplingParameters();
        constants.restirDI.temporalResamplingParams = m_restirDIContext->GetTemporalResamplingParameters();
        constants.restirDI.boilingFilterParams = m_restirDIContext->GetBoilingFilterParameters();
        constants.restirDI.spatialResamplingParams = m_restirDIContext->GetSpatialResamplingParameters();
        constants.restirDI.shadingParams = m_restirDIContext->GetShadingParameters();
    }

    // Initial-pass candidate counts are driven directly by the UI (independent per stream).
    constants.numAnalyticLocalLightSamples = static_cast<uint32_t>(std::max(0, m_introUIData.IntroReSTIRNumAnalyticLocalSamples));
    constants.numTriangleLocalLightSamples = static_cast<uint32_t>(std::max(0, m_introUIData.IntroReSTIRNumTriangleLocalSamples));
    constants.numInfiniteLightSamples      = static_cast<uint32_t>(std::max(0, m_introUIData.IntroReSTIRNumInfiniteSamples));
    // Environment-map candidates are only drawn when the UI enables it. The shader additionally gates on
    // lightControl.EnvironmentLightPresent (set by PrepareLights), so a stale count can never sample a
    // missing env light.
    constants.numEnvironmentSamples        = m_introUIData.IntroReSTIRSampleEnvironmentMap
        ? static_cast<uint32_t>(std::max(0, m_introUIData.IntroReSTIRNumEnvironmentSamples)) : 0u;

    constants.lightRegionCapacity = m_RTXDILightCapacity;
    constants.lightCount = m_currentConstants.lightControl.InfiniteRegionNumLights
        + m_currentConstants.lightControl.LocalRegionNumLights
        + m_currentConstants.lightControl.EnvironmentLightPresent;
    const std::array<uint32_t, 4> candidateCounts = { constants.numAnalyticLocalLightSamples,
        constants.numTriangleLocalLightSamples, constants.numInfiniteLightSamples, constants.numEnvironmentSamples };
    // A disabled emitter/environment stream must not survive in history while the indirect
    // tracer resumes counting those connections. Reset on candidate-budget edits as well.
    constants.temporalHistoryValid = historyValid && candidateCounts == m_previousReSTIRCandidateCounts ? 1u : 0u;
    m_previousReSTIRCandidateCounts = candidateCounts;

    // Only a completed ReSTIR pass below can make history available to the next frame.
    // Other direct-light modes, reference rendering and unavailable pipelines break reuse.
    m_reSTIRHistoryValid = false;

    commandList->writeBuffer(m_introReSTIRConstBuffer, &constants, sizeof(constants));
}

dm::int2 IntroPathTracer::GetViewportComputeDispatchSize() const
{
    nvrhi::Viewport viewport = GetView().GetViewport();
    const uint32_t width = (uint32_t)(viewport.maxX - viewport.minX);
    const uint32_t height = (uint32_t)(viewport.maxY - viewport.minY);
    return { div_ceil(width, NUM_COMPUTE_THREADS_PER_DIM), div_ceil(height, NUM_COMPUTE_THREADS_PER_DIM) };
}

void IntroPathTracer::DispatchComputePassState(
    nvrhi::CommandListHandle commandList,
    nvrhi::IComputePipeline* pipeline,
    const SampleMiniConstants& miniConstants) const
{
    nvrhi::ComputeState state;
    state.bindings = { m_bindingSet, m_DescriptorTable->GetDescriptorTable() };
    state.pipeline = pipeline;
    commandList->setComputeState(state);
    commandList->setPushConstants(&miniConstants, sizeof(miniConstants));

    const dm::int2 dispatchSize = GetViewportComputeDispatchSize();
    commandList->dispatch(dispatchSize.x, dispatchSize.y);
}

bool IntroPathTracer::DispatchComputePass(
    nvrhi::CommandListHandle commandList,
    const std::shared_ptr<ComputeShaderVariant>& variant,
    const SampleMiniConstants& miniConstants,
    const char* perfMarkerName) const
{
    nvrhi::IComputePipeline* pipeline = variant ? variant->GetPipeline() : nullptr;
    if (!pipeline)
        return false;

    if (perfMarkerName)
    {
        ScopedPerfMarker perfMarker(perfMarkerName, commandList);
        DispatchComputePassState(commandList, pipeline, miniConstants);
    }
    else
    {
        DispatchComputePassState(commandList, pipeline, miniConstants);
    }

    return true;
}

void IntroPathTracer::ReSTIRDIDirectLightingPass(nvrhi::CommandListHandle commandList, uint sharcCapacity)
{
    (void)sharcCapacity;

    if (!m_restirDIContext || !m_lightReservoirBuffer)
        return;

    RefreshBindingSetIfNeeded();
    if (!m_bindingSet)
        return;

    InitializeIntroReSTIRNeighborOffsets(commandList);

    commandList->setBufferState(m_lightReservoirBuffer, nvrhi::ResourceStates::UnorderedAccess);

    const SampleMiniConstants miniConstants = { uint4(0, 0, 0, 0) };
    const rtxdi::ReSTIRDI_ResamplingMode mode = m_restirDIContext->GetResamplingMode();

    // Gather initial candidates per pixel and initialize reservoirs.
    if (!DispatchComputePass(commandList, m_reSTIRDIInitialVariant, miniConstants, "Intro ReSTIR DI Initial"))
        return;

    nvrhi::utils::BufferUavBarrier(commandList, m_lightReservoirBuffer);

    // Temporal resampling reads {current initial reservoir, previous frame temporal
    // reservoir, previous-frame G-buffer history} and writes the temporally combined reservoir.
    const bool runTemporal =
        (mode == rtxdi::ReSTIRDI_ResamplingMode::Temporal ||
         mode == rtxdi::ReSTIRDI_ResamplingMode::TemporalAndSpatial);

    if (runTemporal)
    {
        if (!DispatchComputePass(commandList, m_reSTIRDITemporalVariant, miniConstants, "Intro ReSTIR DI Temporal"))
            return;
        nvrhi::utils::BufferUavBarrier(commandList, m_lightReservoirBuffer);
    }

    // Spatial resampling combines the per-pixel reservoir with neighbors whose
    // depth+normal match (via t_NeighborOffsets). Reads spatialResamplingInputBufferIndex
    // which the SDK has pointed at either the temporal output (when temporal also runs)
    // or the initial output (spatial-only mode).
    const bool runSpatial =
        (mode == rtxdi::ReSTIRDI_ResamplingMode::Spatial ||
         mode == rtxdi::ReSTIRDI_ResamplingMode::TemporalAndSpatial);

    if (runSpatial)
    {
        if (!DispatchComputePass(commandList, m_reSTIRDISpatialVariant, miniConstants, "Intro ReSTIR DI Spatial"))
            return;
        nvrhi::utils::BufferUavBarrier(commandList, m_lightReservoirBuffer);
    }

    // Final shading uses shadingInputBufferIndex, which points at the Initial (None),
    // Temporal (Temporal), Spatial (Spatial), or Spatial-after-Temporal (TemporalAndSpatial)
    // output depending on the active mode.
    m_reSTIRHistoryValid = DispatchComputePass(commandList, m_reSTIRDIFinalVariant, miniConstants, "Intro ReSTIR DI Final")
        && m_previousLightsReady;
}

void IntroPathTracer::DirectLightingPass(nvrhi::CommandListHandle commandList, uint sharcCapacity)
{
    ScopedPerfMarker perfMarker("Direct Lighting", commandList);

    if (m_ui.IntroDirectLighting == IntroDirectLightingMode::ReSTIRDI)
    {
        ReSTIRDIDirectLightingPass(commandList, sharcCapacity);
        return;
    }

    std::shared_ptr<ComputeShaderVariant> variant = m_directLightingVariant;
    if (m_ui.IntroDirectLighting == IntroDirectLightingMode::UniformSampling)
        variant = m_stochasticLightingVariant;
    else if (m_ui.IntroDirectLighting == IntroDirectLightingMode::StreamingRIS)
        variant = m_RTXDIStreamingRISVariant;

    const SampleMiniConstants miniConstants = { uint4(sharcCapacity, 0, 0, 0) };
    DispatchComputePass(commandList, variant, miniConstants);
}

void IntroPathTracer::DeferredLightingCombine(nvrhi::CommandListHandle commandList, bool useRTReflections)
{
    ScopedPerfMarker perfMarker("Deferred Lighting", commandList);

    // Debug RTGI: replace deferred lighting with the raw RTGI ray output (irradiance, pre-albedo).
    // Gated on UseRTGI rather than the effective useRTGI local, matching the UI checkbox's own
    // visibility rule: MIS supersedes RTGI, so the toggle (and this override) only make sense
    // when MIS isn't driving indirect lighting instead.
    const bool useRTGIDebug = m_ui.DebugRTGI && m_ui.UseRTGI && !m_ui.UseMISIndirect && m_rtgiDebugOutputVariant;
    if (useRTGIDebug)
    {
        const SampleMiniConstants miniConstants = { uint4(0, 0, 0, 0) };
        DispatchComputePass(commandList, m_rtgiDebugOutputVariant, miniConstants);
        return;
    }

    // Debug SHaRC cache: replace deferred lighting with cache radiance visualization
    const bool useSHaRCDebug = m_ui.DebugSHaRCCache && m_ui.UseSHaRC && m_sharcRenderer && m_sharcRenderer->IsUpdateReady() && m_sharcDebugCacheVariant;
    if (useSHaRCDebug)
    {
        SampleMiniConstants miniConstants = {};
        miniConstants.params.x = m_sharcRenderer->GetCapacity();
        DispatchComputePass(commandList, m_sharcDebugCacheVariant, miniConstants);
        return;
    }

    // Pick the deferred lighting variant (compile-time specular source):
    // - direct read (u_SSRResult UAV, no blur chain): MIS indirect AND RT reflections both write fully-shaded
    //   specular directly and rely on DLSS-RR to denoise
    // - default (SSR blur chain): screen-space reflections
    const bool useDirectSpecular = (m_ui.UseMISIndirect || useRTReflections) && m_directSpecularLightingVariant;
    const std::shared_ptr<ComputeShaderVariant>& variant = useDirectSpecular ? m_directSpecularLightingVariant
                                                                             : m_deferredLightingVariant;

    // params.x flags: bit 0 selects ray-traced GI (u_RTGIOutput) instead of the environment SH
    // irradiance cube; bit 1 isolates indirect specular. RT reflections must NOT force RTGI diffuse.
    const uint32_t useRTGIDiffuse = (m_ui.UseMISIndirect || m_ui.UseRTGI) ? 1u : 0u;
    const uint32_t lightingFlags = useRTGIDiffuse | (m_ui.DebugIndirectSpecular ? 2u : 0u);

    // params.y: when ReSTIR DI is importance-sampling the environment as (shadowed) direct light, drop the
    // env-irradiance IBL diffuse fallback so the environment's diffuse direct lighting isn't counted twice
    // (once by the ReSTIR env stream, once by the unshadowed irradiance cube). Only relevant on the env-cube
    // fallback path (params.x bit 0 clear); when RTGI/MIS drives diffuse indirect there is no env-IBL term to drop.
    const uint32_t suppressEnvIrradiance =
        (m_ui.IntroDirectLighting == IntroDirectLightingMode::ReSTIRDI
         && m_currentConstants.lightControl.EnvironmentLightPresent != 0
         && m_introUIData.IntroReSTIRNumEnvironmentSamples > 0) ? 1u : 0u;

    // params.z/.w carry max-mip indices read by CombineIndirectLight.
    const uint32_t ssrMaxMip = m_depthHierarchyRenderer ? (m_depthHierarchyRenderer->GetNumMips() - 1u) : 0u;
    const SampleMiniConstants miniConstants = { uint4(lightingFlags, suppressEnvIrradiance, kEnvGGXMaxMip, ssrMaxMip) };
    DispatchComputePass(commandList, variant, miniConstants);
}

void IntroPathTracer::PrepareGuideBuffers(nvrhi::CommandListHandle commandList)
{
    ScopedPerfMarker perfMarker("PrepareGuideBuffers", commandList);

    const SampleMiniConstants miniConstants = { uint4(0, 0, 0, 0) };
    DispatchComputePass(commandList, m_prepareGuideBuffersVariant, miniConstants);
}

void IntroPathTracer::CopyGBufferHistory(nvrhi::CommandListHandle commandList)
{
    // Snapshot the current-frame G-buffer textures needed by next-frame temporal reuse
    // (depth for disocclusion checks, packed shading normal for ReSTIR DI surface match).
    if (!m_renderTargets)
        return;

    ScopedPerfMarker perfMarker("GBuffer History Copy", commandList);
    if (m_prevDepth && m_renderTargets->Depth)
        commandList->copyTexture(m_prevDepth, nvrhi::TextureSlice(), m_renderTargets->Depth, nvrhi::TextureSlice());
    if (m_prevSpecNormal && m_renderTargets->SpecNormal)
        commandList->copyTexture(m_prevSpecNormal, nvrhi::TextureSlice(), m_renderTargets->SpecNormal, nvrhi::TextureSlice());
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
    Sample::CreateRTPipelines();

    auto pipelineBaker = GetRTPipelineBaker();

    auto pipelineMacros = std::vector<donut::engine::ShaderMacro>{ {"RTXPT_HIT_SHADER_SET", "RTXPT_HIT_SET_SIMPLIFIED"} };
    m_simplifiedRTPipelines = std::make_shared<PTPipelineBaker>(GetDevice(), m_materialsBaker, m_bindingLayout, m_bindlessLayout,
        [this](std::vector<donut::engine::ShaderMacro>& macros) { this->GetPTPipelineGlobalMacros(macros); }, nullptr, pipelineMacros);

    // these don't actually compile any shaders - this happens later in m_ptPipelineBaker->Update
    m_GBufferPipeline = pipelineBaker->CreateVariant("IntroSample/PopulateGBuffer.hlsl", {}, "GBuf");
    m_realTimePathTracer = pipelineBaker->CreateVariant("IntroSample/IntroPathTracer.hlsl", {}, "Intro");
    m_reflectionRenderer->CreateRTPipeline(*m_simplifiedRTPipelines);
    m_rtgiRenderer->CreateRTPipelines(*m_simplifiedRTPipelines);
    m_indirectPathTracer->CreateRTPipelines(*m_simplifiedRTPipelines);
    m_sharcRenderer->CreateRTPipeline(*m_simplifiedRTPipelines);

    m_sharcRenderer->CreateBuffers();
}

void IntroPathTracer::CreateCustomRenderPasses()
{
    // All these passes only need to be created once. So early out if they exist already. Just need to check one
    if (m_deferredLightingVariant)
        return;

    auto computeBaker = GetComputePipelineBaker();

    // Direct Lighting
    m_directLightingVariant = computeBaker->CreateVariant("IntroSample/DirectLightingPass.hlsl", "main", {}, { m_bindingLayout, m_bindlessLayout }, "DirectLighting");
    m_stochasticLightingVariant = computeBaker->CreateVariant("IntroSample/StochasticLightingPass.hlsl", "main", {}, { m_bindingLayout, m_bindlessLayout }, "StochasticLighting");
    m_RTXDIStreamingRISVariant = computeBaker->CreateVariant("IntroSample/RTXDIStreamingRISLightingPass.hlsl", "main", {}, { m_bindingLayout, m_bindlessLayout }, "IntroRTXDIStreamingRIS");
    m_reSTIRDIInitialVariant  = computeBaker->CreateVariant("IntroSample/IntroReSTIRDI_Initial.hlsl",  "main", {}, { m_bindingLayout, m_bindlessLayout }, "IntroReSTIRDIInitial");
    m_reSTIRDITemporalVariant = computeBaker->CreateVariant("IntroSample/IntroReSTIRDI_Temporal.hlsl", "main", {}, { m_bindingLayout, m_bindlessLayout }, "IntroReSTIRDITemporal");
    m_reSTIRDISpatialVariant  = computeBaker->CreateVariant("IntroSample/IntroReSTIRDI_Spatial.hlsl",  "main", {}, { m_bindingLayout, m_bindlessLayout }, "IntroReSTIRDISpatial");
    m_reSTIRDIFinalVariant    = computeBaker->CreateVariant("IntroSample/IntroReSTIRDI_Final.hlsl",    "main", {}, { m_bindingLayout, m_bindlessLayout }, "IntroReSTIRDIFinal");

    // Emissive-triangle bake (GPU PrepareLights). Uses a dedicated binding layout (scene geometry +
    // SoA light UAVs) plus the global bindless layout for vertex/index/texture access.
    EnsureIntroPrepareLightsResources(1);
    m_introPrepareLightsVariant = computeBaker->CreateVariant("IntroSample/IntroPrepareLights.hlsl", "main", {}, { m_introPrepareLightsBindingLayout, m_bindlessLayout }, "IntroPrepareLights");

    // Deferred Lighting Combine. Single shader with two compile variants (specular source):
    //   - default:  specular from the SSR blur chain
    //   - MIS:      specular from the u_SSRResult UAV
    // The diffuse indirect source (RTGI vs environment cube) is selected at runtime via params.x.
    m_deferredLightingVariant = computeBaker->CreateVariant("IntroSample/DeferredLightingCombine.hlsl", "main", { {"DIRECT_SPECULAR_READ","0"} }, { m_bindingLayout, m_bindlessLayout }, "DeferredLightingCombine");

    m_directSpecularLightingVariant = computeBaker->CreateVariant("IntroSample/DeferredLightingCombine.hlsl", "main", { {"DIRECT_SPECULAR_READ","1"} }, { m_bindingLayout, m_bindlessLayout }, "DirectSpecularLighting");

    // DLSS Guide buffers
    m_prepareGuideBuffersVariant = computeBaker->CreateVariant("IntroSample/PrepareGuideBuffers.hlsl", "main", {}, { m_bindingLayout, m_bindlessLayout }, "PrepareGuideBuffers");

    // RTGI Debug
    m_rtgiDebugOutputVariant = computeBaker->CreateVariant("IntroSample/RTGIDebugOutput.hlsl", "main", {}, { m_bindingLayout, m_bindlessLayout }, "RTGIDebugOutput");

    // SHaRC Debug
    m_sharcDebugCacheVariant = computeBaker->CreateVariant("IntroSample/SHaRCDebugCache.hlsl", "main", {}, { m_bindingLayout, m_bindlessLayout }, "SHaRCDebugCache");

    // Depth hierarchy
    m_depthHierarchyRenderer->CreatePipelines(*computeBaker);

    // GTAO
    m_aoRenderer->CreatePipelines(m_bindingLayout, m_bindlessLayout, *computeBaker);

    // SSR / reflection system
    m_reflectionRenderer->CreatePipelines(*computeBaker);

    // SHaRC
    m_sharcRenderer->CreatePipelines(m_bindingLayout, m_bindlessLayout, *computeBaker);
}
    
void IntroPathTracer::DestroyRTPipelines()
{
    auto pipelineBaker = GetRTPipelineBaker();
    auto computeBaker = GetComputePipelineBaker();
    assert(pipelineBaker);
    assert(computeBaker);
    
    if (m_depthHierarchyRenderer)
    {
        m_depthHierarchyRenderer->DestroyPipelines(*computeBaker);
    }

    if (m_aoRenderer)
    {
        m_aoRenderer->DestroyPipelines(*computeBaker);
    }

    if (m_reflectionRenderer)
    {
        m_reflectionRenderer->DestroyRTPipeline(*m_simplifiedRTPipelines);
        m_reflectionRenderer->DestroyPipelines(*computeBaker);
    }

    if (m_sharcRenderer)
    {
        m_sharcRenderer->DestroyRTPipeline(*m_simplifiedRTPipelines);
        m_sharcRenderer->DestroyPipelines(*computeBaker);
    }

    computeBaker->ReleaseVariant(m_directLightingVariant);
    computeBaker->ReleaseVariant(m_stochasticLightingVariant);
    computeBaker->ReleaseVariant(m_RTXDIStreamingRISVariant);
    computeBaker->ReleaseVariant(m_reSTIRDIInitialVariant);
    computeBaker->ReleaseVariant(m_reSTIRDITemporalVariant);
    computeBaker->ReleaseVariant(m_reSTIRDISpatialVariant);
    computeBaker->ReleaseVariant(m_reSTIRDIFinalVariant);
    computeBaker->ReleaseVariant(m_introPrepareLightsVariant);
    computeBaker->ReleaseVariant(m_deferredLightingVariant);
    computeBaker->ReleaseVariant(m_directSpecularLightingVariant);
    computeBaker->ReleaseVariant(m_prepareGuideBuffersVariant);
    computeBaker->ReleaseVariant(m_rtgiDebugOutputVariant);
    computeBaker->ReleaseVariant(m_sharcDebugCacheVariant);
    
    pipelineBaker->ReleaseVariant(m_realTimePathTracer);
    pipelineBaker->ReleaseVariant(m_GBufferPipeline);

    // Simplified material pipelines
    m_rtgiRenderer->DestroyRTPipelines(*m_simplifiedRTPipelines);
    m_indirectPathTracer->DestroyRTPipelines(*m_simplifiedRTPipelines);

    Sample::DestroyRTPipelines();
}

std::string IntroPathTracer::GetMaterialSpecializationShader() const
{
    return "IntroSample/IntroHitShaders.hlsl";
}

std::unique_ptr<SampleUI> IntroPathTracer::CreateUI(donut::app::DeviceManager* deviceManager, class SampleBaseApp& baseApp, SampleUIData& uiData, bool NVAPI_SERSupported, const struct CommandLineOptions& cmdLine)
{
    auto sampleUI = std::make_unique<IntroSampleUI>(deviceManager, baseApp, *this, uiData, NVAPI_SERSupported, cmdLine);

    // Intro sample overrides of unsupported functionality
    uiData.NEEType = 0; // No NEEAT in the intro sample
    uiData.UseReSTIRDI = false; // Advanced Sample's integration
    uiData.UseReSTIRGI = false; // Advanced Sample's integration

    // Re-apply the launch step preset AFTER the base SampleUI ctor's performance preset has run
    // (it would otherwise clobber RealtimeAA). This is what selects the step's recommended AA mode.
    // kIntroDefaultStep (IntroSampleUI.h) remains the single source of truth for the launch step.
    m_introUIData.IntroStepIndex = kIntroDefaultStep;
    ApplyIntroStepPreset(uiData, kIntroDefaultStep);

#if DONUT_WITH_STREAMLINE
    // Preselect DLAA (full-resolution) so that if the user later steps up to a DLSS/DLSS-RR preset it
    // launches at full quality. Must be set here, after the base SampleUI ctor's performance preset
    // (which would otherwise reset DLSSMode to its Balanced default). Inert at Step 0, which uses TAA.
    uiData.DLSSMode = SI::DLSSMode::eDLAA;
#endif

    return sampleUI;
}
    
void IntroPathTracer::SceneLoaded()
{
    m_previousLightsReady = false;
    m_reSTIRHistoryValid = false;
    Sample::SceneLoaded();

    AllocateEnvIBLTextures();
}

void IntroPathTracer::OnRenderTargetsRecreated()
{
    uint2 renderSize = GetRenderSize();

    // Recreate previous-frame G-buffer history textures to match new resolution.
    // Both are full-res copies of current-frame counterparts taken at the end of the
    // frame via CopyGBufferHistory; ReSTIR DI temporal resampling reads them as SRVs.
    nvrhi::TextureDesc desc;
    desc.width = renderSize.x;
    desc.height = renderSize.y;
    desc.isUAV = false;
    desc.initialState = nvrhi::ResourceStates::ShaderResource;
    desc.keepInitialState = true;

    desc.format = nvrhi::Format::R32_FLOAT;
    desc.debugName = "PrevDepth";
    m_prevDepth = GetDevice()->createTexture(desc);

    desc.format = nvrhi::Format::R32_UINT;     // matches RenderTargets::SpecNormal (octahedral-packed)
    desc.debugName = "PrevSpecNormal";
    m_prevSpecNormal = GetDevice()->createTexture(desc);

    m_depthHierarchyRenderer->OnRenderTargetsRecreated(renderSize.x, renderSize.y);
    m_aoRenderer->OnRenderTargetsRecreated(renderSize.x, renderSize.y);
    m_reflectionRenderer->OnRenderTargetsRecreated(renderSize.x, renderSize.y);
    m_rtgiRenderer->OnRenderTargetsRecreated(renderSize.x, renderSize.y);
    m_indirectPathTracer->OnRenderTargetsRecreated(renderSize.x, renderSize.y);

    m_lightReservoirBuffer = nullptr;
    m_restirDIContext.reset();
    m_reSTIRHistoryValid = false;

    // Defer binding recreation until after materials are loaded (needNewPasses path in Sample::Render).
    InvalidateBindingSet();
}
    
// Override to add custom textures to the main binding set
void IntroPathTracer::AddCustomBindings(nvrhi::BindingSetDesc& bindingSetDesc)
{
    // Replace null placeholder bindings with actual textures.
    // Note: Depth hierarchy UAVs (u80-84) and constant buffer (b11) are in a dedicated
    // binding set, not the global set, to avoid duplicate barriers with t_DepthHierarchy SRV.

    // Make sure the Intro RTXDI light buffer exists before the binding set is created.
    // PrepareLights will resize it on subsequent frames as the light counts change
    // (and call RecreateBindingSet so we land back here). The buffer holds the infinite region
    // (directional lights), the local analytic region (positional lights) and the baked emissive
    // triangles contiguously ([infinite | local analytic | triangles]), so size it for the combined count.
    uint32_t analyticLightCount = 0;
    for (const std::shared_ptr<donut::engine::Light>& light : m_lights)
    {
        if (light && (PackingUtils::IsAnalyticLight(*light) || PackingUtils::IsInfiniteAnalyticLight(*light)))
            analyticLightCount++;
    }
    // Account for baked emissive triangles too, so the buffer is sized for A + T + I and
    // PrepareLights doesn't have to resize (and rebuild the binding set) on the first bake.
    std::vector<PrepareLightsTask> triangleTasks;
    const uint32_t triangleCount = BuildIntroEmissiveTriangleTasks(triangleTasks);
    EnsureIntroRTXDIResources(analyticLightCount + triangleCount);

    const bool envIBLReady = (m_envGGXFilteredCube != nullptr);

    for (auto& binding : bindingSetDesc.bindings)
    {
        if (binding.type == nvrhi::ResourceType::Texture_SRV)
        {
            switch (binding.slot)
            {
                // Environment IBL / reflection textures (only when the IBL cubes have been allocated)
                case 80: if (envIBLReady) binding.resourceHandle = m_envGGXFilteredCube; break;
                case 81: if (envIBLReady) binding.resourceHandle = m_envIrradianceCube; break;
                case 82: if (envIBLReady) binding.resourceHandle = m_reflectionRenderer->GetSSRBlurMipChain(); break;
                case 84: if (envIBLReady) binding.resourceHandle = m_depthHierarchyRenderer->m_depthHierarchy; break;

                // Always-available textures
                case 87:
                    if (m_prevDepth)
                        binding.resourceHandle = m_prevDepth;
                    break;
                case 95:
                    if (m_prevSpecNormal)
                        binding.resourceHandle = m_prevSpecNormal;
                    break;
            }
        }
        else if (binding.type == nvrhi::ResourceType::Texture_UAV)
        {
            switch (binding.slot)
            {
                case 85: if (envIBLReady) binding.resourceHandle = m_reflectionRenderer->GetSSRResult(); break;
                case 88: binding.resourceHandle = m_renderTargets->DirectLighting; break;
                case 89: if (m_rtgiRenderer && m_rtgiRenderer->GetOutput()) binding.resourceHandle = m_rtgiRenderer->GetOutput(); break;
                case 95: if (m_indirectPathTracer && m_indirectPathTracer->GetOutput()) binding.resourceHandle = m_indirectPathTracer->GetOutput(); break;  // u_IndirectOutput (moved from u93 when SHaRC took u90-u93); u94 is u_LightReservoirs, a StructuredBuffer_UAV, so it never matched this Texture_UAV switch
                case 100: binding.resourceHandle = m_renderTargets->BaseColor; break;
                case 101: binding.resourceHandle = m_renderTargets->SpecNormal; break;
                case 102: binding.resourceHandle = m_renderTargets->RoughnessMetal; break;
                case 103: binding.resourceHandle = m_renderTargets->MaterialInfo; break;
            }
        }
        else if (binding.type == nvrhi::ResourceType::StructuredBuffer_UAV)
        {
            switch (binding.slot)
            {
                case 90: if (m_sharcRenderer && m_sharcRenderer->GetHashEntries()) binding.resourceHandle = m_sharcRenderer->GetHashEntries(); break;
                case 91: if (m_sharcRenderer && m_sharcRenderer->GetAccumulation()) binding.resourceHandle = m_sharcRenderer->GetAccumulation(); break;
                case 92: if (m_sharcRenderer && m_sharcRenderer->GetResolved()) binding.resourceHandle = m_sharcRenderer->GetResolved(); break;
                case 94: if (m_lightReservoirBuffer) binding.resourceHandle = m_lightReservoirBuffer; break;
            }
        }
        else if (binding.type == nvrhi::ResourceType::StructuredBuffer_SRV)
        {
            switch (binding.slot)
            {
                case 90: if (m_RTXDILightsBase) binding.resourceHandle = m_RTXDILightsBase; break;
                case 91: if (m_RTXDILightsEx)   binding.resourceHandle = m_RTXDILightsEx;   break;
            }
        }
        else if (binding.type == nvrhi::ResourceType::TypedBuffer_SRV)
        {
            switch (binding.slot)
            {
                case 94:
                    if (m_neighborOffsetsBuffer)
                        binding.resourceHandle = m_neighborOffsetsBuffer;
                    break;
            }
        }
        else if (binding.type == nvrhi::ResourceType::ConstantBuffer ||
                 binding.type == nvrhi::ResourceType::VolatileConstantBuffer)
        {
            // BindingSetItem::ConstantBuffer() picks VolatileConstantBuffer when the
            // placeholder buffer was created with isVolatile = true (which Sample.cpp's
            // m_constantBuffer is). We must accept both types here, otherwise the slot
            // silently keeps the placeholder and the shader reads SampleConstants bytes
            // as IntroReSTIRDIConstants.
            if (binding.slot == 12 && m_introReSTIRConstBuffer)
                binding.resourceHandle = m_introReSTIRConstBuffer;
        }
    }
}

class IntroSample : public SampleBaseApp
{
    void ConfigureDefaultDeviceParams(donut::app::DeviceCreationParameters& deviceParams) const override
    {
#if DONUT_WITH_AFTERMATH
        deviceParams.enableAftermath = true;
        deviceParams.enableDebugRuntime = false;
        deviceParams.enableGPUValidation = false;
#endif
    }

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
