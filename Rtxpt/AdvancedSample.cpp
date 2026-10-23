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
#include "AdvancedSampleUI.h"
#include "ProcessingPasses/DenoisingGuidesBaker.h"
#include "Sharc/SharcIntegration.h"

#include <SampleCommon/AdaptiveSampling.h>
#include <SampleCommon/PTPipelineBaker.h>
#include <SampleCommon/PathTracePerfLog.h>
#include <SampleCommon/SampleBaseApp.h>
#include <SampleCommon/SplashScreen.h>

using namespace donut::engine;

// Full-featured path tracer used by the Rtxpt sample. Owns the
// pipelines that are specific to the advanced path-tracer (reference + stable
// planes);
class AdvancedPathTracer : public Sample
{
public:
    using Sample::Sample;

    virtual const char* GetWindowTitle() override { return "RTX Path Tracing"; };   // future: change to UnifiedSample?

    virtual void SampleRenderCode(nvrhi::IFramebuffer* framebuffer, nvrhi::CommandListHandle commandList, const SampleConstants& constants, bool needNewPasses) override
    {
        if (m_ui.ActualUseRTXDIPasses())
            m_rtxdiPass->BeginFrame(commandList, *m_renderTargets, m_bindingLayout, m_bindingSet);

        PathTrace(framebuffer, constants);

        Denoise(framebuffer);
    }

    // Per-variant macros for m_ptPipelineFillDenoisingLayers. Re-evaluated on every PTPipelineBaker::Update and only
    // triggers a recompile when the produced set actually changes. Sub-sampling lives here rather than in the global
    // set because only this variant reads these macros (the SHaRC update pass supplies its own); keeping them global
    // made the REF and BUILD variants recompile whenever the sub-sampling mode changed, for macros they never use.
    void GetLocalMacros_FillDenoisingLayers(std::vector<donut::engine::ShaderMacro>& macros)
    {
        AdaptiveSampling::GetFillPassMacros(m_ui, macros);

        if (m_sharc && m_ui.ActualRadianceCacheMode() == 1)
        {
            macros.push_back({ "RTXPT_SHARC_CONSUME", "1" });

            macros.push_back({ "RTXPT_SHARC_ENABLE_DEBUG", (m_sharc && m_sharc->EnableDebug()) ? "1" : "0" });
        }

    }

    virtual void CreateRTPipelines() override
    {
        auto pipelineBaker = GetRTPipelineBaker();
        using SM = donut::engine::ShaderMacro;

        // these don't actually compile any shaders - this happens later in m_ptPipelineBaker->Update
        m_ptPipelineReference = pipelineBaker->CreateVariant("PathTracerSample.hlsl", { SM("PATH_TRACER_MODE", "PT_REFERENCE") }, "REF");
        m_ptPipelineBuildDenoisingLayers = pipelineBaker->CreateVariant("PathTracerSample.hlsl", { SM("PATH_TRACER_MODE", "PT_DENOISING_LAYERS_BUILD") }, "BUILD");
        m_ptPipelineFillDenoisingLayers = pipelineBaker->CreateVariant("PathTracerSample.hlsl", { SM("PATH_TRACER_MODE", "PT_DENOISING_LAYERS_FILL") }, "FILL", false,
            [this](std::vector<SM>& macros) { this->GetLocalMacros_FillDenoisingLayers(macros); });

        // SHaRC owns its own update-pass variant. Constructed here rather than waiting for CreateCustomRenderPasses,
        // which runs later (Sample::Render calls CreateRTPipelines before CreateRenderPasses), because the variant's
        // runtime macro callback reads DownscaleFactor() off this object.
        EnsureSharcCreated();
        m_sharc->CreatePipelines(*pipelineBaker, *GetComputePipelineBaker());

        Sample::CreateRTPipelines();
    }

    virtual void DestroyRTPipelines() override
    {
        m_ptPipelineReference = nullptr;
        m_ptPipelineBuildDenoisingLayers = nullptr;
        m_ptPipelineFillDenoisingLayers = nullptr;

        // must happen before the base class lets go of the baker
        if (m_sharc != nullptr)
            m_sharc->DestroyPipelines(*GetRTPipelineBaker(), *GetComputePipelineBaker());

        Sample::DestroyRTPipelines();
    }

    virtual void CreateCustomRenderPasses() override
    {
        m_denoisingGuidesBaker = std::make_shared<DenoisingGuidesBaker>(GetDevice(), m_shaderFactory, m_renderTargets, m_shaderDebug, m_bindingLayout);
        m_adaptiveSampling = std::make_unique<AdaptiveSampling>(GetDevice(), m_shaderFactory, m_renderTargets, m_bindingLayout, m_bindlessLayout);

        EnsureSharcCreated();
    }

    // SHaRC spatial radiance cache (reference-style integration). Created once and kept across pass rebuilds so the
    // cache contents survive shader reloads; owns its own buffers, update variant and resolve pass. The global +
    // bindless binding layouts are handed in so the resolve compute PSO can run against the sample's global binding
    // set (which provides g_Const at b0 and the SHaRC UAVs at u90-u92). Called from both CreateRTPipelines and
    // CreateCustomRenderPasses, whichever runs first.
    void EnsureSharcCreated()
    {
        if (m_sharc == nullptr)
            m_sharc = std::make_unique<SharcIntegration>(GetDevice(), m_bindingLayout, m_bindlessLayout,
                                                        m_bindlessHeapLayout, m_bindlessHeapDescriptorTable);
    }

    // NOTE: there is deliberately no AddCustomBindings override any more. The three cache buffers are reached
    // through ResourceDescriptorHeap (Rtxpt/Shaders/Bindless.hlsli) instead of binding slots, so nothing needs to
    // be patched into the global binding set. u90-u92 remain in the shared layout for IntroSample's own SHaRC.

    // Size SHaRC's resolution-dependent resources. This has to be the early hook rather than
    // UpdateCustomConstants: the path state buffer's bindless descriptor must exist before the base class
    // snapshots the descriptor table for t_BindlessValidationData, or the training pass trips the bindless
    // validation trap for one frame after every (re)allocation.
    virtual void UpdateCustomResources() override
    {
        if (m_sharc != nullptr)
        {
            const nvrhi::Viewport viewport = m_view->GetViewport();
            m_sharc->UpdateResources(m_ui.ActualRadianceCacheMode() == 1,
                                     (uint32_t)(viewport.maxX - viewport.minX), (uint32_t)(viewport.maxY - viewport.minY));
        }
    }

    // Fill ptConsts.sharc just before the SampleConstants upload (base class calls this from Render()).
    virtual void UpdateCustomConstants(SampleConstants& constants, nvrhi::ICommandList* commandList) override
    {
        if (m_sharc != nullptr)
            m_sharc->UpdateConstants(commandList, constants.ptConsts.sharc, *m_view);
    }

    virtual void GetCustomPTPipelineGlobalMacros(std::vector<donut::engine::ShaderMacro>& macros) override
    {
    }

    // Clear the cache when the scene changes (geometry the cache was built against is gone).
    virtual void SceneLoaded() override
    {
        Sample::SceneLoaded();
        if (m_sharc != nullptr)
            m_sharc->NotifySceneReloaded();
    }

    virtual void PrepareDLSSRRGuides(nvrhi::ICommandList* commandList) override
    {
        m_denoisingGuidesBaker->PrepareDLSSRRGuides(commandList, m_bindingSet);
    }

    // Perf log timer queries would be resolved from two queues into nvrhi's one shared readback buffer
    virtual bool SupportsAsyncPrefix() const override { return m_cmdLine.PerfLogPath.empty(); }

    virtual std::string GetMaterialSpecializationShader() const override {
        return "PathTracerMaterialSpecializations.hlsl";
    }

    virtual SharcIntegration* GetSharc() override { return m_sharc.get(); }

    virtual void BeforeCaptureExit() override { m_perfLog.Flush(); }

    virtual std::unique_ptr<SampleUI> CreateUI(donut::app::DeviceManager* deviceManager, class SampleBaseApp& baseApp, SampleUIData& uiData, bool NVAPI_SERSupported, const struct CommandLineOptions& cmdLine) override
    {
        return std::make_unique<AdvancedSampleUI>(deviceManager, baseApp, *this, uiData, NVAPI_SERSupported, cmdLine);
    }

private:
    void PathTrace(nvrhi::IFramebuffer* framebuffer, const SampleConstants& constants);

    std::unique_ptr<AdaptiveSampling>           m_adaptiveSampling;

    PathTracePerfLog                            m_perfLog;
    bool                                        m_perfLogInitialized = false;

    std::shared_ptr<class PTPipelineVariant> m_ptPipelineReference;
    std::shared_ptr<class PTPipelineVariant> m_ptPipelineBuildDenoisingLayers;
    std::shared_ptr<class PTPipelineVariant> m_ptPipelineFillDenoisingLayers;

    std::shared_ptr<DenoisingGuidesBaker>       m_denoisingGuidesBaker;
    std::unique_ptr<SharcIntegration>           m_sharc;
};

void AdvancedPathTracer::PathTrace(nvrhi::IFramebuffer* framebuffer, const SampleConstants & constants)
{
    //m_commandList->beginMarker("MainRendering"); <- removed (for now) since added hierarchy reduces readability
    ZoneScopedN("PathTrace");

    bool useDenoisingLayers = m_ui.RealtimeMode;

    nvrhi::rt::State state;

    nvrhi::rt::DispatchRaysArguments args;
    nvrhi::Viewport viewport = m_view->GetViewport();
    uint32_t width = (uint32_t)(viewport.maxX - viewport.minX);
    uint32_t height = (uint32_t)(viewport.maxY - viewport.minY);
    args.width = width;
    args.height = height;

    // default miniConstants
    SampleMiniConstants miniConstants = { uint4(0, 0, 0, 0) };

    if (!m_perfLogInitialized)
    {
        if (!m_cmdLine.PerfLogPath.empty())
            m_perfLog.Init(GetDevice(), m_cmdLine.PerfLogPath);
        m_perfLogInitialized = true;
    }
    m_perfLog.BeginFrame(m_frameIndex, GetSceneTime(), uint2(width, height), m_ui.ActualSubSamplingMode(), (uint32_t)m_ui.ActualSubSamplingPattern(), m_ui.SubSamplingStratifiedBlockSize());
    RAII_SCOPE(;, m_perfLog.EndFrame(); );

    // Note: SHaRC constants are filled in UpdateCustomConstants, which base Render calls before uploading SampleConstants

    if (useDenoisingLayers)
    {
        {
            RAII_SCOPE(GpuProfiler::Begin(m_commandList, "PathTracePrePass"); m_perfLog.Begin(m_commandList, PathTracePerfLog::Build);, m_perfLog.End(m_commandList, PathTracePerfLog::Build); GpuProfiler::End(m_commandList); );

            m_commandList->setTextureState(m_renderTargets->Depth, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
            m_commandList->setTextureState(m_renderTargets->ScreenMotionVectors, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
            m_commandList->setTextureState(m_renderTargets->Throughput, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
            m_commandList->setTextureState(m_renderTargets->SpecularHitT, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
            state.shaderTable = m_ptPipelineBuildDenoisingLayers->GetShaderTable();
            state.bindings = { m_bindingSet, m_DescriptorTable->GetDescriptorTable(), m_bindlessHeapDescriptorTable->GetDescriptorTable() };
            m_commandList->setRayTracingState(state);
            m_commandList->setPushConstants(&miniConstants, sizeof(miniConstants));
            m_commandList->dispatchRays(args);
            m_commandList->setBufferState(m_renderTargets->DenoisingLayersBuffer, nvrhi::ResourceStates::UnorderedAccess);
            m_commandList->setTextureState(m_renderTargets->Depth, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
            m_commandList->setTextureState(m_renderTargets->ScreenMotionVectors, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
            m_commandList->setTextureState(m_renderTargets->Throughput, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        }

        if (m_ui.DebugView == DebugViewType::VBufferMotionVectors || m_ui.DebugView == DebugViewType::VBufferDepth)  // it only outputs DebugPixel, for these two views
        {
            RTXPT_GPU_SCOPE(m_commandList, "VBufferExport");

            nvrhi::ComputeState state;
            state.bindings = { m_bindingSet, m_DescriptorTable->GetDescriptorTable() };
            state.pipeline = m_exportVBufferPSO;
            m_commandList->setComputeState(state);

            const dm::uint2 dispatchSize = { (width + NUM_COMPUTE_THREADS_PER_DIM - 1) / NUM_COMPUTE_THREADS_PER_DIM, (height + NUM_COMPUTE_THREADS_PER_DIM - 1) / NUM_COMPUTE_THREADS_PER_DIM };
            m_commandList->setPushConstants(&miniConstants, sizeof(miniConstants));
            m_commandList->dispatch(dispatchSize.x, dispatchSize.y);
        }
    }
    else
    {

    }

    // In realtime mode, ScreenMotionVectors reference mode ScreenMotionVectors should be 0
    m_lightsBaker->UpdateEnd(m_commandList, *m_bindingCache, m_scene, m_materialsBaker, m_ommBaker, m_subInstanceBuffer, m_renderTargets->Depth, m_renderTargets->ScreenMotionVectors);  // <- in the future this will provide motion vectors except in case of reference mode

    // SHaRC: train + resolve the cache before the main path-trace pass consumes it. SharcIntegration owns the
    // training pipeline and the whole clear/train/resolve sequence, including its downscaled dispatch size.
    if (m_sharc != nullptr && m_ui.ActualRadianceCacheMode() == 1)      // ActualRadianceCacheMode() is already 0 outside realtime mode
        m_sharc->RenderUpdate(m_commandList, m_bindingSet, m_DescriptorTable->GetDescriptorTable(),
                              m_bindlessHeapDescriptorTable->GetDescriptorTable(), width, height,
                              constants, m_constantBuffer);

    {
        RTXPT_GPU_SCOPE(m_commandList, "PathTrace");

        // Resolve wrote the cache; this pass reads it. That edge has to be declared explicitly now that the buffers
        // are bindless and no longer picked up by setResourceStatesForBindingSet. It lives here rather than in
        // SharcIntegration because RenderUpdate early-returns when training is frozen ('Enable Training' off) while
        // this pass keeps consuming the cache.
        if (m_sharc != nullptr && m_ui.ActualRadianceCacheMode() == 1)
            m_sharc->BarrierCacheBuffers(m_commandList);

        state.shaderTable = ((useDenoisingLayers) ? (m_ptPipelineFillDenoisingLayers) : (m_ptPipelineReference))->GetShaderTable();
        state.bindings = { m_bindingSet, m_DescriptorTable->GetDescriptorTable(), m_bindlessHeapDescriptorTable->GetDescriptorTable() };

        SampleMiniConstants subSamplingConstants = { uint4(0, 0, 0, 0), uint4(0, 0, 0, 0), uint4(0, 0, 0, 0), uint4(0, 0, 0, 0) };
        const nvrhi::rt::DispatchRaysArguments argsInner = (m_ui.ActualSubSamplingMode() > 0)
            ? m_adaptiveSampling->PrepareSubSampling(m_commandList, m_ui, m_bindingSet, m_DescriptorTable->GetDescriptorTable(), m_perfLog, width, height, subSamplingConstants)
            : args;

        m_perfLog.Begin(m_commandList, PathTracePerfLog::Fill);
        RAII_SCOPE(;, m_perfLog.End(m_commandList, PathTracePerfLog::Fill); );
        for (uint subSampleIndex = 0; subSampleIndex < m_ui.ActualSamplesPerPixel(); subSampleIndex++)
        {
            // required to avoid race conditions in back to back dispatchRays
            m_commandList->setBufferState(m_renderTargets->DenoisingLayersBuffer, nvrhi::ResourceStates::UnorderedAccess);
            m_commandList->setTextureState(m_renderTargets->SpecularHitT, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);

            m_commandList->setRayTracingState(state);

            // tell path tracer which subSampleIndex we're processing
            SampleMiniConstants miniConstants = subSamplingConstants;
            miniConstants.params = uint4(subSampleIndex, 0, 0, 0);  //     <- use subSampleIndex to try to figure out why we're losing radiance - is the first one what's left, or the last one?
            m_commandList->setPushConstants(&miniConstants, sizeof(miniConstants));

            m_commandList->dispatchRays(argsInner);
            m_commandList->setTextureState(m_renderTargets->SpecularHitT, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        }

        m_commandList->setBufferState(m_renderTargets->DenoisingLayersBuffer, nvrhi::ResourceStates::UnorderedAccess);
    }

    // this is a performance optimization where final 2 passes from ReSTIR DI and ReSTIR GI are combined to avoid loading GBuffer twice
    static bool enableFusedDIGIFinal = true;
    bool useFusedDIGIFinal = m_ui.ActualUseReSTIRDI() && m_ui.ActualUseReSTIRGI() && enableFusedDIGIFinal;

    if (m_ui.ActualUseRTXDIPasses())
    {
        RTXPT_GPU_SCOPE(m_commandList, "RTXDI");

        // this does all ReSTIR DI magic including applying the final sample into correct radiance buffer (depending on denoiser state)
        if (m_ui.ActualUseReSTIRDI())
            m_rtxdiPass->Execute(m_commandList, m_bindingSet, useFusedDIGIFinal);

        if (m_ui.ActualUseReSTIRGI())
            m_rtxdiPass->ExecuteGI(m_commandList, m_bindingSet, useFusedDIGIFinal);

        if (useFusedDIGIFinal)
            m_rtxdiPass->ExecuteFusedDIGIFinal(m_commandList, m_bindingSet);
    }

    {
        RTXPT_GPU_SCOPE(m_commandList, "Denoising Guides Bake");

        m_denoisingGuidesBaker->DenoiseSpecHitT(m_commandList, m_bindingSet);

        if (m_ui.DebugView != DebugViewType::Disabled)
            m_denoisingGuidesBaker->RenderDebugViz(m_commandList, m_ui.DebugView, m_bindingSet);
    }

    if (useDenoisingLayers && (m_ui.DebugView > DebugViewType::Disabled && m_ui.DebugView <= DebugViewType::DenoisingLayer_SpecRadiance || m_ui.DebugView == DebugViewType::StableRadiance) )
    {
        GpuProfiler::Begin(m_commandList, "DenoisingLayersDebugViz");
        nvrhi::TextureDesc tdesc = m_renderTargets->OutputColor->getDesc();
        m_postProcess->Apply(m_commandList, PostProcess::ComputePassType::DenoisingLayersDebugViz, m_constantBuffer, miniConstants, m_bindingSet, m_bindingLayout, tdesc.width, tdesc.height);
        GpuProfiler::End(m_commandList);

    }
}

class AdvancedSample : public SampleBaseApp
{
    std::unique_ptr<Sample> CreateMainRenderPass(donut::app::DeviceManager& deviceManager, const CommandLineOptions& cmdLineOptions) override
    {
        return std::make_unique<AdvancedPathTracer>(deviceManager, cmdLineOptions);
    }
};

#ifdef _WIN32
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
int main(int __argc, const char** __argv)
#endif
{
    SplashScreen splashScreen;
    splashScreen.Start(L"loading_splash.png");

    AdvancedSample example;

    // Run the sample app
    const auto status = example.Init(__argc, __argv);

    splashScreen.Stop();

    if (status == SampleBaseApp::InitReturnCodes::Success)
    {
        example.RunMainLoop();

        example.End();
    }

    return static_cast<int>(status);
}
