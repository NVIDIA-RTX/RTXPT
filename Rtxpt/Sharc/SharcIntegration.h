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

#include <algorithm>
#include <memory>
#include <vector>
#include <donut/core/math/math.h>
#include <donut/engine/BindingCache.h>
#include <donut/engine/DescriptorTableManager.h>
#include <donut/engine/View.h>

#include "../Shaders/Libraries/SHaRC/SharcConstants.h"

class PTPipelineBaker;
class PTPipelineVariant;
class ComputePipelineBaker;
class ComputeShaderVariant;
struct SampleConstants;

struct SharcSettings
{
    // NOTE: no 'Enabled' here - SampleUIData::ActualRadianceCacheMode() is the single source of truth, and the
    // caller passes it in where it matters (UpdateResources, and the gate around RenderUpdate).
    bool EnableClear = false;
    bool EnableUpdate = true;
    bool EnableResolve = true;
    bool UpdateViewCamera = true;
    bool EnableMaterialDemodulation = true;
    bool EnableSeparateEmissive = true;
    int DebugMode = 0;                   // 0 = off; anything else also compiles the debug-viz code in (see EnableDebug())
    // Bounds for DownScaleFactor below, enforced both by the UI slider and by SharcIntegration::DownscaleFactor().
    // The lower bound is not cosmetic: the per-thread path state buffer is sized to the training dispatch, so
    // halving the factor quadruples it. At 4K that is ~133 MB at 2, and would be ~531 MB at 1.
    static constexpr int DownScaleFactorMin = 2;
    static constexpr int DownScaleFactorMax = 5;
    int DownScaleFactor = 4;
    float SceneScale = 35.0f;
    int AccumulationFrameNum = 10;
    int StaleFrameFrameNum = 64;
    float MinRoughnessOverride = 0.4f;
    // Applied over ptConsts by ApplyTrainingConstantOverrides. Deliberately deeper than the main path tracer's
    // realtime budget - the cache is meant to hold full indirect.
    int BounceCount = 8;
    int DiffuseBounceCount = 4;
    // Training-pass overrides of the equivalent main path tracer settings. The two baked as macros recompile the
    // training pipeline when changed; the other two are runtime constants.
    int NEEFullSamples = 1;                 // -> RTXPT_NEE_FULL_SAMPLE_COUNT
    bool FireflyFilterEnabled = true;       // -> RTXPT_FIREFLY_FILTER
    float FireflyFilterThreshold = 0.50f;
    float AdditionalTexLODBias = 1.0f;      // ADDED to the main 'Texture MIP bias', not a replacement
    float TerminateThreshold = 3.5f;
};

class SharcIntegration
{
public:
    // The resolve pass runs against the sample's global binding set (which provides g_Const at b0); the cache
    // buffers themselves are reached bindlessly, so they need no binding slots of their own. All three binding
    // layouts are needed to create the resolve PSO, and bindlessHeapTable is where the three cache descriptors
    // are allocated - it must be the table built on the MutableSrvUavCbv layout, not the scene one.
    SharcIntegration(nvrhi::IDevice* device,
                     nvrhi::IBindingLayout* globalBindingLayout, nvrhi::IBindingLayout* globalBindlessLayout,
                     nvrhi::IBindingLayout* globalBindlessHeapLayout,
                     const std::shared_ptr<donut::engine::DescriptorTableManager>& bindlessHeapTable);
    ~SharcIntegration();

public:
    const SharcSettings& GetSettings() const { return m_settings; }

    bool EnableUpdate() const { return m_settings.EnableUpdate; }
    bool EnableClear() const { return m_settings.EnableClear; }
    bool EnableResolve() const { return m_settings.EnableResolve; }
    // Gates RTXPT_SHARC_ENABLE_DEBUG, i.e. whether the debug-viz code is compiled into the path tracer at all.
    // Derived from DebugMode rather than stored: picking a mode is the only reason to want it.
    bool EnableDebug() const { return m_settings.DebugMode > 0; }
    // Single source of truth for the update-pass downscale: feeds both the dispatch size and the
    // RTXPT_SUB_SAMPLING_SCALE_X/Y macros baked into the SHaRC update pipeline variant. Those two must agree, or the
    // remapped pixels in RAYGEN_ENTRY stop covering the render target.
    // Clamped rather than asserted: this is the value everything downstream derives from, so it must be in range
    // even if the setting is not (ctrl+click text entry on the slider, or a future settings file).
    uint32_t DownscaleFactor() const { return (uint32_t)std::clamp(m_settings.DownScaleFactor, SharcSettings::DownScaleFactorMin, SharcSettings::DownScaleFactorMax); }

    nvrhi::BufferHandle GetHashEntriesBuffer() const { return m_hashEntriesBuffer; }
    nvrhi::BufferHandle GetAccumulationBuffer() const { return m_accumulationBuffer; }
    nvrhi::BufferHandle GetResolvedBuffer() const { return m_resolvedBuffer; }

    // The cache buffers no longer sit in any binding set, so nvrhi's automatic per-binding-set UAV barriers do not
    // cover them any more - setResourceStatesForBindingSet early-returns for bindless sets. Every producer ->
    // consumer edge has to be declared by hand with this. Also does a real state transition rather than only a UAV
    // barrier, which matters after a clear: on Vulkan clearBufferUInt leaves the buffer in CopyDest.
    void BarrierCacheBuffers(nvrhi::ICommandList* commandList);
    // categoryColor matches the section-heading colour the rest of the UI uses; it is threaded in from
    // AdvancedSampleUI::DrawSampleUI rather than hard-coded so SHaRC's headings stay in step with the theme.
    // discardNonNEELighting / discardNEELighting are SampleUIData::DbgDiscard*. They bake into every PT pipeline as
    // macros so they live with the global UI, not here; the "Disable all non-SHaRC lighting" checkbox drives both.
    bool DebugGUI(float indent, const struct ImVec4& categoryColor, bool& discardNonNEELighting, bool& discardNEELighting);

    // Creates or resizes the per-thread path state scratch buffer for this frame's training dispatch, and
    // allocates its bindless descriptor. width/height are the full render resolution; the training downscale is
    // applied here, identically to RenderUpdate.
    // Must be called once per frame and EARLY - before the sample refreshes its bindless validation data and
    // rebuilds its binding set (Sample::UpdateCustomResources is exactly that point). Calling it later, e.g. from
    // UpdateConstants, leaves a descriptor the validation snapshot has not seen and trips the bindless trap for
    // one frame on every (re)allocation.
    // 'enabled' is ActualRadianceCacheMode()==1; must still be called when false so the buffer is released.
    void UpdateResources(bool enabled, uint32_t width, uint32_t height);

    // Fills the SHaRC constants (CPU-side) for the upcoming SampleConstants upload, and records any
    // cache clear-on-settings-change on the command list. The caller passes &constants.ptConsts.sharc, so
    // SharcIntegration only ever sees SharcConstants. Must be called before the SampleConstants upload.
    void UpdateConstants(nvrhi::ICommandList* commandList, SharcConstants& sharcConsts, const donut::engine::PlanarView& view);
    void ClearBuffers(nvrhi::ICommandList* commandList);

    // The SHaRC update pass is a PathTracerSample.hlsl variant owned here rather than by the sample, so this class
    // owns both of its passes (update + resolve) and all the state that drives them. Call these from the sample's
    // CreatePipelines / DestroyPipelines overrides - the pair must stay balanced, and the release has to happen
    // before the PTPipelineBaker itself is destroyed.
    // Covers BOTH of SHaRC's passes: the training pass (a PathTracerSample.hlsl ray tracing variant) and the resolve
    // pass (a compute shader). Routing resolve through ComputePipelineBaker instead of building its PSO once in the
    // constructor is what makes SharcResolve.hlsl hot-reloadable - this object deliberately outlives pass rebuilds,
    // so a constructor-built PSO would never be refreshed by Ctrl+R.
    void CreatePipelines(PTPipelineBaker& pipelineBaker, ComputePipelineBaker& computeBaker);
    void DestroyPipelines(PTPipelineBaker& pipelineBaker, ComputePipelineBaker& computeBaker);

    // Runs the whole cache-filling sequence (optional clear, update, resolve) for this frame. No-op unless SHaRC is
    // enabled. width/height are the full render resolution; the downscaled dispatch size is derived here.
    // 'constants' must match what the caller last uploaded to 'constantBuffer'; the training dispatch runs against
    // an overridden copy (ApplyTrainingConstantOverrides) and the original is written back before returning.
    void RenderUpdate(nvrhi::ICommandList* commandList, nvrhi::IBindingSet* globalBindingSet,
                      nvrhi::IBindingSet* descriptorTable, nvrhi::IBindingSet* bindlessHeapTable, uint32_t width, uint32_t height,
                      const SampleConstants& constants, nvrhi::IBuffer* constantBuffer);

    // Patches the training pass's view of the shared SampleConstants into a copy - it runs against the same single
    // constant buffer as the rest of the frame, so anything it must see differently goes here. Prefer a macro on the
    // training pipeline variant (CreatePipelines) when the value is known at build time; this is for dynamic ones.
    void ApplyTrainingConstantOverrides(SampleConstants& constants) const;
    // Dispatched against the sample's global binding set + both bindless tables. bindlessHeapTable must be passed
    // even though nothing reads it through a root parameter: nvrhi's validation layer rejects a NULL binding set on
    // compute dispatches, and the layout list this PSO was built with includes it.
    void Resolve(nvrhi::ICommandList* commandList, nvrhi::IBindingSet* globalBindingSet, nvrhi::IBindingSet* descriptorTable, nvrhi::IBindingSet* bindlessHeapTable);

    void NotifySceneReloaded() { m_sceneReloaded = true; }

private:
    const nvrhi::DeviceHandle       m_device;
    SharcSettings                   m_settings;

    // Track changes in settings values. When one of these flips, existing entries are stale and must be cleared.
    // Note: init must match SharcSettings default
    bool                            m_prevEnableMaterialDemodulation = true;
    bool                            m_prevEnableSeparateEmissive     = true;
    bool                            m_sceneReloaded = false;

    static const uint32_t           m_invalidEntry = 0;

    uint32_t                        m_entriesNum = 0;
    nvrhi::BufferHandle             m_hashEntriesBuffer;
    nvrhi::BufferHandle             m_accumulationBuffer;
    nvrhi::BufferHandle             m_resolvedBuffer;

    // Bindless descriptors for the three buffers above. Held for the lifetime of this object - releasing a handle
    // frees the slot, and the shader would then be indexing a recycled descriptor.
    std::shared_ptr<donut::engine::DescriptorTableManager> m_bindlessHeapTable;
    donut::engine::DescriptorHandle m_hashEntriesDescriptor;
    donut::engine::DescriptorHandle m_accumulationDescriptor;
    donut::engine::DescriptorHandle m_resolvedDescriptor;

    // Per-thread scratch holding the SHaRC propagation state across the TraceRay boundary during training - what
    // used to ride in the ray payload (see PathTracerSHaRC.hlsli). One RTXPT_SHARC_PATH_STATE_STRIDE slot per
    // training thread, so it is sized to renderResolution / DownscaleFactor and reallocated when either changes.
    // Released entirely while SHaRC is off: at the low end of the DownScaleFactor range this is not a small buffer.
    // Called from UpdateResources rather than RenderUpdate, because the heap index has to be valid by the time the
    // constants are uploaded.
    void                            UpdatePathStateBuffer(uint32_t trainWidth, uint32_t trainHeight);

    nvrhi::BufferHandle             m_pathStateBuffer;
    donut::engine::DescriptorHandle m_pathStateDescriptor;
    uint32_t                        m_pathStateCapacity = 0;    // thread slots, not bytes

    // Superseded (buffer, descriptor) pairs waiting out the frames still in flight. DescriptorTableManager::
    // ReleaseDescriptor frees the heap slot and writes a null descriptor immediately, so dropping the old handle
    // straight away would leave an in-flight frame indexing a recycled descriptor - and UpdatePathStateBuffer runs
    // with the command list open, so it cannot waitForIdle. Retire instead of releasing.
    struct RetiredPathStateBuffer
    {
        nvrhi::BufferHandle             Buffer;
        donut::engine::DescriptorHandle Descriptor;
        uint32_t                        FramesLeft = 0;
    };
    static const uint32_t           c_pathStateRetireFrames = 4;
    std::vector<RetiredPathStateBuffer> m_retiredPathStateBuffers;

    // kept so the resolve compute variant can be created (and re-created) after construction
    nvrhi::BindingLayoutHandle      m_globalBindingLayout;
    nvrhi::BindingLayoutHandle      m_globalBindlessLayout;
    nvrhi::BindingLayoutHandle      m_globalBindlessHeapLayout;

    std::shared_ptr<PTPipelineVariant>    m_trainPipeline;
    std::shared_ptr<ComputeShaderVariant> m_resolveVariant;
};
