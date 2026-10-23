/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <algorithm>
#include <nvrhi/utils.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include "../SampleCommon/SampleCommon.h"
#include <donut/app/imgui_renderer.h>

#include "SharcIntegration.h"
#include "../SampleCommon/PTPipelineBaker.h"
#include "../SampleCommon/ComputePipelineBaker.h"
#include "../Shaders/SampleConstantBuffer.h"     // SampleMiniConstants (resolve-pass push constants)
#include "../Sample.h"                          // Sample::FireflyFilterThreshold

using namespace donut::math;

SharcIntegration::SharcIntegration(nvrhi::IDevice* device,
                                   nvrhi::IBindingLayout* globalBindingLayout, nvrhi::IBindingLayout* globalBindlessLayout,
                                   nvrhi::IBindingLayout* globalBindlessHeapLayout,
                                   const std::shared_ptr<donut::engine::DescriptorTableManager>& bindlessHeapTable)
    : m_device(device), m_globalBindingLayout(globalBindingLayout), m_globalBindlessLayout(globalBindlessLayout)
    , m_globalBindlessHeapLayout(globalBindlessHeapLayout), m_bindlessHeapTable(bindlessHeapTable)
{
    // Prepare resources for the SHARC copy and resolve compute passes
    {
        m_entriesNum = 4 * 1024 * 1024;

        // Buffers (SHaRC constants live in the shared SampleConstants buffer, so no dedicated CB here)
        nvrhi::BufferDesc bufferDesc;
        bufferDesc.isConstantBuffer = false;
        bufferDesc.isVolatile = false;
        bufferDesc.canHaveUAVs = true;
        bufferDesc.cpuAccess = nvrhi::CpuAccessMode::None;
        bufferDesc.keepInitialState = true;
        bufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;

        bufferDesc.structStride = sizeof(uint64_t);
        bufferDesc.byteSize = m_entriesNum * bufferDesc.structStride;
        bufferDesc.debugName = "SharcHashEntriesBuffer";
        m_hashEntriesBuffer = m_device->createBuffer(bufferDesc);

        bufferDesc.structStride = 4 * sizeof(uint32_t);
        bufferDesc.byteSize = m_entriesNum * bufferDesc.structStride;
        bufferDesc.canHaveRawViews = true;

        bufferDesc.debugName = "SharcAccumulationBuffer";
        m_accumulationBuffer = m_device->createBuffer(bufferDesc);

        bufferDesc.debugName = "SharcResolvedBuffer";
        m_resolvedBuffer = m_device->createBuffer(bufferDesc);
    }

    // Bindless descriptors. The slot index is written into the constants every frame rather than cached, because
    // growing the descriptor table relocates it and invalidates every heap index handed out beforehand.
    assert(m_bindlessHeapTable != nullptr);
    m_hashEntriesDescriptor  = m_bindlessHeapTable->CreateDescriptorHandle(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_hashEntriesBuffer));
    m_accumulationDescriptor = m_bindlessHeapTable->CreateDescriptorHandle(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_accumulationBuffer));
    m_resolvedDescriptor     = m_bindlessHeapTable->CreateDescriptorHandle(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_resolvedBuffer));

    // The resolve compute pipeline is NOT created here - see CreatePipelines. Building it in the constructor is what
    // used to make SharcResolve.hlsl un-hot-reloadable, since this object outlives pass rebuilds.
}

SharcIntegration::~SharcIntegration( )
{}

bool SharcIntegration::DebugGUI(float indent, const ImVec4& categoryColor, bool& discardNonNEELighting, bool& discardNEELighting)
{
    bool changed = false;

    // Own ID scope: ImGui derives widget identity from the label, and some below match the main path tracer's.
    RAII_SCOPE(ImGui::PushID("SHaRC"); , ImGui::PopID(););


    ImGui::TextColored(categoryColor, "SHaRC training PT pass settings:");
    {
        RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent););

        changed |= ImGui::InputInt("Max bounces", &m_settings.BounceCount);
        m_settings.BounceCount = dm::clamp(m_settings.BounceCount, 0, MAX_BOUNCE_COUNT);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Max number of all bounces (including NEE and diffuse bounces), for the training pass only.\nUsually higher than the main path tracer's: the cache is meant to hold full indirect lighting.");
        changed |= ImGui::InputInt("Max diffuse bounces", &m_settings.DiffuseBounceCount);
        m_settings.DiffuseBounceCount = dm::clamp(m_settings.DiffuseBounceCount, 0, MAX_BOUNCE_COUNT);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Max number of diffuse bounces (diffuse lobe and specular with roughness > 0.25 or similar depending on settings), for the training pass only.");

        changed |= ImGui::SliderInt("Training Resolution Downscale", &m_settings.DownScaleFactor, SharcSettings::DownScaleFactorMin, SharcSettings::DownScaleFactorMax, "%d", ImGuiSliderFlags_AlwaysClamp);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("The resolution of the training (RTXPT_SHARC_TRAIN) pass: 'sharcRes = renderResolution.xy / downScaleFactor'\nThe range starts at 2 because the per-thread path state buffer is sized to this dispatch:\nhalving the factor quadruples it, and at 4K a factor of 1 would cost around half a gigabyte.");

        changed |= ImGui::InputInt("NEE Full Samples", &m_settings.NEEFullSamples, 1);
        m_settings.NEEFullSamples = dm::clamp(m_settings.NEEFullSamples, 0, RTXPT_LIGHTING_MAX_SAMPLE_COUNT);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Overrides the main path tracer's NEE 'Full samples' for the training pass: light samples\nshadow-tested and integrated per vertex. Changing this recompiles the training pipeline.");

        changed |= ImGui::InputFloat("Additional Texture MIP bias", &m_settings.AdditionalTexLODBias, 0.1f, 0.5f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("ADDED to the main path tracer's 'Texture MIP bias' for the training pass, not a replacement.\nThe cache is spatially averaged, so blurrier fetches cost little quality here.");

        changed |= ImGui::Checkbox("FireflyFilter", &m_settings.FireflyFilterEnabled);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Overrides the main path tracer's firefly filter for the training pass. Matters more here: a\nfirefly written into a voxel persists across frames and pixels. Toggling recompiles the pipeline.");
        if (m_settings.FireflyFilterEnabled)
        {
            RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent););
            changed |= ImGui::InputFloat("FF Threshold", &m_settings.FireflyFilterThreshold, 0.01f, 0.1f, "%.5f");
            m_settings.FireflyFilterThreshold = dm::clamp(m_settings.FireflyFilterThreshold, 0.00001f, 1000.0f);
        }
    }
    ImGui::TextColored(categoryColor, "SHaRC usage settings:");
    {
        RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent););

        changed |= ImGui::SliderFloat("Terminate Threshold", &m_settings.TerminateThreshold, 0.01f, 10.0f);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Decides when to drop into radiance cache; small value - drop sooner, large value - drop later; this is the main performance<->quality setting");
    }
    ImGui::TextColored(categoryColor, "SHaRC advanced settings:");
    {
        RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent););

        changed |= ImGui::Checkbox("Enable Material Demodulation", &m_settings.EnableMaterialDemodulation);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Divides out the surface BSDF response (diffuse albedo + specular) before storing radiance,\nand re-applies it at query time. Preserves material/texture detail even though one voxel is\nshared by many surfaces. Changing this invalidates stored entries, so the cache is auto-cleared.");
        changed |= ImGui::Checkbox("Enable Separate Emissive", &m_settings.EnableSeparateEmissive);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Keeps surface emission out of the cached (spatially averaged) radiance and re-adds it at query\ntime instead. Keeps emissive surfaces sharp rather than smearing them across the voxel.\nChanging this invalidates stored entries, so the cache is auto-cleared.");
        changed |= ImGui::SliderInt("Accumulation Frame Number", &m_settings.AccumulationFrameNum, 1, 60);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Maximum length of the temporal accumulation window per voxel, in frames.\nHigher - smoother, less noisy cache, but slower to react to lighting changes.\nLower - more responsive but noisier.");
        changed |= ImGui::SliderInt("Stale Frame Number", &m_settings.StaleFrameFrameNum, 8, 128);   // low end is SHARC_STALE_FRAME_NUM_MIN: SharcResolveEntry clamps to it, so anything below 8 is a no-op
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("How many frames a voxel may go without receiving any new sample before it gets evicted.\nHigher keeps off-screen/occluded areas cached (helps when the camera turns back), lower frees\nentries sooner and reduces occupancy and hash collisions. The range starts at 8 because that is the SDK minimum (SHARC_STALE_FRAME_NUM_MIN) it gets clamped to anyway.");
        changed |= ImGui::SliderFloat("Scene Scale", &m_settings.SceneScale, 1.0f, 100.0f);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("How big/small the voxels are. Too big - indirect lighting from SHaRC loses detail. Too small - too many hash collisions");
        changed |= ImGui::SliderFloat("Min Roughness Override", &m_settings.MinRoughnessOverride, 0.0f, 1.0f);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("This will limit min roughness for all surfaces (except pure delta), for the SHaRC Update (training) pass");
    }

    ImGui::TextColored(categoryColor, "Debugging:");
    {
        RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent););
        changed |= ImGui::Checkbox("Enable Clear", &m_settings.EnableClear);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Debug: wipes the whole cache (hash entries, accumulation and resolved buffers) at the\nstart of every frame, so nothing ever accumulates over time. Leave off for normal use.");
        changed |= ImGui::Checkbox("Enable Training", &m_settings.EnableUpdate);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Runs the RTXPT_SHARC_TRAIN (training) pass that populates the cache.\nUnchecking freezes cache contents - the query pass keeps reading whatever was cached last,\nwhich is useful for inspecting what actually got stored. Note: also gates Clear and Resolve below.");
        changed |= ImGui::Checkbox("Enable Resolve", &m_settings.EnableResolve);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Runs the resolve pass: merges this frame's accumulated samples into the resolved buffer,\napplies the temporal accumulation window, blends adjacent grid levels on camera movement\nand evicts stale entries. Off means the query pass reads a resolved buffer that never advances.");
        changed |= ImGui::Checkbox("Update View Camera", &m_settings.UpdateViewCamera);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Keeps the hash grid anchored to the current camera. Voxel LOD is chosen from the distance to\nthis anchor, so unchecking freezes the grid layout and its levels at the current position -\nlets you fly around and inspect the cache as it was built. Debug only; leave on for normal use.");
        // Derived from the two flags rather than stored, so it cannot desync from the equivalent checkboxes in the
        // main Debugging UI. The discards deliberately leave SHaRC's inputs alone, so the cache keeps training on
        // full lighting and what stays on screen is exactly what SHaRC produced.
        bool disableNonSharcLighting = discardNonNEELighting && discardNEELighting;
        if (ImGui::Checkbox("Disable all non-SHaRC lighting", &disableNonSharcLighting))
        {
            discardNonNEELighting = disableNonSharcLighting;
            discardNEELighting    = disableNonSharcLighting;
            changed = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Discards both NEE and non-NEE (path/emissive/sky) lighting from the final image, leaving only\nwhat SHaRC's query pass contributes. The training pass and the cache deposits are NOT affected.\nSame two flags as 'Discard path (non-NEE) lighting' / 'Discard NEE lighting' under Debugging;\ntoggling recompiles the PT pipelines.");

        changed |= ImGui::Combo("Debug Mode", &m_settings.DebugMode,
            "Off\0Cached Radiance\0Colored Hash\0Occupancy\0Hash Collisions\0Voxel Size\0Grid Level\0\0");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Anything other than Off compiles the SHaRC debug visualization code into the path tracer, so\nleaving Off <-> non-Off recompiles the PT pipelines and costs a little perf. Switching between\ntwo non-Off modes is free - the macro does not change, only the constant.");
    }


    return changed;
}

void SharcIntegration::UpdatePathStateBuffer(uint32_t trainWidth, uint32_t trainHeight)
{
    // Age out anything retired by an earlier call. c_pathStateRetireFrames is comfortably past the frames in flight,
    // and this runs once per frame, so the list is empty in the steady state.
    for (auto it = m_retiredPathStateBuffers.begin(); it != m_retiredPathStateBuffers.end(); )
    {
        if (it->FramesLeft == 0)
        {
            it = m_retiredPathStateBuffers.erase(it);
        }
        else
        {
            it->FramesLeft--;
            ++it;
        }
    }

    // Zero when SHaRC is off (UpdateResources passes zero dims) - this is the one SHaRC buffer whose size is driven
    // by the render resolution, so hand the memory back rather than sitting on it.
    const uint32_t capacity = trainWidth * trainHeight;
    if (capacity == m_pathStateCapacity)
        return;

    if (m_pathStateBuffer != nullptr)
    {
        RetiredPathStateBuffer retired;
        retired.Buffer     = m_pathStateBuffer;
        retired.Descriptor = std::move(m_pathStateDescriptor);
        retired.FramesLeft = c_pathStateRetireFrames;
        m_retiredPathStateBuffers.push_back(std::move(retired));

        m_pathStateBuffer = nullptr;
    }

    m_pathStateCapacity = capacity;
    if (capacity == 0)
        return;

    const uint64_t byteSize = (uint64_t)capacity * RTXPT_SHARC_PATH_STATE_STRIDE;

    // The shader addresses this with a 32 bit byte offset (RWByteAddressBuffer), so it has to stay under 4 GB.
    // Unreachable at the current DownScaleFactor bounds, but fail loudly rather than silently wrapping if they move.
    if (byteSize > 0xFFFFFFFFull)
        donut::log::fatal("SHaRC path state buffer would be %.1f MB, over the 4 GB addressable by the shader-side 32 bit offsets - raise 'Training Resolution Downscale'.",
            double(byteSize) / (1024.0 * 1024.0));

    nvrhi::BufferDesc bufferDesc;
    bufferDesc.byteSize         = byteSize;
    bufferDesc.canHaveUAVs      = true;
    bufferDesc.canHaveRawViews  = true;
    bufferDesc.keepInitialState = true;
    bufferDesc.initialState     = nvrhi::ResourceStates::UnorderedAccess;
    bufferDesc.debugName        = "SharcPathStateBuffer";
    m_pathStateBuffer = m_device->createBuffer(bufferDesc);

    m_pathStateDescriptor = m_bindlessHeapTable->CreateDescriptorHandle(nvrhi::BindingSetItem::RawBuffer_UAV(0, m_pathStateBuffer));

    // Logged because it is the one SHaRC allocation that scales with resolution, and at the low end of the
    // DownScaleFactor range it is still worth seeing (~133 MB at 4K with a factor of 2).
    donut::log::info("SHaRC path state buffer: %u x %u training threads, %.1f MB", trainWidth, trainHeight, double(byteSize) / (1024.0 * 1024.0));
}

void SharcIntegration::UpdateResources(bool enabled, uint32_t width, uint32_t height)
{
    // The derivation must stay identical to the one in RenderUpdate - the RTXPT_SHADER_DEBUG bounds check in
    // SharcPathStateOffset() is the backstop if the two ever drift. Zero dims when disabled releases the buffer.
    const uint32_t downScaleFactor = DownscaleFactor();
    UpdatePathStateBuffer(enabled ? div_ceil(width,  downScaleFactor) : 0,
                          enabled ? div_ceil(height, downScaleFactor) : 0);
}

void SharcIntegration::UpdateConstants(nvrhi::ICommandList *commandList, SharcConstants& constants, const donut::engine::PlanarView& view)
{
    // Hash-grid anchor. Frozen while "Update View Camera" is off so the cache keeps accumulating
    // against a fixed grid origin even as the render camera moves.
    static float3 cameraPosition = view.GetViewOrigin();
    static float3 cameraPositionPrev = view.GetViewOrigin();

    if (m_settings.EnableUpdate && m_settings.UpdateViewCamera)
    {
        cameraPositionPrev = cameraPosition;
        cameraPosition = view.GetViewOrigin();
    }

    // Fill the SHaRC constants (in ptConsts.sharc and uploads it to the GPU).
    constants = {};
    constants.AccumulationFrameNum = m_settings.AccumulationFrameNum;
    constants.StaleFrameNum = m_settings.StaleFrameFrameNum;
    constants.SharcDebug = m_settings.DebugMode;
    constants.EntriesNum = m_entriesNum;
    constants.DownscaleFactor = (int)DownscaleFactor();      // the clamped value, matching the dispatch
    constants.SceneScale = m_settings.SceneScale;
    constants.MinRoughnessOverride = m_settings.MinRoughnessOverride;
    constants.TerminateThreshold = m_settings.TerminateThreshold;
    constants.MaterialDemodulation = m_settings.EnableMaterialDemodulation ? 1 : 0;
    constants.SeparateEmissive     = m_settings.EnableSeparateEmissive ? 1 : 0;
    // DescriptorHandle::GetIndexInHeap() returns a SIGNED index and yields -1 for an invalid/expired handle, which
    // would become 0xFFFFFFFF once stored in the constants and send the shader at a wild descriptor. Keep it signed
    // until it has been checked.
    const int heapIndices[3] =
    {
        m_hashEntriesDescriptor.GetIndexInHeap(),
        m_accumulationDescriptor.GetIndexInHeap(),
        m_resolvedDescriptor.GetIndexInHeap()
    };

    // Logged on first use and again whenever they move, which happens when the descriptor table grows and
    // relocates - exactly the case that silently invalidates a cached index.
    static int s_reportedHeapIndices[3] = { -2, -2, -2 };
    if (heapIndices[0] != s_reportedHeapIndices[0] || heapIndices[1] != s_reportedHeapIndices[1] || heapIndices[2] != s_reportedHeapIndices[2])
    {
        std::copy(std::begin(heapIndices), std::end(heapIndices), std::begin(s_reportedHeapIndices));
        donut::log::info("SHaRC bindless descriptors: heapBase=%u, heapIndex hash/accum/resolved = %d/%d/%d, tableRelative = %d/%d/%d",
            m_bindlessHeapTable->GetDescriptorTable()->getFirstDescriptorIndexInHeap(),
            heapIndices[0], heapIndices[1], heapIndices[2],
            m_hashEntriesDescriptor.Get(), m_accumulationDescriptor.Get(), m_resolvedDescriptor.Get());
    }

    if (heapIndices[0] < 0 || heapIndices[1] < 0 || heapIndices[2] < 0)
        donut::log::fatal("SHaRC bindless descriptors are invalid (%d/%d/%d) - the shader would read a wild descriptor",
            heapIndices[0], heapIndices[1], heapIndices[2]);

    constants.HashEntriesBindlessIndex  = (uint32_t)heapIndices[0];
    constants.AccumulationBindlessIndex = (uint32_t)heapIndices[1];
    constants.ResolvedBindlessIndex     = (uint32_t)heapIndices[2];

    // Handled separately from the three above because, unlike them, this buffer only exists while SHaRC is enabled.
    if (m_pathStateBuffer != nullptr)
    {
        const int pathStateHeapIndex = m_pathStateDescriptor.GetIndexInHeap();
        if (pathStateHeapIndex < 0)
            donut::log::fatal("SHaRC path state descriptor is invalid (%d) - the shader would read a wild descriptor", pathStateHeapIndex);

        constants.PathStateBindlessIndex = (uint32_t)pathStateHeapIndex;
        constants.PathStateCapacity      = m_pathStateCapacity;
    }
    else
    {
        // Nothing to point at, and nothing fetches it either - only the training pass does, and it does not run
        // while SHaRC is disabled. 0xFFFFFFFF is deliberately not a valid heap index, so if that ever stops being
        // true the bindless validation in Debug builds traps instead of quietly reading someone else's descriptor.
        constants.PathStateBindlessIndex = 0xFFFFFFFFu;
        constants.PathStateCapacity      = 0;
    }
    constants.CameraPosition.xyz() = cameraPosition;
    constants.CameraPositionPrev.xyz() = cameraPositionPrev;

    // Auto-clear when any setting that changes the radiance encoding flips:
    //   - Material demodulation: entries stored in demodulated vs absolute radiance space.
    //   - Separate emissive: entries stored with vs without vertex emission baked in.
    // Also clear on scene reload because geometry associated with the cache does not exist anymore.
    const bool demodulationFlipped     = m_settings.EnableMaterialDemodulation != m_prevEnableMaterialDemodulation;
    const bool separateEmissiveFlipped = m_settings.EnableSeparateEmissive     != m_prevEnableSeparateEmissive;
    if (demodulationFlipped || separateEmissiveFlipped || m_sceneReloaded)
    {
        m_prevEnableMaterialDemodulation = m_settings.EnableMaterialDemodulation;
        m_prevEnableSeparateEmissive     = m_settings.EnableSeparateEmissive;
        m_sceneReloaded = false;
        ClearBuffers(commandList);
    }
}

void SharcIntegration::CreatePipelines(PTPipelineBaker& pipelineBaker, ComputePipelineBaker& computeBaker)
{
    using SM = donut::engine::ShaderMacro;
    assert(m_trainPipeline == nullptr && m_resolveVariant == nullptr);    // unbalanced with DestroyPipelines

    // Dispatched at renderResolution / DownscaleFactor, so it overrides the UI-driven RTXPT_SUB_SAMPLING_SCALE_X/Y
    // globals with its own downscale; RAYGEN_ENTRY then remaps each thread onto a random pixel inside its block.
    // Re-evaluated on every PTPipelineBaker::Update, recompiling only when the produced value changes.
    // (Bounce limits are NOT baked here - they are runtime constants, see ApplyTrainingConstantOverrides.)
    // PT_ENABLE_RUSSIAN_ROULETTE=0 is mandatory rather than a preference - PathTracerSHaRC.hlsli #errors without it.
    // These override the globals by name (PTPipelineVariant::UpdateStart) instead of appending a second -D, which
    // would be a macro redefinition and a hard error under -WX.
    m_trainPipeline = pipelineBaker.CreateVariant("PathTracerSample.hlsl", { SM("PATH_TRACER_MODE", "PT_REFERENCE"), 
        SM("RTXPT_SHARC_TRAIN", "1"), SM("RTXPT_SKIP_STANDARD_OUTPUTS", "1"), SM("RTXPT_SKIP_NEEAT_FEEDBACK", "1"), SM("PT_ENABLE_RUSSIAN_ROULETTE", "0") },
        "SHARC", false,
        [this](std::vector<SM>& macros)
        {
            const std::string downscale = std::to_string(DownscaleFactor());
            macros.push_back({ "RTXPT_SUB_SAMPLING_SCALE_X", downscale });
            macros.push_back({ "RTXPT_SUB_SAMPLING_SCALE_Y", downscale });

            // Training overrides of the same-named globals from Sample::GetPTPipelineGlobalMacros.
            macros.push_back({ "RTXPT_NEE_FULL_SAMPLE_COUNT", std::to_string(dm::clamp(m_settings.NEEFullSamples, 0, RTXPT_LIGHTING_MAX_SAMPLE_COUNT)) });
            macros.push_back({ "RTXPT_FIREFLY_FILTER", m_settings.FireflyFilterEnabled ? "1" : "0" });
        });

    // Resolve pass. The SHaRC buffers cache buffers come from the bindless heap via ptConsts.sharc indices; 
    // g_Const is in the global set at b0, hence all three layouts.
    //
    // NOTE: the shader MUST live under Rtxpt/Shaders. ComputePipelineBaker compiles from SOURCE at runtime, and only
    // Rtxpt/Shaders (plus Donut/NVAPI/RTXDI/External-SHARC) is copied next to the binaries by the
    // ShaderDynamicAssets_CopyAlways target in Rtxpt/CMakeLists.txt. A shader kept outside that tree resolves fine in
    // a dev build (the source tree is right there) and then fails in a packaged build, where it was never shipped.
    // Adding the folder to the baker's monitor paths does NOT fix that - monitoring is not deployment.
    m_resolveVariant = computeBaker.CreateVariant(
        "Libraries/SHaRC/SharcResolve.hlsl",                        // relative to ShadersPath (Rtxpt/Shaders)
        "main",
        {},
        { m_globalBindingLayout, m_globalBindlessLayout, m_globalBindlessHeapLayout },
        "SharcResolve");
}

void SharcIntegration::DestroyPipelines(PTPipelineBaker& pipelineBaker, ComputePipelineBaker& computeBaker)
{
    if (m_trainPipeline != nullptr)
        pipelineBaker.ReleaseVariant(m_trainPipeline);
    if (m_resolveVariant != nullptr)
        computeBaker.ReleaseVariant(m_resolveVariant);
}

void SharcIntegration::ApplyTrainingConstantOverrides(SampleConstants& constants) const
{
    // Not for ptConsts.sharc - the query pass reads those too, so overriding one here desyncs the two sides.
    constants.ptConsts.bounceCount        = (uint)m_settings.BounceCount;
    constants.ptConsts.diffuseBounceCount = (uint)m_settings.DiffuseBounceCount;
    constants.ptConsts.texLODBias        += m_settings.AdditionalTexLODBias;     // additive, not a replacement

    // Same scaling as the main path tracer (Sample::FireflyFilterThreshold), so the UI number means the same in both
    // places. 0 is the shader's "filter off" value, matching RTXPT_FIREFLY_FILTER being baked to 0 alongside it.
    constants.ptConsts.fireflyFilterThreshold = m_settings.FireflyFilterEnabled
        ? Sample::FireflyFilterThreshold(m_settings.FireflyFilterThreshold, constants.ptConsts.preExposedGrayLuminance)
        : 0.0f;
}

void SharcIntegration::RenderUpdate(nvrhi::ICommandList* commandList, nvrhi::IBindingSet* globalBindingSet,
                                    nvrhi::IBindingSet* descriptorTable, nvrhi::IBindingSet* bindlessHeapTable, uint32_t width, uint32_t height,
                                    const SampleConstants& constants, nvrhi::IBuffer* constantBuffer)
{
    RTXPT_GPU_SCOPE(commandList, "PathTraceSharc");

    if (!m_settings.EnableUpdate)
        return;

    if (m_trainPipeline == nullptr)
    {
        donut::log::fatal("SHaRC: update pipeline missing - CreateRTPipelines was not called");
        return;
    }

    if (m_pathStateBuffer == nullptr)
    {
        donut::log::fatal("SHaRC: path state buffer missing - UpdateResources was not called this frame (see Sample::UpdateCustomResources)");
        return;
    }

    if (m_settings.EnableClear)
    {
        RTXPT_GPU_SCOPE(commandList, "PathTraceSharcClear");
        ClearBuffers(commandList);
    }

    {
        RTXPT_GPU_SCOPE(commandList, "PathTraceSharcUpdate");

        // Every frame's training dispatch reuses the same per-thread slots, so consecutive dispatches must not
        // overlap. Today that is inherited transitively from the cache buffer barriers; declare it properly, since
        // the buffer is bindless and so invisible to nvrhi's per-binding-set state tracking.
        commandList->setBufferState(m_pathStateBuffer, nvrhi::ResourceStates::UnorderedAccess);
        commandList->commitBarriers();

        // The buffer is volatile, so a pass sees whatever was written last before its state was set - hence write
        // here, restore after the dispatch. Unconditional even when nothing is overridden: keeps the per-frame
        // version count fixed and the path exercised.
        SampleConstants trainingConstants = constants;
        ApplyTrainingConstantOverrides(trainingConstants);
        commandList->writeBuffer(constantBuffer, &trainingConstants, sizeof(trainingConstants));

        nvrhi::rt::State state;
        state.shaderTable = m_trainPipeline->GetShaderTable();
        state.bindings = { globalBindingSet, descriptorTable, bindlessHeapTable };
        commandList->setRayTracingState(state);

        SampleMiniConstants miniConstants = { uint4(0, 0, 0, 0) };
        commandList->setPushConstants(&miniConstants, sizeof(miniConstants));

        const uint32_t downScaleFactor = DownscaleFactor();  // must match the macros baked into m_trainPipeline
        // Round up: truncating here left a margin of up to downScaleFactor-1 pixels along the right and bottom
        // edges that never fed the cache. Rounding up means the last block can extend past the viewport, which
        // RAYGEN_ENTRY drops after the sub-sampling remap.
        nvrhi::rt::DispatchRaysArguments args;
        args.width  = div_ceil(width,  downScaleFactor);
        args.height = div_ceil(height, downScaleFactor);
        commandList->dispatchRays(args);

        commandList->writeBuffer(constantBuffer, &constants, sizeof(constants));    // restore for the resolve pass and everything after
    }

    if (m_settings.EnableResolve)
    {
        RTXPT_GPU_SCOPE(commandList, "PathTraceSharcResolve");
        Resolve(commandList, globalBindingSet, descriptorTable, bindlessHeapTable);
    }
}

void SharcIntegration::BarrierCacheBuffers(nvrhi::ICommandList* commandList)
{
    // setBufferState only records the requirement; commitBarriers is what emits it. Doing both here keeps the
    // ordering independent of whatever state-setting call happens to come next.
    commandList->setBufferState(m_hashEntriesBuffer, nvrhi::ResourceStates::UnorderedAccess);
    commandList->setBufferState(m_accumulationBuffer, nvrhi::ResourceStates::UnorderedAccess);
    commandList->setBufferState(m_resolvedBuffer, nvrhi::ResourceStates::UnorderedAccess);
    commandList->commitBarriers();
}

void SharcIntegration::ClearBuffers(nvrhi::ICommandList* commandList)
{
    commandList->clearBufferUInt(m_hashEntriesBuffer, m_invalidEntry);
    commandList->clearBufferUInt(m_accumulationBuffer, 0);
    commandList->clearBufferUInt(m_resolvedBuffer, 0);

    // Mandatory, and not merely a UAV barrier: on Vulkan clearBufferUInt leaves the buffer in CopyDest and
    // nothing else would ever transition it back now that the buffers are not in any binding set.
    BarrierCacheBuffers(commandList);
}

void SharcIntegration::Resolve(nvrhi::ICommandList* commandList, nvrhi::IBindingSet* globalBindingSet, nvrhi::IBindingSet* descriptorTable, nvrhi::IBindingSet* bindlessHeapTable)
{
    auto pipeline = (m_resolveVariant != nullptr) ? m_resolveVariant->GetPipeline() : nullptr;
    if (pipeline == nullptr)
        return;     // never compiled successfully (a failed hot reload keeps the previous pipeline) - skip this frame rather than dispatch garbage

    // The training pass wrote hashEntries and accumulation; resolve reads both and read-modify-writes resolved.
    // Nothing issues this for us any more - see BarrierCacheBuffers.
    BarrierCacheBuffers(commandList);

    nvrhi::ComputeState computeState;
    computeState.bindings = { globalBindingSet, descriptorTable, bindlessHeapTable };
    computeState.pipeline = pipeline;
    commandList->setComputeState(computeState);

    // The global binding layout declares push constants at b1; set even though resolve shader
    // does not read them, so the bound root constants are defined.
    SampleMiniConstants miniConstants = { uint4(0, 0, 0, 0) };
    commandList->setPushConstants(&miniConstants, sizeof(miniConstants));

    const uint groupSize = 256;
    const dm::uint2 dispatchSize = { static_cast<uint>(div_ceil(m_entriesNum, groupSize)), 1U };
    commandList->dispatch(dispatchSize.x, dispatchSize.y);
}
