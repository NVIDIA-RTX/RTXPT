/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "IntroSampleUI.h"
#include "IntroSample.h"
#include "SampleCommon/SampleCommon.h"
#include <donut/core/math/math.h>
#include <imgui_internal.h>

namespace
{
    // ----------------------------------------------------------------------------
    // Intro Sample step presets
    // ----------------------------------------------------------------------------
    // The Intro Sample exposes a dropdown that walks the user through the
    // progression from a screen-space baseline (Step 0) to a fully-featured real-time
    // path tracer (Step 8). Each preset defines the full set of toggles plus
    // the recommended AA/denoiser mode. A trailing "Custom" entry is appended
    // so the user can mix and match toggles freely.
    //
    // If the user toggles any individual switch while a preset is selected,
    // the dropdown automatically flips to "Custom" (see SwitchToCustomIfChanged).
    struct IntroStepPreset
    {
        const char* label;
        bool useRTAO;
        bool useRTGI;
        bool useRTReflections;
        IntroDirectLightingMode directLighting;
        bool useSHaRC;
        bool useMISIndirect;
        // Recommended RealtimeAA: 0 = none, 1 = TAA, 2 = DLSS, 3 = DLSS-RR.
        // Sample.cpp will gracefully downgrade this if the requested mode isn't supported.
        int  realtimeAA;
    };

    // Order matters: Step N must be at index N. Keep in sync with the comment
    // block in SampleUI.h's IntroStepIndex docstring.
    static const IntroStepPreset kIntroStepPresets[] = {
        // label                                      RTAO   RTGI   RTRefl Direct                                      SHaRC  MIS    AA
        { "0: Baseline (IBL + SSAO)",                 false, false, false, IntroDirectLightingMode::Unshadowed,       false, false, 1 },
        { "1: + RTAO",                                true,  false, false, IntroDirectLightingMode::Unshadowed,       false, false, 1 },
        { "2: + RTGI (DLSS-RR)",                      true,  true,  false, IntroDirectLightingMode::Unshadowed,       false, false, 3 },
        { "3: + RT Reflections",                      false, true,  true,  IntroDirectLightingMode::Unshadowed,       false, false, 3 },
        { "4: + Uniform Direct Lighting",             false, true,  true,  IntroDirectLightingMode::UniformSampling,  false, false, 3 },
        { "5: + SHaRC (multi-bounce GI)",             false, true,  true,  IntroDirectLightingMode::UniformSampling,  true,  false, 3 },
        { "6: + MIS Indirect Light",                  false, true,  true,  IntroDirectLightingMode::UniformSampling,  true,  true,  3 },
        { "7: + Streaming RIS Direct Lighting",       false, true,  true,  IntroDirectLightingMode::StreamingRIS,     true,  true,  3 },
        { "8: + ReSTIR DI (spatio-temporal)",         false, true,  true,  IntroDirectLightingMode::ReSTIRDI,         true,  true,  3 },
    };
    static constexpr int kIntroStepCount  = (int)(sizeof(kIntroStepPresets) / sizeof(kIntroStepPresets[0]));
    static constexpr int kIntroStepCustom = kIntroStepCount; // dropdown index of the trailing "Custom" entry

    void ApplyPreset(SampleUIData& ui, const IntroStepPreset& p)
    {
        ui.UseRTAO            = p.useRTAO;
        ui.UseRTGI            = p.useRTGI;
        ui.UseRTReflections   = p.useRTReflections;
        ui.IntroDirectLighting = p.directLighting;
        ui.UseSHaRC           = p.useSHaRC;
        ui.UseMISIndirect     = p.useMISIndirect;
        ui.RealtimeAA         = p.realtimeAA;
        ui.ResetAccumulation  = true;
    }

    // Switch to "Custom" if the change unsettled the current preset. Called
    // after every individual toggle. We don't try to detect whether the user
    // happened to land back on a preset - any manual edit moves to Custom and
    // the user must re-select a preset to snap back.
    inline void SwitchToCustomIfChanged(SampleUIData& ui, IntroSampleUIData& introUI, bool changed)
    {
        if (changed && introUI.IntroStepIndex >= 0 && introUI.IntroStepIndex < kIntroStepCount)
            introUI.IntroStepIndex = kIntroStepCustom;
        if (changed)
            ui.ResetAccumulation = true;
    }
}

void ApplyIntroStepPreset(SampleUIData& ui, int stepIndex)
{
    stepIndex = dm::clamp(stepIndex, 0, kIntroStepCount - 1);
    ApplyPreset(ui, kIntroStepPresets[stepIndex]);
}

void IntroSampleUI::DrawSampleUI(float indent, float defItemWidth, const ImVec4& warnColor, const ImVec4& categoryColor)
{
    (void)defItemWidth;
    (void)warnColor;
    (void)categoryColor;

    // IntroSampleUI is only ever constructed by IntroPathTracer::CreateUI, so m_app
    // is always an IntroPathTracer; the Intro-only UI state lives on it.
    IntroSampleUIData& introUI = static_cast<IntroPathTracer&>(m_app).GetIntroUIData();

    if (ImGui::CollapsingHeader("Intro Sample"))
    {
        int modeIndex = (m_ui.RealtimeMode) ? (1) : (0);
        if (ImGui::Combo("Mode", &modeIndex, "Reference\0Realtime\0\0"))
        {
            m_ui.RealtimeMode = (modeIndex != 0);
            m_ui.ResetAccumulation = true;
        }

        if (m_ui.RealtimeMode)
        {
            RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent););

            // --- Step preset dropdown ---------------------------------------------------
            // Build the items array on the fly so the "Custom" entry is always last.
            const char* items[kIntroStepCount + 1];
            for (int i = 0; i < kIntroStepCount; ++i)
                items[i] = kIntroStepPresets[i].label;
            items[kIntroStepCustom] = "Custom";

            introUI.IntroStepIndex = dm::clamp(introUI.IntroStepIndex, 0, kIntroStepCustom);

            if (ImGui::Combo("Step", &introUI.IntroStepIndex, items, kIntroStepCount + 1))
            {
                if (introUI.IntroStepIndex >= 0 && introUI.IntroStepIndex < kIntroStepCount)
                    ApplyPreset(m_ui, kIntroStepPresets[introUI.IntroStepIndex]);
                else
                    m_ui.ResetAccumulation = true;
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Walks the renderer from a screen-space baseline (Step 0) up to ReSTIR DI (Step 8).\n"
                    "Selecting a step applies the matching preset below.\n"
                    "Toggling any option manually switches the dropdown to 'Custom'.");

            // --- Direct Lighting -------------------------------------------------------
            ImGui::TextColored(categoryColor, "Direct Lighting:");
            {
                RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent););

                const char* directLightingItems[] = {
                    "Unshadowed",
                    "Uniform Sampling",
                    "Streaming RIS",
                    "ReSTIR DI"
                };

                int directLightingMode = (int)m_ui.IntroDirectLighting;
                directLightingMode = dm::clamp(directLightingMode, 0, IM_ARRAYSIZE(directLightingItems) - 1);
                bool changed = ImGui::Combo("Direct Lighting Mode", &directLightingMode, directLightingItems, IM_ARRAYSIZE(directLightingItems));
                if (changed)
                {
                    m_ui.IntroDirectLighting = (IntroDirectLightingMode)directLightingMode;
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "Unshadowed evaluates every analytic light; only directional lights cast shadows.\n"
                        "Uniform Sampling picks one analytic light and traces at most one shadow ray.\n"
                        "Streaming RIS draws several analytic light candidates and traces one shadow ray.\n"
                        "ReSTIR DI uses RTXDI initial sampling with reservoir storage (temporal/spatial in Step 8).");
                SwitchToCustomIfChanged(m_ui, introUI, changed);

                if (m_ui.IntroDirectLighting == IntroDirectLightingMode::ReSTIRDI)
                {
                    RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent););

                    bool temporalChanged = ImGui::Checkbox("Temporal Reuse", &introUI.IntroReSTIRTemporalReuse);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Reuse the previous frame's reservoir via screen-space reprojection.\n"
                            "Disable to fall back to single-frame initial sampling (noisier, no history).");
                    SwitchToCustomIfChanged(m_ui, introUI, temporalChanged);

                    bool spatialChanged = ImGui::Checkbox("Spatial Reuse", &introUI.IntroReSTIRSpatialReuse);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Combine the current reservoir with a few similar-surface neighbors each frame.\n"
                            "Works on top of (or without) temporal reuse - both checkboxes are independent.");
                    SwitchToCustomIfChanged(m_ui, introUI, spatialChanged);

                    bool historyChanged = ImGui::SliderInt("History Length", &introUI.IntroReSTIRMaxHistoryLength, 1, 40);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Maximum previous-frame reservoir M relative to this frame's initial M (1).\n"
                            "Higher retains more history weight; it can reduce noise but increase correlation and lag.");
                    SwitchToCustomIfChanged(m_ui, introUI, historyChanged);

                    bool radiusChanged = ImGui::SliderFloat("Spatial Radius", &introUI.IntroReSTIRSpatialSamplingRadius, 4.0f, 64.0f, "%.1f px");
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Radius in internal render pixels of the disk used to pick spatial neighbors.\n"
                            "Larger reaches farther across surfaces; depth/normal checks reject incompatible neighbors.");
                    SwitchToCustomIfChanged(m_ui, introUI, radiusChanged);

                    bool analyticSamplesChanged = ImGui::SliderInt("Analytic Local Candidates", &introUI.IntroReSTIRNumAnalyticLocalSamples, 0, 32);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Number of analytic local-light (point/spot/sphere) RIS candidates drawn per pixel\n"
                            "in the Initial pass. Higher = better initial light selection, more cost.");
                    SwitchToCustomIfChanged(m_ui, introUI, analyticSamplesChanged);

                    bool triangleSamplesChanged = ImGui::SliderInt("Triangle Local Candidates", &introUI.IntroReSTIRNumTriangleLocalSamples, 0, 32);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Number of emissive-triangle RIS candidates drawn per pixel in the Initial pass.\n"
                            "Sampled as a stream independent from the analytic locals.");
                    SwitchToCustomIfChanged(m_ui, introUI, triangleSamplesChanged);

                    bool infiniteSamplesChanged = ImGui::SliderInt("Infinite Light Candidates", &introUI.IntroReSTIRNumInfiniteSamples, 0, 8);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Number of infinite (directional) light candidates drawn per pixel in the Initial pass.");
                    SwitchToCustomIfChanged(m_ui, introUI, infiniteSamplesChanged);

                    bool sampleEnvChanged = ImGui::Checkbox("Sample Environment Map", &introUI.IntroReSTIRSampleEnvironmentMap);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Importance-sample the environment map as an extra RIS stream in the Initial pass,\n"
                            "so sky/HDRI samples participate in direct-light reuse. With MIS Indirect Light enabled,\n"
                            "covered primary environment connections are then excluded from indirect tracing.");
                    SwitchToCustomIfChanged(m_ui, introUI, sampleEnvChanged);

                    if (introUI.IntroReSTIRSampleEnvironmentMap)
                    {
                        bool envSamplesChanged = ImGui::SliderInt("Environment Candidates", &introUI.IntroReSTIRNumEnvironmentSamples, 0, 32);
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip(
                                "Number of environment-map RIS candidates drawn per pixel in the Initial pass.");
                        SwitchToCustomIfChanged(m_ui, introUI, envSamplesChanged);
                    }

                    bool spatialSamplesChanged = ImGui::SliderInt("Spatial Neighbor Candidates", &introUI.IntroReSTIRSpatialNumSamples, 1, 8);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Number of neighbor reservoirs combined by the Spatial pass (RTXDI numSamples).\n"
                            "Low-M pixels (fresh disocclusions) still get a boosted count internally.");
                    SwitchToCustomIfChanged(m_ui, introUI, spatialSamplesChanged);

                    bool depthThreshChanged = ImGui::SliderFloat("Depth Threshold", &introUI.IntroReSTIRDepthThreshold, 0.001f, 1.0f, "%.3f");
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Relative view-depth tolerance for accepting a temporal/spatial neighbor.\n"
                            "Lower = stricter (less leaking across depth discontinuities), noisier.");
                    SwitchToCustomIfChanged(m_ui, introUI, depthThreshChanged);

                    bool normalThreshChanged = ImGui::SliderFloat("Normal Threshold", &introUI.IntroReSTIRNormalThreshold, 0.0f, 1.0f, "%.2f");
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Minimum dot(N_current, N_neighbor) to accept a temporal/spatial neighbor.\n"
                            "Higher = stricter (rejects across normal discontinuities), noisier.");
                    SwitchToCustomIfChanged(m_ui, introUI, normalThreshChanged);

                    bool discountNaiveChanged = ImGui::Checkbox("Discount Naive Samples", &introUI.IntroReSTIRDiscountNaiveSamples);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Skip very-low-confidence (naive) neighbor reservoirs during spatial reuse.\n"
                            "Reduces bias/noise from neighbors that have almost no history.");
                    SwitchToCustomIfChanged(m_ui, introUI, discountNaiveChanged);

                    bool temporalRTChanged = ImGui::Checkbox("Temporal Ray-Traced Bias Correction", &introUI.IntroReSTIRTemporalBiasRaytraced);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Basic (off): normalizes reuse using target values at the current and previous surfaces.\n"
                            "Ray Traced (on): also tests visibility for the temporal normalization term.\n"
                            "Uses the current TLAS, so visibility under motion remains approximate.");
                    SwitchToCustomIfChanged(m_ui, introUI, temporalRTChanged);

                    bool permutationChanged = ImGui::Checkbox("Permutation Sampling", &introUI.IntroReSTIREnablePermutationSampling);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Jitter the temporal reprojection lookup with a per-frame permutation.\n"
                            "Decorrelates history to reduce structured noise; can add noise on thin/high-detail geometry.");
                    SwitchToCustomIfChanged(m_ui, introUI, permutationChanged);

                    bool boilingChanged = ImGui::Checkbox("Boiling Filter", &introUI.IntroReSTIREnableBoilingFilter);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Group-local filter (in the Temporal pass) that discards outlier reservoirs whose weight\n"
                            "is far above the tile average. Can suppress fireflies but also remove valid bright samples.\n"
                            "Requires temporal reuse; disabled by default.");
                    SwitchToCustomIfChanged(m_ui, introUI, boilingChanged);

                    if (introUI.IntroReSTIREnableBoilingFilter)
                    {
                        RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent););
                        bool boilingStrengthChanged = ImGui::SliderFloat("Boiling Strength", &introUI.IntroReSTIRBoilingFilterStrength, 0.01f, 1.0f, "%.2f");
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip(
                                "Lower = more aggressive culling (discards more outliers); higher = gentler.\n"
                                "A reservoir is discarded when its weight exceeds tileAverage / strength.");
                        SwitchToCustomIfChanged(m_ui, introUI, boilingStrengthChanged);
                    }
                }
            }

            // --- Indirect Lighting -----------------------------------------------------
            ImGui::TextColored(categoryColor, "Indirect Lighting:");
            {
                RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent););

                bool misChanged = ImGui::Checkbox("MIS Indirect Light", &m_ui.UseMISIndirect);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("When enabled, replaces separate reflections and RTGI with a single\nunified pass that stochastically selects specular or diffuse rays.");
                SwitchToCustomIfChanged(m_ui, introUI, misChanged);

                // Reflections + RTGI toggles. Disabled (but still visible) when MIS is on,
                // since MIS supersedes both.
                {
                    UI_SCOPED_DISABLE(m_ui.UseMISIndirect);
                    bool reflChanged = ImGui::Checkbox("Use RT Reflections", &m_ui.UseRTReflections);
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                        ImGui::SetTooltip("When enabled, uses ray-traced reflections instead of screen-space reflections (SSR).\nDisabled while MIS Indirect Light is on (MIS handles reflections internally).");
                    SwitchToCustomIfChanged(m_ui, introUI, reflChanged);

                    bool gIChanged = ImGui::Checkbox("Use RT Global Illumination", &m_ui.UseRTGI);
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                        ImGui::SetTooltip("When enabled, traces one-bounce diffuse GI rays (denoised by DLSS-RR).\nDisabled while MIS Indirect Light is on.");
                    SwitchToCustomIfChanged(m_ui, introUI, gIChanged);

                    if (!m_ui.UseMISIndirect && m_ui.UseRTGI)
                    {
                        RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent););

                        // Kept in sync with kEnvGGXMaxMip in IntroSample.cpp; the host also clamps.
                        constexpr int kEnvMipUIMax = 7;
                        int envMip = (int)m_ui.RTGIEnvMip;
                        bool envMipChanged = ImGui::SliderInt("Env Map Mip", &envMip, 0, kEnvMipUIMax);
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Environment-map mip used on both a GI ray's miss and its hit's own environment sample.\nLower = sharper environment detail but potentially noisier with bright, small features.\nHigher = less noise at the cost of blurring environment lighting. Default: 5.\nDoes not change the directional sun or the background seen directly by the camera.");
                        m_ui.RTGIEnvMip = (uint32_t)envMip;
                        SwitchToCustomIfChanged(m_ui, introUI, envMipChanged);

                        // Debug toggle is intentionally not part of any preset.
                        if (ImGui::Checkbox("Debug RTGI (irradiance only)", &m_ui.DebugRTGI))
                        {
                            if (m_ui.DebugRTGI)
                            {
                                m_ui.DebugIndirectSpecular = false;
                                m_ui.DebugSHaRCCache = false;
                            }
                            m_ui.ResetAccumulation = true;
                            m_ui.ResetRealtimeCaches = true;
                        }
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Displays the GI buffer before primary-surface albedo is applied (an estimate of irradiance / pi for diffuse surfaces).\nIncludes ray hits and environment misses; replaces the combined lighting, but still passes through post-processing.\nIndependent of step presets: turn off to restore the normal view.");
                    }
                }

                // Available for SSR, RT reflections and MIS; independent of step presets.
                if (ImGui::Checkbox("Debug indirect specular", &m_ui.DebugIndirectSpecular))
                {
                    if (m_ui.DebugIndirectSpecular)
                    {
                        m_ui.DebugRTGI = false;
                        m_ui.DebugSHaRCCache = false;
                    }
                    m_ui.ResetAccumulation = true;
                    m_ui.ResetRealtimeCaches = true;
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Displays only indirect specular, including Fresnel and energy compensation.\nWorks with SSR, RT reflections and MIS; still passes through denoising and post-processing.\nIndependent of step presets: turn off to restore the normal view.");

                // SHaRC is meaningful for both MIS Indirect and standalone RTGI.
                {
                    const bool sharcRelevant = m_ui.UseMISIndirect || m_ui.UseRTGI;
                    UI_SCOPED_DISABLE(!sharcRelevant);
                    bool sharcChanged = ImGui::Checkbox("Use SHaRC (Radiance Cache)", &m_ui.UseSHaRC);
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                        ImGui::SetTooltip("When enabled, sparse multi-bounce path tracing fills a world-space radiance cache.\nIndirect lighting queries the cache on hits instead of computing full lighting.\nRequires MIS Indirect or RTGI to be enabled.");
                    SwitchToCustomIfChanged(m_ui, introUI, sharcChanged);

                    if (sharcRelevant && m_ui.UseSHaRC)
                    {
                        RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent););
                        // Debug toggle is intentionally not part of any preset.
                        if (ImGui::Checkbox("Debug SHaRC cache", &m_ui.DebugSHaRCCache))
                        {
                            if (m_ui.DebugSHaRCCache)
                            {
                                m_ui.DebugRTGI = false;
                                m_ui.DebugIndirectSpecular = false;
                            }
                            m_ui.ResetAccumulation = true;
                            m_ui.ResetRealtimeCaches = true;
                        }
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("When enabled, replaces deferred lighting with a visualization of the cached radiance at first-hit positions.\nShows where the cache has data (colored) vs. cache misses (black).");
                    }
                }

                // Match the indirect-lighting consumers in SampleRenderCode: diffuse environment
                // fallback without RTGI, or specular environment fallback with SSR.
                const bool aoUsed = !m_ui.UseMISIndirect && (!m_ui.UseRTGI || !m_ui.UseRTReflections);
                {
                    UI_SCOPED_DISABLE(!aoUsed);
                    bool aoChanged = ImGui::Checkbox("Use RTAO (Ray-Traced AO)", &m_ui.UseRTAO);
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                        ImGui::SetTooltip("When enabled, uses ray-traced AO (inline ray queries) instead of screen-space GTAO.\nAO attenuates the diffuse environment fallback without RTGI, and the specular environment fallback with SSR.\nDisabled when both GI and reflections are ray-traced, or when MIS Indirect handles both terms.\nDirect-lighting mode does not affect whether AO is needed.");
                    SwitchToCustomIfChanged(m_ui, introUI, aoChanged);

                    if (aoUsed && m_ui.UseRTAO)
                    {
                        RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent););
                        bool aoRadiusChanged = ImGui::SliderFloat("AO Radius", &m_ui.AORadius, 1.0f, 200.0f, "%.1f");
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("World-space distance the AO ray searches for an occluder.\nLarger values reach farther occluders but make the AO term less local.");
                        SwitchToCustomIfChanged(m_ui, introUI, aoRadiusChanged);
                    }
                }
            }

            // --- Post processing -------------------------------------------------------
            ImGui::TextColored(categoryColor, "Post processing:");
            {
                RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent); );

#if DONUT_WITH_STREAMLINE
                const bool dlssAvailable   = m_ui.IsDLSSSuported;
                const bool dlssRRAvailable = m_ui.IsDLSSRRSupported;
#else
                const bool dlssAvailable   = false;
                const bool dlssRRAvailable = false;
#endif
                const char* aaItems[] = { "Disabled", "TAA", "DLSS", "DLSS-RR" };
                const int aaItemCount = IM_ARRAYSIZE(aaItems);

                m_ui.RealtimeAA = dm::clamp(m_ui.RealtimeAA, 0, dlssAvailable ? aaItemCount : 1);

                if (ImGui::BeginCombo("AA/SR/Denoising", aaItems[m_ui.RealtimeAA]))
                {
                    for (int i = 0; i < aaItemCount; i++)
                    {
                        bool enabled = false;
                        enabled |= i < 2;
                        enabled |= (i == 2) && dlssAvailable;
                        enabled |= (i == 3) && dlssRRAvailable;
                        UI_SCOPED_DISABLE(!enabled);

                        bool isSelected = (m_ui.RealtimeAA == i);
                        if (ImGui::Selectable(aaItems[i], isSelected))
                            m_ui.RealtimeAA = i;
                        if (isSelected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip(
                    "TAA        - generic temporal anti-aliasing\n"
                    "DLSS       - Nvidia Deep Learning Super Sampling (lower resolution render + upscale)\n"
                    "DLAA       - Nvidia Deep Learning Anti Aliasing (full resolution render)\n"
                    "DLSS-RR    - DLSS + Ray Reconstruction (lower resolution render + denoise & upscale)\n"
                    "\nIndividual DLSS options available under global `DLSS` options"
                );

#if DONUT_WITH_STREAMLINE
                if (m_ui.RealtimeAA == 2 || m_ui.RealtimeAA == 3)
                {
                    RAII_SCOPE(ImGui::Indent(indent); ImGui::PushID("PPDLSSQual"); , ImGui::Unindent(indent); ImGui::PopID(););
                    m_ui.DLSSMode = DLSSModeUI(m_ui.DLSSMode);
                }
#endif
            }
        }
    }
}
