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

#include "SampleUI.h"

// ----------------------------------------------------------------------------
// Intro Sample launch step
// ----------------------------------------------------------------------------
// SINGLE SOURCE OF TRUTH: change this one line to change which step the Intro
// Sample boots into. Every launch default (RT toggles, direct-lighting mode,
// recommended AA) is derived from the matching entry in kIntroStepPresets
// (see IntroSampleUI.cpp) via ApplyIntroStepPreset().
static constexpr int kIntroDefaultStep = 8;

// Applies the full step preset (all RT toggles + the recommended AA mode) for
// the given step index to the UI state. The index is clamped to a valid step.
void ApplyIntroStepPreset(SampleUIData& ui, int stepIndex);

// ----------------------------------------------------------------------------
// Intro Sample-specific UI state
// ----------------------------------------------------------------------------
// State that only the Intro Sample cares about (the step dropdown selection and
// the ReSTIR DI learning-path knobs). Kept out of the shared SampleUIData so the
// base sample doesn't carry Intro-only fields. Owned by IntroPathTracer (the
// renderer and per-frame consumer). IntroSampleUI reaches it via
// IntroPathTracer::GetIntroUIData().
struct IntroSampleUIData
{
    int                                 IntroStepIndex = 0;  // dropdown selection; actual launch step is kIntroDefaultStep, applied by the ctor/CreateUI
    bool                                IntroReSTIRTemporalReuse = true;     // contributes to ReSTIRDI_ResamplingMode selection
    bool                                IntroReSTIRSpatialReuse  = true;     // contributes to ReSTIRDI_ResamplingMode selection
    int                                 IntroReSTIRMaxHistoryLength = 20;    // RTXDI_DITemporalResamplingParameters::maxHistoryLength
    float                               IntroReSTIRSpatialSamplingRadius = 32.0f; // RTXDI_DISpatialResamplingParameters::samplingRadius (pixels)
    int                                 IntroReSTIRNumAnalyticLocalSamples = 8;  // ReSTIR DI Initial: analytic local-light RIS candidate count
    int                                 IntroReSTIRNumTriangleLocalSamples =8;  // ReSTIR DI Initial: emissive-triangle RIS candidate count
    int                                 IntroReSTIRNumInfiniteSamples = 1;       // ReSTIR DI Initial: infinite (directional) light candidate count
    bool                                IntroReSTIRSampleEnvironmentMap = false; // ReSTIR DI Initial: importance-sample the environment map as an extra RIS stream
    int                                 IntroReSTIRNumEnvironmentSamples = 1;    // ReSTIR DI Initial: environment-map RIS candidate count
    int                                 IntroReSTIRSpatialNumSamples = 2;        // RTXDI_DISpatialResamplingParameters::numSamples (neighbors per pixel)
    bool                                IntroReSTIRDiscountNaiveSamples = false; // RTXDI_DISpatialResamplingParameters::discountNaiveSamples
    bool                                IntroReSTIRTemporalBiasRaytraced = true;// temporal biasCorrectionMode: false=Basic, true=Raytraced
    bool                                IntroReSTIREnablePermutationSampling = true; // RTXDI_DITemporalResamplingParameters::enablePermutationSampling
    float                               IntroReSTIRDepthThreshold = 0.1f;        // shared temporal+spatial depthThreshold
    float                               IntroReSTIRNormalThreshold = 0.5f;       // shared temporal+spatial normalThreshold
    bool                                IntroReSTIREnableBoilingFilter = false;  // RTXDI_BoilingFilterParameters::enableBoilingFilter (Temporal pass)
    float                               IntroReSTIRBoilingFilterStrength = 0.2f; // RTXDI_BoilingFilterParameters::boilingFilterStrength (0..1]
};

class IntroSampleUI : public SampleUI
{
public:
    using SampleUI::SampleUI;
    void DrawSampleUI(float indent, float defItemWidth, const struct ImVec4& warnColor, const struct ImVec4& categoryColor) override;
};
