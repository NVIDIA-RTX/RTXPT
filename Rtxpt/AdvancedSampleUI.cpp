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
#include "SampleCommon/SampleCommon.h"
#include "SampleGame/GameScene.h"
#include "Sharc/SharcIntegration.h"
#include <imgui_internal.h>

using namespace donut::app;
using namespace donut::engine;

#define RESET_ON_CHANGE(code) do{if (code) m_ui.ResetAccumulation = true;} while(false)

AdvancedSampleUI::AdvancedSampleUI(donut::app::DeviceManager * deviceManager, class SampleBaseApp& baseApp, class Sample& app, SampleUIData & ui, bool NVAPI_SERSupported, const struct CommandLineOptions& cmdLine)
    : SampleUI(deviceManager, baseApp, app, ui, NVAPI_SERSupported, cmdLine)
{
    ui.EnableAsyncPrefix = cmdLine.AsyncPrefix;
}

void AdvancedSampleUI::DrawSampleUI(float indent, float defItemWidth, const ImVec4& warnColor, const ImVec4& categoryColor)
{
    if (ImGui::CollapsingHeader("Camera", 0/*ImGuiTreeNodeFlags_DefaultOpen*/))
    {
        RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent); );
        std::vector<std::string> options; options.push_back("Free flight");
        for (uint i = 0; i < m_app.GetSceneCameraCount(); i++)
            options.push_back(m_app.GetSceneCameraName(i));
        uint& currentlySelected = m_app.SelectedCameraIndex();
        currentlySelected = std::min(currentlySelected, (uint)m_app.GetSceneCameraCount() - 1);
        if (ImGui::BeginCombo("Motion", options[currentlySelected].c_str()))
        {
            for (uint i = 0; i < m_app.GetSceneCameraCount(); i++)
            {
                bool is_selected = i == currentlySelected;
                if (ImGui::Selectable(options[i].c_str(), is_selected))
                    currentlySelected = i;
                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        if (currentlySelected == 0)
        {
            ImGui::Text("Camera position: ");
            RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent); );
            if (ImGui::Button("Save to file", ImVec2(ImGui::GetFontSize() * 9.0f, ImGui::GetTextLineHeightWithSpacing()))) m_app.SaveCurrentCamera(); ImGui::SameLine();
            if (ImGui::Button("Load from file", ImVec2(ImGui::GetFontSize() * 9.0f, ImGui::GetTextLineHeightWithSpacing()))) m_app.LoadCurrentCamera();
            if (ImGui::Button("Copy TO clipboard", ImVec2(ImGui::GetFontSize() * 9.0f, ImGui::GetTextLineHeightWithSpacing()))) ImGui::SetClipboardText(m_app.GetCurrentCameraPosDirUp().c_str()); ImGui::SameLine();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Ctrl+Shift+C");
            const char* cpbrdtxt = ImGui::GetClipboardText();
            if (ImGui::Button("Copy FROM clipboard", ImVec2(ImGui::GetFontSize() * 9.0f, ImGui::GetTextLineHeightWithSpacing()))) m_app.SetCurrentCameraPosDirUp(cpbrdtxt ? cpbrdtxt : "");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Ctrl+Shift+V");

        }

#if 1
        RESET_ON_CHANGE(ImGui::InputFloat("Aperture", &m_ui.CameraAperture, 0.001f, 0.01f, "%.4f"));
        m_ui.CameraAperture = dm::clamp(m_ui.CameraAperture, 0.0f, 1.0f);

        RESET_ON_CHANGE(ImGui::InputFloat("Focal Distance", &m_ui.CameraFocalDistance, 0.1f));
        m_ui.CameraFocalDistance = dm::clamp(m_ui.CameraFocalDistance, 0.001f, 1e16f);
        ImGui::SliderFloat("Keyboard move speed", &m_ui.CameraMoveSpeed, 0.1f, 10.0f);

        float cameraFOV = dm::degrees(m_app.GetCameraVerticalFOV());
        if (ImGui::InputFloat("Vertical FOV", &cameraFOV, 0.1f))
        {
            cameraFOV = dm::clamp(cameraFOV, 0.5f, 179.0f);
            m_ui.ResetAccumulation = true;
            m_app.SetCameraVerticalFOV(dm::radians(cameraFOV));
        }

        RESET_ON_CHANGE(ImGui::InputFloat("CameraAntiRRSleepJitter", &m_ui.CameraAntiRRSleepJitter, 0.001f));
        m_ui.CameraAntiRRSleepJitter = clamp(m_ui.CameraAntiRRSleepJitter, 0.0f, 1.0f);
#endif
    }

    if (ImGui::CollapsingHeader("Light pre-processing and sampling", 0/*ImGuiTreeNodeFlags_DefaultOpen*/))
    {
        RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent););

        if (!m_ui.UseNEE)
            ImGui::TextColored(warnColor, "NOTE: NEE inactive (enable in `Path Tracer -> Next Event Estimation` settings).");

        ImGui::TextColored(categoryColor, "Info and statistics:");

        {
            RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent););
            if (m_app.GetLightsBaker() != nullptr) // local lights baker can legally be nullptr
                m_ui.ResetAccumulation |= m_app.GetLightsBaker()->InfoGUI(indent);
        }


        //ImGui::TextColored(categoryColor, "Distant lighting (envmap+directional):");
        {
            RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent););
            if (ImGui::CollapsingHeader("Distant lighting (envmap+directional)", 0/*ImGuiTreeNodeFlags_DefaultOpen*/))
            {
                RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent););
                if (m_app.GetEnvMapBaker() != nullptr) // envmap baker can legally be nullptr
                    m_ui.ResetAccumulation |= m_app.GetEnvMapBaker()->DebugGUI(indent);
            }
        }

        ImGui::TextColored(categoryColor, "Importance sampling:");
        {
            RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent););
            if (m_app.GetLightsBaker() != nullptr) // local lights baker can legally be nullptr
            {
                if (m_ui.NEEType != 2)
                {
                    ImGui::TextWrapped("NOTE: NEE-AT inactive (enable in `Path Tracer -> Next Event Estimation` settings).");
                }
                else
                {
                    ImGui::TextColored(categoryColor, "NEE-AT settings:");
                    {
                        RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent););

                        RESET_ON_CHANGE(ImGui::SliderFloat("Global feedback weight", &m_ui.NEEAT_GlobalTemporalFeedbackWeight, 0.0f, 0.95f));
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("How much to rely on last frame's usage statistic as opposed to simple power based sampling.\nSome power based sampling is essential to allow new lights to be considered.");

                        RESET_ON_CHANGE(ImGui::SliderFloat("Local to global sampler ratio", &m_ui.NEEAT_LocalToGlobalSampleRatio, 0.0f, 0.95f));

                        uint localCandidateSamples = NEEAT::ComputeCandidateSampleLocalCount(m_ui.ActualNEEAT_LocalToGlobalSampleRatio(), m_ui.NEECandidateSamples);
                        uint globalCandidateSamples = NEEAT::ComputeCandidateSampleGlobalCount(m_ui.ActualNEEAT_LocalToGlobalSampleRatio(), m_ui.NEECandidateSamples);

                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("When drawing new light candidate samples, how many to draw from Global versus Local (tile) samplers.\n"
                            "Current total candidate sample count is %d, and out of those %d will be Local and %d will be Global",
                            m_ui.NEECandidateSamples, localCandidateSamples, globalCandidateSamples);

                        // ImGui::SliderFloat("BSDF vs NEE-AT MIS boost", &m_ui.NEEAT_MIS_Boost, 0.0f, 1000.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
                        // if (ImGui::IsItemHovered()) ImGui::SetTooltip("Tweak the MIS to give more power to NEE-AT (>1) or to BSDF sampled emissives (<1);\nuseful since NEE-AT is shadow aware and boosting it can provide better overall sampling quality");

                        ImGui::SliderFloat("Distant vs Local initial importance", &m_ui.NEEAT_Distant_vs_Local_Importance, 0.01f, 100.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("The higher the setting, the more initial importance will be given to environment map / sunlight vs local scene lights and vice versa.");

                        RESET_ON_CHANGE(ImGui::Combo("Quality preset", &m_ui.NEEAT_QualityPreset, "Low\0Medium\0High\0\0"));
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Local (tile) sampler tile size and light count - see NEEAT_QUALITY_PRESET in NEEATConfig.h.\nChanging it recompiles the NEE-AT and path tracer shaders.");
                    }
                }

                ImGui::TextColored(categoryColor, "Debugging:");
                {
                    RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent););
                    if (m_app.GetLightsBaker() != nullptr) // local lights baker can legally be nullptr
                        m_ui.ResetAccumulation |= m_app.GetLightsBaker()->DebugGUI(indent);
                }
            }
        }
    }


    if (ImGui::CollapsingHeader("Path Tracer", ImGuiTreeNodeFlags_DefaultOpen))
    {
        RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent); );

        int modeIndex = (m_ui.RealtimeMode) ? (1) : (0);
        if (ImGui::Combo("Mode", &modeIndex, "Reference\0Realtime\0\0"))
        {
            m_ui.RealtimeMode = (modeIndex != 0);
            m_ui.ResetAccumulation = true;
        }

        ImGui::TextColored(categoryColor, "Setup:");
        {
            RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent); );

            if (m_ui.RealtimeMode)
            {
                if (ImGui::Button("Reset##RTMACC"))
                    m_ui.ResetRealtimeCaches = true;
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reset all temporal caches in denoising, lighting and etc");
                ImGui::SameLine();

                {
                    UI_SCOPED_DISABLE((m_ui.ActualUseReSTIRDI() || m_ui.ActualUseReSTIRGI()));
                    ImGui::InputInt("Samples per pixel", &m_ui.RealtimeSamplesPerPixel);
                    m_ui.RealtimeSamplesPerPixel = dm::clamp(m_ui.RealtimeSamplesPerPixel, 1, 64);
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                        ImGui::SetTooltip("How many full paths to trace per pixel from the primary surface\n(camera ray is not re-cast so there is no added AA)\n(currently incompatible with ReSTIR DI & ReSTIR GI)");

                    if (m_ui.RealtimeSamplesPerPixel > 1)
                        m_ui.RealtimeSubSamplingMode = 0;
                }

                {
                    UI_SCOPED_DISABLE((m_ui.ActualUseReSTIRDI() || m_ui.ActualUseReSTIRGI()) || m_ui.RealtimeSamplesPerPixel != 1);
                    RESET_ON_CHANGE(ImGui::Combo("Sub-sampling", (int*)&m_ui.RealtimeSubSamplingMode, "Disabled\0Three quarters\0Half\0Quarter\0\0"));
                    m_ui.RealtimeSubSamplingMode = dm::clamp(m_ui.RealtimeSubSamplingMode, 0, 3);
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                        ImGui::SetTooltip("Opposite of 'Samples per pixel', we can compute lighting for only three quarters, half or a quarter of pixels for better performance.\n(currently incompatible with ReSTIR DI & ReSTIR GI)");

                    if (m_ui.RealtimeSubSamplingMode > 0)
                    {
                        RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent); );

                        RESET_ON_CHANGE(ImGui::Combo("Pattern", &m_ui.RealtimeSubSamplingPattern, "Random per block\0Stochastic (compacted)\0Block stratified\0Sobol (0,2) per block\0\0"));
                        m_ui.RealtimeSubSamplingPattern = dm::clamp(m_ui.RealtimeSubSamplingPattern, 0, (int)SubSamplingPattern::Count - 1);
                        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                            ImGui::SetTooltip("Which pixels get a path (traced pixels are weighted by 2x / 4x / 4/3x, the rest get no noisy radiance):\n"
                                              " Random per block: one random pixel in every 2x1 (half) / 2x2 (quarter) block, or all but one\n"
                                              "   random pixel of every 2x2 block (three quarters) - the original method\n"
                                              " Stochastic (compacted): independent white noise coin flip per pixel; selected pixels are\n"
                                              "   stream-compacted (per group, then globally with atomics) into a 1D ray launch\n"
                                              " Block stratified: every NxN block gets exactly N*N/2 (half), N*N/4 (quarter) or 3*N*N/4\n"
                                              "   (three quarters) pixels, at random distinct positions\n"
                                              " Sobol (0,2) per block: quasirandom - each NxN block takes the next points of its Owen-scrambled\n"
                                              "   Sobol (0,2) sequence; stratified at every power of two aspect, and every pixel is traced\n"
                                              "   exactly once per 2 (half) / 4 (quarter) frames, or 3 times per 4 frames (three quarters)\n"
                                              "Switching patterns recompiles the path tracer's fill pass shaders.");

#if 0 // a bit too advanced for regular UI perhaps - keep it hidden but very useful for testing
                        if (m_ui.RealtimeSubSamplingPattern == (int)SubSamplingPattern::BlockStratified || m_ui.RealtimeSubSamplingPattern == (int)SubSamplingPattern::SobolBlock)
                        {
                            int blockSizeIndex = (m_ui.SubSamplingStratifiedBlockSize() == 16) ? 2 : ((m_ui.SubSamplingStratifiedBlockSize() == 8) ? 1 : 0);
                            if (ImGui::Combo("Block size", &blockSizeIndex, " 4x4\0 8x8\0 16x16\0\0"))
                            {
                                m_ui.RealtimeSubSamplingStratifiedBlockSize = 4 << dm::clamp(blockSizeIndex, 0, 2);
                                m_ui.ResetAccumulation = true;
                            }
                            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                                ImGui::SetTooltip("Size of the block that always gets the same number of samples");
                        }
#endif
                    }
                }
            }
            else
            {
                RESET_ON_CHANGE(ImGui::Button("Reset##REFMACC"));
                ImGui::SameLine();
                RESET_ON_CHANGE(ImGui::InputInt("Sample count", &m_ui.AccumulationTarget));
                m_ui.AccumulationTarget = dm::clamp(m_ui.AccumulationTarget, 1, 4 * 1024 * 1024); // this max is beyond float32 precision threshold; expect some banding creeping in when using more than 500k samples
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Number of path samples per pixel to collect");
                ImGui::Text("Accumulated samples: %d (out of %d target)", m_app.GetAccumulationSampleIndex(), m_ui.AccumulationTarget);
                ImGui::Text("(avg frame time: %.3fms)", m_app.GetAvgTimePerFrame() * 1000.0f);

                RESET_ON_CHANGE(ImGui::Checkbox("Pre-warm real-time caches", &m_ui.AccumulationPreWarmRealtimeCaches));
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("If enabled, various lighting and etc systems will be pre-warmed before sample 0 is \naccumulated; otherwise they're reset and initial few samples will be lower quality.");

                RESET_ON_CHANGE(ImGui::Checkbox("Jitter anti-aliasing", &m_ui.AccumulationAA));
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Each sample will have a random, per pixel jitter emulating box filter\nTODO: add option for Gaussian distribution for better AA");

                m_ui.ResetRealtimeCaches |= m_ui.ResetAccumulation; // if there's a reset for any reason whilst we're in reference mode, reset the realtime caches too for determinism
            }

            RESET_ON_CHANGE(ImGui::InputInt("Max bounces", &m_ui.BounceCount));
            m_ui.BounceCount = dm::clamp(m_ui.BounceCount, 0, MAX_BOUNCE_COUNT);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Max number of all bounces (including NEE and diffuse bounces)");
            RESET_ON_CHANGE(ImGui::InputInt("Max diffuse bounces", &m_ui.DiffuseBounceCount));
            m_ui.DiffuseBounceCount = dm::clamp(m_ui.DiffuseBounceCount, 0, MAX_BOUNCE_COUNT);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Max number of diffuse bounces (diffuse lobe and specular with roughness > 0.25 or similar depending on settings)");

            if (m_ui.RealtimeMode)
            {
                RESET_ON_CHANGE(ImGui::Checkbox("FireflyFilter (realtime)", &m_ui.RealtimeFireflyFilterEnabled));
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Enable smart firefly filter that clamps max radiance based on probability heuristic.");
                if (m_ui.RealtimeFireflyFilterEnabled)
                {
                    RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent); );
                    RESET_ON_CHANGE(ImGui::InputFloat("FF Threshold", &m_ui.RealtimeFireflyFilterThreshold, 0.01f, 0.1f, "%.5f"));
                    m_ui.RealtimeFireflyFilterThreshold = dm::clamp(m_ui.RealtimeFireflyFilterThreshold, 0.00001f, 1000.0f);
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Better light importance sampling allows for setting higher firefly filter threshold and conversely.");
                    //ImGui::SameLine();
                    //RESET_ON_CHANGE( ImGui::Checkbox("RX", &m_ui.RealtimeFireflyFilterRelaxOnNonNoisy) ); 
                    //if (ImGui::IsItemHovered()) ImGui::SetTooltip("Relax on non-noisy (direct, stable) radiance: clamp value will be FIREFLY_FILTER_RELAX_ON_NON_NOISY_K times bigger - helps with blooms");
                }
            }
            else
            {
                RESET_ON_CHANGE(ImGui::Checkbox("FireflyFilter (reference *)", &m_ui.ReferenceFireflyFilterEnabled));
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Enable smart firefly filter that clamps max radiance based on probability heuristic.\n* when both tonemapping autoexposure and firefly filter are enabled\nin reference mode, results are no longer deterministic!");
                if (m_ui.ReferenceFireflyFilterEnabled)
                {
                    RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent); );
                    RESET_ON_CHANGE(ImGui::InputFloat("FF Threshold", &m_ui.ReferenceFireflyFilterThreshold, 0.1f, 0.2f, "%.5f"));
                    m_ui.ReferenceFireflyFilterThreshold = dm::clamp(m_ui.ReferenceFireflyFilterThreshold, 0.01f, 1000.0f);
                }
            }
        }

        ImGui::TextColored(categoryColor, "AA/SR/Denoising:");
        {
            RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent); );

            if (m_ui.RealtimeMode)
            {
#if DONUT_WITH_STREAMLINE
                const bool dlssAvailable = m_ui.IsDLSSSuported;
                const bool dlssRRAvailable = m_ui.IsDLSSRRSupported;
#else
                const bool dlssAvailable = false;
                const bool dlssRRAvailable = false;
#endif
                const char* items[] = { "Disabled", "TAA", "DLSS", "DLSS-RR" };

                const int itemCount = IM_ARRAYSIZE(items);

                m_ui.RealtimeAA = dm::clamp(m_ui.RealtimeAA, 0, dlssAvailable ? itemCount : 1);

                if (ImGui::BeginCombo("AA/SR/Denoising", items[m_ui.RealtimeAA]))
                {
                    for (int i = 0; i < itemCount; i++)
                    {
                        bool enabled = false;
                        enabled |= i < 2;
                        enabled |= (i == 2) && dlssAvailable;
                        enabled |= (i == 3) && dlssRRAvailable;
                        UI_SCOPED_DISABLE(!enabled);

                        bool isSelected = (m_ui.RealtimeAA == i);
                        if (ImGui::Selectable(items[i], isSelected))
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

                {
                    UI_SCOPED_DISABLE(!m_ui.RealtimeMode || m_ui.RealtimeAA == 3);
                    bool notTrue = false;
                    ImGui::Checkbox("Use standalone denoiser (NRD)", (m_ui.RealtimeAA == 3) ? (&notTrue) : (&m_ui.StandaloneDenoiser));
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Enables NVIDIA Real-Time Denoisers (NRD) that execute before TAA/DLSS/DLAA pass\nNote: no built-in denoiser available in 'Reference' mode, however \n'Photo mode screenshot' button launches external denoiser!");
                }
            }
            else // !m_ui.RealtimeMode
            {
                if (ImGui::Button("Photo mode screenshot"))
                    m_ui.ExperimentalPhotoModeScreenshot = true;
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Experimental: Saves a photo.bmp next to where .exe is and applies\n"
                    "denoising using command line tool that wraps OptiX and OIDN denoisers.\n"
                    "No guidance buffers are used and color is in LDR (so not as high quality\n"
                    "as it could be - will get improved in the future). \n"
                    "Command line denoiser wrapper tools by Declan Russel, available at:\n"
                    "https://github.com/DeclanRussell/NvidiaAIDenoiser\n"
                    "https://github.com/DeclanRussell/IntelOIDenoiser");
            }
            {
                ImGui::Checkbox("Enable tone mapping", &m_ui.EnableToneMapping);
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Full tone mapping settings available under global `Tone Mapping` options");
            }
        }

        ImGui::TextColored(categoryColor, "Signal caching/boosting:");
        {
            RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent); );
            if (m_ui.RealtimeMode)
            {
                RESET_ON_CHANGE(ImGui::Combo("Radiance cache", (int*)&m_ui.RadianceCacheMode, "Disabled\0SHaRC\0\0"));
            }
            else
            {
                ImGui::Text("Not yet supported in reference mode.");
            }
        }

        ImGui::TextColored(categoryColor, "Light sampling:");
        {
            RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent); );

            if (m_ui.RealtimeMode)
            {
                {
                    bool nullCheckbox = false;
                    bool disabled = !m_ui.UseNEE || (m_ui.RealtimeAA == 3 && m_ui.DisableReSTIRsWithDLSSRR);
                    UI_SCOPED_DISABLE(disabled);
                    RESET_ON_CHANGE(ImGui::Checkbox("Use ReSTIR DI (RTXDI)", (disabled) ? &nullCheckbox : &m_ui.UseReSTIRDI));
                    if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle) && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                        m_ui.DisableReSTIRsWithDLSSRR = !m_ui.DisableReSTIRsWithDLSSRR;
                }
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("ReSTIR DI (RTXDI) requires Next Event Estimation to be enabled\nand this implementation is currently not tuned to work with DLSS-RR");

                {
                    bool nullCheckbox = false;
                    bool disabled = m_ui.RealtimeAA == 3 && m_ui.DisableReSTIRsWithDLSSRR;
                    UI_SCOPED_DISABLE(disabled);
                    RESET_ON_CHANGE(ImGui::Checkbox("Use ReSTIR GI (RTXDI)", (disabled) ? &nullCheckbox : &m_ui.UseReSTIRGI));
                    if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle) && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                        m_ui.DisableReSTIRsWithDLSSRR = !m_ui.DisableReSTIRsWithDLSSRR;
                }
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("ReSTIR GI (RTXDI) is currently not tuned to work well with DLSS-RR\nUse middle mouse button to enable anyway");
            }

            RESET_ON_CHANGE(ImGui::Checkbox("Use Next Event Estimation", &m_ui.UseNEE));
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("This enables NEE a.k.a. direct light importance sampling (this includes ReSTIR DI but not ReSTIR GI)\nNote: analytic lights currently only come out of NEE so they will be missing when NEE is disabled");

            if (m_ui.UseNEE)
            {
                ImGui::TextColored(categoryColor, "NEE settings: ");
                {
                    RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent); );
                    RESET_ON_CHANGE(ImGui::Combo("Sampling technique", (int*)&m_ui.NEEType, "Uniform\0Power+\0NEE-AT\0\0"));
                    m_ui.NEEType = dm::clamp(m_ui.NEEType, 0, 2);
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Light importance sampling technique to use for NEE.\nNote: Additional NEE-AT settings are exposed in 'Lighting -> NEE-AT' UI section.");

                    RESET_ON_CHANGE(ImGui::InputInt("Candidate samples", &m_ui.NEECandidateSamples, 1));
                    if (ImGui::IsItemHovered())
                    {
                        if (m_ui.NEEType != 2)
                            ImGui::SetTooltip("This is the number of light samples weighted with BSDF used to pick each full sample.");
                        else
                        {
                            uint localCandidateSamples = NEEAT::ComputeCandidateSampleLocalCount(m_ui.ActualNEEAT_LocalToGlobalSampleRatio(), m_ui.NEECandidateSamples);
                            uint globalCandidateSamples = NEEAT::ComputeCandidateSampleGlobalCount(m_ui.ActualNEEAT_LocalToGlobalSampleRatio(), m_ui.NEECandidateSamples);

                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("This is the number of light samples weighted with BSDF used to pick each full sample.\n"
                                "Out of those %d will be Local and %d will be Global NEE-AT samples", localCandidateSamples, globalCandidateSamples);
                        }
                    }
                    m_ui.NEECandidateSamples = dm::clamp(m_ui.NEECandidateSamples, 1, RTXPT_LIGHTING_MAX_SAMPLE_COUNT);

                    RESET_ON_CHANGE(ImGui::InputInt("Full samples", &m_ui.NEEFullSamples, 1));
                    m_ui.NEEFullSamples = dm::clamp(m_ui.NEEFullSamples, 0, RTXPT_LIGHTING_MAX_SAMPLE_COUNT);
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("This is the number of light samples to shadow test and integrate\nNote: Maximum total number of samples is 63");

                    RESET_ON_CHANGE(ImGui::Combo("MIS Type", (int*)&m_ui.NEEMISType, "Full\0ApproxInRealtime\0Approximate\0\0"));
                    m_ui.NEEMISType = dm::clamp(m_ui.NEEMISType, 0, 2);
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Path (BSDF) vs light sampler Multiple Importance Sampling approach.\n'Approximate' is faster and easier to implement but more noisy, with \nthe impact of noise especially detrimental in reference accumulation.");
                }
            }
        }

        if (ImGui::CollapsingHeader("PT: Advanced Settings", 0))
        {
            ImGui::TextColored(categoryColor, "Features:");
            ImGui::Combo("Nested Dielectrics", (int*)&m_ui.NestedDielectricsQuality, "Off\0Fast\0Quality\0"); m_ui.NestedDielectricsQuality = clamp(m_ui.NestedDielectricsQuality, 0, 2);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Priority-based nested dielectrics; 'Quality' allows for more \ncorrect rejections, 'Fast' is.. well, faster.");
            if (m_ui.RealtimeAA == 3)
            {
                RESET_ON_CHANGE(ImGui::InputFloat("RR brightness clamp", &m_ui.DLSSRRBrightnessClampK));
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("RR doesn't handle too bright (relatively) areas, causing unwanted noise;\nEnabling this will clamp brightness at the expense of bloom.\nTODO: replace this with local tonemap or similar");
            }

            ImGui::TextColored(categoryColor, "Performance:");
            {
                RESET_ON_CHANGE(ImGui::InputFloat("Texture MIP bias", &m_ui.TexLODBias));

                RESET_ON_CHANGE(ImGui::InputInt("Diffuse sample envmap MIP level", &m_ui.EnvironmentMapDiffuseSampleMIPLevel));    m_ui.EnvironmentMapDiffuseSampleMIPLevel = dm::clamp(m_ui.EnvironmentMapDiffuseSampleMIPLevel, 0, 16);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Use the specific MIP level to sample environment map texture during light sampling and for main path terminating\ninto sky after a diffuse scatter. Only 0 produces unbiased results.");

                RESET_ON_CHANGE(ImGui::Checkbox("Use Russian Roulette early out", &m_ui.EnableRussianRoulette));
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("This enables stochastic path termination for low throughput diffuse paths");

                if (m_NVAPI_SERSupported)
                {
                    RESET_ON_CHANGE(ImGui::Checkbox("NVAPI HitObject codepath", &m_ui.NVAPIHitObjectExtension)); // <- while there's no need to reset accumulation since this is a performance only feature, leaving the reset in for testing correctness
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("If disabled, traditional TraceRay path is used.\nIf enabled, TraceRayInline->MakeHit->ReorderThread->InvokeHit approach is used!");
                    if (m_ui.NVAPIHitObjectExtension)
                    {
                        RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent); );
                        ImGui::Checkbox("NVAPI ReorderThreads", &m_ui.NVAPIReorderThreads);
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("This enables/disables the actual ReorderThread call in the shader.");
                    }
                    if (m_ui.NVAPIHitObjectExtension)
                        m_ui.DXHitObjectExtension = false;
                }
                // No else: the NVAPI HitObject path has no UI now (superseded by dx::HitObject), but it is still used without Agility SDK 619;
                // the SampleUI constructor already turned it off where SER is unsupported, so don't override it here.

#if RTXPT_D3D_AGILITY_SDK_VERSION >= 619
                {
                    RESET_ON_CHANGE(ImGui::Checkbox("dx::HitObject codepath", &m_ui.DXHitObjectExtension));
                    if (m_ui.DXHitObjectExtension)
                    {
                        RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent); );
                        RESET_ON_CHANGE(ImGui::Checkbox("dx::MaybeReorderThreads ", &m_ui.DXMaybeReorderThreads));
                    }
                    if (m_ui.DXHitObjectExtension)
                        m_ui.NVAPIHitObjectExtension = false;
                }
#else
                ImGui::TextColored(warnColor, "AgilitySDK619 NOT available, no dx::MaybeReorderThreads!");
#endif

                RESET_ON_CHANGE(ImGui::Checkbox("Use explicit fp16 types", &m_ui.UseFp16Types));

                RESET_ON_CHANGE(ImGui::Checkbox("Enable LD sampler for BSDF", &m_ui.EnableLDSamplerForBSDF));
            }
        }
    }

    if (m_app.GetSharc() != nullptr && m_ui.ActualRadianceCacheMode() == 1 && ImGui::CollapsingHeader("SHaRC (radiance cache)"))
    {
        assert( m_ui.RadianceCacheMode == 1 );
        RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent););
        m_app.GetSharc()->DebugGUI(indent, categoryColor, m_ui.DbgDiscardNonNEELighting, m_ui.DbgDiscardNEELighting);
    }

#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
    if (ImGui::CollapsingHeader("Stochastic Texture Filtering"))
    {
        bool changed = false;
        changed = ImGui::Combo("Magnification Method", (int*)&m_ui.STFMagnificationMethod,
            "Default\0"
            "Quad2x2\0"
            "Fine2x2\0"
            "FineTemporal2x2\0"
            "FineAlu3x3\0"
            "FineLut3x3\0"
            "Fine4x4\0"
        );
        if (changed)
        {
            donut::log::debug("Magnification Method ", static_cast<int>(m_ui.STFMagnificationMethod));
        }

        changed = ImGui::Combo("Filter Type", (int*)&m_ui.STFFilterMode,
            "Point\0"
            "Linear\0"
            "Cubic\0"
            "Gaussian\0"
        );
        if (changed)
        {
            donut::log::debug("Filter Type ", static_cast<int>(m_ui.STFFilterMode));
        }

        ImGui::BeginDisabled(m_ui.STFFilterMode != StfFilterMode::Gaussian);
        ImGui::SliderFloat("Sigma", &m_ui.STFGaussianSigma, 0.f, 100.f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::EndDisabled();   // m_ui.STFFilterMode
    }
#endif // RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
    if (ImGui::CollapsingHeader("Denoising Layers"))
    {
        if (m_ui.RealtimeMode)
        {
            ImGui::InputInt("Active denoising layers", &m_ui.DenoisingLayersActiveCount);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("How many denoising layers to allow - 1 is just standard denoising");
            m_ui.DenoisingLayersActiveCount = dm::clamp(m_ui.DenoisingLayersActiveCount, 1, (int)cDenoisingLayerCount);
            ImGui::InputInt("Max denoising layer vertex depth", &m_ui.DenoisingLayersMaxVertexDepth);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("How deep the stable part of path tracing can go");
            m_ui.DenoisingLayersMaxVertexDepth = dm::clamp(m_ui.DenoisingLayersMaxVertexDepth, 2, (int)cDenoisingLayerMaxVertexIndex);
            ImGui::Checkbox("Primary Surface Replacement", &m_ui.AllowPrimarySurfaceReplacement);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("When denoising layers enabled, whether we can use PSR for the first (base) plane");
            ImGui::Checkbox("Suppress primary plane noisy specular", &m_ui.DenoisingLayersSuppressPrimaryIndirectSpecular);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("This will suppress noisy specular to primary denoising layer by specified amount\nbut only if at least 1 denoising layer is also used on the same pixel.\nThis for ex. reduces secondary internal smudgy reflections from internal many bounces in a window.");
            ImGui::SliderFloat("Suppress primary plane noisy specular amount", &m_ui.DenoisingLayersSuppressPrimaryIndirectSpecularK, 0.0f, 1.0f);
            ImGui::SliderFloat("Non-primary plane anti-aliasing fallthrough", &m_ui.DenoisingLayersAntiAliasingFallthrough, 0.0f, 1.0f);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Divert some radiance on highly curved and edge areas from non-0 plane back\nto plane 0. This reduces aliasing on complex boundary bounces.");
        }
        else
        {
            ImGui::Text("Not available in reference mode");
        }
    }

    if ((m_ui.ActualUseReSTIRDI() || m_ui.ActualUseReSTIRGI()) && ImGui::CollapsingHeader("RTXDI Settings"))
    {
        ImGui::TextColored(categoryColor, "ReGIR");
        {
            RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent); );

            if (m_ui.ActualUseReSTIRDI())
            {
                ImGui::PushItemWidth(defItemWidth);

                RESET_ON_CHANGE(ImGui::InputInt("Number of Build Samples", (int*)&m_ui.RTXDI.regir.regirDynamicParameters.regirNumBuildSamples));
                m_ui.RTXDI.regir.regirDynamicParameters.regirNumBuildSamples = dm::clamp(m_ui.RTXDI.regir.regirDynamicParameters.regirNumBuildSamples, 0u, 128u);
                RESET_ON_CHANGE(ImGui::SliderFloat("Cell Size", &m_ui.RTXDI.regir.regirDynamicParameters.regirCellSize, 0.1f, 2.f));
                RESET_ON_CHANGE(ImGui::SliderFloat("Sampling Jitter", &m_ui.RTXDI.regir.regirDynamicParameters.regirSamplingJitter, 0.f, 1.f));

                ImGui::PopItemWidth();
            }
            else
                ImGui::Text("Not used/enabled");
        }

        auto imGuiCheckboxUInt32 = [&](const char* label, uint32_t* v)
            {
                bool pv = (*v) != 0;
                bool ret = ImGui::Checkbox(label, &pv);
                *v = pv ? (1) : (0);
                return ret;
            };

        ImGui::TextColored(categoryColor, "ReSTIR DI");
        {
            RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent); );
            if (m_ui.ActualUseReSTIRDI())
            {
                ImGui::PushItemWidth(defItemWidth);

                RESET_ON_CHANGE(ImGui::Combo("Resampling Mode", (int*)&m_ui.RTXDI.restirDI.resamplingMode,
                    "Disabled\0Temporal\0Spatial\0Temporal & Spatial\0Fused\0\0"));

                RESET_ON_CHANGE(ImGui::Combo("Spatial Bias Correction", (int*)&m_ui.RTXDI.restirDI.spatialResamplingParams.biasCorrectionMode,
                    "Off\0Basic\0Pairwise\0Ray Traced\0\0"));

                RESET_ON_CHANGE(ImGui::Combo("Temporal Bias Correction", (int*)&m_ui.RTXDI.restirDI.temporalResamplingParams.biasCorrectionMode,
                    "Off\0Basic\0Pairwise\0Ray Traced\0\0"));

                RESET_ON_CHANGE(ImGui::Combo("Local Light Sampling Mode", (int*)&m_ui.RTXDI.restirDI.initialSamplingParams.localLightSamplingMode,
                    "Uniform\0Power RIS\0ReGIR RIS\0\0"));

                if (m_ui.RTXDI.restirDI.initialSamplingParams.localLightSamplingMode == ReSTIRDI_LocalLightSamplingMode::ReGIR_RIS)
                {
                    RESET_ON_CHANGE(ImGui::Combo("ReGIR Mode", (int*)&m_ui.RTXDI.regir.regirStaticParams.Mode,
                        "Disabled\0Grid\0Onion\0\0"));
                }

                ImGui::PopItemWidth();

                ImGui::PushItemWidth(defItemWidth * 0.8f);

                ImGui::Text("Number of Primary Samples: ");

                {
                    RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent); );

                    RESET_ON_CHANGE(ImGui::InputInt("ReGir", (int*)&m_ui.RTXDI.regir.regirDynamicParameters.regirNumBuildSamples));
                    m_ui.RTXDI.regir.regirDynamicParameters.regirNumBuildSamples = dm::clamp(m_ui.RTXDI.regir.regirDynamicParameters.regirNumBuildSamples, 0u, 32u);
                    RESET_ON_CHANGE(ImGui::InputInt("Local Light", (int*)&m_ui.RTXDI.restirDI.initialSamplingParams.numLocalLightSamples));
                    m_ui.RTXDI.restirDI.initialSamplingParams.numLocalLightSamples = dm::clamp(m_ui.RTXDI.restirDI.initialSamplingParams.numLocalLightSamples, 0u, 32u);
                    RESET_ON_CHANGE(ImGui::InputInt("BRDF", (int*)&m_ui.RTXDI.restirDI.initialSamplingParams.numBrdfSamples));
                    m_ui.RTXDI.restirDI.initialSamplingParams.numBrdfSamples = dm::clamp(m_ui.RTXDI.restirDI.initialSamplingParams.numBrdfSamples, 0u, 32u);
                    RESET_ON_CHANGE(ImGui::InputInt("Infinite Light", (int*)&m_ui.RTXDI.restirDI.initialSamplingParams.numInfiniteLightSamples));
                    m_ui.RTXDI.restirDI.initialSamplingParams.numInfiniteLightSamples = dm::clamp(m_ui.RTXDI.restirDI.initialSamplingParams.numInfiniteLightSamples, 0u, 32u);
                    RESET_ON_CHANGE(ImGui::InputInt("Environment Light", (int*)&m_ui.RTXDI.restirDI.initialSamplingParams.numEnvironmentSamples));
                    m_ui.RTXDI.restirDI.initialSamplingParams.numEnvironmentSamples = dm::clamp(m_ui.RTXDI.restirDI.initialSamplingParams.numEnvironmentSamples, 0u, 32u);
                }
                RESET_ON_CHANGE(imGuiCheckboxUInt32("Initial visibility test", &m_ui.RTXDI.restirDI.initialSamplingParams.enableInitialVisibility));

                if (ImGui::CollapsingHeader("Fine Tuning"))
                {
                    RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent); );
                    ImGui::PushItemWidth(defItemWidth);
                    RESET_ON_CHANGE(ImGui::SliderFloat("BRDF Cut-off", &m_ui.RTXDI.restirDI.initialSamplingParams.brdfCutoff, 0.0f, 1.0f));
                    ImGui::Separator();
                    RESET_ON_CHANGE(imGuiCheckboxUInt32("Use Permutation Sampling", &m_ui.RTXDI.restirDI.temporalResamplingParams.enablePermutationSampling));
                    RESET_ON_CHANGE(ImGui::SliderFloat("Temporal Depth Threshold", &m_ui.RTXDI.restirDI.temporalResamplingParams.depthThreshold, 0.f, 1.f));
                    RESET_ON_CHANGE(ImGui::SliderFloat("Temporal Normal Threshold", &m_ui.RTXDI.restirDI.temporalResamplingParams.normalThreshold, 0.f, 1.f));
                    RESET_ON_CHANGE(ImGui::SliderFloat("Boiling Filter Strength", &m_ui.RTXDI.restirDI.boilingFilterParams.boilingFilterStrength, 0.f, 1.f));
                    RESET_ON_CHANGE(imGuiCheckboxUInt32("Visibility Shortcut", &m_ui.RTXDI.restirDI.temporalResamplingParams.enableVisibilityShortcut));
                    ImGui::Separator();
                    RESET_ON_CHANGE(ImGui::SliderInt("Spatial Samples", (int*)&m_ui.RTXDI.restirDI.spatialResamplingParams.numSamples, 0, 8));
                    RESET_ON_CHANGE(ImGui::SliderInt("Disocclusion Samples", (int*)&m_ui.RTXDI.restirDI.spatialResamplingParams.numDisocclusionBoostSamples, 0, 8));
                    RESET_ON_CHANGE(ImGui::SliderFloat("Spatial Sampling Radius", &m_ui.RTXDI.restirDI.spatialResamplingParams.samplingRadius, 0.f, 64.f));
                    RESET_ON_CHANGE(ImGui::SliderFloat("Spatial Depth Threshold", &m_ui.RTXDI.restirDI.spatialResamplingParams.depthThreshold, 0.f, 1.f));
                    RESET_ON_CHANGE(ImGui::SliderFloat("Spatial Normal Threshold", &m_ui.RTXDI.restirDI.spatialResamplingParams.normalThreshold, 0.f, 1.f));
                    RESET_ON_CHANGE(imGuiCheckboxUInt32("Discount Naive Samples", &m_ui.RTXDI.restirDI.spatialResamplingParams.discountNaiveSamples));
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Prevents samples which are from the current frame or have no reasonable temporal history merged being spread to neighbors");
                    ImGui::Separator();
                    RESET_ON_CHANGE(ImGui::DragFloat("Ray Epsilon", &m_ui.RTXDI.rayEpsilon, 0.0001f, 0.0001f, 0.01f, "%.4f"));
                    ImGui::PopItemWidth();
                }

                ImGui::PopItemWidth();
            }
            else
                ImGui::Text("Not used/enabled");
        }

        ImGui::TextColored(categoryColor, "ReSTIR GI");
        {
            RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent); );
            if (m_ui.ActualUseReSTIRGI())
            {
                RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent); );
                ImGui::PushItemWidth(defItemWidth);
                RESET_ON_CHANGE(ImGui::Combo("Resampling Mode", (int*)&m_ui.RTXDI.restirGI.resamplingMode,
                    "Disabled\0Temporal\0Spatial\0Temporal & Spatial\0Fused\0\0"));
                ImGui::TextWrapped("Please note: there's a bug in ReSTIRGIContext::UpdateBufferIndices or similar which breaks 'Disabled' or one or the other sampling modes.");
                ImGui::Separator();
                RESET_ON_CHANGE(ImGui::SliderInt("History Length ##GI", (int*)&m_ui.RTXDI.restirGI.temporalResamplingParams.maxHistoryLength, 0, 64));
                RESET_ON_CHANGE(ImGui::SliderInt("Max Reservoir Age ##GI", (int*)&m_ui.RTXDI.restirGI.temporalResamplingParams.maxReservoirAge, 0, 100));
                RESET_ON_CHANGE(imGuiCheckboxUInt32("Permutation Sampling ##GI", &m_ui.RTXDI.restirGI.temporalResamplingParams.enablePermutationSampling));
                RESET_ON_CHANGE(imGuiCheckboxUInt32("Fallback Sampling ##GI", &m_ui.RTXDI.restirGI.temporalResamplingParams.enableFallbackSampling));
                RESET_ON_CHANGE(ImGui::SliderFloat("Boiling Filter Strength##GI", &m_ui.RTXDI.restirGI.boilingFilterParams.boilingFilterStrength, 0.f, 1.f));
                RESET_ON_CHANGE(ImGui::Combo("Temporal Bias Correction ##GI", (int*)&m_ui.RTXDI.restirGI.temporalResamplingParams.biasCorrectionMode,
                    "Off\0Basic\0Ray Traced\0"));
                ImGui::Separator();
                RESET_ON_CHANGE(ImGui::SliderInt("Spatial Samples ##GI", (int*)&m_ui.RTXDI.restirGI.spatialResamplingParams.numSamples, 0, 8));
                RESET_ON_CHANGE(ImGui::SliderFloat("Spatial Sampling Radius ##GI", &m_ui.RTXDI.restirGI.spatialResamplingParams.samplingRadius, 1.f, 64.f));
                RESET_ON_CHANGE(ImGui::Combo("Spatial Bias Correction ##GI", (int*)&m_ui.RTXDI.restirGI.spatialResamplingParams.biasCorrectionMode, "Off\0Basic\0Ray Traced\0"));
                ImGui::Separator();
                RESET_ON_CHANGE(imGuiCheckboxUInt32("Final Visibility ##GI", &m_ui.RTXDI.restirGI.finalShadingParams.enableFinalVisibility));
                RESET_ON_CHANGE(imGuiCheckboxUInt32("Final MIS ##GI", &m_ui.RTXDI.restirGI.finalShadingParams.enableFinalMIS));

                ImGui::PopItemWidth();
            }
        }
    }

    if (m_ui.ActualUseStandaloneDenoiser() && ImGui::CollapsingHeader("Standalone Denoiser (NRD)"))
    {
        RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent); );

        ImGui::InputFloat("Disocclusion Threshold", &m_ui.NRDDisocclusionThreshold);
        ImGui::Checkbox("Use Alternate Disocclusion Threshold Mix", &m_ui.NRDUseAlternateDisocclusionThresholdMix);
        ImGui::InputFloat("Disocclusion Threshold Alt", &m_ui.NRDDisocclusionThresholdAlternate);
        ImGui::InputFloat("Radiance clamping", &m_ui.DenoiserRadianceClampK);

        ImGui::Separator();

        m_ui.NRDModeChanged = ImGui::Combo("Denoiser Mode", (int*)&m_ui.NRDMethod, "REBLUR\0RELAX\0\0");
        m_ui.NRDMethod = dm::clamp(m_ui.NRDMethod, (NrdConfig::DenoiserMethod)0, (NrdConfig::DenoiserMethod)1);

        if (ImGui::CollapsingHeader("Advanced Settings"))
        {
            if (m_ui.NRDMethod == NrdConfig::DenoiserMethod::REBLUR)
            {
                // TODO: make sure these are updated to constants
                ImGui::SliderFloat("Hit Distance A", &m_ui.ReblurSettings.hitDistanceParameters.A, 0.0f, 10.0f);
                ImGui::SliderFloat("Hit Distance B", &m_ui.ReblurSettings.hitDistanceParameters.B, 0.0f, 10.0f);
                ImGui::SliderFloat("Hit Distance C", &m_ui.ReblurSettings.hitDistanceParameters.C, 0.0f, 50.0f);

                ImGui::SliderFloat("Antilag Luminance Sigma Scale", &m_ui.ReblurSettings.antilagSettings.luminanceSigmaScale, 1.0f, 3.0f);
                // ImGui::SliderFloat("Antilag Hit Distance Sigma Scale", &m_ui.ReblurSettings.antilagSettings.hitDistanceSigmaScale, 1.0f, 3.0f);
                ImGui::SliderFloat("Antilag Luminance Sensitivity", &m_ui.ReblurSettings.antilagSettings.luminanceSensitivity, 0.001f, 1.0f);
                // ImGui::SliderFloat("Antilag Hit Distance Sensitivity", &m_ui.ReblurSettings.antilagSettings.hitDistanceSensitivity, 0.001f, 1.0f);

                ImGui::SliderInt("Max Accumulated Frames", (int*)&m_ui.ReblurSettings.maxAccumulatedFrameNum, 0, nrd::REBLUR_MAX_HISTORY_FRAME_NUM);
                ImGui::SliderInt("Fast Max Accumulated Frames", (int*)&m_ui.ReblurSettings.maxFastAccumulatedFrameNum, 0, nrd::REBLUR_MAX_HISTORY_FRAME_NUM);
                ImGui::SliderInt("History Fix Frames", (int*)&m_ui.ReblurSettings.historyFixFrameNum, 0, nrd::REBLUR_MAX_HISTORY_FRAME_NUM);

                ImGui::SliderFloat("Diffuse Prepass Blur Radius (pixels)", &m_ui.ReblurSettings.diffusePrepassBlurRadius, 0.0f, 100.0f);
                ImGui::SliderFloat("Specular Prepass Blur Radius (pixels)", &m_ui.ReblurSettings.specularPrepassBlurRadius, 0.0f, 100.0f);
                ImGui::SliderFloat("Min Blur Radius (pixels)", &m_ui.ReblurSettings.minBlurRadius, 0.0f, 100.0f);
                ImGui::SliderFloat("Max Blur Radius (pixels)", &m_ui.ReblurSettings.maxBlurRadius, 0.0f, 100.0f);

                ImGui::SliderFloat("Lobe Angle Fraction", &m_ui.ReblurSettings.lobeAngleFraction, 0.0f, 1.0f);
                ImGui::SliderFloat("Roughness Fraction", &m_ui.ReblurSettings.roughnessFraction, 0.0f, 1.0f);

                ImGui::SliderFloat("Accumulation Roughness Threshold", &m_ui.ReblurSettings.responsiveAccumulationSettings.roughnessThreshold, 0.0f, 1.0f);

                //ImGui::SliderFloat("Stabilization Strength", &m_ui.ReblurSettings.stabilizationStrength, 0.0f, 1.0f);

                ImGui::SliderFloat("Plane Distance Sensitivity", &m_ui.ReblurSettings.planeDistanceSensitivity, 0.0f, 1.0f);

                // ImGui::Combo("Checkerboard Mode", (int*)&m_ui.ReblurSettings.checkerboardMode, "Off\0Black\0White\0\0");

                // these are uint8_t and ImGUI takes a ptr to int32_t :(
                int hitDistanceReconstructionMode = (int)m_ui.ReblurSettings.hitDistanceReconstructionMode;
                ImGui::Combo("Hit Distance Reconstruction Mode", &hitDistanceReconstructionMode, "Off\0AREA_3X3\0AREA_5X5\0\0");
                m_ui.ReblurSettings.hitDistanceReconstructionMode = (nrd::HitDistanceReconstructionMode)hitDistanceReconstructionMode;

                ImGui::Checkbox("Enable Firefly Filter", &m_ui.ReblurSettings.enableAntiFirefly);

                // ImGui::Checkbox("Enable Diffuse Material Test", &m_ui.ReblurSettings.enableMaterialTestForDiffuse);
                // ImGui::Checkbox("Enable Specular Material Test", &m_ui.ReblurSettings.enableMaterialTestForSpecular);
            }
            else // m_ui.NRDMethod == NrdConfig::DenoiserMethod::RELAX
            {
                ImGui::SliderFloat("Diffuse Prepass Blur Radius", &m_ui.RelaxSettings.diffusePrepassBlurRadius, 0.0f, 100.0f);
                ImGui::SliderFloat("Specular Prepass Blur Radius", &m_ui.RelaxSettings.specularPrepassBlurRadius, 0.0f, 100.0f);

                ImGui::SliderInt("Diffuse Max Accumulated Frames", (int*)&m_ui.RelaxSettings.diffuseMaxAccumulatedFrameNum, 0, nrd::RELAX_MAX_HISTORY_FRAME_NUM);
                ImGui::SliderInt("Specular Max Accumulated Frames", (int*)&m_ui.RelaxSettings.specularMaxAccumulatedFrameNum, 0, nrd::RELAX_MAX_HISTORY_FRAME_NUM);

                ImGui::SliderInt("Diffuse Fast Max Accumulated Frames", (int*)&m_ui.RelaxSettings.diffuseMaxFastAccumulatedFrameNum, 0, 10);   // nrd::RELAX_MAX_HISTORY_FRAME_NUM
                ImGui::SliderInt("Specular Fast Max Accumulated Frames", (int*)&m_ui.RelaxSettings.specularMaxFastAccumulatedFrameNum, 0, 10); // nrd::RELAX_MAX_HISTORY_FRAME_NUM

                ImGui::SliderInt("History Fix Frame Num", (int*)&m_ui.RelaxSettings.historyFixFrameNum, 0, nrd::RELAX_MAX_HISTORY_FRAME_NUM);

                ImGui::SliderFloat("Diffuse Edge Stopping Sensitivity", &m_ui.RelaxSettings.diffusePhiLuminance, 0.0f, 10.0f);
                ImGui::SliderFloat("Specular Edge Stopping Sensitivity", &m_ui.RelaxSettings.specularPhiLuminance, 0.0f, 10.0f);

                ImGui::SliderFloat("Lobe Angle Fraction", &m_ui.RelaxSettings.lobeAngleFraction, 0.0f, 1.0f);
                ImGui::SliderFloat("Roughness Fraction", &m_ui.RelaxSettings.roughnessFraction, 0.0f, 1.0f);

                ImGui::SliderFloat("Specular Variance Boost", &m_ui.RelaxSettings.specularVarianceBoost, 0.0f, 1.0f);

                ImGui::SliderFloat("Specular Lobe Angle Slack", &m_ui.RelaxSettings.specularLobeAngleSlack, 0.0f, 1.0f);

                ImGui::SliderFloat("Normal Edge Stopping Power", &m_ui.RelaxSettings.historyFixEdgeStoppingNormalPower, 0.0f, 30.0f);

                ImGui::SliderFloat("Clamping Color Box Sigma Scale", &m_ui.RelaxSettings.fastHistoryClampingSigmaScale, 0.0f, 3.0f);

                ImGui::SliderInt("Spatial Variance Estimation History Threshold", (int*)&m_ui.RelaxSettings.spatialVarianceEstimationHistoryThreshold, 0, nrd::RELAX_MAX_HISTORY_FRAME_NUM);

                ImGui::SliderInt("Number of Atrous iterations", (int*)&m_ui.RelaxSettings.atrousIterationNum, 2, 8);

                ImGui::SliderFloat("Diffuse Min Luminance Weight", &m_ui.RelaxSettings.diffuseMinLuminanceWeight, 0.0f, 1.0f);
                ImGui::SliderFloat("Specular Min Luminance Weight", &m_ui.RelaxSettings.specularMinLuminanceWeight, 0.0f, 1.0f);

                ImGui::SliderFloat("Edge Stopping Threshold", &m_ui.RelaxSettings.depthThreshold, 0.0f, 0.1f);

                ImGui::SliderFloat("Confidence: Relaxation Multiplier", &m_ui.RelaxSettings.confidenceDrivenRelaxationMultiplier, 0.0f, 1.0f);
                ImGui::SliderFloat("Confidence: Luminance Edge Stopping Relaxation", &m_ui.RelaxSettings.confidenceDrivenLuminanceEdgeStoppingRelaxation, 0.0f, 1.0f);
                ImGui::SliderFloat("Confidence: Normal Edge Stopping Relaxation", &m_ui.RelaxSettings.confidenceDrivenNormalEdgeStoppingRelaxation, 0.0f, 1.0f);

                ImGui::SliderFloat("Luminance Edge Stopping Relaxation", &m_ui.RelaxSettings.luminanceEdgeStoppingRelaxation, 0.0f, 1.0f);
                ImGui::SliderFloat("Normal Edge Stopping Relaxation", &m_ui.RelaxSettings.normalEdgeStoppingRelaxation, 0.0f, 1.0f);

                ImGui::SliderFloat("Roughness Edge Stopping Relaxation", &m_ui.RelaxSettings.roughnessEdgeStoppingRelaxation, 0.0f, 5.0f);

                ImGui::SliderFloat("Antilag Acceleration Amount", &m_ui.RelaxSettings.antilagSettings.accelerationAmount, 0.0f, 1.0f);
                ImGui::SliderFloat("Antilag Spatial Sigma Scale", &m_ui.RelaxSettings.antilagSettings.spatialSigmaScale, 0.0f, 5.0f);
                ImGui::SliderFloat("Antilag Temporal Sigma Scale", &m_ui.RelaxSettings.antilagSettings.temporalSigmaScale, 0.0f, 5.0f);
                ImGui::SliderFloat("Antilag Reset Amount", &m_ui.RelaxSettings.antilagSettings.resetAmount, 0.0f, 1.0f);

                // ImGui::Combo("Checkerboard Mode", (int*)&m_ui.RelaxSettings.checkerboardMode, "Off\0Black\0White\0\0");

                int hitDistanceReconstructionMode = (int)m_ui.RelaxSettings.hitDistanceReconstructionMode;  // these are uint8_t and ImGUI takes a ptr to int32_t :(
                ImGui::Combo("Hit Distance Reconstruction Mode", &hitDistanceReconstructionMode, "Off\0AREA_3X3\0AREA_5X5\0\0");
                m_ui.RelaxSettings.hitDistanceReconstructionMode = (nrd::HitDistanceReconstructionMode)hitDistanceReconstructionMode;

                ImGui::Checkbox("Enable Firefly Filter", &m_ui.RelaxSettings.enableAntiFirefly);

                ImGui::Checkbox("Roughness Edge Stopping", &m_ui.RelaxSettings.enableRoughnessEdgeStopping);

                // ImGui::Checkbox("Enable Diffuse Material Test", &m_ui.RelaxSettings.enableMaterialTestForDiffuse);
                // ImGui::Checkbox("Enable Specular Material Test", &m_ui.RelaxSettings.enableMaterialTestForSpecular);
            }

            // Not really needed for now since we have reference codepath, but it could be used to debug some of the NRD codepaths so leaving in as a reminder
            // ImGui::Checkbox("Reference Accumulation", &m_ui.NRDReferenceSettings.maxAccumulatedFrameNum);
        }
    }
}
