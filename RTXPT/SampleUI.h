/*
* Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <donut/core/math/math.h>
#include <donut/engine/Scene.h>

#include <donut/app/imgui_renderer.h>
#include <donut/app/imgui_console.h>

#include "RTXDI/RtxdiPass.h"

#include <donut/render/TemporalAntiAliasingPass.h>

using namespace donut::math;

#include "ToneMapper/ToneMappingPasses.h"
#include "PathTracer/ShaderDebug.hlsli"

#if ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
#include "ImNodesEz.h"
#endif

#ifdef STREAMLINE_INTEGRATION
#include "sl.h"
#include "sl_dlss.h"
#include "sl_reflex.h"
#include<sl_dlss_g.h>
#endif

#include "NRD/NrdConfig.h"

namespace donut::engine
{
    class SceneGraphNode;
}

struct TogglableNode
{
    donut::engine::SceneGraphNode * SceneNode;
    dm::double3                     OriginalTranslation;
    std::string                     UIName;
    bool                            IsSelected() const;
    void                            SetSelected( bool selected ) ;
};

struct OpacityMicroMapUIData
{
    struct BuildState
    {
        // ~~ Application is expected to tweak these settings ~~ 
        int MaxSubdivisionLevel = 12;
        bool EnableDynamicSubdivision = true;
        float DynamicSubdivisionScale = 1.f;
        nvrhi::rt::OpacityMicromapBuildFlags Flag = nvrhi::rt::OpacityMicromapBuildFlags::FastTrace;
        nvrhi::rt::OpacityMicromapFormat Format = nvrhi::rt::OpacityMicromapFormat::OC1_4_State;

        // ~~ Debug settings, application is expected to leave to default ~~ 
        bool ComputeOnly = true;
        bool LevelLineIntersection = true;
        bool EnableTexCoordDeduplication = true;
        bool Force32BitIndices = false;
        bool EnableNsightDebugMode = false;
        bool EnableSpecialIndices = true;
        int MaxOmmArrayDataSizeInMB = 100;

        bool operator == (const BuildState& o) const {
            return
                MaxSubdivisionLevel == o.MaxSubdivisionLevel &&
                EnableDynamicSubdivision == o.EnableDynamicSubdivision &&
                DynamicSubdivisionScale == o.DynamicSubdivisionScale &&
                Flag == o.Flag &&
                Format == o.Format &&
                ComputeOnly == o.ComputeOnly &&
                LevelLineIntersection == o.LevelLineIntersection &&
                EnableTexCoordDeduplication == o.EnableTexCoordDeduplication &&
                Force32BitIndices == o.Force32BitIndices &&
                EnableNsightDebugMode == o.EnableNsightDebugMode &&
                EnableSpecialIndices == o.EnableSpecialIndices &&
                MaxOmmArrayDataSizeInMB == o.MaxOmmArrayDataSizeInMB
                ;
        }
    };

    bool                                Enable = true;
    bool                                Force2State = false;
    bool                                OnlyOMMs = false;

    // Amortize the builds over multiple frames
    std::optional<BuildState>           ActiveState;
    BuildState                          DesiredState;
    bool                                TriggerRebuild = true;

    // --- Stats --- 
    // build progress of active tasks
    uint32_t                            BuildsLeftInQueue = 0;
    uint32_t                            BuildsQueued = 0;
};

struct AccelerationStructureUIData
{
    // Instance settings (no rebuild required)
    bool                                ForceOpaque = false;

    // BVH settings (require rebuild to take effect)
    bool                                ExcludeTransmissive = false;

    bool                                IsDirty = false;
};

struct EnvironmentMapRuntimeParameters
{
    dm::float3  TintColor = { 1.f, 1.f, 1.f };
    float       Intensity = 1.f;
    dm::float3  RotationXYZ = { 0.f, 0.f, 0.f };
    bool        Enabled = true;
};

struct SampleUIData
{
    bool                                ShowUI = true;
    int                                 FPSLimiter = 0; // 0 - no limit, otherwise limit fps to FPSLimiter and fix scene update deltaTime to 1./FPSLimiter
    bool                                ShowConsole = false;
    bool                                EnableAnimations = false;
    bool                                EnableVsync = false;
    std::shared_ptr<donut::engine::Material> SelectedMaterial;
    bool                                ShaderReloadRequested = false;
    float                               ShaderReloadDelayedRequest = 0.0f;
    std::string                         ScreenshotFileName;
    std::string                         ScreenshotSequencePath = "D:/AnimSequence/";
    bool                                ScreenshotSequenceCaptureActive = false;
    int                                 ScreenshotSequenceCaptureIndex = -64; // -x means x warmup frames for recording to stabilize denoiser
    bool                                LoopLongestAnimation = false; // some animation sequences want to loop only the longest, but some want to loop each independently
    bool                                ExperimentalPhotoModeScreenshot = false;

    bool                                UseStablePlanes = false; // only determines whether UseStablePlanes is used in Accumulate mode (for testing correctness and enabling RTXDI) - in Realtime mode or when using RTXDI UseStablePlanes are necessary
    bool                                AllowRTXDIInReferenceMode = false; // allows use of RTXDI even in reference mode
    bool                                UseNEE                      = true;
    int                                 NEEDistantType              = 2;        // 0 - uniform; 1 - MIP descent; 2 - pre-sampling, 3 - ...
    int                                 NEEDistantCandidateSamples  = 1;        // each full sample is picked from a number of candidate samples
    int                                 NEEDistantFullSamples       = 2;        // each full sample requires a shadow ray!
    int                                 NEELocalType                = 2;        // '0' is uniform, '1' is power (with pre-sampling), '2' is ReGIR; once this solidifies make it a proper enum
    int                                 NEELocalCandidateSamples    = 4;        // each full sample is picked from a number of candidate samples
    int                                 NEELocalFullSamples         = 2;        // each full sample requires a shadow ray!
    int                                 NEEBoostSamplingOnDominantPlane = 2;    // Boost light sampling only on the dominant denoising surface
    float                               NEEMinRadianceThresholdMul = 1e-3f;
    bool                                UseReSTIRDI = false;
    bool                                UseReSTIRGI = false;
    bool                                RealtimeMode = false;
    int                                 RealtimeSamplesPerPixel = 1;        // equivalent to m_ui.AccumulationTarget in reference mode (except looping x times within frame)
    bool                                RealtimeNoise = true;               // stops noise from changing at real-time - useful for reproducing rare bugs
    bool                                RealtimeDenoiser = true;
    bool                                ResetAccumulation = false;
    int                                 BounceCount = 30;
    int                                 ReferenceDiffuseBounceCount = 6;
    int                                 RealtimeDiffuseBounceCount = 3;
    int                                 AccumulationTarget = 4096;
    int                                 AccumulationIndex = 0;
    bool                                AccumulationAA = true;
    int                                 RealtimeAA = 2;                     // 0 - no AA, 1 - TAA, 2 - DLSS,  3 - DLAA
    float                               CameraAperture = 0.0f;
    float                               CameraFocalDistance = 10000.0f;
    float                               CameraMoveSpeed = 2.0f;
    float                               TexLODBias = -1.0f;                 // as small as possible without reducing performance!
    bool                                SuppressPrimaryNEE = false;

    donut::render::TemporalAntiAliasingParameters TemporalAntiAliasingParams;
    donut::render::TemporalAntiAliasingJitter     TemporalAntiAliasingJitter = donut::render::TemporalAntiAliasingJitter::Halton;

    bool                                ContinuousDebugFeedback = false;
    bool                                ShowDebugLines = false;
    donut::math::uint2                  DebugPixel = { 0, 0 };
    donut::math::uint2                  MousePos = { 0, 0 };
    float                               DebugLineScale = 0.2f;

    bool                                ShowSceneTweakerWindow = false;

    EnvironmentMapRuntimeParameters     EnvironmentMapParams;

    bool                                EnableToneMapping = true;
    ToneMappingParameters               ToneMappingParams;

    DebugViewType                       DebugView = DebugViewType::Disabled;
    int                                 DebugViewStablePlaneIndex = -1;
    bool                                ShowWireframe;

    bool                                ReferenceFireflyFilterEnabled = true;
    float                               ReferenceFireflyFilterThreshold = 2.5f;
    bool                                RealtimeFireflyFilterEnabled = true;
    float                               RealtimeFireflyFilterThreshold = 0.25f;

    float                               DenoiserRadianceClampK = 8.0f;

    bool                                EnableRussianRoulette = true;

    bool                                DXRHitObjectExtension = true;
    bool                                ShaderExecutionReordering = true;
    OpacityMicroMapUIData               OpacityMicroMaps;
    AccelerationStructureUIData         AS;

    RtxdiUserSettings                   RTXDI;
    
    bool                                ShowDeltaTree = false;
    bool                                ShowMaterialEditor = true;  // this makes material editor default right click option

#ifdef STREAMLINE_INTEGRATION
    float                               DLSS_Sharpness = 0.f;
    bool                                DLSS_Supported = false;
    static constexpr sl::DLSSMode       DLSS_ModeDefault = sl::DLSSMode::eMaxQuality;
    sl::DLSSMode                        DLSS_Mode = DLSS_ModeDefault;
    bool                                DLSS_Dynamic_Res_change = true;
    donut::math::int2                   DLSS_Last_DisplaySize = { 0,0 };
    sl::DLSSMode                        DLSS_Last_Mode = sl::DLSSMode::eOff;
    int                                 DLSS_Last_RealtimeAA = 0;
    bool                                DLSS_DebugShowFullRenderingBuffer = false;
    bool                                DLSS_lodbias_useoveride = false;
    float                               DLSS_lodbias_overide = 0.f;
    bool                                DLSS_always_use_extents = false;

    // LATENCY specific parameters
    bool                                REFLEX_Supported = false;
    bool                                REFLEX_LowLatencyAvailable = false;
    int                                 REFLEX_Mode = sl::ReflexMode::eOff;
    int                                 REFLEX_CapedFPS = 0;
    std::string                         REFLEX_Stats = "";
    bool                                REFLEX_ShowStats = false;
    int                                 FpsCap = 60;

    // DLFG specific parameters
    bool                                DLSSG_Supported = false;
    sl::DLSSGMode                       DLSSG_mode = sl::DLSSGMode::eOff;
    int                                 DLSSG_multiplier = 1;
#endif

    // See UI tooltips for more info (or search code for ImGui::SetTooltip()!)
    int                                 StablePlanesActiveCount             = cStablePlaneCount;
    int                                 StablePlanesMaxVertexDepth          = std::min(14u, cStablePlaneMaxVertexIndex);
    float                               StablePlanesSplitStopThreshold      = 0.95f;
    float                               StablePlanesMinRoughness            = 0.06f;
    bool                                AllowPrimarySurfaceReplacement      = true;
    bool                                StablePlanesSuppressPrimaryIndirectSpecular = true;
    float                               StablePlanesSuppressPrimaryIndirectSpecularK = 0.6f;
    float                               StablePlanesAntiAliasingFallthrough = 0.6f;
    //bool                                StablePlanesSkipIndirectNoisePlane0 = false;

    std::shared_ptr<std::vector<TogglableNode>> TogglableNodes = nullptr;

    bool                                ActualUseStablePlanes() const               { return UseStablePlanes || RealtimeMode || ((AllowRTXDIInReferenceMode) && (UseReSTIRDI || UseReSTIRGI)); }
    //bool                                ActualSkipIndirectNoisePlane0() const       { return StablePlanesSkipIndirectNoisePlane0 && StablePlanesActiveCount > 2; }

    bool                                ActualUseRTXDIPasses() const                { return (RealtimeMode || AllowRTXDIInReferenceMode) && (UseReSTIRDI || UseReSTIRGI) || ((NEELocalFullSamples>0 && UseNEE)); }
    bool                                ActualUseReSTIRDI() const                   { return UseNEE && (RealtimeMode || AllowRTXDIInReferenceMode) && (UseReSTIRDI); }
    bool                                ActualUseReSTIRGI() const                   { return (RealtimeMode || AllowRTXDIInReferenceMode) && (UseReSTIRGI); }
    uint                                ActualSamplesPerPixel() const               { return (RealtimeMode && !(UseReSTIRDI || UseReSTIRGI))?RealtimeSamplesPerPixel:1u; }

    // Denoiser
    bool                                NRDModeChanged = false;
    NrdConfig::DenoiserMethod           NRDMethod = NrdConfig::DenoiserMethod::RELAX;
    float                               NRDDisocclusionThreshold = 0.01f;
    bool                                NRDUseAlternateDisocclusionThresholdMix = true;
    float                               NRDDisocclusionThresholdAlternate = 0.1f;
    nrd::RelaxDiffuseSpecularSettings   RelaxSettings;
    nrd::ReblurSettings                 ReblurSettings;
    //nrd::ReferenceSettings              NRDReferenceSettings;
};

class SampleUI : public donut::app::ImGui_Renderer
{
private:
    class Sample & m_app;

    ImFont* m_FontDroidMono = nullptr;
    std::pair<ImFont*, float>   m_scaledFonts[14];
    int                         m_currentFontScaleIndex = -1;
    float                       m_currentScale = 1.0f;
    ImGuiStyle                  m_defaultStyle;

    float                       m_showSceneWidgets = 0.0f;

    std::unique_ptr<donut::app::ImGui_Console> m_console;
    std::shared_ptr<donut::engine::Light> m_SelectedLight;

    SampleUIData& m_ui;
    nvrhi::CommandListHandle m_CommandList;

    const bool m_SERSupported;
    const bool m_OMMSupported;

#if ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
    ImNodes::Ez::Context* m_ImNodesContext;
#endif

public:
    SampleUI(donut::app::DeviceManager* deviceManager, class Sample & app, SampleUIData& ui, bool SERSupported, bool OMMSupported);
    virtual ~SampleUI();
protected:
    virtual void buildUI(void) override;
private:
    void buildDeltaTreeViz();

    virtual void Animate(float elapsedTimeSeconds) override;
    virtual bool MousePosUpdate(double xpos, double ypos) override;

    int FindBestScaleFontIndex(float scale);
};

void UpdateTogglableNodes(std::vector<TogglableNode>& TogglableNodes, donut::engine::SceneGraphNode* node);
