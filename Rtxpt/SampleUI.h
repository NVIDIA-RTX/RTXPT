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

#include <donut/core/math/math.h>
#include <donut/engine/Scene.h>

#include <donut/app/imgui_renderer.h>
#include <donut/app/imgui_console.h>

#include "RTXDI/RtxdiPass.h"

#include <donut/render/TemporalAntiAliasingPass.h>

using namespace donut::math;

#include "ToneMapper/ToneMappingPasses.h"
#include "Shaders/PathTracer/PathTracerDebug.hlsli"

#if ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
#include "ImNodesEz.h"
#endif

#if DONUT_WITH_STREAMLINE
#include <donut/app/StreamlineInterface.h>
#endif

#include "NRD/NrdConfig.h"

#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
#include "../../external/RtxTf/STFDefinitions.h"
#endif

namespace donut::engine
{
    class SceneGraphNode;
}

#if DONUT_WITH_STREAMLINE
typedef donut::app::StreamlineInterface SI;
#endif

struct TogglableNode
{
    donut::engine::SceneGraphNode * SceneNode;
    dm::double3                     OriginalTranslation;
    std::string                     UIName;
    bool                            IsSelected() const;
    void                            SetSelected( bool selected ) ;
};

struct AccelerationStructureUIData
{
    // Instance settings (no rebuild required)
    bool                                ForceOpaque = false;

    // BVH settings (require rebuild to take effect)
    bool                                ExcludeTransmissive = false;

    // Build the TLAS instance descriptors with a compute pass instead of the CPU loop
    // (Sample::BuildTLASGPU vs Sample::BuildTLAS). The CPU path stays available for A/B.
    bool                                GPUTLASBuild = true;

    // Frustum + distance culling inside that compute pass. Off by default; a sample that wants it can
    // set it in its constructor. Ignored outside realtime mode either way: rays are not bounded by the
    // view frustum, so dropping instances changes shadows, reflections and GI, which an accumulating
    // reference image must not see.
    bool                                GPUTLASCull = false;
    // Metres the side planes are pushed outward. The skirt is what keeps off-screen shadow casters and
    // reflection sources in the TLAS; the retained set still varies with camera orientation, so a wider
    // one moves that boundary out rather than removing the dependency.
    float                               GPUTLASCullFrustumMargin = 100.0f;
    // Metres beyond which an instance is culled regardless of the frustum. Camera-position dependent
    // only, so unlike the margin it is invariant under pure camera rotation.
    float                               GPUTLASCullDistanceCutoff = 1000.0f;
};

struct EnvironmentMapRuntimeParameters
{
    dm::float3  TintColor = { 1.f, 1.f, 1.f };
    float       Intensity = 1.f;
    dm::float3  RotationXYZ = { 0.f, 0.f, 0.f };
    bool        Enabled = true;
};

#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
enum class StfFilterMode
{
    Point = 0,
    Linear,
    Cubic,
    Gaussian
};

enum class StfMagnificationMethod
{
    Default = 0,
    Quad2x2,
    Fine2x2,
    FineTemporal2x2,
    FineAlu3x3,
    FineLut3x3,
    Fine4x4
};
#endif // RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE

// Which pixels get a path in the realtime sub-sampling modes (sub-1spp). The rate (half/quarter/three quarters) comes
// from RealtimeSubSamplingMode; every pattern weighs a traced pixel by 1/rate and leaves the rest at zero noisy radiance.
// Must match the RTXPT_SUB_SAMPLING_PATTERN_* values in Shaders/Misc/SubSampling.h.
enum class SubSamplingPattern : int
{
    BlockJitter = 0,        // one uniformly random pixel per 2x1 (half) / 2x2 (quarter) block, all but one (three quarters) - the original method
    StochasticCompacted,    // per-pixel white noise coin flip; selected pixels are stream-compacted into a 1D launch
    BlockStratified,        // each NxN block gets exactly N*N*rate pixels, at random distinct positions
    SobolBlock,             // per NxN block, Owen-scrambled Sobol (0,2) sequence points: stratified, every pixel traced exactly at the rate over each 2 / 4 frame cycle
    Count
};

struct PerformancePreset
{
    const char* Name;
    int         SubSamplingMode;
    int         NEECandidateSamples;
    int         NEEFullSamples;
    int         NEEMISType;
    int         RealtimeSamplesPerPixel;
    int         BounceCount;
    int         DiffuseBounceCount;
    float       TexLODBias;
    int         NestedDielectricsQuality;
    int         EnvironmentMapDiffuseSampleMIPLevel;
    int         DenoisingLayersActiveCount;
    bool        AllowPrimarySurfaceReplacement;
    bool        EnableBloom;
    bool        EnableLDSamplerForBSDF;
    float       FireflyThreshold;
    bool        OmmForce2State;
#if DONUT_WITH_STREAMLINE
    SI::DLSSMode DLSSMode;
#endif
};

enum class IntroDirectLightingMode : int
{
    Unshadowed = 0,
    UniformSampling,
    StreamingRIS,
    ReSTIRDI
};

struct SampleUIData
{
    bool                                ActualUseRTXDIPasses() const { return (RealtimeMode) && (ActualUseReSTIRDI() || ActualUseReSTIRGI()); }
    bool                                ActualUseReSTIRDI() const { return UseNEE && (RealtimeMode) && (UseReSTIRDI) && (RealtimeAA < 3 || !DisableReSTIRsWithDLSSRR); }
    bool                                ActualUseReSTIRGI() const { return (RealtimeMode) && (UseReSTIRGI) && (RealtimeAA < 3 || !DisableReSTIRsWithDLSSRR); }
    uint                                ActualSamplesPerPixel() const { return (RealtimeMode && !(ActualUseReSTIRDI() || ActualUseReSTIRGI())) ? RealtimeSamplesPerPixel : 1u; }
    uint                                ActualSubSamplingMode() const { return (RealtimeMode && !(ActualUseReSTIRDI() || ActualUseReSTIRGI())) ? RealtimeSubSamplingMode : 0; }
    uint                                SubSamplingScaleX() const { return ActualSubSamplingMode() > 0 ? 2 : 1; }
    uint                                SubSamplingScaleY() const { return (ActualSubSamplingMode() == 1 || ActualSubSamplingMode() == 3) ? 2 : 1; }
    uint                                SubSamplingRateNum() const { return ActualSubSamplingMode() == 1 ? 3 : 1; }   // traced pixels per ScaleX x ScaleY block
    SubSamplingPattern                  ActualSubSamplingPattern() const { return ActualSubSamplingMode() > 0 ? (SubSamplingPattern)RealtimeSubSamplingPattern : SubSamplingPattern::BlockJitter; }
    uint                                SubSamplingStratifiedBlockSize() const { return (RealtimeSubSamplingStratifiedBlockSize >= 16) ? 16u : ((RealtimeSubSamplingStratifiedBlockSize >= 8) ? 8u : 4u); }  // power of two: the shader's per-block permutation relies on it
    bool                                ActualUseStandaloneDenoiser() const { return (RealtimeMode && RealtimeAA < 3) ? StandaloneDenoiser : false; }
    float                               ActualNEEAT_LocalToGlobalSampleRatio() const { return (NEEType == 2) ? (NEEAT_LocalToGlobalSampleRatio) : (0); }    // make sure we use no local samples when NEE-AT disabled!
    bool                                ActualFireflyFilterEnabled() const { return (RealtimeMode)?RealtimeFireflyFilterEnabled:ReferenceFireflyFilterEnabled; }
    int                                 ActualRadianceCacheMode() const { return (RealtimeMode)?RadianceCacheMode:0; }

#if DONUT_WITH_STREAMLINE
    int                                 ActualReflexMode() const { return (RealtimeMode && IsReflexSupported) ? ((ReflexMode == SI::ReflexMode::eOff && ActualDLSSFGMode() != SI::DLSSGMode::eOff) ? (SI::ReflexMode::eLowLatency) : (ReflexMode)) : (SI::ReflexMode::eOff); }
    SI::DLSSGMode                       ActualDLSSFGMode() const { return (RealtimeMode && IsDLSSFGSupported) ? (DLSSFGMode) : (SI::DLSSGMode::eOff); }

    bool                                ActualEnableVsync() const { return (ActualDLSSFGMode() != SI::DLSSGMode::eOff) ? (false) : (EnableVsync); }
    int                                 ActualFPSLimiter() const { return (ActualDLSSFGMode() != SI::DLSSGMode::eOff) ? (0) : (FPSLimiter); }
#else
    bool                                ActualEnableVsync() const { return EnableVsync; }
    int                                 ActualFPSLimiter() const { return FPSLimiter; }
#endif

    bool                                ActualUseApproximateMIS() const { return (RealtimeMode)?(NEEMISType!=0):(NEEMISType==2); }

    // In the future we'll likely drop support for multi-layer NRD 
    //int                                 ActualDenoisingLayersActiveCount() const { return ActualUseStandaloneDenoiser()?1:DenoisingLayersActiveCount; }
    int                                 ActualDenoisingLayersActiveCount() const { return DenoisingLayersActiveCount; }


    bool                                ShowUI                                  = true;
    int                                 FPSLimiter                              = 0; // 0 - no limit, otherwise limit fps to FPSLimiter and fix scene update deltaTime to 1./FPSLimiter
    bool                                RenderWhenOutOfFocus                    = false; // if window is out of focus window render loop is paused
    bool                                ShowConsole                             = false;
    bool                                EnableAnimations                        = false;
    bool                                EnableVsync                             = false;
    std::shared_ptr<donut::engine::Material> SelectedMaterial;
    bool                                ShaderReloadRequested                   = false;
    bool                                AccelerationStructRebuildRequested      = false;
    float                               ShaderAndACRefreshDelayedRequest        = 0.0f;
    bool                                ExperimentalPhotoModeScreenshot         = false;

    bool                                UseNEE          /*Defaults in CommandLine >*/;
    int                                 NEEType         /*Defaults in CommandLine >*/;      // '0' is uniform, '1' is power, '2' is NEE-AT; once this solidifies make it a proper enum
    int                                 NEECandidateSamples                     = 5;        // each full sample is picked from a number of candidate samples; these are not visibility tested so taking too many can hurt quality in heavily shadowed scenarios
    int                                 NEEFullSamples                          = 1;        // each full sample requires a shadow ray!
    int                                 NEEMISType                              = 1;        // '0' full MIS always; '1' full MIS in reference, approx in realtime; '2' approx MIS always
    //bool                                NEEAT_AntiLagPass                       = false;
    float                               NEEAT_GlobalTemporalFeedbackWeight      = 0.75f;
    float                               NEEAT_LocalToGlobalSampleRatio          = 0.65f;
    //float                               NEEAT_MIS_Boost                         = 1.0f;
    float                               NEEAT_Distant_vs_Local_Importance       = 1.0f;
    int                                 NEEAT_QualityPreset                     = 0;        // NEEAT_QUALITY_PRESET (see NEEATConfig.h); a change re-creates NEE-AT buffers and recompiles its and the path tracer's shaders
    //int                                 NEEBoostSamplingOnDominantPlane = 0;    // Boost light sampling only on the dominant denoising surface
    bool                                UseReSTIRDI /*Defaults in CommandLine >*/;
    bool                                UseReSTIRGI /*Defaults in CommandLine >*/;
    bool                                RealtimeMode = true;
    int                                 RealtimeSamplesPerPixel /*Defaults in CommandLine >*/;        // equivalent to m_ui.AccumulationTarget in reference mode (except looping x times within frame)
    bool                                StandaloneDenoiser /*Defaults in CommandLine >*/;
    bool                                ResetAccumulation = false;
    bool                                ResetRealtimeCaches = false;
    int                                 BounceCount = 20;
    int                                 DiffuseBounceCount = 2;             // should be 2 on default quality, 3 on ultra high and 1 on ultra fast
    int                                 AccumulationTarget /*Defaults in CommandLine >*/;
    bool                                AccumulationPreWarmRealtimeCaches = true;
    bool                                AccumulationAA = true;
    int                                 RealtimeAA /*Defaults in CommandLine >*/;           // 0 - no AA, 1 - TAA, 2 - DLSS, 3 - DLSS-RR (if available)
    int                                 RealtimeSubSamplingMode = 1;                            // 0 - no SubSampling, 1 - three quarters, 2 - half res, 3 - quarter res
    int                                 RealtimeSubSamplingPattern = (int)SubSamplingPattern::SobolBlock;
    int                                 RealtimeSubSamplingStratifiedBlockSize = 8;             // N for SubSamplingPattern::BlockStratified and ::SobolBlock; power of two, 4..16
    float                               CameraAperture = 0.0f;
    float                               CameraFocalDistance = 10000.0f;
    float                               CameraMoveSpeed = 1.0f;
    float                               CameraAntiRRSleepJitter = 0.0f;
    float                               TexLODBias = -1.0f;                 // as small as possible without reducing performance!
    int                                 NestedDielectricsQuality    = 1;    // 0 - off; 1 - fast; 2 - quality
    bool                                UseFp16Types = false;
    bool                                EnableLDSamplerForBSDF = true;
#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
    StfFilterMode                       STFFilterMode = StfFilterMode::Linear;
    StfMagnificationMethod              STFMagnificationMethod = StfMagnificationMethod::Default;
    float                               STFGaussianSigma = 0.3f;
#endif // RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE

    donut::render::TemporalAntiAliasingParameters TemporalAntiAliasingParams;
    donut::render::TemporalAntiAliasingJitter     TemporalAntiAliasingJitter = donut::render::TemporalAntiAliasingJitter::Halton;   // what DLSS is trained on (DLSS guide 3.7.1)

    bool                                ContinuousDebugFeedback = false;
    bool                                ShowDebugLines = false;
    donut::math::uint2                  DebugPixel = { 0, 0 };
    donut::math::uint2                  MousePos = { 0, 0 };
    float                               DebugLineScale = 0.05f;

    bool                                EnableShaderDebug = true;   // see ShaderDebug.hlsli/.h/.cpp

    bool                                ShowSceneTweakerWindow = false;

    EnvironmentMapRuntimeParameters     EnvironmentMapParams;

    bool                                EnableToneMapping = true;
    ToneMappingParameters               ToneMappingParams;

    DebugViewType                       DebugView = DebugViewType::Disabled;
    int                                 DebugViewDenoisingLayerIndex = -1;
    bool                                ShowWireframe;

    bool                                ReferenceFireflyFilterEnabled = true;
    float                               ReferenceFireflyFilterThreshold = 2.5f;
    bool                                RealtimeFireflyFilterEnabled = true;
    float                               RealtimeFireflyFilterThreshold = 0.10f;

    float                               DenoiserRadianceClampK = 8.0f;
    float                               DLSSRRBrightnessClampK = 4096.0f;

    bool                                EnableRussianRoulette = true;

    int                                 EnvironmentMapDiffuseSampleMIPLevel = 2;

    bool                                NVAPIHitObjectExtension = true;
    bool                                NVAPIReorderThreads     = true;

    bool                                DXHitObjectExtension    = false;
    bool                                DXMaybeReorderThreads   = true;

    AccelerationStructureUIData         AS;

    RtxdiUserSettings                   RTXDI;
    
    bool                                ShowDeltaTree = false;
    bool                                ShowMaterialEditor = true;  // this makes material editor default right click option

#if DONUT_WITH_STREAMLINE
    // DLSS specific parameters
    //float                               DLSSSharpness = 0.f;
    //bool                                DLSSDynamicResChange = true;
    //bool                                DLSSDebugShowFullRenderingBuffer = false;
    bool                                IsDLSSSuported = false;
    static constexpr SI::DLSSMode       DLSSModeDefault = SI::DLSSMode::eBalanced;
    SI::DLSSMode                        DLSSMode = DLSSModeDefault;
    SI::DLSSMode                        DLSSLastMode = SI::DLSSMode::eOff;
    donut::math::uint2                  DLSSLastDisplaySize = { 0,0 };
    int                                 DLSSLastRealtimeAA = 0;
    bool                                DLSSLodBiasUseOverride = false;
    float                               DLSSLodBiasOverride = 0.f;
    bool                                DLSSAlwaysUseExtents = false;

    // DLSSFG specific parameters
    bool                                IsDLSSFGSupported = false;
    SI::DLSSGMode                       DLSSFGMode = SI::DLSSGMode::eOff;
    SI::DLSSGOptions                    DLSSFGOptions = {};
    uint32_t                            DLSSFGMultiplier = 1;
    uint32_t                            DLSSFGNumFramesToGenerate = 1;
    uint32_t                            DLSSFGMaxNumFramesToGenerate = 1;

    // Reflex latency specific parameters
    bool                                IsReflexSupported = false;
    bool                                IsReflexLowLatencyAvailable = false;
    bool                                IsReflexFlashIndicatorDriverControlled = false;
    int                                 ReflexMode = SI::ReflexMode::eLowLatency;    // only used when supported, see ActualReflexMode
    int                                 ReflexCappedFps = 0;
    bool                                ReflexShowStats = false;
    std::string                         ReflexStats = "";
    int                                 FpsCap = 60;

    // DLSS-RR specific parameters
    bool                                IsDLSSRRSupported = false;
    SI::DLSSRRPreset                    DLSRRPreset = SI::DLSSRRPreset::ePresetF;    // SL 2.14.1 calls F the latest/default transformer model
#endif // DONUT_WITH_STREAMLINE

    float                               DLSSRRMicroJitter = 0.01f;

    // See UI tooltips for more info (or search code for ImGui::SetTooltip()!)
    int                                 DenoisingLayersActiveCount             = cDenoisingLayerCount;
    int                                 DenoisingLayersMaxVertexDepth          = std::min(9u, cDenoisingLayerMaxVertexIndex); // more is not necessarily better with current heuristics
    float                               DenoisingLayersSplitStopThreshold      = 0.95f;
    bool                                AllowPrimarySurfaceReplacement      = true;
    bool                                DenoisingLayersSuppressPrimaryIndirectSpecular = true;
    float                               DenoisingLayersSuppressPrimaryIndirectSpecularK = 0.6f;
    float                               DenoisingLayersAntiAliasingFallthrough = 0.6f;
    //bool                                DenoisingLayersSkipIndirectNoisePlane0 = false;

    bool                                DisableReSTIRsWithDLSSRR            = true;

    std::shared_ptr<std::vector<TogglableNode>> TogglableNodes = nullptr;

    // Denoiser
    bool                                NRDModeChanged = false;
    NrdConfig::DenoiserMethod           NRDMethod = NrdConfig::DenoiserMethod::REBLUR;
    float                               NRDDisocclusionThreshold = 0.03f;
    bool                                NRDUseAlternateDisocclusionThresholdMix = true;
    float                               NRDDisocclusionThresholdAlternate = 0.2f;
    nrd::RelaxSettings                  RelaxSettings;
    nrd::ReblurSettings                 ReblurSettings;
    //nrd::ReferenceSettings              NRDReferenceSettings;

    int                                 RadianceCacheMode = 1;      // 0 - disabled, 1 - SHaRC, (future, not implemented) 2 - NRC

    // TODO: we need to compartmentalize specific sample UI knobs away from SampleUIData into IntroSampleUIData & AdvancedSampleUIData (inheritance/whatever)
    bool                                PostProcessTestPassHDR = false;
    IntroDirectLightingMode             IntroDirectLighting = IntroDirectLightingMode::Unshadowed;
    bool                                UseRTAO = false;    // false = GTAO (screen-space), true = RTAO (ray-traced)
    float                               AORadius = 50.0f;   // world-space AO occlusion radius, shared by GTAO and RTAO
    bool                                UseRTReflections = false;  // false = SSR (screen-space), true = RT reflections
    bool                                UseRTGI = false;           // false = environment IBL diffuse, true = one-bounce RT GI
    uint32_t                            RTGIEnvMip = 5;            // mip RTGI samples the prefiltered env cube at (GI-ray miss + hit env term); trades environment detail/fireflies for bias
    bool                                DebugRTGI = false;         // when true (and UseRTGI), replaces deferred lighting with raw RTGI irradiance visualization
    bool                                DebugIndirectSpecular = false; // shows only the weighted indirect specular contribution from SSR, RT reflections or MIS
    bool                                UseSHaRC = false;          // when true + UseRTGI, SHaRC radiance cache accelerates RTGI
    bool                                DebugSHaRCCache = false;   // when true (and UseSHaRC), replaces deferred lighting with cache radiance visualization
    bool                                UseMISIndirect = false;    // when true, unified spec+diffuse indirect replaces separate reflections+RTGI


    bool                                PostProcessEdgeDetection = false;
    float                               PostProcessEdgeDetectionThreshold = 0.1f;

    bool                                EnableBloom = true;
    float                               BloomRadius = 8.0f;
    float                               BloomIntensity = 0.004f;

    bool                                DbgFreezeRealtimeNoiseSeed = false;               // stops noise from changing at real-time - useful for reproducing rare bugs
    bool                                DbgDisableSERTerminationHint = false;

    bool                                DbgDiscardNonNEELighting = false;
    bool                                DbgDiscardNEELighting = false;

    bool                                DbgDisablePostProcessFilters = false;

    int                                 MaterialVariantIndex = 0;                       // each scene can have multiple material presets

    bool                                EnableAsyncPrefix = true;                    // only used when supported, see Sample::IsAsyncPrefixAvailable
};

void ApplyPerformancePreset(SampleUIData& ui, uint presetIndex, class OmmBaker* ommBaker);

extern SampleUIData g_sampleUIData;

class SampleUI : public donut::app::ImGui_Renderer
{
public:
    SampleUI(donut::app::DeviceManager* deviceManager, class SampleBaseApp & baseApp, class Sample & app, SampleUIData& ui, bool NVAPI_SERSupported, const struct CommandLineOptions& cmdLine);
    virtual ~SampleUI();
protected:
    virtual void buildUI(void) override;
    virtual void DrawSampleUI(float indent, float defItemWidth, const struct ImVec4& warnColor, const struct ImVec4& categoryColor) { (void)indent; (void)defItemWidth; (void)warnColor; (void)categoryColor; }

#if DONUT_WITH_STREAMLINE
    SI::DLSSMode DLSSModeUI(SI::DLSSMode dlssModeCurrent);
#endif // DONUT_WITH_STREAMLINE

protected:
    class Sample& m_app;
    SampleUIData& m_ui;

    const bool m_NVAPI_SERSupported;
private:
    void buildDeltaTreeViz();

    virtual bool MousePosUpdate(double xpos, double ypos) override;
    virtual void DisplayScaleChanged(float scaleX, float scaleY) override { m_currentScale = scaleX; assert( scaleX == scaleY ); }
    virtual void Animate(float elapsedTimeSeconds) override;

    bool BuildUIScriptsAndEtc(void);
    void BuildUIResolutionPicker();
    void BuildUIPerformancePresets();
    
    void DLSSFGSelectorUI();

private:
    class SampleBaseApp& m_baseApp;

    int                         m_currentFontScaleIndex = -1;
    float                       m_currentScale = 1.0f;
    ImGuiStyle                  m_defaultStyle;

    float                       m_showSceneWidgets = 0.0f;

    std::unique_ptr<donut::app::ImGui_Console> m_console;
    std::shared_ptr<donut::engine::Light> m_SelectedLight;

    nvrhi::CommandListHandle m_commandList;

#if ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
    ImNodes::Ez::Context* m_ImNodesContext;
#endif
};

void UpdateTogglableNodes(std::vector<TogglableNode>& TogglableNodes, donut::engine::SceneGraphNode* node);
