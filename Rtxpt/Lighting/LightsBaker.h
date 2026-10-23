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

#include "Shaders/PathTracer/Config.h"

#include <donut/engine/BindingCache.h>
#include <nvrhi/nvrhi.h>
#include <donut/core/math/math.h>
#include <memory>

#include <donut/core/math/math.h>

#define NEEAT_BAKER_ONLY 1
#include "../Shaders/PathTracer/Lighting/LightingTypes.hlsli"

#include "../Shaders/SubInstanceData.h"

#include "../Shaders/Libraries/NEE-AT/NEEATBaker.hlsli"

// Dual C++/HLSL header; everything outside its '!defined(__cplusplus)' block is shared - SubInstanceLightPatch here.
#include "../Shaders/Libraries/LightsBaker/LightsBaker.hlsl"

#include <filesystem>

using namespace donut::math;

namespace donut::engine
{
    class FramebufferFactory;
    class TextureCache;
    class TextureHandle;
    class ShaderFactory;
    class CommonRenderPasses;
    struct TextureData;
    struct ShaderMacro;
}

class ShaderDebug;
class ExtendedScene;
class SceneRenderCache;
class ComputePipelineBaker;
class ComputeShaderVariant;

// this is a fallback for when the engine light source can't be modified to add tracking (otherwise, see 'struct LightSamplerLink')
// #define HASH_LOOKUP_BASED_HISTORIC_LIGHT_SOURCE_MATCHING

// ProcessEmissiveGeometry's two SubInstanceData light fields are patched on the GPU from a sparse list instead of the
// caller re-uploading the whole sub-instance buffer. Set to 0 for the old CPU writes plus full re-upload.
#define RTXPT_SUBINSTANCE_LIGHT_PATCH_PASS 1

// This prepares all scene lighting (including environment map already partially processed by EnvMapBaker) for sampling in path tracing.
// Supported sampling approaches are Uniform, Power and NEE-AT. All NEE-AT baking logic is included here.
class LightsBaker 
{
public:
    struct BakeSettings
    {
        uint        ImportanceSamplingType      = 0;                    // 0 - uniform; 1 - pure power based; 2 - NEE-AT
        float3      CameraPosition              = float3(0,0,0);
        float3      CameraDirection             = float3(0,0,0);
        float       AverageContentsDistance     = 10.0f;                // rough average distance from camera that most viewed objects will be at - 1-100m is good for FPS, could be 1000 for a flight sim
        uint2       MouseCursorPos              = uint2(0,0);           // only used for debug viz
        float4x4    ViewProjMatrix              = float4x4::identity(); // needed for figuring out frustum planes for optimizations
        
        // NEE-AT settings
        float       GlobalTemporalFeedbackWeight    = 0.75f;    // 0.0 - use no feedback for global sampler, 0.95 use almost feedback only (some power-based input always needed to bring in new lights)
        float       LocalToGlobalSampleRatio        = 0.65f;    // 0.0 - sample exclusively from global sampler, 1.0 sample (almost) exclusively from local sampler
        // float       LightSampling_MIS_Boost         = 1.0f;     // boost light sampling when doing MIS vs BSDF <- TODO: redesign, make it work with 'UseApproximateMIS'
        bool        UseApproximateMIS           = false;

        // frame/global settings
        bool        ResetFeedback = false;
        float2      ViewportSize                    = {0,0};
        float2      PrevViewportSize                = {0,0};

        // environment map parameters
        NEEAT::LightsBakerEnvMapParams EnvMapParams        = {};
        float DistantVsLocalImportanceScale         = 1.0f;

        int64_t     FrameIndex                      = -1;
    };

public:
    LightsBaker(nvrhi::IDevice* device);
    ~LightsBaker();
    LightsBaker(const LightsBaker&) = delete;               // m_passes points into this object
    LightsBaker& operator=(const LightsBaker&) = delete;

    // Reset scene related stuff
    void                            SceneReloaded();

    void                            CreateRenderPasses(const std::shared_ptr<ComputePipelineBaker> & computeBaker, nvrhi::IBindingLayout* bindlessLayout, std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses, std::shared_ptr<ShaderDebug> shaderDebug, const uint2 renderResolution, const uint envMapProcessedResolution, const int qualityPreset);

    // Main and only processing stage is split into UpdateBegin/UpdateEnd. These can be called one after the other as soon as screen space motion vectors are available.
    // The split is purely to facilitate any potential async compute.
    
    // UpdateBegin can happen in parallel with any other ray preparatory tracing work - anything from BVH building to laying down denoising layers. Emissive triangle emission must be accessible at this point.
    void                            UpdateBegin(nvrhi::ICommandList* commandList, donut::engine::BindingCache& bindingCache, const BakeSettings& _settings, double sceneTime, const std::shared_ptr<ExtendedScene>& scene, std::shared_ptr<class MaterialsBaker> materialsBaker, std::shared_ptr<class OmmBaker> ommBaker, nvrhi::BufferHandle subInstanceDataBuffer, std::vector<SubInstanceData>& subInstanceData, nvrhi::TextureHandle envMapProcessed, SceneRenderCache & sceneRenderCache);
    // UpdateEnd must happen BEFORE any light sampling (e.g. PT pass with NEE) but AFTER screen space motion vectors are available for reprojection.
    void                            UpdateEnd(nvrhi::ICommandList * commandList, donut::engine::BindingCache & bindingCache, const std::shared_ptr<ExtendedScene> & scene, std::shared_ptr<class MaterialsBaker> materialsBaker, std::shared_ptr<class OmmBaker> ommBaker, nvrhi::BufferHandle subInstanceDataBuffer, nvrhi::TextureHandle depthBuffer, nvrhi::TextureHandle motionVectors);

    // Applies the patch list ProcessEmissiveGeometry built, in place of the caller's full sub-instance re-upload.
    // Must run after that upload (which resets both fields to invalid) and before anything samples the fields.
    void                            ApplySubInstanceLightPatches(nvrhi::ICommandList* commandList, donut::engine::BindingCache & bindingCache, nvrhi::BufferHandle subInstanceDataBuffer);

    // NEEAT_QUALITY_PRESET applied by the last CreateRenderPasses
    int                             GetQualityPreset() const                    { return m_qualityPreset; }

    nvrhi::BufferHandle             GetControlBuffer() const                    { return m_controlBuffer; }
    nvrhi::BufferHandle             GetLightBuffer() const                      { return m_lightsBuffer; }              // this is the list of lights
    nvrhi::BufferHandle             GetLightExBuffer() const                    { return m_lightsExBuffer; }            // this is the list of light (extended data)
    nvrhi::BufferHandle             GetLightProxyCounters() const               { return m_perLightProxyCounters; }     // these are counters of how many proxies each light has
    nvrhi::BufferHandle             GetLightSamplingProxies() const             { return m_lightSamplingProxies; }      // these are indices into the GetLightBuffer()
    nvrhi::BufferHandle             GetLightWeightsBoost() const                { return m_lightWeightsBoost; }         // per-light importance boost multiplier (>=1) applied to the current frame's weights
    uint                            GetAnalyticLightOffset() const              { return m_currentCtrlBuff.EnvmapQuadNodeCount; }
    uint                            GetAnalyticLightCount() const               { return m_currentCtrlBuff.AnalyticLightCount; }
    uint                            GetTriangleLightOffset() const              { return m_currentCtrlBuff.EnvmapQuadNodeCount + m_currentCtrlBuff.AnalyticLightCount; }
    uint                            GetLightClusterOffset() const               { return GetTriangleLightOffset() + m_currentCtrlBuff.TriangleLightCount; }

    nvrhi::TextureHandle            GetEnvLightLookupMap() const                { return m_envLightLookupMap; }

    nvrhi::BufferHandle             GetLocalSamplingBuffer() const              { return m_NEE_AT_LocalSamplingBuffer; }

    nvrhi::TextureHandle            GetFeedbackTotalWeight() const              { return m_NEE_AT_FeedbackTotalWeight; }
    nvrhi::TextureHandle            GetFeedbackCandidates() const               { return m_NEE_AT_FeedbackCandidates; }


    bool                            InfoGUI(float indent);
    bool                            DebugGUI(float indent);

    void                            SetGlobalShaderMacros(std::vector<donut::engine::ShaderMacro> & macros);

    bool                            TotalLightCountOverflow() const;

private:

    // output goes into m_scratchLightBuffer and 
    static bool                     CollectEnvmapLightPlaceholders(const BakeSettings & settings, NEEAT::ControlData & ctrlBuff, std::vector<PolymorphicLightInfo> & outLightBuffer, std::vector<PolymorphicLightInfoEx> & outLightExBuffer, std::vector<uint> & outLightHistoryRemapCurrentToPastBuffer, std::vector<uint> & outLightHistoryRemapPastToCurrent);
    bool                            CollectAnalyticLightsCPU(const BakeSettings & settings, const std::shared_ptr<ExtendedScene> & scene, NEEAT::ControlData & ctrlBuff, std::vector<PolymorphicLightInfo> & outLightBuffer, std::vector<PolymorphicLightInfoEx> & outLightExBuffer, std::vector<uint> & outLightHistoryRemapCurrentToPast, std::vector<uint> & outLightHistoryRemapPastToCurrent);

    // this creates emissive triangle proc tasks and also does any required geometry instance (subInstance) processing such as analyt light proxies; has to happen AFTER CollectAnalyticLightsCPU
    bool                            ProcessEmissiveGeometry( const BakeSettings & settings, const std::shared_ptr<ExtendedScene> & scene, std::vector<SubInstanceData> & subInstanceData, NEEAT::ControlData & ctrlBuff, std::vector<NEEAT::EmissiveTrianglesProcTask> & tasks, SceneRenderCache & sceneRenderCache );

    void                            FillBindings(nvrhi::BindingSetDesc& outBindingSetDesc, const std::shared_ptr<ExtendedScene> & scene, std::shared_ptr<class MaterialsBaker> materialsBaker, std::shared_ptr<class OmmBaker> ommBaker, nvrhi::BufferHandle subInstanceDataBuffer, nvrhi::TextureHandle depthBuffer, nvrhi::TextureHandle motionVectors, nvrhi::TextureHandle envMapProcessed);

    void                            UpdateFrustumConsts(NEEAT::LightsBakerConstants & outConsts, const LightsBaker::BakeSettings & settings);

    void                            UpdateLocalJitter();

    bool                            AllPassesReady() const;
    void                            ReleasePasses();

private:
    nvrhi::DeviceHandle             m_device;
    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;
    std::shared_ptr<donut::engine::FramebufferFactory> m_framebufferFactory;
    std::shared_ptr<ShaderDebug>    m_shaderDebug;

    std::shared_ptr<ComputeShaderVariant> m_resetPastToCurrentHistory;

    std::shared_ptr<ComputeShaderVariant> m_envLightsBackupPast;
    std::shared_ptr<ComputeShaderVariant> m_envLightsSubdivideBase;
    std::shared_ptr<ComputeShaderVariant> m_envLightsSubdivideBoost;
    std::shared_ptr<ComputeShaderVariant> m_envLightsFillLookupMap;
    std::shared_ptr<ComputeShaderVariant> m_envLightsMapPastToCurrent;

    std::shared_ptr<ComputeShaderVariant> m_bakeEmissiveTriangles;
    std::shared_ptr<ComputeShaderVariant> m_patchSubInstanceLightData;

    std::shared_ptr<ComputeShaderVariant> m_clearFeedbackHistory;
    std::shared_ptr<ComputeShaderVariant> m_processFeedbackHistoryP0;
    std::shared_ptr<ComputeShaderVariant> m_processFeedbackHistoryP1a;
    std::shared_ptr<ComputeShaderVariant> m_processFeedbackHistoryP1b;
    std::shared_ptr<ComputeShaderVariant> m_processFeedbackHistoryP2;
    std::shared_ptr<ComputeShaderVariant> m_processFeedbackHistoryP3;
    std::shared_ptr<ComputeShaderVariant> m_processFeedbackHistoryDebugViz;

    std::shared_ptr<ComputeShaderVariant> m_processFeedbackHistoryPreFilterA;
    std::shared_ptr<ComputeShaderVariant> m_processFeedbackHistoryPreFilterB;

    std::shared_ptr<ComputeShaderVariant> m_resetLightProxyCounters;
    std::shared_ptr<ComputeShaderVariant> m_computeWeights;
    std::shared_ptr<ComputeShaderVariant> m_processClusters;
    std::shared_ptr<ComputeShaderVariant> m_computeProxyCounts;
    std::shared_ptr<ComputeShaderVariant> m_computeProxyBaselineOffsets;
    std::shared_ptr<ComputeShaderVariant> m_createProxyJobs;
    std::shared_ptr<ComputeShaderVariant> m_executeProxyJobs;
    std::shared_ptr<ComputeShaderVariant> m_debugDrawLights;

    std::vector<std::shared_ptr<ComputeShaderVariant> *> m_passes;  // all of the above, for AllPassesReady and ReleasePasses
    std::weak_ptr<ComputePipelineBaker> m_computeBaker;             // the passes were created on this baker...
    nvrhi::BindingLayoutHandle      m_bindlessLayout;               // ...with this bindless layout
    bool                            m_bakeSkipped = false;          // set by UpdateBegin when a pass isn't ready, so UpdateEnd skips too
    bool                            m_controlBufferCleared = false; // m_controlBuffer holds the zeroes a skipped bake wrote
    int                             m_qualityPreset = 0;            // NEEAT_QUALITY_PRESET the buffers are laid out for, and the passes and path tracer compile with
    int                             m_passesQualityPreset = -1;     // m_qualityPreset the passes were created with

    nvrhi::BindingLayoutHandle      m_commonBindingLayout;

    // Separate from m_commonBindingLayout on purpose: that one binds the sub-instance buffer as an SRV, and the same
    // resource cannot be SRV and UAV in one binding set.
    nvrhi::BindingLayoutHandle      m_subInstanceLightPatchBindingLayout;
    nvrhi::BufferHandle             m_subInstanceLightPatchBuffer;      // grow-only; element count: m_subInstanceLightPatchCapacity
    uint32_t                        m_subInstanceLightPatchCapacity = 0;

    BakeSettings                    m_currentSettings;
    NEEAT::ControlData              m_currentCtrlBuff;              // NOTE: this does not include GPU-side changes, only the initial state set in Update

    nvrhi::BufferHandle             m_controlBuffer;

    nvrhi::SamplerHandle            m_pointSampler;
    nvrhi::SamplerHandle            m_linearSampler;

    // nvrhi::BufferHandle             m_lightingConstants;                // same content as in control buffer

    nvrhi::BufferHandle             m_lightsBuffer;                     // element count: NEEAT_MAX_LIGHTS
    nvrhi::BufferHandle             m_lightsExBuffer;                   // element count: NEEAT_MAX_LIGHTS
    nvrhi::BufferHandle             m_scratchBuffer;                    // byte size: LLB_SCRATCH_BUFFER_SIZE
    nvrhi::BufferHandle             m_scratchList;                      // element count: NEEAT_MAX_LIGHTS
    nvrhi::BufferHandle             m_historyRemapCurrentToPastBuffer;  // element count: NEEAT_MAX_LIGHTS
    nvrhi::BufferHandle             m_historyRemapPastToCurrentBuffer;  // element count: NEEAT_MAX_LIGHTS

    nvrhi::BufferHandle             m_controlBufferReadback;        // for showing debug info
    int                             m_framesFromLastReadbackCopy;   // the number of frames that passed since 
    NEEAT::ControlData              m_lastReadback;

    nvrhi::BufferHandle             m_lightWeights;                 // element count: 2 * RTXPT_LIGHTING_WEIGHTS_COUNT_HALF
    nvrhi::BufferHandle             m_lightWeightsBoost;            // element count: 1 * RTXPT_LIGHTING_WEIGHTS_COUNT_HALF
    nvrhi::BufferHandle             m_perLightProxyCounters;        // element count: NEEAT_MAX_LIGHTS
    nvrhi::BufferHandle             m_lightSamplingProxies;         // element count: RTXPT_LIGHTING_MAX_SAMPLING_PROXIES  <- this is the output of the GPUSort and is only used to sort the above 2 arrays

    nvrhi::TextureHandle            m_NEE_AT_FeedbackTotalWeight;
    nvrhi::TextureHandle            m_NEE_AT_FeedbackCandidates;
    nvrhi::TextureHandle            m_NEE_AT_FeedbackTotalWeightScratch;
    nvrhi::TextureHandle            m_NEE_AT_FeedbackCandidatesScratch;
    bool                            m_NEE_AT_FeedbackBufferFilled;

    nvrhi::TextureHandle            m_NEE_AT_FeedbackTotalWeightBlended;
    nvrhi::TextureHandle            m_NEE_AT_FeedbackCandidatesBlended;
    nvrhi::BufferHandle             m_NEE_AT_LocalSamplingBuffer;

    nvrhi::TextureHandle            m_envLightLookupMap;            // used for looking up environment lights by direction for full MIS

    nvrhi::TextureHandle            m_NEE_AT_HistoryDepth;

    std::vector<PolymorphicLightInfo>   m_scratchLightBuffer;                           // these are for scene lights filled in on CPU side
    std::vector<PolymorphicLightInfoEx> m_scratchLightExBuffer;                         // these are for scene lights filled in on CPU side
    std::vector<uint>                   m_scratchLightHistoryRemapCurrentToPastBuffer;  // these are for scene lights filled in on CPU side
    std::vector<uint>                   m_scratchLightHistoryRemapPastToCurrentBuffer;  // these are for scene lights filled in on CPU side
    std::shared_ptr<std::vector<struct NEEAT::EmissiveTrianglesProcTask>>  m_scratchTaskBuffer;
    std::vector<SubInstanceLightPatch>  m_subInstanceLightPatches;                  // rebuilt by ProcessEmissiveGeometry every frame
    uint                                m_historicTotalLightCount;
    uint                                m_historicLightClusterBase;     // previous frame's light index of the first light cluster

    // Historic light-source matching by hash - the fallback used when engine light/instance objects cannot carry a
    // LightSamplerLink (see HASH_LOOKUP_BASED_HISTORIC_LIGHT_SOURCE_MATCHING above).
    //
    // Double-buffered: lookups read the PREVIOUS frame's map, writes go to the CURRENT one, and the two roles flip at
    // the end of UpdateBegin with the new 'current' cleared. Absence from the previous map then means exactly "was not
    // present last frame", which makes this self-cleaning by construction - no erase on transition, no staleness tag,
    // and no need to visit geometry that stopped being emissive just to clean up after it. It also bounds each map to a
    // single frame's live set instead of the union of everything ever seen, which keeps hash collisions down.
#ifdef HASH_LOOKUP_BASED_HISTORIC_LIGHT_SOURCE_MATCHING
    std::unordered_map<size_t, uint32_t> m_historyRemapEmissiveLightBlockOffsets[2];
    std::unordered_map<size_t, uint32_t> m_historyRemapAnalyticLightIndices[2];
    uint32_t                             m_historyRemapWriteIndex = 0;

    std::unordered_map<size_t, uint32_t> & HistoryEmissivePrev() { return m_historyRemapEmissiveLightBlockOffsets[1 - m_historyRemapWriteIndex]; }
    std::unordered_map<size_t, uint32_t> & HistoryEmissiveCurr() { return m_historyRemapEmissiveLightBlockOffsets[m_historyRemapWriteIndex]; }
    std::unordered_map<size_t, uint32_t> & HistoryAnalyticPrev() { return m_historyRemapAnalyticLightIndices[1 - m_historyRemapWriteIndex]; }
    std::unordered_map<size_t, uint32_t> & HistoryAnalyticCurr() { return m_historyRemapAnalyticLightIndices[m_historyRemapWriteIndex]; }
    void                                 FlipHistoryRemapBuffers();
#endif

    float2                          m_localJitterF                      = {0, 0};
    uint2                           m_localJitter                       = {0, 0};
    uint2                           m_prevLocalJitter                   = {0, 0};
    uint                            m_updateCounter                     = 0;

    uint2                           m_renderResolution                  = {0, 0};   // logical feedback/render resolution; do not infer this from feedback texture dimensions - they may be allocated larger (see prefilter ping-pong)
    int                             m_localSamplingBufferWidth          = 0;
    int                             m_localSamplingBufferHeight         = 0;
    int                             m_localSamplingBufferDepth          = 0;
    uint                            m_localSamplingTileSize             = 0;

    // various buffers are ping-ponged where current and history swap places; this bool is inverted at every Update()
    bool                            m_ping                              = false;

    bool                            m_dbgDebugDrawLights                = false;
    bool                            m_dbgDebugDrawTileLightConnections  = false;
    bool                            m_dbgFreezeFeedback                 = false;
    
    NEEAT::DebugViewType            m_dbgDebugDrawType                   = NEEAT::DebugViewType::Disabled;
    bool                            m_dbgDebugDisableJitter             = false;
    bool                            m_dbgDebugDisableLastFrameFeedback  = false;

    float                           m_advSetting_ScreenSpaceVsWorldSpaceThreshold = 0.3f;
    bool                            m_advSetting_SampleBakedEnvironment = true;
    bool                            m_advSetting_BSDFFeedbackEnabled    = true;     // RTXPT_LIGHTING_ENABLE_BSDF_FEEDBACK

    bool                            m_deviceHas32ThreadWaves            = false;

    bool                            m_importanceBoost_IntensityDelta        = true;
    float                           m_importanceBoost_IntensityDeltaMul     = 12.0f;    // seems like a good balance but it's just a rough guess; too much and it can have unintended consequences
    bool                            m_importanceBoost_Frustum               = true;
    float                           m_importanceBoost_FrustumMul            = 8.0f;
    float                           m_importanceBoost_FrustumFadeDistance   = 5.0f;		// in scene units
    float                           m_importanceBoost_FrustumAngleExpand    = 5.0f;     // in degrees - TODO: not implemented
    bool                            m_importanceBoost_PreFilter             = true;

    float                           m_advSetting_reservoirHistoryDropoff    = 0.005f;

    bool                            m_advSetting_LightClustersEnabled       = true;
    float                           m_advSetting_LightClusterLODExponent    = 2.0f;
    float                           m_advSetting_LightClusterLODMultiplier  = 3000.0f;   // with exponent 2, a cluster turns active beyond ~32x its radius (sqrt(1000))
    float                           m_advSetting_LightClusterBrightnessExponent = 0.5f;  // 0 disables; see IsLightClusterActive

    float                           m_depthDisocclusionThreshold            = 1.5f;

    int64_t                         m_lastFrameIndex                        = -1;   // updated in UpdateFrame, no longer valid after
    
    bool                            m_dbgFreezeFrustumUpdates           = false;
    float4                          m_dbgFrozenFrustum[6];

    uint64_t                        m_allocatedVRAM                     = 0;

    bool                            m_noOverflow                        = true;

    int                             m_verifyBeginHasMatchingEnd         = 0;
};
