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

#include <donut/engine/BindingCache.h>
#include <nvrhi/nvrhi.h>
#include <donut/core/math/math.h>
#include <memory>

#include <donut/core/math/math.h>

#include "../Shaders/PathTracer/Lighting/LightingTypes.h"

#include "../ComputePass.h"

#include "../Shaders/SubInstanceData.h"

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
}

class ShaderDebug;
class EnvMapBaker;
class ExtendedScene;

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
        bool        GlobalTemporalFeedbackEnabled       = false;    // <- remove, just use
        float       GlobalTemporalFeedbackRatio         = 0.75f;    // 0.0 - use no feedback, 0.95 use almost feedback only (some power-based input always needed to bring in new lights)
        bool        LocalTemporalFeedbackEnabled       = false;    // <- remove, just use
        float       LocalTemporalFeedbackRatio         = 0.65f;     // 0.0 - use no feedback, 1.0 use feedback only
        float       LightSampling_MIS_Boost             = 1.0f;     // boost light sampling when doing MIS vs BSDF

        // frame/global settings
        bool        ResetFeedback = false;
        float2      ViewportSize                        = {0,0};
        float2      PrevViewportSize                    = {0,0};

        // if GetFeedbackTotalWeightAntiLagPass/GetFeedbackCandidatesAntiLagPass was filled in a pre-pass, use it in the first UpdatePreRender of the frame to feed potential light pool with up to date lights
        bool        EnableAntiLag                   = false;

        // environment map parameters
        EnvMapSceneParams EnvMapParams              = {};
        float DistantVsLocalImportanceScale         = 1.0f;
    };

public:
    LightsBaker(nvrhi::IDevice* device, std::shared_ptr<donut::engine::TextureCache> textureCache, std::shared_ptr<donut::engine::ShaderFactory> shaderFactory, std::shared_ptr<EnvMapBaker> envMapBaker);
    ~LightsBaker();

    // Reset scene related stuff
    void                            SceneReloaded();

    void                            CreateRenderPasses(nvrhi::IBindingLayout* bindlessLayout, std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses, std::shared_ptr<ShaderDebug> shaderDebug, const uint2 renderResolution);

    // this update can happen in parallel with any other ray preparatory tracing work - anything from BVH building to laying down denoising layers
    void                            UpdateFrame(nvrhi::ICommandList * commandList, const BakeSettings & settings, double sceneTime, const std::shared_ptr<ExtendedScene> & scene, std::shared_ptr<class MaterialsBaker> materialsBaker, std::shared_ptr<class OmmBaker> ommBaker, nvrhi::BufferHandle subInstanceDataBuffer, std::vector<SubInstanceData> & subInstanceData);
    // this update must happen before main path tracing (that uses NEE) but ideally after motion vectors are available for reprojection
    void                            UpdatePreRender(nvrhi::ICommandList * commandList, const std::shared_ptr<ExtendedScene> & scene, std::shared_ptr<class MaterialsBaker> materialsBaker, std::shared_ptr<class OmmBaker> ommBaker, nvrhi::BufferHandle subInstanceDataBuffer, nvrhi::TextureHandle depthBuffer, nvrhi::TextureHandle motionVectors);

    // only valid after UpdateFrame()!
    bool                            IsAntiLagActive() const                     { return m_currentSettings.EnableAntiLag; }

    nvrhi::BufferHandle             GetControlBuffer() const                    { return m_controlBuffer; }
    nvrhi::BufferHandle             GetLightBuffer() const                      { return m_lightsBuffer; }              // this is the list of lights
    nvrhi::BufferHandle             GetLightExBuffer() const                    { return m_lightsExBuffer; }            // this is the list of light (extended data)
    nvrhi::BufferHandle             GetLightProxyCounters() const               { return m_perLightProxyCounters; }     // these are counters of how many proxies each light has
    nvrhi::BufferHandle             GetLightSamplingProxies() const             { return m_lightSamplingProxies; }      // these are indices into the GetLightBuffer()

    nvrhi::TextureHandle            GetEnvLightLookupMap() const                { return m_envLightLookupMap; }

#if RTXPT_LIGHTING_LOCAL_SAMPLING_BUFFER_IS_3D_TEXTURE
    nvrhi::TextureHandle            GetLocalSamplingBuffer() const              { return m_NEE_AT_LocalSamplingBuffer; }
#else
    nvrhi::BufferHandle             GetLocalSamplingBuffer() const              { return m_NEE_AT_LocalSamplingBuffer; }
#endif

    nvrhi::TextureHandle            GetFeedbackTotalWeight() const              { return m_NEE_AT_FeedbackTotalWeight; }
    nvrhi::TextureHandle            GetFeedbackCandidates() const               { return m_NEE_AT_FeedbackCandidates; }
    nvrhi::TextureHandle            GetFeedbackTotalWeightAntiLagPass() const   { return m_NEE_AT_FeedbackTotalWeightScratch; }
    nvrhi::TextureHandle            GetFeedbackCandidatesAntiLagPass() const    { return m_NEE_AT_FeedbackCandidatesScratch; }


    bool                            InfoGUI(float indent);
    bool                            DebugGUI(float indent);

    void                            SetGlobalShaderMacros(std::vector<donut::engine::ShaderMacro> & macros);

private:

    // output goes into m_scratchLightBuffer and 
    static void                     CollectEnvmapLightPlaceholders(const BakeSettings & settings, LightingControlData & ctrlBuff, std::vector<PolymorphicLightInfo> & outLightBuffer, std::vector<PolymorphicLightInfoEx> & outLightExBuffer, std::vector<uint> & outLightHistoryRemapCurrentToPastBuffer, std::vector<uint> & outLightHistoryRemapPastToCurrent);
    static void                     CollectAnalyticLightsCPU(const BakeSettings & settings, const std::shared_ptr<ExtendedScene> & scene, LightingControlData & ctrlBuff, std::vector<PolymorphicLightInfo> & outLightBuffer, std::vector<PolymorphicLightInfoEx> & outLightExBuffer, std::unordered_map<size_t, uint32_t> & inoutHistoryRemapAnalyticLightIndices, std::vector<uint> & outLightHistoryRemapCurrentToPast, std::vector<uint> & outLightHistoryRemapPastToCurrent);

    // this creates emissive triangle proc tasks and also does any required geometry instance (subInstance) processing such as analyt light proxies; has to happen AFTER CollectAnalyticLightsCPU
    uint                            ProcessGeometry( const BakeSettings & settings, const std::shared_ptr<ExtendedScene> & scene, std::vector<SubInstanceData> & subInstanceData, LightingControlData & ctrlBuff, std::vector<struct EmissiveTrianglesProcTask> & tasks, std::unordered_map<size_t, uint32_t> & inoutHistoryRemapAnalyticLightIndices );

    void                            FillBindings(nvrhi::BindingSetDesc& outBindingSetDesc, const std::shared_ptr<ExtendedScene> & scene, std::shared_ptr<class MaterialsBaker> materialsBaker, std::shared_ptr<class OmmBaker> ommBaker, nvrhi::BufferHandle subInstanceDataBuffer, nvrhi::TextureHandle depthBuffer, nvrhi::TextureHandle motionVectors);

    void                            UpdateFrustumConsts(LightsBakerConstants & outConsts, const LightsBaker::BakeSettings & settings);

    void                            UpdateLocalJitter();

private:
    nvrhi::DeviceHandle             m_device;
    std::shared_ptr<donut::engine::TextureCache> m_textureCache;
    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;
    std::shared_ptr<donut::engine::FramebufferFactory> m_framebufferFactory;
    std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;
    std::shared_ptr<ShaderDebug>    m_shaderDebug;

    std::shared_ptr<EnvMapBaker>    m_envMapBaker;

    ComputePass                     m_resetPastToCurrentHistory;

    ComputePass                     m_envLightsBackupPast;
    ComputePass                     m_envLightsSubdivideBase;
    ComputePass                     m_envLightsSubdivideBoost;
    ComputePass                     m_envLightsFillLookupMap;
    ComputePass                     m_envLightsMapPastToCurrent;

    ComputePass                     m_bakeEmissiveTriangles;

    ComputePass                     m_clearFeedbackHistory;
    ComputePass                     m_clearAntiLagFeedback;
    ComputePass                     m_processFeedbackHistoryP0;
    ComputePass                     m_processFeedbackHistoryP1a;
    ComputePass                     m_processFeedbackHistoryP1b;
    ComputePass                     m_processFeedbackHistoryP2;
    ComputePass                     m_processFeedbackHistoryP3;
    ComputePass                     m_processFeedbackHistoryDebugViz;
    ComputePass                     m_updateControlBufferMultipass;

    ComputePass                     m_resetLightProxyCounters;
    ComputePass                     m_computeWeights;
    ComputePass                     m_computeProxyCounts;
    ComputePass                     m_computeProxyBaselineOffsets;
    ComputePass                     m_createProxyJobs;
    ComputePass                     m_executeProxyJobs;
    ComputePass                     m_debugDrawLights;
    
    nvrhi::BindingLayoutHandle      m_commonBindingLayout;
    nvrhi::BindingLayoutHandle      m_bindlessLayout;

    donut::engine::BindingCache     m_bindingCache;

    BakeSettings                    m_currentSettings;
    LightingControlData             m_currentCtrlBuff;              // NOTE: this does not include GPU-side changes, only the initial state set in Update
    LightsBakerConstants            m_currentConsts;

    nvrhi::BufferHandle             m_constantBuffer;
    nvrhi::BufferHandle             m_controlBuffer;

    nvrhi::SamplerHandle            m_pointSampler;
    nvrhi::SamplerHandle            m_linearSampler;

    // nvrhi::BufferHandle             m_lightingConstants;                // same content as in control buffer

    nvrhi::BufferHandle             m_lightsBuffer;                     // element count: RTXPT_LIGHTING_MAX_LIGHTS
    nvrhi::BufferHandle             m_lightsExBuffer;                   // element count: RTXPT_LIGHTING_MAX_LIGHTS
    nvrhi::BufferHandle             m_scratchBuffer;                    // byte size: LLB_SCRATCH_BUFFER_SIZE
    nvrhi::BufferHandle             m_scratchList;                      // element count: RTXPT_LIGHTING_MAX_LIGHTS
    nvrhi::BufferHandle             m_historyRemapCurrentToPastBuffer;  // element count: RTXPT_LIGHTING_MAX_LIGHTS
    nvrhi::BufferHandle             m_historyRemapPastToCurrentBuffer;  // element count: RTXPT_LIGHTING_MAX_LIGHTS

    nvrhi::BufferHandle             m_controlBufferReadback;        // for showing debug info
    int                             m_framesFromLastReadbackCopy;   // the number of frames that passed since 
    LightingControlData             m_lastReadback;

    nvrhi::BufferHandle             m_lightWeightsPing;             // element count: RTXPT_LIGHTING_MAX_LIGHTS
    nvrhi::BufferHandle             m_lightWeightsPong;             // element count: RTXPT_LIGHTING_MAX_LIGHTS
    nvrhi::BufferHandle             m_perLightProxyCounters;        // element count: RTXPT_LIGHTING_MAX_LIGHTS
    nvrhi::BufferHandle             m_lightSamplingProxies;         // element count: RTXPT_LIGHTING_MAX_SAMPLING_PROXIES  <- this is the output of the GPUSort and is only used to sort the above 2 arrays

    nvrhi::TextureHandle            m_NEE_AT_FeedbackTotalWeight;
    nvrhi::TextureHandle            m_NEE_AT_FeedbackCandidates;
    nvrhi::TextureHandle            m_NEE_AT_FeedbackTotalWeightScratch;
    nvrhi::TextureHandle            m_NEE_AT_FeedbackCandidatesScratch;
    //nvrhi::TextureHandle            m_NEE_AT_ProcessedFeedbackBuffer;
    //nvrhi::TextureHandle            m_NEE_AT_ReprojectedFeedbackBuffer;
    bool                            m_NEE_AT_FeedbackBufferFilled;

    nvrhi::TextureHandle            m_NEE_AT_FeedbackTotalWeightBlended;
    nvrhi::TextureHandle            m_NEE_AT_FeedbackCandidatesBlended;

#if RTXPT_LIGHTING_LOCAL_SAMPLING_BUFFER_IS_3D_TEXTURE
    nvrhi::TextureHandle            m_NEE_AT_LocalSamplingBuffer;
#else
    nvrhi::BufferHandle             m_NEE_AT_LocalSamplingBuffer;
#endif

    nvrhi::TextureHandle            m_envLightLookupMap;            // looking up environment lights by direction

    nvrhi::TextureHandle            m_NEE_AT_HistoryDepth;

    std::vector<PolymorphicLightInfo>   m_scratchLightBuffer;                           // these are for scene lights filled in on CPU side
    std::vector<PolymorphicLightInfoEx> m_scratchLightExBuffer;                         // these are for scene lights filled in on CPU side
    std::vector<uint>                   m_scratchLightHistoryRemapCurrentToPastBuffer;  // these are for scene lights filled in on CPU side
    std::vector<uint>                   m_scratchLightHistoryRemapPastToCurrentBuffer;  // these are for scene lights filled in on CPU side
    std::shared_ptr<std::vector<struct EmissiveTrianglesProcTask>>  m_scratchTaskBuffer;
    uint                                m_historicTotalLightCount;

    // NOTE: there's no mechanism to erase stale historic indices; it would be ideal to double-buffer both of these and each frame populate the new one afresh, while clearing the old one; that way we would never have leftover historic entries and reduce chance of hash collisions
    std::unordered_map<size_t, uint32_t> m_historyRemapEmissiveLightBlockOffsets;
    std::unordered_map<size_t, uint32_t> m_historyRemapAnalyticLightIndices;

    float2                          m_localJitterF                      = {0, 0};
    uint2                           m_localJitter                       = {0, 0};
    uint2                           m_prevLocalJitter                   = {0, 0};
    uint                            m_updateCounter                     = 0;
    bool                            m_updateFrameCalledBeforePreRender  = false;

    int                             m_localSamplingBufferWidth          = 0;
    int                             m_localSamplingBufferHeight         = 0;
    const int                       m_localSamplingBufferDepth          = RTXPT_LIGHTING_LOCAL_PROXY_COUNT;

    // various buffers are ping-ponged where current and history swap places; this bool is inverted at every Update()
    bool                            m_ping                              = false;

    bool                            m_dbgDebugDrawLights                = false;
    bool                            m_dbgDebugDrawTileLightConnections  = false;
    bool                            m_dbgFreezeUpdates                  = false;
    
    LightingDebugViewType           m_dbgDebugDrawType                  = LightingDebugViewType::Disabled;
    //bool                            m_dbgDebugDrawDirect                = true;
    bool                            m_dbgDebugDisableJitter             = false;

    float                           m_advSetting_DirectVsIndirectThreshold = 0.3f;
    bool                            m_advSetting_SampleBakedEnvironment = true;

    bool                            m_deviceHas32ThreadWaves            = false;

    bool                            m_importanceBoost_IntensityDelta        = true;
    float                           m_importanceBoost_IntensityDeltaMul     = 30.0f;
    bool                            m_importanceBoost_Frustum               = true;
    float                           m_importanceBoost_FrustumMul            = 8.0f;
    float                           m_importanceBoost_FrustumFadeRangeInt   = 20.0f;
    float                           m_importanceBoost_FrustumFadeRangeExt   = 5.0f;

    float                           m_advSetting_reservoirHistoryDropoff    = 0.04f;

    float                           m_depthDisocclusionThreshold            = 1.5f;
    
    bool                            m_dbgFreezeFrustumUpdates           = false;
    float4                          m_dbgFrozenFrustum[6];
};
