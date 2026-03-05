/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __LIGHTING_TYPES_HLSLI__
#define __LIGHTING_TYPES_HLSLI__

#include "LightingConfig.h"

#if defined(__cplusplus)
using namespace donut::math;
#define ROW_MAJOR
#else
#define ROW_MAJOR row_major
#endif

struct LightsBakerEnvMapParams
{
    ROW_MAJOR float3x4  Transform;          ///< Local to world transform.
    ROW_MAJOR float3x4  InvTransform;       ///< World to local transform.
    float3              ColorMultiplier;    ///< Color & radiance scale (Tint * Intensity)
    float               Enabled;            ///< 1 if enabled, 0 if not
};

#define NEEAT_LIGHTS_BAKER_CONSTANTS_SIZE 464

struct LightsBakerConstants
{
    float                   DistantVsLocalRelativeImportance;
    uint                    EnvMapImportanceMapMIPCount;
    uint                    EnvMapImportanceMapResolution;
    uint                    TriangleLightTaskCount;

    uint2                   FeedbackResolution;
    uint2                   BlendedFeedbackResolution;

    uint2                   MouseCursorPos; // for debugging viz only
    float2                  PrevOverCurrentViewportSize; ///< viewPrev.viewportSize / view.viewportSize

    int                     DebugDrawType;
    uint                    DebugDrawTileLights;
    uint                    UpdateCounter; ///< LightBaker's own 'frame' counter (gets reset with LightsBaker::BakeSettings::ResetFeedback and gets incremented on every LightSbaker::UpdateFrame(...) and non-first UpdatePreRender after UpdateFrame )
    uint                    DebugDrawFrustum;

    float                   ImportanceBoostIntensityDelta;
    float                   ImportanceBoostFrustumMul;
    float                   ImportanceBoostFrustumFadeDistance;
    float                   _padding3;

    float3                  SceneCameraPos;
    float                   SceneAverageContentsDistance;

    float                   DepthDisocclusionThreshold;
    uint                    EnableMotionReprojection;
    float                   ReservoirHistoryDropoff;
    uint                    _padding0;
    
    uint                    CurrentWeightsBufferOffset;     ///< Used to ping-pong between current and historic weights; it's either 0 or RTXPT_LIGHTING_WEIGHTS_COUNT_HALF
    uint                    HistoricWeightsBufferOffset;    ///< Used to ping-pong between current and historic weights; it's either 0 or RTXPT_LIGHTING_WEIGHTS_COUNT_HALF
    uint                    _padding1;
    uint                    _padding2;


    float4                  FrustumPlanes[6];               ///< Left Right Top Bottom Near Far
    float4                  FrustumCorners[8];              ///< For debugging only

    LightsBakerEnvMapParams EnvMapParams;
};
STATIC_ASSERT( NEEAT_LIGHTS_BAKER_CONSTANTS_SIZE == sizeof(LightsBakerConstants) );

// Used for building and using light list
struct LightingControlData
{
    uint    TotalLightCount;            ///< Current total count of lights in the light buffer (of PolymorphicLightInfo type, max RTXPT_LIGHTING_MAX_LIGHTS)
    uint    EnvmapQuadNodeCount;        ///< Number of environment map sampling lights in the light buffer (useful for debugging)
    uint    AnalyticLightCount;         ///< Number of analytic lights in the light buffer (useful for debugging)
    uint    TriangleLightCount;         ///< Number of emissive triangle lights in the light buffer (useful for debugging)

    uint    SamplingProxyCount;         ///< Number of the sampling proxies (max RTXPT_LIGHTING_MAX_SAMPLING_PROXIES)
    uint    HistoricTotalLightCount;    ///< Previous frame's TotalLightCount (can be 0)
    uint    LastFrameTemporalFeedbackAvailable; ///< We can use last frame's temporal feedback
    uint    LastFrameLocalSamplesAvailable;     ///< We can use last frame's local (tile) lights (effectively same as LastFrameTemporalFeedbackAvailable from previous frame)

    uint    ProxyBuildTaskCount;        ///< Only used for building proxies (in LightsBaker.*)
    uint    WeightsSumUINT;             ///< Used with interlocked float add - this is actually a float (can be read with 'asfloat')
    uint    ImportanceSamplingType;     ///< From LightsBaker::BakeSettings - should match global NEEType
    uint    padding0;                   // float   LightSampling_MIS_Boost;    ///< This is a fixed pdf used for the sampler when doing MIS with main path BSDF

    uint    TemporalFeedbackRequired;   ///< Whether the path tracing NEE needs to provide temporal feedback
    uint    TotalMaxFeedbackCount;      ///< Copy of 'LightsBakerConstants' value, used for debugging
    float   GlobalFeedbackUseWeight;
    float   LocalToGlobalSampleRatio;

    uint    TileBufferHeight;           ///< Feedback and tile buffers for direct and indirect are stacked one on top of the other; this is the height of just one (and offset to get from direct to indirect)
    float   ScreenSpaceVsWorldSpaceThreshold;  ///< Used to determine whether to use direct vs indirect light caching strategy for current surface
    uint2   LocalSamplingResolution;    ///< The resolution of the screen space local sampling buffer (number of tiles x * y)

    uint2   LocalSamplingTileJitter;
    uint2   LocalSamplingTileJitterPrev;

    uint    ValidFeedbackCount;         ///< For debugging only
    uint    padding1;
    uint    padding2;
    uint    padding3;

#if NEEAT_BAKER_ONLY
    LightsBakerConstants BakerConstants;
#else 
    uint    paddingBK[NEEAT_LIGHTS_BAKER_CONSTANTS_SIZE/4];
#endif

#if !defined(__cplusplus)
    float   WeightsSum()            { return asfloat(WeightsSumUINT); }
#else
    float   WeightsSum()            { return *reinterpret_cast<float*>(&WeightsSumUINT); }
#endif
};

// These should go into LightsBaker.hlsli or similar
enum class LightingDebugViewType : int
{
    Disabled,

    Disocclusion,
    NoHistoryFeedback,
    MissingFeedbackSSC,
    MissingFeedbackWSC,
    FeedbackRawSSC,
    FeedbackRawWSC,
    LowResBlendedFeedback,
    FeedbackAfterClear,

    TileHeatmap,

    ValidateCorrectness,

    MaxCount
};

#define RTXPT_INVALID_LIGHT_INDEX                       0xFFFFFFFF

inline uint ComputeCandidateSampleLocalCount(const float localToGlobalRatio, const uint totalCandidateSamples)
{
#ifdef RTXPT_NEE_LOCAL_CANDIDATE_SAMPLE_COUNT
    return RTXPT_NEE_LOCAL_CANDIDATE_SAMPLE_COUNT;
#else
    return (uint)((float)(totalCandidateSamples-1) * localToGlobalRatio + 0.75f);    // always allow at least 1 global but then even out
#endif
}

inline uint ComputeCandidateSampleGlobalCount(const float localToGlobalRatio, const uint totalCandidateSamples)
{
#ifdef RTXPT_NEE_GLOBAL_CANDIDATE_SAMPLE_COUNT
    return RTXPT_NEE_GLOBAL_CANDIDATE_SAMPLE_COUNT;
#else
    return totalCandidateSamples - ComputeCandidateSampleLocalCount(localToGlobalRatio, totalCandidateSamples);
#endif
}

#if !defined(__cplusplus) || defined(__INTELLISENSE__)

// Note:
//  * these are dependent for maximum number of global lights under 0x007FFFFF and counter under or equal to 0x1FF.
//  * sorting algorithms rely on light index being packed in high bits
//  * sorting/coalescing algorithms rely on counter being packed in low bits so ++ operation is legal
uint PackMiniListLightAndCount(uint globalLightIndex, uint counter)                             { return ((globalLightIndex & 0x007FFFFF) << 9) | ((counter-1) & 0x1FF); }
void UnpackMiniListLightAndCount(uint value, out uint globalLightIndex, out uint counter)       { globalLightIndex = value >> 9; counter = (value & 0x1FF)+1; }
uint UnpackMiniListLight(uint value)                                                            { return value >> 9; }
uint UnpackMiniListCount(uint value)                                                            { return (value & 0x1FF)+1; }

inline uint LLSB_ComputeBaseAddress(uint2 tilePos, uint2 localSamplingResolution)
{
    return (tilePos.x + (tilePos.y * localSamplingResolution.x)) * RTXPT_LIGHTING_LOCAL_PROXY_COUNT;
}

// Weighted Reservoir Sampling helper for storing good lights for later reuse. Since our reuse is entirely statistical, we don't actually keep the weights
// https://www.pbr-book.org/4ed/Sampling_Algorithms/Reservoir_Sampling
struct LightFeedbackReservoir
{
    #define LFR_SCREEN_SPACE_COHERENT_FLAG      0x80000000u
    #define LFR_MAX_WEIGHT                      1e12

    uint2                                       PixelPos;
    RWTexture2D<float>                          TextureTotalWeight;
    RWTexture2D<uint>                           TextureCandidates;

    static LightFeedbackReservoir make(uint2 pixelPos, RWTexture2D<float> textureTotalWeight, RWTexture2D<uint> textureCandidates)
    {
        LightFeedbackReservoir ret;
        ret.PixelPos            = pixelPos;
        ret.TextureTotalWeight  = textureTotalWeight;
        ret.TextureCandidates   = textureCandidates;
        return ret;
    }

    void CloneFrom(LightFeedbackReservoir other, float scale)
    {
        float otherWeight = other.GetTotalWeight();
        if (otherWeight > 0)
        {
            SetTotalWeight(other.GetTotalWeight()*scale);
            SetCandidatesRaw(other.GetCandidatesRaw());
        }
        else
            Clear();
    }

    // Clear reservoir to empty - not necessary to set individual slots to 0 but useful for debugging
    void Clear()
    {
        SetTotalWeight(0);
#if 1 // this is useful for debugging, but should be removed in production
        SetCandidate(RTXPT_INVALID_LIGHT_INDEX, false);
#endif
    }

    bool IsEmpty()
    {
        return GetTotalWeight() == 0;
    }

    float GetTotalWeight()
    {
        return TextureTotalWeight[PixelPos];
    }

    void SetTotalWeight(float totalWeight)
    {
        totalWeight = min( LFR_MAX_WEIGHT, totalWeight );
        TextureTotalWeight[PixelPos] = totalWeight;
    }

    uint GetCandidatesRaw()
    {
        return TextureCandidates[PixelPos];
    }

    void SetCandidatesRaw(uint candidates)
    {
        TextureCandidates[PixelPos] = candidates;
    }

    uint GetCandidateRaw()
    {
        return TextureCandidates[PixelPos];
    }

    void SetCandidateRaw(uint candidateIndex)
    {
        TextureCandidates[PixelPos] = candidateIndex;
    }

    void GetCandidate(out uint candidateIndex, out bool candidateIsScreenSpaceCoherent)
    {
        candidateIndex = RTXPT_INVALID_LIGHT_INDEX;
        candidateIsScreenSpaceCoherent = false;
        if (IsEmpty())
            return;

        candidateIndex = GetCandidateRaw();
        if (candidateIndex != RTXPT_INVALID_LIGHT_INDEX)
        {
            candidateIsScreenSpaceCoherent = (candidateIndex & LFR_SCREEN_SPACE_COHERENT_FLAG) != 0;
            candidateIndex &= ~LFR_SCREEN_SPACE_COHERENT_FLAG;
        }
    }

    void SetCandidate(uint candidateIndex, bool candidateIsScreenSpaceCoherent )
    {
        SetCandidateRaw( candidateIndex | ((candidateIsScreenSpaceCoherent)?(LFR_SCREEN_SPACE_COHERENT_FLAG):(0)) );
    }

    // Add single new light source with weight; it will be added stochastically to as many slots as available
    void Add( const float randomValue, uint candidateIndex, float candidateWeight, bool candidateIsScreenSpaceCoherent )
    {
        candidateWeight = min(LFR_MAX_WEIGHT, candidateWeight);
        // NOTE: caller ensures no race condition possible here
        float totalWeight = GetTotalWeight();
        totalWeight += candidateWeight;
        SetTotalWeight(totalWeight);
        float threshold = saturate(candidateWeight / totalWeight);

        // Bake in indirect flag
        if (candidateIsScreenSpaceCoherent)
            candidateIndex |= LFR_SCREEN_SPACE_COHERENT_FLAG;

        // Stochastically add 
        if (randomValue < threshold)
            SetCandidateRaw(candidateIndex);
    }

    // Merge reservoirs - can be used to merge a 3x3 kernel for ex.
    void Merge( const float randomValue, const LightFeedbackReservoir other, float otherScale = 1.0 )
    {
        float otherTotalWeight = min(LFR_MAX_WEIGHT, other.GetTotalWeight() * otherScale);
        if( otherTotalWeight > 0 )
        {
            uint lightIndex = other.GetCandidateRaw();
            if (lightIndex != RTXPT_INVALID_LIGHT_INDEX)
            {
                bool candidateIsScreenSpaceCoherent = (lightIndex & LFR_SCREEN_SPACE_COHERENT_FLAG) != 0;
                lightIndex &= ~LFR_SCREEN_SPACE_COHERENT_FLAG;
                Add( randomValue, lightIndex, otherTotalWeight, candidateIsScreenSpaceCoherent );
            }
        }
    }

    // we can delete this safely now unless we decide to go back to buffered approach
    void CommitToStorage()
    {
    }

    //void Merge( )
};

/** Describes a light sample, mainly for use in NEE (but could be just a standalone direct illumination light sampler).
    It is considered in the context of a shaded surface point, from which Distance and Direction to light sample are computed.
    In case of emissive triangle light source, it is advisable to compute anti-self-intersection offset before computing
    distance and direction, even though distance shortening is needed anyways for shadow rays due to precision issues.
*/
struct LightSample
{
    float3  Li;                     ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
    //float   Pdf;                    ///< Pdf with respect to solid angle at the shading point with selected light (selectionPDF*solidAnglePdf).
    float   Distance;               ///< Ray distance for visibility evaluation (NOT shortened or offset to avoid self-intersection). Ray starts at shading surface.
    float3  Direction;              ///< Ray direction (normalized). Ray starts at shading surface.
    uint    LightIndex;             ///< Identifier of the source light (index in the light list), 0xFFFFFFFF if not available.
    float   SelectionPdf;           ///< Pdf of just the source light (LightIndex) selection; In contrast to 'LightSample::Pdf' which is a 'selectionPDF * solidAnglePdf'.
    float   SolidAnglePdf;
    bool    LightSampleableByBSDF;  ///< Required for MIS vs BSDF; typically emissive and environment samples can be "seen" by BSDF, while analytic Sphere/Point/Spotlight/etc. are virtual, with non-scene representation
    bool    FromLocalDistribution;  ///< Was drawn from NEE-AT local distribution; false if global or NEE-AT not used
        
    static LightSample make() 
    { 
        LightSample ret; 
        ret.Li = float3(0,0,0); 
        // ret.Pdf = 0; 
        ret.Distance = 0; 
        ret.Direction = float3(0,0,0); 
        ret.LightIndex = 0xFFFFFFFF;
        ret.LightSampleableByBSDF = false;
        ret.SolidAnglePdf = 0;
        ret.SelectionPdf = 0;
        ret.FromLocalDistribution = false;
        return ret; 
    }

    bool Valid()    
    { 
        return any(Li > 0); 
    }
};

#endif // !defined(__cplusplus)

#include "PolymorphicLight.h"

#endif // #define __LIGHTING_TYPES_HLSLI__