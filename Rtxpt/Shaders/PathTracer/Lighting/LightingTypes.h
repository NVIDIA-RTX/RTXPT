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
#endif

// Environment map color/intensity and orientation modifiers ("in-scene" settings)
struct EnvMapSceneParams
{
	float3x4    Transform;              ///< Local to world transform.
	float3x4    InvTransform;           ///< World to local transform.

    float3      ColorMultiplier;        ///< Color & radiance scale (Tint * Intensity)
    float       Enabled;                ///< 1 if enabled, 0 if not
};

// Environment map importance sampling internals
struct EnvMapImportanceSamplingParams
{
    // MIP descent sampling
    float2      ImportanceInvDim;       ///< 1.0 / dimension.
    uint        ImportanceBaseMip;      ///< Mip level for 1x1 resolution.
    uint        padding0;
};

// Returned by importance sampling functions
struct DistantLightSample
{
    float3  Dir;        ///< Sampled direction towards the light in world space.
    float   Pdf;        ///< Probability density function for the sampled direction with respect to solid angle.
    float3  Le;         ///< Emitted radiance.
};

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
    float   LightSampling_MIS_Boost;    ///< This is a fixed pdf used for the sampler when doing MIS with main path BSDF

    uint    TemporalFeedbackRequired;   ///< Whether the path tracing NEE needs to provide temporal feedback
    uint    TotalMaxFeedbackCount;      ///< Copy of 'LightsBakerConstants' value, used for debugging
    float   GlobalFeedbackUseRatio;
    float   LocalFeedbackUseRatio;

    uint    TileBufferHeight;           ///< Feedback and tile buffers for direct and indirect are stacked one on top of the other; this is the height of just one (and offset to get from direct to indirect)
    float   DirectVsIndirectThreshold;  ///< Used to determine whether to use direct vs indirect light caching strategy for current surface
    uint2   LocalSamplingResolution;    ///< The resolution of the screen space local sampling buffer (number of tiles x * y)

    uint2   LocalSamplingTileJitter;
    uint2   LocalSamplingTileJitterPrev;

    uint    ValidFeedbackCount;         ///< For debugging only
    uint    padding1;
    uint    padding2;
    uint    padding3;

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

    MissingFeedbackDirect,
    MissingFeedbackIndirect,
    FeedbackRawDirect,
    FeedbackRawIndirect,
    FeedbackAfterClear,
    LowResBlendedFeedback,

    TileHeatmap,

    Disocclusion,

    ValidateCorrectness,

    MaxCount
};
struct LightsBakerConstants
{
    EnvMapSceneParams       EnvMapParams;
    float                   DistantVsLocalRelativeImportance;
    uint                    EnvMapImportanceMapMIPCount;
    uint                    EnvMapImportanceMapResolution;
    uint                    TriangleLightTaskCount;

    uint2                   LocalSamplingResolution;
    float                   GlobalFeedbackUseRatio;
    float                   LocalFeedbackUseRatio;

    uint2                   FeedbackResolution;
    uint2                   BlendedFeedbackResolution;

    uint                    TotalMaxFeedbackCount;
    uint                    TotalLightCount;
    uint                    UpdateCounter;              ///< LightBaker's own 'frame' counter (gets reset with LightsBaker::BakeSettings::ResetFeedback and gets incremented on every LightSbaker::UpdateFrame(...) and non-first UpdatePreRender after UpdateFrame )
    uint                    LastFrameTemporalFeedbackAvailable;

    uint2                   MouseCursorPos; // for debugging viz only
    float2                  PrevOverCurrentViewportSize; ///< viewPrev.viewportSize / view.viewportSize

    int                     DebugDrawType;
    uint                    DebugDrawTileLights;
    uint                    LastFrameLocalSamplesAvailable;
    uint                    DebugDrawFrustum;

    float                   ImportanceBoostIntensityDelta;
    float                   ImportanceBoostFrustumMul;
    float                   ImportanceBoostFrustumFadeRangeExt;
    float                   ImportanceBoostFrustumFadeRangeInt;

    float3                  SceneCameraPos;
    float                   SceneAverageContentsDistance;

    float                   DepthDisocclusionThreshold;
    uint                    EnableMotionReprojection;
    float                   ReservoirHistoryDropoff;
    uint                    AntiLagEnabled;

    uint2                   LocalSamplingTileJitter;
    uint2                   LocalSamplingTileJitterPrev;

    float4                  FrustumPlanes[6];              ///< Left Right Top Bottom Near Far
    
    float4                  FrustumCorners[8];
};

#define RTXPT_INVALID_LIGHT_INDEX                       0xFFFFFFFF

inline uint ComputeLocalSampleCount(const float localTemporalFeedbackRatio, const uint totalSamples)
{
    return (uint)((float)(totalSamples-1) * localTemporalFeedbackRatio + 0.75f);    // always leave 
}

inline uint ComputeGlobalSampleCount(const float localTemporalFeedbackRatio, const uint totalSamples)
{
    return totalSamples - (uint)((float)(totalSamples - 1) * localTemporalFeedbackRatio + 0.75f);    // always leave 
}

#if !defined(__cplusplus)

// Note:
//  * these are dependent for maximum number of global lights under 0x007FFFFF and counter under or equal to 0x1FF.
//  * sorting algorithms rely on light index being packed in high bits
//  * sorting/coalescing algorithms rely on counter being packed in low bits so ++ operation is legal
uint PackMiniListLightAndCount(uint globalLightIndex, uint counter)                             { return ((globalLightIndex & 0x007FFFFF) << 9) | ((counter-1) & 0x1FF); }
void UnpackMiniListLightAndCount(uint value, out uint globalLightIndex, out uint counter)       { globalLightIndex = value >> 9; counter = (value & 0x1FF)+1; }
uint UnpackMiniListLight(uint value)                                                            { return value >> 9; }
uint UnpackMiniListCount(uint value)                                                            { return (value & 0x1FF)+1; }


#if RTXPT_LIGHTING_LOCAL_SAMPLING_BUFFER_IS_3D_TEXTURE
#define LOCAL_SAMPLING_BUFFER_TYPE_SRV Texture3D<uint>
#define LOCAL_SAMPLING_BUFFER_TYPE_UAV RWTexture3D<uint>
#else
#define LOCAL_SAMPLING_BUFFER_TYPE_SRV Buffer<uint>
#define LOCAL_SAMPLING_BUFFER_TYPE_UAV RWBuffer<uint>
#endif

#if RTXPT_LIGHTING_LOCAL_SAMPLING_BUFFER_IS_3D_TEXTURE==0
inline uint LLSB_ComputeBaseAddress(uint2 tilePos, uint2 localSamplingResolution)
{
    return (tilePos.x + (tilePos.y * localSamplingResolution.x)) * RTXPT_LIGHTING_LOCAL_PROXY_COUNT;
}
#endif



#define LFR_BUFFERED 1

#if RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH == 1
#define RTXPT_FEEDBACK_ACCESS_INDEX(_VAR, _IDX)   _VAR
#else
#define RTXPT_FEEDBACK_ACCESS_INDEX(_VAR, _IDX)   _VAR[_IDX]
#endif


// Weighted Reservoir Sampling helper for storing good lights for later reuse. Since our reuse is entirely statistical, we don't actually keep the weights
// https://www.pbr-book.org/4ed/Sampling_Algorithms/Reservoir_Sampling
struct LightFeedbackReservoir
{
    #define LFR_INDIRECT_CANDIDATE_FLAG         0x80000000u
    #define LFR_MAX_WEIGHT                      1e12

    uint2                                       PixelPos;
    RWTexture2D<float>                          TextureTotalWeight;
    RWTexture2D<NEEAT_FEEDBACK_CANDIDATE_TYPE>  TextureCandidates;
#if LFR_BUFFERED
    float                                       TotalWeight;
    NEEAT_FEEDBACK_CANDIDATE_TYPE               Candidates;
    bool                                        DirtyW;
    bool                                        DirtyC;
#endif

    static LightFeedbackReservoir make(uint2 pixelPos, RWTexture2D<float> textureTotalWeight, RWTexture2D<NEEAT_FEEDBACK_CANDIDATE_TYPE> textureCandidates)
    {
        LightFeedbackReservoir ret;
        ret.PixelPos            = pixelPos;
        ret.TextureTotalWeight  = textureTotalWeight;
        ret.TextureCandidates   = textureCandidates;
#if LFR_BUFFERED
        ret.TotalWeight = min( LFR_MAX_WEIGHT, textureTotalWeight[pixelPos] );
        ret.Candidates  = textureCandidates[pixelPos];
        ret.DirtyW = false;
        ret.DirtyC = false;
#endif
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
        for (int i = 0; i < RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH; i++)
            SetCandidate(i, RTXPT_INVALID_LIGHT_INDEX, false);
#endif
    }

    bool IsEmpty()
    {
        return GetTotalWeight() == 0;
    }

    float GetTotalWeight()
    {
#if LFR_BUFFERED
        return TotalWeight;
#else
        return TextureTotalWeight[PixelPos];
#endif
    }

    void SetTotalWeight(float totalWeight)
    {
        totalWeight = min( LFR_MAX_WEIGHT, totalWeight );
#if LFR_BUFFERED
        TotalWeight = totalWeight;
        DirtyW = true;
#else
        TextureTotalWeight[PixelPos] = totalWeight;
#endif
    }

    NEEAT_FEEDBACK_CANDIDATE_TYPE GetCandidatesRaw()
    {
#if LFR_BUFFERED
        return Candidates;
#else
        return TextureCandidates[PixelPos];
#endif
    }

    void SetCandidatesRaw(NEEAT_FEEDBACK_CANDIDATE_TYPE candidates)
    {
#if LFR_BUFFERED
        Candidates = candidates;
        DirtyC = true;
#else
        TextureCandidates[PixelPos] = candidates;
#endif
    }

    uint GetCandidateRaw(uint i)
    {
#if LFR_BUFFERED
        return RTXPT_FEEDBACK_ACCESS_INDEX(Candidates,i);
#else
        return RTXPT_FEEDBACK_ACCESS_INDEX(TextureCandidates[PixelPos]),i);
#endif
    }

    void SetCandidateRaw(uint i, uint candidateIndex)
    {
#if LFR_BUFFERED
        RTXPT_FEEDBACK_ACCESS_INDEX(Candidates,i) = candidateIndex;
        DirtyC = true;
#else
        RTXPT_FEEDBACK_ACCESS_INDEX(TextureCandidates[PixelPos],i) = candidateIndex;
#endif
    }

    void GetCandidate(uint i, out uint candidateIndex, out bool candidateIsIndirect)
    {
        candidateIndex = RTXPT_INVALID_LIGHT_INDEX;
        candidateIsIndirect = false;
        if (IsEmpty())
            return;

        candidateIndex = GetCandidateRaw(i);
        if (candidateIndex != RTXPT_INVALID_LIGHT_INDEX)
        {
            candidateIsIndirect = (candidateIndex & LFR_INDIRECT_CANDIDATE_FLAG) != 0;
            candidateIndex &= ~LFR_INDIRECT_CANDIDATE_FLAG;
        }
    }

    void SetCandidate(uint i, uint candidateIndex, bool candidateIsIndirect )
    {
        SetCandidateRaw(i, candidateIndex | ((candidateIsIndirect)?(LFR_INDIRECT_CANDIDATE_FLAG):(0)) );
    }

    // Add single new light source with weight; it will be added stochastically to as many slots as available
    void Add( const float randomValues[RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH], uint candidateIndex, float candidateWeight, bool candidateIsIndirect )
    {
        candidateWeight = min(LFR_MAX_WEIGHT, candidateWeight);
        // NOTE: caller ensures no race condition possible here
        float totalWeight = GetTotalWeight();
        totalWeight += candidateWeight;
        SetTotalWeight(totalWeight);
        float threshold = saturate(candidateWeight / totalWeight);

        // Bake in indirect flag
        if (candidateIsIndirect)
            candidateIndex |= LFR_INDIRECT_CANDIDATE_FLAG;

        // Stochastically add 
        for (int i = 0; i < RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH; i++ )
            if (randomValues[i] < threshold)
                SetCandidateRaw(i, candidateIndex);
    }

    // Merge reservoirs - can be used to merge a 3x3 kernel for ex.
    void Merge( const float randomValues[RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH][RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH], const LightFeedbackReservoir other, float otherScale = 1.0 )
    {
        float otherTotalWeight = min(LFR_MAX_WEIGHT, other.GetTotalWeight() * otherScale);
        if( otherTotalWeight > 0 )
        {
            for (int i = 0; i < RTXPT_LIGHTING_FEEDBACK_CANDIDATES_PER_PATH; i++)
            {
                uint lightIndex = other.GetCandidateRaw(i);
                if (lightIndex != RTXPT_INVALID_LIGHT_INDEX)
                {
                    bool candidateIsIndirect = (lightIndex & LFR_INDIRECT_CANDIDATE_FLAG) != 0;
                    lightIndex &= ~LFR_INDIRECT_CANDIDATE_FLAG;
                    Add( randomValues[i], lightIndex, otherTotalWeight, candidateIsIndirect );
                }
            }
        }
    }

    void CommitToStorage()
    {
#if LFR_BUFFERED
        if (DirtyW)
            TextureTotalWeight[PixelPos] = TotalWeight;
        if (DirtyC)
            TextureCandidates[PixelPos] = Candidates;
#endif
    }

    //void Merge( )
};

#endif // !defined(__cplusplus)

#include "PolymorphicLight.h"

#endif // #define __LIGHTING_TYPES_HLSLI__