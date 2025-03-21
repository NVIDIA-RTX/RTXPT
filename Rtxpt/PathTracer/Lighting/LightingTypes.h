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

#if !defined(__cplusplus)
#pragma pack_matrix(row_major)
#else
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
    uint    WeightsSumUINT;
    uint    ImportanceSamplingType;     ///< From LightsBaker::BakeSettings - should match global NEEType
    float   LightSampling_MIS_Boost;    ///< This is a fixed pdf used for the sampler when doing MIS with main path BSDF

    uint    TemporalFeedbackRequired;   ///< Whether the path tracing NEE needs to provide temporal feedback
    uint    ValidFeedbackCount;

    float   GlobalFeedbackUseRatio;
    float   NarrowFeedbackUseRatio;

    uint    FeedbackBufferHeight;       ///< Feedback and tile buffers for direct and indirect are stacked one on top of the other; this is the height of just one (and offset to get from direct to indirect)
    uint    TileBufferHeight;           ///< Feedback and tile buffers for direct and indirect are stacked one on top of the other; this is the height of just one (and offset to get from direct to indirect)
    float   DirectVsIndirectThreshold;  ///< Used to determine whether to use direct vs indirect light caching strategy for current surface
    uint    _padding0;

    uint2   LocalSamplingTileJitter;
    uint2   LocalSamplingTileJitterPrev;

    float4  SceneCameraPos;             ///< Reference center (perhaps rename?) - currently used for debugging viz only
    // float4  SceneWorldMax;

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

    MissingFeedback,
    FeedbackRaw,
    FeedbackProcessed,
    FeedbackHistoric,
    FeedbackLowRes,
    FeedbackReadyForNew,

    TileHeatmap,

    MaxCount
};
struct LightsBakerConstants
{
    EnvMapSceneParams       EnvMapParams;
    float                   DistantVsLocalRelativeImportance;
    uint                    EnvMapImportanceMapMIPCount;
    uint                    EnvMapImportanceMapResolution;
    uint					_padding0;

    uint                    TriangleLightTaskCount;
    float                   _padding1;
    float                   GlobalFeedbackUseRatio;
    float                   NarrowFeedbackUseRatio;

    uint2                   FeedbackResolution;
    uint2                   LRFeedbackResolution;

    uint2                   NarrowSamplingResolution;
    uint                    SampleIndex;
    uint                    _padding2;

    uint2                   MouseCursorPos; // for debugging viz only
    float2                  PrevOverCurrentViewportSize; ///< viewPrev.viewportSize / view.viewportSize

    int                     DebugDrawType;
    uint                    DebugDrawTileLights;
    uint                    DebugDrawDirect;
    uint                    _padding3;
};

#define RTXPT_INVALID_LIGHT_INDEX                       0xFFFFFFFF

// general settings
#define RTXPT_LIGHTING_MAX_LIGHTS                       1 * 1024 * 1024                 // number of PolymorphicLightInfo (currently 48 bytes each) - million lights is the max, and that's it.
#define RTXPT_LIGHTING_SAMPLING_PROXY_RATIO             8                               // every light can have this many proxies on average 
#define RTXPT_LIGHTING_MAX_SAMPLING_PROXIES             RTXPT_LIGHTING_SAMPLING_PROXY_RATIO * RTXPT_LIGHTING_MAX_LIGHTS    // total buffer size required for proxies, worst case scenario
#define RTXPT_LIGHTING_MAX_SAMPLING_PROXIES_PER_LIGHT   32768                           // one light can have no more than this many proxies (puts bounds on power-based importance sampling component; up to 32768 was tested)
#define RTXPT_LIGHTING_MIN_WEIGHT_THRESHOLD             1e-7                            // ignore lights under this threshold

// tile (local) sampling settings
#define RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE        (8) // 6x6 is good; 8x8 is acceptable and will reduce post-processing (P3 pass) cost over 6x6 by around 1.8x; the loss in quality is on small detail and shadows
#define RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE      (10) // has to be same as RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE or n*2+RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE
#define RTXPT_LIGHTING_LR_SAMPLING_BUFFER_SCALE         (4)
#define RTXPT_LIGHTING_LR_SAMPLING_BUFFER_WINDOW_SIZE   (4) // that's times RTXPT_LIGHTING_LR_SAMPLING_BUFFER_SCALE in real world pixels
#define RTXPT_LIGHTING_HISTORIC_SAMPLING_COUNT          12 //(28)

// we take, roughly (see FillTile)
// 1.) one per pixel sample from past most used samples in the RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE window
// 2.) one per pixel sample from low res buffer RTXPT_LIGHTING_LR_SAMPLING_BUFFER_WINDOW_SIZE window - this helps with sharing most important feedback between neighbours
// 3.) exactly RTXPT_LIGHTING_HISTORIC_SAMPLING_COUNT historic samples from previous frame's feedback 
// NOTE: must always be less or equal compared to 256, due to current count packing
#define RTXPT_LIGHTING_NARROW_PROXY_COUNT              (RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE*RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE  \
                                                         + RTXPT_LIGHTING_LR_SAMPLING_BUFFER_WINDOW_SIZE*RTXPT_LIGHTING_LR_SAMPLING_BUFFER_WINDOW_SIZE \
                                                         + RTXPT_LIGHTING_HISTORIC_SAMPLING_COUNT)

// environment map quad tree settings
// (one potential future upgrade is to eliminate subdivisions when child nodes are all equally lit - e.g. there's no point having many nodes within the sun, only enough to capture the sun itself; this could be done with an additional "difference" mip-mapped lookup, produced together with the luminance map)
// (another potential future upgrade is to have nodes optionally capture & emit color themselves, like emissive triangles do, which would reduce noise and also speed up sampling side but is biased)
#define RTXPT_LIGHTING_ENVMAP_QT_BASE_RESOLUTION        16
#define RTXPT_LIGHTING_ENVMAP_QT_SUBDIVISIONS           160    // how many times the nodes will be subdivided; ~150-512 seems like a good balance depending on the number of bright light sources in the environment map
#define RTXPT_LIGHTING_ENVMAP_QT_ADDITIONAL_NODES       (3*RTXPT_LIGHTING_ENVMAP_QT_SUBDIVISIONS) // for each subdivision, one goes out, 4 get added - net is 3 new nodes
#define RTXPT_LIGHTING_ENVMAP_QT_TOTAL_NODE_COUNT       (RTXPT_LIGHTING_ENVMAP_QT_BASE_RESOLUTION*RTXPT_LIGHTING_ENVMAP_QT_BASE_RESOLUTION + RTXPT_LIGHTING_ENVMAP_QT_ADDITIONAL_NODES)

// This will not fully clear reservoirs but diminish existing content to 5% (with reprojection). This will help focus on stronger lights at the expense of less influential ones.
// Note: this doubles the amount of memory used for reservoirs.
#define RTXPT_LIGHTING_NEEAT_ENABLE_RESERVOIR_HISTORY   0
#define RTXPT_LIGHTING_NEEAT_ENABLE_INDIRECT_LOCAL_LAYER 0
#define RTXPT_LIGHTING_NEEAT_ENABLE_BSDF_FEEDBACK       1           //< provide NEE-AT feedback from BSDF rays hitting emissive surface/sky; helps primarily to speed up convergence / reduce lag

#define RTXPT_LIGHTING_NEEAT_MAX_TOTAL_SAMPLE_COUNT     63

// these are for testing/integration only
//#define RTXPT_NEEAT_MIS_OVERRIDE_BSDF_PDF               0.5     //< use constant value BSDF for MIS only; will result in more noise but still be unbiased (used to test BSDF pdf correctness on both MIS ends)
//#define RTXPT_NEEAT_MIS_OVERRIDE_SOLID_ANGLE_PDF        10      //< use constant value light solid angle pdf for MIS only; will result in more noise but still be unbiased (used to test whether solid angle pdf correctness on both MIS ends)

inline uint ComputeNarrowSampleCount(const float narrowTemporalFeedbackRatio, const uint totalSamples)
{
    return (uint)((float)(totalSamples-1) * narrowTemporalFeedbackRatio + 0.75f);    // always leave 
}

#if defined(__cplusplus)
// note: this is because 'groupshared' storage is limited to 32k, and we rely on it to fit preprocessing data. 
static_assert( RTXPT_LIGHTING_ENVMAP_QT_TOTAL_NODE_COUNT < 4096 );
#endif

#if !defined(__cplusplus)

// Note, these are dependent for maximum number of global lights under 0x00FFFFFF and counter under or equal to 0xFF.
uint PackMiniListLightAndCount(uint globalLightIndex, uint counter)                             { return (globalLightIndex & 0x00FFFFFF) | (((counter-1) & 0xFF) << 24); }
void UnpackMiniListLightAndCount(uint value, out uint globalLightIndex, out uint counter)       { globalLightIndex = 0x00FFFFFF & value; counter = (value >> 24)+1; }
uint UnpackMiniListLight(uint value)                                                            { return 0x00FFFFFF & value; }
uint UnpackMiniListCount(uint value)                                                            { return (value >> 24)+1; }

// Weighted Reservoir Sampling helper for storing good lights for later reuse.
// https://www.pbr-book.org/4ed/Sampling_Algorithms/Reservoir_Sampling
struct LightFeedbackReservoir
{
    uint        CandidateIndex;
    float       CandidateWeight;
    float       TotalWeight;

    static LightFeedbackReservoir make()
    {
        LightFeedbackReservoir ret;
        ret.CandidateIndex  = RTXPT_INVALID_LIGHT_INDEX;
        ret.CandidateWeight = 0;
        ret.TotalWeight     = 0;
        return ret;
    }
    
    static LightFeedbackReservoir Unpack12Byte( uint3 packed )
    {
        LightFeedbackReservoir ret;
        ret.CandidateIndex  = packed.x;
        ret.CandidateWeight = asfloat(packed.y);
        ret.TotalWeight     = asfloat(packed.z);
        return ret;
    }

    uint3 Pack12Byte()
    { 
        return uint3(CandidateIndex, asuint(CandidateWeight), asuint(TotalWeight));
    }

    static LightFeedbackReservoir Unpack8Byte(uint2 packed)
    {
        LightFeedbackReservoir ret;
        ret.CandidateIndex    = packed.x;
        ret.CandidateWeight   = f16tof32(packed.y & 0xFFFF);
        ret.TotalWeight    = f16tof32(packed.y >> 16);
        return ret;
    }

    uint2 Pack8Byte()
    {
        return uint2(CandidateIndex, f32tof16(CandidateWeight) | ( f32tof16(TotalWeight) << 16 ) ); 
    }

    void Add( const float randomValue, const uint candidateIndex, const float candidateWeight )
    {
        TotalWeight += candidateWeight;
        float threshold = saturate(candidateWeight / TotalWeight);
        if (randomValue < threshold)
        {
            CandidateIndex = candidateIndex;
            CandidateWeight = candidateWeight;
        }
    }

    void Merge( const float randomValue, const LightFeedbackReservoir other )
    {
        if( other.TotalWeight > 0 )
            Add( randomValue, other.CandidateIndex, other.TotalWeight );
    }

    void Retrieve( out uint candidateIndex, out float candidatePdf )
    {
        candidateIndex  = CandidateIndex;
        candidatePdf    = /*TotalCandidates **/ CandidateWeight / TotalWeight;
    }

    void Scale(float factor)
    {
        CandidateWeight *= factor;
        TotalWeight *= factor;
    }

    //void Merge( )
};

#endif // !defined(__cplusplus)

#include "PolymorphicLight.h"

#endif // #define __LIGHTING_TYPES_HLSLI__