/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __NEEAT_BAKER_HLSL__
#define __NEEAT_BAKER_HLSL__

#include "NEEATShared.hlsli"

// this will enable various valdiation passes which will debug print or otherwise indicate issues with the algorithm or input data / buffers setup
#define LLB_ENABLE_VALIDATION           0

#define LLB_NUM_COMPUTE_THREADS         128
#define LLB_NUM_COMPUTE_THREADS_2D      8
#define LLB_LOCAL_BLOCK_SIZE            32

#define LLB_PREPROCESS_BLOCK_SIZE_OUTER 16
#define LLB_PREPROCESS_BLOCK_SIZE_INNER (LLB_PREPROCESS_BLOCK_SIZE_OUTER-2)

// The prefilter runs twice, ping-ponging through the feedback reservoir textures. To avoid a read-vs-write race between
// adjacent thread groups (each group reads a 1 pixel margin that lies inside its neighbour's write region) the two passes
// read and write disjoint regions: the feedback textures are allocated this many times taller than the logical resolution,
// pass A does [0,H) -> [H,2H) and pass B does [H,2H) -> [0,H). Everything outside the prefilter only ever touches [0,H).
#define LLB_PREPROCESS_PINGPONG_SLICES  2

#define LLB_MAX_PROXIES_PER_TASK        32
#define LLB_MAX_PROXY_PROC_TASKS        (NEEAT_MAX_LIGHTS+((RTXPT_LIGHTING_MAX_SAMPLING_PROXIES+LLB_MAX_PROXIES_PER_TASK-1) / LLB_MAX_PROXIES_PER_TASK))

#define LLB_SCRATCH_BUFFER_SIZE         (64*1024*1024) // this is in bytes
// count in float-s; buffer size in bytes should be 2 * sizeof(float) * RTXPT_LIGHTING_WEIGHTS_COUNT_HALF
#define RTXPT_LIGHTING_WEIGHTS_COUNT_HALF     (NEEAT_MAX_LIGHTS+1)   // +1 is purely because perLightProxyCounters needs one more to store invalid feedback; "half" is because we need 2x so we can ping-pong with historic

#define NEEAT_TOTAL_WEIGHT_CLEAR_THRESHOLD 1e-6

namespace NEEAT
{
    struct EmissiveTrianglesProcTask
    {
        uint InstanceIndex;
        uint GeometryIndex;
        uint TriangleIndexFrom;
        uint TriangleIndexTo;
        uint DestinationBufferOffset;
        uint HistoricBufferOffset;
        uint ClusterBufferOffset;           // light index of the task's first cluster (or NEEAT_INVALID_LIGHT_INDEX if none)
        uint HistoricClusterBufferOffset;   // previous frame's light index of the task's first cluster (or NEEAT_INVALID_LIGHT_INDEX)
    };

    struct SamplingProxyBuildProcTask
    {
        uint LightIndex; // <- index into u_lightsBuffer
        uint ProxyIndexBase; // useful for figuring out sampling proxy index within its own proxies
        uint FillProxyIndexFrom; // this task needs to fill from this index                                  
        uint FillProxyIndexTo; // this task needs to fill to this index                                    
    };

#if defined(__cplusplus)
    static_assert( sizeof(SamplingProxyBuildProcTask) * LLB_MAX_PROXY_PROC_TASKS <= LLB_SCRATCH_BUFFER_SIZE ); // does it fit
#endif

    // These should go into LightsBaker.hlsli or similar
    enum class DebugViewType : int
    {
        Disabled,

        Disocclusion,
        FeedbackHistoryInfo,
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

#if !defined(__cplusplus)

    struct MicroRng
    {
        uint    N;
    
        static MicroRng make( uint seed )
        {
            MicroRng ret;
            ret.N = seed ^ 0x9e3779b9;    // with our hash function, 0 will loop so please don't seed with 0x9e3779b9 :D
            return ret;
        }
    
        static MicroRng make( uint2 seedCoord, uint seedValue )
        {
            MicroRng ret;
            ret.N = ((seedCoord.x << 16) | seedCoord.y) ^ 0x9e3779b9;
            ret.N = ret.N ^ (seedValue + (ret.N << 6) + (ret.N >> 2));
            return ret;
        }

        static MicroRng make( uint2 seedCoord, uint seedValueA, uint seedValueB )
        {
            MicroRng ret;
            ret.N = ((seedCoord.x << 16) | seedCoord.y) ^ 0x9e3779b9;
            ret.N = ret.N ^ (seedValueA + (ret.N << 6) + (ret.N >> 2));
            ret.N = ret.N ^ (seedValueB + (ret.N << 6) + (ret.N >> 2));
            return ret;
        }
        
        uint Next()
        {
            // See https://nullprogram.com/blog/2018/07/31/, "Prospecting for Hash Functions" by Chris Wellons
            // Latest update: https://github.com/skeeto/hash-prospector/issues/19
            N ^= N >> 16;
            N *= 0x21f0aaad;
            N ^= N >> 15;
            N *= 0xf35a2d97;
            N ^= N >> 15;
            return N;
        }
    
        float NextFloat()
        {
            // a.) converting the upper 24bits to [0, 1) because the higher bits have better distribution in some hash algorithms (like sobol)
            // b.) this is a good way to guarantee [0, 1) since float32 mantissa is only 23 bits
            return (Next() >> 8) / float(1 << 24); // same as '/ 16777216.0'
        }

        float2 NextFloat2()  { return float2(NextFloat(), NextFloat()); }
    };

    // Holds all the UAVs/SRVs the NEE-AT baker operates on. This bundles the resources so the baking
    // algorithm can be authored as standalone library code, decoupled from how the host declares and
    // binds the underlying buffers/textures. The host (see LightsBaker.hlsl) fills one of these from its
    // declared resource registers and threads it through the baking passes.
    struct BakerContext
    {
        RWStructuredBuffer<ControlData>     ControlBuffer;                  // control/constant data (element 0 holds ControlData + BakerConstants)

        RWByteAddressBuffer                 ScratchBuffer;                  // general purpose scratch (proc tasks, etc.)
        RWBuffer<uint>                      ScratchList;                    // general purpose scratch list

        RWBuffer<float>                     LightWeights;                   // per-light weights (ping-pongs current/historic via buffer offset)
        RWBuffer<float>                     LightWeightsBoost;              // per-light weight boost, current frame only (no ping-pong, so indexed directly by light index)
        RWBuffer<uint>                      HistoryRemapCurrentToPast;      // maps current-frame light index -> previous-frame light index
        RWBuffer<uint>                      HistoryRemapPastToCurrent;      // maps previous-frame light index -> current-frame light index
        RWBuffer<uint>                      PerLightProxyCounters;          // per-light sampling proxy counters
        RWBuffer<uint>                      LightSamplingProxies;           // global sampling proxy list
        RWTexture2D<uint>                   EnvLightLookupMap;              // environment map -> light index lookup

        // feedback reservoirs
        RWTexture2D<float>                  FeedbackTotalWeight;            // main reservoir working surface (total weight)
        RWTexture2D<uint>                   FeedbackCandidates;             // main reservoir working surface (candidates)
        RWTexture2D<float>                  FeedbackTotalWeightScratch;     // temporary surface reprojected into in P1, consumed by P2 (and in some cases Clear)
        RWTexture2D<uint>                   FeedbackCandidatesScratch;      // temporary surface reprojected into in P1, consumed by P2 (and in some cases Clear)
        RWTexture2D<float>                  FeedbackTotalWeightBlended;     // low res surface where the early feedback is blended together
        RWTexture2D<uint>                   FeedbackCandidatesBlended;      // low res surface where the early feedback is blended together

        RWTexture2D<float>                  HistoryDepth;                   // previous-frame depth used for reprojection/disocclusion
        RWBuffer<uint>                      LocalSamplingBuffer;            // screen-space (tile) local sampling buffer

        Texture2D<float>                    DepthBuffer;                    // engine's depth buffer
        Texture2D<float3>                   MotionVectors;                  // engine's motion vectors
        Texture2D<float4>                   EnvRadianceAndImportanceMap;    // environment radiance + importance
    };

    // Invalidate the previous-frame -> current-frame light remap. Meant to be called once per light index;
    // the host is expected to dispatch this over max(HistoricTotalLightCount, TotalLightCount) threads and
    // pass the global thread index as lightIndex.
    void ResetPastToCurrentHistory( BakerContext context, uint lightIndex )
    {
        uint totalCount = max( context.ControlBuffer[0].HistoricTotalLightCount, context.ControlBuffer[0].TotalLightCount );
        if( lightIndex >= totalCount )
            return;
        context.HistoryRemapPastToCurrent[lightIndex] = NEEAT_INVALID_LIGHT_INDEX;
    }

#endif // !defined(__cplusplus)

} // namespace NEEAT

#endif // __NEEAT_BAKER_HLSL__