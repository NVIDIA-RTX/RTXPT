/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __NEEAT_SHARED_HLSL__
#define __NEEAT_SHARED_HLSL__

#include "NEEATConfig.h"

#if defined(__cplusplus)

#ifdef NATIVE_CPP_VECTOR_TYPES_ENABLE
    NATIVE_CPP_VECTOR_TYPES_ENABLE
#else
    using namespace donut::math;
#endif

#define NEEAT_ROW_MAJOR
#else
#define NEEAT_ROW_MAJOR row_major
#endif

namespace NEEAT
{

    struct LightsBakerEnvMapParams
    {
        NEEAT_ROW_MAJOR float3x4    Transform;          ///< Local to world transform.
        NEEAT_ROW_MAJOR float3x4    InvTransform;       ///< World to local transform.
        float3                      ColorMultiplier;    ///< Color & radiance scale (Tint * Intensity)
        float                       Enabled;            ///< 1 if enabled, 0 if not
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
        float                   LightClusterBrightnessExponent; ///< see IsLightClusterActive; 0 disables

        float3                  SceneCameraPos;
        float                   SceneAverageContentsDistance;

        float                   DepthDisocclusionThreshold;
        uint                    EnableMotionReprojection;
        float                   ReservoirHistoryDropoff;
        uint                    EnableLightClusters;            ///< 0 keeps every light cluster inactive
    
        uint                    CurrentWeightsBufferOffset;     ///< Used to ping-pong between current and historic weights; it's either 0 or RTXPT_LIGHTING_WEIGHTS_COUNT_HALF
        uint                    HistoricWeightsBufferOffset;    ///< Used to ping-pong between current and historic weights; it's either 0 or RTXPT_LIGHTING_WEIGHTS_COUNT_HALF
        float                   LightClusterLODExponent;        ///< Cluster is active when LODMultiplier * pow(radius / distance, LODExponent) < 1
        float                   LightClusterLODMultiplier;


        float4                  FrustumPlanes[6];               ///< Left Right Top Bottom Near Far
        float4                  FrustumCorners[8];              ///< For debugging only

        LightsBakerEnvMapParams EnvMapParams;
    };

    STATIC_ASSERT( NEEAT_LIGHTS_BAKER_CONSTANTS_SIZE == sizeof(LightsBakerConstants) );

    // Used for building and using light list
    struct ControlData
    {
        uint    TotalLightCount;                    ///< Current total count of lights in the light buffer (of PolymorphicLightInfo type, max NEEAT_MAX_LIGHTS)
        uint    EnvmapQuadNodeCount;                ///< Number of environment map sampling lights in the light buffer (useful for debugging)
        uint    AnalyticLightCount;                 ///< Number of analytic lights in the light buffer (useful for debugging)
        uint    TriangleLightCount;                 ///< Number of emissive triangle lights in the light buffer (useful for debugging)

        uint    SamplingProxyCount;                 ///< Number of the sampling proxies (max RTXPT_LIGHTING_MAX_SAMPLING_PROXIES)
        uint    HistoricTotalLightCount;            ///< Previous frame's TotalLightCount (can be 0)
        uint    LastFrameTemporalFeedbackAvailable; ///< We can use last frame's temporal feedback
        uint    LastFrameLocalSamplesAvailable;     ///< We can use last frame's local (tile) lights (effectively same as LastFrameTemporalFeedbackAvailable from previous frame)

        uint    ProxyBuildTaskCount;                ///< Only used for building proxies (in LightsBaker.*)
        uint    WeightsSumUINT;                     ///< Used with interlocked float add - this is actually a float (can be read with 'asfloat')
        uint    ImportanceSamplingType;             ///< From LightsBaker::BakeSettings - should match global NEEType
        uint    ClusteredTriangleCount;             ///< Emissive triangles in active light clusters; counted on the GPU, for debugging only

        uint    TemporalFeedbackRequired;           ///< Whether the path tracing NEE needs to provide temporal feedback
        uint    TotalMaxFeedbackCount;              ///< Copy of 'LightsBakerConstants' value, used for debugging
        float   GlobalFeedbackUseWeight;
        float   LocalToGlobalSampleRatio;

        uint    TileBufferHeight;                   ///< Feedback and tile buffers for direct and indirect are stacked one on top of the other; this is the height of just one (and offset to get from direct to indirect)
        float   ScreenSpaceVsWorldSpaceThreshold;   ///< Used to determine whether to use direct vs indirect light caching strategy for current surface
        uint2   LocalSamplingResolution;            ///< The resolution of the screen space local sampling buffer (number of tiles x * y)

        uint2   LocalSamplingTileJitter;
        uint2   LocalSamplingTileJitterPrev;

        uint    ValidFeedbackCount;                 ///< For debugging only
        uint    DbgFreezeFeedback;                  ///< Everything functions as normal but feedback is frozen - only indices get updated w.r.t. to history
        uint    LightClusterCount;                  ///< Number of light clusters in the light buffer (useful for debugging)
        uint    ActiveLightClusterCount;            ///< Counted on the GPU, for debugging only

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

        // Light buffer layout: [envmap quads][analytic][emissive triangles][light clusters]
        uint    TriangleLightOffset()   { return EnvmapQuadNodeCount + AnalyticLightCount; }
        uint    LightClusterOffset()    { return EnvmapQuadNodeCount + AnalyticLightCount + TriangleLightCount; }
    };

    #define NEEAT_INVALID_LIGHT_INDEX                       0xFFFFFFFF

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
    //  * these are dependent on 1.) maximum number of global lights under 0x007FFFFF and 2.) counter under or equal to 0x1FF.
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
    struct FeedbackReservoir
    {
        #define LFR_SCREEN_SPACE_COHERENT_FLAG      0x80000000u
        #define LFR_MAX_WEIGHT                      6.5504e+4F  // HLF_MAX

        uint2                                       PixelPos;
        RWTexture2D<float>                          TextureTotalWeight;
        RWTexture2D<uint>                           TextureCandidates;

        static FeedbackReservoir make(uint2 pixelPos, RWTexture2D<float> textureTotalWeight, RWTexture2D<uint> textureCandidates)
        {
            FeedbackReservoir ret;
            ret.PixelPos            = pixelPos;
            ret.TextureTotalWeight  = textureTotalWeight;
            ret.TextureCandidates   = textureCandidates;
            return ret;
        }

        void CloneFrom(FeedbackReservoir other, float scale)
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
    #if 1 // this is useful for debugging, but should probably be removed in production
            SetCandidate(NEEAT_INVALID_LIGHT_INDEX, false);
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
            candidateIndex = NEEAT_INVALID_LIGHT_INDEX;
            candidateIsScreenSpaceCoherent = false;
            if (IsEmpty())
                return;

            candidateIndex = GetCandidateRaw();
            if (candidateIndex != NEEAT_INVALID_LIGHT_INDEX)
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
        void Merge( const float randomValue, const FeedbackReservoir other, float otherScale = 1.0 )
        {
            float otherTotalWeight = min(LFR_MAX_WEIGHT, other.GetTotalWeight() * otherScale);
            if( otherTotalWeight > 0 )
            {
                uint lightIndex = other.GetCandidateRaw();
                if (lightIndex != NEEAT_INVALID_LIGHT_INDEX)
                {
                    bool candidateIsScreenSpaceCoherent = (lightIndex & LFR_SCREEN_SPACE_COHERENT_FLAG) != 0;
                    lightIndex &= ~LFR_SCREEN_SPACE_COHERENT_FLAG;
                    Add( randomValue, lightIndex, otherTotalWeight, candidateIsScreenSpaceCoherent );
                }
            }
        }

        //void Merge( )
    };

#endif // !defined(__cplusplus)

} // namespace NEEAT


#endif // __NEEAT_SHARED_HLSL__