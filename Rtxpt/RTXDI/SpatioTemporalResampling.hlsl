/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma pack_matrix(row_major)

#include "RtxdiApplicationBridge.hlsli"


 // Only enable the boiling filter for RayQuery (compute shader) mode because it requires shared memory
#if USE_RAY_QUERY
#define RTXDI_ENABLE_BOILING_FILTER
#define RTXDI_BOILING_FILTER_GROUP_SIZE RTXDI_SCREEN_SPACE_GROUP_SIZE
#endif

#include "../../External/Rtxdi/include/rtxdi/ResamplingFunctions.hlsli"

#if USE_RAY_QUERY
[numthreads(RTXDI_SCREEN_SPACE_GROUP_SIZE, RTXDI_SCREEN_SPACE_GROUP_SIZE, 1)]
void main(uint2 GlobalIndex : SV_DispatchThreadID, uint2 LocalIndex : SV_GroupThreadID, uint2 GroupIdx : SV_GroupID)
#else
[shader("raygeneration")]
void RayGen()
#endif
{
#if !USE_RAY_QUERY
    uint2 GlobalIndex = DispatchRaysIndex().xy;
#endif
    const RTXDI_ResamplingRuntimeParameters params = g_ResamplingConst.runtimeParams;

    uint2 pixelPosition = RTXDI_DIReservoirToPixelPos(GlobalIndex, params);

    RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPosition, 2); //Falcor uses 4

    RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

    //Do we need this
     //bool usePermutationSampling = false;
     //if (g_ResamplingConst.enablePermutationSampling)
     //{
     //    // Permutation sampling makes more noise on thin, high-detail objects.
     //    usePermutationSampling = !IsComplexSurface(pixelPosition, surface);
     //}

    RTXDI_DIReservoir spResult = RTXDI_EmptyDIReservoir();
    int2 temporalSamplePixelPos = -1;

    if (RAB_IsSurfaceValid(surface))
    {
        RTXDI_DIReservoir curSample = RTXDI_LoadDIReservoir(params,
            GlobalIndex, g_ResamplingConst.initialOutputBufferIndex);

        float3 motionVector = u_MotionVectors[pixelPosition].xyz;

        motionVector = convertMotionVectorToPixelSpace(g_ResamplingConst.view, g_ResamplingConst.prevView, pixelPosition, motionVector);

        RTXDI_SpatioTemporalResamplingParameters stparams;
        stparams.screenSpaceMotion = motionVector;
        stparams.sourceBufferIndex = g_ResamplingConst.temporalInputBufferIndex;
        stparams.maxHistoryLength = g_ResamplingConst.maxHistoryLength;
        stparams.biasCorrectionMode = g_ResamplingConst.temporalBiasCorrection;
        stparams.depthThreshold = g_ResamplingConst.temporalDepthThreshold;
        stparams.normalThreshold = g_ResamplingConst.temporalNormalThreshold;
        stparams.numSamples = g_ResamplingConst.numSpatialSamples + 1;
        stparams.numDisocclusionBoostSamples = g_ResamplingConst.numDisocclusionBoostSamples;
        stparams.samplingRadius = g_ResamplingConst.spatialSamplingRadius;
        stparams.enableVisibilityShortcut = g_ResamplingConst.discardInvisibleSamples;
        stparams.enablePermutationSampling = g_ResamplingConst.enablePermutationSampling;

        RAB_LightSample selectedLightSample = (RAB_LightSample)0;

        spResult = RTXDI_SpatioTemporalResampling(pixelPosition, surface, curSample,
            rng, stparams, params, temporalSamplePixelPos, selectedLightSample);
    }

#ifdef RTXDI_ENABLE_BOILING_FILTER
    if (g_ResamplingConst.boilingFilterStrength > 0)
    {
        RTXDI_BoilingFilter(LocalIndex, g_ResamplingConst.boilingFilterStrength, params, spResult);
    }
#endif

    RTXDI_StoreDIReservoir(spResult, params, GlobalIndex, g_ResamplingConst.temporalOutputBufferIndex);
}