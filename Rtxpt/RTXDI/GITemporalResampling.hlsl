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

#define NON_PATH_TRACING_PASS 1

// Only enable the boiling filter for RayQuery (compute shader) mode because it requires shared memory
#if USE_RAY_QUERY
#define RTXDI_ENABLE_BOILING_FILTER
#define RTXDI_BOILING_FILTER_GROUP_SIZE RTXDI_SCREEN_SPACE_GROUP_SIZE
#endif

#include "../RTXDI/RtxdiApplicationBridge.hlsli"
#include <Rtxdi/GI/Reservoir.hlsli>
#include <Rtxdi/GI/BoilingFilter.hlsli>
#include <Rtxdi/GI/TemporalResampling.hlsli>
#include <Rtxdi/GI/SpatioTemporalResampling.hlsli>
#include <Rtxdi/GI/SpatialResampling.hlsli>

[numthreads(RTXDI_SCREEN_SPACE_GROUP_SIZE, RTXDI_SCREEN_SPACE_GROUP_SIZE, 1)]
void main(uint2 GlobalIndex : SV_DispatchThreadID, uint2 LocalIndex : SV_GroupThreadID, uint2 GroupIdx : SV_GroupID)
{
    uint2 pixelPosition = RTXDI_ReservoirPosToPixelPos(GlobalIndex, g_RtxdiBridgeConst.runtimeParams.activeCheckerboardField);

    RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

    RTXDI_GIReservoir resultReservoir = RTXDI_EmptyGIReservoir();
    
    if (RAB_IsSurfaceValid(surface))
    {
        // Create GI reservoir from the position and orientation of the first secondary vertex from the Path Tracer
        const float4 secondaryPositionNormal = u_SecondarySurfacePositionNormal[pixelPosition];
        RTXDI_GIReservoir initialReservoir = RTXDI_MakeGIReservoir(
            secondaryPositionNormal.xyz,
            octToNdirUnorm32(asuint(secondaryPositionNormal.w)),
            u_SecondarySurfaceRadiance[pixelPosition].xyz,
            /* samplePdf = */ 1.0);
        
        if (g_RtxdiBridgeConst.reStirGIEnableTemporalResampling)
        {
            RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPosition, 5);

            float3 motionVector = u_MotionVectors[pixelPosition].xyz;

            motionVector = convertMotionVectorToPixelSpace(g_Const.view, g_Const.previousView, pixelPosition, motionVector);

            RTXDI_GITemporalResamplingParameters tparams;
            tparams.screenSpaceMotion = motionVector;
            tparams.sourceBufferIndex = g_RtxdiBridgeConst.restirGI.bufferIndices.temporalResamplingInputBufferIndex;
            tparams.maxHistoryLength = g_RtxdiBridgeConst.restirGI.temporalResamplingParams.maxHistoryLength;
            tparams.depthThreshold = g_RtxdiBridgeConst.restirGI.temporalResamplingParams.depthThreshold;
            tparams.normalThreshold = g_RtxdiBridgeConst.restirGI.temporalResamplingParams.normalThreshold;
            tparams.enablePermutationSampling = g_RtxdiBridgeConst.restirGI.temporalResamplingParams.enablePermutationSampling;
            tparams.enableFallbackSampling = g_RtxdiBridgeConst.restirGI.temporalResamplingParams.enableFallbackSampling;
            tparams.uniformRandomNumber = g_RtxdiBridgeConst.restirGI.temporalResamplingParams.uniformRandomNumber;
            tparams.maxReservoirAge = g_RtxdiBridgeConst.restirGI.temporalResamplingParams.maxReservoirAge;
            // Max age threshold should vary.
            if (g_RtxdiBridgeConst.reStirGIVaryAgeThreshold)    
                tparams.maxReservoirAge *= (0.5 + RAB_GetNextRandom(rng) * 0.5);

            resultReservoir = RTXDI_GITemporalResampling(
                pixelPosition, 
                surface, 
                initialReservoir,
                rng, 
                g_RtxdiBridgeConst.runtimeParams,
                g_RtxdiBridgeConst.restirGI.reservoirBufferParams,
                tparams);
        }
        else
        {
            resultReservoir = initialReservoir;
        }
    }

    // Sometimes, something in the renderer produces an invalid output, which then propagates
    // through resampling and eventually covers the screen. That shouldn't happen, but if it does,
    // the following filter can be enabled to stop the propagation.
    static const bool c_SanitizeOutput = false;
    if (c_SanitizeOutput)
    {
        if (any(isinf(resultReservoir.position)) || any(isnan(resultReservoir.position)) ||
            any(isinf(resultReservoir.radiance)) || any(isnan(resultReservoir.radiance)) ||
            any(isinf(resultReservoir.normal)) || any(isnan(resultReservoir.normal)) ||
            isinf(resultReservoir.weightSum) || isnan(resultReservoir.weightSum))
        {
            resultReservoir = RTXDI_EmptyGIReservoir();
        }
    }

#ifdef RTXDI_ENABLE_BOILING_FILTER
    if  (g_RtxdiBridgeConst.restirGI.temporalResamplingParams.enableBoilingFilter > 0)
    {
        RTXDI_GIBoilingFilter(
            LocalIndex, 
            g_RtxdiBridgeConst.restirGI.temporalResamplingParams.boilingFilterStrength, 
            resultReservoir);
    }
#endif

    RTXDI_StoreGIReservoir(
        resultReservoir,
        g_RtxdiBridgeConst.restirGI.reservoirBufferParams,
        GlobalIndex,
        g_RtxdiBridgeConst.restirGI.bufferIndices.temporalResamplingOutputBufferIndex);
}