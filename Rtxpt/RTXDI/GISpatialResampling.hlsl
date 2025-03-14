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

#include "../RTXDI/RtxdiApplicationBridge.hlsli"
#include <Rtxdi/GI/Reservoir.hlsli>
#include <Rtxdi/GI/TemporalResampling.hlsli>
#include <Rtxdi/GI/SpatioTemporalResampling.hlsli>
#include <Rtxdi/GI/SpatialResampling.hlsli>

[numthreads(RTXDI_SCREEN_SPACE_GROUP_SIZE, RTXDI_SCREEN_SPACE_GROUP_SIZE, 1)]
void main(uint2 GlobalIndex : SV_DispatchThreadID, uint2 LocalIndex : SV_GroupThreadID, uint2 GroupIdx : SV_GroupID)
{
    uint2 pixelPosition = RTXDI_ReservoirPosToPixelPos(GlobalIndex, g_RtxdiBridgeConst.runtimeParams.activeCheckerboardField);

    RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPosition, 6);

    RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

    RTXDI_GIReservoir resultReservoir = RTXDI_EmptyGIReservoir();
    
    if (RAB_IsSurfaceValid(surface))
    {
        RTXDI_GIReservoir inputReservoir = RTXDI_LoadGIReservoir(
            g_RtxdiBridgeConst.restirGI.reservoirBufferParams,
            GlobalIndex,
            g_RtxdiBridgeConst.restirGI.bufferIndices.spatialResamplingInputBufferIndex);

        RTXDI_GISpatialResamplingParameters sparams;
        sparams.sourceBufferIndex = g_RtxdiBridgeConst.restirGI.bufferIndices.spatialResamplingInputBufferIndex;
        sparams.depthThreshold = g_RtxdiBridgeConst.restirGI.spatialResamplingParams.spatialDepthThreshold;
        sparams.normalThreshold = g_RtxdiBridgeConst.restirGI.spatialResamplingParams.spatialNormalThreshold;
        sparams.numSamples = g_RtxdiBridgeConst.restirGI.spatialResamplingParams.numSpatialSamples;
        sparams.samplingRadius = g_RtxdiBridgeConst.restirGI.spatialResamplingParams.spatialSamplingRadius;
        sparams.biasCorrectionMode = g_RtxdiBridgeConst.restirGI.spatialResamplingParams.spatialBiasCorrectionMode;
        
        resultReservoir = RTXDI_GISpatialResampling(
            pixelPosition, 
            surface, 
            inputReservoir,
            rng, 
            g_RtxdiBridgeConst.runtimeParams,
            g_RtxdiBridgeConst.restirGI.reservoirBufferParams,
            sparams);
    }

    RTXDI_StoreGIReservoir(
        resultReservoir, 
        g_RtxdiBridgeConst.restirGI.reservoirBufferParams, 
        GlobalIndex,
        g_RtxdiBridgeConst.restirGI.bufferIndices.spatialResamplingOutputBufferIndex);
}