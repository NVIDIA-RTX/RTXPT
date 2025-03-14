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
#include <Rtxdi/DI/TemporalResampling.hlsli>
#include <Rtxdi/DI/SpatioTemporalResampling.hlsli>
#include <Rtxdi/DI/SpatialResampling.hlsli>

#if USE_RAY_QUERY
[numthreads(RTXDI_SCREEN_SPACE_GROUP_SIZE, RTXDI_SCREEN_SPACE_GROUP_SIZE, 1)]
void main(uint2 GlobalIndex : SV_DispatchThreadID)
#else
[shader("raygeneration")]
void RayGen()
#endif
{
#if !USE_RAY_QUERY
    uint2 GlobalIndex = DispatchRaysIndex().xy;
#endif
    
    uint2 pixelPosition = RTXDI_ReservoirPosToPixelPos(GlobalIndex, g_RtxdiBridgeConst.runtimeParams.activeCheckerboardField);

    RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPosition, 3);

    RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

    RTXDI_DIReservoir spatialReservoir = RTXDI_EmptyDIReservoir();
    RAB_LightSample lightSample = (RAB_LightSample)0;

    if (RAB_IsSurfaceValid(surface))
    {
        RTXDI_DIReservoir centerSample = RTXDI_LoadDIReservoir(
            g_RtxdiBridgeConst.restirDI.reservoirBufferParams,
            GlobalIndex,
            g_RtxdiBridgeConst.restirDI.bufferIndices.spatialResamplingInputBufferIndex);

        RTXDI_DISpatialResamplingParameters sparams;
        sparams.sourceBufferIndex = g_RtxdiBridgeConst.restirDI.bufferIndices.spatialResamplingInputBufferIndex;
        sparams.numSamples = g_RtxdiBridgeConst.restirDI.spatialResamplingParams.numSpatialSamples;
        sparams.numDisocclusionBoostSamples = g_RtxdiBridgeConst.restirDI.spatialResamplingParams.numDisocclusionBoostSamples;
        sparams.targetHistoryLength = g_RtxdiBridgeConst.restirDI.temporalResamplingParams.maxHistoryLength;
        sparams.biasCorrectionMode = g_RtxdiBridgeConst.restirDI.spatialResamplingParams.spatialBiasCorrection;
        sparams.samplingRadius =  g_RtxdiBridgeConst.restirDI.spatialResamplingParams.spatialSamplingRadius;
        sparams.depthThreshold =  g_RtxdiBridgeConst.restirDI.spatialResamplingParams.spatialDepthThreshold;
        sparams.normalThreshold = g_RtxdiBridgeConst.restirDI.spatialResamplingParams.spatialNormalThreshold;
        sparams.enableMaterialSimilarityTest = true;
        sparams.discountNaiveSamples = g_RtxdiBridgeConst.restirDI.spatialResamplingParams.discountNaiveSamples;

        spatialReservoir = RTXDI_DISpatialResampling(
            pixelPosition,
            surface, 
            centerSample, 
            rng, 
            g_RtxdiBridgeConst.runtimeParams,
            g_RtxdiBridgeConst.restirDI.reservoirBufferParams,
            sparams, 
            lightSample);
    }

    RTXDI_StoreDIReservoir(spatialReservoir, g_RtxdiBridgeConst.restirDI.reservoirBufferParams, GlobalIndex, g_RtxdiBridgeConst.restirDI.bufferIndices.spatialResamplingOutputBufferIndex);

    // useful for debugging!
    DebugContext debug;
    debug.Init(pixelPosition, g_Const.debug, u_FeedbackBuffer, u_DebugLinesBuffer, u_DebugDeltaPathTree, u_DeltaPathSearchStack, u_DebugVizOutput);

    switch (g_Const.debug.debugViewType)
    {
    case (int)DebugViewType::ReSTIRDISpatialOutput:

        // Load the light stored in reservoir and sample it
        uint lightIdx = RTXDI_GetDIReservoirLightIndex(spatialReservoir);
        float2 lightUV = RTXDI_GetDIReservoirSampleUV(spatialReservoir);
        RAB_LightInfo lightInfo = RAB_LoadLightInfo(lightIdx, false);
        RAB_LightSample lightSample = RAB_SamplePolymorphicLight(lightInfo, surface, lightUV);

        float3 Li = lightSample.radiance * RTXDI_GetDIReservoirInvPdf(spatialReservoir) / lightSample.solidAnglePdf;
        debug.DrawDebugViz(pixelPosition, float4(Li, 1));
        break;
    }
}