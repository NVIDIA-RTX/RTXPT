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

// Only enable the boiling filter for RayQuery (compute shader) mode because it requires shared memory
#if USE_RAY_QUERY
#define RTXDI_ENABLE_BOILING_FILTER
#define RTXDI_BOILING_FILTER_GROUP_SIZE RTXDI_SCREEN_SPACE_GROUP_SIZE
#endif

#include "RtxdiApplicationBridge.hlsli"
#include <Rtxdi/Di/BoilingFilter.hlsli>
#include <Rtxdi/Di/SpatialResampling.hlsli>
#include <Rtxdi/Di/SpatioTemporalResampling.hlsli>
#include <Rtxdi/Di/TemporalResampling.hlsli>

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

    uint2 pixelPosition = RTXDI_ReservoirPosToPixelPos(GlobalIndex, g_RtxdiBridgeConst.runtimeParams.activeCheckerboardField);

    RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPosition, 2);

    RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

    //Adds a small random offet to the motion vector
    // We current do not have a method to check for complex surfaces 
   bool usePermutationSampling = false;
   //if (g_ResamplingConst.enablePermutationSampling)
   //{
   //    // Permutation sampling makes more noise on thin, high-detail objects.
   //    usePermutationSampling = !IsComplexSurface(pixelPosition, surface);
   //}

    RTXDI_DIReservoir temporalReservoir = RTXDI_EmptyDIReservoir();
    int2 temporalSamplePixelPos = -1;
    RAB_LightSample lightSample = (RAB_LightSample)0;

    if (RAB_IsSurfaceValid(surface))
    {
        RTXDI_DIReservoir currentSample = RTXDI_LoadDIReservoir(
            g_RtxdiBridgeConst.restirDI.reservoirBufferParams,
            GlobalIndex, 
            g_RtxdiBridgeConst.restirDI.bufferIndices.initialSamplingOutputBufferIndex);

        float3 motionVector = u_MotionVectors[pixelPosition].xyz;
        motionVector = convertMotionVectorToPixelSpace(g_Const.view, g_Const.previousView, pixelPosition, motionVector);

        RTXDI_DITemporalResamplingParameters tparams;
        tparams.screenSpaceMotion = motionVector;
        tparams.sourceBufferIndex = g_RtxdiBridgeConst.restirDI.bufferIndices.temporalResamplingInputBufferIndex;
        tparams.maxHistoryLength = g_RtxdiBridgeConst.restirDI.temporalResamplingParams.maxHistoryLength;
        tparams.biasCorrectionMode = g_RtxdiBridgeConst.restirDI.temporalResamplingParams.temporalBiasCorrection;
        tparams.depthThreshold = g_RtxdiBridgeConst.restirDI.temporalResamplingParams.temporalDepthThreshold;
        tparams.normalThreshold = g_RtxdiBridgeConst.restirDI.temporalResamplingParams.temporalNormalThreshold;
        tparams.enableVisibilityShortcut = g_RtxdiBridgeConst.restirDI.temporalResamplingParams.discardInvisibleSamples;
        tparams.enablePermutationSampling = usePermutationSampling;
        tparams.uniformRandomNumber = g_RtxdiBridgeConst.restirDI.temporalResamplingParams.uniformRandomNumber;
        
        temporalReservoir = RTXDI_DITemporalResampling(
            pixelPosition,
            surface,
            currentSample,
            rng,
            g_RtxdiBridgeConst.runtimeParams,
            g_RtxdiBridgeConst.restirDI.reservoirBufferParams,
            tparams,
            temporalSamplePixelPos,
            lightSample);
    }

#ifdef RTXDI_ENABLE_BOILING_FILTER
    if (g_RtxdiBridgeConst.restirDI.temporalResamplingParams.enableBoilingFilter)
    {
        RTXDI_BoilingFilter(LocalIndex, g_RtxdiBridgeConst.restirDI.temporalResamplingParams.boilingFilterStrength, temporalReservoir);
    }
#endif
    
    RTXDI_StoreDIReservoir(temporalReservoir, g_RtxdiBridgeConst.restirDI.reservoirBufferParams, GlobalIndex, g_RtxdiBridgeConst.restirDI.bufferIndices.temporalResamplingOutputBufferIndex);

    // useful for debugging!
    DebugContext debug;
    debug.Init(pixelPosition, g_Const.debug, u_FeedbackBuffer, u_DebugLinesBuffer, u_DebugDeltaPathTree, u_DeltaPathSearchStack, u_DebugVizOutput);

    switch (g_Const.debug.debugViewType)
    {
    case (int)DebugViewType::ReSTIRDITemporalOutput:

        // Load the light stored in reservoir and sample it
        uint lightIdx = RTXDI_GetDIReservoirLightIndex(temporalReservoir);
        float2 lightUV = RTXDI_GetDIReservoirSampleUV(temporalReservoir);
        RAB_LightInfo lightInfo = RAB_LoadLightInfo(lightIdx, false);
        lightSample = RAB_SamplePolymorphicLight(lightInfo, surface, lightUV);
        
        float3 Li = lightSample.radiance * RTXDI_GetDIReservoirInvPdf(temporalReservoir) / lightSample.solidAnglePdf;
        debug.DrawDebugViz(pixelPosition, float4(Li, 1));
        break;
    }
}