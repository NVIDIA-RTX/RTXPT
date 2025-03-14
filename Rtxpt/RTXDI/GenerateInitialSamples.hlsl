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

#include <Rtxdi/Di/InitialSampling.hlsli>

#if USE_RAY_QUERY
[numthreads(RTXDI_SCREEN_SPACE_GROUP_SIZE, RTXDI_SCREEN_SPACE_GROUP_SIZE, 1)]
void main(uint2 dispatchThreadID : SV_DispatchThreadID)
#else
[shader("raygeneration")]
void RayGen()
#endif
{
#if !USE_RAY_QUERY
    uint2 dispatchThreadID = DispatchRaysIndex().xy;
#endif

   uint2 pixelPosition = RTXDI_ReservoirPosToPixelPos(dispatchThreadID, g_RtxdiBridgeConst.runtimeParams.activeCheckerboardField);

    RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPosition, 1);
    RAB_RandomSamplerState tileRng = RAB_InitRandomSampler(pixelPosition / RTXDI_TILE_SIZE_IN_PIXELS, 1);

    
    RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);
    if (!RAB_IsSurfaceValid(surface))
        return;
  
    RTXDI_SampleParameters sampleParams = RTXDI_InitSampleParameters(
        g_RtxdiBridgeConst.restirDI.initialSamplingParams.numPrimaryLocalLightSamples,
        g_RtxdiBridgeConst.restirDI.initialSamplingParams.numPrimaryInfiniteLightSamples,
        g_RtxdiBridgeConst.restirDI.initialSamplingParams.numPrimaryEnvironmentSamples,
        g_RtxdiBridgeConst.restirDI.initialSamplingParams.numPrimaryBrdfSamples,
        g_RtxdiBridgeConst.restirDI.initialSamplingParams.brdfCutoff,
        0.001f);

    RAB_LightSample lightSample;
    RTXDI_DIReservoir reservoir = RTXDI_SampleLightsForSurface(
        rng, 
        tileRng, 
        surface, 
        sampleParams, 
        g_RtxdiBridgeConst.lightBufferParams,
        g_RtxdiBridgeConst.restirDI.initialSamplingParams.localLightSamplingMode,
#ifdef RTXDI_ENABLE_PRESAMPLING
        g_RtxdiBridgeConst.localLightsRISBufferSegmentParams, 
        g_RtxdiBridgeConst.environmentLightRISBufferSegmentParams,
#if RTXDI_REGIR_MODE != RTXDI_REGIR_MODE_DISABLED
        g_RtxdiBridgeConst.regir,
#endif
#endif
        lightSample);

    DebugContext debug;
    debug.Init(pixelPosition, g_Const.debug, u_FeedbackBuffer, u_DebugLinesBuffer, u_DebugDeltaPathTree, u_DeltaPathSearchStack, u_DebugVizOutput);

    if (g_RtxdiBridgeConst.restirDI.initialSamplingParams.enableInitialVisibility && RTXDI_IsValidDIReservoir(reservoir))
    {
        if (!RAB_GetConservativeVisibility(surface, lightSample))
        {
            RTXDI_StoreVisibilityInDIReservoir(reservoir, 0, true);
#if ENABLE_DEBUG_RTXDI_VIZUALISATION
            if (debug.IsDebugPixel())
            {
               debug.DrawLine(lightSample.position, RAB_GetNewRayOrigin(surface), float3(1, 0, 0), float3(1, 0, 0));
            }
#endif
        }
#if ENABLE_DEBUG_RTXDI_VIZUALISATION
        else //debug only
        {
            if (debug.IsDebugPixel())
            {
                debug.DrawLine(lightSample.position, RAB_GetNewRayOrigin(surface), float3(0, 1, 0), float3(0, 1, 0));
            }
        }
#endif
    }
    
    RTXDI_StoreDIReservoir(reservoir, g_RtxdiBridgeConst.restirDI.reservoirBufferParams, dispatchThreadID, g_RtxdiBridgeConst.restirDI.bufferIndices.initialSamplingOutputBufferIndex);

   
    switch (g_Const.debug.debugViewType)
    {
    case (int)DebugViewType::ReSTIRDIInitialOutput:
        // Load the light stored in reservoir and sample it
        uint lightIdx = RTXDI_GetDIReservoirLightIndex(reservoir);
        float2 lightUV = RTXDI_GetDIReservoirSampleUV(reservoir);
        RAB_LightInfo lightInfo = RAB_LoadLightInfo(lightIdx, false);
        lightSample = RAB_SamplePolymorphicLight(lightInfo, surface, lightUV);

        float3 Li = lightSample.radiance * RTXDI_GetDIReservoirInvPdf(reservoir) / lightSample.solidAnglePdf;
        debug.DrawDebugViz(pixelPosition, float4(Li, 1));
        break;
    }
}