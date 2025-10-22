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

#ifndef NON_PATH_TRACING_PASS
#define NON_PATH_TRACING_PASS 1
#endif

#include "../RTXDI/RtxdiApplicationBridge.hlsli"

#include "../../External/Rtxdi/Include/Rtxdi/GI/Reservoir.hlsli"

static const float kMaxBrdfValue = 1e4;
static const float kMISRoughness = 0.2;

float GetMISWeight(float3 roughBrdf, float3 trueBrdf)
{
    roughBrdf = clamp(roughBrdf, 1e-4, kMaxBrdfValue);
    trueBrdf = clamp(trueBrdf, 0, kMaxBrdfValue);

    const float initWeight = saturate(calcLuminance(trueBrdf) / calcLuminance(trueBrdf + roughBrdf));
    return initWeight * initWeight * initWeight;
}

bool ReSTIRGIFinalContribution(const uint2 pixelPosition, const RAB_Surface surface, out float4 radianceAndSpecAvg, out float hitDistance)
{
    RTXDI_GIReservoir finalReservoir = RTXDI_LoadGIReservoir(
        g_RtxdiBridgeConst.restirGI.reservoirBufferParams,
        pixelPosition, 
        g_RtxdiBridgeConst.restirGI.bufferIndices.finalShadingInputBufferIndex);

    radianceAndSpecAvg = 0;
    hitDistance = 0;

    const float4 secondaryPositionNormal = u_SecondarySurfacePositionNormal[pixelPosition];
    const float3 secondaryPositionRadiance = u_SecondarySurfaceRadiance[pixelPosition].xyz;
    const float  primaryScatterPdf = u_SecondarySurfaceRadiance[pixelPosition].w;

    #if 0
    {
        DebugContext debug;
        debug.Init( g_Const.debug, u_FeedbackBuffer, u_DebugLinesBuffer, u_DebugDeltaPathTree, u_DeltaPathSearchStack, u_DebugVizOutput );
        //debug.DrawDebugViz(pixelPosition, float4(frac(secondaryPositionNormal.rgb), 1.0));
        if (primaryScatterPdf==0)
            debug.DrawDebugViz(pixelPosition, float4(1, 0.5, 0.0, 1.0));
        else
            debug.DrawDebugViz(pixelPosition, float4(frac(secondaryPositionRadiance.rgb), 1.0));
        return false;
    }
    #endif

    if (!RTXDI_IsValidGIReservoir(finalReservoir))
        return false;

    RTXDI_GIReservoir initialReservoir = RTXDI_EmptyGIReservoir();
    if (primaryScatterPdf>0)
        initialReservoir = RTXDI_MakeGIReservoir( secondaryPositionNormal.xyz, octToNdirUnorm32(asuint(secondaryPositionNormal.w)),
                                                    secondaryPositionRadiance.xyz, primaryScatterPdf );


    float3 L = finalReservoir.position - surface.GetPosW();
    // Calculate hitDistance here in case visibility is not enabled
    hitDistance = length(L);
    L /= hitDistance;

    float3 finalRadiance = finalReservoir.radiance * finalReservoir.weightSum;

    if (g_RtxdiBridgeConst.restirGI.finalShadingParams.enableFinalVisibility)
    {
        // TODO: support partial visibility... if that is applicable with this material model.
        const RayDesc ray = setupVisibilityRay(surface, finalReservoir.position, g_RtxdiBridgeConst.rayEpsilon);

        bool visible = GetFinalVisibility(SceneBVH, ray);

        if (!visible)
            finalRadiance = 0;
    }
    
    float4 bsdfThp = surface.Eval(L);
    
    float3 attenuatedRadiance = 0; 

    if (g_RtxdiBridgeConst.restirGI.finalShadingParams.enableFinalMIS)
    {
        float3 L0 = initialReservoir.position - surface.GetPosW();
        float hitDistance0 = length(L0);
        L0 /= hitDistance0;

        float4 bsdfThp0 = surface.Eval(L0);

        float4 roughBrdf    = surface.EvalRoughnessClamp(kMISRoughness, L);
        float4 roughBrdf0   = surface.EvalRoughnessClamp(kMISRoughness, L0);

        const float finalWeight = 1.0 - GetMISWeight(roughBrdf.rgb, bsdfThp.rgb);
        const float initialWeight = GetMISWeight(roughBrdf.rgb, bsdfThp0.rgb);

        const float3 initialRadiance = initialReservoir.radiance * initialReservoir.weightSum;

        attenuatedRadiance = bsdfThp.rgb * finalRadiance * finalWeight + bsdfThp0.rgb * initialRadiance * initialWeight;

        hitDistance = hitDistance * finalWeight     + hitDistance0 * initialWeight;
    }
    else
    {
        attenuatedRadiance = bsdfThp.rgb * finalRadiance;
    }

    if ( any(isinf(attenuatedRadiance)) || any(isnan(attenuatedRadiance)) )
    {
        attenuatedRadiance = 0;
    }

    radianceAndSpecAvg.rgb = attenuatedRadiance;
    radianceAndSpecAvg.a = Average(finalRadiance) * bsdfThp.a;

    DebugContext debug;
    debug.Init( g_Const.debug, u_FeedbackBuffer, u_DebugLinesBuffer, u_DebugDeltaPathTree, u_DeltaPathSearchStack, u_DebugVizOutput );

    switch(g_Const.debug.debugViewType)
    {
    case (int)DebugViewType::SecondarySurfacePosition:
        debug.DrawDebugViz(pixelPosition, float4(abs(finalReservoir.position) * 0.1, 1.0));
        break;
    case (int)DebugViewType::SecondarySurfaceRadiance:
        debug.DrawDebugViz(pixelPosition, float4(finalReservoir.radiance, 1.0));
        break;
    case (int)DebugViewType::ReSTIRGIOutput:
        debug.DrawDebugViz(pixelPosition, float4(radianceAndSpecAvg.rgb, 1.0));
        break;
    }

    return any(radianceAndSpecAvg.rgb)>0;
}

#ifndef USE_AS_INCLUDE
[numthreads(RTXDI_SCREEN_SPACE_GROUP_SIZE, RTXDI_SCREEN_SPACE_GROUP_SIZE, 1)]
void main(uint2 GlobalIndex : SV_DispatchThreadID, uint2 LocalIndex : SV_GroupThreadID, uint2 GroupIdx : SV_GroupID)
{
    const uint2 pixelPos = RTXDI_ReservoirPosToPixelPos(GlobalIndex, g_RtxdiBridgeConst.runtimeParams.activeCheckerboardField);

    RAB_Surface surface = RAB_GetGBufferSurface(pixelPos, false);

    if (!RAB_IsSurfaceValid(surface))
        return;

    float4 newRadianceAndSpecAvg;
    float newSpecHitT;

    if (ReSTIRGIFinalContribution(pixelPos, surface, newRadianceAndSpecAvg, newSpecHitT))
    {
        if (g_Const.ptConsts.denoisingEnabled)
        {
            uint address = StablePlanesContext::ComputeDominantAddress(pixelPos, u_StablePlanesHeader, u_StablePlanesBuffer, u_StableRadiance, g_Const.ptConsts);

            float4 radiance     = Fp16ToFp32(u_StablePlanesBuffer[address].PackedNoisyRadianceAndSpecAvg);
            float specHitDist   = u_StablePlanesBuffer[address].NoisyRadianceSpecHitDist;
            u_StablePlanesBuffer[address].NoisyRadianceSpecHitDist = AccumulateHitT( radiance.a, specHitDist, newRadianceAndSpecAvg.a, newSpecHitT );
            u_StablePlanesBuffer[address].PackedNoisyRadianceAndSpecAvg = Fp32ToFp16(radiance.rgba+newRadianceAndSpecAvg.rgba);
        }
        else
        {
            u_OutputColor[pixelPos] += float4(newRadianceAndSpecAvg.rgb, 0);
        }
    }
}
#endif // #ifndef USE_AS_INCLUDE