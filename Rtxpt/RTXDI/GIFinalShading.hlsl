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

bool ReSTIRGIFinalContribution(const uint2 pixelPosition, const RAB_Surface surface, out float3 diffuseContribution, out float3 specularContribution, out float hitDistance)
{
    RTXDI_GIReservoir finalReservoir = RTXDI_LoadGIReservoir(
        g_RtxdiBridgeConst.restirGI.reservoirBufferParams,
        pixelPosition, 
        g_RtxdiBridgeConst.restirGI.bufferIndices.finalShadingInputBufferIndex);

    diffuseContribution = 0.0.xxx;
    specularContribution = 0.0.xxx;
    hitDistance = 0;

    if (!RTXDI_IsValidGIReservoir(finalReservoir))
        return false;

    const float4 secondaryPositionNormal = u_SecondarySurfacePositionNormal[pixelPosition];
    RTXDI_GIReservoir initialReservoir = RTXDI_MakeGIReservoir(
        secondaryPositionNormal.xyz,
        octToNdirUnorm32(asuint(secondaryPositionNormal.w)),
        u_SecondarySurfaceRadiance[pixelPosition].xyz,
        /* samplePdf = */ 1.0);


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
    
    float3 diffuseBrdf, specularBrdf;
    surface.Eval(L, diffuseBrdf, specularBrdf);
    
    diffuseContribution = 0; 
    specularContribution = 0;

    if (g_RtxdiBridgeConst.restirGI.finalShadingParams.enableFinalMIS)
    {
        float3 L0 = initialReservoir.position - surface.GetPosW();
        float hitDistance0 = length(L0);
        L0 /= hitDistance0;

        float3 diffuseBrdf0, specularBrdf0;
        surface.Eval(L0, diffuseBrdf0, specularBrdf0);

        float3 roughDiffuseBrdf, roughSpecularBrdf;
        float3 roughDiffuseBrdf0, roughSpecularBrdf0;
        surface.EvalRoughnessClamp(kMISRoughness, L, roughDiffuseBrdf, roughSpecularBrdf);
        surface.EvalRoughnessClamp(kMISRoughness, L0, roughDiffuseBrdf0, roughSpecularBrdf0);

        const float finalWeight = 1.0 - GetMISWeight(roughDiffuseBrdf + roughSpecularBrdf, diffuseBrdf + specularBrdf);
        const float initialWeight = GetMISWeight(roughDiffuseBrdf0 + roughSpecularBrdf0, diffuseBrdf0 + specularBrdf0);

        const float3 initialRadiance = initialReservoir.radiance * initialReservoir.weightSum;

        diffuseContribution = diffuseBrdf * finalRadiance * finalWeight
                            + diffuseBrdf0 * initialRadiance * initialWeight;

        specularContribution = specularBrdf * finalRadiance * finalWeight
                             + specularBrdf0 * initialRadiance * initialWeight;

        hitDistance = hitDistance * finalWeight
                    + hitDistance0 * initialWeight;
    }
    else
    {
        diffuseContribution = diffuseBrdf * finalRadiance;
        specularContribution = specularBrdf * finalRadiance;
    }

    if (any(isinf(diffuseContribution)) || any(isnan(diffuseContribution)) ||
        any(isinf(specularContribution)) || any(isnan(specularContribution)))
    {
        diffuseContribution = 0;
        specularContribution = 0;
    }

    DebugContext debug;
    debug.Init( pixelPosition, g_Const.debug, u_FeedbackBuffer, u_DebugLinesBuffer, u_DebugDeltaPathTree, u_DeltaPathSearchStack, u_DebugVizOutput );

    switch(g_Const.debug.debugViewType)
    {
    case (int)DebugViewType::SecondarySurfacePosition:
        debug.DrawDebugViz(pixelPosition, float4(abs(finalReservoir.position) * 0.1, 1.0));
        break;
    case (int)DebugViewType::SecondarySurfaceRadiance:
        debug.DrawDebugViz(pixelPosition, float4(finalReservoir.radiance, 1.0));
        break;
    case (int)DebugViewType::ReSTIRGIOutput:
        debug.DrawDebugViz(pixelPosition, float4(diffuseContribution + specularContribution, 1.0));
        break;
    }

    return any(diffuseContribution+specularContribution)>0;
}

#ifndef USE_AS_INCLUDE
[numthreads(RTXDI_SCREEN_SPACE_GROUP_SIZE, RTXDI_SCREEN_SPACE_GROUP_SIZE, 1)]
void main(uint2 GlobalIndex : SV_DispatchThreadID, uint2 LocalIndex : SV_GroupThreadID, uint2 GroupIdx : SV_GroupID)
{
   const uint2 pixelPosition = RTXDI_ReservoirPosToPixelPos(GlobalIndex, g_RtxdiBridgeConst.runtimeParams.activeCheckerboardField);

    RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

    if (!RAB_IsSurfaceValid(surface))
        return;

    float3 diffuseContribution, specularContribution;
    float hitDistance;

    if (ReSTIRGIFinalContribution(pixelPosition, surface, diffuseContribution, specularContribution, hitDistance))
    {
        if (g_Const.ptConsts.denoisingEnabled)
        {
            StablePlanesContext stablePlanes = StablePlanesContext::make(pixelPosition, u_StablePlanesHeader, u_StablePlanesBuffer, u_StableRadiance, u_SecondarySurfaceRadiance, g_Const.ptConsts);

            uint dominantStablePlaneIndex = stablePlanes.LoadDominantIndexCenter();
            uint address = stablePlanes.PixelToAddress(pixelPosition, dominantStablePlaneIndex);

            float4 denoiserDiffRadianceHitDist;
            float4 denoiserSpecRadianceHitDist;
            UnpackTwoFp32ToFp16(u_StablePlanesBuffer[address].DenoiserPackedRadianceHitDist, denoiserDiffRadianceHitDist, denoiserSpecRadianceHitDist);

            denoiserDiffRadianceHitDist = StablePlaneCombineWithHitTCompensation(denoiserDiffRadianceHitDist, diffuseContribution, hitDistance);
            denoiserSpecRadianceHitDist = StablePlaneCombineWithHitTCompensation(denoiserSpecRadianceHitDist, specularContribution, hitDistance);

            u_StablePlanesBuffer[address].DenoiserPackedRadianceHitDist = PackTwoFp32ToFp16(denoiserDiffRadianceHitDist, denoiserSpecRadianceHitDist);
        }
        else
        {
            u_Output[pixelPosition] += float4(diffuseContribution + specularContribution, 0);
        }
    }
}
#endif // #ifndef USE_AS_INCLUDE