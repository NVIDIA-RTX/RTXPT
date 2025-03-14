/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#define USE_AS_INCLUDE 1
#include "DIFinalShading.hlsl"
#include "GIFinalShading.hlsl"
#undef USE_AS_INCLUDE 

[numthreads(RTXDI_SCREEN_SPACE_GROUP_SIZE, RTXDI_SCREEN_SPACE_GROUP_SIZE, 1)]
void main(uint2 dispatchThreadID : SV_DispatchThreadID)
{
    const uint2 reservoirPos = dispatchThreadID.xy;
    const uint2 pixelPos = RTXDI_ReservoirPosToPixelPos(reservoirPos, g_RtxdiBridgeConst.runtimeParams.activeCheckerboardField);

    RAB_Surface surface = RAB_GetGBufferSurface(pixelPos, false);

    if (!RAB_IsSurfaceValid(surface))
        return;

    float3 diffuseContributionDI, specularContributionDI, diffuseContributionGI, specularContributionGI;
    float hitDistanceDI, hitDistanceGI;

    bool hasGI = ReSTIRGIFinalContribution(pixelPos, surface, diffuseContributionGI, specularContributionGI, hitDistanceGI);
    bool hasDI = ReSTIRDIFinalContribution(reservoirPos, pixelPos, surface, diffuseContributionDI, specularContributionDI, hitDistanceDI);

    if (hasGI || hasDI)
    {
        if (g_Const.ptConsts.denoisingEnabled)
        {
            StablePlanesContext stablePlanes = StablePlanesContext::make(pixelPos, u_StablePlanesHeader, u_StablePlanesBuffer, u_StableRadiance, u_SecondarySurfaceRadiance, g_Const.ptConsts);

            uint dominantStablePlaneIndex = stablePlanes.LoadDominantIndexCenter();
            uint address = stablePlanes.PixelToAddress(pixelPos, dominantStablePlaneIndex);

            float4 denoiserDiffRadianceHitDist;
            float4 denoiserSpecRadianceHitDist;
            UnpackTwoFp32ToFp16(u_StablePlanesBuffer[address].DenoiserPackedRadianceHitDist, denoiserDiffRadianceHitDist, denoiserSpecRadianceHitDist);

            if (hasDI)
            {
                denoiserDiffRadianceHitDist = StablePlaneCombineWithHitTCompensation(denoiserDiffRadianceHitDist, diffuseContributionDI, hitDistanceDI);
                denoiserSpecRadianceHitDist = StablePlaneCombineWithHitTCompensation(denoiserSpecRadianceHitDist, specularContributionDI, hitDistanceDI);
            }
            if (hasGI)
            {
                denoiserDiffRadianceHitDist = StablePlaneCombineWithHitTCompensation(denoiserDiffRadianceHitDist, diffuseContributionGI, hitDistanceGI);
                denoiserSpecRadianceHitDist = StablePlaneCombineWithHitTCompensation(denoiserSpecRadianceHitDist, specularContributionGI, hitDistanceGI);
            }

            u_StablePlanesBuffer[address].DenoiserPackedRadianceHitDist = PackTwoFp32ToFp16(denoiserDiffRadianceHitDist, denoiserSpecRadianceHitDist);
        }
        else
        {
            float3 combined = 0;
            if (hasDI)
                combined += diffuseContributionDI + specularContributionDI;
            if (hasGI)
                combined += diffuseContributionGI + specularContributionGI;
            u_Output[pixelPos] += float4(combined, 0);
        }
    }

}	