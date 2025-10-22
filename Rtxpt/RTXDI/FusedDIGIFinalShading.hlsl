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

    float4 radianceAndSpecAvgDI, radianceAndSpecAvgGI;
    float hitDistanceDI, hitDistanceGI;

    bool hasGI = ReSTIRGIFinalContribution(pixelPos, surface, radianceAndSpecAvgGI, hitDistanceGI);
    bool hasDI = ReSTIRDIFinalContribution(reservoirPos, pixelPos, surface, radianceAndSpecAvgDI, hitDistanceDI);

    if (hasGI || hasDI)
    {
        if (g_Const.ptConsts.denoisingEnabled)
        {
            uint address = StablePlanesContext::ComputeDominantAddress(pixelPos, u_StablePlanesHeader, u_StablePlanesBuffer, u_StableRadiance, g_Const.ptConsts);
            float4 radiance     = Fp16ToFp32(u_StablePlanesBuffer[address].PackedNoisyRadianceAndSpecAvg);
            float specHitDist   = u_StablePlanesBuffer[address].NoisyRadianceSpecHitDist;

            if (hasDI)
            {
                specHitDist = AccumulateHitT( radiance.a, specHitDist, radianceAndSpecAvgDI.a, hitDistanceDI );
                radiance   += radianceAndSpecAvgDI;
            }
            if (hasGI)
            {
                specHitDist = AccumulateHitT( radiance.a, specHitDist, radianceAndSpecAvgGI.a, hitDistanceGI );
                radiance   += radianceAndSpecAvgGI;
            }

            u_StablePlanesBuffer[address].NoisyRadianceSpecHitDist = specHitDist;
            u_StablePlanesBuffer[address].PackedNoisyRadianceAndSpecAvg = Fp32ToFp16( radiance );
        }
        else
        {
            float3 combined = 0;
            if (hasDI)
                combined += radianceAndSpecAvgDI.rgb;
            if (hasGI)
                combined += radianceAndSpecAvgGI.rgb;
            u_OutputColor[pixelPos] += float4(combined, 0);
        }
    }
}	