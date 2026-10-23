/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __INTRO_RESTIR_DI_SHADING_HLSLI__
#define __INTRO_RESTIR_DI_SHADING_HLSLI__

#include "IntroSampleShadowRay.hlsli"
#include "IntroReSTIRDICommon.hlsli"

// Shade using the RIS-selected light sample (same path as Step 7 Streaming RIS).
// Do not re-sample the light from the reservoir; the stored UV/light index can disagree
// with the streamed sample (notably directional lights), which drops contributions.
float3 IntroReSTIRDI_ComputeDirectLightingWithSample(
    IntroRTXDISurface surface,
    float depth,
    RTXDI_DIReservoir reservoir,
    RAB_LightSample selectedSample)
{
    if (!RTXDI_IsValidDIReservoir(reservoir) || selectedSample.solidAnglePdf <= 0)
        return 0;

    float3 lightDir;
    float lightDistance;
    RAB_GetLightDirDistance(surface, selectedSample, lightDir, lightDistance);

    float NdotL = saturate(dot(surface.normal, lightDir));
    if (NdotL <= 0)
        return 0;

    float rayOffset = max(kIntroRTXDIRayEpsilon, kIntroRTXDIRayEpsilon * (0.001 / max(depth, 1e-6)));
    if (!TraceVisibilityRay(surface.worldPos + surface.normal * rayOffset, lightDir, max(rayOffset, lightDistance - rayOffset)))
        return 0;

    float invPdf = RTXDI_GetDIReservoirInvPdf(reservoir);
    float3 incidentRadiance = (selectedSample.radiance / selectedSample.solidAnglePdf) * invPdf;
    return IntroRTXDIEvaluateBRDF(surface, lightDir, NdotL) * incidentRadiance * NdotL;
}

#endif // __INTRO_RESTIR_DI_SHADING_HLSLI__
