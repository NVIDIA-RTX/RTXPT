/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Intro direct-lighting pass that uses RTXDI's initial sampling code as a
// Streaming RIS implementation. Samples local analytic lights (point/sphere/spot)
// and infinite analytic lights (directional disk lights) in two parallel reservoirs,
// combines them, and performs one final visibility ray.
//
// Directional lights are sampled analytically here; unlike the full sample, the intro
// pipeline does not bake them into the environment cubemap.

#define IGNORE_RAY_PAYLOAD 1

// This pass samples analytic local lights only (it reads the analytic-only sub-region), so the
// triangle light type is never selected here. Compile it out of the polymorphic light dispatch.
#define INTRO_RTXDI_ENABLE_TRIANGLE_LIGHTS 0

#include "../Bindings/ShaderResourceBindings.hlsli"
#include "../Bindings/GBufferBindings.hlsli"
#include "../Bindings/SamplerBindings.hlsli"
#include "IntroBridge.hlsli"
#include "IntroSampleCommon.hlsli"
#include "IntroSampleShadowRay.hlsli"
#include "IntroRTXDIBridge.hlsli"

#include <Rtxdi/DI/InitialSampling.hlsli>

float3 EvaluateRTXDIDirectLighting(float depth, IntroRTXDISurface surface, RNGState rng)
{
    // Analytic locals only (excludes baked emissive triangles): uniform RIS over a small analytic
    // set keeps spot/point/sphere lights well-sampled. Triangles are handled by ReSTIR DI (Step 8).
    RTXDI_LightBufferRegion localRegion = MakeIntroRTXDILocalLightRegionAnalyticOnly();
    RTXDI_LightBufferRegion infiniteRegion = MakeIntroRTXDIInfiniteLightRegion();
    if (!surface.valid || (localRegion.numLights == 0 && infiniteRegion.numLights == 0))
        return 0;

    RTXDI_DIInitialSamplingParameters initialSamplingParams = MakeIntroRTXDIInitialSamplingParameters();

    RNGState coherentRng = rng;
    RTXDI_InitialSamplingMisData misData = RTXDI_ComputeInitialSamplingMisData(initialSamplingParams);

    // Sample local (positional) analytic lights.
    RAB_LightSample localSample = RAB_EmptyLightSample();
    RTXDI_DIReservoir localReservoir = RTXDI_SampleLocalLights(
        rng,
        coherentRng,
        surface,
        initialSamplingParams,
        misData,
        initialSamplingParams.localLightSamplingMode,
        localRegion,
        localSample);

    // Sample infinite (directional) analytic lights into a parallel reservoir.
    RAB_LightSample infiniteSample = RAB_EmptyLightSample();
    RTXDI_DIReservoir infiniteReservoir = RTXDI_SampleInfiniteLights(
        rng,
        surface,
        initialSamplingParams.numInfiniteLightSamples,
        infiniteRegion,
        infiniteSample);

    // Combine the two reservoirs the same way RTXDI_SampleLightsForSurface does.
    RTXDI_DIReservoir reservoir = RTXDI_EmptyDIReservoir();
    RTXDI_CombineDIReservoirs(reservoir, localReservoir, 0.5, localReservoir.targetPdf);
    bool selectInfinite = RTXDI_CombineDIReservoirs(reservoir, infiniteReservoir, RTXDI_GetNextRandom(rng), infiniteReservoir.targetPdf);
    RTXDI_FinalizeResampling(reservoir, 1.0, 1.0);
    reservoir.M = 1;

    // HLSL's ternary doesn't accept struct results; use an if-else.
    RAB_LightSample selectedSample = localSample;
    if (selectInfinite)
        selectedSample = infiniteSample;

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

[numthreads(NUM_COMPUTE_THREADS_PER_DIM, NUM_COMPUTE_THREADS_PER_DIM, 1)]
void main(uint2 dispatchThreadID : SV_DispatchThreadID)
{
    const uint2 pixelPos = dispatchThreadID.xy;
    if (any(pixelPos >= uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight)))
        return;

    PackedGBufferSurface packedSurface;
    packedSurface.LoadFromRenderTargets(pixelPos);
    float depth = u_Depth[pixelPos];

    RNGState rng;
    rng.initRngSeed(pixelPos, g_Const.ptConsts.frameIndex * 37);

    float3 directLight = 0;
    if (depth > 0)
    {
        IntroRTXDISurface surface = BuildIntroRTXDISurface(pixelPos, depth, packedSurface);
        directLight = EvaluateRTXDIDirectLighting(depth, surface, rng);
    }

    u_DirectLighting[pixelPos] = float4(directLight, 0);
}
