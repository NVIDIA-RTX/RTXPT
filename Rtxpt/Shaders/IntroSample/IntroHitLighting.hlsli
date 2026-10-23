/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __INTRO_HIT_LIGHTING_HLSLI__
#define __INTRO_HIT_LIGHTING_HLSLI__

// Lighting-only helpers for the Intro secondary-hit passes (RTGI, RT reflections, the SHaRC cache
// update and the unified indirect path tracer). Included by FastHitShaders.hlsl; traversal and
// payload handling stay in the calling passes.
//
// The including shader must already provide (order matters - include this after these):
//   - SceneBVH                              (for the inline shadow rays; via SceneBindings.hlsli)
//   - g_Const / g_MiniConst                 (light-region indices + prefiltered-env mip count)
//   - RNGState, K_1_PI                       (IntroSampleCommon.hlsli / math headers)
//   - UnpackLightColor, EvaluateSpotAttenuation, EvaluatePolymorphicLightIrradiance (IntroSampleCommon)
//   - OctToNDirUnorm32, SampleDiskUniform, BranchlessONB                            (math/geometry)
//   - t_EnvIrradianceCube, t_PrefilteredEnvCube, t_BRDFLUT, s_EnvironmentMapSampler (env bindings)
//   - ShaderIdEmissive                       (GBufferBindings.hlsli)
//   - PolymorphicLightInfo / PolymorphicLightInfoEx (PolymorphicLight.h via IntroLightBindings.hlsli)

#include "IntroSampleShadowRay.hlsli" // TraceVisibilityRay (needs SceneBVH, declared by the includer above)

// ---------------------------------------------------------------------------------------------------
// Shared single-sample analytic direct lighting for the Intro secondary-hit passes. Lights are read
// from the Intro RTXDI light buffers:
//   - local analytic lights (point / spot / sphere) from the analytic sub-range of the LocalRegion
//   - directional (infinite disk) lights from the InfiniteRegion
// One light is drawn uniformly from each region and shadow-tested. The contribution is divided by the
// uniform pick pdf so the result is an unbiased estimate of each region's total contribution. The hit
// is shaded as a diffuse (Lambertian) surface.
// ---------------------------------------------------------------------------------------------------

// Same buffers/bindings the RTXDI/ReSTIR + direct-lighting passes read (see IntroRTXDIBridge.hlsli).
StructuredBuffer<PolymorphicLightInfo>   t_IntroRTXDILightsBase : register(t90);
StructuredBuffer<PolymorphicLightInfoEx> t_IntroRTXDILightsEx   : register(t91);

// Evaluate one-sample analytic direct lighting at a diffuse secondary hit.
//   hitPos          - shaded surface point (used to build light vectors / distances)
//   shadowRayOrigin - normal-offset origin for shadow rays (caller controls the bias magnitude)
//   normal          - shading normal at the hit
//   diffuseAlbedo   - Lambertian albedo; the diffuse BRDF (albedo / pi) is applied here
//   rng             - consumes 1 sample for the local pick, then 2 (disk) + 1 (pick) for the directional
// Returns the outgoing diffuse radiance from the sampled analytic lights.
float3 IntroEvaluateHitDirectLighting(float3 hitPos, float3 shadowRayOrigin, float3 normal, float3 diffuseAlbedo, inout RNGState rng)
{
    // Accumulates irradiance * NdotL (pre-BRDF), pick-pdf corrected, over the sampled lights.
    float3 irradiance = 0;

    // --- One local analytic light (point / spot / sphere) from the LocalRegion analytic range ---
    uint numLocal = g_Const.lightControl.LocalRegionNumAnalyticLights;
    if (numLocal > 0)
    {
        uint k = min(uint(rng.getNextUniformRand() * numLocal), numLocal - 1);
        uint lightIndex = g_Const.lightControl.LocalRegionFirstLightIndex + k;

        PolymorphicLightInfo lightBase = t_IntroRTXDILightsBase[lightIndex];
        PolymorphicLightInfoEx lightEx = t_IntroRTXDILightsEx[lightIndex];

        float3 L = lightBase.Center - hitPos;
        float dist2 = dot(L, L);
        float dist = sqrt(dist2);
        L /= dist;

        float NdotL = saturate(dot(normal, L));
        if (NdotL > 0 && TraceVisibilityRay(shadowRayOrigin, L, dist * 0.995))
        {
            float spotAtten = EvaluateSpotAttenuation(lightBase, lightEx, hitPos);
            if (spotAtten > 0)
            {
                // Uniform pick over numLocal => divide by pdf = 1 / numLocal.
                irradiance += EvaluatePolymorphicLightIrradiance(lightBase, dist2, spotAtten) * NdotL * numLocal;
            }
        }
    }

    // --- One directional (infinite) light from the InfiniteRegion ---
    uint numInfinite = g_Const.lightControl.InfiniteRegionNumLights;
    if (numInfinite > 0)
    {
        uint k = min(uint(rng.getNextUniformRand() * numInfinite), numInfinite - 1);
        uint lightIndex = g_Const.lightControl.InfiniteRegionFirstLightIndex + k;

        PolymorphicLightInfo lightBase = t_IntroRTXDILightsBase[lightIndex];

        float3 lightDir = OctToNDirUnorm32(lightBase.Direction1); // light -> world travel direction
        float halfAngle = f16tof32(lightBase.Scalars);
        float solidAngle = f16tof32(lightBase.Scalars >> 16);
        float sinHalfAngle = sin(halfAngle);
        float3 radiance = UnpackLightColor(lightBase);

        // Sample the light's solid-angle disk uniformly (mirrors the Step 4 directional NEE sample).
        float2 diskSample = SampleDiskUniform(rng.getNextUniformRand2());
        float3 t, b;
        BranchlessONB(lightDir, t, b);
        float3 distantSample = lightDir + (t * diskSample.x + b * diskSample.y) * sinHalfAngle;
        float3 L = normalize(-distantSample); // toLight

        float NdotL = saturate(dot(normal, L));
        if (NdotL > 0 && TraceVisibilityRay(shadowRayOrigin, L, 1e6))
        {
            // Disk pdf = 1 / solidAngle cancels radiance = color / solidAngle; uniform pick over
            // numInfinite => multiply by numInfinite (divide by pick pdf = 1 / numInfinite).
            irradiance += radiance * NdotL * solidAngle * numInfinite;
        }
    }

    return diffuseAlbedo * K_1_PI * irradiance;
}

// Shade a secondary (GI / reflection) hit as Lambertian: sampled analytic direct lighting,
// optional emission, and an optional cosine-weighted environment sample with visibility.
// Shared by RTGI, RT reflections, cache training and the unified indirect path tracer.
//   viewDir - direction from the hit back toward the ray origin (i.e. -rayDirection)
//   envMip  - mip level the environment term (see sampleEnvMap) is evaluated at; a fixed prefilter
//             level trades environment detail for firefly suppression (see Step 2's "Misses and
//             fireflies"). Defaults to 5, RTGI's own value; other callers don't override it.
float3 ComputeHitLighting(float3 hitPos, float3 baseColor, float3 normal, uint shaderId, float3 viewDir, inout RNGState rng, bool includeEmission = false, bool sampleEnvMap = false, float envMip = 5.0f)
{
    float3 hitRadiance = includeEmission && (shaderId == ShaderIdEmissive) ? baseColor : 0;

    // Hit direct lighting from the shared RTXDI-buffer sampler (one local + one directional light,
    // shadow-tested). Offset the shadow-ray origin by distance from camera, like the primary GI ray.
    float distFromCamera = length(hitPos - g_Const.ptConsts.camera.PosW);
    float3 shadowRayOrigin = hitPos + normal * max(1e-3, 1e-4 * distFromCamera);
    hitRadiance += IntroEvaluateHitDirectLighting(hitPos, shadowRayOrigin, normal, saturate(baseColor), rng);

    // Simplified IBL lighting with AO on hits
    if (sampleEnvMap)
    {
        float3 rayDir = SampleHemisphereCosineWeighted(rng.getNextUniformRand2(), normal);
        if (TraceVisibilityRay(shadowRayOrigin, rayDir, 1e6))
        {
            EnvMap envMap = EnvMap::make(t_EnvironmentMap, s_EnvironmentMapSampler, g_Const.envMapSceneParams);
            float3 envRadiance = envMap.Eval(rayDir, envMip);
            hitRadiance += saturate(baseColor) * envRadiance;
        }
    }

    return hitRadiance;
}

#endif // __INTRO_HIT_LIGHTING_HLSLI__
