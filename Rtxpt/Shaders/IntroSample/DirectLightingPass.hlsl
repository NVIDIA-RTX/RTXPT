/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Direct lighting pass: evaluates every analytic light - local (point/spot/sphere) and directional
// (infinite disks) - in a simple sequential loop and writes the result to u_DirectLighting for the
// deferred combine pass. All lights are read from a single source: the Intro RTXDI light buffers
// (t_IntroRTXDILightsBase / t_IntroRTXDILightsEx). Local analytic lights live in the analytic subset
// of the LocalRegion; directional lights live in the InfiniteRegion. Using the one RTXDI-baked
// buffer for every light (rather than mixing in the LightsBaker t_Lights buffer for local lights)
// keeps this pass consistent with the other RTXDI-based passes.
//
// Local (point/spot/sphere) lights are evaluated unshadowed, as in Intro Steps 0-3. Directional
// lights are evaluated along their central direction (no stochastic disk sampling) but DO trace a
// shadow ray, so cast shadows from the sun/directional lights show up here. Step 4's
// StochasticLightingPass.hlsl switches to one shadow-tested light sample: local centers from
// LightsBaker and directional disks from the Intro bake.

#define IGNORE_RAY_PAYLOAD 1

#include "../Bindings/ShaderResourceBindings.hlsli"
#include "../Bindings/GBufferBindings.hlsli"
#include "../Bindings/SamplerBindings.hlsli"
#include "IntroBridge.hlsli"
#include "IntroSampleCommon.hlsli"
#include "IntroSampleShadowRay.hlsli"

// Intro RTXDI light buffers. The infinite-first bake lays each ping-pong half out as
// [infinite (directional) | local analytic | emissive triangles]; this pass reads the directional
// and local-analytic sub-ranges (region indices come from g_Const.lightControl). Same bindings
// (t90/t91) the RTXDI/ReSTIR passes use.
StructuredBuffer<PolymorphicLightInfo>   t_IntroRTXDILightsBase : register(t90);
StructuredBuffer<PolymorphicLightInfoEx> t_IntroRTXDILightsEx   : register(t91);

float3 EvaluateLocalLight(uint lightIndex, float NdotV, float3 surfaceWorldPos, float3 normal, float3 viewDir, float3 diffuseColor, float3 specularF0, float alpha2)
{
    PolymorphicLightInfo lightBase = t_IntroRTXDILightsBase[lightIndex];
    PolymorphicLightInfoEx lightEx = t_IntroRTXDILightsEx[lightIndex];

    float3 lightPos = lightBase.Center;
    float3 L = lightPos - surfaceWorldPos;
    float dist2 = dot(L, L);
    float dist = sqrt(dist2);
    L /= dist;

    float NdotL = dot(normal, L);
    if (NdotL <= 0)
        return 0;

    NdotL = saturate(NdotL);

    // Spot cone attenuation
    float spotAtten = EvaluateSpotAttenuation(lightBase, lightEx, surfaceWorldPos);
    if (spotAtten <= 0)
        return 0;

    float3 radiance = EvaluatePolymorphicLightIrradiance(lightBase, dist2, spotAtten);

    // Specular BRDF (denominator clamped to avoid division by zero)
    float3 specular = CookTorranceBRDF(L, normal, viewDir, specularF0, alpha2, NdotL, NdotV);

    // Diffuse (Lambertian, energy-conserved)
    float3 diffuse = diffuseColor * K_1_PI;

    return (diffuse + specular) * radiance * NdotL;
}

// Evaluate one directional (infinite) light along its central direction - no disk sampling, but
// with a shadow ray so directional lights (e.g. the sun) cast shadows. Mirrors the analytic form of
// StochasticLightingPass.hlsl's EvaluateDirectionalLight but with L fixed to the light's central
// direction, so the disk-sampling Monte Carlo estimator collapses to its expected value:
// (BRDF) * radiance * NdotL * solidAngle. viewDist is the linear distance from the camera to the
// shaded point, used to scale the shadow-ray origin offset so the bias grows with distance.
float3 EvaluateDirectionalLight(uint lightIndex, float NdotV, float3 surfaceWorldPos, float3 normal, float3 viewDir, float3 diffuseColor, float3 specularF0, float alpha2, float viewDist)
{
    PolymorphicLightInfo lightBase = t_IntroRTXDILightsBase[lightIndex];

    float3 lightDir = OctToNDirUnorm32(lightBase.Direction1); // light -> world travel direction
    float solidAngle = f16tof32(lightBase.Scalars >> 16);
    float3 radiance = UnpackLightColor(lightBase);

    float3 L = -lightDir; // toLight (central direction)

    float NdotL = dot(normal, L);
    if (NdotL <= 0)
        return 0;

    // Trace a shadow ray toward "infinity" (directional lights have no position). Offset the origin
    // along the normal by a distance-scaled bias to avoid self-intersection / shadow acne.
    float shadowBias = max(1e-4, 1e-4 * viewDist);
    //float3 origin = 
    if (!TraceVisibilityRay(surfaceWorldPos + normal * shadowBias, normalize(L), 1e6))
    {
        return 0;
    }

    NdotL = saturate(NdotL);

    float3 specular = CookTorranceBRDF(L, normal, viewDir, specularF0, alpha2, NdotL, NdotV);
    float3 diffuse = diffuseColor * K_1_PI;

    return (diffuse + specular) * radiance * NdotL * solidAngle;
}

// Evaluate direct lighting from every analytic light: local (point/spot/sphere) then directional.
// All indices reference the single Intro RTXDI light buffer.
float3 EvaluateDirectLighting(float3 worldPos, float3 normal, float3 viewDir, float3 baseColor, float metallic, float roughness, float viewDist)
{
    // Local-analytic sub-range of the LocalRegion (point/spot/sphere only, excluding the trailing
    // emissive triangles that share the region).
    uint localStart = g_Const.lightControl.LocalRegionFirstLightIndex;
    uint localEnd = localStart + g_Const.lightControl.LocalRegionNumAnalyticLights;

    // Directional lights occupy the whole InfiniteRegion.
    uint infiniteStart = g_Const.lightControl.InfiniteRegionFirstLightIndex;
    uint infiniteEnd = infiniteStart + g_Const.lightControl.InfiniteRegionNumLights;

    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);
    float3 diffuseColor = baseColor * (1 - metallic) * (1 - F0); // Very rough energy conservation
    float NdotV = saturate(dot(normal, viewDir));

    float r2 = max(roughness * roughness, 1e-5);
    float alpha2 = r2 * r2;

    float3 totalLight = 0;

    // Local analytic lights (point / spot / sphere).
    for (uint i = localStart; i < localEnd; i++)
    {
        totalLight += EvaluateLocalLight(i, NdotV, worldPos, normal, viewDir, diffuseColor, F0, alpha2);
    }

    // Directional (infinite) lights.
    for (uint j = infiniteStart; j < infiniteEnd; j++)
    {
        totalLight += EvaluateDirectionalLight(j, NdotV, worldPos, normal, viewDir, diffuseColor, F0, alpha2, viewDist);
    }

    return totalLight;
}

[numthreads(NUM_COMPUTE_THREADS_PER_DIM, NUM_COMPUTE_THREADS_PER_DIM, 1)]
void main(uint2 dispatchThreadID : SV_DispatchThreadID)
{
    const uint2 pixelPos = dispatchThreadID.xy;
    if (any(pixelPos >= uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight)))
        return;

    // Load GBuffer
    PackedGBufferSurface surface;
    surface.LoadFromRenderTargets(pixelPos);
    float depth = u_Depth[pixelPos];

    float3 directLight = 0;
    if (depth > 0)
    {
        float3 baseColor = surface.GetBaseColor();
        float metal = surface.GetMetalnes();
        float3 normal = surface.GetSpecNormal();
        float roughness = surface.GetRoughness();

        float3 rayDir = Bridge::computeCameraRay(pixelPos).dir;
        float3 worldPos = ReconstructWorldPos(pixelPos, depth, g_Const.view.pixelOffset, g_Const.view.viewportSizeInv, g_Const.view.matClipToWorldNoOffset);

        // Linear distance from camera to the shaded point, used to scale the directional shadow-ray bias.
        float viewDist = length(worldPos - g_Const.ptConsts.camera.PosW);

        directLight = EvaluateDirectLighting(worldPos, normal, -rayDir, baseColor, metal, roughness, viewDist);
    }

    u_DirectLighting[pixelPos] = float4(directLight, 0);
}
