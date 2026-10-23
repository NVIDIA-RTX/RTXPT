/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Direct lighting pass: stochastically samples both local analytic lights (point/spot/sphere
// from t_Lights/t_LightsEx, the LightsBaker output) and infinite analytic lights (directional
// disks from the Intro RTXDI bake's InfiniteRegion of t_IntroRTXDILightsBase). Each pixel takes
// one sample from the union: with probability pInfinite a directional light is picked uniformly
// from the InfiniteRegion, otherwise a local light is picked uniformly. The contribution is
// divided by the per-light pick probability. This removes bias from light selection; local
// center sampling and the small-angle directional disk still approximate the light shapes.
// Output is written to u_DirectLighting for the deferred combine pass.

#define IGNORE_RAY_PAYLOAD 1
#define SHARC_UPDATE 1

static const uint kStochasticRISCandidates = 8;

#define ActiveBSDF StandardBSDF

#include "../Bindings/ShaderResourceBindings.hlsli"
#include "../Bindings/GBufferBindings.hlsli"
#include "../Bindings/SamplerBindings.hlsli"
#include "IntroBridge.hlsli"
#include "IntroSampleCommon.hlsli"
#include "IntroSampleShadowRay.hlsli"

RWStructuredBuffer<uint64_t> u_SHaRCHashEntries : register(u90);
RWStructuredBuffer<SharcAccumulationData> u_SHaRCAccumulation : register(u91);
RWStructuredBuffer<SharcPackedData> u_SHaRCResolved : register(u92);

// Intro RTXDI light buffers (Base only; directional sampling never needs Extended). These are
// declared here with the same registers used by IntroRTXDIBridge.hlsli so this pass can read
// directly from the InfiniteRegion of the Intro RTXDI bake without going through RTXDI itself.
StructuredBuffer<PolymorphicLightInfo> t_IntroRTXDILightsBase : register(t90);

// Reconstruct world position from pixel coordinates and NDC depth
float3 ReconstructWorldPosition(uint2 pixelPos, float depth)
{
    float2 uv = (float2(pixelPos) + 0.5 - g_Const.view.pixelOffset) * g_Const.view.viewportSizeInv;
    float4 clipPos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    float4 worldPos = mul(clipPos, g_Const.view.matClipToWorldNoOffset);
    return worldPos.xyz / worldPos.w;
}

float LightTargetPdf(uint lightIndex, float3 surfaceWorldPos, float3 normal)
{
    PolymorphicLightInfo lightBase = t_Lights[lightIndex];
    PolymorphicLightInfoEx lightEx = t_LightsEx[lightIndex];

    float3 lightPos = lightBase.Center;
    float3 L = lightPos - surfaceWorldPos;
    float dist2 = dot(L, L);
    float dist = sqrt(dist2);
    L /= dist;
    
    float NdotL = saturate(dot(normal, L));
    if (NdotL == 0)
        return 0;
    
    // Spot cone attenuation
    float spotAtten = EvaluateSpotAttenuation(lightBase, lightEx, surfaceWorldPos);
    if (spotAtten <= 0)
        return 0;

    float radiance = EvaluatePolymorphicLightIrradianceScalar(lightBase, dist2, spotAtten);
    
    return radiance * NdotL;
}

float3 EvaluateLocalLight(float viewDist, uint lightIndex, float NdotV, float3 surfaceWorldPos, float3 normal, float3 viewDir, float3 diffuseColor, float3 specularF0, float alpha2)
{
    PolymorphicLightInfo lightBase = t_Lights[lightIndex];
    PolymorphicLightInfoEx lightEx = t_LightsEx[lightIndex];

    float3 lightPos = lightBase.Center;
    float3 L = lightPos - surfaceWorldPos;
    float dist2 = dot(L, L);
    float dist = sqrt(dist2);
    L /= dist;

    float NdotL = saturate(dot(normal, L));
    if (NdotL == 0)
        return 0;

    // Trace a shadow ray
    if(!TraceVisibilityRay(surfaceWorldPos + normal * max(1e-4, 1e-4 * viewDist), normalize(L), dist*0.995))
    {
        return 0;
    }

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

// Evaluate one directional-light NEE sample. Mirrors PolymorphicLight.hlsli's
// DirectionalLight::CalcSample math, written inline here to avoid pulling in the full
// PolymorphicLight.hlsli machinery from this lean direct-lighting pass.
//
// Returns the per-sample contribution (BRDF * radiance * NdotL * solidAngle).
// Solid-angle pdf cancels with `radiance = color * irradiance / solidAngle` so the
// final scalar matches what RTXDI sees on the StreamingRIS path.
float3 EvaluateDirectionalLight(float viewDist, uint lightIndex, float NdotV,
    float3 surfaceWorldPos, float3 normal, float3 viewDir,
    float3 diffuseColor, float3 specularF0, float alpha2,
    float2 diskRand)
{
    PolymorphicLightInfo lightBase = t_IntroRTXDILightsBase[lightIndex];

    float3 lightDir = OctToNDirUnorm32(lightBase.Direction1);          // travel direction (light -> world)
    float halfAngle = f16tof32(lightBase.Scalars);
    float solidAngle = f16tof32(lightBase.Scalars >> 16);
    float sinHalfAngle, cosHalfAngle;
    sincos(halfAngle, sinHalfAngle, cosHalfAngle);
    float3 radiance = UnpackLightColor(lightBase);

    // Uniform planar disk sampling approximates uniform solid angle for a small sun disk.
    float2 diskSample = SampleDiskUniform(diskRand);
    float3 t, b;
    BranchlessONB(lightDir, t, b);
    float3 distantSample = lightDir + t * diskSample.x * sinHalfAngle + b * diskSample.y * sinHalfAngle;
    float3 L = -distantSample;                                          // toLight (approximately unit length)

    float NdotL = saturate(dot(normal, L));
    if (NdotL == 0)
        return 0;

    // Long shadow ray to "infinity"; directional lights have no physical position.
    if (!TraceVisibilityRay(surfaceWorldPos + normal * max(1e-4, 1e-4 * viewDist), normalize(L), 1e6))
        return 0;

    float3 specular = CookTorranceBRDF(L, normal, viewDir, specularF0, alpha2, NdotL, NdotV);
    float3 diffuse = diffuseColor * K_1_PI;

    // Single-sample MC estimator with pdf = 1/solidAngle on the disk:
    //   L_o ≈ (BRDF) * radiance * NdotL / pdf = (BRDF) * radiance * NdotL * solidAngle
    return (diffuse + specular) * radiance * NdotL * solidAngle;
}

SharcParameters GetSharcParameters()
{
    SharcParameters params = GetDefaultSharcParams(g_Const.ptConsts.camera.PosW);

    params.hashGridData.capacity = g_MiniConst.params.x;
    params.hashGridData.hashEntriesBuffer = u_SHaRCHashEntries;
    params.accumulationBuffer = u_SHaRCAccumulation;
    params.resolvedBuffer = u_SHaRCResolved;
    
    return params;
}

// Estimate direct lighting by choosing one local or directional analytic light.
float3 EvaluateDirectLighting(float viewDist, float3 worldPos, float3 viewDir, PackedGBufferSurface surface, RNGState rng)
{
    float3 baseColor = surface.GetBaseColor();
    float metal = surface.GetMetalnes();
    float3 normal = normalize(surface.GetSpecNormal());
    float roughness = surface.GetRoughness();

    uint analyticStart = g_Const.lightControl.AnalyticLightOffset;
    uint analyticEnd = analyticStart + g_Const.lightControl.AnalyticLightCount;

    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metal);
    float3 diffuseColor = baseColor * (1 - metal) * (1 - F0); // Very rough energy conservation
    float NdotV = saturate(dot(normal, viewDir));

    float r2 = max(roughness * roughness, 1e-5);
    float alpha2 = r2 * r2;

    uint numLocalLights = analyticEnd - analyticStart;
    uint infiniteStart  = g_Const.lightControl.InfiniteRegionFirstLightIndex;
    uint numInfiniteLights = g_Const.lightControl.InfiniteRegionNumLights;

    if (numLocalLights == 0 && numInfiniteLights == 0)
        return 0;

    // === Single-sample stochastic direct lighting with a 2-group probability split ===
    // The "local" group (point/spot/sphere from LightsBaker) and the "infinite" group
    // (directional disks from the Intro RTXDI bake) are sampled with a fixed probability
    // mixture: the infinite group gets pInfinite of the budget and locals get the rest.
    // Within the chosen group the light is picked uniformly. A 50/50 split keeps directional
    // lights sampled frequently even when there are many locals; it does not account for power.
    float pInfinite = (numInfiniteLights > 0) ? ((numLocalLights > 0) ? 0.5 : 1.0) : 0.0;

    float u = rng.getNextUniformRand();
    float3 totalLight = 0;

    if (u < pInfinite)
    {
        // Re-stratify u into [0, 1) for picking within the infinite group.
        float v = u / pInfinite;
        uint k = min(uint(numInfiniteLights * v), numInfiniteLights - 1);
        uint dirIdx = infiniteStart + k;

        // Two-component disk sample for the directional cone.
        float2 diskRand = float2(rng.getNextUniformRand(), rng.getNextUniformRand());
        float3 contrib = EvaluateDirectionalLight(viewDist, dirIdx, NdotV, worldPos, normal, viewDir,
            diffuseColor, F0, alpha2, diskRand);

        // Estimator: contrib / (pInfinite * (1 / numInfinite)) = contrib * numInfinite / pInfinite.
        totalLight = contrib * (float(numInfiniteLights) / pInfinite);
    }
    else
    {
        float pLocal = 1.0 - pInfinite;
        float v = (u - pInfinite) / pLocal;
        uint candidateLight = analyticStart + min(uint(numLocalLights * v), numLocalLights - 1);
        float3 radiance = EvaluateLocalLight(viewDist, candidateLight, NdotV, worldPos, normal, viewDir, diffuseColor, F0, alpha2);

        // Estimator: radiance / (pLocal * (1 / numLocal)) = radiance * numLocal / pLocal.
        totalLight = radiance * (float(numLocalLights) / pLocal);
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

    RNGState rng;
    rng.initRngSeed(pixelPos, g_Const.ptConsts.frameIndex * 37);

    float3 directLight = 0;
    if (depth > 0)
    {
        float3 viewDir = -normalize(Bridge::computeCameraRay(pixelPos).dir);
        float3 worldPos = ReconstructWorldPos(pixelPos, depth, g_Const.view.pixelOffset, g_Const.view.viewportSizeInv, g_Const.view.matClipToWorldNoOffset);

        // Use the reconstructed position rather than assuming a fixed camera near plane.
        float viewDist = length(worldPos - g_Const.ptConsts.camera.PosW);
        directLight = EvaluateDirectLighting(viewDist, worldPos, viewDir, surface, rng);
    }

    u_DirectLighting[pixelPos] = float4(directLight, 0);
}
