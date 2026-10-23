/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Ray-Traced Reflections Pass
// Per-pixel GGX reflection rays traced from GBuffer surfaces, with diffuse shading at their hits.
// Shares u_SSRResult with SSR, but writes BRDF-weighted specular for direct reading by the combine
// pass. No SSR blur or split-sum weighting; denoising is left to DLSS-RR after lighting is combined.

#define ActiveBSDF StandardBSDF

#include "../PathTracer/Config.h" // must always be included first

#include "../PathTracer/PathTracerTypes.hlsli"

#include "../Bindings/ShaderResourceBindings.hlsli"
#include "../Bindings/GBufferBindings.hlsli"
#include "IntroLightBindings.hlsli"
#include "../Bindings/SamplerBindings.hlsli"

#include "IntroBridge.hlsli"

#include "FastHitShaders.hlsl"

[shader("raygeneration")]
void RAYGEN_ENTRY()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 screenSize = DispatchRaysDimensions().xy;

    if (any(pixel >= screenSize))
        return;

    float depth = u_Depth[pixel];
    PackedGBufferSurface packedSurface;
    packedSurface.LoadFromRenderTargets(pixel);
    float metal = packedSurface.GetMetalnes();
    float roughness = packedSurface.GetRoughness();

    if (depth <= 0)
    {
        u_SSRResult[pixel] = float4(0, 0, 0, 0);
        return;
    }

    // Reconstruct world position and read GBuffer
    float3 worldPos = ReconstructWorldPos(pixel, depth, g_Const.view.pixelOffset, g_Const.view.viewportSizeInv, g_Const.view.matClipToWorldNoOffset);
    float3 worldNormal = packedSurface.GetSpecNormal();

    float3 cameraPos = g_Const.ptConsts.camera.PosW.xyz;
    float3 viewDir = normalize(cameraPos - worldPos);
    
    RNGState rng;
    rng.initRngSeed(pixel, g_Const.ptConsts.frameIndex);
    float2 u = rng.getNextUniformRand2();
    float3 halfVector = SampleVNDF(u, viewDir, worldNormal, roughness * roughness);
    float3 reflectDir = reflect(-viewDir, halfVector);
    float invPdfVNDF = VNDF_SpecularBRDF(roughness, worldNormal, halfVector, viewDir, reflectDir);
    
    // Skip grazing angles where reflections degenerate
    float NdotV = saturate(dot(worldNormal, viewDir));
    if (NdotV < 0.01)
    {
        u_SSRResult[pixel] = float4(0, 0, 0, 0);
        return;
    }

    RayDesc rayDesc;
    rayDesc.Origin = worldPos + worldNormal * 0.01;
    rayDesc.Direction = reflectDir;
    rayDesc.TMin = 0.0;
    rayDesc.TMax = 1000.0;

    PayloadSimplified payload;
    payload.hitDistance = -1;
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xff, 0, 1, 0, rayDesc, payload);

    float3 reflectionColor = 0;
    float confidence = 0;

    if (payload.hitDistance >= 0)
    {
        float3 hitPos = rayDesc.Origin + reflectDir * payload.hitDistance;

        bool countEmissive = roughness < 0.25; // Explicit light sampling doesn't contribute much to smooth reflections.
        // A full implementation would MIS between the two strategies, but this is good enough for a temporary step.
        reflectionColor = ComputeHitLighting(hitPos, payload.baseColor, payload.normal, payload.shaderId, -reflectDir, rng, countEmissive, true);
        confidence = 1.0;
    }
    else // Reflection ray missed: fall back to the environment so misses reflect the sky rather than going black.
    {
        EnvMap envMap = EnvMap::make(t_EnvironmentMap, s_EnvironmentMapSampler, g_Const.envMapSceneParams);
        reflectionColor = envMap.Eval(reflectDir, 0.f);
        confidence = 1.0;
    }
    reflectionColor *= invPdfVNDF;

    // Full specular BRDF Fresnel/F0 (incl. the metal tint). The deferred combine reads u_SSRResult directly
    // (no split-sum term), so the whole specular BRDF must be applied here - matching IndirectPathTracer.
    float HdV = saturate(dot(viewDir, halfVector));
    float3 F0 = lerp(0.04, packedSurface.GetBaseColor(), metal);
    reflectionColor *= Schlick_Fresnel(F0, HdV);

    reflectionColor = min(reflectionColor, 65504);

    u_SSRResult[pixel] = float4(reflectionColor, confidence);
}

[shader("miss")]
void MISS_ENTRY(inout PayloadSimplified payload : SV_RayPayload)
{
}
