/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Ray-Traced Global Illumination (RTGI)
// Traces a single cosine-weighted hemisphere ray per pixel, shading the hit
// with simple IBL to produce one-bounce diffuse GI. Output (rgb=irradiance)
// is blended into diffuse lighting by the DeferredLightingCombine pass.
// Denoising is handled by DLSS-RR.

#if RTGI_SHARC_ENABLED
#define SHARC_QUERY 1
#endif

#define ActiveBSDF StandardBSDF

#include "../PathTracer/Config.h" // must always be included first

#include "../PathTracer/PathTracerTypes.hlsli"

#include "../Bindings/ShaderResourceBindings.hlsli"
#include "../Bindings/GBufferBindings.hlsli"
#include "IntroLightBindings.hlsli"
#include "../Bindings/SamplerBindings.hlsli"

#include "IntroBridge.hlsli"
#include "../PathTracer/Utils/Math/HashUtils.hlsli"
#include "../PathTracer/Utils/Math/MathHelpers.hlsli"
#include "../PathTracer/Utils/Geometry.hlsli"

#include "FastHitShaders.hlsl" // provides IntroEvaluateHitDirectLighting + TraceVisibilityRay

#if RTGI_SHARC_ENABLED

RWStructuredBuffer<uint64_t>                u_SHaRCHashEntries   : register(u90);
RWStructuredBuffer<SharcAccumulationData>   u_SHaRCAccumulation  : register(u91);
RWStructuredBuffer<SharcPackedData>         u_SHaRCResolved      : register(u92);

#endif

[shader("raygeneration")]
void RAYGEN_ENTRY()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 screenSize = DispatchRaysDimensions().xy;

    if (any(pixel >= screenSize))
        return;

    float depth = u_Depth[pixel];
    PackedGBufferSurface packedSurface;
    packedSurface.packedSpecNormal = u_GBufferSpecNormal32[pixel];
    packedSurface.packedMaterialInfo = u_GBufferMaterialInfo[pixel];

    // Making these non-dependent reads by placing them before the depth check.
    float3 worldNormal = packedSurface.GetSpecNormal(); // Same as diffuse normal for simplicity
    uint shaderId = packedSurface.GetShaderId();
    
    if (depth <= 0)
    {
        u_RTGIOutput[pixel] = float4(0, 0, 0, 0);
        return;
    }

    float3 worldPos = ReconstructWorldPos(pixel, depth, g_Const.view.pixelOffset, g_Const.view.viewportSizeInv, g_Const.view.matClipToWorldNoOffset);

    // Fixed mip the environment is sampled at, both on this ray's miss and (via ComputeHitLighting)
    // its hit's own environment term. UI-configurable; see "Misses and fireflies" in the Step 2 docs
    // for the accuracy/noise trade-off. Set on the CPU, clamped to the prefiltered cube's max mip.
    float envMip = (float)g_MiniConst.params.z;

    // Cosine-weighted hemisphere sample in tangent space (+z = normal)
    RNGState rng;
    rng.initRngSeed(pixel, g_Const.ptConsts.frameIndex);
    float3 rayDir = SampleHemisphereCosineWeighted(rng.getNextUniformRand2(), worldNormal);

    // Transform to world space
    float3 cameraRelativePos = worldPos.xyz - g_Const.ptConsts.camera.PosW;
    float hitDistance = length(cameraRelativePos);
    float3 viewDir = cameraRelativePos / hitDistance;
    
    uint rayFlags = RAY_FLAG_NONE;
    float3 irradiance = 1;
    if (shaderId == ShaderIdGlassGGX)
    {
        rayDir = viewDir;
        float F0 = 0.08; // Twice the common dielectric reflectance to account for both the in and out bounces.
        worldNormal = -worldNormal; // Inverts the ray start offset
        float F = Schlick_Fresnel(F0, saturate(dot(viewDir, worldNormal)));
        
        rayFlags = RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_CULL_NON_OPAQUE;
        irradiance *= 1-F; // Minus two reflection events. TODO: Better fresnel
        
        rayDir = normalize(rayDir + 0.5 * worldNormal);
        worldPos += worldNormal * 1e-3; // Skip the glass
    }

    RayDesc rayDesc;
    rayDesc.Origin = worldPos + worldNormal * max(1e-3, 1e-4 * hitDistance);
    rayDesc.Direction = rayDir;
    rayDesc.TMin = 1e-3;
    rayDesc.TMax = 1000.0;

    PayloadSimplified payload;
    payload.hitDistance = -1;
    TraceRay(SceneBVH, rayFlags, 0xff, 0, 1, 0, rayDesc, payload);
        
    if (payload.hitDistance >= 0)
    {
#if RTGI_SHARC_ENABLED
        SharcParameters sharcParams;
        sharcParams.hashGridParameters.cameraPosition = g_Const.ptConsts.camera.PosW;
        sharcParams.hashGridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
        sharcParams.hashGridParameters.sceneScale = SHARC_SCENE_SCALE;
        sharcParams.hashGridParameters.levelBias = SHARC_GRID_LEVEL_BIAS;
        sharcParams.hashGridData.capacity = g_MiniConst.params.x;
        sharcParams.hashGridData.hashEntriesBuffer = u_SHaRCHashEntries;
        sharcParams.radianceScale = 1e3f;
        sharcParams.accumulationBuffer = u_SHaRCAccumulation;
        sharcParams.resolvedBuffer = u_SHaRCResolved;

        SharcHitData sharcHitData;
        sharcHitData.positionWorld = rayDesc.Origin + rayDir * payload.hitDistance;
        sharcHitData.normalWorld = payload.normal;

        float3 cachedRadiance;
        if (payload.hitDistance >= 0.5 && shaderId != ShaderIdGlassGGX && SharcGetCachedRadiance(sharcParams, sharcHitData, cachedRadiance, false))
        {
            irradiance *= cachedRadiance;
        }
        else
#endif // RTGI_SHARC_ENABLED
        {
            // Only count emission at the GI hit when direct lighting isn't already sampling
            // emissive triangles explicitly (params.y, set on the CPU). When explicit emissive
            // NEE is active (e.g. ReSTIR DI), letting RTGI's first hit add emission double counts
            // the "primary surface -> emitter" transport path.
            bool hitContributeEmissive = g_MiniConst.params.y != 0;
            // Outgoing radiance at the GI hit = IBL approximation + one-sample analytic direct
            // lighting (one local + one directional light, shadow-tested). In the SHaRC variant this
            // branch is only the cache-miss fallback, so no double counting with the cached radiance.
            float3 hitPos = rayDesc.Origin + rayDir * payload.hitDistance;
            float3 hitRadiance = ComputeHitLighting(hitPos, payload.baseColor, payload.normal, payload.shaderId, -rayDir, rng, hitContributeEmissive, true, envMip);
            irradiance *= hitRadiance;
        }
    }
    else
    {
        EnvMap envMap = EnvMap::make(t_EnvironmentMap, s_EnvironmentMapSampler, g_Const.envMapSceneParams);
        irradiance *= envMap.Eval(rayDir, envMip); // Sample a higher mip to reduce fireflies on high contrast HDR content.
    }

    irradiance = min(irradiance, float3(65504, 65504, 65504));

    u_RTGIOutput[pixel] = float4(irradiance, 1);
}

[shader("miss")]
void MISS_ENTRY(inout PayloadSimplified payload : SV_RayPayload)
{
}
