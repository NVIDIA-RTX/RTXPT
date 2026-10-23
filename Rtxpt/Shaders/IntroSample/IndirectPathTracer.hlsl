/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Unified indirect lighting: select one primary BSDF lobe and trace its continuation path.
#if IPT_SHARC_ENABLED
#define SHARC_QUERY 1
#endif
#define ActiveBSDF StandardBSDF

#include "../PathTracer/Config.h"
#include "../PathTracer/PathTracerTypes.hlsli"
#include "../Bindings/ShaderResourceBindings.hlsli"
#include "../Bindings/GBufferBindings.hlsli"
#include "IntroLightBindings.hlsli"
#include "../Bindings/SamplerBindings.hlsli"
#include "IntroBridge.hlsli"
#include "FastHitShaders.hlsl"

#if IPT_SHARC_ENABLED
RWStructuredBuffer<uint64_t> u_SHaRCHashEntries : register(u90);
RWStructuredBuffer<SharcAccumulationData> u_SHaRCAccumulation : register(u91);
RWStructuredBuffer<SharcPackedData> u_SHaRCResolved : register(u92);
#endif

[shader("raygeneration")]
void RAYGEN_ENTRY()
{
    uint2 pixel = DispatchRaysIndex().xy;
    float depth = u_Depth[pixel];
    if (depth <= 0)
    {
        u_SSRResult[pixel] = 0;
        u_RTGIOutput[pixel] = 0;
        return;
    }

    PackedGBufferSurface surface;
    surface.LoadFromRenderTargets(pixel);
    float metal = surface.GetMetalnes();
    float roughness = surface.GetRoughness();
    float3 normal = surface.GetSpecNormal();
    uint shaderId = surface.GetShaderId();
    float3 worldPos = ReconstructWorldPos(pixel, depth, g_Const.view.pixelOffset,
        g_Const.view.viewportSizeInv, g_Const.view.matClipToWorldNoOffset);
    float cameraDistance = length(worldPos - g_Const.ptConsts.camera.PosW);
    float3 viewDir = (g_Const.ptConsts.camera.PosW - worldPos) / cameraDistance;

    RNGState rng;
    rng.initRngSeed(pixel, g_Const.ptConsts.frameIndex);
    float specProbability = lerp(0.2, 1, metal);
    bool specularLobe = rng.getNextUniformRand() < specProbability;
    float3 F0 = lerp(0.04, surface.GetBaseColor(), metal);
    float3 throughput = 1;
    float3 direction;
    bool scatterWasDelta = false;

    if (specularLobe)
    {
        scatterWasDelta = roughness <= RTXPT_INTRO_MAX_DELTA_ROUGHNESS;
        float3 halfVector = scatterWasDelta ? normal
            : SampleVNDF(rng.getNextUniformRand2(), viewDir, normal, max(roughness * roughness, 1e-4));
        direction = reflect(-viewDir, halfVector);
        float weight = scatterWasDelta ? 1
            : SmithGGX_G2_over_G1v(max(1e-2, roughness), saturate(dot(viewDir, normal)), saturate(dot(direction, normal)));
        throughput = Schlick_Fresnel(F0, saturate(dot(viewDir, halfVector))) * weight / specProbability;
    }
    else
    {
        throughput = (1 - Schlick_Fresnel(F0, saturate(dot(viewDir, normal)))) / (1 - specProbability);
        if (shaderId == ShaderIdGlassGGX)
        {
            normal = -normal;
            direction = normalize(-viewDir + 0.25 * normal);
            scatterWasDelta = true;
        }
        else
        {
            direction = SampleHemisphereCosineWeighted(rng.getNextUniformRand2(), normal);
        }
    }

    RayDesc ray;
    ray.Origin = worldPos + normal * max(1e-3, 1e-4 * cameraDistance);
    ray.Direction = direction;
    ray.TMin = 1e-4;
    ray.TMax = 1e6;
    uint rayFlags = shaderId == ShaderIdGlassGGX ? RAY_FLAG_CULL_BACK_FACING_TRIANGLES : RAY_FLAG_NONE;

    // params.y describes primary direct-light sampling. Delta directions are not covered by NEE.
    bool emissionCovered = !scatterWasDelta && (g_MiniConst.params.y & 1u) != 0;
    bool environmentCovered = !scatterWasDelta && (g_MiniConst.params.y & 2u) != 0;
    bool cameThroughDelta = scatterWasDelta;
    bool lightedReplacedSurface = false;
    float3 radiance = 0;

    // Two shaded vertices; a glass crossing consumes only a quarter of a vertex.
    // Crossings require room for a following vertex, bounding this loop to five rays.
    [loop] for (float vertex = 0; vertex < 2; vertex += 1)
    {
        PayloadSimplified payload;
        payload.hitDistance = -1;
        TraceRay(SceneBVH, rayFlags, 0xff, 0, 1, 0, ray, payload);

        // Copy all hit data out of the annotated payload before shading or continuing the path.
        float3 hitBaseColor = payload.baseColor;
        float3 hitNormal = payload.normal;
        uint hitShaderId = payload.shaderId;
        float hitDistance = payload.hitDistance;
        if (hitDistance < 0)
        {
            if (!environmentCovered)
            {
                EnvMap env = EnvMap::make(t_EnvironmentMap, s_EnvironmentMapSampler, g_Const.envMapSceneParams);
                radiance += throughput * env.Eval(ray.Direction, 0.f);
            }
            break;
        }

        float3 hitPos = ray.Origin + ray.Direction * hitDistance;
        if (hitShaderId == ShaderIdEmissive)
        {
            if (!emissionCovered)
                radiance += throughput * hitBaseColor;
            break;
        }

        bool canContinue = vertex + 1 < 2;
        if (hitShaderId == ShaderIdGlassGGX)
        {
            if (!canContinue)
                break;

            // Keep Intro's approximate refraction and single-interface tint model.
            throughput *= (1 - Schlick_Fresnel(float3(0.04, 0.04, 0.04), saturate(dot(-ray.Direction, hitNormal))))
                * saturate(hitBaseColor);
            ray.Origin = hitPos - hitNormal * max(1e-3, 1e-4 * hitDistance);
            ray.Direction = normalize(ray.Direction - 0.25 * hitNormal);
            rayFlags = RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
            cameThroughDelta = true;
            emissionCovered = false;
            environmentCovered = false;
            vertex -= 0.75;
            continue;
        }

        if (cameThroughDelta && !lightedReplacedSurface && canContinue)
        {
            // This visible reflected/transmitted surface was not lit by the primary direct pass.
            radiance += throughput * ComputeHitLighting(hitPos, hitBaseColor, hitNormal, hitShaderId,
                -ray.Direction, rng, false, false);
            throughput *= saturate(hitBaseColor);
            ray.Origin = hitPos + hitNormal * max(1e-3, 1e-4 * hitDistance);
            ray.Direction = SampleHemisphereCosineWeighted(rng.getNextUniformRand2(), hitNormal);
            rayFlags = RAY_FLAG_NONE;
            lightedReplacedSurface = true;
            cameThroughDelta = false;
            // Intro's secondary NEE samples analytic lights only; no emitter or sky coverage.
            emissionCovered = false;
            environmentCovered = false;
            continue;
        }

#if IPT_SHARC_ENABLED
        SharcParameters sharc = GetDefaultSharcParams(g_Const.ptConsts.camera.PosW);
        sharc.hashGridData.capacity = g_MiniConst.params.x;
        sharc.hashGridData.hashEntriesBuffer = u_SHaRCHashEntries;
        sharc.accumulationBuffer = u_SHaRCAccumulation;
        sharc.resolvedBuffer = u_SHaRCResolved;

        SharcHitData sharcHit;
        sharcHit.positionWorld = hitPos;
        sharcHit.normalWorld = hitNormal;
        uint gridLevel = HashGridGetLevel(hitPos, sharc.hashGridParameters);
        float voxelSize = HashGridGetVoxelSize(gridLevel, sharc.hashGridParameters);
        float3 cachedRadiance;
        if (!cameThroughDelta && hitDistance >= voxelSize && SharcGetCachedRadiance(sharc, sharcHit, cachedRadiance, false))
        {
            radiance += throughput * cachedRadiance;
        }
        else
#endif
        {
            radiance += throughput * ComputeHitLighting(hitPos, hitBaseColor, hitNormal, hitShaderId,
                -ray.Direction, rng, false, true);
        }
        break;
    }

    float4 result = float4(min(radiance, 65504), 1);
    u_SSRResult[pixel] = specularLobe ? result : 0;
    u_RTGIOutput[pixel] = specularLobe ? 0 : result;
}

[shader("miss")]
void MISS_ENTRY(inout PayloadSimplified payload : SV_RayPayload)
{
}
