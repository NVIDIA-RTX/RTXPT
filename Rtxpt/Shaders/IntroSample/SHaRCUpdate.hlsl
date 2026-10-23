/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// SHaRC Update pass: sparse multi-bounce path tracing to populate the radiance cache.
// Dispatched at 1/4 x 1/4 resolution. Each quarter-res thread maps to a random
// full-res pixel within its 4x4 block using a temporal hash.
// Based on IntroPathTracer.hlsl, but with SHaRC API calls instead of GBuffer writes.

#define ActiveBSDF StandardBSDF

#include "../PathTracer/Config.h"

#define SHARC_UPDATE 1

#include "../PathTracer/PathTracerTypes.hlsli"

#include "../Bindings/ShaderResourceBindings.hlsli"

#include "IntroBridge.hlsli"
#include "../Bindings/GBufferBindings.hlsli"
#include "FastHitShaders.hlsl" // provides IntroEvaluateHitDirectLighting + TraceVisibilityRay

// SHaRC buffer declarations with correct types (not in ShaderResourceBindings.hlsli to avoid type dependency)
RWStructuredBuffer<uint64_t>                u_SHaRCHashEntries   : register(u90);
RWStructuredBuffer<SharcAccumulationData>   u_SHaRCAccumulation  : register(u91);
RWStructuredBuffer<SharcPackedData>         u_SHaRCResolved      : register(u92);

SharcParameters GetSharcParameters()
{
    SharcParameters params = GetDefaultSharcParams(g_Const.ptConsts.camera.PosW);

    params.hashGridData.capacity = g_MiniConst.params.x;
    params.hashGridData.hashEntriesBuffer = u_SHaRCHashEntries;
    params.accumulationBuffer = u_SHaRCAccumulation;
    params.resolvedBuffer = u_SHaRCResolved;
    
    return params;
}

struct PathContextLite
{
    float3 rayOrigin;
    float3 rayDirection;
    float curBounce;
    
    RNGState rng;
};

[shader("raygeneration")]
void RAYGEN_ENTRY()
{
    uint2 quarterPixel = DispatchRaysIndex().xy;
    uint2 quarterSize = DispatchRaysDimensions().xy;
    uint frameIndex = g_Const.ptConsts.frameIndex;

    // Map quarter-res pixel to a random full-res pixel within its 4x4 block
    uint blockSeed = Hash32Combine(Hash32(frameIndex), (quarterPixel.x << 16) | quarterPixel.y);
    uint offsetX = blockSeed & 3;
    uint offsetY = (blockSeed >> 2) & 3;
    uint2 fullResSize = uint2(g_Const.view.viewportSize);
    uint2 pixelPos = min(quarterPixel * 4 + uint2(offsetX, offsetY), fullResSize - 1);

    // Read GBuffer at the selected full-res pixel
    float depth = u_Depth[pixelPos];
    
    PackedGBufferSurface packedSurface;
    packedSurface.LoadFromRenderTargets(pixelPos);
    float3 normal = packedSurface.GetSpecNormal();
    uint shaderId = packedSurface.GetShaderId();
    
    if (depth <= 0) return;
    
    float3 worldPos = ReconstructWorldPos(pixelPos, depth, g_Const.view.pixelOffset, g_Const.view.viewportSizeInv, g_Const.view.matClipToWorldNoOffset);

    PathContextLite pathContext;
    pathContext.curBounce = 0;
    pathContext.rng.initRngSeed(pixelPos, g_Const.ptConsts.frameIndex);
    
    EnvMap envMap = EnvMap::make(t_EnvironmentMap, s_EnvironmentMapSampler, g_Const.envMapSceneParams);

    float3 cameraRelativePos = worldPos - g_Const.ptConsts.camera.PosW;
    float cameraDist = length(cameraRelativePos);
    float3 viewDir = -cameraRelativePos / cameraDist;
    
    // Initialize SHaRC state
    SharcParameters sharcParams = GetSharcParameters();
    SharcState sharcState;
    SharcInit(sharcState);
    
    // SHaRC hit update
    SharcHitData sharcHitData;
    sharcHitData.positionWorld = worldPos.xyz;
    sharcHitData.normalWorld = normal;
    
    // Direct lighting
    float3 directLighting = 0;

    // Cache outgoing radiance, including emission at the primary surface just as at later hits.
    // Environment illumination is supplied by path misses, not a second environment sample here.
    directLighting = ComputeHitLighting(worldPos, packedSurface.GetBaseColor(), normal, shaderId, viewDir, pathContext.rng, true, false);
    if (!SharcUpdateHit(sharcParams, sharcState, sharcHitData, directLighting, pathContext.rng.getNextUniformRand()))
        return;
    
    // Handle glass: transmit through instead
    float3 worldOffset = normal * max(4e-3, 4e-3 * cameraDist);
    if (shaderId == ShaderIdGlassGGX)
    {
        pathContext.rayDirection = -viewDir;
        worldOffset = -2 * worldOffset;
    }
    else
    {        
        pathContext.rayDirection = SampleHemisphereCosineWeighted(pathContext.rng.getNextUniformRand2(), normal);

        // Lambertian BRDF * cosine / cosine-weighted PDF = albedo.
        SharcSetThroughput(sharcState, saturate(packedSurface.GetBaseColor()));
    }
    pathContext.rayOrigin = worldPos + worldOffset;
    
    const int kMaxBounces = SHARC_PROPAGATION_DEPTH - 1; // -1 because the first bounce was executed before the loop

    [unroll]
    for (int bounce = 0; bounce < kMaxBounces; ++bounce)
    {
        RayDesc rayDesc;
        rayDesc.Origin = pathContext.rayOrigin;
        rayDesc.Direction = pathContext.rayDirection;
        rayDesc.TMin = 1e-3;
        rayDesc.TMax = 1000;

        PayloadSimplified payload;
        payload.hitDistance = -1;
        TraceRay(SceneBVH, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xff, 0, 1, 0, rayDesc, payload);
        
        if (payload.hitDistance < 0)
        {
            float3 envRadiance = envMap.Eval(pathContext.rayDirection, 0.f);
            SharcUpdateMiss(sharcParams, sharcState, envRadiance);
            break;
        }
        worldPos = rayDesc.Origin + payload.hitDistance * rayDesc.Direction;

        // Correct for back facing normals
        normal = payload.normal;
        if (dot(pathContext.rayDirection, normal) > 0)
            normal = -normal;

        // NEE: one local and one directional analytic light, plus surface emission.
        directLighting = 0;
        {
            viewDir = -pathContext.rayDirection;
            directLighting = ComputeHitLighting(worldPos, payload.baseColor, normal, payload.shaderId, viewDir, pathContext.rng, true, false);
        }

        // Update sharcHitData to THIS hit's position/normal before caching
        sharcHitData.positionWorld = worldPos;
        sharcHitData.normalWorld = normal;
                    
        if (!SharcUpdateHit(sharcParams, sharcState, sharcHitData, directLighting, pathContext.rng.getNextUniformRand()))
            break;

        pathContext.rayOrigin = worldPos + max(1e-4, 1e-3 * payload.hitDistance) * normal;
        pathContext.rayDirection = SampleHemisphereCosineWeighted(pathContext.rng.getNextUniformRand2(), normal);

        SharcSetThroughput(sharcState, saturate(payload.baseColor));
    }
}

[shader("miss")]
void MISS_ENTRY(inout PayloadSimplified payload : SV_RayPayload)
{
}
