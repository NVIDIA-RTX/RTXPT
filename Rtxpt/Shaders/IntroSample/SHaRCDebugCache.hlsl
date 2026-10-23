/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Debug visualization: displays SHaRC's cached radiance at GBuffer first-hit positions.
// Uses depth and normal to reconstruct world position, fetches from the cache, and
// outputs the retrieved radiance. Cache misses show black; sky shows environment.
// Replaces the deferred lighting pass when "Debug SHaRC cache" is enabled.

#define IGNORE_RAY_PAYLOAD 1

#include "../Bindings/ShaderResourceBindings.hlsli"
#include "../Bindings/GBufferBindings.hlsli"
#include "../Bindings/SamplerBindings.hlsli"
#include "IntroLightBindings.hlsli"
#include "IntroBridge.hlsli"
#include "IntroCommon.hlsli"

RWStructuredBuffer<uint64_t>                u_SHaRCHashEntries   : register(u90);
RWStructuredBuffer<SharcAccumulationData>   u_SHaRCAccumulation  : register(u91);
RWStructuredBuffer<SharcPackedData>         u_SHaRCResolved      : register(u92);

[numthreads(NUM_COMPUTE_THREADS_PER_DIM, NUM_COMPUTE_THREADS_PER_DIM, 1)]
void main(uint2 dispatchThreadID : SV_DispatchThreadID)
{
    const uint2 pixelPos = dispatchThreadID.xy;
    if (any(pixelPos >= uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight)))
        return;

    float depth = u_Depth[pixelPos];
    float3 rayDir = Bridge::computeCameraRay(pixelPos).dir;

    float3 radiance = 0;
    if (depth <= 0)
    {
        // Sky / no-hit: sample environment
        EnvMap envMap = EnvMap::make(t_EnvironmentMap, s_EnvironmentMapSampler, g_Const.envMapSceneParams);
        radiance = envMap.Eval(rayDir, 0.f);
    }
    else
    {
        // Reconstruct world position and normal from GBuffer
        PackedGBufferSurface surface;
        surface.LoadFromRenderTargets(pixelPos);
        float3 worldNormal = surface.GetSpecNormal();
        float3 worldPos = ReconstructWorldPos(pixelPos, depth, g_Const.view.pixelOffset, g_Const.view.viewportSizeInv, g_Const.view.matClipToWorldNoOffset);

        // Build SHaRC parameters and hit data for cache lookup
        SharcParameters sharcParams = GetDefaultSharcParams(g_Const.ptConsts.camera.PosW);
        
        sharcParams.hashGridData.capacity = g_MiniConst.params.x;
        sharcParams.hashGridData.hashEntriesBuffer = u_SHaRCHashEntries;
        sharcParams.accumulationBuffer = u_SHaRCAccumulation;
        sharcParams.resolvedBuffer = u_SHaRCResolved;

        SharcHitData sharcHitData;
        sharcHitData.positionWorld = worldPos;
        sharcHitData.normalWorld = worldNormal;
#if SHARC_SEPARATE_EMISSIVE
        sharcHitData.emissive = 0;
#endif // SHARC_SEPARATE_EMISSIVE

        float3 cachedRadiance;
        if (SharcGetCachedRadiance(sharcParams, sharcHitData, cachedRadiance, false))
        {
            radiance = cachedRadiance;
        }
        // else: cache miss -> black (shows where cache has no data)
    }

    u_OutputColor[pixelPos] = float4(radiance, 1.0);
}
