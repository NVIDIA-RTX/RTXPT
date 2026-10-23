/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Ray-Traced Ambient Occlusion (RTAO)
// Traces a single cosine-weighted hemisphere ray per half-res pixel per frame.
// Relies on spatial filtering + temporal accumulation for convergence.

#ifndef __RTAOTRACE_HLSL__
#define __RTAOTRACE_HLSL__

#define IGNORE_RAY_PAYLOAD 1

#include "IntroSampleCommon.hlsli"
#include "../Bindings/ShaderResourceBindings.hlsli"
#include "../PathTracer/Utils/Math/HashUtils.hlsli"
#include "../PathTracer/Utils/Math/MathHelpers.hlsli"
#include "../PathTracer/Utils/Geometry.hlsli"

// ============================================================================
// RTAO - Inline ray query AO
// ============================================================================

// TLAS from global binding set (space0, register t0)
// Cannot include SceneBindings.hlsli directly because it declares bindless
// resources in space1, which conflicts with the AO binding layout in space1.
RaytracingAccelerationStructure SceneBVH : register(t0);

// Dedicated binding set (space1): AO-specific resources
// Same layout as GTAOPass.hlsl
cbuffer GTAOComputeCB : register(b0, space1) { AOConstants g_GTAO; }

Texture2D<uint>     t_GTAO_Normal   : register(t0, space1);
RWTexture2D<float>  u_GTAO_RawAO    : register(u0, space1);
SamplerState        s_GTAO_Point    : register(s0, space1);

// Small normal bias to prevent self-intersection
#define RTAO_NORMAL_BIAS 0.01

[numthreads(8, 8, 1)]
void main(uint2 dispatchID : SV_DispatchThreadID)
{
    if (any(dispatchID >= uint2(g_GTAO.HalfWidth, g_GTAO.HalfHeight)))
        return;

    // Map half-res pixel to full-res using spatially interleaved sub-pixel offset
    uint2 fullResPixel = HalfToFullRes(dispatchID, g_GTAO.FullWidth, g_GTAO.FullHeight);

    float depth = t_DepthHierarchy.Load(int3(fullResPixel, 0));
    if (depth <= 0)
    {
        u_GTAO_RawAO[dispatchID] = 1.0;
        return;
    }

    // Reconstruct world-space position
    float3 worldPos = ReconstructWorldPos(fullResPixel, depth, g_GTAO.subPixelOffset, g_GTAO.ViewportInvSize, g_GTAO.MatClipToWorldNoOffset);

    // Decode world-space normal
    float3 worldNormal = octToNdirUnorm32(t_GTAO_Normal[fullResPixel]);

    // Cosine-weighted hemisphere sample in tangent space (+z = normal)
    RNGState rng;
    rng.initRngSeed(dispatchID, g_GTAO.FrameIndex);
    float3 localDir = SampleHemisphereCosineWeighted(rng.getNextUniformRand2());

    // Build orthonormal basis around world normal
    float3 tangent, bitangent;
    BranchlessONB(worldNormal, tangent, bitangent);

    // Transform to world space
    float3 rayDir = localDir.x * tangent + localDir.y * bitangent + localDir.z * worldNormal;

    // Trace AO ray
    RayDesc ray;
    ray.Origin = worldPos + worldNormal * RTAO_NORMAL_BIAS;
    ray.Direction = rayDir;
    ray.TMin = 0;
    ray.TMax = g_GTAO.Radius;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(SceneBVH, RAY_FLAG_FORCE_OPAQUE, 0xff, ray);
    query.Proceed();

    // Unoccluded = 1.0, occluded = 0.0
    float ao = (query.CommittedStatus() == COMMITTED_NOTHING) ? 1.0 : 0.0;
    u_GTAO_RawAO[dispatchID] = ao;
}

#endif // __RTAOTRACE_HLSL__
