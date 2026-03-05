/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Ground Truth Ambient Occlusion (GTAO)
// Based on: Jimenez et al., "Practical Real-Time Strategies for Accurate Indirect Occlusion"
//
// Three compute passes:
//   GTAOComputeCS     - Horizon-based AO with analytic inner integral (half-res)
//   GTAOSpatialFilterCS - 4x4 bilateral spatial filter (half-res)
//   GTAOTemporalCS    - Temporal accumulation + bilateral upscale (full-res output)

#ifndef __GTAO_PASSES_HLSL__
#define __GTAO_PASSES_HLSL__

#include <donut/shaders/utils.hlsli>
#include <donut/shaders/binding_helpers.hlsli>

#pragma pack_matrix(row_major)

#define GTAO_THREADS 8

#define GTAO_NUM_STEPS 12
#define GTAO_NUM_TEMPORAL_ROTATIONS 6

// Thickness heuristic decay rate (Eq. 13)
#define GTAO_THICKNESS_BETA 0.05

// Maximum screen-space radius in pixels at half-res
#define GTAO_MAX_PIXEL_RADIUS 64

static const float GTAO_PI = 3.14159265358979323846;
static const float GTAO_HALF_PI = 1.57079632679489661923;

// Must match GTAOConstants in GTAORenderer.cpp
struct GTAOConstants
{
    float4x4    MatWorldToView;
    float4x4    MatClipToWorldNoOffset;
    float       ProjScale;          // matViewToClip[0][0]
    float       Radius;
    float       FalloffEnd;
    float       TemporalAlpha;
    uint        HalfWidth;
    uint        HalfHeight;
    uint        FullWidth;
    uint        FullHeight;
    uint        FrameIndex;
    float       ViewportSizeInvX;
    float       ViewportSizeInvY;
    uint        Padding;
};

// ============================================================================
// Shared utility functions
// ============================================================================

// Reconstruct view-space position from NDC depth and pixel coordinates
float3 GTAO_ReconstructViewPos(uint2 fullResPixel, float ndcDepth, GTAOConstants gtao)
{
    float2 uv = (float2(fullResPixel) + 0.5) * float2(gtao.ViewportSizeInvX, gtao.ViewportSizeInvY);
    float4 clipPos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, ndcDepth, 1.0);
    float4 worldPos = mul(clipPos, gtao.MatClipToWorldNoOffset);
    worldPos.xyz /= worldPos.w;
    float4 viewPos = mul(float4(worldPos.xyz, 1.0), gtao.MatWorldToView);
    return viewPos.xyz;
}

// Fast acos approximation (Eberly 2014, max error ~0.0017 rad)
float GTAO_FastAcos(float x)
{
    float ax = abs(x);
    float res = -0.156583 * ax + GTAO_HALF_PI;
    res *= sqrt(1.0 - ax);
    return (x >= 0) ? res : GTAO_PI - res;
}

// 4x4 spatial noise directions, uniformly spaced over [0, pi)
// These tile every 4 pixels in each dimension, giving 16 unique directions
float GTAO_SpatialDirection(uint2 pixel, uint frameIndex)
{
    // Interleaved gradient-style noise using a 4x4 tile
    // 16 base directions + 6 temporal rotations = 96 effective directions
    uint tileX = pixel.x & 3;
    uint tileY = pixel.y & 3;
    uint idx = tileY * 4 + tileX;

    // Hashed rotation offset per tile slot
    float baseAngle = float(idx) * (GTAO_PI / 16); // Only rotate within the top semicircle. We'll sample along both + and - the direction.

    // Temporal rotation: cycle through 6 rotations
    // TODO: Maybe apply the rotation randomly here according to a low discrepancy sequence?
    float temporalOffset = float(frameIndex % GTAO_NUM_TEMPORAL_ROTATIONS)* (GTAO_PI / float(16 * GTAO_NUM_TEMPORAL_ROTATIONS));

    return baseAngle + temporalOffset;
}

// Analytic inner integral of the AO equation (Eq. 7)
// theta1, theta2: horizon angles, gamma: angle between projected normal and view direction
float GTAO_IntegrateArc(float theta1, float theta2, float gamma)
{
    float cosGamma = cos(gamma);
    float sinGamma = sin(gamma);

    return 0.25 *(2 * theta2 * sinGamma + cosGamma - cos(2 * theta2 - gamma)
                + 2 * theta1 * sinGamma + cosGamma - cos(2 * theta1 - gamma));
}

// ============================================================================
// GTAOComputeCS - Main horizon search (Paper Section 4.1)
// ============================================================================

#ifdef __GTAO_COMPUTE_CS__

#define NON_PATH_TRACING_PASS 1
#include "Bindings/ShaderResourceBindings.hlsli"

// Dedicated binding set (space1): GTAO-specific resources
// Push constants (g_MiniConst) come from global binding set (space0) via ShaderResourceBindings.hlsli
// Depth is read from t_DepthHierarchy (space0, register t84) via the global binding set
cbuffer GTAOComputeCB : register(b0, space1) { GTAOConstants g_GTAO; }

Texture2D<uint>     t_GTAO_Normal   : register(t0, space1);
RWTexture2D<float>  u_GTAO_RawAO    : register(u0, space1);
SamplerState        s_GTAO_Point    : register(s0, space1);

[numthreads(GTAO_THREADS, GTAO_THREADS, 1)]
void GTAOComputeCS(uint2 dispatchID : SV_DispatchThreadID)
{
    if (any(dispatchID >= uint2(g_GTAO.HalfWidth, g_GTAO.HalfHeight)))
        return;

    // Map half-res pixel to full-res center
    uint2 fullResPixel = dispatchID * 2 + 1; // TODO: Use a halton sequence here to choose a pixel within the 2x2
    fullResPixel = min(fullResPixel, uint2(g_GTAO.FullWidth - 1, g_GTAO.FullHeight - 1));

    float depth = t_DepthHierarchy.Load(int3(fullResPixel, 0));
    if (depth <= 0)
    {
        u_GTAO_RawAO[dispatchID] = 1.0;
        return;
    }

    float3 viewPos = GTAO_ReconstructViewPos(fullResPixel, depth, g_GTAO);

    float3 worldNormal = octToNdirUnorm32(t_GTAO_Normal[fullResPixel]);
    float3 viewNormal = normalize(mul(worldNormal, (float3x3)g_GTAO.MatWorldToView));

    // View direction (from surface toward camera, in view space)
    float3 viewDir = normalize(-viewPos);

    // Screen-space step size scaled by world-space radius and distance from camera
    float viewDist = length(viewPos);
    float screenRadius = (g_GTAO.Radius * g_GTAO.ProjScale) / viewDist * 0.25;
    // Convert to half-res pixel units and clamp
    float pixelRadius = screenRadius * float(g_GTAO.HalfWidth);
    pixelRadius = min(pixelRadius, float(GTAO_MAX_PIXEL_RADIUS));

    if (pixelRadius < 1.0)
    {
        u_GTAO_RawAO[dispatchID] = 1.0;
        return;
    }

    float stepSize = pixelRadius / float(GTAO_NUM_STEPS);

    // Choose azimuthal direction from spatial noise + temporal rotation
    float phi = GTAO_SpatialDirection(dispatchID, g_GTAO.FrameIndex);
    float2 direction = float2(cos(phi), -sin(phi)); // Directions in pixel space, y sign is flipped

    // Search for horizon angles in both directions along the slice
    float horizonCos1 = -1.0;
    float horizonCos2 = -1.0;

    [unroll]
    for (int step = 1; step <= GTAO_NUM_STEPS; step++)
    {
        float radius = float(step) * stepSize;

        // Positive direction
        {
            float2 sampleCoord = float2(dispatchID) + 0.5 + direction * radius;
            int2 samplePixel = clamp(int2(sampleCoord), 0, int2(g_GTAO.HalfWidth - 1, g_GTAO.HalfHeight - 1));

            // Sample depth from hierarchy mip 1 (half-res), more bandwidth-efficient than sparse full-res reads
            float sampleDepth = t_DepthHierarchy.Load(int3(samplePixel, 1));
            if (sampleDepth > 0)
            {
                // Reconstruct view-space position using the half-res pixel mapped back to full-res
                uint2 sampleFullRes = min(uint2(samplePixel) * 2 + 1, uint2(g_GTAO.FullWidth - 1, g_GTAO.FullHeight - 1));
                float3 sampleViewPos = GTAO_ReconstructViewPos(sampleFullRes, sampleDepth, g_GTAO);
                float3 omega = normalize(sampleViewPos - viewPos);
                float sampleCos = dot(omega, viewDir);

                if (sampleCos > horizonCos1)
                    horizonCos1 = sampleCos;
                else
                    horizonCos1 -= GTAO_THICKNESS_BETA; // Thickness heuristic (Eq. 13)
            }
        }

        // Negative direction
        {
            float2 sampleCoord = float2(dispatchID) + 0.5 - direction * radius;
            int2 samplePixel = clamp(int2(sampleCoord), 0, int2(g_GTAO.HalfWidth - 1, g_GTAO.HalfHeight - 1));

            // Sample depth from hierarchy mip 1 (half-res)
            float sampleDepth = t_DepthHierarchy.Load(int3(samplePixel, 1));
            if (sampleDepth > 0)
            {
                uint2 sampleFullRes = min(uint2(samplePixel) * 2 + 1, uint2(g_GTAO.FullWidth - 1, g_GTAO.FullHeight - 1));
                float3 sampleViewPos = GTAO_ReconstructViewPos(sampleFullRes, sampleDepth, g_GTAO);
                float3 omega = normalize(sampleViewPos - viewPos);
                float sampleCos = dot(omega, viewDir);

                if (sampleCos > horizonCos2)
                    horizonCos2 = sampleCos;
                else
                    horizonCos2 -= GTAO_THICKNESS_BETA;
            }
        }
    }

    // Convert horizon cosines to angles
    // Theta goes from -pi/2 to pi/2, so one of the two angles has to be negative. In the computation of gamma below, we chose
    // to make gamma negative when it's dot product with the scree space direction 1 (the slice tangent) is positive.
    // That means that theta1 is also positive, since it follows the same positive direction.
    float theta1 = -GTAO_FastAcos(horizonCos1);
    float theta2 = GTAO_FastAcos(horizonCos2);
    
    // Project normal onto the slice plane (Eq. 8-9)
    float3 sliceTangent = float3(direction.x, -direction.y, 0); // Direction is in pixel space, so we need to flip y to get tangent in view space
    float3 sliceBitangent = cross(viewDir, sliceTangent);
    float3 projNormal = viewNormal - sliceBitangent * dot(viewNormal, sliceBitangent);
    float projNormalLen = length(projNormal);
    
    // Angle between projected normal and view direction
    float cosGamma = clamp(dot(normalize(projNormal), viewDir), -1.0, 1.0);
    float gamma = -sign(dot(projNormal.xy, sliceTangent.xy)) * GTAO_FastAcos(cosGamma);
    
    // Clamp horizon angles to the normal hemisphere
    theta1 = max(theta1, gamma - GTAO_HALF_PI);
    theta2 = min(theta2, gamma + GTAO_HALF_PI);
    
    // Analytic inner integral (Eq. 7), weighted by projected normal length (Eq. 9)
    // The 1/4 factor in IntegrateArc already normalizes the result to [0, 1]
    float ao = GTAO_IntegrateArc(theta1, theta2, gamma);
    //float ao = projNormalLen * GTAO_IntegrateArc(theta1, theta2, gamma);

    u_GTAO_RawAO[dispatchID] = saturate(ao) / projNormalLen; // saturate(ao);
}

#endif // __GTAO_COMPUTE_CS__

// ============================================================================
// GTAOSpatialFilterCS - 4x4 depth-aware bilateral filter (Paper Section 4.3)
// ============================================================================

#ifdef __GTAO_SPATIAL_FILTER_CS__

cbuffer GTAOFilterCB : register(b0) { GTAOConstants g_GTAOFilter; }
struct GTAOFilterMiniConst { uint4 p; uint4 p1; uint4 p2; uint4 p3; };
VK_PUSH_CONSTANT ConstantBuffer<GTAOFilterMiniConst> g_MiniConst : register(b1);

Texture2D<float>    t_GTAO_RawAO        : register(t0);
Texture2D<float>    t_GTAO_FilterDepth   : register(t1);
RWTexture2D<float>  u_GTAO_FilteredAO    : register(u0);
SamplerState        s_GTAO_FilterPoint   : register(s0);

[numthreads(GTAO_THREADS, GTAO_THREADS, 1)]
void GTAOSpatialFilterCS(uint2 dispatchID : SV_DispatchThreadID)
{
    if (any(dispatchID >= uint2(g_GTAOFilter.HalfWidth, g_GTAOFilter.HalfHeight)))
        return;

    // Reference depth at center pixel (full-res)
    uint2 centerFullRes = min(dispatchID * 2 + 1, uint2(g_GTAOFilter.FullWidth - 1, g_GTAOFilter.FullHeight - 1));
    float centerDepth = t_GTAO_FilterDepth[centerFullRes];

    if (centerDepth <= 0)
    {
        u_GTAO_FilteredAO[dispatchID] = 1.0;
        return;
    }

    // 4x4 bilateral gather matching the spatial noise tile
    float totalAO = 0;
    float totalWeight = 0;

    // Depth sensitivity for bilateral weights
    float depthThreshold = centerDepth * 0.05;

    int2 baseCoord = int2(dispatchID) - 1; // Start 1 pixel before center for 4x4 kernel

    [unroll]
    for (int y = 0; y < 4; y++)
    {
        [unroll]
        for (int x = 0; x < 4; x++)
        {
            int2 sampleCoord = baseCoord + int2(x, y);
            sampleCoord = clamp(sampleCoord, 0, int2(g_GTAOFilter.HalfWidth - 1, g_GTAOFilter.HalfHeight - 1));

            float sampleAO = t_GTAO_RawAO[sampleCoord];

            // Bilateral depth weight
            uint2 sampleFullRes = min(uint2(sampleCoord) * 2 + 1, uint2(g_GTAOFilter.FullWidth - 1, g_GTAOFilter.FullHeight - 1));
            float sampleDepth = t_GTAO_FilterDepth[sampleFullRes];

            float depthDiff = abs(sampleDepth - centerDepth);
            float weight = (sampleDepth > 0 && depthDiff < depthThreshold) ? 1.0 : 0.0;

            totalAO += sampleAO * weight;
            totalWeight += weight;
        }
    }

    u_GTAO_FilteredAO[dispatchID] = (totalWeight > 0) ? (totalAO / totalWeight) : t_GTAO_RawAO[dispatchID];
}

#endif // __GTAO_SPATIAL_FILTER_CS__

// ============================================================================
// GTAOTemporalCS - Temporal accumulation + bilateral upscale to full-res
// ============================================================================

#ifdef __GTAO_TEMPORAL_CS__

cbuffer GTAOTemporalCB : register(b0) { GTAOConstants g_GTAOTemporal; }
struct GTAOTemporalMiniConst { uint4 p; uint4 p1; uint4 p2; uint4 p3; };
VK_PUSH_CONSTANT ConstantBuffer<GTAOTemporalMiniConst> g_MiniConst : register(b1);

Texture2D<float>    t_GTAO_Filtered      : register(t0);
Texture2D<float>    t_GTAO_History       : register(t1);
Texture2D<float4>   t_GTAO_MotionVecs    : register(t2);
Texture2D<float>    t_GTAO_TempDepth     : register(t3);
Texture2D<float>    t_GTAO_PrevDepth     : register(t4);
RWTexture2D<float>  u_GTAO_Output        : register(u0);
RWTexture2D<float>  u_GTAO_HistoryWrite  : register(u1);
SamplerState        s_GTAO_TempPoint     : register(s0);

[numthreads(GTAO_THREADS, GTAO_THREADS, 1)]
void GTAOTemporalCS(uint2 dispatchID : SV_DispatchThreadID)
{
    if (any(dispatchID >= uint2(g_GTAOTemporal.FullWidth, g_GTAOTemporal.FullHeight)))
        return;

    float depth = t_GTAO_TempDepth[dispatchID];
    if (depth <= 0)
    {
        u_GTAO_Output[dispatchID] = 1.0;
        return;
    }

    // Bilateral upscale from half-res to full-res
    // Find the 4 nearest half-res texels and weight by depth similarity
    float2 halfResCoord = (float2(dispatchID) + 0.5) * 0.5 - 0.5;
    int2 halfResBase = int2(floor(halfResCoord));
    float2 frac_ = halfResCoord - float2(halfResBase);

    float currentAO = 0;
    float totalWeight = 0;
    float depthThreshold = depth * 0.05;

    [unroll]
    for (int y = 0; y <= 1; y++)
    {
        [unroll]
        for (int x = 0; x <= 1; x++)
        {
            int2 sampleCoord = clamp(halfResBase + int2(x, y), 0,
                int2(g_GTAOTemporal.HalfWidth - 1, g_GTAOTemporal.HalfHeight - 1));

            float sampleAO = t_GTAO_Filtered[sampleCoord];

            // Depth-based bilateral weight
            uint2 sampleFullRes = min(uint2(sampleCoord) * 2 + 1,
                uint2(g_GTAOTemporal.FullWidth - 1, g_GTAOTemporal.FullHeight - 1));
            float sampleDepth = t_GTAO_TempDepth[sampleFullRes];
            float depthDiff = abs(sampleDepth - depth);

            // Bilinear weight * depth weight
            float bilinearWeight = ((x == 0) ? (1.0 - frac_.x) : frac_.x) *
                                   ((y == 0) ? (1.0 - frac_.y) : frac_.y);
            float depthWeight = (sampleDepth > 0 && depthDiff < depthThreshold) ? 1.0 : 0.0;
            float weight = bilinearWeight * depthWeight;

            currentAO += sampleAO * weight;
            totalWeight += weight;
        }
    }

    currentAO = (totalWeight > 0) ? (currentAO / totalWeight) : t_GTAO_Filtered[clamp(halfResBase, 0,
        int2(g_GTAOTemporal.HalfWidth - 1, g_GTAOTemporal.HalfHeight - 1))];

    // Temporal reprojection: motion vectors are pixel-space offsets (previous - current)
    float3 motionVec = t_GTAO_MotionVecs[dispatchID].xyz;
    float2 prevPixel = float2(dispatchID) + 0.5 + motionVec.xy;
    float2 prevUV = prevPixel / float2(g_GTAOTemporal.FullWidth, g_GTAOTemporal.FullHeight);

    // Map to half-res coordinates
    float2 prevHalfRes = prevUV * float2(g_GTAOTemporal.HalfWidth, g_GTAOTemporal.HalfHeight) - 0.5;
    int2 prevHalfResPixel = clamp(int2(round(prevHalfRes)), 0,
        int2(g_GTAOTemporal.HalfWidth - 1, g_GTAOTemporal.HalfHeight - 1));

    float historyAO = t_GTAO_History[prevHalfResPixel];

    // Disocclusion detection: reject history if reprojected position is off-screen
    // or the surface at the reprojected position has changed (depth mismatch)
    bool validHistory = false;// all(prevUV >= 0) && all(prevUV < 1);

    if (validHistory)
    {
        // Compare previous frame depth at the reprojected position with current depth.
        // A large mismatch means a different surface is now visible (disocclusion).
        int2 prevFullResPixel = clamp(int2(round(prevPixel - 0.5)), 0,
            int2(g_GTAOTemporal.FullWidth - 1, g_GTAOTemporal.FullHeight - 1));
        float prevDepth = t_GTAO_PrevDepth[prevFullResPixel];
        float depthRelDiff = abs(depth - prevDepth) / max(depth, 1e-6);
        if (depthRelDiff > 0.1)
            validHistory = false;
    }

    float outputAO;
    if (validHistory)
    {
        outputAO = lerp(currentAO, historyAO, g_GTAOTemporal.TemporalAlpha);
    }
    else
    {
        outputAO = currentAO;
    }

    u_GTAO_Output[dispatchID] = saturate(outputAO);

    // Write to half-res history for next frame
    uint2 halfResPixel = dispatchID / 2;
    // Only write from the pixel that owns this half-res texel (top-left of each 2x2 block)
    if (all((dispatchID & 1) == 0))
    {
        u_GTAO_HistoryWrite[halfResPixel] = saturate(outputAO);
    }
}

#endif // __GTAO_TEMPORAL_CS__

#endif // __GTAO_PASSES_HLSL__
