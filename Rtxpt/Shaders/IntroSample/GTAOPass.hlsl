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

#ifndef __GTAOPASS_HLSL__
#define __GTAOPASS_HLSL__

#define IGNORE_RAY_PAYLOAD 1

#include "IntroSampleCommon.hlsli"
#include "../Bindings/ShaderResourceBindings.hlsli"

// ============================================================================
// GTAOComputeCS - Main horizon search (Paper Section 4.1)
// ============================================================================

#define GTAO_NUM_STEPS 12
#define GTAO_NUM_TEMPORAL_ROTATIONS 6

// Thickness heuristic decay rate (Eq. 13)
#define GTAO_THICKNESS_BETA 0.05

// Maximum screen-space radius in pixels at half-res
#define GTAO_MAX_PIXEL_RADIUS 64

// Dedicated binding set (space1): GTAO-specific resources
// Push constants (g_MiniConst) come from global binding set (space0) via ShaderResourceBindings.hlsli
// Depth is read from t_DepthHierarchy (space0, register t84) via the global binding set
cbuffer GTAOComputeCB : register(b0, space1) { AOConstants g_GTAO; }

Texture2D<uint>     t_GTAO_Normal   : register(t0, space1);
RWTexture2D<float>  u_GTAO_RawAO    : register(u0, space1);
SamplerState        s_GTAO_Point    : register(s0, space1);

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

    float3 viewPos = ReconstructViewPos(fullResPixel, depth, g_GTAO.subPixelOffset, 
    g_GTAO.ViewportInvSize, g_GTAO.MatClipToWorldNoOffset, g_GTAO.MatWorldToView);

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
    float phi = AO_SpatialDirection(dispatchID, g_GTAO.FrameIndex, GTAO_NUM_TEMPORAL_ROTATIONS);
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
                uint2 sampleFullRes = HalfToFullRes(uint2(samplePixel), g_GTAO.FullWidth, g_GTAO.FullHeight);
                float3 sampleViewPos = ReconstructViewPos(sampleFullRes, sampleDepth, g_GTAO.subPixelOffset, 
                    g_GTAO.ViewportInvSize, g_GTAO.MatClipToWorldNoOffset, g_GTAO.MatWorldToView);
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
                uint2 sampleFullRes = HalfToFullRes(uint2(samplePixel), g_GTAO.FullWidth, g_GTAO.FullHeight);
                float3 sampleViewPos = ReconstructViewPos(sampleFullRes, sampleDepth, g_GTAO.subPixelOffset,
                    g_GTAO.ViewportInvSize, g_GTAO.MatClipToWorldNoOffset, g_GTAO.MatWorldToView);
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
    float theta1 = -FastACos(horizonCos1);
    float theta2 = FastACos(horizonCos2);
    
    // Project normal onto the slice plane (Eq. 8-9)
    float3 sliceTangent = float3(direction.x, -direction.y, 0); // Direction is in pixel space, so we need to flip y to get tangent in view space
    float3 sliceBitangent = cross(viewDir, sliceTangent);
    float3 projNormal = viewNormal - sliceBitangent * dot(viewNormal, sliceBitangent);
    float projNormalLen = length(projNormal);
    
    // Angle between projected normal and view direction
    float cosGamma = clamp(dot(normalize(projNormal), viewDir), -1.0, 1.0);
    float gamma = -sign(dot(projNormal.xy, sliceTangent.xy)) * FastACos(cosGamma);
    
    // Clamp horizon angles to the normal hemisphere
    theta1 = max(theta1, gamma - K_PI_2);
    theta2 = min(theta2, gamma + K_PI_2);
    
    // Analytic inner integral (Eq. 7), weighted by projected normal length (Eq. 9)
    // The 1/4 factor in IntegrateArc already normalizes the result to [0, 1]
    float ao = AO_IntegrateArc(theta1, theta2, gamma);
    //float ao = projNormalLen * AO_IntegrateArc(theta1, theta2, gamma);

    u_GTAO_RawAO[dispatchID] = projNormalLen > 1e-3 ? (saturate(ao) / projNormalLen) : 1;
}

#endif // __GTAOPASS_HLSL__
