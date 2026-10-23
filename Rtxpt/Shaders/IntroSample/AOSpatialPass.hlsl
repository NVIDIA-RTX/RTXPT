/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// AO Spatial Filter - 4x4 depth-aware bilateral filter (half-res)
// Based on: Jimenez et al., "Practical Real-Time Strategies for Accurate Indirect Occlusion", Section 4.3

#ifndef __AO_SPATIAL_PASS_HLSL__
#define __AO_SPATIAL_PASS_HLSL__

#include "IntroSampleCommon.hlsli"

// ============================================================================
// AO SpatialFilter - 4x4 depth-aware bilateral filter (Paper Section 4.3)
// ============================================================================

cbuffer AOFilterCB : register(b0) { AOConstants g_AOFilter; }
struct AOFilterMiniConst { uint4 p; uint4 p1; uint4 p2; uint4 p3; };

VK_PUSH_CONSTANT ConstantBuffer<AOFilterMiniConst> g_MiniConst : register(b1);

Texture2D<float>    t_GTAO_RawAO        : register(t0);
Texture2D<float>    t_GTAO_FilterDepth   : register(t1);
RWTexture2D<float>  u_GTAO_FilteredAO    : register(u0);
SamplerState        s_GTAO_FilterPoint   : register(s0);

[numthreads(8, 8, 1)]
void main(uint2 dispatchID : SV_DispatchThreadID)
{
    if (any(dispatchID >= uint2(g_AOFilter.HalfWidth, g_AOFilter.HalfHeight)))
        return;

    // Reference depth at the full-res pixel this half-res texel represents
    uint2 centerFullRes = HalfToFullRes(dispatchID, g_AOFilter.FullWidth, g_AOFilter.FullHeight);
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
            sampleCoord = clamp(sampleCoord, 0, int2(g_AOFilter.HalfWidth - 1, g_AOFilter.HalfHeight - 1));

            float sampleAO = t_GTAO_RawAO[sampleCoord];

            // Bilateral depth weight
            uint2 sampleFullRes = HalfToFullRes(uint2(sampleCoord), g_AOFilter.FullWidth, g_AOFilter.FullHeight);
            float sampleDepth = t_GTAO_FilterDepth[sampleFullRes];

            float depthDiff = abs(sampleDepth - centerDepth);
            float weight = (sampleDepth > 0 && depthDiff < depthThreshold) ? 1.0 : 0.0;

            totalAO += sampleAO * weight;
            totalWeight += weight;
        }
    }

    u_GTAO_FilteredAO[dispatchID] = (totalWeight > 0) ? (totalAO / totalWeight) : t_GTAO_RawAO[dispatchID];
}

#endif // __AO_SPATIAL_PASS_HLSL__
