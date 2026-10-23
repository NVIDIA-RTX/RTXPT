/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// AO Temporal Accumulation + Bilateral Upscale (half-res to full-res)
// Based on: Jimenez et al., "Practical Real-Time Strategies for Accurate Indirect Occlusion"

#ifndef __AO_TEMPORAL_UPSCALE_HLSL__
#define __AO_TEMPORAL_UPSCALE_HLSL__

#include "IntroSampleCommon.hlsli"

// ============================================================================
// AO Temporal Accumulation + Bilateral Upscale to full-res
// ============================================================================

cbuffer GTAOTemporalCB : register(b0) { AOConstants g_GTAOTemporal; }
struct AOTemporalMiniConst { uint4 p; uint4 p1; uint4 p2; uint4 p3; };
VK_PUSH_CONSTANT ConstantBuffer<AOTemporalMiniConst> g_MiniConst : register(b1);

Texture2D<float>    t_GTAO_Filtered      : register(t0);
Texture2D<float>    t_GTAO_History       : register(t1);
Texture2D<float4>   t_GTAO_MotionVecs    : register(t2);
Texture2D<float>    t_Depth              : register(t3);
Texture2D<float>    t_GTAO_PrevDepth     : register(t4);
// Final AO is folded straight into the GBuffer material-info target (low byte = material AO).
// Layout mirrors u_GBufferMaterialInfo in GBufferBindings.hlsli.
RWTexture2D<uint>   u_GBufferMaterialInfo : register(u0);
RWTexture2D<float>  u_HistoryOutput      : register(u1);
SamplerState        s_GTAO_TempPoint     : register(s0);

[numthreads(8, 8, 1)]
void main(uint2 dispatchID : SV_DispatchThreadID)
{
    if (any(dispatchID >= uint2(g_GTAOTemporal.FullWidth, g_GTAOTemporal.FullHeight)))
        return;

    float depth = t_Depth[dispatchID];
    if (depth <= 0)
    {
        // Sky/miss: leave the GBuffer material AO byte untouched (PopulateGBuffer already wrote 1.0).
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

            // Depth-based bilateral weight (read depth at the full-res pixel this half-res texel represents)
            uint2 sampleFullRes = HalfToFullRes(uint2(sampleCoord), g_GTAOTemporal.FullWidth, g_GTAOTemporal.FullHeight);
            float sampleDepth = t_Depth[sampleFullRes];
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

    // Expected previous view-space depth for depth-based history rejection
    float zNear = g_GTAOTemporal.ZNear;
    float currentViewZ = zNear / depth;
    float expectedPrevViewZ = currentViewZ + motionVec.z; // motionVec.z = prevViewZ - currentViewZ

    // Bilateral history sampling: sample the nearest 2x2 half-res history texels,
    // weighted by bilinear proximity and depth similarity to the expected surface.
    // This handles sub-pixel reprojection smoothly while rejecting history from
    // different surfaces (disocclusion) in a single pass.
    // Convert the reprojected full-res position to half-res using the exact same 2x convention as
    // the forward upscale above (half texel h is centered at full-res 2h+1). Scaling by
    // HalfDim/FullDim instead is only equal to 0.5 at even resolutions; at odd resolutions
    // HalfDim = (FullDim+1)/2, so that scale is slightly larger than 0.5 and introduces a
    // coordinate-dependent bias that grows toward the bottom/right. That bias misaligns the
    // history fetch from the forward sample, so temporal accumulation never converges and the
    // noise appears to "bubble" up/down.
    float2 prevHalfRes = prevPixel * 0.5 - 0.5;
    int2 prevHalfBase = int2(floor(prevHalfRes));
    float2 prevFrac = prevHalfRes - float2(prevHalfBase);

    float historyAO = 0;
    float historyWeight = 0;
    bool onScreen = all(prevUV >= 0) && all(prevUV < 1);

    if (onScreen)
    {
        [unroll]
        for (int hy = 0; hy <= 1; hy++)
        {
            [unroll]
            for (int hx = 0; hx <= 1; hx++)
            {
                int2 sampleHalf = clamp(prevHalfBase + int2(hx, hy), 0,
                    int2(g_GTAOTemporal.HalfWidth - 1, g_GTAOTemporal.HalfHeight - 1));

                float sampleAO = t_GTAO_History[sampleHalf];

                // Map half-res texel to the full-res pixel it represents (same offset pattern as prev frame)
                uint2 sampleFull = HalfToFullRes(uint2(sampleHalf), g_GTAOTemporal.FullWidth, g_GTAOTemporal.FullHeight);
                float prevNdcDepth = t_GTAO_PrevDepth[sampleFull];
                float actualPrevViewZ = zNear / max(prevNdcDepth, 1e-7);

                // Depth weight: accept history only if the surface matches
                float depthRelDiff = abs(expectedPrevViewZ - actualPrevViewZ) / max(abs(expectedPrevViewZ), 1e-4);
                float dw = (prevNdcDepth > 0 && depthRelDiff < 0.1) ? 1.0 : 0.0;

                // Bilinear weight
                float bw = ((hx == 0) ? (1.0 - prevFrac.x) : prevFrac.x) *
                           ((hy == 0) ? (1.0 - prevFrac.y) : prevFrac.y);

                float w = bw * dw;
                historyAO += sampleAO * w;
                historyWeight += w;
            }
        }
    }
        
    float outputAO;
    if (historyWeight > 0)
    {
        historyAO /= historyWeight;
        outputAO = lerp(currentAO, historyAO, g_GTAOTemporal.TemporalAlpha);
    }
    else
    {
        outputAO = currentAO;
    }

    float screenAO = saturate(outputAO);

    // Combine the screen-space AO with the baked material AO stored in the GBuffer,
    // then write the product back into the material-info low byte. The GBuffer is
    // regenerated every frame, so this read-modify-write never compounds over time.
    uint packedMaterialInfo = u_GBufferMaterialInfo[dispatchID];
    float materialAO = (packedMaterialInfo & 0xff) / 255.0;
    uint packedAO = min(255u, uint(min(screenAO, materialAO) * 255.0 + 0.5));
    u_GBufferMaterialInfo[dispatchID] = (packedMaterialInfo & 0xffffff00u) | packedAO;

    // Write to half-res history for next frame (pure screen-space AO, never the combined value).
    // Each half-res texel is written by the single full-res pixel it represents. Using
    // HalfToFullRes (which clamps to the valid full-res range) guarantees that every half-res
    // texel has an in-range writer even when the render resolution is odd. A naive
    // (dispatchID & 1) == subPixelOffset test would leave the bottom row / right column of
    // half-res texels unwritten at odd resolutions, so their uninitialized (black) history
    // would bleed a "fade to black" band along those borders through temporal accumulation.
    uint2 halfResPixel = dispatchID / 2;
    if (all(dispatchID == HalfToFullRes(halfResPixel, g_GTAOTemporal.FullWidth, g_GTAOTemporal.FullHeight)))
    {
        u_HistoryOutput[halfResPixel] = screenAO;
    }
}

#endif // __AO_TEMPORAL_UPSCALE_HLSL__
