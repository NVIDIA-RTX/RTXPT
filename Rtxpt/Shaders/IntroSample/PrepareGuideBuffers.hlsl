/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// DLSS-RR Guide Buffer Preparation
// Runs after deferred lighting when DLSS-RR is active (RealtimeAA == 3).
// Reads the GBuffer to populate RR guide buffers (diffuse/specular albedo,
// normals+roughness, specular motion vectors) and clamps OutputColor radiance
// using DLSSRRBrightnessClampK.

#define IGNORE_RAY_PAYLOAD 1

#include "../Bindings/ShaderResourceBindings.hlsli"
#include "../Bindings/GBufferBindings.hlsli"
#include "../Bindings/SamplerBindings.hlsli"
#include "IntroBridge.hlsli"

[numthreads(NUM_COMPUTE_THREADS_PER_DIM, NUM_COMPUTE_THREADS_PER_DIM, 1)]
void main(uint2 dispatchThreadID : SV_DispatchThreadID)
{
    const uint2 pixelPos = dispatchThreadID.xy;
    if (any(pixelPos >= uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight)))
        return;

    float depth = u_Depth[pixelPos];

    if (depth > 0)
    {
        PackedGBufferSurface surface;
        surface.LoadFromRenderTargets(pixelPos);

        float3 baseColor = surface.GetBaseColor();
        float metal = surface.GetMetalnes();
        float3 normal = surface.GetSpecNormal();
        float roughness = surface.GetRoughness();
        
        float3 F0 = lerp(0.04, baseColor, metal);
        
        float3 viewDir = -Bridge::computeCameraRay(pixelPos).dir;
        float NdotV = saturate(dot(viewDir, normal));
        
        // Clamp sampler: (NdotV, roughness) LUT must not wrap at its [0,1] edges (see DeferredLightingCombine).
        float2 brdfLUT = t_BRDFLUT.SampleLevel(s_LinearClampSampler, float2(NdotV, roughness), 0).rg;
        float3 avgSpecular = F0 * brdfLUT.x + brdfLUT.y;

        u_RRDiffuseAlbedo[pixelPos]       = float4(baseColor * (1 - metal), 1);
        u_RRSpecAlbedo[pixelPos] = float4(avgSpecular, 1);
        u_RRNormalsAndRoughness[pixelPos] = float4(normal, roughness);
    }
    else
    {
        float3 rayDir = Bridge::computeCameraRay(pixelPos).dir;
        u_RRDiffuseAlbedo[pixelPos]       = float4(0.5, 0.5, 0.5, 1);
        u_RRSpecAlbedo[pixelPos]          = float4(0.5, 0.5, 0.5, 1);
        u_RRNormalsAndRoughness[pixelPos] = float4(-normalize(rayDir), 0.5);
    }

    u_RRSpecMotionVectors[pixelPos] = u_MotionVectors[pixelPos].xy;

    // Clamp radiance for DLSS-RR
    float4 color = u_OutputColor[pixelPos];
    float maxRadiance = max(color.r, max(color.g, color.b));
    if (maxRadiance > g_Const.ptConsts.DLSSRRBrightnessClampK)
        color.rgb *= g_Const.ptConsts.DLSSRRBrightnessClampK / maxRadiance;
    u_OutputColor[pixelPos] = color;
}
