/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __SAMPLE_CONSTANT_BUFFER_H__
#define __SAMPLE_CONSTANT_BUFFER_H__

#if !defined(__cplusplus) // not needed in the port so far
#pragma pack_matrix(row_major) // matrices below are expected in row_major
#else
using namespace donut::math;
#endif

#include "PathTracer/PathTracerShared.h"

#include "PathTracer/PathTracerDebug.hlsli"

#include "PathTracer/Lighting/LightingTypes.hlsli"
#include "PathTracer/Lighting/EnvMap.hlsli"


struct SimpleViewConstants
{
    float4x4    matWorldToView;
    float4x4    matViewToClip;
    float4x4    matWorldToClip;
    float4x4    matWorldToClipNoOffset;
    float4x4    matClipToWorldNoOffset;
    // 
    float2      viewportOrigin;
    float2      viewportSize;
    // 
    float2      viewportSizeInv;
    float2      pixelOffset;
    // 
    float2      clipToWindowScale;
    float2      clipToWindowBias;
};

struct LightControlData
{
    uint AnalyticLightOffset;             // Start index of analytic lights in t_Lights (LightsBaker output).
    uint AnalyticLightCount;
    uint LocalRegionFirstLightIndex;      // Start index of the active local analytic-lights region (point/spot/sphere) inside t_IntroRTXDILights (Intro RTXDI bake output).
    uint LocalRegionNumLights;            // Number of local analytic lights in that region.
    uint InfiniteRegionFirstLightIndex;   // Start index of the active infinite analytic-lights region (directional lights) inside t_IntroRTXDILights.
    uint InfiniteRegionNumLights;         // Number of infinite analytic lights in that region.
    uint LocalRegionNumAnalyticLights;    // A only: local analytic lights, excluding triangles. Streaming RIS (Step 7) samples this subset to avoid triangle dilution.
    uint EnvironmentLightIndex;           // Index of the synthetic kEnvironment light appended after the triangles (valid iff EnvironmentLightPresent). Sampled only by ReSTIR DI's custom env stream.
    uint EnvironmentLightPresent;         // 1 when ReSTIR DI environment-map sampling is enabled and the env light was appended, else 0.
    uint _lightControlPad1;
    uint _lightControlPad2;
    uint _lightControlPad3;
};

struct SampleConstants
{    
    SimpleViewConstants view;
    SimpleViewConstants previousView;
    EnvMapSceneParams envMapSceneParams;
    EnvMapImportanceSamplingParams envMapImportanceSamplingParams;
    PathTracerConstants ptConsts;
    DebugConstants debug;
    float3 denoisingHitParamConsts;

    uint MaterialCount;

    LightControlData lightControl;
};

// Must match SSRPushConstants in SSRPasses.hlsl (96 bytes)
struct SSRPushConstants
{
    float4 viewRow0_P00;      // View row 0 + proj[0][0]
    float4 viewRow1_P11;      // View row 1 + proj[1][1]
    float4 viewRow2_P22;      // View row 2 + proj[2][2]
    float4 projParams;        // rcpP00, rcpP11, P32, nearZ
    float4 screenAndParams;   // width, height, envGGXMaxMip, maxRayDistance
    float4 ssrParams;         // maxSteps (asuint), thickness, jitter, maxMipLevel (asuint)

    uint frameIndex;
};

// Must match SSRBlurConstants in SSRPasses.hlsl (32 bytes)
// Passed as push constants.
struct SSRBlurConstants
{
    uint2   SrcSize;
    uint2   DstSize;

    uint32_t    SrcMipLevel;
    uint32_t    DstMipLevel;
    uint32_t    Padding0;
    uint32_t    Padding1;
};

// Used in a couple of places like multipass postprocess where you want to keep SampleConstants the same for all passes, but send just a few additional per-pass parameters 
// In path tracing used to pass subSampleIndex (when enabled).
// Set as 'push constants' (root constants)
struct SampleMiniConstants
{
    uint4 params;
    uint4 params1;
    uint4 params2;
    uint4 params3;
};

#if !defined(__cplusplus) 
// 2.5D motion vectors
float3 ComputeMotionVector(SimpleViewConstants view, SimpleViewConstants previousView, float3 posW, float3 prevPosW)
{
    float4 clipPos = mul(float4(posW, 1), view.matWorldToClipNoOffset);
    clipPos.xyz /= clipPos.w;
    float4 prevClipPos = mul(float4(prevPosW, 1), previousView.matWorldToClipNoOffset);
    prevClipPos.xyz /= prevClipPos.w;

    if (clipPos.w <= 0 || prevClipPos.w <= 0)
        return float3(0, 0, 0);

    float3 motion;
    motion.xy = (prevClipPos.xy - clipPos.xy) * view.clipToWindowScale;
    //motion.xy += (view.pixelOffset - previousView.pixelOffset); //<- no longer needed, using NoOffset matrices
    motion.z = prevClipPos.w - clipPos.w; // Use view depth

    return motion;
}
#endif

#endif // __SAMPLE_CONSTANT_BUFFER_H__