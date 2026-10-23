/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Cubemap Processing Shaders
// - GGX pre-filtering for specular IBL
// - SH-based diffuse irradiance convolution

#ifndef __CUBEMAP_PROCESSING_HLSL__
#define __CUBEMAP_PROCESSING_HLSL__

#define CUBEMAP_PROCESS_THREADS 8

// Core config first so STATIC_ASSERT (and pack_matrix) are defined before EnvMap.hlsli pulls in
// LightingTypes.hlsli -> NEEATShared.hlsli, which uses STATIC_ASSERT unguarded. Other shaders get
// this ordering implicitly via their bindings preamble; this standalone pass needs it explicitly.
#include "../../PathTracer/Config.h"
// Shared C++/HLSL. Provides EnvMapSceneParams (and, in the shader, EnvMap) so the prefilter
// passes can sample the source cube through the same rotation/intensity/tint transform the
// runtime path uses.
#include "../../PathTracer/Lighting/EnvMap.hlsli"

struct GGXPrefilterConstants
{
    uint    SrcCubemapSize;
    uint    DstMipSize;
    uint    MipLevel;
    uint    MaxMipLevels;
    uint    SampleCount;
    float   Roughness;
    uint    Padding1;
    uint    Padding2;
    EnvMapSceneParams EnvParams;
};

struct IrradianceConvolveConstants
{
    uint    SrcCubemapSize;
    uint    DstCubemapSize;
    uint    SampleCount;
    uint    Padding0;
    EnvMapSceneParams EnvParams;
};

#if !defined(__cplusplus)

// Low-discrepancy sampling (Hammersley/RadicalInverse_VdC)
#include "../../PathTracer/Utils/NoiseAndSequences.hlsli"
#include "CubemapUtils.hlsli"

//==================================================================================================
// Common Utilities
//==================================================================================================

// GGX importance sampling - returns halfway vector in tangent space
float3 ImportanceSampleGGX(float2 Xi, float roughness)
{
    float a = roughness * roughness;
    
    float phi = K_2PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

// GGX Normal Distribution Function
float DistributionGGX(float NdotH, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;
    
    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = K_PI * denom * denom;
    
    return nom / max(denom, 0.0001);
}

//==================================================================================================
// GGX Pre-filtering Shader
//==================================================================================================

ConstantBuffer<GGXPrefilterConstants>   g_GGXConst          : register(b0);
TextureCube<float4>                     t_SrcCubemap        : register(t0);
RWTexture2DArray<float4>                u_DstCubemapMip     : register(u0);
SamplerState                            s_LinearSampler     : register(s0);

[numthreads(CUBEMAP_PROCESS_THREADS, CUBEMAP_PROCESS_THREADS, 1)]
void GGXPrefilterCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint x = dispatchThreadID.x;
    uint y = dispatchThreadID.y;
    uint face = dispatchThreadID.z;
    
    if (x >= g_GGXConst.DstMipSize || y >= g_GGXConst.DstMipSize)
        return;
    
    EnvMap envMap = EnvMap::make(t_SrcCubemap, s_LinearSampler, g_GGXConst.EnvParams);
    
    // Mip 0: direct copy (no filtering for roughness=0)
    float2 uv = (float2(x, y) + 0.5) / float(g_GGXConst.DstMipSize);
    float3 dir = CubemapGetDirectionFor(face, uv);
    if (g_GGXConst.MipLevel == 0)
    {
        u_DstCubemapMip[uint3(x, y, face)] = float4(envMap.Eval(dir, 0), 1);
        return;
    }
    
    float roughness = g_GGXConst.Roughness;
    
    float3 N = dir;
    
    // Build tangent space basis
    float3 tangent, bitangent;
    BranchlessONB(N, tangent, bitangent);
    
    float3 prefilteredColor = 0;
    float totalWeight = 0;
    float3 faceAvg = envMap.Eval(dir, 9);
    
    uint sampleCount = g_GGXConst.SampleCount;
    for (uint i = 0; i < 64; i++)
    {
        float2 Xi = Hammersley(i, sampleCount);
        Xi.y *= 0.9;
        float3 H = ImportanceSampleGGX(Xi, roughness);
        float3 L = 2.0 * H.z * H - float3(0, 0, 1);
        L = L.x * tangent + L.y * bitangent + L.z * N;
        
        float NdotL = dot(N, L);
        if (NdotL > 0)
        {
            // PDF-based LOD to reduce noise
            float NdotH = max(H.z, 0);
            float D = DistributionGGX(NdotH, roughness);
            float pdf = D / 4.0;
            
            float solidAngleTexel = K_4PI / (6.0 * g_GGXConst.SrcCubemapSize * g_GGXConst.SrcCubemapSize); // Area of the unit sphere over total number of texels.
            float solidAngleSample = K_4PI / (sampleCount * D + 0.0001);
            float lod = max(0, min(0.5 * log2(solidAngleSample / solidAngleTexel), g_GGXConst.MipLevel));
            
            float3 radiance = envMap.Eval(L, lod);
            radiance = min(radiance, 4 * faceAvg); // Limit fireflies
            prefilteredColor += radiance * NdotL;
            totalWeight += NdotL;
        }
    }
    
    u_DstCubemapMip[uint3(x, y, face)] = float4(prefilteredColor / max(totalWeight, 0.001), 1);
}

//==================================================================================================
// SH-Based Irradiance Convolution Shader
//==================================================================================================

struct SHOrder2
{
    float3 c[9];
};

SHOrder2 ProjectToSH(float3 n, float3 color)
{
    SHOrder2 result;
    result.c[0] = color;                              // L0
    result.c[1] = color * n.x;                        // L1
    result.c[2] = color * n.y;
    result.c[3] = color * n.z;
    result.c[4] = color * (n.x * n.y);                // L2
    result.c[5] = color * (n.y * n.z);
    result.c[6] = color * (3 * n.z * n.z - 1);
    result.c[7] = color * (n.z * n.x);
    result.c[8] = color * (n.x * n.x - n.y * n.y);
    return result;
}

float3 EvaluateSH(SHOrder2 sh, float3 n)
{
    float3 result = sh.c[0];
    result += sh.c[1] * (2 * n.x) + sh.c[2] * (2 * n.y) + sh.c[3] * (2 * n.z);
    result += sh.c[4] * (15.0 / 4.0 * n.x * n.y);
    result += sh.c[5] * (15.0 / 4.0 * n.y * n.z);
    result += sh.c[6] * (5.0 / 16.0 * (3 * n.z * n.z - 1));
    result += sh.c[7] * (15.0 / 4.0 * n.z * n.x);
    result += sh.c[8] * (15.0 / 16.0 * (n.x * n.x - n.y * n.y));
    return result;
}

SHOrder2 WaveReduceSH(SHOrder2 sh)
{
    [unroll]
    for (int i = 0; i < 9; i++)
        sh.c[i] = WaveActiveSum(sh.c[i]) / float(WaveGetLaneCount());
    return sh;
}

ConstantBuffer<IrradianceConvolveConstants> g_IrradConst        : register(b0);
TextureCube<float4>                         t_SrcCubemapIrrad   : register(t0);
RWTexture2DArray<float4>                    u_DstIrradianceCube : register(u0);
SamplerState                                s_LinearSamplerIrrad : register(s0);

groupshared SHOrder2 s_SH[32];

[numthreads(1024, 1, 1)]
void ConvolveIrradianceCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint threadIdx = dispatchThreadID.x;
    uint face = dispatchThreadID.z;
    
    // Sample sphere using Fibonacci distribution
    float3 sampleDir = GenerateFibonacciSphereSample(threadIdx, g_IrradConst.SampleCount);
    // Sample at a higher mip level for better integration
    EnvMap envMap = EnvMap::make(t_SrcCubemapIrrad, s_LinearSamplerIrrad, g_IrradConst.EnvParams);
    float3 radiance = envMap.Eval(sampleDir, 5.0);
    
    SHOrder2 sh = ProjectToSH(sampleDir, radiance);
    
    // Wave reduction (assumes 32 threads per wave)
    sh = WaveReduceSH(sh);
    
    // First lane of each wave writes to shared memory
    uint waveIdx = threadIdx / WaveGetLaneCount();
    if (WaveIsFirstLane())
        s_SH[waveIdx] = sh;
    
    GroupMemoryBarrierWithGroupSync();
    
    // Second reduction across waves
    uint x = threadIdx % 32;
    uint y = threadIdx / 32;
    sh = s_SH[x];
    
    sh = WaveReduceSH(sh);

    // Each thread writes one output texel (32x32 = 1024)
    float2 uv = (float2(x, y) + 0.5) / float(g_IrradConst.DstCubemapSize);
    float3 dir = CubemapGetDirectionFor(face, uv);
    float3 irradiance = max(0, EvaluateSH(sh, dir));

    u_DstIrradianceCube[uint3(x, y, face)] = float4(irradiance, 1);
}

#endif // !defined(__cplusplus)

#endif // __CUBEMAP_PROCESSING_HLSL__
