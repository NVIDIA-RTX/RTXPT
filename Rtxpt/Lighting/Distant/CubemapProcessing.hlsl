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
};

struct IrradianceConvolveConstants
{
    uint    SrcCubemapSize;
    uint    DstCubemapSize;
    uint    SampleCount;
    uint    Padding;
};

#if !defined(__cplusplus)

static const float PI = 3.14159265359;

//==================================================================================================
// Common Utilities
//==================================================================================================

// Convert cubemap face UV to 3D direction vector
float3 CubemapUVToDirection(uint face, float2 uv)
{
    float2 ndc = uv * 2.0 - 1.0;
    
    float3 dir;
    switch (face)
    {
        case 0: dir = float3(1.0, -ndc.y, -ndc.x); break;  // +X
        case 1: dir = float3(-1.0, -ndc.y, ndc.x); break;  // -X
        case 2: dir = float3(ndc.x, 1.0, ndc.y); break;    // +Y
        case 3: dir = float3(ndc.x, -1.0, -ndc.y); break;  // -Y
        case 4: dir = float3(ndc.x, -ndc.y, 1.0); break;   // +Z
        case 5: dir = float3(-ndc.x, -ndc.y, -1.0); break; // -Z
        default: dir = float3(0, 0, 1); break;
    }
    
    return normalize(dir);
}

// Van der Corput radical inverse (base 2)
float RadicalInverse_VdC(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

// Hammersley low-discrepancy sequence
float2 Hammersley(uint i, uint N)
{
    return float2(float(i) / float(N), RadicalInverse_VdC(i));
}

// Build orthonormal basis from normal
void BuildOrthonormalBasis(float3 normal, out float3 tangent, out float3 bitangent)
{
    float3 up = abs(normal.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    tangent = normalize(cross(up, normal));
    bitangent = cross(normal, tangent);
}

// Transform from tangent space to world space
float3 TangentToWorld(float3 dirTS, float3 normal, float3 tangent, float3 bitangent)
{
    return dirTS.x * tangent + dirTS.y * bitangent + dirTS.z * normal;
}

// GGX importance sampling - returns halfway vector in tangent space
float3 ImportanceSampleGGX(float2 Xi, float roughness)
{
    float a = roughness * roughness;
    
    float phi = 2.0 * PI * Xi.x;
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
    denom = PI * denom * denom;
    
    return nom / max(denom, 0.0001);
}

// Fibonacci sphere sampling for uniform distribution
float3 GenerateFibonacciSphereSample(uint sampleIndex, uint totalSamples)
{
    float goldenRatio = 1.618033988749895;
    float angleIncrement = PI * 2.0 * goldenRatio;
    
    float i = float(sampleIndex);
    float n = float(totalSamples);
    
    float a = 1 - 2 * (i / (n - 1));
    float b = sqrt(1 - a * a);
    float phi = angleIncrement * i;
    
    return float3(cos(phi) * b, sin(phi) * b, a);
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
    
    // Mip 0: direct copy (no filtering for roughness=0)
    float2 uv = (float2(x, y) + 0.5) / float(g_GGXConst.DstMipSize);
    float3 dir = CubemapUVToDirection(face, uv);
    if (g_GGXConst.MipLevel == 0)
    {
        // Sample at mip 0 using the cubemap SRV
        u_DstCubemapMip[uint3(x, y, face)] = t_SrcCubemap.SampleLevel(s_LinearSampler, dir, 0);
        return;
    }
    
    float roughness = g_GGXConst.Roughness;
    
    float3 N = dir;
    
    // Build tangent space basis
    float3 tangent, bitangent;
    BuildOrthonormalBasis(N, tangent, bitangent);
    
    float3 prefilteredColor = 0;
    float totalWeight = 0;
    float3 faceAvg = t_SrcCubemap.SampleLevel(s_LinearSampler, dir, 9).rgb;
    
    uint sampleCount = g_GGXConst.SampleCount;
    for (uint i = 0; i < 64; i++)
    {
        float2 Xi = Hammersley(i, sampleCount);
        Xi.y *= 0.9;
        float3 H = ImportanceSampleGGX(Xi, roughness);
        float3 L = 2.0 * H.z * H - float3(0, 0, 1);
        L = TangentToWorld(L, N, tangent, bitangent);
        
        float NdotL = dot(N, L);
        if (NdotL > 0)
        {
            // PDF-based LOD to reduce noise
            float NdotH = max(H.z, 0);
            float D = DistributionGGX(NdotH, roughness);
            float pdf = D / 4.0;
            
            float solidAngleTexel = 4.0 * PI / (6.0 * g_GGXConst.SrcCubemapSize * g_GGXConst.SrcCubemapSize); // Area of the unit sphere over total number of texels.
            float solidAngleSample = 4.0 * PI / (sampleCount * D + 0.0001);
            float lod = max(0, min(0.5 * log2(solidAngleSample / solidAngleTexel), g_GGXConst.MipLevel));
            
            float3 radiance = t_SrcCubemap.SampleLevel(s_LinearSampler, L, lod).rgb;
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
    float3 radiance = t_SrcCubemapIrrad.SampleLevel(s_LinearSamplerIrrad, sampleDir, 5.0).rgb;
    
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
    float3 dir = CubemapUVToDirection(face, uv);
    float3 irradiance = max(0, EvaluateSH(sh, dir));
        
    u_DstIrradianceCube[uint3(x, y, face)] = float4(irradiance, 1);
}

#endif // !defined(__cplusplus)

#endif // __CUBEMAP_PROCESSING_HLSL__
