/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// BRDF Integration LUT Generation for Split-Sum IBL Approximation
// Based on Karis 2013: https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
//
// This generates a 2D lookup table where:
//   X-axis: NdotV (cos theta between normal and view) [0, 1]
//   Y-axis: roughness [0, 1]
//   Output: RG16F with (scale, bias) for F0 * scale + bias

#ifndef __BRDF_LUT_GENERATOR_HLSL__
#define __BRDF_LUT_GENERATOR_HLSL__

#define BRDF_LUT_SIZE 64
#define BRDF_LUT_SAMPLE_COUNT 1024

struct BRDFLUTConstants
{
    uint    LUTSize;
    uint    SampleCount;
    uint    Padding0;
    uint    Padding1;
};

#if !defined(__cplusplus)

ConstantBuffer<BRDFLUTConstants>    g_BRDFConst     : register(b0);
RWTexture2D<float2>                 u_BRDFLUT       : register(u0);

static const float PI = 3.14159265359;

//--------------------------------------------------------------------------------------------------
// Van der Corput radical inverse (base 2)
//--------------------------------------------------------------------------------------------------
float RadicalInverse_VdC(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

//--------------------------------------------------------------------------------------------------
// Hammersley low-discrepancy sequence
//--------------------------------------------------------------------------------------------------
float2 Hammersley(uint i, uint N)
{
    return float2(float(i) / float(N), RadicalInverse_VdC(i));
}

//--------------------------------------------------------------------------------------------------
// GGX importance sampling
// Returns halfway vector in tangent space (Z-up)
//--------------------------------------------------------------------------------------------------
float3 ImportanceSampleGGX(float2 Xi, float roughness)
{
    float a = roughness * roughness;
    
    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    
    return float3(
        cos(phi) * sinTheta,
        sin(phi) * sinTheta,
        cosTheta
    );
}

//--------------------------------------------------------------------------------------------------
// Geometry function (Schlick-GGX with k for IBL)
//--------------------------------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float a = roughness;
    float k = (a * a) / 2.0;
    
    return NdotV / (NdotV * (1.0 - k) + k);
}

//--------------------------------------------------------------------------------------------------
// Smith's geometry function (combined shadowing-masking)
//--------------------------------------------------------------------------------------------------
float GeometrySmith(float NdotV, float NdotL, float roughness)
{
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    return ggx1 * ggx2;
}

//--------------------------------------------------------------------------------------------------
// Integrate GGX BRDF for given NdotV and roughness
// Returns (scale, bias) for F0 * scale + bias
//--------------------------------------------------------------------------------------------------
float2 IntegrateBRDF(float NdotV, float roughness, uint sampleCount)
{
    // View direction in tangent space (Z-up)
    float3 V = float3(sqrt(1.0 - NdotV * NdotV), 0.0, NdotV);
    
    // Normal points straight up in tangent space
    float3 N = float3(0.0, 0.0, 1.0);
    
    float A = 0.0;
    float B = 0.0;
    
    for (uint i = 0; i < sampleCount; i++)
    {
        float2 Xi = Hammersley(i, sampleCount);
        float3 H = ImportanceSampleGGX(Xi, roughness);
        
        // Compute light direction from halfway vector
        float3 L = 2.0 * dot(V, H) * H - V;
        
        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);
        
        if (NdotL > 0.0)
        {
            // Geometry term
            float G = GeometrySmith(NdotV, NdotL, roughness);
            
            // Visibility term
            float G_Vis = (G * VdotH) / (NdotH * NdotV + 0.0001);
            
            // Fresnel term (Schlick approximation)
            float Fc = pow(1.0 - VdotH, 5.0);
            
            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    
    A /= float(sampleCount);
    B /= float(sampleCount);
    
    return float2(A, B);
}

//--------------------------------------------------------------------------------------------------
// Main compute shader - generate BRDF integration LUT
//--------------------------------------------------------------------------------------------------
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint x = dispatchThreadID.x;
    uint y = dispatchThreadID.y;
    
    // Early out if beyond LUT size
    if (x >= g_BRDFConst.LUTSize || y >= g_BRDFConst.LUTSize)
        return;
    
    // Map pixel coordinates to (NdotV, roughness)
    // X-axis: NdotV [0, 1]
    // Y-axis: roughness [0, 1]
    float NdotV = (float(x) + 0.5) / float(g_BRDFConst.LUTSize);
    float roughness = (float(y) + 0.5) / float(g_BRDFConst.LUTSize);
    
    // Clamp NdotV to avoid division by zero
    NdotV = max(NdotV, 0.001);
    
    // Clamp roughness to avoid issues at 0
    roughness = max(roughness, 0.001);
    
    // Integrate BRDF for this (NdotV, roughness) pair
    float2 brdf = IntegrateBRDF(NdotV, roughness, g_BRDFConst.SampleCount);
    
    // Write to LUT (R = scale, G = bias)
    u_BRDFLUT[uint2(x, y)] = brdf;
}

#endif // !defined(__cplusplus)

#endif // __BRDF_LUT_GENERATOR_HLSL__
