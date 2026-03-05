/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __GBUFFER_BINDINGS_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __GBUFFER_BINDINGS_HLSLI__

#include "../PathTracer/Utils/utils.hlsli"
#include <donut/shaders/utils.hlsli>

enum GBufferShaderId
{
    ShaderIdInvalid = 0,
    ShaderIdEmissive = 1,
    ShaderIdStandardGGX = 2,
    ShaderIdGlassGGX = 3
};

// G-Buffer bindings
RWTexture2D<float4> u_GBufferBaseColor : register(u100); // RGB_10_10_11
RWTexture2D<uint> u_GBufferSpecNormal32 : register(u101); // u32 octahedral packed
RWTexture2D<float2> u_GBufferRoughnessMetal : register(u102); // rg16f: roughness + metalness
RWTexture2D<uint> u_GBufferMaterialInfo : register(u103); // u32: 8 bit "BxDF index", 8 bit ambient occlusion, 8 custom BxDF packed data

struct PackedGBufferSurface
{
    half3 baseColor;
    
    uint packedSpecNormal;
    half2 roughnessMetal;
    
    uint packedMaterialInfo;
    
    void SetMaterialInfo(uint shaderModel, float AO)
    {
        uint packedAO = min(255, uint(AO * 255 + 0.5));
        packedMaterialInfo = ((shaderModel & 0xff) << 24) | (packedAO << 16);
    }
    
    uint getShaderModel()
    {
        return packedMaterialInfo >> 24;
    }
    
    float getAO()
    {
        return ((packedMaterialInfo >> 16) & 0xff) / 255.0;
    }
    
    void SetBaseColor(float3 color)
    {
        baseColor = half3(color);
    }
    
    float3 GetBaseColor()
    {
        return baseColor;
    }
    
    void SetSpecNormal(float3 normal)
    {
        packedSpecNormal = ndirToOctUnorm32(normal);
    }
    
    float3 GetSpecNormal()
    {
        return octToNdirUnorm32(packedSpecNormal);
    }
    
    void SetRoughnessMetal(float roughness, float metalness)
    {
        roughnessMetal = half2(roughness, metalness);
    }
    
    float GetRoughness()
    {
        return float(roughnessMetal.x);
    }
    
    float GetMetalnes()
    {
        return float(roughnessMetal.y);
    }
    
    void StoreIntoRenderTargets(uint2 pixelPos)
    {
        u_GBufferBaseColor[pixelPos] = half4(baseColor, 1);
        u_GBufferSpecNormal32[pixelPos] = packedSpecNormal;
        u_GBufferRoughnessMetal[pixelPos] = roughnessMetal;
        u_GBufferMaterialInfo[pixelPos] = packedMaterialInfo;
    }
    
    void LoadFromRenderTargets(uint2 pixelPos)
    {
        baseColor = (half3)u_GBufferBaseColor[pixelPos].xyz;
        packedSpecNormal = u_GBufferSpecNormal32[pixelPos];
        roughnessMetal = (half2)u_GBufferRoughnessMetal[pixelPos];
        packedMaterialInfo = u_GBufferMaterialInfo[pixelPos];
    }
};

#endif // #ifndef __GBUFFER_BINDINGS_HLSLI__
