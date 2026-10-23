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
    ShaderIdGlassGGX = 3,
    ShaderIdDiffuseTx = 4
};

// G-Buffer bindings
RWTexture2D<float4> u_GBufferBaseColor : register(u100); // RGB_10_10_11
RWTexture2D<uint> u_GBufferSpecNormal32 : register(u101); // u32 octahedral packed
RWTexture2D<float2> u_GBufferRoughnessMetal : register(u102); // rg16f: roughness + metalness
RWTexture2D<uint> u_GBufferMaterialInfo : register(u103); // u32: 8 bit "BxDF index", 8 bit opacity, 8 bit diffuse transmission, 8 bit free

struct PackedGBufferSurface
{
    half3 baseColor;
    
    uint packedSpecNormal;
    half2 roughnessMetal;
    
    uint packedMaterialInfo;
    
    void SetMaterialInfo(uint shaderId, float opacity, float diffTx = 0)
    {
        uint packedOpacity = min(255, uint(opacity * 255 + 0.5));
        uint packedTx = min(255, uint(diffTx * 255 + 0.5));
        // Preserve the low byte (material AO), it may have been set independently.
        packedMaterialInfo = ((shaderId & 0xff) << 24) | (packedOpacity << 16) | (packedTx << 8) | (packedMaterialInfo & 0xff);
    }

    // Material ambient occlusion is stored in the otherwise-free low byte of packedMaterialInfo.
    void SetMaterialAO(float ao)
    {
        uint packedAO = min(255, uint(saturate(ao) * 255 + 0.5));
        packedMaterialInfo = (packedMaterialInfo & 0xffffff00) | packedAO;
    }

    float GetMaterialAO()
    {
        return (packedMaterialInfo & 0xff) / 255.0;
    }
    
    uint GetShaderId()
    {
        return packedMaterialInfo >> 24;
    }
    
    float GetOpacity()
    {
        return ((packedMaterialInfo >> 16) & 0xff) / 255.0;
    }
    
    float GetDiffuseTx()
    {
        return ((packedMaterialInfo >> 8) & 0xff) / 255.0;
    }
    
    void SetBaseColor(float3 color)
    {
        baseColor = half3(color);
    }
    
    float3 GetBaseColor()
    {
        float3 color = baseColor;
        if (GetShaderId() == ShaderIdEmissive)
        {
            // Emissives are stored in baseColor and can be above 1. To conserve energy, normalize by the highest component.
            color /= max(color.x, max(color.y, color.z));
        }
        return color;
    }
    
    float3 GetEmissive()
    {
        // Emissives are stored in baseColor in this GBuffer layout as an optimization.
        return (GetShaderId() == ShaderIdEmissive) ? baseColor : 0;
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
