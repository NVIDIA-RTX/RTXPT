/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __INTRO_MATERIAL_EVAL_SIMPLIFIED_HLSLI__
#define __INTRO_MATERIAL_EVAL_SIMPLIFIED_HLSLI__

#include "../PathTracer/PathTracerTypes.hlsli"
#include "../Bindings/ShaderResourceBindings.hlsli"
#include "../Bindings/GBufferBindings.hlsli"

struct SimplifiedSurface
{
    float3 baseColor;
    uint shaderId;
};

float4 sampleTextureLOD(uint textureId, float2 uv, uint lod)
{
    const uint textureIndex = textureId & 0xFFFF;
    return t_BindlessTextures[NonUniformResourceIndex(textureIndex)].SampleLevel(s_MaterialSampler, uv, lod);
}

MaterialTextureSample EvalSimplifiedMaterialTextures(PTMaterialData material, float2 uv, uint lod)
{
    MaterialTextureSample textures = DefaultMaterialTextures();
    
    if ((material.Flags & PTMaterialFlags_UseBaseOrDiffuseTexture) != 0)
    {
        textures.baseOrDiffuse = sampleTextureLOD(material.BaseOrDiffuseTextureIndex, uv, lod);
    }

    if ((material.Flags & PTMaterialFlags_UseEmissiveTexture) != 0)
    {
        textures.emissive = sampleTextureLOD(material.EmissiveTextureIndex, uv, lod);
    }
        
    return textures;
}

SimplifiedSurface EvalSimplifiedSurface(uint materialIndex, float2 uv)
{
    PTMaterialData material = t_PTMaterialData[materialIndex];
    
    const uint texLOD = 5; // Enough offset to make this sample cheaper
    MaterialTextureSample textures = EvalSimplifiedMaterialTextures(material, uv, texLOD);

    // Initialize material surface with basic material scalers
    SimplifiedSurface surface;
    surface.baseColor = material.BaseOrDiffuseColor.rgb;
    
    if ((material.Flags & PTMaterialFlags_UseBaseOrDiffuseTexture) != 0)
        surface.baseColor *= textures.baseOrDiffuse.rgb;
    
    // Opacity
    float opacity = saturate(material.Opacity);

#if !defined(RTXPT_MATERIAL_HAS_TRANSMISSION) || RTXPT_MATERIAL_HAS_TRANSMISSION
    float transmission = material.TransmissionFactor;
    float diffTx = material.DiffuseTransmissionFactor;
    
    if (diffTx > 0)
    {
        transmission = 0; // Disambiguate transparent vs diffuse transmission
    }
    opacity *= saturate(1 - transmission);
    
    if (opacity < 1)
    {
        surface.baseColor = material.Volume.AttenuationColor;
    }
#endif
    
    // Emissive
    float3 emissiveColor = material.EmissiveColor;
    if ((material.Flags & PTMaterialFlags_UseEmissiveTexture) != 0)
    {
        emissiveColor *= textures.emissive.rgb;
    }
    float maxEmissive = max(emissiveColor.x, max(emissiveColor.y, emissiveColor.z));
    surface.baseColor = maxEmissive > 0 ? emissiveColor : surface.baseColor;
    
    // Shader ID
    surface.shaderId = maxEmissive > 0 ? ShaderIdEmissive : (opacity < 1 ? ShaderIdGlassGGX : ShaderIdStandardGGX);
    
    return surface;
}

#endif // __INTRO_MATERIAL_EVAL_SIMPLIFIED_HLSLI__
