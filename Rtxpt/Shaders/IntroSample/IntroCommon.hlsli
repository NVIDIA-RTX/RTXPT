/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef __INTRO_COMMON_HLSLI__
#define __INTRO_COMMON_HLSLI__

#include "IntroSampleCommon.hlsli"

#define CLOSEST_HIT_HIGH 0
#define CLOSEST_HIT_LOW 1

struct PBRSurface
{
    float3 baseColor;
    float metal;
    float3 normal;
    float roughness;
    float3 emissiveColor;
    float opacity;
    float diffTx;
    float ao;       // baked material ambient occlusion (ORM red channel)
};

PBRSurface GetDefaultSurface()
{
    PBRSurface result;

    result.baseColor = 1;
    result.metal = 0;
    result.roughness = 0.5;
    result.emissiveColor = 0;
    result.opacity = 1;
    result.diffTx = 0;
    result.ao = 1;

    return result;
}


struct [raypayload] PayloadLite
{
    float3 baseColor : read(caller) : write(closesthit);
    float metal : read(caller) : write(closesthit);

    float3 normal : read(caller) : write(closesthit);
    float roughness : read(caller) : write(closesthit);

    float3 worldPos : read(caller) : write(closesthit);
    float hitDistance : read(caller) : write(closesthit, miss);

    float3 motionVector : read(caller) : write(closesthit);
    uint shaderId : read(caller) : write(closesthit);

    float opacity : read(caller) : write(closesthit);
    float diffTx : read(caller) : write(closesthit);

    float ao : read(caller) : write(closesthit);
};

float4 sampleTextureLOD0(uint textureId, float2 uv)
{
    const uint textureIndex = textureId & 0xFFFF;
    return t_BindlessTextures[NonUniformResourceIndex(textureIndex)].SampleLevel(s_MaterialSampler, uv, 0);
}

MaterialTextureSample EvalMaterialTextures(PTMaterialData material, float2 uv)
{
    MaterialTextureSample textures = DefaultMaterialTextures();

    if ((material.Flags & PTMaterialFlags_UseBaseOrDiffuseTexture) != 0)
        textures.baseOrDiffuse = sampleTextureLOD0(material.BaseOrDiffuseTextureIndex, uv);

    if ((material.Flags & PTMaterialFlags_UseEmissiveTexture) != 0)
        textures.emissive = sampleTextureLOD0(material.EmissiveTextureIndex, uv);

    if ((material.Flags & PTMaterialFlags_UseMetalRoughOrSpecularTexture) != 0)
        textures.metalRoughOrSpecular = sampleTextureLOD0(material.MetalRoughOrSpecularTextureIndex, uv);

#if !defined(RTXPT_MATERIAL_HAS_TRANSMISSION) || RTXPT_MATERIAL_HAS_TRANSMISSION
    if ((material.Flags & PTMaterialFlags_UseTransmissionTexture) != 0)
        textures.transmission = sampleTextureLOD0(material.TransmissionTextureIndex, uv);
#endif

    return textures;
}

PBRSurface EvalMaterialSurface(uint materialIndex, float2 uv)
{
    PTMaterialData material = t_PTMaterialData[materialIndex];

    MaterialTextureSample textures = EvalMaterialTextures(material, uv);

    // Initialize material surface with basic material scalers
    PBRSurface surface = GetDefaultSurface();
    surface.baseColor = material.BaseOrDiffuseColor.rgb * textures.baseOrDiffuse.rgb;

    if ((material.Flags & PTMaterialFlags_UseSpecularGlossModel) != 0)
    {
        // Specular-glossiness workflow. The rest of the Intro pipeline is metal-rough only
        // (the GBuffer stores baseColor + metalness), so reconstruct those here.
        float3 diffuseColor = surface.baseColor;
        float3 specularColor = material.SpecularColor.rgb * textures.metalRoughOrSpecular.rgb;
        surface.roughness = 1.0 - textures.metalRoughOrSpecular.a * (1.0 - material.Roughness);
        ConvertSpecularGlossToMetalRough(diffuseColor, specularColor, surface.baseColor, surface.metal);
    }
    else
    {
        // Metal-rough workflow.
        surface.metal = material.Metalness;
        if ((material.Flags & PTMaterialFlags_MetalnessInRedChannel) != 0)
            surface.metal = material.Metalness * textures.metalRoughOrSpecular.r;
        else
            surface.metal = material.Metalness * textures.metalRoughOrSpecular.b;

        surface.roughness = material.Roughness * textures.metalRoughOrSpecular.g;
    }

    // Material ambient occlusion from the dedicated occlusion texture (glTF stores AO in the red channel).
    // MaterialsBaker points OcclusionTextureIndex at the real occlusion image (which aliases the ORM
    // texture for packed assets and is a standalone texture otherwise), so .r is always occlusion here.
    if ((material.Flags & PTMaterialFlags_UseOcclusionTexture) != 0)
        surface.ao = sampleTextureLOD0(material.OcclusionTextureIndex, uv).r;

    // Opacity
    surface.opacity = saturate(material.Opacity);

#if !defined(RTXPT_MATERIAL_HAS_TRANSMISSION) || RTXPT_MATERIAL_HAS_TRANSMISSION
    float transmission = material.TransmissionFactor;
    surface.diffTx = material.DiffuseTransmissionFactor;
    if ((material.Flags & PTMaterialFlags_UseTransmissionTexture) != 0)
    {
        transmission *= textures.transmission.r;
        surface.diffTx *= textures.transmission.r;
    }
    if(surface.diffTx > 0)
    {
        transmission = 0; // Disambiguate transparent vs diffuse transmission
    }
    surface.opacity *= saturate(1 - transmission);

    if(surface.opacity < 1)
    {
        surface.baseColor = material.Volume.AttenuationColor;
    }
#endif

    // Emissive
    surface.emissiveColor = material.EmissiveColor;
    if ((material.Flags & PTMaterialFlags_UseEmissiveTexture) != 0)
        surface.emissiveColor *= textures.emissive.rgb;
    
    return surface;
}

// Apply tangent-space normal map to a geometry normal.
// Returns the shading normal in world space, or the geometry normal if no normal map is available.
float3 ApplyNormalMap(uint materialIndex, float2 uv, float3 geoNormal, float4 tangent)
{
    PTMaterialData material = t_PTMaterialData[materialIndex];

    if ((material.Flags & PTMaterialFlags_UseNormalTexture) == 0)
        return geoNormal;

    float squareTangentLength = dot(tangent.xyz, tangent.xyz);
    if (squareTangentLength == 0 || tangent.w == 0)
        return geoNormal;

    float4 normalSample = sampleTextureLOD0(material.NormalTextureIndex, uv);

    // Decode normal map (supports both RG-only and RGB encoding)
    normalSample.xy = normalSample.xy * 2.0 - 1.0;
    normalSample.xy *= material.NormalTextureScale;

    if (normalSample.z <= 0)
        normalSample.z = sqrt(saturate(1.0 - normalSample.x * normalSample.x - normalSample.y * normalSample.y));
    else
        normalSample.z = abs(normalSample.z * 2.0 - 1.0);

    float squareLen = dot(normalSample.xyz, normalSample.xyz);
    if (squareLen == 0)
        return geoNormal;

    float3 localNormal = normalSample.xyz / sqrt(squareLen);

    // Build tangent frame from world-space tangent and geometry normal
    float3 T = tangent.xyz * rsqrt(squareTangentLength);
    float3 B = cross(geoNormal, T) * tangent.w;

    return normalize(T * localNormal.x + B * localNormal.y + geoNormal * localNormal.z);
}


#endif // __INTRO_COMMON_HLSLI__