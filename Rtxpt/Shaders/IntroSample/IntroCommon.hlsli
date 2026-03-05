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


struct PBRSurface
{
    float3 baseColor;
    float3 normal;
    float metal;
    float emissiveScale;
    float roughness;
    float alpha;
};

struct [raypayload] PayloadLite
{
    float3 baseColor : read(caller) : write(closesthit);
    float metal : read(caller) : write(closesthit);

    float3 normal : read(caller) : write(closesthit);
    float roughness : read(caller) : write(closesthit);

    float3 worldPos : read(caller) : write(closesthit);
    float3 motionVector : read(caller) : write(closesthit);
    float ambientOcclusion : read(caller) : write(closesthit);
    float hitDistance : read(caller) : write(closesthit, miss);

    uint shaderId : read(caller) : write(closesthit);
};

float4 sampleTextureLOD0(uint textureId, float2 uv)
{
    const uint textureIndex = textureId & 0xFFFF;
    return t_BindlessTextures[textureIndex].SampleLevel(s_MaterialSampler, uv, 0);
}

PBRSurface EvalMaterialSurface(uint materialIndex, float2 uv, float3 wsNormal)
{
    PTMaterialData material = t_PTMaterialData[materialIndex];

    MaterialTextureSample textures = DefaultMaterialTextures();

    // Initialize material surface with basic material scalers
    PBRSurface surface;
    surface.baseColor = material.BaseOrDiffuseColor.rgb;
    surface.metal = material.Metalness;
    surface.emissiveScale = 0;
    surface.roughness = material.Roughness;

    if ((material.Flags & PTMaterialFlags_UseEmissiveTexture) != 0)
        material.EmissiveColor *= sampleTextureLOD0(material.EmissiveTextureIndex, uv).xyz;

    if ((material.Flags & PTMaterialFlags_UseBaseOrDiffuseTexture) != 0)
        textures.baseOrDiffuse = sampleTextureLOD0(material.BaseOrDiffuseTextureIndex, uv);

    if ((material.Flags & PTMaterialFlags_UseMetalRoughOrSpecularTexture) != 0)
        textures.metalRoughOrSpecular = sampleTextureLOD0(material.MetalRoughOrSpecularTextureIndex, uv);

    if ((material.Flags & PTMaterialFlags_MetalnessInRedChannel) != 0)
        surface.metal *= textures.metalRoughOrSpecular.r;
    else
        surface.metal *= textures.metalRoughOrSpecular.b;

    surface.roughness *= textures.metalRoughOrSpecular.g;
    surface.baseColor *= textures.baseOrDiffuse.rgb;
    surface.normal = wsNormal; // TODO: Normal maps

    if (any(material.EmissiveColor) > 0)
    {
        surface.emissiveScale = 1;
        surface.baseColor = material.EmissiveColor;
    }

    float opacity = material.Opacity;
    if ((material.Flags & MaterialFlags_UseOpacityTexture) != 0)
        opacity *= textures.opacity;
    else if ((material.Flags & MaterialFlags_UseBaseOrDiffuseTexture) != 0)
        opacity *= textures.baseOrDiffuse.a;
    
    opacity = saturate(opacity);
    if (material.TransmissionFactor > 0.9) // Hack to separate vegetation assets from some glass in Bistro.
        opacity *= (1 - material.TransmissionFactor);
        
    if (opacity < 1)
    {
        surface.baseColor = material.Volume.AttenuationColor;
    }
    
    // Assuming alpha blend here, since alpha test must be handled either with OMMs or in the anyHit shader
    surface.alpha = opacity;
    
    if ((material.Flags & PTMaterialFlags_UseSpecularGlossModel) != 0)
    {
        surface.roughness = lpfloat(1.0 - textures.metalRoughOrSpecular.a * (1.0 - material.Roughness));
    }
    
    return surface;
}

float ToFloat0To1(uint u)
{
    return asfloat(0x3f800000 | (u & 0x7fffff)) - 1.f;
}

float D_GGX(float3 m, float a2)
{
    float D = a2 / max(1e-4, M_PI * square(m.z * a2 + (1 - m.z)));
    return D;
}

float GGXReflectionPDF(float3 i, float3 m, float alpha, float a2)
{
    float ndf = D_GGX(m, a2);
    float ai = alpha * i.x;
    float len2 = ai * ai;
    float t = sqrt(len2 + i.z * i.z);
    if (i.z >= 0.0f)
    {
        float s = 1.0f + i.x; // Omit sgn for a <=1
        float s2 = s * s;
        float k = (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z); // Eq. 5
        return max(1e-3, ndf / (2.0 * (k * i.z + t))); // Eq. 8 * || dm/do ||
    }
    // Numerically stable form of the previous PDF for i.z < 0
    return ndf * (t - i.z) / (2.0f * len2); // = Eq. 7 * || dm/do ||
}

float3 SampleVNDF(float2 rand, float3 view, float3 normal, float alpha, out float invPdfVNDF)
{
    // Project incoming direction into a local space
    float3 i;
    i.z = dot(view, normal);
    i.x = sqrt(1 - i.z * i.z);
    i.y = 0;
        
    float r = sqrt(max(1e-3, alpha));
    float3 H = ImportanceSampleGGX_VNDF(rand, r, i, 0.98);
    
    // Project m into the original space
    float3 tangent = normalize(view - normal * i.z);
    float3 bitangent = cross(normal, tangent);
    
    float3 m = normalize(tangent * H.x + bitangent * H.y + normal * H.z);
    
    float3 refl = reflect(-view, m);
    
    invPdfVNDF = 1.0 / ImportanceSampleGGX_VNDF_PDF(r, normal, view, refl);
    
    // return the half vector in world space
    return m;
}


#endif // __INTRO_COMMON_HLSLI__