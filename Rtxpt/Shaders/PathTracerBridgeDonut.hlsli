/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __PATH_TRACER_BRIDGE_DONUT_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __PATH_TRACER_BRIDGE_DONUT_HLSLI__

// easier if we let Donut do this!
#define ENABLE_METAL_ROUGH_RECONSTRUCTION 1

#include "PathTracerBridge.hlsli"
#include "PathTracer/Materials/MaterialTypes.hlsli"

#include "Misc/OmmDebug.hlsli"

// Donut-specific (native engine - we can include before PathTracer to avoid any collisions)
#include <donut/shaders/bindless.h>
#include <donut/shaders/utils.hlsli>
#include <donut/shaders/binding_helpers.hlsli>
#include <donut/shaders/surface.hlsli>
#include <donut/shaders/scene_material.hlsli>

#include "Bindings/SceneBindings.hlsli"
#include "Bindings/LightingBindings.hlsli"
#include "Bindings/SamplerBindings.hlsli"

#include "Libraries\MicroRng.hlsli"


#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES // build
#if PT_USE_RESTIR_GI || PT_USE_RESTIR_DI
#define EXPORT_GBUFFER 1
#endif
#endif

#if EXPORT_GBUFFER
#include "Misc/GBufferHelpers.hlsli"
#endif

enum DonutGeometryAttributes
{
    GeomAttr_Position       = 0x01,
    GeomAttr_TexCoord       = 0x02,
    GeomAttr_Normal         = 0x04,
    GeomAttr_Tangents       = 0x08,
    GeomAttr_PrevPosition   = 0x10,

    GeomAttr_All            = 0x1F
};

struct DonutGeometrySample
{
    InstanceData instance;

    // this should be removed
    GeometryData geometry;

    // this should be under a macro
    GeometryDebugData geometryDebug;

    float3 vertexPositions[3];      //< object space vertex positions, for world pos do "mul(instance.transform, float4(positions[0], 1)).xyz"
    //float3 prevVertexPositions[3]; <- not needed for anything yet so we just use local variables and compute prevObjectSpacePosition
    float2 vertexTexcoords[3];

    float3 objectSpacePosition;
    float3 prevObjectSpacePosition;
    float2 texcoord;
    float3 flatNormal;
    float3 geometryNormal;
    float4 tangent;
    bool frontFacing;
    float curvatureWS;                //< rough, approx world space relative curvature metric
};

float3 SafeNormalize(float3 input)
{
    float lenSq = dot(input,input);
    return input * rsqrt(max( 1.175494351e-38, lenSq));
}

float3 FlipIfOpposite(float3 normal, float3 referenceNormal)
{
    return (dot(normal, referenceNormal)>=0)?(normal):(-normal);
}

// Returns a rough curvature proxy for a single triangle, in ~1/length units.
// Based on fitting a linear normal field over the triangle in a local 2D basis.
// Inputs:
//   vertexPositions[3] : triangle positions in the same space/units you care about (world or object).
//   vertexNormals[3]   : per-vertex normals (need not be normalized; we normalize them).
//
// Output:
//   curvature : higher => normals vary rapidly over short distances (more "curved").
//
// Notes:
// - Degenerate triangles return 0.
// - If your vertex normals are not consistently oriented / smoothed, curvature will be noisy by design.
float TriangleCurvatureApprox_GradN(float3 vertexPositions[3], float3 vertexNormals[3], float3x4 transform)
{
    const float eps = 1e-8f;

    // Build orthonormal basis on the triangle plane.
    float3 e10 = mul((float3x3)transform, vertexPositions[1] - vertexPositions[0]);
    float  e10Len = length(e10);
    if (e10Len < eps)
    {
        return 0.0f;
    }

    float3 e1 = e10 / e10Len;

    float3 e20 = mul((float3x3)transform, vertexPositions[2] - vertexPositions[0]);
    float  u2  = dot(e20, e1);

    float3 t   = e20 - e1 * u2;
    float  tLen = length(t);
    if (tLen < eps)
    {
        // Nearly colinear triangle; no stable 2D basis.
        return 0.0f;
    }

    float3 e2 = t / tLen;

    // 2D coordinates in this basis:
    // p1 is at (u1, v1) = (e10Len, 0)
    float u1 = e10Len;
    float v2 = dot(e20, e2); // should be ~= tLen (signed)

    // Fit N(u,v) = n0 + a*u + b*v.
    float3 dn1 = vertexNormals[1] - vertexNormals[0];
    float3 dn2 = vertexNormals[2] - vertexNormals[0];

    // a = dN/du
    float3 a = dn1 / max(u1, eps);

    // b = dN/dv
    float denomV = abs(v2) < eps ? (v2 >= 0.0f ? eps : -eps) : v2;
    float3 b = (dn2 - a * u2) / denomV;

    // Scalar curvature proxy: RMS magnitude of the normal gradient. // Units: 1 / position_units.
    float curvature = sqrt(dot(a, a) + dot(b, b));
    return curvature;
}

DonutGeometrySample getGeometryFromHit(
    uint instanceIndex,
    uint geometryIndex,
    uint triangleIndex,
    float2 rayBarycentrics,
    DonutGeometryAttributes attributes,
    StructuredBuffer<InstanceData> instanceBuffer,
    StructuredBuffer<GeometryData> geometryBuffer,
    StructuredBuffer<GeometryDebugData> geometryDebugBuffer,
    float3 rayDirection, 
    DebugContext debug)
{
    DonutGeometrySample gs = (DonutGeometrySample)0;

    gs.instance = instanceBuffer[instanceIndex];
    gs.geometry = geometryBuffer[gs.instance.firstGeometryIndex + geometryIndex];
    gs.geometryDebug = geometryDebugBuffer[gs.instance.firstGeometryIndex + geometryIndex];
    
    ByteAddressBuffer indexBuffer = t_BindlessBuffers[NonUniformResourceIndex(gs.geometry.indexBufferIndex)];
    ByteAddressBuffer vertexBuffer = t_BindlessBuffers[NonUniformResourceIndex(gs.geometry.vertexBufferIndex)];

    float3 barycentrics;
    barycentrics.yz = rayBarycentrics;
    barycentrics.x = 1.0 - (barycentrics.y + barycentrics.z);

    uint3 indices = indexBuffer.Load3(gs.geometry.indexOffset + triangleIndex * c_SizeOfTriangleIndices);

    if (attributes & GeomAttr_Position)
    {
        gs.vertexPositions[0] = asfloat(vertexBuffer.Load3(gs.geometry.positionOffset + indices[0] * c_SizeOfPosition));
        gs.vertexPositions[1] = asfloat(vertexBuffer.Load3(gs.geometry.positionOffset + indices[1] * c_SizeOfPosition));
        gs.vertexPositions[2] = asfloat(vertexBuffer.Load3(gs.geometry.positionOffset + indices[2] * c_SizeOfPosition));
        gs.objectSpacePosition = interpolate(gs.vertexPositions, barycentrics);
    }

    if (attributes & GeomAttr_PrevPosition)
    {
        if( gs.geometry.prevPositionOffset != 0xFFFFFFFF )  // only present for skinned objects
        {
            float3 prevVertexPositions[3];
            /*gs.*/prevVertexPositions[0]   = asfloat(vertexBuffer.Load3(gs.geometry.prevPositionOffset + indices[0] * c_SizeOfPosition));
            /*gs.*/prevVertexPositions[1]   = asfloat(vertexBuffer.Load3(gs.geometry.prevPositionOffset + indices[1] * c_SizeOfPosition));
            /*gs.*/prevVertexPositions[2]   = asfloat(vertexBuffer.Load3(gs.geometry.prevPositionOffset + indices[2] * c_SizeOfPosition));
            gs.prevObjectSpacePosition  = interpolate(/*gs.*/prevVertexPositions, barycentrics);
        }
        else
            gs.prevObjectSpacePosition  = gs.objectSpacePosition;
    }

    if ((attributes & GeomAttr_TexCoord) && gs.geometry.texCoord1Offset != ~0u)
    {
        gs.vertexTexcoords[0] = asfloat(vertexBuffer.Load2(gs.geometry.texCoord1Offset + indices[0] * c_SizeOfTexcoord));
        gs.vertexTexcoords[1] = asfloat(vertexBuffer.Load2(gs.geometry.texCoord1Offset + indices[1] * c_SizeOfTexcoord));
        gs.vertexTexcoords[2] = asfloat(vertexBuffer.Load2(gs.geometry.texCoord1Offset + indices[2] * c_SizeOfTexcoord));
        gs.texcoord = interpolate(gs.vertexTexcoords, barycentrics);
    }


    float3 vBA = gs.vertexPositions[1] - gs.vertexPositions[0];
    float3 vCA = gs.vertexPositions[2] - gs.vertexPositions[0];

    float3 objectSpaceFlatNormal = SafeNormalize(cross(vBA, vCA));


    if ((attributes & GeomAttr_Normal) && gs.geometry.normalOffset != ~0u)
    {
        float3 normals[3];
        normals[0] = normalize(Unpack_RGB8_SNORM(vertexBuffer.Load(gs.geometry.normalOffset + indices[0] * c_SizeOfNormal)));
        normals[1] = normalize(Unpack_RGB8_SNORM(vertexBuffer.Load(gs.geometry.normalOffset + indices[1] * c_SizeOfNormal)));
        normals[2] = normalize(Unpack_RGB8_SNORM(vertexBuffer.Load(gs.geometry.normalOffset + indices[2] * c_SizeOfNormal)));

		// we want the geometry normal to be on the same hemisphere as the triangle normal (should be guaranteed on tools side, but isn't always)
        normals[0] = FlipIfOpposite(normals[0], objectSpaceFlatNormal);
        normals[1] = FlipIfOpposite(normals[1], objectSpaceFlatNormal);
        normals[2] = FlipIfOpposite(normals[2], objectSpaceFlatNormal);

        gs.curvatureWS = TriangleCurvatureApprox_GradN(gs.vertexPositions, normals, gs.instance.transform);

        gs.geometryNormal = interpolate(normals, barycentrics);

        gs.geometryNormal = mul(gs.instance.transform, float4(gs.geometryNormal, 0.0)).xyz;
        gs.geometryNormal = SafeNormalize(gs.geometryNormal);
    }
    else
        gs.curvatureWS = 0;

    if ((attributes & GeomAttr_Tangents) && gs.geometry.tangentOffset != ~0u)
    {
        float4 tangents[3];
        tangents[0] = Unpack_RGBA8_SNORM(vertexBuffer.Load(gs.geometry.tangentOffset + indices[0] * c_SizeOfNormal));
        tangents[1] = Unpack_RGBA8_SNORM(vertexBuffer.Load(gs.geometry.tangentOffset + indices[1] * c_SizeOfNormal));
        tangents[2] = Unpack_RGBA8_SNORM(vertexBuffer.Load(gs.geometry.tangentOffset + indices[2] * c_SizeOfNormal));

        gs.tangent.xyz = interpolate(tangents, barycentrics).xyz;
        gs.tangent.xyz = mul(gs.instance.transform, float4(gs.tangent.xyz, 0.0)).xyz;
        gs.tangent.xyz = SafeNormalize(gs.tangent.xyz);
        gs.tangent.w = tangents[0].w;
    }

    gs.flatNormal   = SafeNormalize(mul(gs.instance.transform, float4(objectSpaceFlatNormal, 0.0)).xyz);

    gs.frontFacing  = dot( -rayDirection, gs.flatNormal ) >= 0.0;

    return gs;
}

enum MaterialAttributes
{
    MatAttr_BaseColor    = 0x01,
    MatAttr_Emissive     = 0x02,
    MatAttr_Normal       = 0x04,
    MatAttr_MetalRough   = 0x08,
    MatAttr_Transmission = 0x10,

    MatAttr_All          = 0x1F
};

float4 sampleTexture(uint textureIndexAndInfo, SamplerState materialSampler, const ActiveTextureSampler textureSampler, float2 uv)
{
    uint textureIndex = textureIndexAndInfo & 0xFFFF;
    uint baseLOD = textureIndexAndInfo>>24;
    uint mipLevels = (textureIndexAndInfo>>16) & 0xFF;

    Texture2D tex2D = t_BindlessTextures[NonUniformResourceIndex(textureIndex)];
    
    return textureSampler.sampleTexture(tex2D, materialSampler, uv, baseLOD, mipLevels);
}

void ApplyNormalMapRTXPT(inout MaterialProperties result, float4 tangent, float4 normalsTextureValue, float normalTextureScale)
{
    float squareTangentLength = dot(tangent.xyz, tangent.xyz);
    if (squareTangentLength == 0)
        return;
    
    if (tangent.w == 0)
        return;

    normalsTextureValue.xy = normalsTextureValue.xy * 2.0 - 1.0;
    normalsTextureValue.xy *= normalTextureScale;

    if (normalsTextureValue.z <= 0)
        normalsTextureValue.z = sqrt(saturate(1.0 - square(normalsTextureValue.x) - square(normalsTextureValue.y)));
    else
        normalsTextureValue.z = abs(normalsTextureValue.z * 2.0 - 1.0);

    float squareNormalMapLength = dot(normalsTextureValue.xyz, normalsTextureValue.xyz);

    if (squareNormalMapLength == 0)
        return;
        
    float normalMapLen = sqrt(squareNormalMapLength);
    float3 localNormal = normalsTextureValue.xyz / normalMapLen;

    tangent.xyz *= rsqrt(squareTangentLength);
    float3 bitangent = cross(result.geometryNormal, tangent.xyz) * tangent.w;

    result.shadingNormal = normalize(tangent.xyz * localNormal.x + bitangent.xyz * localNormal.y + result.geometryNormal.xyz * localNormal.z);
}

MaterialProperties EvaluateSceneMaterialRTXPT(float3 normal, float4 tangent, PTMaterialData material, MaterialTextureSample textures)
{
    MaterialProperties result = MaterialProperties::make();
    result.geometryNormal   = normalize(normal);
    result.shadingNormal    = result.geometryNormal;
    result.flags = material.Flags;
    
    if ((material.Flags & PTMaterialFlags_UseSpecularGlossModel) != 0)
    {
        float3 diffuseColor = material.BaseOrDiffuseColor.rgb * textures.baseOrDiffuse.rgb;
        float3 specularColor = material.SpecularColor.rgb * textures.metalRoughOrSpecular.rgb;
        result.roughness = lpfloat(1.0 - textures.metalRoughOrSpecular.a * (1.0 - material.Roughness));

#if ENABLE_METAL_ROUGH_RECONSTRUCTION
        ConvertSpecularGlossToMetalRough(diffuseColor, specularColor, result.baseColor, result.metalness);
        //result.hasMetalRoughParams = true;
#endif

        // Compute the BRDF inputs for the specular-gloss model
        // https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_materials_pbrSpecularGlossiness/README.md#specular---glossiness
        result.diffuseAlbedo = lpfloat3(diffuseColor * (1.0 - max(specularColor.r, max(specularColor.g, specularColor.b))));
        result.specularF0 = lpfloat3(specularColor);
    }
    else
    {
        result.baseColor = lpfloat3(material.BaseOrDiffuseColor.rgb * textures.baseOrDiffuse.rgb);
        result.roughness = lpfloat(material.Roughness * textures.metalRoughOrSpecular.g);
        if ((material.Flags & PTMaterialFlags_MetalnessInRedChannel) != 0)
            result.metalness = lpfloat(material.Metalness * textures.metalRoughOrSpecular.r);
        else
            result.metalness = lpfloat(material.Metalness * textures.metalRoughOrSpecular.b);
        //result.hasMetalRoughParams = true;

        // Compute the BRDF inputs for the metal-rough model
        // https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#metal-brdf-and-dielectric-brdf
        result.diffuseAlbedo = result.baseColor * (1.0 - result.metalness); // Don't compensate for specular energy here. Energy compensation is built into Frostbite's diffuse, so this would be double dipping.
        result.specularF0 = lpfloat3( lerp(c_DielectricSpecular, result.baseColor.rgb, result.metalness) );
    }

#if 0    
    result.occlusion = 1.0;
    if ((material.Flags & PTMaterialFlags_UseOcclusionTexture) != 0)
    {
        result.occlusion = lpfloat( textures.occlusion.r );
    }
    result.occlusion = lpfloat( lerp(1.0, result.occlusion, material.OcclusionStrength) );
#endif

    result.opacity = lpfloat( material.Opacity );
    if ((material.Flags & MaterialFlags_UseBaseOrDiffuseTexture) != 0)
        result.opacity *= lpfloat( textures.baseOrDiffuse.a );
    result.opacity = saturate(result.opacity);

#if !defined(RTXPT_MATERIAL_HAS_TRANSMISSION) || RTXPT_MATERIAL_HAS_TRANSMISSION
    result.transmission = lpfloat( material.TransmissionFactor );
    result.diffuseTransmission = lpfloat( material.DiffuseTransmissionFactor );
    if ((material.Flags & PTMaterialFlags_UseTransmissionTexture) != 0)
    {
        result.transmission *= lpfloat( textures.transmission.r );
        result.diffuseTransmission *= lpfloat( textures.transmission.r );
    }
#endif
    
    result.emissiveColor = lpfloat3( material.EmissiveColor );
    if ((material.Flags & PTMaterialFlags_UseEmissiveTexture) != 0)
        result.emissiveColor *= lpfloat3( textures.emissive.rgb );

    result.ior = lpfloat( material.IoR );
    
    result.shadowNoLFadeout = lpfloat( material.ShadowNoLFadeout );
    
    #if defined(RTXPT_MATERIAL_USE_NORMAL_TEXTURE)
        #if RTXPT_MATERIAL_USE_NORMAL_TEXTURE
            ApplyNormalMapRTXPT(result, tangent, textures.normal, material.NormalTextureScale);  // there's an incorrect "error X3508: 'ApplyNormalMap': output parameter 'result' not completely initialized" if this line happens before result is fully initialized
        #endif
    #else
        if ((material.Flags & PTMaterialFlags_UseNormalTexture) != 0)
            ApplyNormalMapRTXPT(result, tangent, textures.normal, material.NormalTextureScale);  // there's an incorrect "error X3508: 'ApplyNormalMap': output parameter 'result' not completely initialized" if this line happens before result is fully initialized
    #endif

    return result;
}

MaterialProperties sampleGeometryMaterialRTXPT(const DonutGeometrySample gs, uint materialIndex, const MaterialAttributes attributes, const SamplerState materialSampler, const ActiveTextureSampler textureSampler)
{
    MaterialTextureSample textures = DefaultMaterialTextures();

    PTMaterialData material = t_PTMaterialData[materialIndex];

    //if( !optimizationHints.NoTextures )
    {
        if ((attributes & MatAttr_BaseColor) && (material.Flags & PTMaterialFlags_UseBaseOrDiffuseTexture) != 0)
            textures.baseOrDiffuse = sampleTexture(material.BaseOrDiffuseTextureIndex, materialSampler, textureSampler, gs.texcoord);

        if ((attributes & MatAttr_Emissive) && (material.Flags & PTMaterialFlags_UseEmissiveTexture) != 0)
            textures.emissive = sampleTexture(material.EmissiveTextureIndex, materialSampler, textureSampler, gs.texcoord);
    
        #if defined(RTXPT_MATERIAL_USE_NORMAL_TEXTURE)
            #if RTXPT_MATERIAL_USE_NORMAL_TEXTURE
                if (attributes & MatAttr_Normal)
                    textures.normal = sampleTexture(material.NormalTextureIndex, materialSampler, textureSampler, gs.texcoord);
            #endif
        #else
            if ((attributes & MatAttr_Normal) && (material.Flags & PTMaterialFlags_UseNormalTexture) != 0)
                textures.normal = sampleTexture(material.NormalTextureIndex, materialSampler, textureSampler, gs.texcoord);
        #endif

        if ((attributes & MatAttr_MetalRough) && (material.Flags & PTMaterialFlags_UseMetalRoughOrSpecularTexture) != 0)
            textures.metalRoughOrSpecular = sampleTexture(material.MetalRoughOrSpecularTextureIndex, materialSampler, textureSampler, gs.texcoord);

#if !defined(RTXPT_MATERIAL_HAS_TRANSMISSION) || RTXPT_MATERIAL_HAS_TRANSMISSION
        if ((attributes & MatAttr_Transmission) && (material.Flags & PTMaterialFlags_UseTransmissionTexture) != 0)
            textures.transmission = sampleTexture(material.TransmissionTextureIndex, materialSampler, textureSampler, gs.texcoord);
#endif
    }

    return EvaluateSceneMaterialRTXPT(gs.geometryNormal, gs.tangent, material, textures);
}

static OpacityMicroMapDebugInfo loadOmmDebugInfo(const DonutGeometrySample donutGS, const uint triangleIndex, float2 barycentrics)
{
    OpacityMicroMapDebugInfo ommDebug = OpacityMicroMapDebugInfo::initDefault();

#if (OMM_DEBUG_VIEW_IN_WORLD || OMM_DEBUG_VIEW_OVERLAY) && !NON_PATH_TRACING_PASS && ENABLE_DEBUG_VIZUALISATIONS
    if (donutGS.geometryDebug.ommIndexBufferIndex != -1 &&
        donutGS.geometryDebug.ommIndexBufferOffset != 0xFFFFFFFF)
    {
        ByteAddressBuffer ommIndexBuffer = t_BindlessBuffers[NonUniformResourceIndex(donutGS.geometryDebug.ommIndexBufferIndex)];
        ByteAddressBuffer ommDescArrayBuffer = t_BindlessBuffers[NonUniformResourceIndex(donutGS.geometryDebug.ommDescArrayBufferIndex)];
        ByteAddressBuffer ommArrayDataBuffer = t_BindlessBuffers[NonUniformResourceIndex(donutGS.geometryDebug.ommArrayDataBufferIndex)];

        OpacityMicroMapContext ommContext = OpacityMicroMapContext::make(
            ommIndexBuffer, donutGS.geometryDebug.ommIndexBufferOffset, donutGS.geometryDebug.ommIndexBuffer16Bit,
            ommDescArrayBuffer, donutGS.geometryDebug.ommDescArrayBufferOffset,
            ommArrayDataBuffer, donutGS.geometryDebug.ommArrayDataBufferOffset,
            triangleIndex,
            barycentrics
        );

        ommDebug.hasOmmAttachment = true;
        ommDebug.opacityStateDebugColor = OpacityMicroMapDebugViz(ommContext);
    }
#endif

    return ommDebug;
}

static void surfaceDebugViz(uint2 pixelPos, const PathTracer::SurfaceData surfaceData, const float2 barycentrics, const float3 rayDir, const RayCone rayCone, const int pathVertexIndex, const OpacityMicroMapDebugInfo ommDebug, uint materialID, DebugContext debug)
{
#if ENABLE_DEBUG_VIZUALISATIONS && !NON_PATH_TRACING_PASS && ENABLE_DEBUG_SURFACE_VIZ
    if (g_Const.debug.debugViewType == (int)DebugViewType::Disabled || pathVertexIndex != 1)
        return;

#if 0
    float3 camPos = debug.constants.cameraPosW;
    float3 diff = surfaceData.shadingData.posW - camPos;

    float3 dsig = sign(diff);
    diff = abs(diff);
    diff = diff / (diff+20); // Reinhard-like mapping
    diff *= dsig;
    diff = (diff * 0.5 + 0.5);

    debug.DrawDebugViz( float4( frac( diff * 1024.0 ) < 0.1, 1.0 ) );
#endif

    //const VertexData vd     = surfaceData.vd;
    const ShadingData shadingData = surfaceData.shadingData;
    const ActiveBSDF bsdf = surfaceData.bsdf;

    // these work only when ActiveBSDF is StandardBSDF - make an #ifdef if/when this becomes a problem
    StandardBSDFData bsdfData = bsdf.data;

    uint shaderID = 0xFFFFFFFF;
    #ifdef RTXPT_SHADER_ID
    shaderID = RTXPT_SHADER_ID+1;   // 0 will result in black, so start from 1 and leave 0 to represent ubershader
    #endif

    switch (g_Const.debug.debugViewType)
    {
    case ((int)DebugViewType::FirstHit_Barycentrics):                debug.DrawDebugViz(pixelPos, float4(barycentrics, 0.0, 1.0)); break;
    case ((int)DebugViewType::FirstHit_FaceNormal):                  debug.DrawDebugViz(pixelPos, float4(DbgShowNormalSRGB(shadingData.faceNCorrected), 1.0)); break;
    case ((int)DebugViewType::FirstHit_GeometryNormal):              debug.DrawDebugViz(pixelPos, float4(DbgShowNormalSRGB(shadingData.vertexN), 1.0)); break;
    case ((int)DebugViewType::FirstHit_ShadingNormal):               debug.DrawDebugViz(pixelPos, float4(DbgShowNormalSRGB(shadingData.N), 1.0)); break;
    case ((int)DebugViewType::FirstHit_ShadingTangent):              debug.DrawDebugViz(pixelPos, float4(DbgShowNormalSRGB(shadingData.T), 1.0)); break;
    case ((int)DebugViewType::FirstHit_ShadingBitangent):            debug.DrawDebugViz(pixelPos, float4(DbgShowNormalSRGB(shadingData.B), 1.0)); break;
    case ((int)DebugViewType::FirstHit_FrontFacing):                 debug.DrawDebugViz(pixelPos, float4(saturate(float3(0.15, 0.1 + shadingData.frontFacing, 0.15)), 1.0)); break;
    case ((int)DebugViewType::FirstHit_ThinSurface):                 debug.DrawDebugViz(pixelPos, float4(saturate(float3(0.15, 0.1 + shadingData.mtl.isThinSurface(), 0.15)), 1.0)); break;
    case ((int)DebugViewType::FirstHit_Diffuse):                     debug.DrawDebugViz(pixelPos, float4(bsdfData.Diffuse().xyz, 1.0)); break;
    case ((int)DebugViewType::FirstHit_Specular):                    debug.DrawDebugViz(pixelPos, float4(bsdfData.Specular().xyz, 1.0)); break;
    case ((int)DebugViewType::FirstHit_Roughness):                   debug.DrawDebugViz(pixelPos, float4(bsdfData.Roughness().xxx, 1.0)); break;
    case ((int)DebugViewType::FirstHit_Metallic):                    debug.DrawDebugViz(pixelPos, float4(bsdfData.Metallic().xxx, 1.0)); break;
    case ((int)DebugViewType::FirstHit_ShaderID):                    debug.DrawDebugViz(pixelPos, float4( ColorFromHash(Hash32(shaderID)), 1.0)); break;
    case ((int)DebugViewType::FirstHit_MaterialID):                  debug.DrawDebugViz(pixelPos, float4( ColorFromHash(Hash32(materialID)), 1.0)); break;
    default: break;
    }
#endif
}

uint Bridge::getSampleIndex()
{
    return g_Const.ptConsts.sampleBaseIndex + g_MiniConst.params.x;
}

float Bridge::getNoisyRadianceAttenuation()
{
    // When using multiple samples within pixel in realtime mode (which share identical camera ray), only noisy part of radiance (i.e. not direct sky) needs to be attenuated!
#if PATH_TRACER_MODE != PATH_TRACER_MODE_BUILD_STABLE_PLANES
    return g_Const.ptConsts.invSubSampleCount;
#else
    return 1.0;
#endif
}

uint Bridge::getMaxBounceLimit()
{
#if PT_BOUNCE_COUNT
    return PT_BOUNCE_COUNT;
#else
    return g_Const.ptConsts.bounceCount;
#endif
}

uint Bridge::getMaxDiffuseBounceLimit()
{
#if PT_DIFFUSE_BOUNCE_COUNT
    return PT_DIFFUSE_BOUNCE_COUNT;
#else
    return g_Const.ptConsts.diffuseBounceCount;
#endif
}

Ray Bridge::computeCameraRay(const uint2 pixelPos)
{
    SampleGenerator sampleGenerator = SampleGenerator::make( SampleGeneratorVertexBase::make( pixelPos, 0, Bridge::getSampleIndex() ) );

    // compute camera ray! would make sense to compile out if unused
    float2 subPixelOffset = g_Const.ptConsts.camera.Jitter + (sampleNext2D(sampleGenerator) - 0.5.xx) * g_Const.ptConsts.perPixelJitterAAScale;
    const float2 cameraDoFSample = sampleNext2D(sampleGenerator);
    //return ComputeRayPinhole( g_Const.ptConsts.camera, pixelPos, subPixelOffset );
    Ray ray = ComputeRayThinlens( g_Const.ptConsts.camera, pixelPos, subPixelOffset, cameraDoFSample ); 

#if 0  // fallback: use inverted matrix) useful for correctness validation; with DoF disabled (apertureRadius/focalDistance == near zero), should provide same rays as above code - otherwise something's really broken
    PlanarViewConstants view = g_Const.view;
    float2 uv = (float2(pixelPos) + 0.5) * view.viewportSizeInv;
    float4 clipPos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1e-7, 1);
    float4 worldPos = mul(clipPos, view.matClipToWorld);
    worldPos.xyz /= worldPos.w;
        
    ray.origin  = view.cameraDirectionOrPosition.xyz;
    ray.dir     = normalize(worldPos.xyz - ray.origin);
#endif
    return ray;
}

/** Helper to create a texture sampler instance.
The method for computing texture level-of-detail depends on the configuration.
\param[in] path Path state.
\param[in] isPrimaryTriangleHit True if primary hit on a triangle.
\return Texture sampler instance.
*/
ActiveTextureSampler Bridge::createTextureSampler(
    const RayCone rayCone,
    const float3 rayDir,
    float coneTexLODValue,
    float3 normalW,
    bool isPrimaryHit,
    bool isTriangleHit,
    float texLODBias
#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
    ,STF_SamplerState stfSamplerState
#endif
)
{
#if ACTIVE_LOD_TEXTURE_SAMPLER == LOD_TEXTURE_SAMPLER_EXPLICIT
    return ExplicitLodTextureSampler::make(texLODBias
#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
        ,stfSamplerState
#endif
    );
#elif ACTIVE_LOD_TEXTURE_SAMPLER == LOD_TEXTURE_SAMPLER_RAY_CONES
    float lambda = rayCone.computeLOD(coneTexLODValue, rayDir, normalW, true);
    lambda += texLODBias;
    return ExplicitRayConesLodTextureSampler::make(lambda
#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
        ,stfSamplerState
#endif
    );
#endif  // ACTIVE_LOD_TEXTURE_SAMPLER
}

PathTracer::SurfaceData Bridge::loadSurface( const TriangleHit triangleHit, 
    const float3 rayDir, const RayCone rayCone, const int pathVertexIndex, const uint2 pixelPos, DebugContext debug)
{
    const uint instanceIndex    = triangleHit.instanceID.getInstanceIndex();
    const uint geometryIndex    = triangleHit.instanceID.getGeometryIndex();
    const uint triangleIndex    = triangleHit.primitiveIndex;
    const float2 barycentrics   = triangleHit.barycentrics;
    return Bridge::loadSurface( instanceIndex, geometryIndex, triangleIndex, barycentrics, rayDir, rayCone, pathVertexIndex, pixelPos, debug );
}

static PathTracer::SurfaceData Bridge::loadSurface( const uint instanceIndex, const uint geometryIndex, const uint triangleIndex, const float2 barycentrics,
    const float3 rayDir, const RayCone rayCone, const int pathVertexIndex, const uint2 pixelPos, DebugContext debug )
{
    const bool isPrimaryHit     = pathVertexIndex == 1;

    DonutGeometryAttributes attributes = GeomAttr_TexCoord | GeomAttr_Position | GeomAttr_Normal | GeomAttr_Tangents;
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES // otherwise motion vectors not needed
    attributes |= GeomAttr_PrevPosition;
#endif

    DonutGeometrySample donutGS = getGeometryFromHit( instanceIndex, geometryIndex, triangleIndex, barycentrics, 
        attributes, t_InstanceData, t_GeometryData, t_GeometryDebugData, rayDir, debug );

    // Convert Donut to RTXPT data! 

    // World pos and prev world pos
    float3 posW     = mul(donutGS.instance.transform, float4(donutGS.objectSpacePosition, 1.0)).xyz;

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES // otherwise motion vectors not needed
    float3 prevPosW = mul(donutGS.instance.prevTransform, float4(donutGS.prevObjectSpacePosition, 1.0)).xyz;
#endif

    // transpose is to go from Donut row_major to Falcor column_major; it is likely unnecessary here since both should work the same for this specific function, but leaving in for correctness
    float coneTexLODValue = computeRayConeTriangleLODValue( donutGS.vertexPositions, donutGS.vertexTexcoords, transpose((float3x3)donutGS.instance.transform) );
      
#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
    STF_SamplerState stfSamplerState;
    float4 u;
    #if 0
    if (g_Const.ptConsts.STFUseBlueNoise)
    {
        u = SpatioTemporalBlueNoise2DWhiteNoise2D(pixelPos, Bridge::getSampleIndex(), t_STBN2DTexture);
    } 
    else
    #endif
    {
        SampleGeneratorVertexBase sampleGeneratorVertexBase = SampleGeneratorVertexBase::make(pixelPos, pathVertexIndex, Bridge::getSampleIndex());       
        SampleGenerator sampleGenerator = SampleGenerator::make(sampleGeneratorVertexBase); 
        u = sampleNext4D(sampleGenerator);
    }
    stfSamplerState = STF_SamplerState::Create(u);
    stfSamplerState.SetFrameIndex(Bridge::getSampleIndex());
    stfSamplerState.SetFilterType(g_Const.ptConsts.STFFilterMode);
    stfSamplerState.SetMagMethod(g_Const.ptConsts.STFMagnificationMethod);
    stfSamplerState.SetSigma(g_Const.ptConsts.STFGaussianSigma);  
#endif
    
    // using flat (triangle) normal makes more sense since actual triangle surface is where the textures are sampled on (plus geometry normals are borked in some datasets)
    ActiveTextureSampler textureSampler = createTextureSampler( rayCone, rayDir, coneTexLODValue, donutGS.flatNormal/*donutGS.geometryNormal*/, isPrimaryHit, true, g_Const.ptConsts.texLODBias
#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
        ,stfSamplerState
#endif
    );

    // See MaterialFactory.hlsli in Falcor
    ShadingData ptShadingData = ShadingData::make();

    ptShadingData.posW = posW;
    //ptShadingData.uv   = lpfloat2(donutGS.texcoord);
    ptShadingData.V    = -rayDir;
    ptShadingData.N    = donutGS.geometryNormal;

    uint subInstanceDataIndex = donutGS.instance.firstGeometryInstanceIndex + geometryIndex;

    uint materialIndex = t_SubInstanceData[subInstanceDataIndex].GlobalGeometryIndex_PTMaterialDataIndex & 0xFFFF;

    // Get donut material (normal map is evaluated here)
    MaterialProperties donutMaterial = sampleGeometryMaterialRTXPT(donutGS, materialIndex, MatAttr_All, s_MaterialSampler, textureSampler);

    bool ignoreTangent = (donutMaterial.flags & PTMaterialFlags_IgnoreMeshTangentSpace) != 0;

    // after this point we have valid tangent space in ptShadingData.N/.T/.B using geometry (interpolated) normal, but without normalmap yet
    computeTangentSpace(ptShadingData, donutGS.tangent, ignoreTangent);

    // Primitive data
    ptShadingData.faceNCorrected = (donutGS.frontFacing)?(donutGS.flatNormal):(-donutGS.flatNormal);
    ptShadingData.vertexN = (donutGS.frontFacing)?(donutGS.geometryNormal):(-donutGS.geometryNormal);
    ptShadingData.frontFacing = donutGS.frontFacing;

    ptShadingData.N = (donutGS.frontFacing)?(donutMaterial.shadingNormal):(-donutMaterial.shadingNormal);

    // Donut -> Falcor
    const bool donutMaterialThinSurface = (donutMaterial.flags & PTMaterialFlags_ThinSurface) != 0;
    ptShadingData.materialID = materialIndex;
    ptShadingData.mtl = MaterialHeader::make();
    ptShadingData.mtl.setNestedPriority( min( InteriorList::kMaxNestedPriority, 1 + (uint(donutMaterial.flags) >> PTMaterialFlags_NestedPriorityShift)) );   // priorities are from (1, ... kMaxNestedPriority) because 0 is used to mark empty slots and remapped to kMaxNestedPriority
    ptShadingData.mtl.setThinSurface( donutMaterialThinSurface );
    ptShadingData.mtl.setPSDExclude( (donutMaterial.flags & PTMaterialFlags_PSDExclude) != 0 );
    ptShadingData.mtl.setPSDDominantDeltaLobeP1( (donutMaterial.flags & PTMaterialFlags_PSDDominantDeltaLobeP1Mask) >> PTMaterialFlags_PSDDominantDeltaLobeP1Shift );


    // stopping motion vectors from being calculated behind/beyond this surface
    {
        // types are 0 - Off; 1 - AutoLow; 2 - AutoHigh; 3 - Full
        const int blockType = ((donutMaterial.flags & PTMaterialFlags_PSDBlockMVsAtSurfaceTypeB0) != 0) + ((donutMaterial.flags & PTMaterialFlags_PSDBlockMVsAtSurfaceTypeB1) != 0) * 2;
        bool blockMVs = (blockType) == 3;

        if (blockType == 1 || blockType == 2)
        {
            float projectionTerm = abs(dot(rayDir, -ptShadingData.N/*-ptShadingData.vertexN*//*-ptShadingData.faceNCorrected*/));
            float pixelCurvature = (donutGS.curvatureWS * rayCone.getWidth()) / max(projectionTerm, 1e-6f);
            const float threshold = (blockType==1)?(0.03):(0.0005);
            MicroRng rng = MicroRng::make(pixelPos, pathVertexIndex, Bridge::getSampleIndex());
            blockMVs |= pixelCurvature > ((rng.NextFloat()*0.9+0.3)*threshold);
        }
        // if (blockMVs && pathVertexIndex == 1) DebugPixel(pixelPos, float4(1,0,0,1));
        ptShadingData.mtl.setPSDBlockMotionVectorsAtSurface( blockMVs );
    }

    // Helper function to adjust the shading normal to reduce black pixels due to back-facing view direction. Note: This breaks the reciprocity of the BSDF!
    // This also reorthonormalizes the frame based on the normal map, which is necessary (see `ptShadingData.N = donutMaterial.shadingNormal;` line above)
    adjustShadingNormal( ptShadingData, donutGS.tangent, true, ignoreTangent );
    // ^^ this should be part of material processing code

    // ptShadingData.opacity = donutMaterial.opacity;

    ptShadingData.shadowNoLFadeout = donutMaterial.shadowNoLFadeout;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Now load the actual BSDF! Equivalent to StandardBSDF::setupBSDF
    lpfloat3    bsdfDataDiffuse              = 0;
    lpfloat     bsdfDataRoughness            = 0;
    lpfloat3    bsdfDataSpecular             = 0;
    lpfloat     bsdfDataMetallic             = 0;
    lpfloat     bsdfDataEta                  = 0;


    // A.k.a. interiorIoR
    lpfloat matIoR = donutMaterial.ior;

#if !defined(RTXPT_MATERIAL_HAS_TRANSMISSION) || RTXPT_MATERIAL_HAS_TRANSMISSION
    // from https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_transmission/README.md#refraction
    // "This microfacet lobe is exactly the same as the specular lobe except sampled along the line of sight through the surface."
    lpfloat     bsdfDataSpecularTransmission = donutMaterial.transmission * (1 - donutMaterial.metalness);    // (1 - donutMaterial.metalness) is from https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_transmission/README.md#transparent-metals
    lpfloat     bsdfDataDiffuseTransmission = donutMaterial.diffuseTransmission * (1 - donutMaterial.metalness);    // (1 - donutMaterial.metalness) is from https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_transmission/README.md#transparent-metals
    lpfloat3    bsdfDataTransmission = donutMaterial.baseColor;
#else
    lpfloat     bsdfDataDiffuseTransmission  = 0;
    lpfloat     bsdfDataSpecularTransmission = 0;    
    lpfloat3    bsdfDataTransmission         = 0;
#endif

    /*LobeType*/ uint lobeType = (uint)LobeType::All;

#if defined(RTXPT_MATERIAL_HAS_TRANSMISSION) && !RTXPT_MATERIAL_HAS_TRANSMISSION
    lobeType &= ~(uint)LobeType::Transmission;//~((uint)LobeType::DiffuseReflection | (uint)LobeType::SpecularReflection | (uint)LobeType::DeltaReflection);
#endif

    ptShadingData.mtl.setActiveLobes( lobeType );

    // Sample base color.
    lpfloat3 baseColor = donutMaterial.baseColor;

    // OMM Debug evaluates the OMM state at a given triangle + hit BC color codes the result for the corresonding state.
    OpacityMicroMapDebugInfo ommDebug = loadOmmDebugInfo(donutGS, triangleIndex, barycentrics);
#if OMM_DEBUG_VIEW_IN_WORLD && !NON_PATH_TRACING_PASS && ENABLE_DEBUG_VIZUALISATIONS
    if (ommDebug.hasOmmAttachment)
        baseColor = (lpfloat3)ommDebug.opacityStateDebugColor;
#endif

#if ENABLE_METAL_ROUGH_RECONSTRUCTION == 0
#error we rely on Donut to do the conversion! for more info on how to do it manually search for MATERIAL_SYSTEM_HAS_SPEC_GLOSS_MATERIALS 
#endif

    // Calculate the specular reflectance for dielectrics from the IoR, as in the Disney BSDF [Burley 2015].
    // UE4 uses 0.08 multiplied by a default specular value of 0.5, hence F0=0.04 as default. The default IoR=1.5 gives the same result.
    float f = (matIoR - 1.f) / (matIoR + 1.f);
    float F0 = f * f;

    // G - Roughness; B - Metallic
    bsdfDataDiffuse = lerp(baseColor, lpfloat3(0,0,0), donutMaterial.metalness);
    bsdfDataSpecular = lerp(lpfloat3(F0,F0,F0), baseColor, donutMaterial.metalness);
    bsdfDataRoughness = donutMaterial.roughness;
    bsdfDataMetallic = donutMaterial.metalness;

    // Assume the default IoR for vacuum on the front-facing side.
    // The renderer may override this for nested dielectrics (see 'handleNestedDielectrics' calling Bridge::updateOutsideIoR)
    ptShadingData.IoR = 1.f;
    bsdfDataEta = ptShadingData.IoR / matIoR; 
    if (!ptShadingData.mtl.isThinSurface() && !ptShadingData.frontFacing) 
        bsdfDataEta = (matIoR / ptShadingData.IoR);

    // Sample the emissive texture.
    // The standard material supports uniform emission over the hemisphere.
    // Note: we only support single sided emissives at the moment; If upgrading, make sure to upgrade NEE codepath as well (i.e. PolymorphicLight.hlsli)

    uint neeTriangleLightIndex = RTXPT_INVALID_LIGHT_INDEX;
    uint neeAnalyticLightIndex = RTXPT_INVALID_LIGHT_INDEX;

#if !defined(RTXPT_MATERIAL_IS_EMISSIVE) || RTXPT_MATERIAL_IS_EMISSIVE
    if (ptShadingData.frontFacing && any(donutMaterial.emissiveColor>0))
    {
        ptShadingData.emission = donutMaterial.emissiveColor;

#if !RTXPT_USE_APPROXIMATE_MIS
        uint baseIndex = t_SubInstanceData[subInstanceDataIndex].EmissiveLightMappingOffset;
        if (baseIndex != 0xFFFFFFFF)
            neeTriangleLightIndex = baseIndex + triangleIndex;
#endif
        //if( debug.IsDebugPixel() )
        //{
        //    DebugPrint( "a {0}; b {1}, c {2}", geometryIndex, baseIndex, neeTriangleLightIndex );
        // }

#if 0
        LightSampler lightSampler = Bridge::CreateLightSampler( debug.pixelPos, false/*doesn't matter in this case*/, false );
        float3 v0 = mul(donutGS.instance.transform, float4(donutGS.vertexPositions[0], 1)).xyz;
        float3 v1 = mul(donutGS.instance.transform, float4(donutGS.vertexPositions[1], 1)).xyz;
        float3 v2 = mul(donutGS.instance.transform, float4(donutGS.vertexPositions[2], 1)).xyz;
        bool OK = lightSampler.ValidateTriangleLightIndex( neeTriangleLightIndex, v0, v1, v2, donutGS.flatNormal );
        debug.DrawDebugViz( float4(1-OK, OK, 0, 1) );
#endif
    }
#endif

#if !defined(RTXPT_MATERIAL_IS_ANALYTIC_LIGHT_PROXY) || RTXPT_MATERIAL_IS_ANALYTIC_LIGHT_PROXY
    if ( (donutMaterial.flags & PTMaterialFlags_EnableAsAnalyticLightProxy) != 0 )
        neeAnalyticLightIndex = t_SubInstanceData[subInstanceDataIndex].AnalyticProxyLightIndex;
#endif

    StandardBSDF bsdf = StandardBSDF::make(
        StandardBSDFData::make( bsdfDataDiffuse, bsdfDataSpecular, bsdfDataRoughness, bsdfDataMetallic, bsdfDataEta, bsdfDataTransmission, bsdfDataDiffuseTransmission, bsdfDataSpecularTransmission ) );

    // if you think tangent space is broken, test with this (won't make it correctly oriented)
    //ConstructONB( ptShadingData.N, ptShadingData.T, ptShadingData.B );

    PathTracer::SurfaceData ret = PathTracer::SurfaceData::make(/*ptVertex, */ptShadingData, bsdf, 
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES // otherwise motion vectors not needed
                                    prevPosW, 
#endif
                                    matIoR, neeTriangleLightIndex, neeAnalyticLightIndex);

#if ENABLE_DEBUG_VIZUALISATIONS && !NON_PATH_TRACING_PASS
    if( debug.IsDebugPixel(pixelPos) && pathVertexIndex==1 && !debug.constants.exploreDeltaTree )
        debug.SetPickedMaterial( materialIndex );
    surfaceDebugViz( pixelPos, ret, barycentrics, rayDir, rayCone, pathVertexIndex, ommDebug, materialIndex, debug );
#if OMM_DEBUG_VIEW_OVERLAY
    debug.DrawDebugViz(float4(ommDebug.opacityStateDebugColor, 1.0));
#endif
#endif
    return ret;
}

void Bridge::updateOutsideIoR(inout PathTracer::SurfaceData surfaceData, lpfloat outsideIoR)
{
    surfaceData.shadingData.IoR = outsideIoR;

    ///< Relative index of refraction (incident IoR / transmissive IoR), dependent on whether we're exiting or entering
    surfaceData.bsdf.data.SetEta( surfaceData.shadingData.frontFacing ? (surfaceData.shadingData.IoR / surfaceData.interiorIoR) : (surfaceData.interiorIoR / surfaceData.shadingData.IoR) ); 
}

lpfloat Bridge::loadIoR(const uint materialDataIndex)
{
    if( materialDataIndex >= g_Const.MaterialCount )
        return 1.0;
    else
        return (lpfloat)t_PTMaterialData[materialDataIndex].IoR;
}

HomogeneousVolumeData Bridge::loadHomogeneousVolumeData(const uint materialDataIndex)
{
    HomogeneousVolumeData ptVolume;
    ptVolume.sigmaS = float3(0,0,0); 
    ptVolume.sigmaA = float3(0,0,0); 
    ptVolume.g = 0.0;

    if( materialDataIndex >= g_Const.MaterialCount )
        return ptVolume;

    VolumePTConstants volumeInfo = t_PTMaterialData[materialDataIndex].Volume;
        
    // these should be precomputed on the C++ side!!
    ptVolume.sigmaS = float3(0,0,0); // no scattering yet
    ptVolume.sigmaA = -log( clamp( volumeInfo.AttenuationColor, 1e-7, 1 ) ) / max( 1e-30, volumeInfo.AttenuationDistance.xxx );
    return ptVolume;
}

// 2.5D motion vectors
float3 Bridge::computeMotionVector( float3 posW, float3 prevPosW )
{
    SimpleViewConstants view = g_Const.view;
    SimpleViewConstants previousView = g_Const.previousView;

    float4 clipPos = mul(float4(posW, 1), view.matWorldToClipNoOffset);
    clipPos.xyz /= clipPos.w;
    float4 prevClipPos = mul(float4(prevPosW, 1), previousView.matWorldToClipNoOffset);
    prevClipPos.xyz /= prevClipPos.w;

    if (clipPos.w <= 0 || prevClipPos.w <= 0)
        return float3(0,0,0);

    float3 motion;
    motion.xy = (prevClipPos.xy - clipPos.xy) * view.clipToWindowScale;
    //motion.xy += (view.pixelOffset - previousView.pixelOffset); //<- no longer needed, using NoOffset matrices
    motion.z = prevClipPos.w - clipPos.w; // Use view depth

    return motion;
}
// 2.5D motion vectors
float3 Bridge::computeSkyMotionVector( const uint2 pixelPos )
{
    SimpleViewConstants view = g_Const.view;
    SimpleViewConstants previousView = g_Const.previousView;

    float4 clipPos = float4( (pixelPos + 0.5.xx)/g_Const.view.clipToWindowScale+float2(-1,1), 1e-7, 1.0);
    float4 viewPos = mul( clipPos, view.matClipToWorldNoOffset ); viewPos.xyzw /= viewPos.w;
    float4 prevClipPos = mul(viewPos, previousView.matWorldToClipNoOffset);
    prevClipPos.xyz /= prevClipPos.w;

    float3 motion;
    motion.xy = (prevClipPos.xy - clipPos.xy) * view.clipToWindowScale;
    //motion.xy += (view.pixelOffset - previousView.pixelOffset); <- no longer needed, using NoOffset matrices
    motion.z = 0; //prevClipPos.w - clipPos.w; // Use view depth

    return motion;
}

bool AlphaTestImpl(SubInstanceData subInstanceData, uint triangleIndex, float2 rayBarycentrics)
{
    bool alphaTested = (subInstanceData.FlagsAndAlphaInfo & SubInstanceData::Flags_AlphaTested) != 0;
    if( !alphaTested ) // note: with correct use of D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE this is unnecessary, but there are cases (such as disabling texture but leaving alpha tested state) in which this isn't handled correctly
        return true;
        
    // have to do all this to figure out UVs!
    float2 texcoord;
    {
#if !SUBINSTANCEDATA_EXTENDED
        GeometryData geometry = t_GeometryData[subInstanceData.GlobalGeometryIndex_PTMaterialDataIndex>>16];

        ByteAddressBuffer indexBuffer = t_BindlessBuffers[NonUniformResourceIndex(geometry.indexBufferIndex)];
        ByteAddressBuffer vertexBuffer = t_BindlessBuffers[NonUniformResourceIndex(geometry.vertexBufferIndex)];

        uint3 indices = indexBuffer.Load3(geometry.indexOffset + triangleIndex * c_SizeOfTriangleIndices);

        float2 vertexTexcoords[3];
        vertexTexcoords[0] = asfloat(vertexBuffer.Load2(geometry.texCoord1Offset + indices[0] * c_SizeOfTexcoord));
        vertexTexcoords[1] = asfloat(vertexBuffer.Load2(geometry.texCoord1Offset + indices[1] * c_SizeOfTexcoord));
        vertexTexcoords[2] = asfloat(vertexBuffer.Load2(geometry.texCoord1Offset + indices[2] * c_SizeOfTexcoord));
#else
        ByteAddressBuffer indexBuffer = t_BindlessBuffers[NonUniformResourceIndex( subInstanceData.IndexBufferIndex_VertexBufferIndex >> 16 /*geometry.indexBufferIndex*/ )];
        ByteAddressBuffer vertexBuffer = t_BindlessBuffers[NonUniformResourceIndex(subInstanceData.IndexBufferIndex_VertexBufferIndex & 0xFFFF /*geometry.vertexBufferIndex*/ )];

        uint3 indices = indexBuffer.Load3(subInstanceData.IndexOffset + triangleIndex * c_SizeOfTriangleIndices);

        float2 vertexTexcoords[3];
        vertexTexcoords[0] = asfloat(vertexBuffer.Load2(subInstanceData.TexCoord1Offset + indices[0] * c_SizeOfTexcoord));
        vertexTexcoords[1] = asfloat(vertexBuffer.Load2(subInstanceData.TexCoord1Offset + indices[1] * c_SizeOfTexcoord));
        vertexTexcoords[2] = asfloat(vertexBuffer.Load2(subInstanceData.TexCoord1Offset + indices[2] * c_SizeOfTexcoord));
#endif

        float3 barycentrics;
        barycentrics.yz = rayBarycentrics;
        barycentrics.x = 1.0 - (barycentrics.y + barycentrics.z);
        texcoord = interpolate(vertexTexcoords, barycentrics);
    }
    // sample the alpha (opacity) texture and test vs the threshold
    Texture2D diffuseTexture = t_BindlessTextures[NonUniformResourceIndex(subInstanceData.AlphaTextureIndex())];
    float opacityValue = diffuseTexture.SampleLevel(s_MaterialSampler, texcoord, 0).a; // <- hard coded to .a channel but we might want a separate alpha only texture, maybe in .g of BC1
    return opacityValue >= subInstanceData.AlphaCutoff();
}

bool Bridge::AlphaTest(uint instanceID, uint instanceIndex, uint geometryIndex, uint triangleIndex, float2 rayBarycentrics)
{
    SubInstanceData subInstanceData = t_SubInstanceData[(instanceID + geometryIndex)];

    return AlphaTestImpl(subInstanceData, triangleIndex, rayBarycentrics);
}

bool Bridge::AlphaTestVisibilityRay(uint instanceID, uint instanceIndex, uint geometryIndex, uint triangleIndex, float2 rayBarycentrics)
{
    SubInstanceData subInstanceData = t_SubInstanceData[(instanceID + geometryIndex)];

    bool excludeFromNEE = (subInstanceData.FlagsAndAlphaInfo & SubInstanceData::Flags_ExcludeFromNEE) != 0;
    if (excludeFromNEE)
        return false;

    return AlphaTestImpl(subInstanceData, triangleIndex, rayBarycentrics);
}

// There's a relatively high cost to this when used in large shaders just due to register allocation required for alphaTest, even if all geometries are opaque.
// Consider simplifying alpha testing - perhaps splitting it up from the main geometry path, load it with fewer indirections or something like that.
bool Bridge::traceVisibilityRay(RayDesc ray, const RayCone rayCone, const int pathVertexIndex, DebugContext debug)
{
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, RTXPT_FLAG_ALLOW_OPACITY_MICROMAPS> rayQuery;
    rayQuery.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xff, ray);

    while (rayQuery.Proceed())
    {
        if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            [branch]if (Bridge::AlphaTestVisibilityRay(
                rayQuery.CandidateInstanceID(),
                rayQuery.CandidateInstanceIndex(),
                rayQuery.CandidateGeometryIndex(),
                rayQuery.CandidatePrimitiveIndex(),
                rayQuery.CandidateTriangleBarycentrics()
                //, debug
                ) )
            {
                rayQuery.CommitNonOpaqueTriangleHit();
                //break; // <- TODO: revisit - not needed when using RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH?
            }
        }
    }
        
#if ENABLE_DEBUG_VIZUALISATIONS && ENABLE_DEBUG_LINES_VIZ && !NON_PATH_TRACING_PASS && PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES
    float visible = rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    if (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        ray.TMax = rayQuery.CommittedRayT();    // <- this gets passed via NvMakeHitWithRecordIndex/NvInvokeHitObject as RayTCurrent() or similar in ubershader path

    if( debug.IsDebugPixel() )
        debug.DrawLine(ray.Origin, ray.Origin+ray.Direction*ray.TMax, float4(visible.x, visible.x, 0.8, 0.2), float4(visible.x, visible.x, 0.8, 0.2));
#endif

    return !rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

void Bridge::traceScatterRay(const PathState path, inout RayQuery<RAY_FLAG_NONE, RTXPT_FLAG_ALLOW_OPACITY_MICROMAPS> rayQuery, const float2 tMinMax, DebugContext debug)
{
    RayDesc ray = path.getScatterRay().toRayDesc();
    ray.TMin = tMinMax.x;
    ray.TMax = tMinMax.y;
    rayQuery.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xff, ray);

    while (rayQuery.Proceed())
    {
        if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            // A.k.a. 'Anyhit' shader!
            [branch]if (Bridge::AlphaTest(
                rayQuery.CandidateInstanceID(),
                rayQuery.CandidateInstanceIndex(),
                rayQuery.CandidateGeometryIndex(),
                rayQuery.CandidatePrimitiveIndex(),
                rayQuery.CandidateTriangleBarycentrics()
                //, workingContext.Debug
                )
            )
            {
                rayQuery.CommitNonOpaqueTriangleHit();
            }
        }
    }
}

EnvMap Bridge::CreateEnvMap()
{
    return EnvMap::make( t_EnvironmentMap, s_EnvironmentMapSampler, g_Const.envMapSceneParams );
}

EnvMapSampler Bridge::CreateEnvMapImportanceSampler()
{
    return EnvMapSampler::make(
        s_EnvironmentMapImportanceSampler,
        t_EnvironmentMapImportanceMap,
        g_Const.envMapImportanceSamplingParams,
        t_EnvironmentMap,
        s_EnvironmentMapSampler,
        g_Const.envMapSceneParams/*,
        t_PresampledEnvMapBuffer*/
    );
}

LightSampler Bridge::CreateLightSampler( const uint2 pixelPos, float rayConeWidth, float totalPathLength )
{
    bool isScreenSpaceCoherent = LightSampler::IsScreenSpaceCoherentHeuristic( t_LightsCB, rayConeWidth, totalPathLength );
    return LightSampler::make( t_LightsCB, t_Lights, t_LightsEx, t_LightProxyCounters, t_LightProxyIndices, t_LightLocalSamplingBuffer, t_EnvLookupMap, u_LightFeedbackTotalWeight, u_LightFeedbackCandidates, pixelPos, isScreenSpaceCoherent );
}

LightSampler Bridge::CreateLightSampler( const uint2 pixelPos, bool isScreenSpaceCoherent )
{
    return LightSampler::make( t_LightsCB, t_Lights, t_LightsEx, t_LightProxyCounters, t_LightProxyIndices, t_LightLocalSamplingBuffer, t_EnvLookupMap, u_LightFeedbackTotalWeight, u_LightFeedbackCandidates, pixelPos, isScreenSpaceCoherent );
}

bool Bridge::HasEnvMap()
{
    return g_Const.envMapSceneParams.Enabled;
}

float Bridge::DiffuseEnvironmentMapMIPOffset( )
{
    return g_Const.ptConsts.EnvironmentMapDiffuseSampleMIPLevel;
}

void Bridge::ExportSurfaceInit(uint2 pixelPos)
{
    u_Depth[pixelPos] = 0;                  // this is a signal that data is invalid - there's (rare) cases where neither ExportSurface or ExportNonSurface get called
    u_SpecularHitT[pixelPos] = 0;           // it is common for this to be missing
    
    // u_MotionVectors[pixelPos] = float4( 0, 0, 0, 0 );   // this should not be strictly necessary as we already know from u_Depth[] that the signal is invalid
    // DebugPixel( pixelPos.xy, float4( 0.0.xxx, 1 ) ); 
}

void Bridge::ExportSurface(const PathState path, PathTracer::SurfaceData surfaceData, float sceneLength, float3 motionVectors )
{
    uint2 pixelPos = path.GetPixelPos();

    u_MotionVectors[pixelPos]   = float4(motionVectors, 0);

    const Ray cameraRay = Bridge::computeCameraRay( pixelPos );

    float3 virtualWorldPos = cameraRay.origin + cameraRay.dir * sceneLength;
    float4 clipPos = mul(float4(/*bridgedData.shadingData.posW*/virtualWorldPos, 1), g_Const.view.matWorldToClip);
    u_Depth[pixelPos] = clipPos.z / clipPos.w;
    u_Throughput[pixelPos] = Pack_R11G11B10_FLOAT(saturate(path.GetThp()));
    
#if EXPORT_GBUFFER
    if (g_Const.ptConsts.useReSTIRDI || g_Const.ptConsts.useReSTIRGI)
    {
        // compute address for current output - it ping pongs based on frameIndex!
        const uint idxPingPong = (g_Const.ptConsts.frameIndex % 2) == (uint)0;
        const uint idx = GenericTSPixelToAddress(pixelPos, idxPingPong, g_Const.ptConsts.genericTSLineStride, g_Const.ptConsts.genericTSPlaneStride);

        u_SurfaceData[idx] = RunCompress( CollectGBufferSurface(path, surfaceData, cameraRay) ); //ExtractPackedGbufferSurfaceData(pixelPos, sp, dominantStablePlaneIndex, stableBranchID);
    }
#endif
    // DebugPixel( pixelPos.xy, float4( 0, 0.2, 0, 1 ) );
}

void Bridge::ExportNonSurface(const PathState path, float3 virtualWorldPos, float3 motionVectors )
{
    uint2 pixelPos = path.GetPixelPos();

    u_MotionVectors[pixelPos]   = float4(motionVectors, 0);

    float4 clipPos = mul(float4( /*bridgedData.shadingData.posW*/virtualWorldPos, 1), g_Const.view.matWorldToClip);
    u_Depth[pixelPos] = clipPos.z / clipPos.w;
    u_Throughput[pixelPos] = 0;

    //DebugPixel( pixelPos.xy, float4( 0, 0, 0.2, 1 ) );

#if EXPORT_GBUFFER
    if (g_Const.ptConsts.useReSTIRDI || g_Const.ptConsts.useReSTIRGI)
    {
        // compute address for current output - it ping pongs based on frameIndex!
        const uint idxPingPong = (g_Const.ptConsts.frameIndex % 2) == (uint)0;
        const uint idx = GenericTSPixelToAddress(pixelPos, idxPingPong, g_Const.ptConsts.genericTSLineStride, g_Const.ptConsts.genericTSPlaneStride);

        u_SurfaceData[idx] = RunCompress(PathTracerCollectedSurfaceData::makeEmpty()); 
    }
#endif
}

void Bridge::ExportSpecHitTStart(const PathState path)
{
    u_SpecularHitT[path.GetPixelPos()] = -path.GetSceneLength();
}

void Bridge::ExportSpecHitTStop(const PathState path)
{
    uint2 pixelPos = path.GetPixelPos();

    float denoisingSceneLength = u_SpecularHitT[pixelPos];
    if (denoisingSceneLength < 0)  // 0 means not initialized - nothing to do here (shouldn't happen really!); >0 means we've already filled it up in previous pass when using multiple samples per pixel - again, nothing to do here
    {
        // if (denoisingSceneLength > 0)  
        //      DebugPixel( pixelPos.xy, float4( 1.0, 0, 0.0, 1 ) ); // bug!
    
        float specHitT = max( 0, path.GetSceneLength() + denoisingSceneLength );
        u_SpecularHitT[pixelPos] = specHitT;
    }
//     if (specHitT > 0.0)
//         //DebugPixel( pixelPos.xy, float4( 0.0, saturate(specHitT/10.0), 1.0, 1 ) );
//        DebugPixel( pixelPos.xy, float4( saturate(specHitT/10.0).xxx, 1 ) );
}

PathTracer::WorkingContext GetWorkingContext()
{
    PathTracer::WorkingContext ret;
    ret.PtConsts = g_Const.ptConsts;
    ret.Debug.Init( g_Const.debug, u_FeedbackBuffer, u_DebugLinesBuffer, u_DebugDeltaPathTree, u_DeltaPathSearchStack );
    ret.StablePlanes = StablePlanesContext::make(u_StablePlanesHeader, u_StablePlanesBuffer, u_StableRadiance, g_Const.ptConsts);
    ret.OutputColor = u_OutputColor;
    return ret;
}


#endif // __PATH_TRACER_BRIDGE_DONUT_HLSLI__
