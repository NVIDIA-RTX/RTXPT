/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __PATH_TRACER_SURFACE_BRIDGE_DONUT_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __PATH_TRACER_SURFACE_BRIDGE_DONUT_HLSLI__

// easier if we let Donut do this!
#define ENABLE_METAL_ROUGH_RECONSTRUCTION 1

#include "PathTracer/Config.h"
#include "PathTracerSurfaceBridge.hlsli"
#include "Libraries/ShaderDebug/ShaderDebug.hlsl"
#include "PathTracer/PathTracerTypes.hlsli"

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

#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD // build
#if PT_USE_RESTIR_GI || PT_USE_RESTIR_DI
#define EXPORT_GBUFFER 1
#endif
#endif

#if EXPORT_GBUFFER
#include "Misc/GBufferHelpers.hlsli"
#endif

#include "DonutGeometryHelpers.hlsli"

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
    
    // Texture applied before the conversion, so a dark texel can bring an over-range factor back into fp16 range
    const float3 emissiveColor = ((material.Flags & PTMaterialFlags_UseEmissiveTexture) == 0) ? material.EmissiveColor : (material.EmissiveColor * textures.emissive.rgb);
    result.emissiveColor = ToLpRadiance( emissiveColor );

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

static void surfaceDebugViz(uint2 pixelPos, const PathTracer::Surface surface, const float2 barycentrics, const float3 rayDir, const RayCone rayCone, const int pathVertexIndex, const OpacityMicroMapDebugInfo ommDebug, uint materialID, DebugContext debug)
{
#if ENABLE_DEBUG_VIZUALISATIONS && !NON_PATH_TRACING_PASS && ENABLE_DEBUG_SURFACE_VIZ
    if (g_Const.debug.debugViewType == (int)DebugViewType::Disabled || pathVertexIndex != 1)
        return;

    const ShadingData shadingData = surface.shadingData;
    const ActiveBSDF bsdf = surface.bsdf;

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
    case ((int)DebugViewType::FirstHit_GeometryNormal):              debug.DrawDebugViz(pixelPos, float4(DbgShowNormalSRGB(shadingData.GeometryNormal()), 1.0)); break;
    case ((int)DebugViewType::FirstHit_ShadingNormal):               debug.DrawDebugViz(pixelPos, float4(DbgShowNormalSRGB(shadingData.N), 1.0)); break;
    case ((int)DebugViewType::FirstHit_ShadingTangent):              debug.DrawDebugViz(pixelPos, float4(DbgShowNormalSRGB(shadingData.T()), 1.0)); break;
    case ((int)DebugViewType::FirstHit_ShadingBitangent):            debug.DrawDebugViz(pixelPos, float4(DbgShowNormalSRGB(shadingData.B()), 1.0)); break;
    case ((int)DebugViewType::FirstHit_FrontFacing):                 debug.DrawDebugViz(pixelPos, float4(saturate(float3(0.15, 0.1 + shadingData.frontFacing, 0.15)), 1.0)); break;
    case ((int)DebugViewType::FirstHit_ThinSurface):                 debug.DrawDebugViz(pixelPos, float4(saturate(float3(0.15, 0.1 + shadingData.shadingFlags.isThinSurface(), 0.15)), 1.0)); break;
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

PathTracer::Surface Bridge::loadSurface( const uint instanceIndex, const uint geometryIndex, const uint triangleIndex, const float2 barycentrics,
    const float3 rayDir, const RayCone rayCone, const int pathVertexIndex, const uint2 pixelPos, DebugContext debug )
{
    DonutGeometryAttributes attributes = GeomAttr_TexCoord | GeomAttr_Position | GeomAttr_Normal | GeomAttr_Tangents;
#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD // otherwise motion vectors not needed
    attributes |= GeomAttr_PrevPosition;
#endif

    DonutGeometrySample donutGS = getGeometryFromHit( instanceIndex, geometryIndex, triangleIndex, barycentrics, 
        attributes, t_InstanceData, t_GeometryData, t_GeometryDebugData, rayDir, debug );

    // Convert Donut to RTXPT data! 

    // World pos and prev world pos
    float3 posW     = mul(donutGS.instance.transform, float4(donutGS.objectSpacePosition, 1.0)).xyz;
    float3 prevPosW = mul(donutGS.instance.prevTransform, float4(donutGS.prevObjectSpacePosition, 1.0)).xyz;

    // posW's fp32 error scale for ComputeRayOrigin: the larger of |posW| and the triangle's object-space extent through the instance's linear part
    float3 objExtentW   = mul(abs((float3x3)donutGS.instance.transform), max(abs(donutGS.vertexPositions[0]), max(abs(donutGS.vertexPositions[1]), abs(donutGS.vertexPositions[2]))));
    float posErrorScale = max(max(abs(posW.x), max(abs(posW.y), abs(posW.z))), max(objExtentW.x, max(objExtentW.y, objExtentW.z)));

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
    ActiveTextureSampler textureSampler = createTextureSampler( rayCone, rayDir, coneTexLODValue, donutGS.flatNormal/*donutGS.geometryNormal*/, true, g_Const.ptConsts.texLODBias
#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
        ,stfSamplerState
#endif
    );

    // See MaterialFactory.hlsli in Falcor
    ShadingData ptShadingData = ShadingData::make();

    ptShadingData.posW = posW;
    ptShadingData.toPrevPosW = prevPosW - posW;

    //ptShadingData.uv   = lpfloat2(donutGS.texcoord);
    ptShadingData.V    = -rayDir;
    ptShadingData.N    = donutGS.geometryNormal;

    uint subInstanceDataIndex = donutGS.instance.firstGeometryInstanceIndex + geometryIndex;

    uint materialIndex = t_SubInstanceData[subInstanceDataIndex].AlphaTextureIndex_PTMaterialDataIndex & 0xFFFF;

    // Get donut material (normal map is evaluated here)
    MaterialProperties donutMaterial = sampleGeometryMaterialRTXPT(donutGS, materialIndex, MatAttr_All, s_MaterialSampler, textureSampler);

    // Primitive data
    ptShadingData.faceNCorrected = (donutGS.frontFacing)?(donutGS.flatNormal):(-donutGS.flatNormal);
    ptShadingData.frontFacing = donutGS.frontFacing;

    ptShadingData.SetGeometryNormalAndShadowTerminatorK( (donutGS.frontFacing)?(donutGS.geometryNormal):(-donutGS.geometryNormal), donutMaterial.shadowNoLFadeout );

    ptShadingData.N = (donutGS.frontFacing)?(donutMaterial.shadingNormal):(-donutMaterial.shadingNormal);
    ptShadingData.SetTangentW(donutGS.tangent);

    // Helper function to adjust the shading normal to reduce black pixels due to back-facing view direction. Note: This breaks the reciprocity of the BSDF!
    // This also reorthonormalizes the frame based on the normal map, which is necessary (see `ptShadingData.N = donutMaterial.shadingNormal;` line above)
    bool ignoreTangent = (donutMaterial.flags & PTMaterialFlags_IgnoreMeshTangentSpace) != 0;
    ptShadingData.AdjustShadingNormalAndComputeTangentSpace( true, ignoreTangent );
    // ^^ this should be part of material processing code

    // Donut -> Falcor
    const bool donutMaterialThinSurface = (donutMaterial.flags & PTMaterialFlags_ThinSurface) != 0;
    ptShadingData.SetMaterialIDAndPosErrorScale(materialIndex, posErrorScale);
    ptShadingData.shadingFlags = ShadingFlags::make();
    ptShadingData.shadingFlags.setNestedPriority( min( InteriorList::kMaxNestedPriority, 1 + (uint(donutMaterial.flags) >> PTMaterialFlags_NestedPriorityShift)) );   // priorities are from (1, ... kMaxNestedPriority) because 0 is used to mark empty slots and remapped to kMaxNestedPriority
    ptShadingData.shadingFlags.setThinSurface( donutMaterialThinSurface );
    ptShadingData.shadingFlags.setPSDExclude( (donutMaterial.flags & PTMaterialFlags_PSDExclude) != 0 );
    ptShadingData.shadingFlags.setPSDDominantDeltaLobeP1( (donutMaterial.flags & PTMaterialFlags_PSDDominantDeltaLobeP1Mask) >> PTMaterialFlags_PSDDominantDeltaLobeP1Shift );


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
        ptShadingData.shadingFlags.setPSDBlockMotionVectorsAtSurface( blockMVs );
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Now load the actual BSDF! Equivalent to StandardBSDF::setupBSDF
    lpfloat3    bsdfDataDiffuse              = 0;
    lpfloat     bsdfDataRoughness            = 0;
    lpfloat3    bsdfDataSpecular             = 0;
    lpfloat     bsdfDataMetallic             = 0;
    lpfloat     bsdfDataEta                  = 0;


    // A.k.a. interiorIoR
    lpfloat matIoR = donutMaterial.ior;
    // Assume the default IoR for vacuum on the front-facing side.
    // The renderer may override this for nested dielectrics (see 'handleNestedDielectrics' calling Bridge::updateOutsideIoR)
    ptShadingData.SetIoR(1.f, matIoR);

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

    ptShadingData.shadingFlags.setActiveLobes( lobeType );

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

    // Sample the emissive texture.
    // The standard material supports uniform emission over the hemisphere.
    // Note: we only support single sided emissives at the moment; If upgrading, make sure to upgrade NEE codepath as well (i.e. PolymorphicLight.hlsli)

    uint neeTriangleLightIndex = NEEAT_INVALID_LIGHT_INDEX;
    uint neeAnalyticLightIndex = NEEAT_INVALID_LIGHT_INDEX;

    if (ptShadingData.frontFacing && any(donutMaterial.emissiveColor>0))
    {
        ptShadingData.SetEmission(donutMaterial.emissiveColor);

#if !RTXPT_USE_APPROXIMATE_MIS || RTXPT_LIGHTING_ENABLE_BSDF_FEEDBACK
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

    if ( (donutMaterial.flags & PTMaterialFlags_EnableAsAnalyticLightProxy) != 0 )
        neeAnalyticLightIndex = t_SubInstanceData[subInstanceDataIndex].AnalyticProxyLightIndex;

    StandardBSDF bsdf = StandardBSDF::make(
        StandardBSDFData::make( bsdfDataDiffuse, bsdfDataSpecular, bsdfDataRoughness, bsdfDataMetallic, bsdfDataTransmission, bsdfDataDiffuseTransmission, bsdfDataSpecularTransmission ) );

    PathTracer::Surface ret = PathTracer::Surface::make(/*ptVertex, */ptShadingData, bsdf,
                                    neeTriangleLightIndex, neeAnalyticLightIndex);

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

void Bridge::ExportSurface(const PathState path, PathTracer::Surface surface, float sceneLength, float3 motionVectors )
{
    uint2 pixelPos = path.GetPixelPos();

#if !RTXPT_SKIP_STANDARD_OUTPUTS
    u_MotionVectors[pixelPos]   = float4(motionVectors, 0);
#endif

    const Ray cameraRay = Bridge::computeCameraRay( pixelPos );

#if !RTXPT_SKIP_STANDARD_OUTPUTS
    float3 virtualWorldPos = cameraRay.origin + cameraRay.dir * sceneLength;
    float4 clipPos = mul(float4(/*bridgedData.shadingData.posW*/virtualWorldPos, 1), g_Const.view.matWorldToClip);
    u_Depth[pixelPos] = clipPos.z / clipPos.w;
    u_Throughput[pixelPos] = Pack_R11G11B10_FLOAT(saturate(path.GetThp()));
#endif
    
#if EXPORT_GBUFFER
    if (g_Const.ptConsts.useReSTIRDI || g_Const.ptConsts.useReSTIRGI)
    {
        // compute address for current output - it ping pongs based on frameIndex!
        const uint idxPingPong = (g_Const.ptConsts.frameIndex % 2) == (uint)0;
        const uint idx = GenericTSPixelToAddress(pixelPos, idxPingPong, g_Const.ptConsts.genericTSLineStride, g_Const.ptConsts.genericTSPlaneStride);

#if !RTXPT_SKIP_STANDARD_OUTPUTS
        u_SurfaceData[idx] = RunCompress( CollectGBufferSurface(path, surface, cameraRay) ); //ExtractPackedGbufferSurfaceData(pixelPos, sp, dominantDenoisingLayerIndex, stableBranchID);
#endif
    }
#endif
    // DebugPixel( pixelPos.xy, float4( 0, 0.2, 0, 1 ) );
}

#endif // __PATH_TRACER_SURFACE_BRIDGE_DONUT_HLSLI__