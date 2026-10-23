/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __DONUT_BRIDGE_COMMON_HLSLI__
#define __DONUT_BRIDGE_COMMON_HLSLI__

#include "PathTracer/Config.h"
#include "PathTracer/PathTracerTypes.hlsli"

#include "Bindings/ShaderResourceBindings.hlsli"
#include "Bindings/SceneBindings.hlsli"
#include "Bindings/SamplerBindings.hlsli"

#include <donut/shaders/bindless.h>
#include <donut/shaders/utils.hlsli>

namespace Bridge
{

static Ray computeCameraRay(const uint2 pixelPos)
{
    uint sampleIndex = g_Const.ptConsts.sampleBaseIndex + g_MiniConst.params.x;
    SampleGenerator sampleGenerator = SampleGenerator::make(SampleGeneratorVertexBase::make(pixelPos, 0, sampleIndex));

    // compute camera ray! would make sense to compile out if unused
    float2 subPixelOffset = g_Const.ptConsts.camera.Jitter + (sampleNext2D(sampleGenerator) - 0.5.xx) * g_Const.ptConsts.perPixelJitterAAScale;
    const float2 cameraDoFSample = sampleNext2D(sampleGenerator);
    //return ComputeRayPinhole( g_Const.ptConsts.camera, pixelPos, subPixelOffset );
    Ray ray = ComputeRayThinlens(g_Const.ptConsts.camera, pixelPos, subPixelOffset, cameraDoFSample);

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

// 2.5D motion vectors
static float3 computeMotionVector(float3 posW, float3 prevPosW)
{
    return ComputeMotionVector( g_Const.view, g_Const.previousView, posW, prevPosW );
}

// 2.5D motion vectors
static float3 computeSkyMotionVector(const uint2 pixelPos)
{
    SimpleViewConstants view = g_Const.view;
    SimpleViewConstants previousView = g_Const.previousView;

    float4 clipPos = float4((pixelPos + 0.5.xx) / g_Const.view.clipToWindowScale + float2(-1, 1), 1e-7, 1.0);
    float4 viewPos = mul(clipPos, view.matClipToWorldNoOffset); viewPos.xyzw /= viewPos.w;
    float4 prevClipPos = mul(viewPos, previousView.matWorldToClipNoOffset);
    prevClipPos.xyz /= prevClipPos.w;

    float3 motion;
    motion.xy = (prevClipPos.xy - clipPos.xy) * view.clipToWindowScale;
    //motion.xy += (view.pixelOffset - previousView.pixelOffset); <- no longer needed, using NoOffset matrices
    motion.z = 0; //prevClipPos.w - clipPos.w; // Use view depth

    return motion;
}

// DXIL does not allow user-defined struct parameters in [[noinline]] functions, so SubInstanceData is
// decomposed into individual scalars at the call sites. AlphaCutoff/AlphaTextureIndex are inlined manually.
//[[noinline]]
static bool AlphaTestImpl(
    uint GlobalGeometryIndex_FlagsAndAlphaCutoff, uint AlphaTextureIndex_PTMaterialDataIndex,
#if SUBINSTANCEDATA_EXTENDED
    uint IndexBufferIndex_VertexBufferIndex, uint IndexOffset, uint TexCoord1Offset,
#endif
    uint triangleIndex, float2 rayBarycentrics)
{
    bool alphaTested = (GlobalGeometryIndex_FlagsAndAlphaCutoff & SubInstanceData::Flags_AlphaTested) != 0;
    if (!alphaTested) // note: with correct use of D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE this is unnecessary, but there are cases (such as disabling texture but leaving alpha tested state) in which this isn't handled correctly
        return true;

    // have to do all this to figure out UVs!
    float3 barycentrics;
    barycentrics.yz = rayBarycentrics;
    barycentrics.x = 1.0 - (barycentrics.y + barycentrics.z);

    float2 texcoord;
#if !SUBINSTANCEDATA_EXTENDED
    {
        GeometryData geometry = t_GeometryData[GlobalGeometryIndex_FlagsAndAlphaCutoff & SubInstanceData::GlobalGeometryIndexMask];
        ByteAddressBuffer indexBuffer = t_BindlessBuffers[NonUniformResourceIndex(geometry.indexBufferIndex)];
        ByteAddressBuffer vertexBuffer = t_BindlessBuffers[NonUniformResourceIndex(geometry.vertexBufferIndex)];
        uint3 indices = indexBuffer.Load3(geometry.indexOffset + triangleIndex * c_SizeOfTriangleIndices);
        texcoord  = barycentrics.x * asfloat(vertexBuffer.Load2(geometry.texCoord1Offset + indices[0] * c_SizeOfTexcoord));
        texcoord += barycentrics.y * asfloat(vertexBuffer.Load2(geometry.texCoord1Offset + indices[1] * c_SizeOfTexcoord));
        texcoord += barycentrics.z * asfloat(vertexBuffer.Load2(geometry.texCoord1Offset + indices[2] * c_SizeOfTexcoord));
    }
#else
    {
        ByteAddressBuffer indexBuffer = t_BindlessBuffers[NonUniformResourceIndex(IndexBufferIndex_VertexBufferIndex >> 16)];
        ByteAddressBuffer vertexBuffer = t_BindlessBuffers[NonUniformResourceIndex(IndexBufferIndex_VertexBufferIndex & 0xFFFF)];
        uint3 indices = indexBuffer.Load3(IndexOffset + triangleIndex * c_SizeOfTriangleIndices);
        texcoord  = barycentrics.x * asfloat(vertexBuffer.Load2(TexCoord1Offset + indices[0] * c_SizeOfTexcoord));
        texcoord += barycentrics.y * asfloat(vertexBuffer.Load2(TexCoord1Offset + indices[1] * c_SizeOfTexcoord));
        texcoord += barycentrics.z * asfloat(vertexBuffer.Load2(TexCoord1Offset + indices[2] * c_SizeOfTexcoord));
    }
#endif

    // sample the alpha (opacity) texture and test vs the threshold
    uint alphaTextureIndex = AlphaTextureIndex_PTMaterialDataIndex >> 16;                          // AlphaTextureIndex()
    float alphaCutoff = SubInstanceData::DecodeAlphaCutoff(GlobalGeometryIndex_FlagsAndAlphaCutoff >> SubInstanceData::Flags_AlphaOffsetOffset);
    Texture2D diffuseTexture = t_BindlessTextures[NonUniformResourceIndex(alphaTextureIndex)];
    float opacityValue = diffuseTexture.SampleLevel(s_MaterialSampler, texcoord, 0).a; // <- hard coded to .a channel but we might want a separate alpha only texture, maybe in .g of BC1
    return opacityValue >= alphaCutoff;
}

static bool AlphaTest(uint instanceID, uint instanceIndex, uint geometryIndex, uint triangleIndex, float2 rayBarycentrics)
{
    SubInstanceData subInstanceData = t_SubInstanceData[(instanceID + geometryIndex)];
    return AlphaTestImpl(
        subInstanceData.GlobalGeometryIndex_FlagsAndAlphaCutoff,
        subInstanceData.AlphaTextureIndex_PTMaterialDataIndex,
#if SUBINSTANCEDATA_EXTENDED
        subInstanceData.IndexBufferIndex_VertexBufferIndex,
        subInstanceData.IndexOffset,
        subInstanceData.TexCoord1Offset,
#endif
        triangleIndex, rayBarycentrics);
}

static bool AlphaTestVisibilityRay(uint instanceID, uint instanceIndex, uint geometryIndex, uint triangleIndex, float2 rayBarycentrics)
{
    SubInstanceData subInstanceData = t_SubInstanceData[(instanceID + geometryIndex)];

    bool excludeFromNEE = (subInstanceData.GlobalGeometryIndex_FlagsAndAlphaCutoff & SubInstanceData::Flags_ExcludeFromNEE) != 0;
    if (excludeFromNEE)
        return false;

    return AlphaTestImpl(
        subInstanceData.GlobalGeometryIndex_FlagsAndAlphaCutoff,
        subInstanceData.AlphaTextureIndex_PTMaterialDataIndex,
#if SUBINSTANCEDATA_EXTENDED
        subInstanceData.IndexBufferIndex_VertexBufferIndex,
        subInstanceData.IndexOffset,
        subInstanceData.TexCoord1Offset,
#endif
        triangleIndex, rayBarycentrics);
}

} // namespace Bridge

#endif // __DONUT_BRIDGE_COMMON_HLSLI__
