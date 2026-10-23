/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef __INTRO_FASTHITSHADERS_HLSL__
#define __INTRO_FASTHITSHADERS_HLSL__

#include "../PathTracer/PathTracerTypes.hlsli"

#include "../Bindings/ShaderResourceBindings.hlsli"

#include "IntroBridge.hlsli"
#include "../DonutGeometryHelpers.hlsli"
#include "../Bindings/GBufferBindings.hlsli"

#include "MaterialEvalSimplified.hlsli"
#include "IntroSampleCommon.hlsli"
#include "IntroHitLighting.hlsli" // ComputeHitLighting + IntroEvaluateHitDirectLighting (+ t90/t91, shadow rays)

#define CLOSEST_HIT_HIGH 0
#define CLOSEST_HIT_LOW 1

struct [raypayload] PayloadSimplified
{
    float3 baseColor : read(caller) : write(closesthit);
    uint shaderId : read(caller) : write(closesthit);

    float3 normal : read(caller) : write(closesthit);
    float hitDistance : read(caller) : write(caller, closesthit);
};

[shader("closesthit")]
void CLOSESTHIT_ENTRY(inout PayloadSimplified payload : SV_RayPayload, in BuiltInTriangleIntersectionAttributes attrib)
{
    // Load geometry data from hit indices.
    DebugContext dummy;
    const DonutGeometryAttributes attrToLoad = GeomAttr_TexCoord | GeomAttr_Normal;
    DonutGeometrySample donutGS = getGeometryFromHit(InstanceIndex(), GeometryIndex(), PrimitiveIndex(),
        attrib.barycentrics, attrToLoad,
        t_InstanceData, t_GeometryData, t_GeometryDebugData, WorldRayDirection(), dummy);

    // Load material data    
    uint subInstanceDataIndex = donutGS.instance.firstGeometryInstanceIndex + GeometryIndex();
    uint materialIndex = t_SubInstanceData[subInstanceDataIndex].AlphaTextureIndex_PTMaterialDataIndex & 0xFFFF;

    // Use the interpolated normal, oriented toward the ray origin; no normal map at secondary hits.
    float3 geoNormal = donutGS.frontFacing ? donutGS.geometryNormal : -donutGS.geometryNormal;

    SimplifiedSurface surface = EvalSimplifiedSurface(materialIndex, donutGS.texcoord);

    // Store surface info in the payload
    payload.baseColor = surface.baseColor;
    payload.normal = geoNormal;
    payload.shaderId = surface.shaderId;
    payload.hitDistance = RayTCurrent();
}

[shader("anyhit")]
void ANYHIT_ENTRY(inout PayloadSimplified payload : SV_RayPayload, in BuiltInTriangleIntersectionAttributes attrib)
{
    if (!Bridge::AlphaTest(InstanceID(), InstanceIndex(), GeometryIndex(), PrimitiveIndex(), attrib.barycentrics))
        IgnoreHit();
}

#endif // __INTRO_FASTHITSHADERS_HLSL__
