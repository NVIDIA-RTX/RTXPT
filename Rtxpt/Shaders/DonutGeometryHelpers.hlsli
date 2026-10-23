/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __DONUT_GEOMETRY_HELPERS_HLSLI__
#define __DONUT_GEOMETRY_HELPERS_HLSLI__

#include "PathTracer/Config.h"
#include "PathTracer/PathTracerTypes.hlsli"
#include "Libraries/ShaderDebug/ShaderDebug.hlsl"

#include <donut/shaders/bindless.h>
#include <donut/shaders/utils.hlsli>
#include <donut/shaders/binding_helpers.hlsli>
#include <donut/shaders/surface.hlsli>
#include <donut/shaders/scene_material.hlsli>

#include "Bindings/SceneBindings.hlsli"

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
        // SafeNormalize avoids NaN propagation when a vertex has all-zero packed normal
        // bytes (Unpack_RGB8_SNORM returns (0,0,0), and normalize((0,0,0)) is NaN).
        // The NaN otherwise propagates into payload.normal, then into the next ray's
        // direction, leading to undefined BVH traversal and a TDR on RTX 3060/596.36.
        normals[0] = SafeNormalize(Unpack_RGB8_SNORM(vertexBuffer.Load(gs.geometry.normalOffset + indices[0] * c_SizeOfNormal)));
        normals[1] = SafeNormalize(Unpack_RGB8_SNORM(vertexBuffer.Load(gs.geometry.normalOffset + indices[1] * c_SizeOfNormal)));
        normals[2] = SafeNormalize(Unpack_RGB8_SNORM(vertexBuffer.Load(gs.geometry.normalOffset + indices[2] * c_SizeOfNormal)));

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

#endif // __DONUT_GEOMETRY_HELPERS_HLSLI__
