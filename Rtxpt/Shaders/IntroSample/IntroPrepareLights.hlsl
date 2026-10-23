/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Intro sample emissive-triangle bake. Modeled on the triangle branch of
// RTXDI's prepare-lights pass (see Rtxpt/RTXDI/PrepareLights.hlsl): the CPU builds one
// PrepareLightsTask per emissive geometry, this shader binary-searches the task list,
// reads the triangle through the bindless vertex/index buffers, evaluates emissive
// radiance (with an anisotropic emissive-texture fetch), and packs a TriangleLight.
//
// Differences from the full RTXDI port:
//  - Writes the Intro's two parallel SoA light buffers (Base + Extended) instead of a
//    single AoS buffer, so the bake output drops straight into t_IntroRTXDILightsBase /
//    t_IntroRTXDILightsEx consumed by the ReSTIR DI passes.
//  - Triangles are appended into the local light region after the CPU-baked analytic
//    local lights, at g_Const.triangleDestBaseIndex.
//  - No PDF texture, light-index mapping, environment or primitive-light paths: the Intro
//    uses uniform local-light sampling, remaps between buffer halves and resets reuse if the light layout changes.

#pragma pack_matrix(row_major)

#include <donut/shaders/bindless.h>
#include <donut/shaders/binding_helpers.hlsli>
#include "../Bindings/PrepareLightsTask.h"
#include "../PathTracer/Lighting/PolymorphicLight.h"   // PolymorphicLightInfo*
#include "../PathTracer/Utils/Utils.hlsli"
#include "../SubInstanceData.h"
#include "../PathTracer/Materials/MaterialPT.h"

// Enable just the polymorphic light types the Intro bridge uses (environment map disabled),
// then pull in PolymorphicLight to get TriangleLight::Store with the same encoding the
// ReSTIR DI passes decode through PolymorphicLight::CalcSample.
#define POLYLIGHT_OVERRIDE_CONFIG
#define POLYLIGHT_SPHERE_ENABLE         1
#define POLYLIGHT_POINT_ENABLE          1
#define POLYLIGHT_TRIANGLE_ENABLE       1
#define POLYLIGHT_DIRECTIONAL_ENABLE    1
#define POLYLIGHT_ENV_ENABLE            0
#define POLYLIGHT_QT_ENV_ENABLE         0
#define POLYLIGHT_LIGHT_CLUSTER         0
#define POLYLIGHT_CONFIGURED
#include "../PathTracer/Lighting/PolymorphicLight.hlsli"

struct IntroPrepareLightsConstants
{
    uint numTasks;
    uint triangleDestBaseIndex;     // first slot in the light buffer for baked triangles (current ping-pong region + analytic local count)
    uint2 _padding;
};

ConstantBuffer<IntroPrepareLightsConstants>  g_Const             : register(b0);
RWStructuredBuffer<PolymorphicLightInfo>     u_LightsBase        : register(u0);
RWStructuredBuffer<PolymorphicLightInfoEx>   u_LightsEx          : register(u1);
StructuredBuffer<PrepareLightsTask>          t_TaskBuffer        : register(t0);
StructuredBuffer<SubInstanceData>            t_SubInstanceData   : register(t1);
StructuredBuffer<InstanceData>               t_InstanceData      : register(t2);
StructuredBuffer<GeometryData>               t_GeometryData      : register(t3);
StructuredBuffer<PTMaterialData>             t_PTMaterialData    : register(t5);
SamplerState                                 s_MaterialSampler   : register(s0);

VK_BINDING(0, 1) ByteAddressBuffer t_BindlessBuffers[]           : register(t0, space1);
VK_BINDING(1, 1) Texture2D t_BindlessTextures[]                  : register(t0, space2);

bool FindTask(uint dispatchThreadId, out PrepareLightsTask task)
{
    // Binary search for the task that contains the current thread's output index:
    //   task.lightBufferOffset <= dispatchThreadId < (task.lightBufferOffset + task.triangleCount)
    int left = 0;
    int right = int(g_Const.numTasks) - 1;

    while (right >= left)
    {
        int middle = (left + right) / 2;
        task = t_TaskBuffer[middle];

        int tri = int(dispatchThreadId) - int(task.lightBufferOffset); // signed

        if (tri < 0)
            right = middle - 1;     // Go left
        else if (tri < task.triangleCount)
            return true;            // Found it!
        else
            left = middle + 1;      // Go right
    }

    return false;
}

[numthreads(256, 1, 1)]
void main(uint dispatchThreadId : SV_DispatchThreadID, uint groupThreadId : SV_GroupThreadID)
{
    PrepareLightsTask task = (PrepareLightsTask)0;

    if (!FindTask(dispatchThreadId, task))
        return;

    uint triangleIdx = dispatchThreadId - task.lightBufferOffset;

    InstanceData instance = t_InstanceData[(task.instanceAndGeometryIndex >> 12) & 0x7FFFF];
    GeometryData geometry = t_GeometryData[instance.firstGeometryIndex + (task.instanceAndGeometryIndex & 0xfff)];

    uint materialIndex = t_SubInstanceData[instance.firstGeometryInstanceIndex + (task.instanceAndGeometryIndex & 0xfff)].AlphaTextureIndex_PTMaterialDataIndex & 0xFFFF;
    PTMaterialData material = t_PTMaterialData[materialIndex];

    ByteAddressBuffer indexBuffer = t_BindlessBuffers[NonUniformResourceIndex(geometry.indexBufferIndex)];
    ByteAddressBuffer vertexBuffer = t_BindlessBuffers[NonUniformResourceIndex(geometry.vertexBufferIndex)];

    uint3 indices = indexBuffer.Load3(geometry.indexOffset + triangleIdx * c_SizeOfTriangleIndices);

    float3 positions[3];
    positions[0] = asfloat(vertexBuffer.Load3(geometry.positionOffset + indices[0] * c_SizeOfPosition));
    positions[1] = asfloat(vertexBuffer.Load3(geometry.positionOffset + indices[1] * c_SizeOfPosition));
    positions[2] = asfloat(vertexBuffer.Load3(geometry.positionOffset + indices[2] * c_SizeOfPosition));

    positions[0] = mul(instance.transform, float4(positions[0], 1)).xyz;
    positions[1] = mul(instance.transform, float4(positions[1], 1)).xyz;
    positions[2] = mul(instance.transform, float4(positions[2], 1)).xyz;

    float3 radiance = material.EmissiveColor;

    if ((material.EmissiveTextureIndex != 0xFFFFFFFF) && (geometry.texCoord1Offset != ~0u) && ((material.Flags & PTMaterialFlags_UseEmissiveTexture) != 0))
    {
        Texture2D emissiveTexture = t_BindlessTextures[NonUniformResourceIndex(material.EmissiveTextureIndex & 0xFFFF)];

        // Load the vertex UVs
        float2 uvs[3];
        uvs[0] = asfloat(vertexBuffer.Load2(geometry.texCoord1Offset + indices[0] * c_SizeOfTexcoord));
        uvs[1] = asfloat(vertexBuffer.Load2(geometry.texCoord1Offset + indices[1] * c_SizeOfTexcoord));
        uvs[2] = asfloat(vertexBuffer.Load2(geometry.texCoord1Offset + indices[2] * c_SizeOfTexcoord));

        // Calculate the triangle edges and edge lengths in UV space
        float2 edges[3];
        edges[0] = uvs[1] - uvs[0];
        edges[1] = uvs[2] - uvs[1];
        edges[2] = uvs[0] - uvs[2];

        float3 edgeLengths;
        edgeLengths[0] = length(edges[0]);
        edgeLengths[1] = length(edges[1]);
        edgeLengths[2] = length(edges[2]);

        // Find the shortest edge and the other two (longer) edges
        float2 shortEdge;
        float2 longEdge1;
        float2 longEdge2;

        if (edgeLengths[0] < edgeLengths[1] && edgeLengths[0] < edgeLengths[2])
        {
            shortEdge = edges[0];
            longEdge1 = edges[1];
            longEdge2 = edges[2];
        }
        else if (edgeLengths[1] < edgeLengths[2])
        {
            shortEdge = edges[1];
            longEdge1 = edges[2];
            longEdge2 = edges[0];
        }
        else
        {
            shortEdge = edges[2];
            longEdge1 = edges[0];
            longEdge2 = edges[1];
        }

        // Use anisotropic sampling with the sample ellipse axes parallel to the short edge
        // and the median from the opposite vertex to the short edge.
        float2 shortGradient = shortEdge * (2.0 / 3.0);
        float2 longGradient = (longEdge1 + longEdge2) / 3.0;

        float2 centerUV = (uvs[0] + uvs[1] + uvs[2]) / 3.0;
        float3 emissiveMask = emissiveTexture.SampleGrad(s_MaterialSampler, centerUV, shortGradient, longGradient).rgb;

        radiance *= emissiveMask;
    }

    radiance.rgb = max(0, radiance.rgb);

    // Check if the transform flips the coordinate system handedness (negative determinant);
    // if so, swap the edges so the single-sided triangle normal faces outward.
    float3x3 transform;
    transform._m00_m01_m02 = (instance.transform._m00_m01_m02);
    transform._m10_m11_m12 = (instance.transform._m10_m11_m12);
    transform._m20_m21_m22 = (instance.transform._m20_m21_m22);

    bool isFlipped = determinant(transform) < 0.f;

    TriangleLight triLight;
    triLight.base = positions[0];
    if (!isFlipped)
    {
        triLight.edge1 = positions[1] - positions[0];
        triLight.edge2 = positions[2] - positions[0];
    }
    else
    {
        triLight.edge1 = positions[2] - positions[0];
        triLight.edge2 = positions[1] - positions[0];
    }
    triLight.radiance = radiance;

    PolymorphicLightInfoFull lightInfo = triLight.Store();

    uint dst = g_Const.triangleDestBaseIndex + task.lightBufferOffset + triangleIdx;
    u_LightsBase[dst] = lightInfo.Base;
    u_LightsEx[dst] = lightInfo.Extended;
}
