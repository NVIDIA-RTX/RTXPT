/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Shared C++/HLSL data model for the GPU TLAS cull-and-build (TLASCull.hlsl, Sample::BuildTLASGPU /
// Sample::UploadTLASInstanceData): the static per-mesh/per-instance tables the cull pass reads, its
// per-frame constants, and its diagnostic counters.

#ifndef __GPU_TLAS_CULL_DATA_H__
#define __GPU_TLAS_CULL_DATA_H__

#if !defined(__cplusplus)
#pragma pack_matrix(row_major)
#else
using namespace donut::math;
#endif

// Object-space AABB, flat min/max layout for GPU upload - mirrors donut::math::box3
// (MeshInfo::objectSpaceBounds), which is CPU-only today.
// Field order matters: HLSL forbids a float3/float4 from straddling a 16-byte boundary and pads to the
// next one if it would, while C++ packs tightly - a float3 immediately followed by another float3 makes
// the two languages disagree on every offset from the second field on, and on the whole struct's size,
// corrupting every array element via a wrong stride. Each 16-byte block below is filled exactly.
struct TLASObjectBounds
{
    float3 Min;
    uint   Pad0;
    float3 Max;
    uint   Pad1;
};

static const uint kTLASInstanceData_IsSkinned = 1; // bit0 of Flags

// Per-mesh-instance BLAS resolution, indexed by SceneGraph instance index (position in
// GetMeshInstances(), == MeshInstance::GetInstanceIndex(), the same index Donut's InstanceData buffer
// uses). Rebuilt when the BLAS pool generation moves or a handle changes, never per frame Compacted
// BLASes do relocate, so the address here is only valid for that generation (see
// Sample::UploadTLASInstanceData).
// All-scalar: two 8-byte addresses then two 4-byte fields, so there is no vector packing rule to get
// wrong on either side.
struct TLASInstanceData
{
    uint64_t BlasAddress;
    uint64_t OmmBlasAddress; // 0 if the mesh has no attached OMM BLAS
    uint32_t MeshIndex;      // MeshInfo::globalMeshIndex - looks up TLASObjectBounds in the mesh bounds buffer
    uint32_t Flags;          // bit0 = IsSkinned
};

// Per-frame constants for TLASCull.hlsl. Loose frustum: side planes only (near/far dropped - rays need
// geometry outside the tight camera frustum for reflections/shadows/GI), grown by a margin, plus a
// distance cutoff. Params.y == 0 disables both tests and keeps only the GPU build.
struct TLASCullConstants
{
    float4 FrustumPlanes[4];    // world-space side planes (left/right/top/bottom), grown by a margin; xyz = outward normal, w = plane distance
    float4 CameraPosAndCutoff;  // xyz = camera world position, w = distance cutoff
    uint4  Params;              // x = instance count, y = cull enable, z = bit0 OnlyOMMs | bit1 Force2State | bit2 OMM enable, w = ForceOpaque
};

#if defined(__cplusplus)
// Tripwire against a future field reorder silently reintroducing an HLSL/C++ layout mismatch - see the
// field-order comment on TLASObjectBounds.
static_assert(sizeof(TLASObjectBounds) == 32);
static_assert(sizeof(TLASInstanceData) == 24);
static_assert(sizeof(TLASCullConstants) == 96);
#endif

#if !defined(__cplusplus)
// Transforms an object-space AABB to world space and tests it against 4 frustum side planes (xyz =
// outward normal, w = plane distance). Also returns worldCenter/worldExtent so a caller needing a
// distance cutoff does not redo the transform. Returns true when the box is fully outside.
// Takes loose min/max rather than a bounds struct so the per-geometry cull passes, whose bounds rows
// carry extra fields, can share it.
bool CullObjectBoundsAgainstFrustum(float3 boundsMin, float3 boundsMax, float3x4 instanceTransform, float4 frustumPlanes[4], out float3 worldCenter, out float3 worldExtent)
{
    float3 centerLocal = (boundsMin + boundsMax) * 0.5;
    float3 extentLocal = (boundsMax - boundsMin) * 0.5;

    worldCenter = mul(instanceTransform, float4(centerLocal, 1)).xyz;
    // Per-axis world extent: dot each output row's absolute basis with the local extent (standard
    // AABB-through-affine-transform bound; exact for any rotation/non-uniform scale).
    worldExtent = float3(
        dot(abs(instanceTransform[0].xyz), extentLocal),
        dot(abs(instanceTransform[1].xyz), extentLocal),
        dot(abs(instanceTransform[2].xyz), extentLocal));

    bool culled = false;
    [unroll]
    for (uint i = 0; i < 4; i++)
    {
        float3 n = frustumPlanes[i].xyz;
        float d = dot(n, worldCenter) - frustumPlanes[i].w;
        float r = dot(abs(n), worldExtent);
        if (d - r > 0)
            culled = true;
    }
    return culled;
}
#endif // !defined(__cplusplus)

#endif // __GPU_TLAS_CULL_DATA_H__
