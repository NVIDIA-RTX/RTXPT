/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// GPU TLAS instance cull-and-build - the GPU counterpart of Sample::BuildTLAS. One thread per mesh
// instance: optionally frustum(+distance)-culls the instance's object-space AABB and writes a raw
// instance-descriptor record into the buffer Sample::BuildTLASGPU feeds to
// buildTopLevelAccelStructFromBuffer. Culled instances get a zero-mask record rather than being
// compacted out (AllowEmptyInstances covers the null-ish entries), matching the zero-mask convention
// Sample::BuildTLAS already uses for its OnlyOMMs debug view.
//
// Also reproduces the per-instance OMM-BLAS / force-opaque / instance-mask selection Sample::BuildTLAS
// does on the CPU, combining the static per-instance TLASInstanceData table with the per-frame UI flags
// in g_Const.

#pragma pack_matrix(row_major)

#include <donut/shaders/bindless.h> // InstanceData
#include "GPUTLASCullData.h"

// Raw nvrhi::rt::InstanceDesc layout (== D3D12_RAYTRACING_INSTANCE_DESC / VkAccelerationStructureInstanceKHR),
// 64 bytes total (nvrhi.h static_asserts sizeof(InstanceDesc) == 64). HLSL has no bitfields, so
// instanceID/instanceMask and contribution/flags are packed by hand into the low/high bytes of a uint,
// matching the C++ struct's bit layout exactly (instanceID/contribution in the low 24 bits).
struct GPUTLASInstanceDesc
{
    float4 TransformRow0;
    float4 TransformRow1;
    float4 TransformRow2;
    uint   InstanceIDAndMask;    // instanceID : 24, instanceMask : 8
    uint   ContributionAndFlags; // instanceContributionToHitGroupIndex : 24, flags : 8
    uint2  BlasAddress;          // uint64_t device address, packed as (low32, high32)
};

static const uint kInstanceFlags_ForceOpaque    = 4;  // nvrhi::rt::InstanceFlags::ForceOpaque
static const uint kInstanceFlags_ForceOMM2State = 16; // nvrhi::rt::InstanceFlags::ForceOMM2State

ConstantBuffer<TLASCullConstants>      g_Const         : register(b0);
StructuredBuffer<InstanceData>         t_InstanceData  : register(t0);
StructuredBuffer<TLASInstanceData>     t_InstanceAccel : register(t1);
StructuredBuffer<TLASObjectBounds>     t_MeshBounds    : register(t2);
RWStructuredBuffer<GPUTLASInstanceDesc> u_TLASInstances : register(u0);

[numthreads(256, 1, 1)]
void main(uint dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId >= g_Const.Params.x)
        return;

    InstanceData instData = t_InstanceData[dispatchThreadId];
    TLASInstanceData accel = t_InstanceAccel[dispatchThreadId];

    const bool cullEnable = g_Const.Params.y != 0;
    // Skinned instances carry their skin prototype's bind-pose bounds (SkinnedMeshInstance copies them
    // and nothing refreshes them per frame), so the box does not contain the posed geometry and both
    // tests below would be answering the wrong question.
    const bool isSkinned = (accel.Flags & kTLASInstanceData_IsSkinned) != 0;

    bool culled = false;
    if (cullEnable && !isSkinned)
    {
        TLASObjectBounds bounds = t_MeshBounds[accel.MeshIndex];

        float3 worldCenter, worldExtent;
        culled = CullObjectBoundsAgainstFrustum(bounds.Min, bounds.Max, instData.transform, g_Const.FrustumPlanes, worldCenter, worldExtent);

        float boundRadius = length(worldExtent);
        float distToCam = length(worldCenter - g_Const.CameraPosAndCutoff.xyz) - boundRadius;
        if (distToCam > g_Const.CameraPosAndCutoff.w)
            culled = true;
    }

    const bool onlyOMMs    = (g_Const.Params.z & 1) != 0;
    const bool force2State = (g_Const.Params.z & 2) != 0;
    const bool ommEnable   = (g_Const.Params.z & 4) != 0;
    const bool forceOpaque = g_Const.Params.w != 0;

    const bool hasAttachmentOMM = accel.OmmBlasAddress != 0;
    const bool useOmmBLAS = ommEnable && hasAttachmentOMM && !forceOpaque;
    const uint64_t blasAddress = useOmmBLAS ? accel.OmmBlasAddress : accel.BlasAddress;

    uint instanceMask = (onlyOMMs && !hasAttachmentOMM) ? 0 : 1;
    if (culled)
        instanceMask = 0;

    uint flags = force2State ? kInstanceFlags_ForceOMM2State : 0;
    if (forceOpaque)
        flags |= kInstanceFlags_ForceOpaque;

    // Sample::BuildTLAS convention: instanceID == instanceContributionToHitGroupIndex ==
    // InstanceData::firstGeometryInstanceIndex (the flat sub-instance index this instance's geometries start at).
    const uint instanceIdAndContribution = instData.firstGeometryInstanceIndex;

    GPUTLASInstanceDesc outDesc;
    outDesc.TransformRow0 = instData.transform[0];
    outDesc.TransformRow1 = instData.transform[1];
    outDesc.TransformRow2 = instData.transform[2];
    outDesc.InstanceIDAndMask = (instanceIdAndContribution & 0xFFFFFF) | (instanceMask << 24);
    outDesc.ContributionAndFlags = (instanceIdAndContribution & 0xFFFFFF) | (flags << 24);
    outDesc.BlasAddress = uint2(uint(blasAddress & 0xFFFFFFFF), uint(blasAddress >> 32));

    u_TLASInstances[dispatchThreadId] = outDesc;
}
