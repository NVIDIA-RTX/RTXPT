/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// SHaRC Resolve pass: combines new cache data with previous frame data.
// One thread per SHaRC element, dispatched over the entire buffer capacity.

#define IGNORE_RAY_PAYLOAD 1

#include "../Bindings/ShaderResourceBindings.hlsli"

#include "IntroSampleCommon.hlsli"

RWStructuredBuffer<uint64_t>                u_SHaRCHashEntries   : register(u90);
RWStructuredBuffer<SharcAccumulationData>   u_SHaRCAccumulation  : register(u91);
RWStructuredBuffer<SharcPackedData>         u_SHaRCResolved      : register(u92);

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint entryIndex = dispatchThreadID.x;
    uint capacity = g_MiniConst.params.x;

    if (entryIndex >= capacity)
        return;

    // Resolve must use the same grid scale and encoding as update and query.
    SharcParameters sharcParams = GetDefaultSharcParams(g_Const.ptConsts.camera.PosW);
    sharcParams.hashGridData.capacity = capacity;
    sharcParams.hashGridData.hashEntriesBuffer = u_SHaRCHashEntries;
    sharcParams.accumulationBuffer = u_SHaRCAccumulation;
    sharcParams.resolvedBuffer = u_SHaRCResolved;

    SharcResolveParameters resolveParams = (SharcResolveParameters)0;
    resolveParams.cameraPositionPrev = g_Const.ptConsts.prevCamera.PosW;
    resolveParams.accumulationFrameNum = 64;
    resolveParams.staleFrameNumMax = 2 * SHARC_STALE_FRAME_NUM_MIN;

    SharcResolveEntry(entryIndex, sharcParams, resolveParams);
}
