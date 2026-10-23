/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

// SHaRC constants live in the main sample constant buffer (g_Const.ptConsts.sharc).
// ShaderResourceBindings.hlsli declares g_Const; the other globals it declares are unused here.
// SharcBindings.hlsli pulls in the SharcCommon SDK types + SHaRC UAV buffers (u90-u92).
#include "../../Bindings/ShaderResourceBindings.hlsli"
#include "../../Bindings/SharcBindings.hlsli"

#define LINEAR_BLOCK_SIZE   256

[numthreads(LINEAR_BLOCK_SIZE, 1, 1)]
void main(in uint2 did : SV_DispatchThreadID)
{
    SharcParameters sharcParameters = GetSharcParameters(g_Const.ptConsts.sharc);

    SharcResolveParameters resolveParameters = (SharcResolveParameters)0;
    resolveParameters.accumulationFrameNum = g_Const.ptConsts.sharc.AccumulationFrameNum;
    resolveParameters.staleFrameNumMax = g_Const.ptConsts.sharc.StaleFrameNum;
    resolveParameters.cameraPositionPrev = g_Const.ptConsts.sharc.CameraPositionPrev.xyz;

    SharcResolveEntry(did.x, sharcParameters, resolveParameters);
}
