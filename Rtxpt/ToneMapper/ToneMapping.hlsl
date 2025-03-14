/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma pack_matrix(row_major)

#include "ToneMapping_cb.h"
#include "ToneMapping.ps.hlsli"

void main_ps(
    in float4 pos : SV_Position,
    in float2 uv : UV,
    out float4 o_rgba : SV_Target)
{
    //Pass to Falcor tonemapping implementation 
    o_rgba = applyToneMapping(uv);
}

RWBuffer<float>     u_CaptureTarget         : register(u0);
Texture2D<float>    t_CaptureSource         : register(t0);

[numthreads(1, 1, 1)]
void capture_cs( uint3 dispatchThreadID : SV_DispatchThreadID )
{
    uint dummy0, dummy1, mipLevels; t_CaptureSource.GetDimensions(0,dummy0,dummy1,mipLevels); 
    float avgLum = t_CaptureSource.Load( int3(0, 0, mipLevels-1) );
    u_CaptureTarget[0] = avgLum;
}