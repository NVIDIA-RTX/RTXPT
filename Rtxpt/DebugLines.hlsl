/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __SHADER_DEBUG_HLSL__
#define __SHADER_DEBUG_HLSL__

#ifndef __cplusplus

#include "SampleConstantBuffer.h"
ConstantBuffer<SampleConstants> g_Const : register(b0);

#ifdef DRAW_LINES_SHADERS

Texture2D<float> t_Depth        : register(t0);

void main_vs(
	float3 i_pos : POSITION,
	float4 i_col : COLOR,
	out float4 o_pos : SV_Position,
	out float4 o_col : COLOR,
    out float4 projPos : TEXCOORD
)
{
	float4 worldPos = float4(i_pos, 1.0);
	float4 viewPos = mul(worldPos, g_Const.view.matWorldToView);
	projPos = mul(viewPos, g_Const.view.matViewToClip);
    o_pos = projPos;
	o_col = i_col;
}

void main_ps(
	in float4 i_pos : SV_Position,
	in float4 i_col : COLOR,
    in float4 projPos : TEXCOORD,
	out float4 o_color : SV_Target0
)
{
    uint2 upos = uint2(i_pos.xy);
    float depth = t_Depth[upos];

    bool behind = depth > (projPos.z/projPos.w*1.00001);
    //bool checkerboard = (upos.x%2)==(upos.y%2);
    //if (behind && checkerboard)
    //    discard;

	o_color = float4( lerp(i_col.rgb, i_col.rgb*0.7+0.2, behind), i_col.a * ((behind)?(0.5):(1)));
}

#endif // #ifndef DRAW_LINES_SHADERS

#ifdef UPDATE_LINES_SHADERS
RWStructuredBuffer<DebugFeedbackStruct> u_FeedbackBuffer        : register(u51);
RWStructuredBuffer<DebugLineStruct>     u_DebugLinesBuffer      : register(u52);

[numthreads(256, 1, 1)]
void AddExtraLinesCS(uint dispatchThreadID : SV_DispatchThreadID)
{
    if( dispatchThreadID != 0 )
        return;

    float3 start = float3(0,0,0);
    float3 stop = float3(0,1000,0);
    float3 col1 = float3(1,1,0);
    float3 col2 = float3(1,0,1);

    uint lineVertexCount;
    InterlockedAdd( u_FeedbackBuffer[0].lineVertexCount, 2, lineVertexCount );
	if ( (lineVertexCount + 1) < MAX_DEBUG_LINES )
	{
		u_DebugLinesBuffer[lineVertexCount].pos = float4(start, 1);
		u_DebugLinesBuffer[lineVertexCount].col = float4(col1, 1);
		u_DebugLinesBuffer[lineVertexCount + 1].pos = float4(stop, 1);
		u_DebugLinesBuffer[lineVertexCount + 1].col = float4(col2, 1);
	}
    else
        InterlockedAdd( u_FeedbackBuffer[0].lineVertexCount, -2 );
}
#endif

#endif // #ifndef __cplusplus

#endif // __SHADER_DEBUG_HLSL__