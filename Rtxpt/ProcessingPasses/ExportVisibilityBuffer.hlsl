/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#define NON_PATH_TRACING_PASS 1

#include "../Shaders/Bindings/ShaderResourceBindings.hlsli"

#include "../Shaders/PathTracerBridgeDonut.hlsli"
#include "../Shaders/PathTracer/PathTracer.hlsli"

[numthreads(NUM_COMPUTE_THREADS_PER_DIM, NUM_COMPUTE_THREADS_PER_DIM, 1)]
void main( uint2 dispatchThreadID : SV_DispatchThreadID )
{
    const uint2 pixelPos = dispatchThreadID.xy;
    if( any(pixelPos >= uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight) ) )
        return;

    // there used to be more code here but it was moved elsewhere - still, useful for debugging

    if (g_Const.debug.debugViewType==(int)DebugViewType::VBufferMotionVectors)
        DebugPixel( pixelPos, float4( 0.5.xxx + u_MotionVectors[pixelPos].xyz * float3(0.2, 0.2, 10), 1) );
    else if (g_Const.debug.debugViewType==(int)DebugViewType::VBufferDepth)
        DebugPixel( pixelPos, float4( u_Depth[pixelPos].xxx * 100.0, 1 ) );

}