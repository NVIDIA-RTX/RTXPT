/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Debug visualization: displays u_RTGIOutput before primary-surface albedo is applied.
// For diffuse surfaces this estimates irradiance / pi, including both hits and environment misses.
// Replaces the deferred lighting pass when "Debug RTGI" is enabled; post-processing still runs.

#define IGNORE_RAY_PAYLOAD 1

#include "../Bindings/ShaderResourceBindings.hlsli"
#include "../Bindings/GBufferBindings.hlsli"
#include "IntroBridge.hlsli"

[numthreads(NUM_COMPUTE_THREADS_PER_DIM, NUM_COMPUTE_THREADS_PER_DIM, 1)]
void main(uint2 dispatchThreadID : SV_DispatchThreadID)
{
    const uint2 pixelPos = dispatchThreadID.xy;
    if (any(pixelPos >= uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight)))
        return;

    u_OutputColor[pixelPos] = float4(u_RTGIOutput[pixelPos].rgb, 1.0);
}
