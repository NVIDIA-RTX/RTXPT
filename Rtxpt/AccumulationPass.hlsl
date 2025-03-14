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

#include "RTXDI/ShaderParameters.h"
#include <donut/shaders/binding_helpers.hlsli>

VK_PUSH_CONSTANT ConstantBuffer<AccumulationConstants> g_Const : register(b0);

RWTexture2D<float4> u_AccumulatedColor : register(u0);

Texture2D<float4> t_InputColor : register(t0);

SamplerState s_Sampler : register(s0);

[numthreads(8, 8, 1)]
void main(uint2 globalIdx : SV_DispatchThreadID)
{
    if (any(globalIdx > int2(g_Const.outputSize)))
        return;

    float4 prevColor = u_AccumulatedColor[globalIdx];

    float4 compositedColor;
    if (all(g_Const.inputSize == g_Const.outputSize))
    {
        compositedColor = t_InputColor[globalIdx];
    }
    else
    {
        float2 inputPos = (float2(globalIdx) + 0.5) * (g_Const.inputSize / g_Const.outputSize) + g_Const.pixelOffset;
        float2 inputUV = inputPos * g_Const.inputTextureSizeInv;
        
        compositedColor = t_InputColor.SampleLevel(s_Sampler, inputUV, 0);
    }
    
    float4 outputColor;
    if (g_Const.blendFactor < 1)
        outputColor = lerp(prevColor, compositedColor, g_Const.blendFactor);
    else
        outputColor = compositedColor;

    u_AccumulatedColor[globalIdx] = outputColor;
}
