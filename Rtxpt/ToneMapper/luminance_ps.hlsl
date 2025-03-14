/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

//Copy of Luminance.ps.hlsli 
SamplerState gColorSampler : register(s1);

Texture2D gColorTex : register(t0);

float luminance(float3 color)
{
    return dot(color, float3(0.299, 0.587, 0.114));
}

//float4 main(float2 texC : TEXCOORD) : SV_TARGET0
void main(
	in float4 pos : SV_Position,
	in float2 uv : UV,
	out float4 o_rgba : SV_Target)
{
    float4 color = gColorTex.Sample(gColorSampler, uv);
    float logLuminance = log2(max(0.0001, luminance(color.xyz)));
    o_rgba = float4(logLuminance, 0, 0, 1);
}
