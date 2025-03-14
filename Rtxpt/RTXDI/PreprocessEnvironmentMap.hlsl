/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "ShaderParameters.h"
#include <donut/shaders/binding_helpers.hlsli>
#include "../../External/Rtxdi/Include/Rtxdi/Utils/Math.hlsli"
#include "../PathTracer/Utils/Math/MathHelpers.hlsli"
#include "../PathTracer/Utils/Utils.hlsli"

RWTexture2D<float> u_IntegratedMips[] : register(u0);

VK_PUSH_CONSTANT ConstantBuffer<PreprocessEnvironmentMapConstants> g_Const : register(b0);

#if INPUT_ENVIRONMENT_MAP

// Note: we added a layer that converts cubemap into env
#define ENVIRONMENT_MAP_IS_CUBEMAP 1


#if ENVIRONMENT_MAP_IS_CUBEMAP
TextureCube<float4> t_EnvironmentMap : register(t0);
SamplerState s_LinearSampler : register(s0);
#else
Texture2D<float4> t_EnvironmentMap : register(t0);
#endif

float getPixelWeight(uint2 position)
{
#if ENVIRONMENT_MAP_IS_CUBEMAP
    // Note: we added a simple adapter here that converts the input cubemap into equirectangular environment map used for importance
    // sampling; What we possibly want to do in future is to use Falcor-style MIP descent in Equal Area Octahedral Projection pdf
    // (https://www.sciencedirect.com/science/article/pii/S0898122114000133) or similar.
    float3 dir = latlong_map_to_world( float2(position+0.5)/float2(g_Const.sourceSize.xy) );
    float3 color = t_EnvironmentMap.SampleLevel(s_LinearSampler, dir, 0).rgb;
#else
    float3 color = t_EnvironmentMap[position].rgb;
#endif
    float luma = max(Luminance(color), 0);

    // Do not sample invalid colors.
    if (isinf(luma) || isnan(luma))
        return 0;
    
    // Compute the solid angle of the pixel assuming equirectangular projection.
    // We don't need the absolute value of the solid angle here, just one at the same scale as the other pixels.
    float elevation = ((float(position.y) + 0.5) / float(g_Const.sourceSize.y) - 0.5) * K_PI;
    float relativeSolidAngle = cos(elevation);

    const float maxWeight = 65504.0; // maximum value that can be encoded in a float16 texture

    return clamp(luma * relativeSolidAngle, 0, maxWeight);
}
#endif

groupshared float s_weights[16];

// Warning: do not change the group size. The algorithm is hardcoded to process 16x16 tiles.
[numthreads(256, 1, 1)]
void main(uint2 GroupIndex : SV_GroupID, uint ThreadIndex : SV_GroupThreadID)
{
    uint2 LocalIndex = RTXDI_LinearIndexToZCurve(ThreadIndex);
    uint2 GlobalIndex = (GroupIndex * 16) + LocalIndex;

    // Step 0: Load a 2x2 quad of pixels from the source texture or the source mip level.
    float4 sourceWeights;
#if INPUT_ENVIRONMENT_MAP
    if (g_Const.sourceMipLevel == 0)
    {
        uint2 sourcePos = GlobalIndex.xy * 2;

        sourceWeights.x = getPixelWeight(sourcePos + int2(0, 0));
        sourceWeights.y = getPixelWeight(sourcePos + int2(0, 1));
        sourceWeights.z = getPixelWeight(sourcePos + int2(1, 0));
        sourceWeights.w = getPixelWeight(sourcePos + int2(1, 1));

        RWTexture2D<float> dest = u_IntegratedMips[0];
        dest[sourcePos + int2(0, 0)] = sourceWeights.x;
        dest[sourcePos + int2(0, 1)] = sourceWeights.y;
        dest[sourcePos + int2(1, 0)] = sourceWeights.z;
        dest[sourcePos + int2(1, 1)] = sourceWeights.w;
    }
    else
#endif
    {
        uint2 sourcePos = GlobalIndex.xy * 2;

        RWTexture2D<float> src = u_IntegratedMips[g_Const.sourceMipLevel];
        sourceWeights.x = src[sourcePos + int2(0, 0)];
        sourceWeights.y = src[sourcePos + int2(0, 1)];
        sourceWeights.z = src[sourcePos + int2(1, 0)];
        sourceWeights.w = src[sourcePos + int2(1, 1)];
    }

    uint mipLevelsToWrite = g_Const.numDestMipLevels - g_Const.sourceMipLevel - 1;
    if (mipLevelsToWrite < 1) return;

    // Average those weights and write out the first mip.
    float weight = (sourceWeights.x + sourceWeights.y + sourceWeights.z + sourceWeights.w) * 0.25;

    u_IntegratedMips[g_Const.sourceMipLevel + 1][GlobalIndex.xy] = weight;

    if (mipLevelsToWrite < 2) return;

    // The following sequence is an optimized hierarchical downsampling algorithm using wave ops.
    // It assumes that the wave size is at least 16 lanes, which is true for both NV and AMD GPUs.
    // It also assumes that the threads are laid out in the group using the Z-curve pattern.

    // Step 1: Average 2x2 groups of pixels.
    uint lane = WaveGetLaneIndex();
    weight = (weight 
        + WaveReadLaneAt(weight, lane + 1)
        + WaveReadLaneAt(weight, lane + 2)
        + WaveReadLaneAt(weight, lane + 3)) * 0.25;

    if ((lane & 3) == 0)
    {
        u_IntegratedMips[g_Const.sourceMipLevel + 2][GlobalIndex.xy >> 1] = weight;
    }

    if (mipLevelsToWrite < 3) return;

    // Step 2: Average the previous results from 2 pixels away.
    weight = (weight 
        + WaveReadLaneAt(weight, lane + 4)
        + WaveReadLaneAt(weight, lane + 8)
        + WaveReadLaneAt(weight, lane + 12)) * 0.25;

    if ((lane & 15) == 0)
    {
        u_IntegratedMips[g_Const.sourceMipLevel + 3][GlobalIndex.xy >> 2] = weight;

        // Store the intermediate result into shared memory.
        s_weights[ThreadIndex >> 4] = weight;
    }

    if (mipLevelsToWrite < 4) return;

    GroupMemoryBarrierWithGroupSync();

    // The rest operates on a 4x4 group of values for the entire thread group
    if (ThreadIndex >= 16)
        return;

    // Load the intermediate results
    weight = s_weights[ThreadIndex];

    // Change the output texture addressing because we'll be only writing a 2x2 block of pixels
    GlobalIndex = (GroupIndex * 2) + (LocalIndex >> 1);

    // Step 3: Average the previous results from adjacent threads, meaning from 4 pixels away.
    weight = (weight 
        + WaveReadLaneAt(weight, lane + 1)
        + WaveReadLaneAt(weight, lane + 2)
        + WaveReadLaneAt(weight, lane + 3)) * 0.25;

    if ((lane & 3) == 0)
    {
        u_IntegratedMips[g_Const.sourceMipLevel + 4][GlobalIndex.xy] = weight;
    }

    if (mipLevelsToWrite < 5) return;

    // Step 4: Average the previous results from 8 pixels away.
    weight = (weight 
        + WaveReadLaneAt(weight, lane + 4)
        + WaveReadLaneAt(weight, lane + 8)
        + WaveReadLaneAt(weight, lane + 12)) * 0.25;

    if (lane == 0)
    {
        u_IntegratedMips[g_Const.sourceMipLevel + 5][GlobalIndex.xy >> 1] = weight;
    }
}