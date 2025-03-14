/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __SAMPLE_GENERATORS_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __SAMPLE_GENERATORS_HLSLI__

#include "Config.h"

enum class SampleGeneratorEffectSeed : uint32_t
{
    Base                            = 0,        // this is the default state that the SampleGenerator starts from
    ScatterBSDF                     = 1,
    NextEventEstimation             = 2,
    NextEventEstimationLightSampler = 3,
    NextEventEstimationFeedback     = 4,
    RussianRoulette                 = 5,
};

// performance optimization for LD sampling - will stop using LD sampling after selected diffuse bounces (1 to stop after first diffuse, 2 after 2nd, etc.)
#define DisableLowDiscrepancySamplingAfterDiffuseBounceCount    1

// Note, compared to Falcor we've completely switched to 'stateless' random/sample generators to avoid storing additional per-path data.
// That means that each sample generator is seeded not only by pixel position and sample index but also 'vertexIndex', which increases
// chances of overlap.

#include "StatelessSampleGenerators.hlsli"
#if 0   // fast uniform
#define SampleGenerator UniformSampleSequenceGenerator
#elif 1 // low discrepancy
#define SampleGenerator SampleSequenceGenerator
#else   // slow uniform with good distribution for many-sample (1mil+) references
#define SampleGenerator HQUniformSampleSequenceGenerator
#endif
    
// Convenience functions for generating 1D/2D/3D values in the range [0,1) using a sample generator template.
template<typename SampleGeneratorType>
float sampleNext1D( inout SampleGeneratorType sampleGenerator )
{
    uint bits = sampleGenerator.Next();
    // a.) converting the upper 24bits to [0, 1) because the higher bits have better distribution in some hash algorithms (like sobol)
    // b.) this is a good way to guarantee [0, 1) since float32 mantissa is only 23 bits
    return (bits>>8) / float(1 << 24); // same as '/ 16777216.0'        // in theory "(bits >> 6) / float( 1 << 26 )" guarantees < 1 as well on the CPU fp math side, but not sure on GPU
}

template<typename SampleGeneratorType>
float2 sampleNext2D( inout SampleGeneratorType sampleGenerator )
{
    float2 sample;
    // Not using float4 initializer to ensure consistent order of evaluation.
    sample.x = sampleNext1D(sampleGenerator);
    sample.y = sampleNext1D(sampleGenerator);
    return sample;
}

template<typename SampleGeneratorType>
float3 sampleNext3D( inout SampleGeneratorType sampleGenerator )
{
    float3 sample;
    // Not using float4 initializer to ensure consistent order of evaluation.
    sample.x = sampleNext1D(sampleGenerator);
    sample.y = sampleNext1D(sampleGenerator);
    sample.z = sampleNext1D(sampleGenerator);
    return sample;
}

template<typename SampleGeneratorType>
float4 sampleNext4D( inout SampleGeneratorType sampleGenerator )
{
    float4 sample;
    // Not using float4 initializer to ensure consistent order of evaluation.
    sample.x = sampleNext1D(sampleGenerator);
    sample.y = sampleNext1D(sampleGenerator);
    sample.z = sampleNext1D(sampleGenerator);
    sample.w = sampleNext1D(sampleGenerator);
    return sample;
}

// Stochastic texture filtering white noise sampling functions.
float STSampleUniform(inout uint hash)
{
    // Use upper 24 bits and divide by 2^24 to get a number u in [0,1).
    // In floating-point precision this also ensures that 1.0-u != 0.0.
    hash = Hash32(hash);
    return (hash >> 8) / float(1 << 24);
}

float STWNWhiteNoise1D(uint2 screenCoord, uint frameIndex)
{
    uint hash = Hash32Combine(Hash32(frameIndex + 0x035F9F29), (screenCoord.x << 16) | screenCoord.y);
    return STSampleUniform(hash);
}

float2 STWNWhiteNoise2D(uint2 screenCoord, uint frameIndex)
{
    uint hash = Hash32Combine(Hash32(frameIndex + 0x035F9F29), (screenCoord.x << 16) | screenCoord.y);
    return float2(STSampleUniform(hash), STSampleUniform(hash));
}

float3 STWNWhiteNoise3D(uint2 screenCoord, uint frameIndex)
{
    uint hash = Hash32Combine(Hash32(frameIndex + 0x035F9F29), (screenCoord.x << 16) | screenCoord.y);
    return float3(STSampleUniform(hash), STSampleUniform(hash), STSampleUniform(hash));
}

// Stochastic texture filtering blue noise sampling functions.
float STBNBlueNoise1D(uint2 screenCoord, uint frameIndex, Texture2D spatioTemporalBlueNoiseTex)
{
    uint3 WrappedCoordinate = uint3(screenCoord % 128, frameIndex % 64);
    uint3 TextureCoordinate = uint3(WrappedCoordinate.x, WrappedCoordinate.z * 128 + WrappedCoordinate.y, 0);
    return spatioTemporalBlueNoiseTex.Load(TextureCoordinate, 0).x;
}

float2 STBNBlueNoise2D(uint2 screenCoord, uint frameIndex, Texture2D spatioTemporalBlueNoiseTex)
{
    uint3 WrappedCoordinate = uint3(screenCoord % 128, frameIndex % 64);
    uint3 TextureCoordinate = uint3(WrappedCoordinate.x, WrappedCoordinate.z * 128 + WrappedCoordinate.y, 0);
    return spatioTemporalBlueNoiseTex.Load(TextureCoordinate, 0).xy;
}

float3 SpatioTemporalBlueNoise1DWhiteNoise2D(float2 pixel, uint frameIndex, Texture2D spatioTemporalBlueNoiseTex)
{
    float3 u;
    u.x = STBNBlueNoise1D(uint2(pixel.xy), frameIndex, spatioTemporalBlueNoiseTex);
    u.yz = STWNWhiteNoise2D(uint2(pixel.xy), frameIndex);
    return u;
}

float3 SpatioTemporalBlueNoise2DWhiteNoise1D(float2 pixel, uint frameIndex, Texture2D spatioTemporalBlueNoiseTex)
{
    float3 u;
    u.xy = STBNBlueNoise2D(uint2(pixel.xy), frameIndex, spatioTemporalBlueNoiseTex);
    u.z  = STWNWhiteNoise1D(uint2(pixel.xy), frameIndex);
    return u;
}

float4 SpatioTemporalBlueNoise2DWhiteNoise2D(float2 pixel, uint frameIndex, Texture2D spatioTemporalBlueNoiseTex)
{
    float4 u;
    u.xy = STBNBlueNoise2D(uint2(pixel.xy), frameIndex, spatioTemporalBlueNoiseTex);
    u.zw  = STWNWhiteNoise2D(uint2(pixel.xy), frameIndex);
    return u;
}

float3 SpatioTemporalWhiteNoise3D(float2 pixel, uint frameIndex)
{
    float3 u;
    u = STWNWhiteNoise3D(uint2(pixel.xy), frameIndex);
    return u;
}

#endif // __SAMPLE_GENERATORS_HLSLI__
