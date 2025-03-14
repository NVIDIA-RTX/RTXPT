/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __ENVMAP_IMPORTANCE_SAMPLING_BAKER_HLSL__
#define __ENVMAP_IMPORTANCE_SAMPLING_BAKER_HLSL__

#define EMISB_NUM_COMPUTE_THREADS_PER_DIM       16
#define EMISB_IMPORTANCE_MAP_DIM                512
#define EMISB_IMPORTANCE_SAMPLES_PER_PIXEL      32          // should have integer square root

struct EnvMapImportanceSamplingBakerConstants
{
    // source info
	uint                SourceCubeDim;
    uint                SourceCubeMIPCount;
    uint                SampleIndex;                 // used for RTG seed for pre-sampling; could be a frame index (unless there's multiple per frame)
    uint                Padding1;

    // importance map (Equal Area Octahedral Projection)
    uint2               ImportanceMapDim;           // Resolution of the importance map in texels.
    uint2               ImportanceMapDimInSamples;  // Resolution of the importance map in samples.

	uint2               ImportanceMapNumSamples;    // Per-texel subsamples s.xy at finest mip.
	float               ImportanceMapInvSamples;    // 1 / (s.x*s.y).
    uint                ImportanceMapBaseMip;
};

#if !defined(__cplusplus)

#define EMIS_ENABLE_CORE_ONLY 1
#include "../../PathTracer/Utils/Math/MathHelpers.hlsli"
#include "../../PathTracer/Lighting/EnvMap.hlsli"

#include "../../PathTracer/SampleGenerators.hlsli"
#include "../../PathTracer/Utils/Utils.hlsli"

ConstantBuffer<EnvMapImportanceSamplingBakerConstants>  g_BuilderConsts  : register(b0);

SamplerState                            s_PointClamp                : register(s0);
SamplerState                            s_LinearWrap                : register(s1);

TextureCube<float4>                     t_EnvMapCube                : register(t0);
RWTexture2D<float>                      u_ImportanceMap             : register(u0);

Texture2D<float>                        t_ImportanceMap             : register(t1);
RWBuffer<uint2>                         u_PresampledBuffer          : register(u0);

[numthreads(EMISB_NUM_COMPUTE_THREADS_PER_DIM, EMISB_NUM_COMPUTE_THREADS_PER_DIM, 1)]
void BuildMIPDescentImportanceMapCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
   uint2 outputDim          = g_BuilderConsts.ImportanceMapDim;
   uint2 outputDimInSamples = g_BuilderConsts.ImportanceMapDimInSamples;  
   uint2 numSamples         = g_BuilderConsts.ImportanceMapNumSamples;         
   float invSamples         = g_BuilderConsts.ImportanceMapInvSamples;         

    uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= outputDim)) return;

    float L = 0.f;
    for (uint y = 0; y < numSamples.y; y++)
    {
        for (uint x = 0; x < numSamples.x; x++)
        {
            // Compute sample pos p in [0,1)^2 in octahedral map.
            uint2 samplePos = pixel * numSamples + uint2(x, y);
            float2 p = ((float2)samplePos + 0.5f) / outputDimInSamples;

            // Convert p to environment map space direction (note: additional rotation matrix exists that can be used to rotate envmap before sampling; it is ignored here - importance sampling works in environment map space)
            float3 dir = oct_to_ndir_equal_area_unorm(p);
            float3 radiance = t_EnvMapCube.SampleLevel(s_LinearWrap, dir, 0).rgb;
            
            L += Luminance(radiance);
        }
    }

    // Store average radiance for this texel.
    u_ImportanceMap[pixel] = L * invSamples;
}

#if 0

// Each frame, this will generate 1 environment sample per thread for a total of ENVMAP_PRESAMPLED_COUNT samples
// These samples are then used by the path tracer for a much faster sampling.
[numthreads(256, 1, 1)]
void PreSampleCS(uint dispatchThreadID : SV_DispatchThreadID)
{
    EnvMapSceneParams envParams;
    envParams.Transform     = float3x4( float4( 1, 0, 0, 0 ), float4( 0, 1, 0, 0 ), float4( 0, 0, 1, 0 ) );
    envParams.InvTransform  = float3x4( float4( 1, 0, 0, 0 ), float4( 0, 1, 0, 0 ), float4( 0, 0, 1, 0 ) );
    envParams.ColorMultiplier = float3( 1, 1, 1 );
    envParams.Enabled       = true;

    EnvMapImportanceSamplingParams isParams;
    isParams.ImportanceInvDim = 1.0.xx / g_BuilderConsts.ImportanceMapDim.xy;
    isParams.ImportanceBaseMip = g_BuilderConsts.ImportanceMapBaseMip;
    isParams.padding0 = 0;

    EnvMapSampler envMapSampler = EnvMapSampler::make( s_PointClamp, t_ImportanceMap, isParams, t_EnvMapCube, s_LinearWrap, envParams );

    SampleGenerator sampleGenerator = SampleGenerator::make( SampleGeneratorVertexBase::make( uint2(0,0), 0, g_BuilderConsts.SampleIndex * ENVMAP_PRESAMPLED_COUNT + dispatchThreadID ) );

    // StatelessUniformSampleGenerator sampleGenerator = StatelessUniformSampleGenerator::make( uint2(0,0), 0, g_BuilderConsts.SampleIndex * ENVMAP_PRESAMPLED_COUNT + dispatchThreadID );
    // ^ switch to this after testing done - it's slightly faster
    
    float2 rnd = sampleNext2D(sampleGenerator);
    
    DistantLightSample result = envMapSampler.MIPDescentSample(rnd);
    
    // More precise variant would be to store final sampling uv in equal-area-octahedral space; that requires extending EnvMapSampler interface
    // but also using oct_to_ndir_equal_area_unorm and transform to world space (envmap can have rotation) on the unpacking side. The 
    // performance cost of this is noticeable especially with multiple samples and I did not test quality difference.
    u_PresampledBuffer[dispatchThreadID] = uint2( PackTwoFp32ToFp16(result.Dir.x, result.Dir.y), PackTwoFp32ToFp16(result.Dir.z, result.Pdf) );
}

#endif


#endif // #if !defined(__cplusplus)

#endif // __ENVMAP_COMPOSER_HLSL__