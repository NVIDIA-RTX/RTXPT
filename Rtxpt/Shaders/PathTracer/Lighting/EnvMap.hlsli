/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __ENV_MAP_HLSLI__
#define __ENV_MAP_HLSLI__

#if !defined(__cplusplus)
#pragma pack_matrix(row_major)
#endif

#include "LightingTypes.h"
#include "../Utils/Geometry.hlsli"

// Core envmap access
struct EnvMap
{
    TextureCube<float4> Texture;            ///< Environment map texture.
    SamplerState        TextureSampler;     ///< Environment map sampler (linear, wrap).
    EnvMapSceneParams   SceneParams;        ///< Environment map scene parameters.
    
    static EnvMap make( TextureCube<float4>     texture         ///< Environment map texture.
                        , SamplerState          textureSampler  ///< Environment map texture sampler.
                        , EnvMapSceneParams     sceneParams     ///< Environment map data.
                       )
    {
        EnvMap envMap;
        envMap.Texture = texture;
        envMap.TextureSampler = textureSampler;
        envMap.SceneParams = sceneParams;
        return envMap;
    }

   // Transform direction from local to world space.
    float3 ToWorld(float3 dir)
    {
        return mul(dir, (float3x3)SceneParams.Transform);
    }

    // Transform direction from world to local space.
    float3 ToLocal(float3 dir)
    {
        return mul(dir, (float3x3)SceneParams.InvTransform);
    } 

    float3 EvalLocal(float3 localDir, float lod = 0.f)
    {
        return Texture.SampleLevel(TextureSampler, localDir, lod).rgb * SceneParams.ColorMultiplier.rgb;
    }

    float3 Eval(float3 worldDir, float lod = 0.f)
    {
        return EvalLocal(ToLocal(worldDir), lod);
    }
};

// Runtime envmap sampler & importance sampler
struct EnvMapSampler
{
    EnvMap              EnvironmentMap;

    // MIP descent sampler
    SamplerState        PointClampSampler;
    Texture2D<float>    ImportanceMap;          ///< Hierarchical importance map (entire mip chain).
    
    EnvMapImportanceSamplingParams
                        ImportanceSamplingParams;
    
// #ifndef EMIS_ENABLE_CORE_ONLY
//     Buffer<uint2>       PresampledBuffer;       ///< ENVMAP_PRESAMPLED_COUNT number of (encoded) samples pre-sampled for each frame or PT pass
// #endif 
    
    static EnvMapSampler make(
          SamplerState          pointClampSampler                   ///< Point sampling with clamp to edge.
        , Texture2D<float>      importanceMap                       ///< Hierarchical importance map (entire mip chain).
        , EnvMapImportanceSamplingParams importanceSamplingParams   ///< Data needed by importance sampling

        , TextureCube<float4>   environmentMap                      ///< Environment map texture.
        , SamplerState          environmentMapTextureSampler        ///< Environment map texture sampler.
        , EnvMapSceneParams     envMapSceneParams                   ///< Environment map data.        
    
// #ifndef EMIS_ENABLE_CORE_ONLY
//         , Buffer<uint2>         presampledBuffer
// #endif
        ) 
    {
        EnvMapSampler envMapSampler;

        envMapSampler.EnvironmentMap        = EnvMap::make( environmentMap, environmentMapTextureSampler, envMapSceneParams );
            
        envMapSampler.PointClampSampler     = pointClampSampler;
        envMapSampler.ImportanceMap         = importanceMap;    
        envMapSampler.ImportanceSamplingParams = importanceSamplingParams; 
        
// #ifndef EMIS_ENABLE_CORE_ONLY
//         envMapSampler.PresampledBuffer      = presampledBuffer;
// #endif
        
        return envMapSampler;
    }
    
    // Transform direction from local to world space.
    float3 ToWorld(float3 dir)
    {
        return EnvironmentMap.ToWorld(dir);
    }

    // Transform direction from world to local space.
    float3 ToLocal(float3 dir)
    {
        return EnvironmentMap.ToLocal(dir);
    } 

    float3 Eval(float3 worldDir, float lod = 0.f)
    {
        return EnvironmentMap.Eval(worldDir, lod);
    }
    
    DistantLightSample UniformSample(const float2 rnd)
    {
        DistantLightSample result;
        float3 dir = SampleSphereUniform(rnd);
        result.Le = EnvironmentMap.EvalLocal(dir);
        result.Dir = ToWorld(dir);
        result.Pdf = SampleSphereUniformPDF();
        return result;
    }
    float UniformEvalPdf(const float3 worldDir)
    {
        return SampleSphereUniformPDF();
    }

    // Importance sampling of the environment map using the MIP descent approach
    DistantLightSample MIPDescentSample(const float2 rnd)
    {
        DistantLightSample result;
        
        float2 p = rnd;     // Random sample in [0,1)^2.
        uint2 pos = 0;      // Top-left texel pos of current 2x2 region.

        // Iterate over mips of 2x2...NxN resolution.
        for (int mip = ImportanceSamplingParams.ImportanceBaseMip - 1; mip >= 0; mip--)
        {
            // Scale position to current mip.
            pos *= 2;

            // Load the four texels at the current position.
            float w[4];
            w[0] = ImportanceMap.Load(int3(pos, mip));
            w[1] = ImportanceMap.Load(int3(pos + uint2(1, 0), mip));
            w[2] = ImportanceMap.Load(int3(pos + uint2(0, 1), mip));
            w[3] = ImportanceMap.Load(int3(pos + uint2(1, 1), mip));

            float q[2];
            q[0] = w[0] + w[2];
            q[1] = w[1] + w[3];

            uint2 off;

            // Horizontal warp.
            float d = saturate(q[0] / (q[0] + q[1]));   // saturate is to guard against div-by-zero. In case both probabilities are 0, d will be 1 by convention (doesn't really matter either way).

            if (p.x < d) // left
            {
                off.x = 0;
                p.x = p.x / d;
            }
            else // right
            {
                off.x = 1;
                p.x = (p.x - d) / (1.f - d);
            }

            // Vertical warp.
            // Avoid stack allocation by not using dynamic indexing.
            // float e = w[off.x] / q[off.x];
            float e = off.x == 0 ? (w[0] / q[0]) : (w[1] / q[1]);

            if (p.y < e) // bottom
            {
                off.y = 0;
                p.y = p.y / e;
            }
            else // top
            {
                off.y = 1;
                p.y = (p.y - e) / (1.f - e);
            }

            pos += off;
        }

        // At this point, we have chosen a texel 'pos' in the range [0,dimension) for each component.
        // The 2D sample point 'p' has been warped along the way, and is in the range [0,1) representing sub-texel location.

#if 0 // if correctness testing vs envMapEvalPdf - due to differences in UV math samples on the border can be different; this is rare and is fine/acceptable but use this to temporarily ensure it's only that
        float eps = 2e-4f; // // eps found empirically
        p = p * (1.0-2.0*eps).xx+eps.xx;
#endif

        // Compute final sample position and map to direction.
        float2 uv = ((float2)pos + p) * ImportanceSamplingParams.ImportanceInvDim;     // Final sample in [0,1)^2.
        float3 dir = oct_to_ndir_equal_area_unorm(uv);

        // Compute final pdf.
        // We sample exactly according to the intensity of where the final samples lies in the octahedral map, normalized to its average intensity.
        float avg_w = ImportanceMap.Load(int3(0, 0, ImportanceSamplingParams.ImportanceBaseMip)); // 1x1 mip holds integral over importance map. TODO: Replace by constant or rescale in setup so that the integral is 1.0
        float pdf = ImportanceMap[pos] / avg_w;

        result.Le = EnvironmentMap.EvalLocal(dir);
        result.Dir = ToWorld(dir);
        result.Pdf = pdf / (4.0 * K_PI);

        return result;
    }

    // Evaluates the probability density function for a specific direction when using MIPDescentSample importance sampling.
    //    Note that the sample() function already returns the pdf for the sampled location.
    //    But, in some cases we need to evaluate the pdf for other directions (e.g. for MIS).
    float MIPDescentEvalPdf(const float3 worldDir)
    {
        float2 uv = ndir_to_oct_equal_area_unorm(ToLocal(worldDir));
        float avg_w = ImportanceMap.Load(int3(0, 0, ImportanceSamplingParams.ImportanceBaseMip)); // 1x1 mip holds integral over importance map. TODO: Replace by constant or rescale in setup so that the integral is 1.0
        float pdf = ImportanceMap.SampleLevel(PointClampSampler, uv, 0) / avg_w;
        return pdf / (4.0 * K_PI);
    }

};



#endif // #define __ENV_MAP_HLSLI__