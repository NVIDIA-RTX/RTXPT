/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Unified deferred lighting combine pass. Two compile variants selected via DIRECT_SPECULAR_READ:
//   - default:              specular overlay sampled from the SSR blur chain (t_SSRBlurChain)
//   - DIRECT_SPECULAR_READ: fully-shaded specular read directly from the u_SSRResult UAV (no blur, no
//                           split-sum) - used by both MIS indirect and RT reflections; DLSS-RR denoises
// Runtime flags in g_MiniConst.params.x:
//   - bit 0 set: ray-traced GI from u_RTGIOutput; clear: environment SH irradiance cube
//   - bit 1 set: output only the weighted indirect specular contribution

#define IGNORE_RAY_PAYLOAD 1

#include "../Bindings/ShaderResourceBindings.hlsli"
#include "../Bindings/GBufferBindings.hlsli"
#include "../Bindings/SamplerBindings.hlsli"
#include "../PathTracer/BSDFs/BxDF.hlsli"
#include "IntroBridge.hlsli"
#include "IntroCommon.hlsli"

// IBL with split-sum approximation, prefiltered environment, and SSR
float3 CombineIndirectLight(uint2 pixelPos, float3 viewDir, float2 screenUV, PackedGBufferSurface surface, bool specularOnly)
{
    // Load material properties from GBuffer
    float3 baseColor = surface.GetBaseColor();
    float metallic = surface.GetMetalnes();
    float3 normal = surface.GetSpecNormal();
    float roughness = surface.GetRoughness();
    const float ao = surface.GetMaterialAO(); // combined material + screen-space AO, baked into the GBuffer by the AO pass
    
    // For metals, F0 is the albedo color; for dielectrics, it's typically 0.04
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);
    float3 reflDir = reflect(-viewDir, normal);
    float NdotV = saturate(dot(normal, viewDir));
    float3 diffuseAlbedo = baseColor * (1.0 - metallic);

    // === DIFFUSE IBL ===
    // Diffuse irradiance source is selected at runtime: ray-traced GI when RTGI is
    // active, otherwise the SH-convolved environment cube.
    float3 irradiance;
    if ((g_MiniConst.params.x & 1u) != 0) // Some form of raytraced GI enabled.
    {
        irradiance = u_RTGIOutput[pixelPos].xyz;
    }
    else if (g_MiniConst.params.y != 0) // ReSTIR DI samples the environment as (shadowed) direct light; the
    {                                    // unshadowed irradiance IBL below would double-count that env diffuse.
        irradiance = 0;
    }
    else // Use prefiltered environment irradiance as a fallback.
    {
        irradiance = t_EnvIrradianceCube.SampleLevel(s_EnvironmentMapSampler, normal, 0).rgb;
        // Irradiance fallback doesn't support built in transmission/refraction for glass
        if (surface.GetShaderId() == ShaderIdGlassGGX)
        {
            irradiance = 0;
        }
        float3 multiBounceAO = ao > 1e-3 ? ao / (1.0 - saturate(diffuseAlbedo) * (1.0 - ao)) : ao;
        irradiance *= multiBounceAO;
    }

    float3 diffuse = diffuseAlbedo * irradiance;

    // The BRDF LUT is needed on both path because it's used for energy conservation/preservation
    // Clamp sampler: the BRDF LUT is indexed by (NdotV, roughness) in [0,1]; wrapping folds the high-roughness
    // / grazing-angle edge back to the opposite edge and corrupts the split-sum term.
    float2 brdfLUT = t_BRDFLUT.SampleLevel(s_LinearClampSampler, float2(NdotV, roughness), 0).rg;

#if DIRECT_SPECULAR_READ
    // The specular pass (MIS indirect or RT reflections) writes fully-shaded, unblurred specular directly to
    // u_SSRResult. Read it straight from the UAV with no split-sum.
    float3 specular = u_SSRResult[pixelPos].xyz;
#else
    // SPECULAR IBL (Split-Sum Approximation)
    float ggxMip = roughness * g_MiniConst.params.z; // g_MiniConst.params.z carries the prefiltered environment max sampleable mip (set on the CPU).
    float3 prefilteredColor = t_PrefilteredEnvCube.SampleLevel(s_EnvironmentMapSampler, reflDir, ggxMip).rgb; // Pre-filtered environment (GGX mip chain).

    // Blend in screen space or ray traced contribution
    float ssrMip = roughness * g_MiniConst.params.w; // g_MiniConst.params.w carries the SSR blur-chain max mip (resolution dependent).
    float4 ssrResult = t_SSRBlurChain.SampleLevel(s_EnvironmentMapSampler, screenUV, ssrMip);
    prefilteredColor = lerp(prefilteredColor * ao, ssrResult.rgb, ssrResult.a);

    // BRDF LUT lookup (split-sum second term)
    float3 specular = prefilteredColor * (F0 * brdfLUT.x + brdfLUT.y);
#endif

    // Energy conservation
    float3 FssEss = F0 * brdfLUT.x + brdfLUT.y;
    float  Ess = brdfLUT.x + brdfLUT.y;
    float  Ems = 1.0 - Ess;
    float3 Favg = F0 + (1.0 - F0) / 21.0;
    float3 Fms =  Favg / (1.0 - Ems * Favg);
    float3 FmsEms = FssEss * Fms * Ems;
    
    specular *= (1 + Fms * Ems);
    diffuse *= (1 - FssEss + Fms * Ems);

    return specularOnly ? specular : diffuse + specular;
}

[numthreads(NUM_COMPUTE_THREADS_PER_DIM, NUM_COMPUTE_THREADS_PER_DIM, 1)]
void main( uint2 dispatchThreadID : SV_DispatchThreadID )
{
    // TODO: Morton code thread reordering for better locality
    const uint2 pixelPos = dispatchThreadID.xy;
    if( any(pixelPos >= uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight) ) )
        return;

    // Load GBuffer
    PackedGBufferSurface surface;
    surface.LoadFromRenderTargets(pixelPos);
    float depth = u_Depth[pixelPos];

    // Combine lighting
    float3 rayDir = Bridge::computeCameraRay(pixelPos).dir;
    const bool specularOnly = (g_MiniConst.params.x & 2u) != 0;

    float3 radiance = 0;
    if (depth > 0)  // Check for valid depth (NDC depth, 0 = sky/miss)
    {
        radiance = specularOnly ? 0 : surface.GetEmissive();

        // Compute screen UV for SSR sampling
        float2 screenUV = (float2(pixelPos) + 0.5) / float2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight);

        // Use IBL with SSR, prefiltered environment, and split-sum
        radiance += CombineIndirectLight(pixelPos, -rayDir, screenUV, surface, specularOnly);

        // Read pre-computed direct lighting from the direct lighting pass
        if (!specularOnly)
            radiance += u_DirectLighting[pixelPos].rgb;
    }
    else if (!specularOnly) // no-hit, sample the background unless isolating reflections
    {
        EnvMap envMap = EnvMap::make(t_EnvironmentMap, s_EnvironmentMapSampler, g_Const.envMapSceneParams);
        radiance = envMap.Eval(rayDir, 0.f);
    }

    // Output hdr color
    u_OutputColor[pixelPos] = float4(radiance, 1.0);
}
