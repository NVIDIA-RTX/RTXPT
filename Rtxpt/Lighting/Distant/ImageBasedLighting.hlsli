#pragma once


// IBL with split-sum approximation, local cubemap, and SSR
float3 EvaluateIBL(uint2 pixelPos, float3 normal, float3 viewDir, float2 screenUV, float3 baseColor, float metallic, float roughness, float ao)
{
    // For metals, F0 is the albedo color; for dielectrics, it's typically 0.04
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);
    float3 reflDir = reflect(-viewDir, normal);
    float NdotV = saturate(dot(normal, viewDir));
    // === DIFFUSE IBL ===
    // Sample diffuse irradiance from local SH-based cubemap and global envmap
    float3 irradiance = t_DiffuseIrradianceCube.SampleLevel(s_EnvironmentMapSampler, normal, 0).rgb;
    
    // Cheap energy conservation
    float3 kD = (1.0 - F0) * (1.0 - metallic);
    float3 diffuse = kD * baseColor * irradiance;
    
    // === SPECULAR IBL (Split-Sum Approximation) ===
    // 1. Pre-filtered environment (GGX mip chain)
    float ggxMip = roughness * 7;
    float3 prefilteredColor = t_LocalCubemapGGX.SampleLevel(s_EnvironmentMapSampler, reflDir, ggxMip).rgb * ao;
    
    // 2. SSR overlay (highest priority for nearby reflections)
    float ssrMip = roughness * 10;
    float4 ssrResult = t_SSRBlurChain.SampleLevel(s_EnvironmentMapSampler, screenUV, ssrMip);
    //prefilteredColor = lerp(prefilteredColor, ssrResult.rgb, ssrResult.a);
    
    // 3. BRDF LUT lookup (split-sum second term)
    float2 brdfLUT = t_BRDFLUT.SampleLevel(s_EnvironmentMapSampler, float2(NdotV, roughness), 0).rg;
    float3 specular = prefilteredColor * (F0 * brdfLUT.x + brdfLUT.y);
    
    // Multi-bounce AO approximation (Jimenez et al., Eq. 12)
    // Recovers energy lost from ignoring near-field interreflections,
    // avoiding the typical over-darkening at corners and crevices
    float3 diffuseAlbedo = kD * baseColor;
    float3 multiBounceAO = ao / (1.0 - saturate(diffuseAlbedo) * (1.0 - ao));

    return diffuse * multiBounceAO + specular;
}