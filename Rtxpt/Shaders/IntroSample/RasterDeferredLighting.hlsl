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

#include "Bindings/ShaderResourceBindings.hlsli"
#include "Bindings/GBufferBindings.hlsli"
#include "Bindings/LightingBindings.hlsli"
#include "Bindings/SamplerBindings.hlsli"
#include "PathTracerBridgeDonut.hlsli"
#include "IntroCommon.hlsli"
#include "../Lighting/Distant/ImageBasedLighting.hlsli"

cbuffer ReflectionConstants : register(b10)
{
    float   g_LocalCubeWeight;          // Blend weight for local cubemap (0 = global only, 1 = local only)
    float   g_LocalCubeMaxMip;          // Max mip level of local cubemap
    float   g_SSRMaxMip;                // Max mip level of SSR blur chain
    float   g_ReflectionPadding;
};

// Reconstruct world position from pixel coordinates and NDC depth
float3 ReconstructWorldPosition(uint2 pixelPos, float depth)
{
    float2 uv = (float2(pixelPos) + 0.5) * g_Const.view.viewportSizeInv;
    float4 clipPos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    float4 worldPos = mul(clipPos, g_Const.view.matClipToWorldNoOffset);
    return worldPos.xyz / worldPos.w;
}

// Fresnel-Schlick (exact, no roughness term)
float3 FresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

// Unpack radiance from log-encoded uint16
float UnpackLightRadiance(uint logRadiance)
{
    return (logRadiance == 0) ? 0 : exp2((float(logRadiance - 1) / 65534.0) * (kPolymorphicLightMaxLog2Radiance - kPolymorphicLightMinLog2Radiance) + kPolymorphicLightMinLog2Radiance);
}

// Unpack RGB color and radiance from polymorphic light
float3 UnpackLightColor(PolymorphicLightInfo lightInfo)
{
    float3 color = Unpack_R8G8B8_UFLOAT(lightInfo.ColorTypeAndFlags);
    float radiance = UnpackLightRadiance(lightInfo.LogRadiance & 0xffff);
    return color * radiance;
}

// Decode the polymorphic light type from the packed flags
PolymorphicLightType DecodeLightType(PolymorphicLightInfo lightInfo)
{
    uint typeCode = (lightInfo.ColorTypeAndFlags >> kPolymorphicLightTypeShift) & kPolymorphicLightTypeMask;
    return (PolymorphicLightType)typeCode;
}

// Evaluate spot light cone attenuation
float EvaluateSpotAttenuation(PolymorphicLightInfo lightBase, PolymorphicLightInfoEx lightEx, float3 worldPos)
{
    bool hasShaping = (lightBase.ColorTypeAndFlags & kPolymorphicLightShapingEnableBit) != 0;
    if (!hasShaping)
        return 1.0;

    float3 spotDir = OctToNDirUnorm32(lightEx.PrimaryAxis);
    float cosConeAngle = f16tof32(lightEx.CosConeAngleAndSoftness);
    float coneSoftness = f16tof32(lightEx.CosConeAngleAndSoftness >> 16);
    float minFalloff = (lightBase.ColorTypeAndFlags & kPolymorphicLightShapingUseMinFalloff) ? kMinSpotlightFalloff : 0.0;

    float3 lightToSurface = normalize(worldPos - lightBase.Center);
    float cosTheta = dot(spotDir, lightToSurface);
    float smoothFalloff = smoothstep(cosConeAngle, cosConeAngle + coneSoftness, cosTheta);

    return max(minFalloff, smoothFalloff);
}

// Evaluate direct lighting from all analytic lights (point and spot)
float3 EvaluateDirectLighting(float3 worldPos, float3 normal, float3 viewDir, float3 baseColor, float metallic, float roughness)
{
    LightingControlData ctrl = t_LightsCB[0];
    uint analyticStart = ctrl.EnvmapQuadNodeCount;
    uint analyticEnd = analyticStart + ctrl.AnalyticLightCount;

    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);
    float NdotV = saturate(dot(normal, viewDir));

    float a = roughness * roughness;
    float a2 = a * a;

    // Schlick-GGX geometry term factor
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float G1V = NdotV / (NdotV * (1.0 - k) + k);

    float3 totalLight = 0;

    for (uint i = analyticStart; i < analyticEnd; i++)
    {
        PolymorphicLightInfo lightBase = t_Lights[i];
        PolymorphicLightInfoEx lightEx = t_LightsEx[i];

        float3 lightPos = lightBase.Center;
        float3 L = lightPos - worldPos;
        float dist2 = dot(L, L);
        float dist = sqrt(dist2);
        L /= dist;

        float NdotL = dot(normal, L);
        if (NdotL <= 0)
            continue;
        NdotL = saturate(NdotL);

        // Spot cone attenuation
        float spotAtten = EvaluateSpotAttenuation(lightBase, lightEx, worldPos);
        if (spotAtten <= 0)
            continue;

        // Compute incident irradiance based on light type
        float3 lightColor = UnpackLightColor(lightBase);
        float3 irradiance;

        PolymorphicLightType lightType = DecodeLightType(lightBase);
        if (lightType == PolymorphicLightType::kPoint)
        {
            // Point light: color is flux, inverse-square falloff
            irradiance = lightColor / dist2;
        }
        else // kSphere
        {
            // Sphere light: color is surface radiance, compute subtended solid angle
            float radius = f16tof32(lightBase.Scalars);
            float sinThetaMax2 = min((radius * radius) / dist2, 1.0);
            float solidAngle = 2.0 * K_PI * (1.0 - sqrt(max(0.0, 1.0 - sinThetaMax2)));
            irradiance = lightColor * solidAngle;
        }

        irradiance *= spotAtten;

        // --- Cook-Torrance BRDF ---

        float3 H = normalize(viewDir + L);
        float NdotH = saturate(dot(normal, H));
        float VdotH = saturate(dot(viewDir, H));

        // GGX NDF
        float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
        float D = a2 / (K_PI * denom * denom);

        // Fresnel
        float3 F = FresnelSchlick(VdotH, F0);

        // Smith's geometry term
        float G1L = NdotL / (NdotL * (1.0 - k) + k);
        float G = G1V * G1L;

        // Specular BRDF (denominator clamped to avoid division by zero)
        float3 specular = (D * F * G) / max(4.0 * NdotV * NdotL, 1e-4);

        // Diffuse (Lambertian, energy-conserved)
        float3 kD = (1.0 - F) * (1.0 - metallic);
        float3 diffuse = kD * baseColor * K_1_PI;

        totalLight += (diffuse + specular) * irradiance * NdotL;
    }

    return totalLight;
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
    
    // Basic lighting
    float3 rayDir = Bridge::computeCameraRay(pixelPos).dir;

    float3 radiance = 0;
    if (depth > 0)  // Check for valid depth (NDC depth, 0 = sky/miss)
    {
        // Load material properties from GBuffer
        float3 baseColor = surface.GetBaseColor();
        float metal = surface.GetMetalnes();
        float3 normal = surface.GetSpecNormal();
        float roughness = surface.GetRoughness();
        const float ao = t_GTAOOutput[pixelPos];

        // Compute screen UV for SSR sampling
        float2 screenUV = (float2(pixelPos) + 0.5) / float2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight);
                
        // Use IBL with SSR, local cubemap, and split-sum
        radiance = EvaluateIBL(pixelPos, normal, -rayDir, screenUV, baseColor, metal, roughness, ao);

        // Direct lighting from analytic lights (point and spot)
        float3 worldPos = ReconstructWorldPosition(pixelPos, depth);
        //radiance += EvaluateDirectLighting(worldPos, normal, -rayDir, baseColor, metal, roughness);
    }
    else // no-hit, sample the background
    {
        radiance = t_EnvironmentMap.SampleLevel(s_EnvironmentMapSampler, rayDir, 0).xyz;
    }

    // Output hdr color
    u_OutputColor[pixelPos] = float4(radiance, 1.0);
}