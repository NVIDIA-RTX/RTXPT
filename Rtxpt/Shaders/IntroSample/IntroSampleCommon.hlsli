/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Shared types and utilities for screen-space ambient occlusion passes (GTAO / RTAO).
// Based on: Jimenez et al., "Practical Real-Time Strategies for Accurate Indirect Occlusion"

#ifndef __INTRO_SAMPLE_COMMON_HLSL__
#define __INTRO_SAMPLE_COMMON_HLSL__

#define SHARC_SEPARATE_EMISSIVE 0
#define SHARC_ENABLE_CACHE_RESAMPLING 0
// Require more than two resolved samples before accepting a cache entry.
#define SHARC_SAMPLE_NUM_THRESHOLD 2

#include <donut/shaders/brdf.hlsli>
#include <donut/shaders/utils.hlsli>
#include <donut/shaders/binding_helpers.hlsli>
#include <donut/shaders/surface.hlsli>
#include <donut/shaders/scene_material.hlsli>
#include "../PathTracer/Lighting/PolymorphicLight.h"
#include "../PathTracer/Utils/Math/MathConstants.hlsli"
#include "../PathTracer/Utils/NoiseAndSequences.hlsli"
#include "../PathTracer/Utils/Utils.hlsli"
#include "../PathTracer/Utils/Geometry.hlsli"

#include "SHARC/include/SharcCommon.h"

static const float SHARC_SCENE_SCALE = 25.0f;

// Roughness at or below this value is treated as a delta (mirror-like) path by the Intro path tracers.
// Cache acceptance follows the most recent scattering event, which may be diffuse after a delta path.
static const float RTXPT_INTRO_MAX_DELTA_ROUGHNESS = 0.1f;

HashGridParameters GetDefaultGridParams(float3 cameraPos)
{
    HashGridParameters gridParameters;

    gridParameters.cameraPosition = cameraPos;
    gridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
    gridParameters.sceneScale = SHARC_SCENE_SCALE;
    gridParameters.levelBias = SHARC_GRID_LEVEL_BIAS;

    return gridParameters;
}

SharcParameters GetDefaultSharcParams(float3 cameraPos)
{
    SharcParameters sharcParams;

    sharcParams.hashGridParameters = GetDefaultGridParams(cameraPos);
    sharcParams.radianceScale = 1e3f;

    return sharcParams;
}

#pragma pack_matrix(row_major)

// Must match AOConstants in AORenderer.cpp
struct AOConstants
{
    float4x4 MatWorldToView;
    float4x4 MatClipToWorldNoOffset;

    float2 ViewportInvSize;
    float ProjScale; // matViewToClip[0][0]
    float Radius;

    float FalloffEnd;
    float TemporalAlpha;
    uint HalfWidth;
    uint HalfHeight;

    uint FullWidth;
    uint FullHeight;
    uint FrameIndex;
    float ZNear;            // Near plane distance (for linearizing reverse-Z depth)
    
    float2 subPixelOffset;
    uint denoise;
};

struct RNGState
{
    uint seed;

    void initRngSeed(uint2 pixelPos, uint frameIndex)
    {
        seed = Hash32Combine(Hash32(frameIndex), (pixelPos.x << 16) | pixelPos.y);
    }

    float getNextUniformRand()
    {
        uint u1 = Hash32(seed);
        seed = u1;

        return Hash32ToFloat(u1);
    }

    float2 getNextUniformRand2()
    {
        uint u1 = Hash32(seed);
        uint u2 = Hash32(u1);
        seed = u2;

        return float2(Hash32ToFloat(u1), Hash32ToFloat(u2));
    }
};

// ============================================================================
// Shared utility functions
// ============================================================================

// Sub-pixel offset for half-res rendering: each half-res pixel samples a different
// position within its 2x2 full-res block using a spatially interleaved pattern.
// offset = (y%2, x%2), which cycles all 4 sub-positions within each 2x2 group
// of half-res pixels, providing full coverage every frame.
uint2 HalfResSubPixelOffset(uint2 halfResPixel)
{
    return uint2(halfResPixel.y & 1, halfResPixel.x & 1);
}

// Map a half-res pixel to the full-res pixel it represents
uint2 HalfToFullRes(uint2 halfResPixel, uint fullWidth, uint fullHeight)
{
    return min(halfResPixel * 2 + HalfResSubPixelOffset(halfResPixel),
               uint2(fullWidth - 1, fullHeight - 1));
}

// Reconstruct view-space position from NDC depth and pixel coordinates
float3 ReconstructViewPos(uint2 fullResPixel, float ndcDepth, float2 subPixelOffset, float2 ViewportInvSize, float4x4 MatClipToWorldNoOffset, float4x4 MatWorldToView)
{
    float2 uv = (float2(fullResPixel) + 0.5 - subPixelOffset) * ViewportInvSize;
    float4 clipPos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, ndcDepth, 1.0);
    float4 worldPos = mul(clipPos, MatClipToWorldNoOffset);
    worldPos.xyz /= worldPos.w;
    float4 viewPos = mul(float4(worldPos.xyz, 1.0), MatWorldToView);
    return viewPos.xyz;
}

// Reconstruct world-space position from NDC depth and pixel coordinates
float3 ReconstructWorldPos(uint2 fullResPixel, float ndcDepth, float2 subPixelOffset, float2 ViewportInvSize, float4x4 MatClipToWorldNoOffset)
{
    float2 uv = (float2(fullResPixel) + 0.5 - subPixelOffset) * ViewportInvSize;
    float4 clipPos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, ndcDepth, 1.0);
    float4 worldPos = mul(clipPos, MatClipToWorldNoOffset);
    return worldPos.xyz / worldPos.w;
}

// 4x4 spatial noise directions, uniformly spaced over [0, pi)
// These tile every 4 pixels in each dimension, giving 16 unique directions
float AO_SpatialDirection(uint2 pixel, uint frameIndex, uint numTemporalRotations)
{
    // Interleaved gradient-style noise using a 4x4 tile
    // 16 base directions + temporal rotations = many effective directions
    uint tileX = pixel.x & 3;
    uint tileY = pixel.y & 3;
    uint idx = tileY * 4 + tileX;

    // Hashed rotation offset per tile slot
    float baseAngle = float(idx) * (K_PI / 16); // Only rotate within the top semicircle. We'll sample along both + and - the direction.

    // Temporal rotation: cycle through N rotations
    // TODO: Maybe apply the rotation randomly here according to a low discrepancy sequence?
    float temporalOffset = float(frameIndex % numTemporalRotations) * (K_PI / float(16 * numTemporalRotations));

    return baseAngle + temporalOffset;
}

// Analytic inner integral of the AO equation (Eq. 7)
// theta1, theta2: horizon angles, gamma: angle between projected normal and view direction
float AO_IntegrateArc(float theta1, float theta2, float gamma)
{
    float cosGamma = cos(gamma);
    float sinGamma = sin(gamma);

    return 0.25 * (2 * theta2 * sinGamma + cosGamma - cos(2 * theta2 - gamma)
        + 2 * theta1 * sinGamma + cosGamma - cos(2 * theta1 - gamma));
}

// Reconstruct world position from pixel coordinates and NDC depth
float3 ReconstructWorldPosition(uint2 pixelPos, float depth, float4x4 matClipToWorldNoOffset, float2 viewportSizeInv)
{
    float2 uv = (float2(pixelPos) + 0.5) * viewportSizeInv;
    float4 clipPos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    float4 worldPos = mul(clipPos, matClipToWorldNoOffset);
    return worldPos.xyz / worldPos.w;
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
    return (PolymorphicLightType) typeCode;
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

// Evaluate incident irradiance from a polymorphic light (point or sphere) at a surface point.
// Returns irradiance (color * geometric term), including spot attenuation.
float3 EvaluatePolymorphicLightIrradiance(PolymorphicLightInfo lightBase, float dist2, float spotAtten)
{
    float3 lightColor = UnpackLightColor(lightBase);
    float3 irradiance;

    PolymorphicLightType lightType = DecodeLightType(lightBase);
    if (lightType == PolymorphicLightType::kPoint)
    {
        irradiance = lightColor / dist2;
    }
    else // kSphere
    {
        float radius = f16tof32(lightBase.Scalars);
        float sinThetaMax2 = min((radius * radius) / dist2, 1.0);
        float solidAngle = 2.0 * K_PI * (1.0 - sqrt(max(0.0, 1.0 - sinThetaMax2)));
        irradiance = lightColor * solidAngle;
    }

    return irradiance * spotAtten;
}

// Scalar variant for target-PDF computation (uses max-component luminance).
float EvaluatePolymorphicLightIrradianceScalar(PolymorphicLightInfo lightBase, float dist2, float spotAtten)
{
    float3 lc = UnpackLightColor(lightBase);
    float lightColor = max(lc.r, max(lc.g, lc.b));
    float irradiance;

    PolymorphicLightType lightType = DecodeLightType(lightBase);
    if (lightType == PolymorphicLightType::kPoint)
    {
        irradiance = lightColor / dist2;
    }
    else // kSphere
    {
        float radius = f16tof32(lightBase.Scalars);
        float sinThetaMax2 = min((radius * radius) / dist2, 1.0);
        float solidAngle = 2.0 * K_PI * (1.0 - sqrt(max(0.0, 1.0 - sinThetaMax2)));
        irradiance = lightColor * solidAngle;
    }

    return irradiance * spotAtten;
}

// GGX Normal Distribution Function
float DistributionGGX(float NdotH, float alpha2)
{
    float NdotH2 = NdotH * NdotH;
    
    float denom = (NdotH2 * (alpha2 - 1.0) + 1.0);
    denom = max(K_PI * denom * denom, 1e-6);
    
    return alpha2 / denom;
}
// Smith Shadowing/masking function for GGX's denominator. Useful for computing G2 / G1
float SmithGGX_G1_denom(float alpha, float NdV)
{
    float a2 = square(alpha);
    float root = a2 + (1 - a2) * NdV * NdV;
    float den = sqrt(root) + NdV;
    return den;
}

// The full, correct G2 Smith GGX term
float SmithGGX_G2(float alpha, float NdV, float NdL)
{
    float num = 2 * NdL * NdV;
    float a2 = square(alpha);
    float g1 = NdV * sqrt(a2 + (1 - a2) * NdL * NdL);
    float g2 = NdL * sqrt(a2 + (1 - a2) * NdV * NdV);

    return num / (g1 + g2);
}

// GGX Term for integration with VNDF importance sampled directions
float SmithGGX_G2_over_G1v(float roughness, float NdV, float NdL)
{
    float alpha = max(1e-4, square(roughness));
    float a2 = square(alpha);
    float g1 = NdV * sqrt(a2 + (1.0 - a2) * square(NdL));
    float g2 = NdL * sqrt(a2 + (1.0 - a2) * square(NdV));
    float g1Denom = SmithGGX_G1_denom(alpha, NdV);
    return NdL * g1Denom / max(g1 + g2, 1e-4);
}

// The G2 Smith GGX term, predivided by the numerator, so you can avoid the division in CookTorrance's brdf
float G2_Smith_predivv(float alpha2, float NdV, float NdL)
{
    float g1 = NdV * sqrt(alpha2 + (1.0 - alpha2) * square(NdL));
    float g2 = NdL * sqrt(alpha2 + (1.0 - alpha2) * square(NdV));
    return 2.0 / max(g1 + g2, 1e-4);
}

// No NdL included!
float3 CookTorranceBRDF(float3 L, float3 normal, float3 viewDir, float3 F0, float alpha2, float NdotL, float NdotV)
{
    float3 H = normalize(viewDir + L);
    float NdotH = saturate(dot(normal, H));
    float VdotH = saturate(dot(viewDir, H));

    // GGX NDF
    float D = DistributionGGX(NdotH, alpha2);

    // Fresnel
    float3 F = Schlick_Fresnel(F0, VdotH);

    // Smith's geometry term
    float G = G2_Smith_predivv(alpha2, NdotV, NdotL);

    // Specular BRDF (denominator clamped to avoid division by zero)
    float3 specular = F * ((D * G) / 4); // Multipy by F last, so everything else stays scalar math

    return specular;
}

float D_GGX(float3 m, float a2)
{
    float D = a2 / max(1e-4, M_PI * square(m.z * a2 + (1 - m.z)));
    return D;
}

float GGXReflectionPDF(float3 i, float3 m, float alpha, float a2)
{
    float ndf = D_GGX(m, a2);
    float ai = alpha * i.x;
    float len2 = ai * ai;
    float t = sqrt(len2 + i.z * i.z);
    if (i.z >= 0.0f)
    {
        float s = 1.0f + i.x; // Omit sgn for a <=1
        float s2 = s * s;
        float k = (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z); // Eq. 5
        return max(1e-3, ndf / (2.0 * (k * i.z + t))); // Eq. 8 * || dm/do ||
    }
    // Numerically stable form of the previous PDF for i.z < 0
    return ndf * (t - i.z) / (2.0f * len2); // = Eq. 7 * || dm/do ||
}

float VNDF_SpecularBRDF(float roughness, float3 normal, float3 halfV, float3 view, float3 light)
{    
    float NoL = saturate(dot(normal, light));
    float NoV = saturate(dot(normal, view));
    float VoH = saturate(dot(halfV, view));
        
    if (NoL > 0)
    {
        return SmithGGX_G2_over_G1v(roughness, NoV, NoL);
    }
    return 0;
}

float3 SampleVNDF(float2 rand, float3 view, float3 normal, float alpha)
{
    // Any orthonormal frame with z = normal works for VNDF sampling; the only requirement is that the SAME
    // frame is used to express the view for sampling and to transform the sampled half vector back to world.
    // (Caller must pass a view-side normal, i.e. dot(view, normal) >= 0.)
    float3 tangent, bitangent;
    BranchlessONB(normal, tangent, bitangent);

    // View in that frame; ImportanceSampleGGX_VNDF accepts an arbitrary (non-azimuth-0) view direction.
    float3 i = float3(dot(view, tangent), dot(view, bitangent), dot(view, normal));

    float r = sqrt(max(1e-3, alpha));
    float3 H = ImportanceSampleGGX_VNDF(rand, r, i, 0.98);

    // Half vector back in world space, using the same frame.
    return normalize(tangent * H.x + bitangent * H.y + normal * H.z);
}

#endif // __INTRO_SAMPLE_COMMON_HLSL__
