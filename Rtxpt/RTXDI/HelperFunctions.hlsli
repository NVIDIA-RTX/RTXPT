/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef HELPER_FUNCTIONS_HLSLI
#define HELPER_FUNCTIONS_HLSLI

#include "../../External/Rtxdi/Include/Rtxdi/Utils/Math.hlsli"
#include "../PathTracer/Utils/Geometry.hlsli"

static const float c_pi = 3.1415926535;

struct RandomSamplerState
{
    uint seed;
    uint index;
};

RandomSamplerState initRandomSampler(uint2 pixelPos, uint frameIndex)
{
    RandomSamplerState state;

    uint linearPixelIndex = RTXDI_ZCurveToLinearIndex(pixelPos);

    state.index = 1;
    state.seed = RTXDI_JenkinsHash(linearPixelIndex) + frameIndex;

    return state;
}

uint murmur3(inout RandomSamplerState r)
{
#define ROT32(x, y) ((x << y) | (x >> (32 - y)))

    // https://en.wikipedia.org/wiki/MurmurHash
    uint c1 = 0xcc9e2d51;
    uint c2 = 0x1b873593;
    uint r1 = 15;
    uint r2 = 13;
    uint m = 5;
    uint n = 0xe6546b64;

    uint hash = r.seed;
    uint k = r.index++;
    k *= c1;
    k = ROT32(k, r1);
    k *= c2;

    hash ^= k;
    hash = ROT32(hash, r2) * m + n;

    hash ^= 4;
    hash ^= (hash >> 16);
    hash *= 0x85ebca6b;
    hash ^= (hash >> 13);
    hash *= 0xc2b2ae35;
    hash ^= (hash >> 16);

#undef ROT32

    return hash;
}

float sampleUniformRng(inout RandomSamplerState r)
{
    uint v = murmur3(r);
    const uint one = asuint(1.f);
    const uint mask = (1 << 23) - 1;
    return asfloat((mask & v) | one) - 1.f;
}

// Maps ray hit UV into triangle barycentric coordinates
float3 hitUVToBarycentric(float2 hitUV)
{
    return float3(1 - hitUV.x - hitUV.y, hitUV.x, hitUV.y);
}

// Inverse of sampleTriangle
float2 randomFromBarycentric(float3 barycentric)
{
    float sqrtx = 1 - barycentric.x;
    return float2(sqrtx * sqrtx, barycentric.z / sqrtx);
}

float3 sampleCosHemisphere(float2 rand, out float solidAnglePdf)
{
    float2 tangential = SampleDiskUniform(rand);
    float elevation = sqrt(saturate(1.0 - rand.y));

    solidAnglePdf = elevation / c_pi;

    return float3(tangential.xy, elevation);
}

float3 sampleSphere(float2 rand, out float solidAnglePdf)
{
    // See (6-8) in https://mathworld.wolfram.com/SpherePointPicking.html

    rand.y = rand.y * 2.0 - 1.0;

    float2 tangential = SampleDiskUniform(float2(rand.x, 1.0 - sq(rand.y)));
    float elevation = rand.y;

    solidAnglePdf = 0.25f / c_pi;

    return float3(tangential.xy, elevation);
}

// This is not a "correct" luminance if 'color' is linear radiance but it's left in for historical reasons
// (it does NOT return a relative luminance of an input linear RGB color in the ITU-R BT.709 color space - that would be with float3(0.2126f, 0.7152f, 0.0722f))
float calcLuminance(float3 color)
{
    return dot(color.xyz, float3(0.299f, 0.587f, 0.114f));
}

float3 sphericalDirection(float sinTheta, float cosTheta, float sinPhi, float cosPhi, float3 x, float3 y, float3 z)
{
    return sinTheta * cosPhi * x + sinTheta * sinPhi * y + cosTheta * z;
}

void getReflectivity(float metalness, float3 baseColor, out float3 o_albedo, out float3 o_baseReflectivity)
{
    const float dielectricSpecular = 0.04;
    o_albedo = lerp(baseColor * (1.0 - dielectricSpecular), 0, metalness);
    o_baseReflectivity = lerp(dielectricSpecular, baseColor, metalness);
}


float3 sampleGGX_VNDF(float3 Ve, float roughness, float2 random)
{
    float alpha = sq(roughness);

    float3 Vh = normalize(float3(alpha * Ve.x, alpha * Ve.y, Ve.z));

    float lensq = sq(Vh.x) + sq(Vh.y);
    float3 T1 = lensq > 0.0 ? float3(-Vh.y, Vh.x, 0.0) / sqrt(lensq) : float3(1.0, 0.0, 0.0);
    float3 T2 = cross(Vh, T1);

    float r = sqrt(random.x);
    float phi = 2.0 * c_pi * random.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - sq(t1)) + s * t2;

    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - sq(t1) - sq(t2))) * Vh;

    // Tangent space H
    float3 Ne = float3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z));
    return Ne;
}

float2 directionToEquirectUV(float3 normalizedDirection)
{
    float elevation = asin(normalizedDirection.y);
    float azimuth = 0;
    if (abs(normalizedDirection.y) < 1.0)
        azimuth = atan2(normalizedDirection.z, normalizedDirection.x);

    float2 uv;
    uv.x = azimuth / (2 * c_pi) - 0.25;
    uv.y = 0.5 - elevation / c_pi;

    return uv;
}

float3 equirectUVToDirection(float2 uv, out float cosElevation)
{
    float azimuth = (uv.x + 0.25) * (2 * c_pi);
    float elevation = (0.5 - uv.y) * c_pi;
    cosElevation = cos(elevation);

    return float3(
        cos(azimuth) * cosElevation,
        sin(elevation),
        sin(azimuth) * cosElevation
    );
}

// Pack a normal into 2x 16 bit. Signed normal converted to octahedral mapping.
uint packNormal2x16(float3 normal)
{
    return NDirToOctUnorm32(normal);
    //float2 octNormal = ndirToOctSigned(normal);
    //return PackTwoFp32ToFp16(octNormal);
}

// Unpack a normal from 2x 16 bit. Signed normal converted from octahedral mapping.
float3 unpackNormal2x16(uint packedNormal)
{
    return OctToNDirUnorm32(packedNormal);
    // float2 octNormal = UnpackTwoFp32ToFp16(packedNormal);
    // return octToNdirSigned(octNormal);
}

#endif // HELPER_FUNCTIONS_HLSLI